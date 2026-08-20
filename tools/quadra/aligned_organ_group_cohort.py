#!/usr/bin/env python
"""Resumable aligned UAE-S organ-group cohort cycle-error runner.

This command promotes the Stage 5R engineering configuration to a deliberately
bounded technical cohort run.  It does not claim scientific validation, does
not run registration, and never persists CT arrays or embedding tensors.

The module remains syntactically compatible with Python 3.7 for the pinned
UAE-S image.
"""

from __future__ import print_function

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra import body_envelope_audit as stage1  # noqa: E402
from tools.quadra import memory_configuration_screen as stage3  # noqa: E402
from tools.quadra import organ_group_lattice_alignment as stage5r  # noqa: E402
from tools.quadra.environment import canonical_layout  # noqa: E402


SCHEMA_VERSION = 1
COHORT_ID = "quadra-uaes-aligned-100mm-cohort-v1"
RUN_PREFIX = "uaes-aligned-100mm-"
EXPECTED_BASE_COMMIT = "5e63838c6c33aaf19abfa4bd9670cacd8c9d5cdd"
EXPECTED_STAGE5R_SELECTED_SHA256 = (
    "acff1a602d0b2661f01cfca1f3970759a15e7697739789e8725aeb6af8311c3c"
)
EXPECTED_SUBJECTS = 28
EXPECTED_SCANS = 56
EXPECTED_MALE_SUBJECTS = 12
EXPECTED_FEMALE_SUBJECTS = 16
EXPECTED_MASKS = 2208
EXPECTED_QUERIES = 110400
POINTS_PER_MASK = 100
SEED = 20260721
GROUP_ORDER = ("pelvis", "abdomen", "thorax", "head_neck")
SUBJECT_GATE = "quadra_hc_030"
VRAM_CEILING_MIB = stage5r.VRAM_CEILING_MIB
MIN_FREE_DISK_GIB = 10.0
SUBJECT_TIMEOUT_SECONDS = 6 * 60 * 60


class CohortError(RuntimeError):
    pass


class InfrastructureError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise CohortError("Required file is missing: {}".format(path))
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise CohortError("Cannot read JSON {}: {}".format(path, exc))
    if not isinstance(value, dict):
        raise CohortError("Expected a JSON object: {}".format(path))
    return value


def atomic_json(path, value, refuse=False):
    path = Path(path)
    if refuse and path.exists():
        raise CohortError("Refusing to overwrite: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_csv(path, rows, fieldnames=None, refuse=False):
    path = Path(path)
    if refuse and path.exists():
        raise CohortError("Refusing to overwrite: {}".format(path))
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git_output(args):
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT)] + list(args), text=True).strip()


def validate_repository(require_clean=True):
    head = git_output(["rev-parse", "HEAD"])
    try:
        subprocess.check_call(
            ["git", "-C", str(PROJECT_ROOT), "merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT, head],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        raise CohortError("Repository does not descend from accepted base {}".format(EXPECTED_BASE_COMMIT))
    dirty = git_output(["status", "--porcelain"])
    if require_clean and dirty:
        raise CohortError("Repository is dirty; refusing cohort execution")
    return {
        "path": str(PROJECT_ROOT), "branch": git_output(["branch", "--show-current"]),
        "execution_commit": head, "accepted_base_commit": EXPECTED_BASE_COMMIT,
        "clean": not bool(dirty),
    }


def validate_stage5r_checkpoint(path):
    checkpoint_identity = file_identity(path)
    checkpoint = load_json(path)
    gates = checkpoint.get("gates", {})
    selected_ref = checkpoint.get("selected_workflow")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("stage") != 5
        or checkpoint.get("substage") != "R"
        or checkpoint.get("resolution_id") != stage5r.RESOLUTION_ID
        or checkpoint.get("status") != "PASS"
        or gates.get("restricted_dense_chunked_equivalence_passed") is not True
        or gates.get("all_three_contrasts_passed_pooled_and_per_group") is not True
        or gates.get("zero_matches_outside_shared_admissible_domain") is not True
        or gates.get("memory_headroom_passed") is not True
        or not isinstance(selected_ref, dict)
    ):
        raise CohortError("Stage 5R checkpoint is not the authoritative PASS evidence")
    if file_identity(selected_ref.get("path")) != selected_ref:
        raise CohortError("Stage 5R selected workflow identity changed")
    if selected_ref["sha256"] != EXPECTED_STAGE5R_SELECTED_SHA256:
        raise CohortError("Unexpected Stage 5R selected-manifest hash")
    selected = load_json(selected_ref["path"])
    if (
        selected.get("status") != "PASS"
        or selected.get("selected_workflow") != "organ_group_aligned_100mm_fp32_global_nn"
        or selected.get("precision") != "fp32"
        or selected.get("matching_mode") != "global_nn"
    ):
        raise CohortError("Stage 5R selected workflow contract changed")
    return checkpoint_identity, checkpoint, selected_ref


def registry_records(registry_path):
    import yaml

    payload = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    records = list(payload.get("organs", [])) + list(payload.get("derived_organs", []))
    names = [item["filename"] for item in records]
    if len(names) != 40 or len(set(names)) != 40:
        raise CohortError("Expected a fixed 40-mask global registry")
    configured = [name for group in stage3.GROUPS.values() for name in group]
    if set(configured) != set(names) or len(configured) != 40:
        raise CohortError("Organ groups do not partition the fixed mask registry")
    group_by_mask = {}
    for group, masks in stage3.GROUPS.items():
        for name in masks:
            group_by_mask[name] = group
    return [dict(item, registry_index=index, group_name=group_by_mask[item["filename"]]) for index, item in enumerate(records)]


def sample_unique_mask_points(mask_path, count, seed):
    import nibabel as nib
    import numpy as np

    image = nib.load(str(mask_path))
    data = np.asanyarray(image.dataobj)
    indices_xyz = np.argwhere(data > 0)
    if len(indices_xyz) < count:
        raise CohortError("Mask has fewer than {} unique voxels: {}".format(count, mask_path))
    rng = np.random.default_rng(int(seed))
    chosen = indices_xyz[rng.choice(len(indices_xyz), size=count, replace=False)]
    if len({tuple(value) for value in chosen.tolist()}) != count:
        raise CohortError("Duplicate query voxel sampled from {}".format(mask_path))
    return image, chosen.astype(np.int64)


def _stage1_sources(selected_path):
    selected_identity = file_identity(selected_path)
    selected = load_json(selected_path)
    if selected_identity["sha256"] != stage3.EXPECTED_SELECTED_SHA256:
        raise CohortError("Frozen Stage 1 selected body-envelope hash changed")
    audit_ref = selected.get("audit_manifest")
    if not isinstance(audit_ref, dict) or file_identity(audit_ref.get("path")) != audit_ref:
        raise CohortError("Stage 1 audit identity changed")
    audit = load_json(audit_ref["path"])
    mask_ref = audit.get("outputs", {}).get("mask_clearance.csv")
    if not isinstance(mask_ref, dict) or file_identity(mask_ref.get("path")) != mask_ref:
        raise CohortError("Stage 1 mask-clearance evidence changed")
    return selected_identity, selected, audit_ref, Path(mask_ref["path"])


def build_aligned_plans(selected, mask_csv, registry_path):
    derived = stage3.derive_spatial_plans(selected, mask_csv, registry_path)
    plans = {}
    for old in derived["organ_group"]:
        aligned = stage5r.aligned_plan_from_union(old, 100)
        key = (aligned["subject_id"], aligned["session"], aligned["group_name"])
        if key in plans:
            raise CohortError("Duplicate aligned plan: {}".format(key))
        plans[key] = aligned
    if len(plans) != 224:
        raise CohortError("Expected 224 aligned cohort plans, found {}".format(len(plans)))
    return plans


def _query_rows(scans, registry, plans):
    import numpy as np

    test_scans = {item["subject_id"]: item for item in scans if item["session"] == "test"}
    rows = []
    for subject in sorted(test_scans):
        scan = test_scans[subject]
        accepted = set(scan["expected_masks"])
        expected = {item["filename"] for item in registry if not (item["filename"] == "prostate" and scan["sex"] != "M")}
        if accepted != expected:
            raise CohortError("Manifest mask set differs from fixed registry for {}".format(subject))
        for item in registry:
            mask_name = item["filename"]
            if mask_name not in expected:
                continue
            mask_path = Path(scan["mask_directory"]) / (mask_name + ".nii.gz")
            image, points = sample_unique_mask_points(
                mask_path, POINTS_PER_MASK, SEED + int(item["registry_index"])
            )
            ct_plan = plans[(subject, "test", item["group_name"])]
            if tuple(image.shape[:3]) != tuple(ct_plan["source_ct"]["native_shape_xyz"]):
                raise CohortError("Mask/CT shape mismatch: {}".format(mask_path))
            if not np.allclose(image.affine, np.asarray(ct_plan["source_ct"]["affine"]), atol=1e-5, rtol=0.0):
                raise CohortError("Mask/CT affine mismatch: {}".format(mask_path))
            model = stage5r.apply_affine(points, ct_plan["raw_to_model_continuous_affine"])
            valid = np.asarray(ct_plan["valid_model_box_xyz"], dtype=np.float64)
            if np.any(model < valid[0] - 1e-6) or np.any(model >= valid[1] + 1e-6):
                raise CohortError("Sampled mask point lies outside aligned plan: {}".format(mask_path))
            for point_index, point in enumerate(points):
                rows.append(
                    {
                        "query_id": "{}:{}:{:03d}".format(subject, mask_name, point_index),
                        "subject_id": subject, "sex": scan["sex"],
                        "group_name": item["group_name"], "mask_name": mask_name,
                        "mask_registry_index": int(item["registry_index"]),
                        "point_index": point_index,
                        "raw_x": int(point[0]), "raw_y": int(point[1]), "raw_z": int(point[2]),
                        "sampling_seed": SEED + int(item["registry_index"]),
                    }
                )
    keys = {row["query_id"] for row in rows}
    if len(rows) != EXPECTED_QUERIES or len(keys) != EXPECTED_QUERIES:
        raise CohortError("Expected {} unique queries, found {}/{}".format(EXPECTED_QUERIES, len(rows), len(keys)))
    return rows


def _plan_record(path, plan):
    return dict(file_identity(path), subject_id=plan["subject_id"], session=plan["session"], group_name=plan["group_name"])


def run_prepare(args):
    storage_root = Path(args.storage_root).resolve()
    layout = canonical_layout(storage_root)
    repository = validate_repository()
    try:
        environment = stage3.read_profile_fingerprint(storage_root, "uae")
    except stage3.Stage3Error as exc:
        raise CohortError(str(exc))
    stage5r_identity, stage5r_checkpoint, selected5r = validate_stage5r_checkpoint(args.stage5r_checkpoint)
    stage1_identity, selected1, audit_ref, mask_csv = _stage1_sources(args.stage1_selected)
    registry_path = Path(args.registry or PROJECT_ROOT / "tools/quadra/totalsegmentator/organs.yaml").resolve()
    registry = registry_records(registry_path)
    cohort_path = Path(args.cohort_manifest or layout["manifests"] / stage1.COHORT_MANIFEST_NAME).resolve()
    cohort_identity = file_identity(cohort_path)
    _, scans = stage1.load_cohort(cohort_path, layout["whole_body_ct"], layout["totalsegmentator_outputs"])
    subjects = sorted({item["subject_id"] for item in scans})
    sexes = {subject: next(item["sex"] for item in scans if item["subject_id"] == subject) for subject in subjects}
    if len(subjects) != EXPECTED_SUBJECTS or sum(value == "M" for value in sexes.values()) != EXPECTED_MALE_SUBJECTS or sum(value == "F" for value in sexes.values()) != EXPECTED_FEMALE_SUBJECTS:
        raise CohortError("Cohort subject/sex denominator changed")
    plans = build_aligned_plans(selected1, mask_csv, registry_path)

    output_root = Path(args.output_root or storage_root / "runs/cohort").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not run_id.startswith(RUN_PREFIX):
        raise CohortError("Invalid cohort run id")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise CohortError("Refusing to overwrite cohort run: {}".format(run_dir))
    run_dir.mkdir(parents=True)
    (run_dir / "plans").mkdir()
    (run_dir / "group_results").mkdir()
    (run_dir / "subject_results").mkdir()
    (run_dir / "logs").mkdir()

    plan_records = []
    for key in sorted(plans):
        plan = plans[key]
        target = run_dir / "plans" / "{}-{}-{}.json".format(plan["subject_id"], plan["session"], plan["group_name"])
        atomic_json(target, plan, refuse=True)
        plan_records.append(_plan_record(target, plan))
    query_rows = _query_rows(scans, registry, plans)
    query_path = run_dir / "frozen_queries_raw_itk.csv"
    atomic_csv(query_path, query_rows, refuse=True)
    counts = {}
    for row in query_rows:
        counts[row["subject_id"]] = counts.get(row["subject_id"], 0) + 1
    if counts.get(SUBJECT_GATE) != 3900:
        raise CohortError("Subject 030 gate denominator changed")

    config = file_identity(args.config)
    checkpoint = file_identity(args.checkpoint)
    if config["sha256"] != stage3.EXPECTED_CONFIG_SHA256 or checkpoint["sha256"] != stage3.EXPECTED_CHECKPOINT_SHA256:
        raise CohortError("UAE-S config/checkpoint identity changed")
    subject_order = [SUBJECT_GATE] + [item for item in subjects if item != SUBJECT_GATE]
    manifest = {
        "schema_version": SCHEMA_VERSION, "cohort_id": COHORT_ID, "status": "prepared",
        "created_at": utc_now(), "run_directory": str(run_dir), "repository": repository,
        "environment": environment,
        "stage5r_checkpoint": stage5r_identity, "stage5r_selected_workflow": selected5r,
        "stage1_selected_body_envelope": stage1_identity, "stage1_audit": audit_ref,
        "cohort_manifest": cohort_identity, "registry": file_identity(registry_path),
        "config": config, "checkpoint": checkpoint,
        "settings": {
            "workflow": "organ_group_aligned_100mm_fp32_global_nn",
            "spacing_xyz_mm": [2.0, 2.0, 2.0], "margin_mm": 100,
            "stride_xyz": list(stage5r.STRIDE_XYZ), "model_precision": "fp32",
            "embedding_dtype": "fp16", "similarity_dtype": "fp32",
            "coordinate_space": "raw_itk_voxel", "primary_error_unit": "mm",
            "query_batch_size": stage5r.QUERY_BATCH_SIZE,
            "match_chunk_xyz": list(stage5r.MATCH_CHUNK_XYZ),
            "points_per_mask": POINTS_PER_MASK, "seed": SEED,
            "group_order": list(GROUP_ORDER), "subject_order": subject_order,
            "subject_gate": SUBJECT_GATE, "vram_ceiling_mib": VRAM_CEILING_MIB,
        },
        "denominators": {
            "subjects": EXPECTED_SUBJECTS, "scans": EXPECTED_SCANS,
            "male_subjects": EXPECTED_MALE_SUBJECTS, "female_subjects": EXPECTED_FEMALE_SUBJECTS,
            "masks": EXPECTED_MASKS, "queries": EXPECTED_QUERIES,
            "subject_query_counts": counts,
        },
        "outputs": {"queries": file_identity(query_path), "plans": plan_records},
        "scope": {
            "technical_cohort": True, "registration": False, "fixed_point": False,
            "superpoint": False, "scientific_validation_claimed": False,
            "embeddings_persisted": False, "prepared_volumes_persisted": False,
        },
    }
    manifest["contract_signature"] = sha256_payload({key: manifest[key] for key in ("cohort_id", "repository", "environment", "stage5r_checkpoint", "stage5r_selected_workflow", "cohort_manifest", "registry", "config", "checkpoint", "settings", "denominators", "outputs", "scope")})
    atomic_json(run_dir / "cohort_manifest.json", manifest, refuse=True)
    status = initial_status(manifest)
    atomic_json(run_dir / "cohort_status.json", status, refuse=True)
    print("Cohort preparation PASS", flush=True)
    print("Run directory: {}".format(run_dir), flush=True)
    print("Frozen queries: {}".format(len(query_rows)), flush=True)
    return run_dir


def initial_status(manifest):
    return {
        "schema_version": SCHEMA_VERSION, "cohort_id": COHORT_ID,
        "status": "prepared", "updated_at": utc_now(), "heartbeat_at": utc_now(),
        "controller_pid": None, "current_subject": None, "current_group": None,
        "subjects_total": EXPECTED_SUBJECTS, "subjects_completed": 0,
        "groups_total": EXPECTED_SUBJECTS * len(GROUP_ORDER), "groups_completed": 0,
        "queries_total": EXPECTED_QUERIES, "queries_completed": 0,
        "failed_subjects": [], "last_error": None,
        "gate_subject": SUBJECT_GATE, "gate_passed": False,
        "run_directory": manifest["run_directory"],
    }


def _load_manifest(run_dir, statuses=None):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "cohort_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("cohort_id") != COHORT_ID or manifest.get("schema_version") != SCHEMA_VERSION:
        raise CohortError("Not a compatible aligned cohort run")
    if statuses is not None and manifest.get("status") not in statuses:
        raise CohortError("Cohort manifest status is not compatible: {}".format(manifest.get("status")))
    validate_stage5r_checkpoint(manifest["stage5r_checkpoint"]["path"])
    if file_identity(manifest["outputs"]["queries"]["path"]) != manifest["outputs"]["queries"]:
        raise CohortError("Frozen query CSV changed")
    for record in manifest["outputs"]["plans"]:
        if file_identity(record["path"]) != {key: record[key] for key in ("path", "bytes", "sha256")}:
            raise CohortError("Frozen aligned plan changed: {}".format(record["path"]))
    return run_dir, manifest_path, manifest


def _plan_lookup(manifest):
    result = {}
    for record in manifest["outputs"]["plans"]:
        plan = load_json(record["path"])
        result[(plan["subject_id"], plan["session"], plan["group_name"])] = (record, plan)
    return result


def _rows_for_subject(manifest, subject):
    return [row for row in read_csv(manifest["outputs"]["queries"]["path"]) if row["subject_id"] == subject]


def _group_signature(manifest, subject, group, rows, test_record, retest_record):
    return sha256_payload(
        {
            "contract_signature": manifest["contract_signature"], "subject_id": subject,
            "group_name": group, "query_ids": [row["query_id"] for row in rows],
            "test_plan": {key: test_record[key] for key in ("path", "bytes", "sha256")},
            "retest_plan": {key: retest_record[key] for key in ("path", "bytes", "sha256")},
        }
    )


RESULT_FIELDS = (
    "query_id", "subject_id", "sex", "group_name", "mask_name", "mask_registry_index",
    "point_index", "query_raw_x", "query_raw_y", "query_raw_z",
    "matched_raw_x", "matched_raw_y", "matched_raw_z",
    "matched_raw_rounded_x", "matched_raw_rounded_y", "matched_raw_rounded_z",
    "returned_raw_x", "returned_raw_y", "returned_raw_z",
    "returned_raw_rounded_x", "returned_raw_rounded_y", "returned_raw_rounded_z",
    "query_physical_x", "query_physical_y", "query_physical_z",
    "matched_physical_x", "matched_physical_y", "matched_physical_z",
    "returned_physical_x", "returned_physical_y", "returned_physical_z",
    "cycle_error_mm", "score_forward", "score_backward",
)


def validate_group_result(run_dir, manifest, subject, group, rows, test_record, retest_record):
    meta_path = Path(run_dir) / "group_results" / subject / (group + ".json")
    csv_path = Path(run_dir) / "group_results" / subject / (group + ".csv")
    if not meta_path.exists() and not csv_path.exists():
        return None
    # The metadata file is the commit marker.  A lone CSV can only be an
    # interrupted, uncommitted current-group write and is safe to regenerate.
    if csv_path.is_file() and not meta_path.exists():
        csv_path.unlink()
        return None
    if not meta_path.is_file() or not csv_path.is_file():
        raise CohortError("Incomplete group output pair: {} {}".format(subject, group))
    meta = load_json(meta_path)
    signature = _group_signature(manifest, subject, group, rows, test_record, retest_record)
    if (
        meta.get("schema_version") != SCHEMA_VERSION or meta.get("cohort_id") != COHORT_ID
        or meta.get("status") != "success" or meta.get("group_signature") != signature
        or meta.get("rows") != len(rows) or file_identity(csv_path) != meta.get("result_csv")
    ):
        raise CohortError("Incompatible group resume result: {} {}".format(subject, group))
    result_rows = read_csv(csv_path)
    if len(result_rows) != len(rows) or {item["query_id"] for item in result_rows} != {item["query_id"] for item in rows}:
        raise CohortError("Group row identity mismatch: {} {}".format(subject, group))
    return meta


def _write_status(run_dir, update):
    import fcntl

    path = Path(run_dir) / "cohort_status.json"
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status = load_json(path)
        status.update(update)
        status["updated_at"] = utc_now()
        status["heartbeat_at"] = utc_now()
        atomic_json(path, status)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return status


class Heartbeat(threading.Thread):
    def __init__(self, run_dir, interval_seconds=60):
        threading.Thread.__init__(self)
        self.daemon = True
        self.run_dir = Path(run_dir)
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.wait(self.interval_seconds):
            try:
                _write_status(self.run_dir, {})
            except Exception:
                # The foreground worker remains authoritative and will expose a
                # real failure through its atomic subject failure record.
                pass

    def stop(self):
        self.stop_event.set()
        self.join(timeout=5)


def process_group(run_dir, manifest, subject, group, rows, test_record, test_plan, retest_record, retest_plan):
    import numpy as np
    import torch
    from tools.quadra.streaming_cycle_error import stream_global_match_uaes

    signature = _group_signature(manifest, subject, group, rows, test_record, retest_record)
    model = None
    test_cache = None
    retest_cache = None
    sampler = None
    started = time.time()
    try:
        sampler = stage3.NvidiaProcessSampler(os.getpid())
        sampler.start()
        model, _ = stage3._load_model(Path(manifest["config"]["path"]), Path(manifest["checkpoint"]["path"]), "fp32")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.backends.cudnn.benchmark = stage3.CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = stage3.CUDNN_DETERMINISTIC
        test_cache, test_extract = stage5r._extract_cache(model, test_plan)
        retest_cache, retest_extract = stage5r._extract_cache(model, retest_plan)
        query_raw = np.asarray([[float(row[key]) for key in ("raw_x", "raw_y", "raw_z")] for row in rows], dtype=np.float64)
        query_model = np.stack([stage5r._raw_to_model_index(point, test_plan) for point in query_raw])
        retest_box = np.asarray(retest_plan["valid_model_box_xyz"], dtype=np.int64).tolist()
        matched_model, score_forward, forward_profile = stream_global_match_uaes(
            test_cache, retest_cache, query_model, stage5r.QUERY_BATCH_SIZE,
            stage5r.MATCH_CHUNK_XYZ, output_space="native", admissible_target_box_xyz=retest_box,
        )
        test_box = np.asarray(test_plan["valid_model_box_xyz"], dtype=np.int64).tolist()
        returned_model, score_backward, backward_profile = stream_global_match_uaes(
            retest_cache, test_cache, matched_model, stage5r.QUERY_BATCH_SIZE,
            stage5r.MATCH_CHUNK_XYZ, output_space="native", admissible_target_box_xyz=test_box,
        )
        result_rows = []
        for index, row in enumerate(rows):
            matched_raw = np.asarray(stage5r._model_to_raw(matched_model[index], retest_plan), dtype=np.float64)
            returned_raw = np.asarray(stage5r._model_to_raw(returned_model[index], test_plan), dtype=np.float64)
            query_physical = np.asarray(stage5r._physical(query_raw[index], test_plan), dtype=np.float64)
            matched_physical = np.asarray(stage5r._physical(matched_raw, retest_plan), dtype=np.float64)
            returned_physical = np.asarray(stage5r._physical(returned_raw, test_plan), dtype=np.float64)
            rounded_matched = np.rint(matched_raw).astype(np.int64)
            rounded_returned = np.rint(returned_raw).astype(np.int64)
            record = {
                "query_id": row["query_id"], "subject_id": subject, "sex": row["sex"],
                "group_name": group, "mask_name": row["mask_name"],
                "mask_registry_index": row["mask_registry_index"], "point_index": row["point_index"],
                "cycle_error_mm": float(np.linalg.norm(query_physical - returned_physical)),
                "score_forward": float(score_forward[index]), "score_backward": float(score_backward[index]),
            }
            for prefix, value in (("query_raw", query_raw[index]), ("matched_raw", matched_raw), ("returned_raw", returned_raw), ("query_physical", query_physical), ("matched_physical", matched_physical), ("returned_physical", returned_physical)):
                for axis, component in zip(("x", "y", "z"), value):
                    record[prefix + "_" + axis] = float(component)
            for prefix, value in (("matched_raw_rounded", rounded_matched), ("returned_raw_rounded", rounded_returned)):
                for axis, component in zip(("x", "y", "z"), value):
                    record[prefix + "_" + axis] = int(component)
            result_rows.append(record)
        if not all(np.isfinite(float(row["cycle_error_mm"])) for row in result_rows):
            raise CohortError("Non-finite cycle error")
        peaks = [
            float(test_extract["torch_peak_reserved_bytes"]) / (1024.0 ** 2),
            float(retest_extract["torch_peak_reserved_bytes"]) / (1024.0 ** 2),
            float(forward_profile["peak_gpu_memory_bytes"]) / (1024.0 ** 2),
            float(backward_profile["peak_gpu_memory_bytes"]) / (1024.0 ** 2),
        ]
        if sampler is not None:
            sampler.stop()
            process_peak = sampler.maximum
            sampler = None
        else:
            process_peak = None
        if process_peak is not None:
            peaks.append(float(process_peak))
        result_dir = Path(run_dir) / "group_results" / subject
        result_dir.mkdir(parents=True, exist_ok=True)
        csv_path = result_dir / (group + ".csv")
        atomic_csv(csv_path, result_rows, fieldnames=list(RESULT_FIELDS), refuse=True)
        meta = {
            "schema_version": SCHEMA_VERSION, "cohort_id": COHORT_ID, "status": "success",
            "subject_id": subject, "group_name": group, "group_signature": signature,
            "rows": len(result_rows), "result_csv": file_identity(csv_path),
            "completed_at": utc_now(), "wall_time_seconds": time.time() - started,
            "peak_gpu_memory_mib": max(peaks), "process_gpu_peak_mib": process_peak,
            "test_extraction": test_extract,
            "retest_extraction": retest_extract, "forward_match": forward_profile,
            "backward_match": backward_profile,
        }
        if meta["peak_gpu_memory_mib"] > VRAM_CEILING_MIB:
            raise CohortError("Group exceeded VRAM ceiling")
        atomic_json(result_dir / (group + ".json"), meta, refuse=True)
        return meta
    finally:
        if sampler is not None:
            sampler.stop()
        del test_cache, retest_cache, model
        gc.collect()
        if "torch" in sys.modules:
            import torch
            torch.cuda.empty_cache()


def _subject_worker_impl(args):
    run_dir, _, manifest = _load_manifest(args.run_directory, statuses=("prepared", "running"))
    subject = args.subject
    plan_lookup = _plan_lookup(manifest)
    subject_rows = _rows_for_subject(manifest, subject)
    if len(subject_rows) != int(manifest["denominators"]["subject_query_counts"][subject]):
        raise CohortError("Subject query denominator changed")
    completed = []
    for group in GROUP_ORDER:
        rows = [row for row in subject_rows if row["group_name"] == group]
        test_record, test_plan = plan_lookup[(subject, "test", group)]
        retest_record, retest_plan = plan_lookup[(subject, "retest", group)]
        existing = validate_group_result(run_dir, manifest, subject, group, rows, test_record, retest_record)
        if existing is None:
            _write_status(run_dir, {"current_subject": subject, "current_group": group})
            existing = process_group(run_dir, manifest, subject, group, rows, test_record, test_plan, retest_record, retest_plan)
        completed.append(existing)
        current_status = load_json(Path(run_dir) / "cohort_status.json")
        progress = _progress(run_dir, manifest, current_status.get("failed_subjects", []))
        _write_status(run_dir, dict(progress, current_subject=subject, current_group=group))
    result = {
        "schema_version": SCHEMA_VERSION, "cohort_id": COHORT_ID, "status": "success",
        "subject_id": subject, "groups": [item["group_name"] for item in completed],
        "rows": sum(int(item["rows"]) for item in completed), "completed_at": utc_now(),
        "peak_gpu_memory_mib": max(float(item["peak_gpu_memory_mib"]) for item in completed),
        "group_results": [file_identity(Path(run_dir) / "group_results" / subject / (group + ".json")) for group in GROUP_ORDER],
    }
    atomic_json(Path(run_dir) / "subject_results" / (subject + ".json"), result)
    print("Subject {} PASS ({} queries)".format(subject, result["rows"]), flush=True)
    return 0


def subject_worker(args):
    failure_path = Path(args.run_directory) / "subject_results" / (args.subject + ".failure.json")
    heartbeat = Heartbeat(args.run_directory)
    heartbeat.start()
    try:
        result = _subject_worker_impl(args)
        if failure_path.exists():
            failure_path.unlink()
        return result
    except BaseException as exc:
        message = str(exc)
        lowered = message.lower()
        if isinstance(exc, CohortError) or any(token in lowered for token in ("hash changed", "geometry changed", "identity changed", "contract", "incompatible")):
            classification = "contract_or_integrity"
        elif isinstance(exc, (ImportError, OSError)) or any(token in lowered for token in ("cuda is unavailable", "no module named", "driver")):
            classification = "infrastructure"
        else:
            classification = "isolated_runtime"
        atomic_json(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION, "cohort_id": COHORT_ID,
                "status": "failed", "subject_id": args.subject,
                "failure_classification": classification,
                "exception_type": type(exc).__name__, "error": message[-4000:],
                "failed_at": utc_now(),
            },
        )
        print("Subject {} failed ({}): {}".format(args.subject, classification, message), file=sys.stderr, flush=True)
        return 3
    finally:
        heartbeat.stop()


def _subject_completed(run_dir, manifest, subject):
    path = Path(run_dir) / "subject_results" / (subject + ".json")
    if not path.is_file():
        return None
    value = load_json(path)
    expected = int(manifest["denominators"]["subject_query_counts"][subject])
    if value.get("status") != "success" or value.get("subject_id") != subject or value.get("rows") != expected or value.get("groups") != list(GROUP_ORDER):
        raise CohortError("Incompatible subject resume result: {}".format(subject))
    return value


def _progress(run_dir, manifest, failed):
    subject_values = []
    group_values = []
    for subject in manifest["settings"]["subject_order"]:
        value = _subject_completed(run_dir, manifest, subject)
        if value:
            subject_values.append(value)
        for group in GROUP_ORDER:
            path = Path(run_dir) / "group_results" / subject / (group + ".json")
            if path.is_file():
                group_values.append(load_json(path))
    return {
        "subjects_completed": len(subject_values), "groups_completed": len(group_values),
        "queries_completed": sum(int(item.get("rows", 0)) for item in group_values),
        "failed_subjects": sorted(set(failed)),
        "gate_passed": _subject_completed(run_dir, manifest, SUBJECT_GATE) is not None,
    }


def _launch_subject(run_dir, subject, log_path):
    command = [sys.executable, "-m", "tools.quadra.aligned_organ_group_cohort", "_subject", "--run-directory", str(run_dir), "--subject", subject]
    with Path(log_path).open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=SUBJECT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def free_disk_gib(path):
    usage = shutil.disk_usage(str(path))
    return float(usage.free) / (1024.0 ** 3)


def forbidden_outputs(run_dir):
    """Return persisted full-volume/tensor artifacts that violate the contract."""
    forbidden = []
    for path in Path(run_dir).rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith((".npy", ".npz", ".nii", ".nii.gz", ".pt", ".pth", ".ckpt")):
            forbidden.append(str(path))
    return sorted(forbidden)


def run_controller(args):
    run_dir, manifest_path, manifest = _load_manifest(args.run_directory, statuses=("prepared", "running"))
    validate_repository()
    try:
        current_environment = stage3.read_profile_fingerprint(Path(manifest["run_directory"]).parents[2], "uae")
    except stage3.Stage3Error as exc:
        raise CohortError(str(exc))
    if current_environment.get("image_digest") != manifest.get("environment", {}).get("image_digest"):
        raise CohortError("UAE environment profile changed after preparation")
    if free_disk_gib(run_dir) < MIN_FREE_DISK_GIB:
        raise CohortError("Disk guard failed before launch")
    try:
        stage3.require_idle_gpu()
    except stage3.Stage3Error as exc:
        raise CohortError(str(exc))
    status = load_json(run_dir / "cohort_status.json")
    status.update({"status": "running", "controller_pid": os.getpid(), "started_at": status.get("started_at") or utc_now(), "last_error": None})
    atomic_json(run_dir / "cohort_status.json", status)
    manifest["status"] = "running"
    manifest["started_at"] = manifest.get("started_at") or utc_now()
    atomic_json(manifest_path, manifest)
    failed = list(status.get("failed_subjects", []))
    try:
        for subject in manifest["settings"]["subject_order"]:
            retained = forbidden_outputs(run_dir)
            if retained:
                raise CohortError("Forbidden full-volume output retained: {}".format(retained[0]))
            if _subject_completed(run_dir, manifest, subject):
                progress = _progress(run_dir, manifest, failed)
                _write_status(run_dir, dict(progress, current_subject=subject, current_group=None))
                continue
            if free_disk_gib(run_dir) < MIN_FREE_DISK_GIB:
                raise CohortError("Disk guard failed during cohort")
            success = False
            for attempt in (1, 2):
                _write_status(run_dir, {"current_subject": subject, "current_group": None, "attempt": attempt})
                returncode = _launch_subject(run_dir, subject, run_dir / "logs" / (subject + ".log"))
                if returncode == 0 and _subject_completed(run_dir, manifest, subject):
                    success = True
                    break
                if returncode == 124 or (returncode is not None and returncode < 0):
                    raise InfrastructureError("Subject worker was timed out or killed: {}".format(subject))
                failure_path = run_dir / "subject_results" / (subject + ".failure.json")
                if failure_path.is_file():
                    failure = load_json(failure_path)
                    if failure.get("failure_classification") == "contract_or_integrity":
                        raise CohortError("Subject integrity failure: {}: {}".format(subject, failure.get("error")))
                    if failure.get("failure_classification") == "infrastructure":
                        raise InfrastructureError("Subject infrastructure failure: {}: {}".format(subject, failure.get("error")))
            if not success:
                if subject == SUBJECT_GATE:
                    raise CohortError("Subject 030 technical gate failed after retry")
                failed.append(subject)
            progress = _progress(run_dir, manifest, failed)
            _write_status(run_dir, dict(progress, current_subject=subject, current_group=None))
        terminal = "PARTIAL" if failed else "TECHNICAL_PASS"
        retained = forbidden_outputs(run_dir)
        if retained:
            raise CohortError("Forbidden full-volume output retained: {}".format(retained[0]))
        progress = _progress(run_dir, manifest, failed)
        _write_status(run_dir, dict(progress, status=terminal, current_subject=None, current_group=None, completed_at=utc_now()))
        manifest["status"] = terminal.lower()
        manifest["completed_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        print("Cohort controller {}".format(terminal), flush=True)
        return 0 if terminal == "TECHNICAL_PASS" else 4
    except CohortError as exc:
        _write_status(run_dir, {"status": "BLOCKED", "last_error": str(exc), "completed_at": utc_now()})
        manifest["status"] = "blocked"
        manifest["completed_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        raise
    except InfrastructureError as exc:
        _write_status(run_dir, {"status": "INCOMPLETE", "last_error": str(exc), "completed_at": utc_now()})
        manifest["status"] = "incomplete"
        manifest["completed_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        raise


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def gpu_snapshot():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, text=True,
        ).strip()
        name, used, total, utilization = [item.strip() for item in output.split(",")]
        return {"name": name, "memory_used_mib": int(used), "memory_total_mib": int(total), "utilization_percent": int(utilization)}
    except Exception as exc:
        return {"error": str(exc)}


def status_payload(run_dir):
    status = load_json(Path(run_dir) / "cohort_status.json")
    status["controller_process_alive"] = _pid_alive(status.get("controller_pid"))
    status["gpu"] = gpu_snapshot()
    status["disk_free_gib"] = free_disk_gib(run_dir)
    return status


def run_status(args):
    payload = status_payload(Path(args.run_directory).resolve())
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("{}: {}/{} subjects, {}/{} groups, {}/{} queries".format(
            payload["status"], payload["subjects_completed"], payload["subjects_total"],
            payload["groups_completed"], payload["groups_total"],
            payload["queries_completed"], payload["queries_total"],
        ))
    return 0


def _summaries(rows, key_fields):
    import numpy as np

    buckets = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        buckets.setdefault(key, []).append(float(row["cycle_error_mm"]))
    result = []
    for key in sorted(buckets):
        values = np.asarray(buckets[key], dtype=np.float64)
        record = {field: value for field, value in zip(key_fields, key)}
        record.update({"count": len(values), "mean_cycle_error_mm": float(values.mean()), "median_cycle_error_mm": float(np.median(values)), "p95_cycle_error_mm": float(np.percentile(values, 95))})
        result.append(record)
    return result


def run_finalize(args):
    run_dir, manifest_path, manifest = _load_manifest(args.run_directory, statuses=("technical_pass", "partial"))
    status = load_json(run_dir / "cohort_status.json")
    if status.get("status") not in ("TECHNICAL_PASS", "PARTIAL"):
        raise CohortError("Cohort is not ready to finalize")
    retained = forbidden_outputs(run_dir)
    if retained:
        raise CohortError("Forbidden full-volume output retained: {}".format(retained[0]))
    checkpoint_path = run_dir / "checkpoint_summary.json"
    if checkpoint_path.exists():
        raise CohortError("Cohort has already been finalized")
    rows = []
    successful_subjects = []
    for subject in manifest["settings"]["subject_order"]:
        if _subject_completed(run_dir, manifest, subject) is None:
            continue
        successful_subjects.append(subject)
        subject_rows = _rows_for_subject(manifest, subject)
        for group in GROUP_ORDER:
            group_rows = [row for row in subject_rows if row["group_name"] == group]
            test_record, _ = _plan_lookup(manifest)[(subject, "test", group)]
            retest_record, _ = _plan_lookup(manifest)[(subject, "retest", group)]
            validate_group_result(run_dir, manifest, subject, group, group_rows, test_record, retest_record)
            rows.extend(read_csv(run_dir / "group_results" / subject / (group + ".csv")))
    if len(rows) != int(status["queries_completed"]) or len({row["query_id"] for row in rows}) != len(rows):
        raise CohortError("Consolidated row denominator or uniqueness failed")
    consolidated = run_dir / "cycle_error_points.csv"
    atomic_csv(consolidated, rows, fieldnames=list(RESULT_FIELDS), refuse=True)
    summary_outputs = {}
    for name, fields in (("subject", ["subject_id"]), ("group", ["group_name"]), ("mask", ["mask_name"])):
        path = run_dir / (name + "_summary.csv")
        atomic_csv(path, _summaries(rows, fields), refuse=True)
        summary_outputs[name] = file_identity(path)
    report = (
        "# Aligned UAE-S cohort completion\n\n"
        "Status: **{}**\n\n"
        "- Successful subjects: {}/28\n"
        "- Validated queries: {}/110400\n"
        "- Workflow: `organ_group_aligned_100mm_fp32_global_nn`\n"
        "- Registration: deferred; use the immutable frozen query CSV.\n\n"
        "This is a technical cohort execution, not proof of biological or clinical validity. "
        "Organ-group crops impose an anatomical search prior, Stage 5R validation was one-subject, "
        "and fixed-point matching remains deferred.\n"
    ).format(status["status"], len(successful_subjects), len(rows))
    report_path = run_dir / "completion_report.md"
    report_path.write_text(report, encoding="utf-8")
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "cohort_id": COHORT_ID, "status": status["status"],
        "created_at": utc_now(), "subjects_validated": len(successful_subjects),
        "queries_validated": len(rows), "failed_subjects": status.get("failed_subjects", []),
        "gates": {
            "subject_030_gate_passed": status.get("gate_passed") is True,
            "all_completed_outputs_validated": True,
            "query_ids_unique": len({row["query_id"] for row in rows}) == len(rows),
            "complete_denominator": len(rows) == EXPECTED_QUERIES,
            "forbidden_full_volume_outputs_retained": False,
            "registration_executed": False,
        },
        "outputs": {
            "points": file_identity(consolidated), "summaries": summary_outputs,
            "queries": manifest["outputs"]["queries"], "report": file_identity(report_path),
        },
    }
    atomic_json(checkpoint_path, checkpoint, refuse=True)
    manifest["status"] = "finalized"
    manifest["checkpoint"] = file_identity(checkpoint_path)
    atomic_json(manifest_path, manifest)
    print("Cohort finalize {}".format(status["status"]), flush=True)
    print("Checkpoint: {}".format(checkpoint_path), flush=True)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--stage5r-checkpoint", required=True)
    prepare.add_argument("--stage1-selected", required=True)
    prepare.add_argument("--storage-root", default="/workspace/quadra")
    prepare.add_argument("--repository-root", default=str(PROJECT_ROOT))
    prepare.add_argument("--cohort-manifest")
    prepare.add_argument("--registry")
    prepare.add_argument("--config", default="configs/samv2/samv2_NIHLN.py")
    prepare.add_argument("--checkpoint", default="checkpoints/SAMv2_iter_20000.pth")
    prepare.add_argument("--output-root")
    prepare.add_argument("--run-id")
    run = subparsers.add_parser("run")
    run.add_argument("--run-directory", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--run-directory", required=True)
    status.add_argument("--json", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-directory", required=True)
    worker = subparsers.add_parser("_subject")
    worker.add_argument("--run-directory", required=True)
    worker.add_argument("--subject", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            run_prepare(args)
            return 0
        if args.command == "run":
            return run_controller(args)
        if args.command == "status":
            return run_status(args)
        if args.command == "finalize":
            return run_finalize(args)
        if args.command == "_subject":
            return subject_worker(args)
        parser.print_help()
        return 2
    except (CohortError, InfrastructureError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
