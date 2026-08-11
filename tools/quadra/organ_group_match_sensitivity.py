#!/usr/bin/env python
"""Stage 5 match-level crop sensitivity for the provisional 100 mm workflow.

Preparation freezes the Stage 4A raw-ITK points and 100/120 mm plans. GPU
workers extract dense organ-group UAE-S features in FP32-model mode, retain the
architecture's FP16 outputs only in CPU memory, and exhaustively search every
target location with FP32 similarity arithmetic. Selection can freeze only the
global-NN workflow; fixed-point uses eight deliberately bounded sentinels and
therefore remains provisional.

This module stays syntactically compatible with Python 3.7 for the legacy UAE
container.
"""

from __future__ import print_function

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra import memory_configuration_screen as stage3  # noqa: E402
from tools.quadra import organ_group_numerical_validation as stage4  # noqa: E402
from tools.quadra import organ_group_workflow_decision as stage4c  # noqa: E402


SCHEMA_VERSION = 1
VALIDATION_ID = "quadra-organ-group-match-sensitivity-v1"
EXPECTED_BRANCH = "codex/quadra-memory-optimization"
RUN_PREFIX = "stage5-match-sensitivity-"
WORKER_TIMEOUT_SECONDS = 60 * 60
QUERY_BATCH_SIZE = 64
MATCH_CHUNK_XYZ = (32, 32, 16)
VRAM_CEILING_MIB = stage3.VRAM_CEILING_MIB
WITHIN_2MM_RATE_MIN = 0.95
DISPLACEMENT_MEDIAN_MAX_MM = 2.0
DISPLACEMENT_P95_MAX_MM = 4.0
CYCLE_DELTA_MEDIAN_MAX_MM = 1.0
CYCLE_DELTA_P95_MAX_MM = 4.0
MATCH_SCORE_ATOL = 1e-5
FIXED_POINT_MARGIN_XYZ = (2, 2, 2)
FIXED_POINT_ITERATIONS = 4
FIXED_POINT_SCORE_THRESHOLD = 0.8
FIXED_POINT_MAX_RETURN_MM = 100.0


class Stage5Error(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise Stage5Error("Cannot read JSON {}: {}".format(path, exc))
    if not isinstance(value, dict):
        raise Stage5Error("Expected a JSON object: {}".format(path))
    return value


def atomic_json(path, value, refuse=False):
    path = Path(path)
    if refuse and path.exists():
        raise Stage5Error("Refusing to overwrite existing file: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value.rstrip() + "\n")
    os.replace(str(temporary), str(path))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path = Path(path)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(str(temporary), str(path))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_identity(path):
    return stage4.file_identity(Path(path))


def sha256_payload(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_repository(repository=PROJECT_ROOT):
    branch = stage3.git_output(["symbolic-ref", "--short", "HEAD"], repository)
    commit = stage3.git_output(["rev-parse", "HEAD"], repository)
    dirty = stage3.git_output(["status", "--porcelain"], repository)
    ancestor = subprocess.call(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", "e66ebd5", "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0
    if branch != EXPECTED_BRANCH or dirty or not ancestor:
        raise Stage5Error(
            "Repository contract failed: branch={!r}, clean={}, Stage4B ancestor={}".format(
                branch, not bool(dirty), ancestor
            )
        )
    return {
        "path": str(Path(repository).resolve()),
        "branch": branch,
        "execution_commit": commit,
        "clean": True,
    }


def validate_stage4c_checkpoint(path):
    identity = file_identity(path)
    checkpoint = load_json(path)
    gates = checkpoint.get("gates", {})
    decision_ref = checkpoint.get("limitation_acceptance")
    if (
        checkpoint.get("stage") != 4
        or checkpoint.get("substage") != "C"
        or checkpoint.get("status") != "PROVISIONAL"
        or checkpoint.get("decision_id") != stage4c.DECISION_ID
        or checkpoint.get("next_stage") != "match_level_crop_sensitivity_validation"
        or gates.get("stage4a_blocked_status_preserved") is not True
        or gates.get("stage4b_blocked_status_preserved") is not True
        or gates.get("human_limitation_acceptance_recorded") is not True
        or gates.get("100mm_fp32_technical_extraction_gates_passed") is not True
        or gates.get("descriptor_boundary_invariance_established") is not False
        or gates.get("match_level_validation_complete") is not False
        or gates.get("production_workflow_frozen") is not False
        or not isinstance(decision_ref, dict)
        or file_identity(decision_ref.get("path")) != decision_ref
    ):
        raise Stage5Error("Stage 4C checkpoint does not authorize match-level validation")
    decision = load_json(decision_ref["path"])
    selected = decision.get("selected_candidate", {})
    interpretation = decision.get("evidence_interpretation", {})
    if (
        decision.get("status") != "PROVISIONAL_ACCEPTANCE"
        or selected.get("spatial_configuration") != "organ_group_100mm"
        or selected.get("precision") != "fp32"
        or selected.get("spacing_xyz_mm") != [2.0, 2.0, 2.0]
        or selected.get("coordinate_space") != "raw_itk_voxel"
        or selected.get("subject_id") != stage4.EXPECTED_SUBJECT
        or selected.get("groups") != list(stage3.GROUPS)
        or interpretation.get("stage4a_status_preserved") != "BLOCKED"
        or interpretation.get("stage4b_status_preserved") != "BLOCKED"
        or interpretation.get("larger_margin_established_invariance") is not False
    ):
        raise Stage5Error("Stage 4C selected-candidate contract changed")
    stage4a = decision.get("sources", {}).get("stage4a")
    stage4b = decision.get("sources", {}).get("stage4b")
    observed_a = stage4c.validate_stage4a(stage4a.get("checkpoint", {}).get("path"))
    observed_b = stage4c.validate_stage4b(
        stage4b.get("checkpoint", {}).get("path"), observed_a
    )
    if observed_a != stage4a or observed_b != stage4b:
        raise Stage5Error("Stage 4C source evidence changed")
    return identity, checkpoint, decision, observed_a


def apply_affine(points_xyz, affine):
    import numpy as np

    points = np.asarray(points_xyz, dtype=np.float64)
    single = points.ndim == 1
    points = points.reshape(-1, 3)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    transformed = homogeneous.dot(np.asarray(affine, dtype=np.float64).T)[:, :3]
    return transformed[0] if single else transformed


def point_inside_model(raw_xyz, plan, atol=1e-6):
    import numpy as np

    model = apply_affine(raw_xyz, plan["raw_to_model_continuous_affine"])
    shape = np.asarray(plan["padded_shape_xyz"], dtype=np.float64)
    return bool(np.all(model >= -atol) and np.all(model <= shape - 1.0 + atol)), model


def physical_xyz(raw_xyz, plan):
    return apply_affine(raw_xyz, plan["source_ct"]["affine"])


def plan_lookup(stage4a):
    pair_plan = load_json(stage4a["pair_plan"]["path"])
    lookup = {}
    for reference in pair_plan.get("plans", []):
        if file_identity(reference.get("path")) != {
            key: reference[key] for key in ("path", "bytes", "sha256")
        }:
            raise Stage5Error("A frozen Stage 4 plan changed: {}".format(reference.get("path")))
        plan = load_json(reference["path"])
        key = (plan["session"], plan["group_name"], int(round(float(plan["margin_mm"]))))
        if key in lookup:
            raise Stage5Error("Duplicate frozen Stage 4 plan: {}".format(key))
        lookup[key] = (reference, plan)
    expected = set(
        (session, group, margin)
        for session in ("test", "retest")
        for group in stage3.GROUPS
        for margin in (100, 120)
    )
    if set(lookup) != expected:
        raise Stage5Error("Stage 4A does not contain the exact 16 frozen 100/120 mm plans")
    return lookup


def _number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        raise Stage5Error("Invalid numeric field {} in frozen query row".format(key))


def freeze_queries(stage4a, plans, run_dir):
    import numpy as np

    source_path = Path(stage4a["samples"]["path"])
    if file_identity(source_path) != stage4a["samples"]:
        raise Stage5Error("Frozen Stage 4A sample CSV changed")
    rows = read_csv(source_path)
    required = {"point_id", "scan_key", "subject_id", "session", "group_name", "mask_name", "raw_x", "raw_y", "raw_z", "coord_space"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise Stage5Error("Frozen Stage 4A query CSV is incomplete")
    frozen = []
    seen = set()
    for row in rows:
        if row["coord_space"] != "raw_itk_voxel" or row["group_name"] not in stage3.GROUPS:
            raise Stage5Error("Frozen query coordinate or group contract changed")
        key = (row["session"], row["group_name"], row["point_id"])
        if key in seen:
            raise Stage5Error("Duplicate frozen query identity: {}".format(key))
        seen.add(key)
        raw = [_number(row, "raw_x"), _number(row, "raw_y"), _number(row, "raw_z")]
        for margin in (100, 120):
            inside, model = point_inside_model(raw, plans[(row["session"], row["group_name"], margin)][1])
            if not inside:
                raise Stage5Error("Frozen query lies outside the {} mm plan: {}".format(margin, key))
        frozen.append({
            "query_id": "{}-{}-{}".format(row["session"], row["group_name"], row["point_id"]),
            "point_id": int(row["point_id"]),
            "scan_key": row["scan_key"],
            "subject_id": row["subject_id"],
            "source_session": row["session"],
            "group_name": row["group_name"],
            "mask_name": row["mask_name"],
            "raw_x": raw[0], "raw_y": raw[1], "raw_z": raw[2],
            "coord_space": "raw_itk_voxel",
        })
    write_csv(run_dir / "global_nn_queries_raw_itk.csv", frozen)

    sentinels = []
    for group in stage3.GROUPS:
        candidates = [row for row in frozen if row["source_session"] == "test" and row["group_name"] == group]
        if len(candidates) < 2:
            raise Stage5Error("Need at least two Test sentinels for {}".format(group))
        plan = plans[("test", group, 100)][1]
        shape = np.asarray(plan["padded_shape_xyz"], dtype=np.float64)
        centre_model = (shape - 1.0) / 2.0
        centre_raw = apply_affine(centre_model, plan["model_to_raw_continuous_affine"])
        centre_physical = physical_xyz(centre_raw, plan)
        ranked_centre = []
        ranked_boundary = []
        for row in candidates:
            raw = np.asarray([row["raw_x"], row["raw_y"], row["raw_z"]], dtype=np.float64)
            model = apply_affine(raw, plan["raw_to_model_continuous_affine"])
            clearance = float(np.min(np.concatenate((model, shape - 1.0 - model))) * 2.0)
            distance = float(np.linalg.norm(physical_xyz(raw, plan) - centre_physical))
            stable = (int(row["point_id"]), row["query_id"])
            ranked_centre.append((distance, stable, row))
            ranked_boundary.append((clearance, stable, row))
        centre = min(ranked_centre)[2]
        boundary = next(item[2] for item in sorted(ranked_boundary) if item[2]["query_id"] != centre["query_id"])
        for role, row in (("group_centre", centre), ("minimum_100mm_clearance", boundary)):
            sentinels.append(dict(row, sentinel_role=role))
    if len(sentinels) != 8 or len({row["query_id"] for row in sentinels}) != 8:
        raise Stage5Error("Fixed-point sentinel selection did not produce eight unique queries")
    write_csv(run_dir / "fixed_point_sentinels_raw_itk.csv", sentinels)
    return frozen, sentinels


def _run_directory(output_root, run_id=None, resume=None):
    if resume:
        run_dir = Path(resume).resolve()
        if not run_dir.is_dir():
            raise Stage5Error("Resume directory is missing: {}".format(run_dir))
        return run_dir, True
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / (run_id or RUN_PREFIX + timestamp)
    if run_dir.exists():
        raise Stage5Error("Refusing to overwrite existing Stage 5 directory: {}".format(run_dir))
    run_dir.mkdir(parents=True)
    return run_dir, False


def run_prepare(args):
    from tools.quadra import optimization_baseline as baseline

    repository = validate_repository(Path(args.repository_root))
    stage4c_identity, stage4c_checkpoint, decision, stage4a = validate_stage4c_checkpoint(args.stage4c_checkpoint)
    stage4a_manifest = load_json(stage4a["manifest"]["path"])
    baseline_record = stage4a_manifest.get("baseline_manifest")
    baseline.validate_locked_contract(
        Path(baseline_record["path"]),
        repository_root=Path(args.repository_root),
        storage_root=Path(args.storage_root),
        required_profile="preprocess",
    )
    plans = plan_lookup(stage4a)
    output_root = Path(args.output_root or Path(args.storage_root) / "runs/memory_optimization")
    run_dir, resuming = _run_directory(output_root, args.run_id, args.resume_run_directory)
    manifest_path = run_dir / "stage5_manifest.json"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "validation_id": VALIDATION_ID,
        "status": "preparing",
        "created_at": utc_now(),
        "repository": repository,
        "baseline_manifest": baseline_record,
        "stage4c_checkpoint": stage4c_identity,
        "stage4c_decision": stage4c_checkpoint["limitation_acceptance"],
        "stage4a_samples": stage4a["samples"],
        "subject_id": stage4.EXPECTED_SUBJECT,
        "settings": {
            "candidate_margin_mm": 100,
            "reference_margin_mm": 120,
            "precision": "fp32",
            "spacing_xyz_mm": [2.0, 2.0, 2.0],
            "coordinate_space": "raw_itk_voxel",
            "groups": list(stage3.GROUPS),
            "matching_modes": ["global_nn", "fixed_point"],
            "query_batch_size": QUERY_BATCH_SIZE,
            "match_chunk_xyz": list(MATCH_CHUNK_XYZ),
            "fixed_point": {
                "margin_xyz": list(FIXED_POINT_MARGIN_XYZ),
                "iterations": FIXED_POINT_ITERATIONS,
                "score_threshold": FIXED_POINT_SCORE_THRESHOLD,
                "max_return_distance_mm": FIXED_POINT_MAX_RETURN_MM,
                "scope": "eight_bounded_sentinels_only",
            },
            "vram_ceiling_mib": VRAM_CEILING_MIB,
        },
        "scope": {
            "largest_pair_only": True,
            "global_nn_all_frozen_points": True,
            "fixed_point_bounded_diagnostic": True,
            "cohort_authorized": False,
            "embeddings_saved": False,
        },
    }
    if resuming:
        existing = load_json(manifest_path)
        if existing.get("status") != "preparing":
            raise Stage5Error("Completed Stage 5 preparation is immutable")
        for key in ("validation_id", "repository", "baseline_manifest", "stage4c_checkpoint", "stage4c_decision", "stage4a_samples", "subject_id", "settings", "scope"):
            if existing.get(key) != contract.get(key):
                raise Stage5Error("Stage 5 resume contract changed: {}".format(key))
        contract = existing
    else:
        atomic_json(manifest_path, contract, refuse=True)
    plans_dir = run_dir / "plans"
    plans_dir.mkdir(exist_ok=True)
    frozen_plans = []
    for key in sorted(plans):
        reference, plan = plans[key]
        filename = "{}-{}-m{:03d}.json".format(plan["scan_key"], plan["group_name"], key[2])
        target = plans_dir / filename
        if not target.exists():
            atomic_json(target, plan, refuse=True)
        frozen_plans.append(dict(file_identity(target), session=key[0], group_name=key[1], margin_mm=key[2], source_identity=reference))
    global_rows, sentinels = freeze_queries(stage4a, plans, run_dir)
    contract.update({
        "status": "prepared",
        "prepared_at": utc_now(),
        "plans": frozen_plans,
        "global_query_count": len(global_rows),
        "fixed_point_sentinel_count": len(sentinels),
        "outputs": {
            "global_queries": file_identity(run_dir / "global_nn_queries_raw_itk.csv"),
            "fixed_point_sentinels": file_identity(run_dir / "fixed_point_sentinels_raw_itk.csv"),
        },
    })
    atomic_json(manifest_path, contract)
    print("Stage 5 prepare PASS", flush=True)
    print("Run directory: {}".format(run_dir), flush=True)
    return run_dir


class InMemoryUaesCache(object):
    def __init__(self, fine, coarse, semantic, native_shape_xyz, cache_dir="in_memory"):
        import numpy as np

        self.fine = fine
        self.coarse = coarse
        self.semantic = semantic
        self._native_shape_xyz = tuple(int(value) for value in native_shape_xyz)
        self.cache_dir = cache_dir
        self._norm_ratio_xyz = np.ones(3, dtype=np.float64)

    def feature_shape_xyz(self, level):
        value = getattr(self, level)
        return int(value.shape[3]), int(value.shape[2]), int(value.shape[1])

    @property
    def native_shape_xyz(self):
        return self._native_shape_xyz

    @property
    def norm_ratio_xyz(self):
        return self._norm_ratio_xyz

    @property
    def manifest(self):
        return {"native_spacing_xyz": [2.0, 2.0, 2.0], "norm_ratio_xyz": [1.0, 1.0, 1.0]}

    def valid_array(self, level):
        return getattr(self, level)


def _extract_cache(model, plan):
    import numpy as np
    import torch

    started = time.time()
    data = stage3._legacy_prepare(plan)
    preprocessing = time.time() - started
    tensor = torch.from_numpy(data)[None, None].cuda(non_blocking=False)
    del data
    torch.cuda.reset_peak_memory_stats()
    forward_started = time.time()
    with torch.no_grad():
        outputs = model.extract_feat(tensor)
    torch.cuda.synchronize()
    forward = time.time() - forward_started
    expected = stage3._expected_feature_shapes([int(value) for value in tensor.shape])
    arrays = []
    output_records = []
    for name, value in zip(("fine", "coarse", "semantic"), outputs):
        record = {
            "name": name,
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all().item()),
        }
        if record["shape"] != expected[name] or record["dtype"] != "torch.float16" or not record["finite"]:
            raise Stage5Error("UAE-S output contract failed for {}".format(name))
        arrays.append(value[0].detach().cpu().numpy())
        output_records.append(record)
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    del tensor, outputs
    torch.cuda.empty_cache()
    cache = InMemoryUaesCache(
        arrays[0], arrays[1], arrays[2], plan["padded_shape_xyz"],
        cache_dir="{}/{}mm".format(plan["scan_key"], int(round(plan["margin_mm"]))),
    )
    return cache, {
        "preprocessing_seconds": preprocessing,
        "forward_seconds": forward,
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "features": output_records,
    }


def _raw_to_model_index(raw_xyz, plan):
    import numpy as np

    model = apply_affine(raw_xyz, plan["raw_to_model_continuous_affine"])
    index = np.rint(model).astype(np.int64)
    shape = np.asarray(plan["padded_shape_xyz"], dtype=np.int64)
    if np.any(index < 0) or np.any(index >= shape):
        raise Stage5Error("Raw query rounds outside the model grid")
    return index


def _model_to_raw(model_xyz, plan):
    return apply_affine(model_xyz, plan["model_to_raw_continuous_affine"])


def _cycle_mm(raw_query, raw_back, source_plan):
    import numpy as np

    return float(np.linalg.norm(physical_xyz(raw_query, source_plan) - physical_xyz(raw_back, source_plan)))


def _global_records(source_session, group, margin, source_cache, target_cache, source_plan, target_plan, rows):
    import numpy as np
    from tools.quadra.streaming_cycle_error import stream_global_match_uaes

    query_raw = np.asarray([[float(row[axis]) for axis in ("raw_x", "raw_y", "raw_z")] for row in rows], dtype=np.float64)
    query_model = np.stack([_raw_to_model_index(point, source_plan) for point in query_raw])
    target_model, score_forward, profile_forward = stream_global_match_uaes(
        source_cache, target_cache, query_model, QUERY_BATCH_SIZE, MATCH_CHUNK_XYZ, output_space="native"
    )
    back_model, score_back, profile_back = stream_global_match_uaes(
        target_cache, source_cache, target_model, QUERY_BATCH_SIZE, MATCH_CHUNK_XYZ, output_space="native"
    )
    records = []
    for index, row in enumerate(rows):
        matched_raw = _model_to_raw(target_model[index], target_plan)
        returned_raw = _model_to_raw(back_model[index], source_plan)
        records.append({
            "query_id": row["query_id"], "point_id": int(row["point_id"]),
            "source_session": source_session,
            "target_session": "retest" if source_session == "test" else "test",
            "group_name": group, "mask_name": row["mask_name"], "margin_mm": margin,
            "query_raw_xyz": query_raw[index].tolist(),
            "matched_raw_xyz": matched_raw.tolist(),
            "returned_raw_xyz": returned_raw.tolist(),
            "matched_physical_xyz": physical_xyz(matched_raw, target_plan).tolist(),
            "returned_physical_xyz": physical_xyz(returned_raw, source_plan).tolist(),
            "cycle_error_mm": _cycle_mm(query_raw[index], returned_raw, source_plan),
            "score_forward": float(score_forward[index]), "score_back": float(score_back[index]),
            "status": "success",
        })
    return records, {"forward": profile_forward, "backward": profile_back}


def _fixed_records(group, margin, test_cache, retest_cache, test_plan, retest_plan, rows):
    import numpy as np
    from tools.quadra.uaes_matching import FixedPointSettings, fixed_point_match_batch

    settings = FixedPointSettings(
        margin_xyz=FIXED_POINT_MARGIN_XYZ,
        iterations=FIXED_POINT_ITERATIONS,
        score_threshold=FIXED_POINT_SCORE_THRESHOLD,
        max_return_distance_mm=FIXED_POINT_MAX_RETURN_MM,
    )
    query_raw = np.asarray([[float(row[axis]) for axis in ("raw_x", "raw_y", "raw_z")] for row in rows], dtype=np.float64)
    query_model = np.stack([_raw_to_model_index(point, test_plan) for point in query_raw])
    forward, forward_profile = fixed_point_match_batch(
        test_cache, retest_cache, query_model, settings, QUERY_BATCH_SIZE, MATCH_CHUNK_XYZ
    )
    successful = [index for index, result in enumerate(forward) if result["status"] == "success"]
    backward_by_index = {}
    backward_profile = None
    if successful:
        targets = np.stack([forward[index]["point_xyz"] for index in successful])
        backward, backward_profile = fixed_point_match_batch(
            retest_cache, test_cache, targets, settings, QUERY_BATCH_SIZE, MATCH_CHUNK_XYZ
        )
        backward_by_index = dict(zip(successful, backward))
    records = []
    for index, row in enumerate(rows):
        first = forward[index]
        second = backward_by_index.get(index)
        success = first["status"] == "success" and second is not None and second["status"] == "success"
        matched_raw = _model_to_raw(first["point_xyz"], retest_plan) if first["point_xyz"] is not None else None
        returned_raw = _model_to_raw(second["point_xyz"], test_plan) if second is not None and second["point_xyz"] is not None else None
        records.append({
            "query_id": row["query_id"], "point_id": int(row["point_id"]),
            "sentinel_role": row["sentinel_role"], "source_session": "test", "target_session": "retest",
            "group_name": group, "mask_name": row["mask_name"], "margin_mm": margin,
            "query_raw_xyz": query_raw[index].tolist(),
            "matched_raw_xyz": None if matched_raw is None else matched_raw.tolist(),
            "returned_raw_xyz": None if returned_raw is None else returned_raw.tolist(),
            "matched_physical_xyz": None if matched_raw is None else physical_xyz(matched_raw, retest_plan).tolist(),
            "returned_physical_xyz": None if returned_raw is None else physical_xyz(returned_raw, test_plan).tolist(),
            "cycle_error_mm": None if returned_raw is None else _cycle_mm(query_raw[index], returned_raw, test_plan),
            "status": "success" if success else "failed",
            "forward_status": first["status"], "backward_status": None if second is None else second["status"],
            "failure_reason": first.get("failure_reason") or (None if second is None else second.get("failure_reason")),
            "stable_anchor_count_forward": first.get("stable_anchor_count"),
            "stable_anchor_count_back": None if second is None else second.get("stable_anchor_count"),
        })
    return records, {"forward": forward_profile, "backward": backward_profile}


def _worker_signature(kind, plan_refs, queries, config, checkpoint):
    return sha256_payload({
        "validation_id": VALIDATION_ID, "kind": kind, "plans": plan_refs,
        "queries": queries, "config": config, "checkpoint": checkpoint,
        "query_batch_size": QUERY_BATCH_SIZE, "match_chunk_xyz": MATCH_CHUNK_XYZ,
    })


def run_worker(args):
    result_path = Path(args.result_path)
    started = time.time()
    sampler = None
    result = {
        "schema_version": SCHEMA_VERSION, "validation_id": VALIDATION_ID,
        "kind": "group_margin", "status": "running", "started_at": utc_now(),
        "worker_signature": args.worker_signature, "pid": os.getpid(),
    }
    try:
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise Stage5Error("CUDA is unavailable")
        torch.manual_seed(stage3.SEED); np.random.seed(stage3.SEED)
        torch.backends.cudnn.benchmark = stage3.CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = stage3.CUDNN_DETERMINISTIC
        test_plan = load_json(args.test_plan); retest_plan = load_json(args.retest_plan)
        group = test_plan["group_name"]; margin = int(round(test_plan["margin_mm"]))
        if retest_plan["group_name"] != group or int(round(retest_plan["margin_mm"])) != margin:
            raise Stage5Error("Worker Test/Retest plans disagree")
        config = Path(args.config); checkpoint = Path(args.checkpoint)
        if stage4.sha256_file(config) != stage3.EXPECTED_CONFIG_SHA256 or stage4.sha256_file(checkpoint) != stage3.EXPECTED_CHECKPOINT_SHA256:
            raise Stage5Error("Config or checkpoint hash mismatch")
        model_started = time.time()
        model, hook_present = stage3._load_model(config, checkpoint, "fp32")
        torch.cuda.synchronize(); model_seconds = time.time() - model_started
        if str(next(model.parameters()).dtype) != "torch.float32" or not hook_present:
            raise Stage5Error("FP32 model precision contract failed")
        sampler = stage3.NvidiaProcessSampler(os.getpid()); sampler.start()
        test_cache, test_extract = _extract_cache(model, test_plan)
        retest_cache, retest_extract = _extract_cache(model, retest_plan)
        queries = read_csv(args.queries)
        global_records = []
        profiles = {}
        for source_session, source_cache, target_cache, source_plan, target_plan in (
            ("test", test_cache, retest_cache, test_plan, retest_plan),
            ("retest", retest_cache, test_cache, retest_plan, test_plan),
        ):
            subset = [row for row in queries if row["source_session"] == source_session and row["group_name"] == group]
            records, profile = _global_records(source_session, group, margin, source_cache, target_cache, source_plan, target_plan, subset)
            global_records.extend(records); profiles["global_{}".format(source_session)] = profile
        sentinels = [row for row in read_csv(args.sentinels) if row["group_name"] == group]
        fixed_records, fixed_profile = _fixed_records(group, margin, test_cache, retest_cache, test_plan, retest_plan, sentinels)
        profiles["fixed_point"] = fixed_profile
        sampler.stop(); process_peak = sampler.maximum; sampler = None
        peaks = [
            float(test_extract["torch_peak_reserved_bytes"]) / 1048576.0,
            float(retest_extract["torch_peak_reserved_bytes"]) / 1048576.0,
        ]
        for profile in profiles.values():
            for direction in ("forward", "backward"):
                item = profile.get(direction) if isinstance(profile, dict) else None
                if isinstance(item, dict) and item.get("peak_gpu_memory_bytes") is not None:
                    peaks.append(float(item["peak_gpu_memory_bytes"]) / 1048576.0)
        if process_peak is not None:
            peaks.append(float(process_peak))
        result.update({
            "status": "success", "failure_classification": None,
            "scan_key_test": test_plan["scan_key"], "scan_key_retest": retest_plan["scan_key"],
            "group_name": group, "margin_mm": margin,
            "model_dtype": "torch.float32", "returned_embedding_dtype": "torch.float16",
            "global_records": global_records, "fixed_records": fixed_records,
            "extractions": {"test": test_extract, "retest": retest_extract},
            "profiles": profiles, "model_load_seconds": model_seconds,
            "measured_peak_mib": max(peaks),
            "process_gpu_peak_mib": process_peak,
            "cpu_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        })
        del test_cache, retest_cache, model
        gc.collect(); torch.cuda.empty_cache()
    except RuntimeError as exc:
        message = str(exc); oom = "out of memory" in message.lower() and "cuda" in message.lower()
        result.update({"status": "failed", "failure_classification": "cuda_oom" if oom else "model_error", "error": message[-4000:]})
    except Exception as exc:
        result.update({"status": "failed", "failure_classification": "environment_error", "error": repr(exc)})
    finally:
        if sampler is not None:
            sampler.stop()
    result["completed_at"] = utc_now(); result["wall_time_seconds"] = time.time() - started
    atomic_json(result_path, result)
    return 0 if result.get("status") == "success" else 3


def _bounded_cache(cache, size_xyz=(24, 24, 12)):
    import numpy as np

    fine_shape = np.asarray(cache.feature_shape_xyz("fine"), dtype=np.int64)
    size = np.minimum(fine_shape, np.asarray(size_xyz, dtype=np.int64))
    start = (fine_shape - size) // 2
    stop = start + size
    fine = cache.fine[:, start[2]:stop[2], start[1]:stop[1], start[0]:stop[0]]
    semantic = cache.semantic[:, start[2]:stop[2], start[1]:stop[1], start[0]:stop[0]]
    coarse_shape = np.asarray(cache.feature_shape_xyz("coarse"), dtype=np.int64)
    coarse_start = np.floor(start.astype(np.float64) * coarse_shape / fine_shape).astype(np.int64)
    coarse_stop = np.ceil(stop.astype(np.float64) * coarse_shape / fine_shape).astype(np.int64)
    coarse = cache.coarse[:, coarse_start[2]:coarse_stop[2], coarse_start[1]:coarse_stop[1], coarse_start[0]:coarse_stop[0]]
    bounded = InMemoryUaesCache(fine, coarse, semantic, size, cache_dir="bounded_real_embedding")
    bounded._norm_ratio_xyz = np.asarray([2.0, 2.0, 2.0], dtype=np.float64)
    return bounded


def run_equivalence_worker(args):
    result_path = Path(args.result_path)
    result = {"schema_version": SCHEMA_VERSION, "validation_id": VALIDATION_ID, "kind": "bounded_equivalence", "status": "running", "worker_signature": args.worker_signature}
    try:
        import numpy as np
        import torch
        from tools.quadra.streaming_cycle_error import stream_global_match_uaes
        from tools.quadra.uaes_matching import FixedPointSettings, fixed_point_match_batch
        from tools.quadra.validate_uaes_streaming import dense_match_uaes

        if not torch.cuda.is_available():
            raise Stage5Error("CUDA is unavailable")
        plan = load_json(args.plan)
        model, _ = stage3._load_model(Path(args.config), Path(args.checkpoint), "fp32")
        cache, extraction = _extract_cache(model, plan)
        bounded = _bounded_cache(cache)
        shape = np.asarray(bounded.native_shape_xyz, dtype=np.int64)
        queries = np.asarray([shape // 2, np.minimum(shape - 1, shape // 2 + 1)], dtype=np.int64)
        dense_points, dense_scores, _ = dense_match_uaes(bounded, bounded, queries, 2, device="cuda:0", output_space="fine")
        stream_points, stream_scores, _ = stream_global_match_uaes(bounded, bounded, queries, 2, MATCH_CHUNK_XYZ, output_space="fine")
        coordinate_rate = float(np.mean(np.all(dense_points == stream_points, axis=1)))
        max_score = float(np.max(np.abs(dense_scores - stream_scores)))
        settings = FixedPointSettings(margin_xyz=FIXED_POINT_MARGIN_XYZ, iterations=FIXED_POINT_ITERATIONS, score_threshold=FIXED_POINT_SCORE_THRESHOLD, max_return_distance_mm=FIXED_POINT_MAX_RETURN_MM)
        dense_fixed, dense_profile = fixed_point_match_batch(bounded, bounded, queries[:1], settings, 64, MATCH_CHUNK_XYZ, match_function=dense_match_uaes)
        stream_fixed, stream_profile = fixed_point_match_batch(bounded, bounded, queries[:1], settings, 64, MATCH_CHUNK_XYZ)
        hashes_equal = all(
            left["matched_fine_sha256"] == right["matched_fine_sha256"]
            for left, right in zip(dense_profile["iterations"], stream_profile["iterations"])
        )
        final_distance = None
        if dense_fixed[0]["point_xyz"] is not None and stream_fixed[0]["point_xyz"] is not None:
            final_distance = float(np.linalg.norm(dense_fixed[0]["point_xyz"] - stream_fixed[0]["point_xyz"]))
        passed = coordinate_rate == 1.0 and max_score <= MATCH_SCORE_ATOL and hashes_equal and final_distance is not None and final_distance <= 1.0
        result.update({
            "status": "success" if passed else "failed",
            "failure_classification": None if passed else "matcher_equivalence",
            "global_argmax_coordinate_rate": coordinate_rate,
            "global_max_score_abs_difference": max_score,
            "fixed_internal_hashes_equal": hashes_equal,
            "fixed_final_distance_model_voxels": final_distance,
            "extraction": extraction,
        })
        del bounded, cache, model; gc.collect(); torch.cuda.empty_cache()
    except RuntimeError as exc:
        message = str(exc); oom = "out of memory" in message.lower() and "cuda" in message.lower()
        result.update({"status": "failed", "failure_classification": "cuda_oom" if oom else "model_error", "error": message[-4000:]})
    except Exception as exc:
        result.update({"status": "failed", "failure_classification": "environment_error", "error": repr(exc)})
    result["completed_at"] = utc_now()
    atomic_json(result_path, result)
    return 0 if result.get("status") == "success" else 3


def _load_prepared(run_dir, storage_root, profile):
    manifest_path = Path(run_dir) / "stage5_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") not in ("prepared", "benchmarked") or manifest.get("validation_id") != VALIDATION_ID:
        raise Stage5Error("Stage 5 preparation is incomplete")
    repository = validate_repository(PROJECT_ROOT)
    if repository["execution_commit"] != manifest.get("repository", {}).get("execution_commit"):
        raise Stage5Error("Repository commit differs from Stage 5 preparation")
    validate_stage4c_checkpoint(manifest["stage4c_checkpoint"]["path"])
    profile_record = stage3.read_profile_fingerprint(Path(storage_root), profile)
    baseline = stage3.require_model_contract(manifest["baseline_manifest"]["path"])
    stage3.require_gpu_matches_baseline(baseline, profile_record)
    return manifest_path, manifest, profile_record


def _frozen_plan_paths(run_dir, manifest=None):
    lookup = {}
    for path in sorted((Path(run_dir) / "plans").glob("*.json")):
        plan = load_json(path)
        key = (plan["session"], plan["group_name"], int(round(plan["margin_mm"])))
        lookup[key] = path
    if len(lookup) != 16:
        raise Stage5Error("Expected 16 frozen Stage 5 plans")
    if manifest is not None:
        expected = {Path(item["path"]).resolve(): item for item in manifest.get("plans", [])}
        observed = {path.resolve(): file_identity(path) for path in lookup.values()}
        if len(expected) != 16 or set(expected) != set(observed):
            raise Stage5Error("Stage 5 frozen-plan inventory changed")
        for path, identity in observed.items():
            recorded = expected[path]
            if any(recorded.get(key) != identity[key] for key in ("path", "bytes", "sha256")):
                raise Stage5Error("A frozen Stage 5 plan changed: {}".format(path))
    return lookup


def _validate_frozen_outputs(manifest):
    for name in ("global_queries", "fixed_point_sentinels"):
        record = manifest.get("outputs", {}).get(name)
        if not isinstance(record, dict) or file_identity(record.get("path")) != record:
            raise Stage5Error("Frozen Stage 5 output changed: {}".format(name))


def _launch(command, result_path, log_path, timeout):
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            returncode = None
    if not Path(result_path).is_file():
        classification = "timeout" if returncode is None else ("process_kill" if returncode < 0 else "process_crash")
        atomic_json(result_path, {"schema_version": SCHEMA_VERSION, "validation_id": VALIDATION_ID, "status": "failed", "failure_classification": classification, "returncode": returncode, "completed_at": utc_now()})
    return load_json(result_path)


def run_benchmark(args):
    run_dir = Path(args.run_directory).resolve()
    manifest_path, manifest, profile = _load_prepared(run_dir, args.storage_root, "uae")
    if (run_dir / "checkpoint_summary.json").exists():
        raise Stage5Error("Stage 5 already has a final checkpoint")
    stage3.require_idle_gpu()
    _validate_frozen_outputs(manifest)
    plans = _frozen_plan_paths(run_dir, manifest)
    result_dir = run_dir / "worker_results"; result_dir.mkdir(exist_ok=True)
    logs_dir = run_dir / "worker_logs"; logs_dir.mkdir(exist_ok=True)
    config = file_identity(args.config); checkpoint = file_identity(args.checkpoint)
    smallest = min((path for key, path in plans.items() if key[2] == 100), key=lambda path: load_json(path)["padded_2mm_voxels"])
    equivalence_path = result_dir / "bounded_equivalence.json"
    equivalence_signature = _worker_signature("bounded_equivalence", [file_identity(smallest)], None, config, checkpoint)
    if equivalence_path.exists():
        equivalence = load_json(equivalence_path)
        if equivalence.get("worker_signature") != equivalence_signature:
            raise Stage5Error("Incompatible bounded-equivalence resume result")
    else:
        command = [sys.executable, "-m", "tools.quadra.organ_group_match_sensitivity", "_equivalence_worker", "--plan", str(smallest), "--config", str(args.config), "--checkpoint", str(args.checkpoint), "--result-path", str(equivalence_path), "--worker-signature", equivalence_signature]
        equivalence = _launch(command, equivalence_path, logs_dir / "bounded_equivalence.log", args.timeout_seconds)
    results = []
    if equivalence.get("status") == "success":
        for group in stage3.GROUPS:
            for margin in (100, 120):
                test_plan = plans[("test", group, margin)]; retest_plan = plans[("retest", group, margin)]
                signature = _worker_signature("group_margin", [file_identity(test_plan), file_identity(retest_plan)], [manifest["outputs"]["global_queries"], manifest["outputs"]["fixed_point_sentinels"]], config, checkpoint)
                result_path = result_dir / "{}-m{:03d}.json".format(group, margin)
                if result_path.exists():
                    result = load_json(result_path)
                    if result.get("worker_signature") != signature:
                        raise Stage5Error("Incompatible resumable worker result: {}".format(result_path))
                else:
                    stage3.require_idle_gpu()
                    command = [sys.executable, "-m", "tools.quadra.organ_group_match_sensitivity", "_worker", "--test-plan", str(test_plan), "--retest-plan", str(retest_plan), "--queries", manifest["outputs"]["global_queries"]["path"], "--sentinels", manifest["outputs"]["fixed_point_sentinels"]["path"], "--config", str(args.config), "--checkpoint", str(args.checkpoint), "--result-path", str(result_path), "--worker-signature", signature]
                    result = _launch(command, result_path, logs_dir / "{}-m{:03d}.log".format(group, margin), args.timeout_seconds)
                results.append(result)
                if result.get("status") != "success":
                    break
            if len(results) and results[-1].get("status") != "success":
                break
    memory_rows = []
    for result in results:
        memory_rows.append({
            "group_name": result.get("group_name", ""), "margin_mm": result.get("margin_mm", ""),
            "status": result.get("status", ""), "failure_classification": result.get("failure_classification") or "",
            "measured_peak_mib": result.get("measured_peak_mib", ""), "process_gpu_peak_mib": result.get("process_gpu_peak_mib", ""),
            "wall_time_seconds": result.get("wall_time_seconds", ""),
        })
    write_csv(run_dir / "memory_profile.csv", memory_rows)
    manifest.update({
        "status": "benchmarked", "benchmarked_at": utc_now(), "uae_profile": profile,
        "equivalence_result": file_identity(equivalence_path),
        "worker_result_count": len(results),
        "worker_results": [file_identity(path) for path in sorted(result_dir.glob("*-m*.json"))],
        "outputs": dict(manifest["outputs"], memory_profile=file_identity(run_dir / "memory_profile.csv")),
    })
    atomic_json(manifest_path, manifest)
    print("Stage 5 benchmark complete; run select under the preprocessing profile", flush=True)
    print("Run directory: {}".format(run_dir), flush=True)
    return run_dir


def percentile(values, q):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if len(values) else float("nan")


def _distance(left, right):
    import numpy as np
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def _comparison_rows(results, mode):
    by_key = {}
    field = "global_records" if mode == "global_nn" else "fixed_records"
    for result in results:
        for row in result.get(field, []):
            by_key[(row["query_id"], int(row["margin_mm"]))] = row
    rows = []
    query_ids = sorted({key[0] for key in by_key})
    for query_id in query_ids:
        selected = by_key.get((query_id, 100)); reference = by_key.get((query_id, 120))
        if selected is None or reference is None:
            continue
        paired = selected.get("status") == "success" and reference.get("status") == "success"
        forward = _distance(selected["matched_physical_xyz"], reference["matched_physical_xyz"]) if paired else None
        backward = _distance(selected["returned_physical_xyz"], reference["returned_physical_xyz"]) if paired else None
        cycle_delta = abs(float(selected["cycle_error_mm"]) - float(reference["cycle_error_mm"])) if paired else None
        rows.append({
            "query_id": query_id, "point_id": selected["point_id"],
            "source_session": selected["source_session"], "target_session": selected["target_session"],
            "group_name": selected["group_name"], "mask_name": selected["mask_name"],
            "sentinel_role": selected.get("sentinel_role", ""),
            "status_100mm": selected.get("status"), "status_120mm": reference.get("status"),
            "paired_success": paired,
            "forward_displacement_mm": forward, "backward_displacement_mm": backward,
            "cycle_error_100mm": selected.get("cycle_error_mm"), "cycle_error_120mm": reference.get("cycle_error_mm"),
            "cycle_error_abs_delta_mm": cycle_delta,
            "stable_anchor_count_100mm": selected.get("stable_anchor_count_forward", ""),
            "stable_anchor_count_120mm": reference.get("stable_anchor_count_forward", ""),
        })
    return rows


def _outside_100mm(rows, plans):
    result_lookup = {}
    for result in rows:
        for record in result.get("global_records", []):
            result_lookup[(record["query_id"], int(record["margin_mm"]))] = record
    outside = {}
    for query_id in {key[0] for key in result_lookup}:
        reference = result_lookup.get((query_id, 120))
        if reference is None:
            continue
        target_session = reference["target_session"]; group = reference["group_name"]
        plan = plans[(target_session, group, 100)]
        inside, _ = point_inside_model(reference["matched_raw_xyz"], plan)
        outside[query_id] = not inside
    return outside


def summarize_global(rows):
    import numpy as np

    summaries = []
    for label, subset in [("ALL_GROUPS", rows)] + [(group, [row for row in rows if row["group_name"] == group]) for group in stage3.GROUPS]:
        paired = [row for row in subset if row["paired_success"]]
        directional = np.asarray([value for row in paired for value in (float(row["forward_displacement_mm"]), float(row["backward_displacement_mm"]))], dtype=np.float64)
        cycle = np.asarray([float(row["cycle_error_abs_delta_mm"]) for row in paired], dtype=np.float64)
        complete = len(paired) == len(subset) and len(subset) > 0
        within = float(np.mean(directional <= 2.0)) if len(directional) else 0.0
        row = {
            "scope": label, "queries": len(subset), "paired_successes": len(paired),
            "complete": complete, "directional_within_2mm_rate": within,
            "displacement_median_mm": percentile(directional, 50), "displacement_p95_mm": percentile(directional, 95),
            "cycle_delta_median_mm": percentile(cycle, 50), "cycle_delta_p95_mm": percentile(cycle, 95),
        }
        row["passed"] = bool(
            complete
            and row["displacement_median_mm"] <= DISPLACEMENT_MEDIAN_MAX_MM
            and row["displacement_p95_mm"] <= DISPLACEMENT_P95_MAX_MM
            and row["cycle_delta_median_mm"] <= CYCLE_DELTA_MEDIAN_MAX_MM
            and row["cycle_delta_p95_mm"] <= CYCLE_DELTA_P95_MAX_MM
            and (label != "ALL_GROUPS" or within >= WITHIN_2MM_RATE_MIN)
        )
        summaries.append(row)
    return summaries


def render_report(status, global_summary, fixed_rows, outside_count, technical_failures):
    lines = [
        "# Stage 5 organ-group match sensitivity", "", "## Outcome", "",
        "Stage 5 status: **{}**.".format(status), "",
        "Stage 4A and Stage 4B remain BLOCKED at descriptor level. This stage tests whether the 100-120 mm crop change materially alters matching outputs.", "",
        "## Global-NN sensitivity", "",
        "| Scope | Queries | Complete | Direction <=2 mm | Median displacement | P95 displacement | Median cycle delta | P95 cycle delta | Pass |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in global_summary:
        lines.append("| {scope} | {queries} | {complete} | {directional_within_2mm_rate:.4f} | {displacement_median_mm:.4f} | {displacement_p95_mm:.4f} | {cycle_delta_median_mm:.4f} | {cycle_delta_p95_mm:.4f} | {passed} |".format(**row))
    lines.extend(["", "120 mm matches outside the corresponding 100 mm crop: **{}**.".format(outside_count), "", "## Fixed-point diagnostic", "", "Eight deterministic Test sentinels were evaluated. This bounded evidence cannot authorize cohort fixed-point analysis.", "", "Paired sentinel rows: **{}**.".format(sum(bool(row["paired_success"]) for row in fixed_rows)), "", "## Technical failures", "", ", ".join(technical_failures) if technical_failures else "None.", "", "## Interpretation", "", "A PASS freezes only `organ_group_100mm_fp32_global_nn` for a later largest-pair cycle-error pilot. It does not retrospectively make Stage 4 pass and does not authorize a cohort run."])
    return "\n".join(lines)


def make_figures(run_dir, global_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = Path(run_dir) / "figures"; figures.mkdir(exist_ok=True)
    directional = [float(value) for row in global_rows if row["paired_success"] for value in (row["forward_displacement_mm"], row["backward_displacement_mm"])]
    cycle = [float(row["cycle_error_abs_delta_mm"]) for row in global_rows if row["paired_success"]]
    for name, values, label in (("correspondence_displacement.png", directional, "100-120 mm correspondence displacement (mm)"), ("cycle_error_delta.png", cycle, "Absolute cycle-error difference (mm)")):
        fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(values, bins=30); ax.set_xlabel(label); ax.set_ylabel("Count"); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(figures / name, dpi=160); plt.close(fig)
    return {path.name: file_identity(path) for path in figures.glob("*.png")}


def run_select(args):
    run_dir = Path(args.run_directory).resolve()
    manifest_path, manifest, profile = _load_prepared(run_dir, args.storage_root, "preprocess")
    if manifest.get("status") != "benchmarked":
        raise Stage5Error("Stage 5 benchmark is incomplete")
    selected_path = run_dir / "selected_stage5_workflow.json"; checkpoint_path = run_dir / "checkpoint_summary.json"
    if selected_path.exists() or checkpoint_path.exists():
        raise Stage5Error("Stage 5 selection is immutable")
    _validate_frozen_outputs(manifest)
    if file_identity(manifest["equivalence_result"]["path"]) != manifest["equivalence_result"]:
        raise Stage5Error("Bounded matcher-equivalence evidence changed")
    equivalence = load_json(manifest["equivalence_result"]["path"])
    results = []
    for item in manifest.get("worker_results", []):
        if file_identity(item["path"]) != item:
            raise Stage5Error("A Stage 5 worker result changed: {}".format(item["path"]))
        results.append(load_json(item["path"]))
    technical_failures = []
    if equivalence.get("status") != "success":
        technical_failures.append("bounded_matcher_equivalence")
    if len(results) != 8:
        technical_failures.append("incomplete_worker_count")
    for result in results:
        if result.get("status") != "success":
            technical_failures.append("{}:{}".format(result.get("group_name", "unknown"), result.get("failure_classification", "failed")))
        if float(result.get("measured_peak_mib", float("inf"))) > VRAM_CEILING_MIB:
            technical_failures.append("memory_ceiling")
    retained = forbidden_outputs(run_dir)
    if retained:
        technical_failures.append("forbidden_full_volume_output_retained")
    plans = {key: load_json(path) for key, path in _frozen_plan_paths(run_dir, manifest).items()}
    global_rows = _comparison_rows(results, "global_nn")
    fixed_rows = _comparison_rows(results, "fixed_point")
    if len(global_rows) != int(manifest.get("global_query_count", -1)):
        technical_failures.append("global_query_denominator")
    if len(fixed_rows) != int(manifest.get("fixed_point_sentinel_count", -1)):
        technical_failures.append("fixed_point_sentinel_denominator")
    outside = _outside_100mm(results, plans)
    for row in global_rows:
        row["matched_120mm_outside_100mm_crop"] = bool(outside.get(row["query_id"], False))
    outside_count = sum(outside.values())
    global_summary = summarize_global(global_rows) if global_rows else []
    global_passed = bool(not technical_failures and len(global_summary) == 5 and all(row["passed"] for row in global_summary) and outside_count == 0)
    fixed_concerns = []
    for row in fixed_rows:
        if row["status_100mm"] != row["status_120mm"]:
            fixed_concerns.append("{}:status_disagreement".format(row["query_id"]))
        if row["paired_success"]:
            if max(float(row["forward_displacement_mm"]), float(row["backward_displacement_mm"]), float(row["cycle_error_abs_delta_mm"])) > 4.0:
                fixed_concerns.append("{}:greater_than_4mm".format(row["query_id"]))
            left = row["stable_anchor_count_100mm"]; right = row["stable_anchor_count_120mm"]
            if left not in (None, "") and right not in (None, "") and int(left) < int(right):
                fixed_concerns.append("{}:fewer_100mm_stable_anchors".format(row["query_id"]))
    status = "INCOMPLETE" if technical_failures else ("PASS" if global_passed else "BLOCKED")
    write_csv(run_dir / "global_nn_sensitivity.csv", global_rows)
    write_csv(run_dir / "fixed_point_sentinel_sensitivity.csv", fixed_rows)
    write_csv(run_dir / "global_nn_summary.csv", global_summary)
    figures = make_figures(run_dir, global_rows)
    report_path = run_dir / "match_sensitivity_report.md"
    atomic_text(report_path, render_report(status, global_summary, fixed_rows, outside_count, technical_failures))
    selection = {
        "schema_version": SCHEMA_VERSION, "validation_id": VALIDATION_ID,
        "status": status, "created_at": utc_now(),
        "selected_workflow": "organ_group_100mm_fp32_global_nn" if status == "PASS" else None,
        "spatial_configuration": "organ_group_100mm" if status == "PASS" else None,
        "precision": "fp32" if status == "PASS" else None,
        "matching_mode": "global_nn" if status == "PASS" else None,
        "fixed_point_status": "PROVISIONAL_NO_CONCERN" if not fixed_concerns else "PROVISIONAL_CONCERN",
        "fixed_point_concerns": fixed_concerns,
        "limitations": [
            "Stage 4 descriptor-level crop-context invariance remains unresolved.",
            "Fixed-point evidence is limited to eight deterministic sentinels.",
            "Validation covers only the largest organ-group Test/Retest pair.",
            "Organ-group crops impose an anatomical search prior.",
            "No cohort analysis is authorized.",
        ],
        "technical_failures": sorted(set(technical_failures)),
    }
    atomic_json(selected_path, selection, refuse=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "stage": 5, "status": status,
        "created_at": utc_now(), "validation_id": VALIDATION_ID,
        "selected_workflow": file_identity(selected_path),
        "stage4c_checkpoint": manifest["stage4c_checkpoint"],
        "gates": {
            "stage4a_and_stage4b_blocked_evidence_preserved": True,
            "stage4c_provisional_acceptance_validated": True,
            "bounded_dense_streamed_equivalence_passed": equivalence.get("status") == "success",
            "all_eight_group_margin_workers_completed": len(results) == 8 and all(item.get("status") == "success" for item in results),
            "global_nn_all_queries_completed": bool(global_summary) and all(row["complete"] for row in global_summary),
            "global_nn_crop_sensitivity_passed": global_passed,
            "no_120mm_match_outside_100mm_crop": outside_count == 0,
            "memory_headroom_passed": "memory_ceiling" not in technical_failures,
            "fixed_point_bounded_diagnostic_only": True,
            "cohort_authorized": False,
            "embeddings_or_prepared_volumes_retained": bool(retained),
        },
        "fixed_point_status": selection["fixed_point_status"],
        "next_stage": "largest_pair_global_nn_cycle_error_pilot" if status == "PASS" else ("resolve_match_sensitivity" if status == "BLOCKED" else "resolve_technical_failure"),
        "outputs": {
            "global_sensitivity": file_identity(run_dir / "global_nn_sensitivity.csv"),
            "fixed_point_sensitivity": file_identity(run_dir / "fixed_point_sentinel_sensitivity.csv"),
            "global_summary": file_identity(run_dir / "global_nn_summary.csv"),
            "memory_profile": manifest["outputs"]["memory_profile"],
            "report": file_identity(report_path),
            "figures": figures,
        },
    }
    atomic_json(checkpoint_path, checkpoint, refuse=True)
    manifest.update({"status": status.lower(), "completed_at": utc_now(), "preprocess_profile_at_selection": profile, "selection": file_identity(selected_path), "checkpoint": file_identity(checkpoint_path)})
    atomic_json(manifest_path, manifest)
    print("Stage 5 {}".format(status), flush=True)
    print("Selected workflow: {}".format(selection["selected_workflow"]), flush=True)
    print("Fixed-point status: {}".format(selection["fixed_point_status"]), flush=True)
    print("Checkpoint: {}".format(checkpoint_path), flush=True)
    return status == "PASS"


def forbidden_outputs(run_dir):
    forbidden = (".nii", ".nii.gz", ".npy", ".npz", ".mmap", ".pt", ".pth")
    return [str(path) for path in Path(run_dir).rglob("*") if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in forbidden)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    prepare = sub.add_parser("prepare", help="Freeze Stage 4C evidence, queries, sentinels and 100/120 mm plans.")
    prepare.add_argument("--stage4c-checkpoint", required=True)
    prepare.add_argument("--storage-root", default="/workspace/quadra")
    prepare.add_argument("--repository-root", default=str(PROJECT_ROOT))
    prepare.add_argument("--output-root", default=None); prepare.add_argument("--run-id", default=None)
    prepare.add_argument("--resume-run-directory", default=None)
    benchmark = sub.add_parser("benchmark", help="Run bounded equivalence and eight fresh group-margin GPU workers.")
    benchmark.add_argument("--run-directory", required=True); benchmark.add_argument("--storage-root", default="/workspace/quadra")
    benchmark.add_argument("--config", default="configs/samv2/samv2_NIHLN.py"); benchmark.add_argument("--checkpoint", default="checkpoints/SAMv2_iter_20000.pth")
    benchmark.add_argument("--timeout-seconds", type=int, default=WORKER_TIMEOUT_SECONDS)
    select = sub.add_parser("select", help="Apply global-NN sensitivity gates and freeze only a passing workflow.")
    select.add_argument("--run-directory", required=True); select.add_argument("--storage-root", default="/workspace/quadra")
    worker = sub.add_parser("_worker")
    worker.add_argument("--test-plan", required=True); worker.add_argument("--retest-plan", required=True)
    worker.add_argument("--queries", required=True); worker.add_argument("--sentinels", required=True)
    worker.add_argument("--config", required=True); worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--result-path", required=True); worker.add_argument("--worker-signature", required=True)
    eq = sub.add_parser("_equivalence_worker")
    eq.add_argument("--plan", required=True); eq.add_argument("--config", required=True); eq.add_argument("--checkpoint", required=True)
    eq.add_argument("--result-path", required=True); eq.add_argument("--worker-signature", required=True)
    return parser


def main(argv=None):
    parser = build_parser(); args = parser.parse_args(argv)
    if not args.command:
        parser.print_help(); return 0
    try:
        if args.command == "prepare": run_prepare(args)
        elif args.command == "benchmark": run_benchmark(args)
        elif args.command == "select": return 0 if run_select(args) else 3
        elif args.command == "_worker": return run_worker(args)
        elif args.command == "_equivalence_worker": return run_equivalence_worker(args)
    except Stage5Error as exc:
        parser.error("Stage 5 failed: {}".format(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
