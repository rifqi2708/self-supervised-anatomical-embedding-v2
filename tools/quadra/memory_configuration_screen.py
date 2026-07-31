#!/usr/bin/env python
"""Stage 3 dense UAE-S memory screen for the largest spatial candidates.

The file intentionally remains syntactically compatible with Python 3.7: the
``prepare`` command runs in the preprocessing image, while ``benchmark`` and
its private workers run in the pinned legacy UAE image.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import resource
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
SCHEMA_VERSION = 1
SCREEN_ID = "quadra-uaes-memory-screen-v1"
EXPECTED_BRANCH = "codex/quadra-memory-optimization"
EXPECTED_STAGE2_COMMIT = "7051e0b"
EXPECTED_BASELINE = Path(
    "/workspace/quadra/runs/memory_optimization/stage0-20260731T085944Z/baseline_manifest.json"
)
EXPECTED_BASELINE_SHA256 = "c331396b41a5cd03c039700b37c79e54fc920944c3e8ff304c2b0175c94d4a47"
EXPECTED_STAGE1 = Path(
    "/workspace/quadra/runs/memory_optimization/stage1-audit-20260731T110726Z/checkpoint_summary.json"
)
EXPECTED_SELECTED_SHA256 = "ec3df0f1c9ed3148058a850ac4591a2dd5a84e029786a5c6c52db140783b5b6c"
EXPECTED_STAGE2 = Path(
    "/workspace/quadra/runs/memory_optimization/stage2-crop-20260731T144720Z/checkpoint_summary.json"
)
EXPECTED_STAGE2_SHA256 = "593116d3288c964ebe0ec6a539feb9caeb644564ee4a5eadda8b1ec20aa84aba"
EXPECTED_CONFIG_SHA256 = "cb45a8790c9524fb93cb1725b9604741cfc01de7a352bf9b2773718101126ba2"
EXPECTED_CHECKPOINT_SHA256 = "a094d5eef867504defdc4c8e1d950835c4eb8aaa19de2027bb1a194781e423e3"
EXPECTED_UAE_IMAGE_DIGEST = "sha256:2c0edd4a205c3c5d9d027b6c9f96f83626eb2cc3810da7876e32d4bf36653d61"
EXPECTED_PREPROCESS_IMAGE_DIGEST = "sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5"
EXPECTED_GPU = "NVIDIA RTX A6000"
EXPECTED_VRAM_MIB = 49140
VRAM_CEILING_MIB = 39312
TARGET_SPACING = (2.0, 2.0, 2.0)
MODEL_STRIDE = (16, 16, 4)
ORGAN_MARGIN_MM = 100.0
WORKER_TIMEOUT_SECONDS = 30 * 60
SEED = 20260721
RUN_PREFIX = "stage3-screen-"

GROUPS = {
    "head_neck": (
        "brain", "eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right",
        "parotid_gland_left", "parotid_gland_right", "skull", "vertebrae_C1",
        "vertebrae_C4", "vertebrae_C7", "spinal_cord_cervical",
    ),
    "thorax": (
        "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
        "lung_middle_lobe_right", "lung_lower_lobe_right", "heart", "esophagus",
        "vertebrae_T4", "vertebrae_T8", "vertebrae_T12", "rib_left_6",
        "rib_right_6", "spinal_cord_thoracic",
    ),
    "abdomen": (
        "liver", "kidney_left", "kidney_right", "pancreas", "duodenum",
        "small_bowel", "colon", "vertebrae_L1", "vertebrae_L3", "vertebrae_L5",
    ),
    "pelvis": ("urinary_bladder", "sacrum", "hip_left", "hip_right", "prostate"),
}
PRECISIONS = ("fp32", "amp", "full_fp16")
SPATIAL_ORDER = ("whole_body", "body_envelope", "organ_group")
PRECISION_ORDER = ("fp32", "amp", "full_fp16")


class Stage3Error(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise Stage3Error("Required file is missing: {}".format(path))
    stat = path.stat()
    return {"path": str(path), "bytes": int(stat.st_size), "sha256": sha256_file(path)}


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise Stage3Error("Cannot read JSON {}: {}".format(path, exc))
    if not isinstance(value, dict):
        raise Stage3Error("Expected a JSON object: {}".format(path))
    return value


def atomic_json(path, value, refuse=False):
    path = Path(path)
    if refuse and path.exists():
        raise Stage3Error("Refusing to overwrite existing file: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_csv(path, rows):
    if not rows:
        raise Stage3Error("Refusing to write an empty CSV: {}".format(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def emit(message, log_path=None):
    print(message, flush=True)
    if log_path is not None:
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write("{} {}\n".format(utc_now(), message))


def git_output(args, repository=PROJECT_ROOT):
    return subprocess.check_output(
        ["git", "-C", str(repository)] + list(args), stderr=subprocess.STDOUT
    ).decode("utf-8").strip()


def validate_repository(repository=PROJECT_ROOT):
    # ``git branch --show-current`` was introduced after the Git 2.17 client
    # shipped in the pinned UAE container.  symbolic-ref provides the same
    # fail-closed branch identity check on both the preprocessing and UAE
    # profiles.
    branch = git_output(["symbolic-ref", "--short", "HEAD"], repository)
    commit = git_output(["rev-parse", "HEAD"], repository)
    dirty = git_output(["status", "--porcelain"], repository)
    ancestor = subprocess.call(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", EXPECTED_STAGE2_COMMIT, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0
    if branch != EXPECTED_BRANCH or dirty or not ancestor:
        raise Stage3Error(
            "Repository contract failed: branch={!r}, clean={}, Stage2 ancestor={}".format(
                branch, not bool(dirty), ancestor
            )
        )
    return {"path": str(Path(repository).resolve()), "branch": branch, "execution_commit": commit, "clean": True}


def validate_identity(path, expected_path, expected_hash=None):
    path = Path(path).resolve()
    if path != Path(expected_path).resolve():
        raise Stage3Error("Unexpected accepted artifact path: {}".format(path))
    identity = file_identity(path)
    if expected_hash and identity["sha256"] != expected_hash:
        raise Stage3Error("Accepted artifact hash mismatch: {}".format(path))
    return identity


def read_profile_fingerprint(storage_root, required_profile):
    path = Path(storage_root) / "runtime/profiles/{}-fingerprint.json".format(required_profile)
    fingerprint = load_json(path)
    if fingerprint.get("profile") != required_profile:
        raise Stage3Error("Wrong runtime profile: {}".format(fingerprint.get("profile")))
    expected = EXPECTED_PREPROCESS_IMAGE_DIGEST if required_profile == "preprocess" else EXPECTED_UAE_IMAGE_DIGEST
    environment_manifest_path = Path(storage_root) / "metadata/manifests/environment.json"
    environment_manifest = load_json(environment_manifest_path)
    profile_record = environment_manifest.get("profiles", {}).get(required_profile, {})
    digest = profile_record.get("image_digest")
    if digest != expected:
        raise Stage3Error("{} image digest mismatch: {}".format(required_profile, digest))
    if fingerprint.get("image_ref") != profile_record.get("image_ref"):
        raise Stage3Error("{} image reference differs from its persistent manifest".format(required_profile))
    gpu = current_gpu_record()
    if gpu["name"] != EXPECTED_GPU or gpu["memory_total_mib"] != EXPECTED_VRAM_MIB:
        raise Stage3Error("Stage 3 requires one 49140 MiB NVIDIA RTX A6000")
    return {"identity": file_identity(path), "environment_manifest": file_identity(environment_manifest_path),
            "image_digest": digest, "fingerprint": fingerprint, "current_gpu": gpu}


def current_gpu_record():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], stderr=subprocess.STDOUT).decode("utf-8").strip().splitlines()
    if len(output) != 1:
        raise Stage3Error("Stage 3 requires exactly one GPU")
    fields = [value.strip() for value in output[0].split(",")]
    if len(fields) != 4:
        raise Stage3Error("Cannot parse nvidia-smi GPU identity")
    return {"index": int(fields[0]), "name": fields[1],
            "memory_total_mib": int(float(fields[2])), "driver_version": fields[3]}


def validate_stage2_checkpoint(path):
    identity = validate_identity(path, EXPECTED_STAGE2, EXPECTED_STAGE2_SHA256)
    value = load_json(path)
    required = (
        "stage0_contract_validated", "stage1_selection_validated",
        "crop_resample_padding_geometry_passed", "coordinate_roundtrip_passed",
        "continuous_inverse_coordinates_preserved", "repository_remained_clean",
        "source_identity_and_geometry_passed", "stride_compatibility_passed",
    )
    gates = value.get("gates", {})
    if value.get("stage") != 2 or value.get("status") != "PASS" or any(gates.get(k) is not True for k in required):
        raise Stage3Error("Stage 2 checkpoint failed a required gate")
    if gates.get("model_or_cuda_computation_launched") is not False or gates.get("full_prepared_volumes_retained") is not False:
        raise Stage3Error("Stage 2 violated its computation or persistence boundary")
    return identity, value


def require_model_contract(baseline_manifest):
    manifest = load_json(baseline_manifest)
    model = manifest.get("model", {})
    config = model.get("config", {})
    checkpoint = model.get("checkpoint", {})
    if config.get("sha256") != EXPECTED_CONFIG_SHA256 or checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise Stage3Error("Stage 0 model identity differs from the frozen UAE-S contract")
    scientific = manifest.get("scientific_contract", {})
    if (scientific.get("norm_spacing_xyz_mm") != [2.0, 2.0, 2.0]
            or scientific.get("coordinate_space") != "raw_itk_voxel"
            or scientific.get("seed") != SEED
            or scientific.get("matching_modes") != ["global_nn", "fixed_point"]):
        raise Stage3Error("Stage 0 scientific contract changed")
    for record in (config, checkpoint):
        observed = file_identity(Path(record["path"]))
        if observed["bytes"] != record["bytes"] or observed["sha256"] != record["sha256"]:
            raise Stage3Error("Frozen model asset changed: {}".format(record["path"]))
    return manifest


def require_gpu_matches_baseline(manifest, profile_record):
    accepted = manifest.get("environment", {}).get("gpu", {})
    current = profile_record["current_gpu"]
    if (accepted.get("name") != current["name"]
            or accepted.get("memory_total_mib") != current["memory_total_mib"]
            or accepted.get("driver_version") != current["driver_version"]):
        raise Stage3Error("Current GPU/driver differs from the accepted Stage 0 environment")


def _plan_from_bounds(base, start, end, strategy, group_name=None, margin_mm=0.0):
    import numpy as np
    from tools.quadra import body_envelope_audit as stage1

    source = base["source_ct"]
    native_shape = np.asarray(source["native_shape_xyz"], dtype=np.int64)
    spacing = np.asarray(source["spacing_xyz_mm"], dtype=float)
    raw_affine = np.asarray(source["affine"], dtype=float)
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    if np.any(start < 0) or np.any(end > native_shape) or np.any(end <= start):
        raise Stage3Error("Invalid {} bounds for {}".format(strategy, base["scan_key"]))
    crop_shape = end - start
    target = stage1.torchio_target_shape(crop_shape, spacing, TARGET_SPACING)
    lower, upper, padded = stage1.symmetric_stride_padding(target, MODEL_STRIDE)
    crop_affine = stage1.crop_affine(raw_affine, start)
    resampled_affine, padded_affine = stage1.resampled_and_padded_affines(
        crop_affine, crop_shape, target, lower, TARGET_SPACING
    )
    raw_to_model = np.linalg.inv(padded_affine) @ raw_affine
    return {
        "subject_id": base["subject_id"], "session": base["session"],
        "scan_key": base["scan_key"], "sex": base.get("sex"),
        "strategy": strategy, "group_name": group_name,
        "axis_policy": "xyz" if strategy == "organ_group" else ("none" if strategy == "whole_body" else "xy"),
        "margin_mm": float(margin_mm), "source_ct": source,
        "crop_start_xyz": start.tolist(), "crop_end_xyz": end.tolist(),
        "crop_shape_xyz": crop_shape.tolist(), "target_shape_xyz": target.tolist(),
        "padding_lower_xyz": lower.tolist(), "padding_upper_xyz": upper.tolist(),
        "padded_shape_xyz": padded.tolist(), "model_tensor_shape_zyx": padded[::-1].tolist(),
        "padded_2mm_voxels": int(np.prod(padded, dtype=np.int64)),
        "minimum_artificial_mask_clearance_mm": None,
        "native_crop_affine": crop_affine.tolist(),
        "resampled_2mm_affine": resampled_affine.tolist(),
        "padded_2mm_affine": padded_affine.tolist(),
        "raw_to_model_continuous_affine": raw_to_model.tolist(),
        "model_to_raw_continuous_affine": np.linalg.inv(raw_to_model).tolist(),
    }


def required_group_masks(group_name, sex):
    if group_name not in GROUPS:
        raise Stage3Error("Unknown organ group: {}".format(group_name))
    return [name for name in GROUPS[group_name] if not (name == "prostate" and sex != "M")]


def derive_spatial_plans(selected, mask_csv_path, registry_path):
    import numpy as np
    import yaml
    from tools.quadra import body_envelope_audit as stage1

    registry = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    available = {item["filename"] for item in registry.get("organs", [])}
    available.update(item["filename"] for item in registry.get("derived_organs", []))
    configured = set(name for names in GROUPS.values() for name in names)
    if configured - available:
        raise Stage3Error("Organ groups reference unknown registry masks: {}".format(sorted(configured - available)))
    plans = selected.get("scan_plans", [])
    if len(plans) != 56 or len({p["scan_key"] for p in plans}) != 56:
        raise Stage3Error("Stage 1 must provide 56 unique scan plans")

    bbox = {}
    with Path(mask_csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["candidate_id"] != "xy_m010":
                continue
            key = (row["scan_key"], row["mask_name"])
            if key in bbox:
                raise Stage3Error("Duplicate mask bounding box: {}".format(key))
            bbox[key] = (json.loads(row["mask_start_xyz"]), json.loads(row["mask_end_xyz"]))

    whole, body, organs = [], [], []
    for frozen in plans:
        source = frozen["source_ct"]
        shape = np.asarray(source["native_shape_xyz"], dtype=np.int64)
        whole.append(_plan_from_bounds(frozen, [0, 0, 0], shape, "whole_body"))
        body_plan = dict(frozen)
        body_plan["strategy"] = "body_envelope"
        body_plan["group_name"] = None
        body.append(body_plan)
        sex = frozen.get("sex")
        if not sex:
            # Sex is recorded in Stage 1 scan results but not in the selected plan.
            sex = "M" if (frozen["scan_key"], "prostate") in bbox else "F"
        for group, configured_names in GROUPS.items():
            names = required_group_masks(group, sex)
            missing = [name for name in names if (frozen["scan_key"], name) not in bbox]
            if missing:
                raise Stage3Error("Missing required masks for {} {}: {}".format(frozen["scan_key"], group, missing))
            starts = np.asarray([bbox[(frozen["scan_key"], name)][0] for name in names], dtype=np.int64)
            ends = np.asarray([bbox[(frozen["scan_key"], name)][1] for name in names], dtype=np.int64)
            union_start, union_end = starts.min(axis=0), ends.max(axis=0)
            expanded_start, expanded_end = stage1.expand_bounds(
                union_start, union_end, shape, source["spacing_xyz_mm"],
                axis_policy="xyz", margin_mm=ORGAN_MARGIN_MM,
            )
            organ = _plan_from_bounds(
                dict(frozen, sex=sex), expanded_start, expanded_end,
                "organ_group", group_name=group, margin_mm=ORGAN_MARGIN_MM,
            )
            organ["included_masks"] = names
            organ["mask_union_start_xyz"] = union_start.tolist()
            organ["mask_union_end_xyz"] = union_end.tolist()
            organs.append(organ)
    if len(bbox) != 2208 or len(organs) != 224:
        raise Stage3Error("Unexpected bbox/organ plan counts: {}/{}".format(len(bbox), len(organs)))
    return {"whole_body": whole, "body_envelope": body, "organ_group": organs}


def plan_row(plan, selected=False):
    return {
        "strategy": plan["strategy"], "group_name": plan.get("group_name") or "",
        "scan_key": plan["scan_key"], "subject_id": plan["subject_id"],
        "session": plan["session"], "sex": plan.get("sex") or "",
        "crop_shape_xyz": json.dumps(plan["crop_shape_xyz"]),
        "target_shape_xyz": json.dumps(plan["target_shape_xyz"]),
        "padded_shape_xyz": json.dumps(plan["padded_shape_xyz"]),
        "padded_2mm_voxels": plan["padded_2mm_voxels"],
        "largest_for_strategy": bool(selected),
    }


def make_qc(prepared, plan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    volume = prepared.data_zyx
    z, y, x = [size // 2 for size in volume.shape]
    views = ((volume[z], "axial"), (volume[:, y, :], "coronal"), (volume[:, :, x], "sagittal"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, pair in zip(axes, views):
        axis.imshow(pair[0], cmap="gray", vmin=-50, vmax=80, origin="lower")
        axis.set_title(pair[1]); axis.axis("off")
    title = "{} — {}".format(plan["scan_key"], plan["strategy"])
    if plan.get("group_name"):
        title += "/" + plan["group_name"]
    fig.suptitle(title); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=140); plt.close(fig)


def run_prepare(args):
    import gc
    from tools.quadra import body_envelope_audit as stage1
    from tools.quadra import coordinate_preserving_crop as stage2
    from tools.quadra import optimization_baseline as baseline

    storage_root = Path(args.storage_root).resolve()
    repository = validate_repository(Path(args.repository_root))
    baseline_identity = validate_identity(args.baseline_manifest, EXPECTED_BASELINE, EXPECTED_BASELINE_SHA256)
    baseline.validate_locked_contract(
        Path(args.baseline_manifest), repository_root=Path(args.repository_root),
        storage_root=storage_root, required_profile="preprocess",
    )
    stage1_contract = stage2.validate_stage1_contract(
        Path(args.stage1_checkpoint), repository_root=Path(args.repository_root), storage_root=storage_root
    )
    stage2_identity, stage2_checkpoint = validate_stage2_checkpoint(args.stage2_checkpoint)
    profile = read_profile_fingerprint(storage_root, "preprocess")
    model_contract = require_model_contract(args.baseline_manifest)
    require_gpu_matches_baseline(model_contract, profile)

    selected = stage1_contract["selected"]
    audit_path = Path(selected["audit_manifest"]["path"])
    audit = load_json(audit_path)
    stage1.verify_audit_outputs(audit_path.parent, audit)
    mask_csv = Path(audit["outputs"]["mask_clearance.csv"]["path"])
    registry = PROJECT_ROOT / "tools/quadra/totalsegmentator/organs.yaml"
    derived = derive_spatial_plans(selected, mask_csv, registry)
    largest = {key: max(values, key=lambda p: (p["padded_2mm_voxels"], p["scan_key"], p.get("group_name") or "")) for key, values in derived.items()}
    if largest["body_envelope"]["scan_key"] != "quadra_hc_044-test":
        raise Stage3Error("Frozen body-envelope largest scan changed")

    output_root = Path(args.output_root or storage_root / "runs/memory_optimization").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.resume_run_directory:
        run_dir = Path(args.resume_run_directory).resolve()
        if not str(run_dir).startswith(str(output_root) + os.sep):
            raise Stage3Error("Resume directory escapes the Stage 3 output root")
        manifest_path = run_dir / "stage3_plan.json"
        existing = load_json(manifest_path)
        if existing.get("status") == "prepared":
            raise Stage3Error("Completed Stage 3 preparation is immutable")
    else:
        run_id = args.run_id or (RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        if not run_id.startswith(RUN_PREFIX):
            raise Stage3Error("Invalid Stage 3 run id")
        run_dir = output_root / run_id
        if run_dir.exists():
            raise Stage3Error("Refusing to overwrite Stage 3 run: {}".format(run_dir))
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / "stage3_plan.json"

    log_path = run_dir / "stage3.log"
    plan = {
        "schema_version": SCHEMA_VERSION, "screen_id": SCREEN_ID, "status": "preparing",
        "created_at": utc_now(), "baseline_manifest": baseline_identity,
        "stage1_checkpoint": stage1_contract["checkpoint_identity"],
        "selected_body_envelope": stage1_contract["selected_identity"],
        "stage1_audit_manifest": stage1_contract["audit_identity"],
        "stage2_checkpoint": stage2_identity, "repository": repository,
        "preprocess_profile": profile, "config": model_contract["model"]["config"],
        "checkpoint": model_contract["model"]["checkpoint"],
        "settings": {
            "spacing_xyz_mm": list(TARGET_SPACING), "model_stride_xyz": list(MODEL_STRIDE),
            "coordinate_space": "raw_itk_voxel", "seed": SEED,
            "organ_margin_mm": ORGAN_MARGIN_MM, "organ_groups": {k: list(v) for k, v in GROUPS.items()},
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS, "vram_ceiling_mib": VRAM_CEILING_MIB,
            "ranking": "spatial_coverage_then_precision_then_memory_then_runtime",
        },
        "largest_spatial_plans": largest,
        "spatial_plans": derived,
        "spatial_plan_counts": {k: len(v) for k, v in derived.items()},
        "scientific_scope": {"embedding_extraction_only": True, "matching": False, "cycle_error": False, "segmentation": False},
    }
    if manifest_path.exists():
        comparable = ("schema_version", "screen_id", "baseline_manifest", "stage1_checkpoint",
                      "selected_body_envelope", "stage1_audit_manifest", "stage2_checkpoint",
                      "repository", "settings", "largest_spatial_plans", "spatial_plans",
                      "spatial_plan_counts", "scientific_scope")
        if any(existing.get(key) != plan.get(key) for key in comparable):
            raise Stage3Error("Stage 3 resume contract changed")
        plan = existing
    else:
        atomic_json(manifest_path, plan, refuse=True)
    emit("Stage 3 prepare: realizing three largest plans sequentially", log_path)
    qc_dir = run_dir / "qc"
    cpu_results = []
    for strategy in SPATIAL_ORDER:
        spatial = largest[strategy]
        emit("Preparing {}: {}".format(strategy, spatial["scan_key"]), log_path)
        prepared = stage2.prepare_scan_from_plan(Path(spatial["source_ct"]["path"]), spatial)
        checks = stage2.coordinate_check_rows(prepared, spatial)
        max_voxel = max(row["max_raw_voxel_error"] for row in checks)
        max_mm = max(row["physical_error_mm"] for row in checks)
        if max_voxel > stage2.ROUNDTRIP_VOXEL_ATOL or max_mm > stage2.ROUNDTRIP_PHYSICAL_ATOL_MM:
            raise Stage3Error("Coordinate validation failed for {}".format(strategy))
        qc_path = qc_dir / "{}.png".format(strategy)
        make_qc(prepared, spatial, qc_path)
        cpu_results.append({
            "strategy": strategy, "scan_key": spatial["scan_key"],
            "group_name": spatial.get("group_name"), "tensor_shape_ncdhw": list(prepared.tensor_shape_ncdhw),
            "dtype": str(prepared.data_zyx.dtype), "max_raw_roundtrip_error": max_voxel,
            "max_physical_roundtrip_error_mm": max_mm, "qc": file_identity(qc_path), "status": "PASS",
        })
        del prepared
        gc.collect()
    all_rows = []
    for strategy in SPATIAL_ORDER:
        for item in derived[strategy]:
            all_rows.append(plan_row(item, item is largest[strategy]))
    write_csv(run_dir / "spatial_candidates.csv", all_rows)
    plan.update({
        "status": "prepared", "prepared_at": utc_now(), "cpu_validation": cpu_results,
        "outputs": {"spatial_candidates": file_identity(run_dir / "spatial_candidates.csv"),
                    "qc": {item["strategy"]: item["qc"] for item in cpu_results}},
    })
    atomic_json(manifest_path, plan)
    emit("Stage 3 prepare PASS", log_path)
    emit("Run directory: {}".format(run_dir), log_path)
    return run_dir


def _legacy_prepare(plan):
    """Python-3.7-compatible realization of the Stage 2 frozen geometry."""
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk

    source = plan["source_ct"]
    path = Path(source["path"])
    if file_identity(path)["sha256"] != source["sha256"]:
        raise Stage3Error("Source CT hash changed: {}".format(path))
    image_nib = nib.load(str(path))
    raw_affine = np.asarray(source["affine"], dtype=float)
    if tuple(image_nib.shape[:3]) != tuple(source["native_shape_xyz"]) or not np.allclose(image_nib.affine, raw_affine, atol=1e-5, rtol=0):
        raise Stage3Error("Source CT geometry changed: {}".format(plan["scan_key"]))
    image = sitk.ReadImage(str(path))
    start = np.asarray(plan["crop_start_xyz"], dtype=np.int64)
    end = np.asarray(plan["crop_end_xyz"], dtype=np.int64)
    cropped = sitk.RegionOfInterest(image, [int(x) for x in end - start], [int(x) for x in start])
    affine = np.asarray(plan["resampled_2mm_affine"], dtype=float)
    ras_lps = np.diag([-1.0, -1.0, 1.0, 1.0]).dot(affine)
    spacing = np.linalg.norm(ras_lps[:3, :3], axis=0)
    direction = ras_lps[:3, :3] / spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(x) for x in plan["target_shape_xyz"]])
    resampler.SetOutputSpacing(tuple(float(x) for x in spacing))
    resampler.SetOutputOrigin(tuple(float(x) for x in ras_lps[:3, 3]))
    resampler.SetOutputDirection(tuple(float(x) for x in direction.reshape(-1)))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear); resampler.SetDefaultPixelValue(-1024.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampled = resampler.Execute(cropped)
    padded = sitk.ConstantPad(
        resampled, [int(x) for x in plan["padding_lower_xyz"]],
        [int(x) for x in plan["padding_upper_xyz"]], -1024.0,
    )
    actual_spacing = np.asarray(padded.GetSpacing(), dtype=float)
    actual_direction = np.asarray(padded.GetDirection(), dtype=float).reshape(3, 3)
    actual_origin = np.asarray(padded.GetOrigin(), dtype=float)
    actual_lps = np.eye(4, dtype=float)
    actual_lps[:3, :3] = actual_direction * actual_spacing
    actual_lps[:3, 3] = actual_origin
    actual_ras = np.diag([-1.0, -1.0, 1.0, 1.0]).dot(actual_lps)
    if not np.allclose(actual_ras, np.asarray(plan["padded_2mm_affine"], dtype=float), atol=1e-5, rtol=0):
        raise Stage3Error("Prepared affine mismatch for {}".format(plan["scan_key"]))
    data = sitk.GetArrayFromImage(padded).astype(np.float32, copy=False)
    if data.shape != tuple(plan["model_tensor_shape_zyx"]):
        raise Stage3Error("Prepared shape mismatch for {}".format(plan["scan_key"]))
    if not np.isfinite(data).all():
        raise Stage3Error("Prepared CT contains non-finite values")
    np.clip(data, -1024.0, 3071.0, out=data)
    data -= np.float32(-1024.0); data *= np.float32(255.0 / 4095.0); data -= np.float32(50.0)
    if not np.isfinite(data).all() or float(data.min()) < -50.0001 or float(data.max()) > 205.0001:
        raise Stage3Error("Prepared CT normalization failed")
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    return data


def _sample_descriptors(outputs):
    import torch
    result = []
    for name, value in zip(("fine", "coarse", "semantic"), outputs):
        if value.ndim != 5 or value.shape[0] != 1:
            raise Stage3Error("Invalid {} output shape: {}".format(name, tuple(value.shape)))
        z, y, x = [int(v) for v in value.shape[2:]]
        locations = [(0, 0, 0), (z // 2, y // 2, x // 2), (z - 1, y - 1, x - 1)]
        descriptors = torch.stack([value[0, :, a, b, c].float().cpu() for a, b, c in locations])
        finite = bool(torch.isfinite(descriptors).all().item())
        norms = torch.norm(descriptors, p=2, dim=1)
        result.append({
            "name": name, "shape": [int(v) for v in value.shape], "dtype": str(value.dtype),
            "finite": finite, "norms": [float(v) for v in norms.tolist()],
            "samples": [[float(v) for v in row] for row in descriptors.tolist()],
        })
    return result


def _tensor_dtypes(value):
    try:
        import torch
        if torch.is_tensor(value):
            return [str(value.dtype)]
    except Exception:
        return []
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_tensor_dtypes(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_tensor_dtypes(item))
        return result
    return []


def _expected_feature_shapes(input_shape):
    n, c, z, y, x = input_shape
    return {
        "fine": [n, 128, z // 2, y // 2, x // 2],
        "coarse": [n, 128, z // 4, y // 16, x // 16],
        "semantic": [n, 128, z // 2, y // 2, x // 2],
    }


class NvidiaProcessSampler(threading.Thread):
    def __init__(self, pid):
        threading.Thread.__init__(self)
        self.daemon = True; self.pid = int(pid); self.maximum = None
        self.pid_matched = False; self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
                ], stderr=subprocess.DEVNULL).decode("utf-8")
                rows = []
                for line in output.splitlines():
                    fields = [part.strip() for part in line.split(",")]
                    if len(fields) == 2:
                        rows.append((int(fields[0]), float(fields[1])))
                matches = [value for pid, value in rows if pid == self.pid]
                if matches:
                    self.pid_matched = True; value = max(matches)
                    self.maximum = value if self.maximum is None else max(self.maximum, value)
                elif len(rows) == 1:
                    # Container PID namespaces can differ from the host PID
                    # reported by nvidia-smi. The controller requires an idle
                    # GPU before launch, so the sole compute process is this worker.
                    value = rows[0][1]
                    self.maximum = value if self.maximum is None else max(self.maximum, value)
            except Exception:
                pass
            self.stop_event.wait(0.05)

    def stop(self):
        self.stop_event.set(); self.join(timeout=2.0)


def _load_model(config_path, checkpoint_path, precision):
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector
    import sam  # noqa: F401

    config = Config.fromfile(str(config_path))
    training_fp16_present = config.get("fp16", None) is not None
    # Deliberately do not call mmcv.runner.wrap_fp16_model. The training hook
    # remains visible for provenance but is not applied to this inference model.
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model.eval()
    if precision == "full_fp16":
        model.half()
    else:
        model.float()
    model.cuda()
    return model, training_fp16_present


def require_idle_gpu():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
        ], stderr=subprocess.STDOUT).decode("utf-8").strip()
    except subprocess.CalledProcessError as exc:
        raise Stage3Error("Cannot inspect active GPU processes: {}".format(exc.output.decode("utf-8", "replace")))
    if output:
        raise Stage3Error("A6000 is not idle; active compute process(es): {}".format(output.replace("\n", "; ")))


def run_worker(args):
    result_path = Path(args.result_path)
    started = time.time()
    sampler = None
    result = {
        "schema_version": SCHEMA_VERSION, "screen_id": SCREEN_ID,
        "kind": args.kind, "precision": args.precision, "status": "running",
        "started_at": utc_now(), "pid": os.getpid(), "worker_signature": args.worker_signature,
    }
    try:
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise Stage3Error("CUDA is unavailable")
        torch.manual_seed(SEED); np.random.seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        config_path = Path(args.config); checkpoint_path = Path(args.checkpoint)
        if sha256_file(config_path) != EXPECTED_CONFIG_SHA256 or sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
            raise Stage3Error("Config or checkpoint hash mismatch")
        prep_started = time.time()
        if args.kind == "smoke":
            data = np.zeros((32, 64, 64), dtype=np.float32)
            spatial = {"strategy": "bounded_smoke", "scan_key": None, "group_name": None}
            spatial_identity = None
        else:
            spatial = load_json(args.plan_path)
            spatial_identity = file_identity(args.plan_path)
            data = _legacy_prepare(spatial)
        prep_seconds = time.time() - prep_started
        model_started = time.time()
        model, hook_present = _load_model(config_path, checkpoint_path, args.precision)
        torch.cuda.synchronize()
        model_seconds = time.time() - model_started
        baseline_allocated = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        tensor = torch.from_numpy(data)[None, None]
        transfer_started = time.time()
        tensor = tensor.cuda(non_blocking=False)
        if args.precision == "full_fp16":
            tensor = tensor.half()
        torch.cuda.synchronize(); transfer_seconds = time.time() - transfer_started
        input_shape = [int(v) for v in tensor.shape]
        input_dtype = str(tensor.dtype)
        model_dtype = str(next(model.parameters()).dtype)
        torch.cuda.reset_peak_memory_stats()
        sampler = NvidiaProcessSampler(os.getpid()); sampler.start()
        activation_dtypes = []
        def capture_activation(module, inputs, output):
            del module, inputs
            activation_dtypes.extend(_tensor_dtypes(output))
        activation_hook = model.backbone.register_forward_hook(capture_activation)
        forward_started = time.time()
        with torch.no_grad():
            if args.precision == "amp":
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model.extract_feat(tensor)
            else:
                outputs = model.extract_feat(tensor)
        torch.cuda.synchronize(); forward_seconds = time.time() - forward_started
        activation_hook.remove()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        samples = _sample_descriptors(outputs)
        expected = _expected_feature_shapes(input_shape)
        geometry_passed = all(item["shape"] == expected[item["name"]] for item in samples)
        normalized = all(all(abs(v - 1.0) <= 0.03 for v in item["norms"]) for item in samples)
        finite = all(item["finite"] for item in samples)
        expected_compute_dtype = "torch.float16" if args.precision == "full_fp16" else "torch.float32"
        expected_activation_dtype = "torch.float32" if args.precision == "fp32" else "torch.float16"
        precision_passed = bool(
            hook_present and model_dtype == expected_compute_dtype and input_dtype == expected_compute_dtype
            and all(item["dtype"] == "torch.float16" for item in samples)
            and activation_dtypes and all(value == expected_activation_dtype for value in activation_dtypes)
        )
        sampler.stop(); process_peak = sampler.maximum; process_pid_matched = sampler.pid_matched; sampler = None
        result.update({
            "status": "success", "failure_classification": None,
            "source": {"strategy": spatial.get("strategy"), "scan_key": spatial.get("scan_key"),
                       "group_name": spatial.get("group_name"), "source_ct": spatial.get("source_ct"),
                       "spatial_plan": spatial_identity},
            "input_shape_ncdhw": input_shape, "input_dtype": input_dtype,
            "model_dtype": model_dtype, "activation_precision": args.precision,
            "sampled_activation_dtypes": sorted(set(activation_dtypes)),
            "output_contract": {"explicit_uae_fp16_outputs": True, "features": samples,
                                "expected_shapes": expected, "geometry_passed": geometry_passed,
                                "finite_passed": finite, "normalized_passed": normalized},
            "precision_contract": {"training_fp16_hook_present": hook_present,
                                   "training_fp16_hook_ignored": True,
                                   "autocast_enabled": args.precision == "amp",
                                   "full_model_half": args.precision == "full_fp16",
                                   "passed": precision_passed},
            "timing_seconds": {"preprocessing": prep_seconds, "model_load": model_seconds,
                               "transfer": transfer_seconds, "forward": forward_seconds,
                               "total": time.time() - started},
            "memory": {"model_resident_allocated_bytes": baseline_allocated,
                       "model_resident_reserved_bytes": baseline_reserved,
                       "torch_peak_allocated_bytes": peak_allocated,
                       "torch_peak_reserved_bytes": peak_reserved,
                       "process_gpu_peak_mib": process_peak,
                       "process_gpu_pid_matched": process_pid_matched,
                       "cpu_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
        })
    except RuntimeError as exc:
        message = str(exc)
        oom = "out of memory" in message.lower() and "cuda" in message.lower()
        result.update({"status": "failed", "failure_classification": "cuda_oom" if oom else "model_error", "error": message[-4000:]})
    except Exception as exc:
        result.update({"status": "failed", "failure_classification": "environment_error", "error": repr(exc)})
    finally:
        if sampler is not None:
            sampler.stop()
        result["completed_at"] = utc_now(); result["wall_time_seconds"] = time.time() - started
        atomic_json(result_path, result)
    return 0 if result["status"] == "success" else 3


def worker_signature(kind, precision, plan_identity=None):
    value = {"screen_id": SCREEN_ID, "kind": kind, "precision": precision,
             "config_sha256": EXPECTED_CONFIG_SHA256,
             "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
             "plan_identity": plan_identity}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _worker_command(args, kind, precision, result_path, plan_path=None):
    plan_identity = file_identity(plan_path) if plan_path is not None else None
    signature = worker_signature(kind, precision, plan_identity)
    command = [sys.executable, "-m", "tools.quadra.memory_configuration_screen", "_worker",
               "--kind", kind, "--precision", precision, "--config", args.config,
               "--checkpoint", args.checkpoint, "--result-path", str(result_path),
               "--worker-signature", signature]
    if plan_path is not None:
        command.extend(["--plan-path", str(plan_path)])
    return command


def validate_reusable_worker_result(result, expected_signature, path):
    if result.get("worker_signature") != expected_signature:
        raise Stage3Error("Incompatible resumable worker result: {}".format(path))
    return result


def launch_worker(command, result_path, log_path, timeout):
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
        classification = classify_missing_worker(returncode, returncode is None)
        signature = command[command.index("--worker-signature") + 1]
        atomic_json(result_path, {"schema_version": SCHEMA_VERSION, "screen_id": SCREEN_ID,
                                  "status": "failed", "failure_classification": classification,
                                  "worker_signature": signature, "returncode": returncode,
                                  "completed_at": utc_now()})
    return load_json(result_path)


def classify_missing_worker(returncode, timed_out=False):
    if timed_out:
        return "timeout"
    if returncode is not None and returncode < 0:
        return "process_kill"
    return "process_crash"


def should_run_full_fp16(amp_result):
    return amp_result.get("failure_classification") == "cuda_oom"


def _cosine_rows(reference, candidate):
    import numpy as np
    rows = []
    for ref, other in zip(reference["output_contract"]["features"], candidate["output_contract"]["features"]):
        a = np.asarray(ref["samples"], dtype=float); b = np.asarray(other["samples"], dtype=float)
        cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        rows.append({"feature": ref["name"], "minimum_cosine": float(cosine.min()), "mean_cosine": float(cosine.mean())})
    return rows


def validate_plan_for_benchmark(run_dir, storage_root):
    plan_path = Path(run_dir) / "stage3_plan.json"
    plan = load_json(plan_path)
    if plan.get("status") not in ("prepared", "benchmarked") or plan.get("screen_id") != SCREEN_ID:
        raise Stage3Error("Stage 3 plan is not prepared")
    repository = validate_repository(PROJECT_ROOT)
    if repository["execution_commit"] != plan.get("repository", {}).get("execution_commit"):
        raise Stage3Error("Repository commit differs from the Stage 3 prepare commit")
    validate_identity(plan["baseline_manifest"]["path"], EXPECTED_BASELINE, EXPECTED_BASELINE_SHA256)
    validate_identity(plan["stage1_checkpoint"]["path"], EXPECTED_STAGE1)
    validate_identity(plan["stage2_checkpoint"]["path"], EXPECTED_STAGE2, EXPECTED_STAGE2_SHA256)
    baseline_manifest = require_model_contract(plan["baseline_manifest"]["path"])
    profile = read_profile_fingerprint(storage_root, "uae")
    require_gpu_matches_baseline(baseline_manifest, profile)
    return plan_path, plan, profile


def run_benchmark(args):
    run_dir = Path(args.run_directory).resolve()
    plan_path, plan, profile = validate_plan_for_benchmark(run_dir, Path(args.storage_root))
    if (run_dir / "checkpoint_summary.json").exists():
        raise Stage3Error("Stage 3 already has a final checkpoint")
    log_path = run_dir / "stage3.log"
    require_idle_gpu()
    smoke_dir = run_dir / "loader_smoke"; smoke_dir.mkdir(exist_ok=True)
    logs_dir = run_dir / "candidate_logs"; logs_dir.mkdir(exist_ok=True)
    results_dir = run_dir / "candidate_results"; results_dir.mkdir(exist_ok=True)
    spatial_dir = run_dir / "benchmark_plans"; spatial_dir.mkdir(exist_ok=True)

    smoke = {}
    for precision in ("fp32", "amp"):
        result_path = smoke_dir / "{}.json".format(precision)
        expected_signature = worker_signature("smoke", precision)
        if result_path.exists():
            result = validate_reusable_worker_result(load_json(result_path), expected_signature, result_path)
        else:
            result = launch_worker(
                _worker_command(args, "smoke", precision, result_path), result_path,
                logs_dir / "smoke_{}.log".format(precision), 300,
            )
        if result.get("status") != "success":
            raise Stage3Error("Bounded {} loader smoke failed".format(precision))
        smoke[precision] = result
    cosine = _cosine_rows(smoke["fp32"], smoke["amp"])
    if any(row["minimum_cosine"] < 0.99 for row in cosine):
        raise Stage3Error("FP32/AMP bounded descriptor cosine fell below 0.99")
    smoke_summary = {"status": "PASS", "fp32": file_identity(smoke_dir / "fp32.json"),
                     "amp": file_identity(smoke_dir / "amp.json"), "cosine": cosine,
                     "full_fp16": None}
    atomic_json(smoke_dir / "summary.json", smoke_summary)

    results = []
    for strategy in SPATIAL_ORDER:
        spatial = plan["largest_spatial_plans"][strategy]
        spatial_path = spatial_dir / "{}.json".format(strategy)
        if not spatial_path.exists():
            atomic_json(spatial_path, spatial, refuse=True)
        amp_oom = False
        for precision in PRECISIONS:
            candidate_id = "{}_{}".format(strategy, precision)
            if precision == "full_fp16" and not amp_oom:
                plan_identity = file_identity(spatial_path)
                result = {"schema_version": SCHEMA_VERSION, "screen_id": SCREEN_ID,
                          "candidate_id": candidate_id, "strategy": strategy,
                          "precision": precision, "status": "not_run",
                          "worker_signature": worker_signature("candidate", precision, plan_identity),
                          "failure_classification": "not_triggered_amp_did_not_oom"}
                result_path = results_dir / "{}.json".format(candidate_id)
                if not result_path.exists(): atomic_json(result_path, result, refuse=True)
                else: result = load_json(result_path)
                results.append(result); continue
            if precision == "full_fp16" and smoke_summary["full_fp16"] is None:
                full_path = smoke_dir / "full_fp16.json"
                full_signature = worker_signature("smoke", precision)
                if full_path.exists():
                    full = validate_reusable_worker_result(load_json(full_path), full_signature, full_path)
                else:
                    full = launch_worker(_worker_command(args, "smoke", precision, full_path), full_path,
                                         logs_dir / "smoke_full_fp16.log", 300)
                if full.get("status") != "success":
                    raise Stage3Error("Conditional full-FP16 bounded smoke failed")
                full_cosine = _cosine_rows(smoke["fp32"], full)
                if any(row["minimum_cosine"] < 0.99 for row in full_cosine):
                    raise Stage3Error("FP32/full-FP16 bounded descriptor cosine fell below 0.99")
                smoke_summary["full_fp16"] = {"identity": file_identity(full_path), "cosine": full_cosine}
                atomic_json(smoke_dir / "summary.json", smoke_summary)
            result_path = results_dir / "{}.json".format(candidate_id)
            candidate_signature = worker_signature("candidate", precision, file_identity(spatial_path))
            if result_path.exists():
                result = validate_reusable_worker_result(load_json(result_path), candidate_signature, result_path)
                if precision == "full_fp16" and amp_oom and result.get("status") == "not_run":
                    result = launch_worker(
                        _worker_command(args, "candidate", precision, result_path, spatial_path),
                        result_path, logs_dir / "{}.log".format(candidate_id), args.timeout_seconds,
                    )
            else:
                emit("Benchmarking {}".format(candidate_id), log_path)
                result = launch_worker(
                    _worker_command(args, "candidate", precision, result_path, spatial_path),
                    result_path, logs_dir / "{}.log".format(candidate_id), args.timeout_seconds,
                )
            if result.get("candidate_id") != candidate_id or result.get("strategy") != strategy:
                result["candidate_id"] = candidate_id; result["strategy"] = strategy
                atomic_json(result_path, result)
            results.append(result)
            if precision == "amp":
                amp_oom = should_run_full_fp16(result)

    rows = [result_row(r) for r in results]
    write_csv(run_dir / "memory_screen.csv", rows)
    atomic_text(run_dir / "memory_screen_report.md", render_memory_report(rows, smoke_summary))
    plan.update({"status": "benchmarked", "benchmarked_at": utc_now(), "uae_profile": profile,
                 "loader_smoke": file_identity(smoke_dir / "summary.json"),
                 "outputs": dict(plan.get("outputs", {}), memory_screen=file_identity(run_dir / "memory_screen.csv"),
                                 report=file_identity(run_dir / "memory_screen_report.md"))})
    atomic_json(plan_path, plan)
    emit("Stage 3 benchmark complete; run select after result inspection", log_path)
    return run_dir


def result_row(result):
    memory = result.get("memory", {})
    timing = result.get("timing_seconds", {})
    contract = result.get("output_contract", {})
    precision_contract = result.get("precision_contract", {})
    peak_reserved = memory.get("torch_peak_reserved_bytes")
    peak_reserved_mib = None if peak_reserved is None else float(peak_reserved) / 1048576.0
    process_peak = memory.get("process_gpu_peak_mib")
    measured_peak = max([v for v in (peak_reserved_mib, process_peak) if v is not None] or [float("inf")])
    eligible = bool(
        result.get("status") == "success" and contract.get("geometry_passed") is True
        and contract.get("finite_passed") is True and contract.get("normalized_passed") is True
        and precision_contract.get("passed") is True
        and measured_peak <= VRAM_CEILING_MIB
    )
    if eligible:
        reason = "eligible"
    elif result.get("failure_classification"):
        reason = result["failure_classification"]
    elif process_peak is None:
        reason = "process_gpu_metric_missing"
    elif precision_contract.get("passed") is not True:
        reason = "precision_contract_failed"
    elif measured_peak > VRAM_CEILING_MIB:
        reason = "memory_ceiling_exceeded"
    else:
        reason = "output_contract_failed"
    return {
        "candidate_id": result.get("candidate_id"), "strategy": result.get("strategy"),
        "precision": result.get("precision"), "status": result.get("status"),
        "failure_classification": result.get("failure_classification") or "",
        "input_shape_ncdhw": json.dumps(result.get("input_shape_ncdhw")),
        "model_dtype": result.get("model_dtype") or "", "input_dtype": result.get("input_dtype") or "",
        "torch_peak_allocated_mib": "" if memory.get("torch_peak_allocated_bytes") is None else float(memory["torch_peak_allocated_bytes"]) / 1048576.0,
        "torch_peak_reserved_mib": "" if peak_reserved_mib is None else peak_reserved_mib,
        "process_gpu_peak_mib": "" if process_peak is None else process_peak,
        "measured_peak_mib": measured_peak, "forward_seconds": timing.get("forward", ""),
        "total_seconds": timing.get("total", result.get("wall_time_seconds", "")),
        "geometry_passed": contract.get("geometry_passed", False),
        "finite_passed": contract.get("finite_passed", False),
        "normalized_passed": contract.get("normalized_passed", False),
        "precision_contract_passed": precision_contract.get("passed", False),
        "eligible": eligible, "eligibility_reason": reason,
    }


def render_memory_report(rows, smoke):
    lines = ["# Stage 3 largest-case UAE-S memory screen", "", "## Scope", "",
             "Dense embedding extraction only; no matching, cycle error, segmentation, or saved embeddings.", "",
             "## Bounded loader smoke", "",
             "FP32 and AMP passed with minimum sampled cosine >= 0.99.", "",
             "## Candidates", "", "| Candidate | Status | Peak MiB | Forward s | Eligible | Reason |",
             "|---|---|---:|---:|---|---|"]
    for row in rows:
        lines.append("| {candidate_id} | {status} | {measured_peak_mib} | {forward_seconds} | {eligible} | {eligibility_reason} |".format(**row))
    lines.extend(["", "Whole-body coverage is ranked above body-envelope coverage, which is ranked above organ-group coverage. Precision equivalence is not established here.", ""])
    return "\n".join(lines)


def _read_screen_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["eligible"] = row["eligible"].lower() == "true"
        row["measured_peak_mib"] = float(row["measured_peak_mib"])
        try: row["total_seconds"] = float(row["total_seconds"])
        except (TypeError, ValueError): row["total_seconds"] = float("inf")
    return rows


def select_configurations(rows):
    rank = {(space, precision): (i, j) for i, space in enumerate(SPATIAL_ORDER) for j, precision in enumerate(PRECISION_ORDER)}
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: (rank[(row["strategy"], row["precision"])], row["measured_peak_mib"], row["total_seconds"]))
    if len(eligible) < 2:
        return {"status": "BLOCKED", "preferred": eligible[0] if eligible else None,
                "fallback": None, "rejections": {row["candidate_id"]: row["eligibility_reason"] for row in rows if not row["eligible"]}}
    chosen = {eligible[0]["candidate_id"], eligible[1]["candidate_id"]}
    rejections = {}
    for row in rows:
        if row["candidate_id"] not in chosen:
            rejections[row["candidate_id"]] = row["eligibility_reason"] if not row["eligible"] else "eligible_but_lower_ranked"
    return {"status": "PASS", "preferred": eligible[0], "fallback": eligible[1], "rejections": rejections}


def run_select(args):
    run_dir = Path(args.run_directory).resolve()
    plan_path = run_dir / "stage3_plan.json"
    plan = load_json(plan_path)
    if plan.get("status") != "benchmarked":
        raise Stage3Error("Benchmark must complete before selection")
    validate_repository(PROJECT_ROOT)
    selected_path = run_dir / "selected_configuration.json"
    checkpoint_path = run_dir / "checkpoint_summary.json"
    if selected_path.exists() or checkpoint_path.exists():
        raise Stage3Error("Stage 3 selection is immutable")
    rows = _read_screen_rows(run_dir / "memory_screen.csv")
    if len(rows) != 9 or len({row["candidate_id"] for row in rows}) != 9:
        raise Stage3Error("Stage 3 candidate table is incomplete")
    selection = select_configurations(rows)
    payload = {
        "schema_version": SCHEMA_VERSION, "screen_id": SCREEN_ID,
        "created_at": utc_now(), "status": selection["status"],
        "selection_policy": "spatial_coverage_then_precision_then_memory_then_runtime",
        "preferred": selection["preferred"], "fallback": selection["fallback"],
        "rejections": selection["rejections"],
        "limitations": [
            "Embedding extraction only; matching memory was not measured.",
            "AMP/full-FP16 numerical equivalence was not established.",
            "Organ-group crops impose an anatomical search prior.",
        ],
    }
    atomic_json(selected_path, payload, refuse=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "stage": 3, "status": selection["status"],
        "created_at": utc_now(), "screen_id": SCREEN_ID,
        "selected_configuration": file_identity(selected_path),
        "preferred_candidate": selection["preferred"]["candidate_id"] if selection["preferred"] else None,
        "fallback_candidate": selection["fallback"]["candidate_id"] if selection["fallback"] else None,
        "gates": {
            "stage0_contract_validated": True, "stage1_selection_validated": True,
            "stage2_geometry_validated": True, "three_largest_spatial_plans_realized": True,
            "bounded_fp32_amp_smoke_passed": True, "fresh_process_per_candidate": True,
            "two_eligible_candidates_exist": selection["fallback"] is not None,
            "matching_or_cycle_error_run": False, "embeddings_or_prepared_volumes_retained": False,
        },
        "next_stage": "numerical_validation" if selection["status"] == "PASS" else "resolve_stage3_blocker",
    }
    atomic_json(checkpoint_path, checkpoint, refuse=True)
    plan.update({"status": "selected" if selection["status"] == "PASS" else "blocked",
                 "completed_at": utc_now(), "selected_configuration": file_identity(selected_path),
                 "checkpoint_summary": file_identity(checkpoint_path)})
    atomic_json(plan_path, plan)
    print("Stage 3 {}".format(selection["status"]), flush=True)
    print("Preferred: {}".format(checkpoint["preferred_candidate"]), flush=True)
    print("Fallback: {}".format(checkpoint["fallback_candidate"]), flush=True)
    print("Checkpoint: {}".format(checkpoint_path), flush=True)
    return selection["status"] == "PASS"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    prepare = sub.add_parser("prepare", help="Build and CPU-validate the three largest spatial plans.")
    prepare.add_argument("--baseline-manifest", default=str(EXPECTED_BASELINE))
    prepare.add_argument("--stage1-checkpoint", default=str(EXPECTED_STAGE1))
    prepare.add_argument("--stage2-checkpoint", default=str(EXPECTED_STAGE2))
    prepare.add_argument("--storage-root", default="/workspace/quadra")
    prepare.add_argument("--repository-root", default=str(PROJECT_ROOT))
    prepare.add_argument("--output-root", default=None); prepare.add_argument("--run-id", default=None)
    prepare.add_argument("--resume-run-directory", default=None)
    benchmark = sub.add_parser("benchmark", help="Run bounded loader smoke and fresh-process GPU candidates.")
    benchmark.add_argument("--run-directory", required=True); benchmark.add_argument("--storage-root", default="/workspace/quadra")
    benchmark.add_argument("--config", default="configs/samv2/samv2_NIHLN.py")
    benchmark.add_argument("--checkpoint", default="checkpoints/SAMv2_iter_20000.pth")
    benchmark.add_argument("--timeout-seconds", type=int, default=WORKER_TIMEOUT_SECONDS)
    select = sub.add_parser("select", help="Apply the frozen feasibility ranking.")
    select.add_argument("--run-directory", required=True)
    worker = sub.add_parser("_worker")
    worker.add_argument("--kind", choices=("smoke", "candidate"), required=True)
    worker.add_argument("--precision", choices=PRECISIONS, required=True)
    worker.add_argument("--config", required=True); worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--result-path", required=True); worker.add_argument("--plan-path", default=None)
    worker.add_argument("--worker-signature", required=True)
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
    except Stage3Error as exc:
        parser.error("Stage 3 failed: {}".format(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
