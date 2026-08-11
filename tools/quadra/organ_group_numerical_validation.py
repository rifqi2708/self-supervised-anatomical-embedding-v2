#!/usr/bin/env python
"""Stage 4 numerical validation for the selected organ-group UAE-S workflow.

The preprocessing command freezes raw-ITK foreground samples and candidate/
reference organ-group plans for the largest Stage 3 Test/Retest pair. The
original Stage 4A compares 100/120 mm; the isolated Stage 4B resolution run
reuses the exact samples and compares 120/150 mm. The benchmark
command runs dense UAE-S extraction in fresh group/session subprocesses and
retains only sampled descriptor comparisons. The select command freezes the
candidate workflow only after every numerical, geometry, memory, and provenance
gate passes.  Matching and cycle-error analysis are deliberately out of scope.

This module remains syntactically compatible with Python 3.7 because its GPU
workers execute inside the pinned legacy UAE-S container.
"""

from __future__ import print_function

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import shutil
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


SCHEMA_VERSION = 1
VALIDATION_ID = "quadra-organ-group-numerical-validation-v1"
RESOLUTION_VALIDATION_ID = "quadra-organ-group-boundary-resolution-v1"
EXPECTED_BRANCH = "codex/quadra-memory-optimization"
EXPECTED_STAGE3_ANCESTOR = "eb607b2"
EXPECTED_STAGE4A_ANCESTOR = "817176f"
EXPECTED_STAGE3 = Path(
    "/workspace/quadra/runs/memory_optimization/"
    "stage3-screen-20260731T170831Z/checkpoint_summary.json"
)
EXPECTED_STAGE3_SELECTION_SHA256 = (
    "93eafbd9c92913eac22aeaad93c4c4fc71624777084cc9a7580c8b685986158d"
)
EXPECTED_SUBJECT = "quadra_hc_030"
SELECTED_MARGIN_MM = 100.0
REFERENCE_MARGIN_MM = 120.0
RESOLUTION_SELECTED_MARGIN_MM = 120.0
RESOLUTION_REFERENCE_MARGIN_MM = 150.0
EXPECTED_STAGE4A_CHECKPOINT = Path(
    "/workspace/quadra/runs/memory_optimization/"
    "stage4-validation-20260811T031529Z/checkpoint_summary.json"
)
EXPECTED_STAGE4A_SELECTION_SHA256 = (
    "d47a0008dfe5f51509c6490ad677c4bde54641d135c31cca6bb87aef88265411"
)
MAX_POINTS_PER_MASK = 32
RANDOM_POINTS_PER_MASK = 25
COSINE_MEDIAN_MIN = 0.99
COSINE_P01_MIN = 0.95
ROUNDTRIP_VOXEL_ATOL = 1e-6
ROUNDTRIP_PHYSICAL_ATOL_MM = 1e-5
RUN_PREFIX = "stage4-validation-"
RESOLUTION_RUN_PREFIX = "stage4b-resolution-"
PRECISION_ORDER = ("fp32", "amp")
WORKER_TIMEOUT_SECONDS = 30 * 60


class Stage4Error(RuntimeError):
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
        raise Stage4Error("Required file is missing: {}".format(path))
    stat = path.stat()
    return {"path": str(path), "bytes": int(stat.st_size), "sha256": sha256_file(path)}


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise Stage4Error("Cannot read JSON {}: {}".format(path, exc))
    if not isinstance(value, dict):
        raise Stage4Error("Expected a JSON object: {}".format(path))
    return value


def atomic_json(path, value, refuse=False):
    path = Path(path)
    if refuse and path.exists():
        raise Stage4Error("Refusing to overwrite existing file: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_csv(path, rows):
    if not rows:
        raise Stage4Error("Refusing to write an empty CSV: {}".format(path))
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


def validate_repository(repository=PROJECT_ROOT):
    branch = stage3.git_output(["symbolic-ref", "--short", "HEAD"], repository)
    commit = stage3.git_output(["rev-parse", "HEAD"], repository)
    dirty = stage3.git_output(["status", "--porcelain"], repository)
    ancestor = subprocess.call(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", EXPECTED_STAGE3_ANCESTOR, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0
    if branch != EXPECTED_BRANCH or dirty or not ancestor:
        raise Stage4Error(
            "Repository contract failed: branch={!r}, clean={}, Stage3 ancestor={}".format(
                branch, not bool(dirty), ancestor
            )
        )
    return {
        "path": str(Path(repository).resolve()),
        "branch": branch,
        "execution_commit": commit,
        "clean": True,
    }


def validate_stage3_checkpoint(path):
    identity = file_identity(path)
    if Path(path).resolve() != EXPECTED_STAGE3.resolve():
        raise Stage4Error("Unexpected Stage 3 checkpoint path: {}".format(path))
    checkpoint = load_json(path)
    gates = checkpoint.get("gates", {})
    required = (
        "stage0_contract_validated",
        "stage1_selection_validated",
        "stage2_geometry_validated",
        "three_largest_spatial_plans_realized",
        "bounded_all_precision_smoke_passed",
        "all_nine_candidates_attempted",
        "fresh_process_per_candidate",
        "two_eligible_candidates_exist",
    )
    if (
        checkpoint.get("stage") != 3
        or checkpoint.get("status") != "PASS"
        or checkpoint.get("preferred_candidate") != "organ_group_fp32"
        or checkpoint.get("fallback_candidate") != "organ_group_amp"
        or any(gates.get(key) is not True for key in required)
        or gates.get("matching_or_cycle_error_run") is not False
        or gates.get("embeddings_or_prepared_volumes_retained") is not False
    ):
        raise Stage4Error("Stage 3 checkpoint does not freeze the accepted organ-group workflow")
    selected = checkpoint.get("selected_configuration", {})
    if selected.get("sha256") != EXPECTED_STAGE3_SELECTION_SHA256:
        raise Stage4Error("Stage 3 selected-configuration identity changed")
    observed = file_identity(selected.get("path"))
    if observed != selected:
        raise Stage4Error("Stage 3 selected-configuration file changed")
    selected_payload = load_json(observed["path"])
    if (
        selected_payload.get("status") != "PASS"
        or selected_payload.get("preferred", {}).get("candidate_id") != "organ_group_fp32"
        or selected_payload.get("fallback", {}).get("candidate_id") != "organ_group_amp"
    ):
        raise Stage4Error("Stage 3 selection payload is incompatible")
    return identity, checkpoint, selected_payload


def stage3_plan_from_checkpoint(checkpoint):
    selected_path = Path(checkpoint["selected_configuration"]["path"])
    run_dir = selected_path.parent
    plan_path = run_dir / "stage3_plan.json"
    plan = load_json(plan_path)
    if plan.get("status") != "selected" or plan.get("screen_id") != stage3.SCREEN_ID:
        raise Stage4Error("Stage 3 plan is not selected")
    if plan.get("repository", {}).get("execution_commit") != EXPECTED_STAGE3_ANCESTOR + "":
        # The stored value is a full hash; a prefix is intentionally accepted.
        if not str(plan.get("repository", {}).get("execution_commit", "")).startswith(EXPECTED_STAGE3_ANCESTOR):
            raise Stage4Error("Stage 3 plan commit changed")
    return plan_path, plan


def validate_stage4a_checkpoint(path):
    """Validate the immutable BLOCKED Stage 4A evidence used by Stage 4B."""
    identity = file_identity(path)
    if Path(path).resolve() != EXPECTED_STAGE4A_CHECKPOINT.resolve():
        raise Stage4Error("Unexpected Stage 4A checkpoint path: {}".format(path))
    checkpoint = load_json(path)
    gates = checkpoint.get("gates", {})
    required_true = (
        "stage0_to_stage3_contracts_validated",
        "all_four_groups_both_sessions_completed",
        "foreground_samples_contained",
        "coordinate_roundtrip_passed",
        "output_geometry_precision_and_finiteness_passed",
        "same_crop_repeatability_passed",
        "memory_headroom_passed",
    )
    if (
        checkpoint.get("stage") != 4
        or checkpoint.get("status") != "BLOCKED"
        or checkpoint.get("next_stage") != "resolve_stage4_blocker"
        or any(gates.get(key) is not True for key in required_true)
        or gates.get("100mm_vs_120mm_boundary_gate_passed") is not False
        or gates.get("matching_or_cycle_error_run") is not False
        or gates.get("embeddings_or_prepared_volumes_retained") is not False
        or gates.get("full_fp16_used") is not False
    ):
        raise Stage4Error("Stage 4A checkpoint is not the accepted boundary-only blocker")
    selection_reference = checkpoint.get("selected_configuration", {})
    if selection_reference.get("sha256") != EXPECTED_STAGE4A_SELECTION_SHA256:
        raise Stage4Error("Stage 4A selected-configuration identity changed")
    if file_identity(selection_reference.get("path")) != selection_reference:
        raise Stage4Error("Stage 4A selected-configuration file changed")
    selection = load_json(selection_reference["path"])
    if (
        selection.get("status") != "BLOCKED"
        or selection.get("failures") != ["boundary_sensitivity"]
        or selection.get("selected_spatial_configuration") is not None
        or selection.get("selected_precision") is not None
    ):
        raise Stage4Error("Stage 4A failure is not isolated to boundary sensitivity")
    run_dir = Path(path).resolve().parent
    manifest_path = run_dir / "stage4_manifest.json"
    manifest = load_json(manifest_path)
    settings = manifest.get("settings", {})
    if (
        manifest.get("validation_id") != VALIDATION_ID
        or manifest.get("status") != "blocked"
        or not str(manifest.get("repository", {}).get("execution_commit", "")).startswith(
            EXPECTED_STAGE4A_ANCESTOR
        )
        or float(settings.get("selected_margin_mm", -1)) != SELECTED_MARGIN_MM
        or float(settings.get("reference_margin_mm", -1)) != REFERENCE_MARGIN_MM
    ):
        raise Stage4Error("Stage 4A manifest contract changed")
    pair_plan_path = run_dir / "pair_plan.json"
    pair_plan = load_json(pair_plan_path)
    samples = manifest.get("outputs", {}).get("samples")
    containment = manifest.get("outputs", {}).get("containment")
    if (
        not isinstance(samples, dict)
        or not isinstance(containment, dict)
        or pair_plan.get("validation_id") != VALIDATION_ID
        or file_identity(pair_plan_path) != manifest.get("pair_plan")
        or file_identity(samples.get("path")) != samples
        or file_identity(containment.get("path")) != containment
    ):
        raise Stage4Error("Stage 4A compact evidence changed")
    return {
        "checkpoint": identity,
        "selection": selection_reference,
        "manifest": file_identity(manifest_path),
        "pair_plan": file_identity(pair_plan_path),
        "samples": samples,
        "containment": containment,
        "payload": pair_plan,
    }


def validate_resolution_source(manifest):
    """Revalidate Stage 4A immediately before Stage 4B state transitions."""
    if manifest.get("validation_id") != RESOLUTION_VALIDATION_ID:
        return
    source = manifest.get("source_stage4a")
    if not isinstance(source, dict) or not source.get("path"):
        raise Stage4Error("Stage 4B source Stage 4A checkpoint is missing")
    observed = validate_stage4a_checkpoint(source["path"])
    if observed["checkpoint"] != source:
        raise Stage4Error("Stage 4B source Stage 4A checkpoint changed")


def derive_margin_plan(plan_100, margin_mm):
    import numpy as np
    from tools.quadra import body_envelope_audit as stage1

    source = plan_100["source_ct"]
    start, end = stage1.expand_bounds(
        plan_100["mask_union_start_xyz"],
        plan_100["mask_union_end_xyz"],
        source["native_shape_xyz"],
        source["spacing_xyz_mm"],
        axis_policy="xyz",
        margin_mm=float(margin_mm),
    )
    derived = stage3._plan_from_bounds(
        plan_100,
        start,
        end,
        "organ_group",
        group_name=plan_100["group_name"],
        margin_mm=float(margin_mm),
    )
    derived["included_masks"] = list(plan_100["included_masks"])
    derived["mask_union_start_xyz"] = list(plan_100["mask_union_start_xyz"])
    derived["mask_union_end_xyz"] = list(plan_100["mask_union_end_xyz"])
    derived["original_fov_clamped_lower_xyz"] = (np.asarray(start) == 0).tolist()
    derived["original_fov_clamped_upper_xyz"] = (
        np.asarray(end) == np.asarray(source["native_shape_xyz"])
    ).tolist()
    return derived


def select_pair_plans(stage3_plan):
    plans = stage3_plan.get("spatial_plans", {}).get("organ_group", [])
    largest = stage3_plan.get("largest_spatial_plans", {}).get("organ_group", {})
    subject = largest.get("subject_id")
    if subject != EXPECTED_SUBJECT:
        raise Stage4Error("Largest organ-group subject changed: {}".format(subject))
    selected = [plan for plan in plans if plan.get("subject_id") == subject]
    expected = {(session, group) for session in ("test", "retest") for group in stage3.GROUPS}
    observed = {(plan.get("session"), plan.get("group_name")) for plan in selected}
    if len(selected) != 8 or observed != expected:
        raise Stage4Error("Largest pair does not have four unique group plans per session")
    for plan in selected:
        if float(plan.get("margin_mm", -1)) != SELECTED_MARGIN_MM:
            raise Stage4Error("Stage 3 organ-group margin changed")
    selected.sort(key=lambda item: (item["session"], item["group_name"]))
    return selected


def _seed_for(*parts):
    payload = "|".join(str(value) for value in parts).encode("utf-8")
    return (stage3.SEED + int(hashlib.sha256(payload).hexdigest()[:8], 16)) % (2 ** 32 - 1)


def sample_foreground_points(mask_data, centroid_xyz, limit=MAX_POINTS_PER_MASK):
    """Return deterministic, unique foreground XYZ voxels without ``argwhere`` on 3D data."""
    import numpy as np

    mask = np.asarray(mask_data) != 0
    if mask.ndim != 3 or not np.any(mask):
        raise Stage4Error("Mask must be non-empty and three-dimensional")
    total = int(np.count_nonzero(mask))
    rng = np.random.RandomState(_seed_for(mask.shape, total, centroid_xyz))
    wanted = min(RANDOM_POINTS_PER_MASK, total)
    ranks = sorted(int(value) for value in rng.choice(total, size=wanted, replace=False))
    rank_cursor = 0
    cumulative = 0
    random_points = []
    nearest = None
    nearest_distance = None
    extrema = {}
    centroid = np.asarray(centroid_xyz, dtype=float)
    for z in range(mask.shape[2]):
        xy = np.argwhere(mask[:, :, z])
        if not len(xy):
            continue
        count = len(xy)
        while rank_cursor < len(ranks) and ranks[rank_cursor] < cumulative + count:
            x, y = xy[ranks[rank_cursor] - cumulative]
            random_points.append((int(x), int(y), int(z)))
            rank_cursor += 1
        xyz = np.column_stack((xy, np.full(count, z, dtype=np.int64)))
        distances = np.sum((xyz.astype(float) - centroid[None, :]) ** 2, axis=1)
        index = int(np.argmin(distances))
        if nearest_distance is None or float(distances[index]) < nearest_distance:
            nearest_distance = float(distances[index])
            nearest = tuple(int(value) for value in xyz[index])
        for axis, label in enumerate(("x", "y", "z")):
            low = xyz[int(np.argmin(xyz[:, axis]))]
            high = xyz[int(np.argmax(xyz[:, axis]))]
            if label + "_min" not in extrema or int(low[axis]) < extrema[label + "_min"][axis]:
                extrema[label + "_min"] = tuple(int(value) for value in low)
            if label + "_max" not in extrema or int(high[axis]) > extrema[label + "_max"][axis]:
                extrema[label + "_max"] = tuple(int(value) for value in high)
        cumulative += count
    ordered = [nearest] + [extrema[name] for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")] + random_points
    unique = []
    seen = set()
    for point in ordered:
        if point is not None and point not in seen:
            seen.add(point)
            unique.append(point)
        if len(unique) == min(int(limit), total):
            break
    if not unique or any(not mask[point] for point in unique):
        raise Stage4Error("Foreground sampling produced an invalid point")
    return unique


def _mask_records_for_pair(stage3_plan, pair_plans):
    from tools.quadra import body_envelope_audit as stage1

    audit_path = Path(stage3_plan["stage1_audit_manifest"]["path"])
    audit = load_json(audit_path)
    cohort_path = Path(audit["cohort_manifest"]["path"])
    layout = stage1.canonical_layout(Path("/workspace/quadra"))
    _, scans = stage1.load_cohort(
        cohort_path,
        Path(layout["whole_body_ct"]),
        Path(layout["totalsegmentator_outputs"]),
    )
    by_key = {scan["key"]: scan for scan in scans}
    needed = sorted({plan["scan_key"] for plan in pair_plans})
    if any(key not in by_key for key in needed):
        raise Stage4Error("Cannot resolve largest-pair mask directories")
    return {key: by_key[key] for key in needed}, audit_path


def prepare_samples(pair_plans, stage3_plan, run_dir):
    import nibabel as nib
    import numpy as np
    from tools.quadra import body_envelope_audit as stage1

    scans, audit_path = _mask_records_for_pair(stage3_plan, pair_plans)
    scan_results = audit_path.parent / "scan_results"
    group_by_mask = {}
    for group, names in stage3.GROUPS.items():
        for name in names:
            if name in group_by_mask:
                raise Stage4Error("Mask is assigned to more than one organ group: {}".format(name))
            group_by_mask[name] = group

    samples = []
    containment = []
    mask_identities = []
    point_id = 0
    plan_by_scan_group = {(p["scan_key"], p["group_name"]): p for p in pair_plans}
    for scan_key in sorted(scans):
        scan = scans[scan_key]
        result = load_json(scan_results / "{}.json".format(scan_key))
        masks_from_audit = {
            item["mask_name"]: item
            for item in result["candidates"][0]["masks"]
        }
        for mask_name in scan["expected_masks"]:
            if mask_name not in group_by_mask:
                continue
            group = group_by_mask[mask_name]
            plan = plan_by_scan_group[(scan_key, group)]
            if mask_name not in plan["included_masks"]:
                continue
            mask_path = Path(scan["mask_directory"]) / "{}.nii.gz".format(mask_name)
            image = nib.load(str(mask_path))
            if tuple(image.shape[:3]) != tuple(plan["source_ct"]["native_shape_xyz"]):
                raise Stage4Error("Mask shape changed: {}".format(mask_path))
            if not np.allclose(image.affine, np.asarray(plan["source_ct"]["affine"]), atol=1e-5, rtol=0):
                raise Stage4Error("Mask affine changed: {}".format(mask_path))
            data = np.asanyarray(image.dataobj)
            audit_mask = masks_from_audit.get(mask_name)
            if audit_mask is None:
                raise Stage4Error("Stage 1 mask evidence is missing: {}".format(mask_name))
            points = sample_foreground_points(data, audit_mask["centroid_raw_xyz"])
            mask_identities.append(dict(file_identity(mask_path), scan_key=scan_key, mask_name=mask_name))
            start = np.asarray(plan["crop_start_xyz"], dtype=float)
            end = np.asarray(plan["crop_end_xyz"], dtype=float)
            outside = 0
            for local_index, point in enumerate(points):
                raw = np.asarray(point, dtype=float)
                contained = bool(np.all(raw >= start) and np.all(raw < end))
                outside += 0 if contained else 1
                samples.append({
                    "point_id": point_id,
                    "scan_key": scan_key,
                    "subject_id": scan["subject_id"],
                    "session": scan["session"],
                    "group_name": group,
                    "mask_name": mask_name,
                    "mask_point_index": local_index,
                    "raw_x": float(raw[0]),
                    "raw_y": float(raw[1]),
                    "raw_z": float(raw[2]),
                    "coord_space": "raw_itk_voxel",
                })
                point_id += 1
            containment.append({
                "scan_key": scan_key,
                "session": scan["session"],
                "group_name": group,
                "mask_name": mask_name,
                "sample_count": len(points),
                "outside_selected_crop": outside,
                "mask_voxels": int(np.count_nonzero(data)),
                "status": "PASS" if outside == 0 else "FAIL",
            })
            del data
    if not samples or any(row["outside_selected_crop"] for row in containment):
        raise Stage4Error("Frozen foreground samples are incomplete or outside the 100 mm crop")
    return samples, containment, mask_identities


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def frozen_stage4a_samples(stage4a, candidate_plans, run_dir):
    """Copy Stage 4A sample points and recheck them against the 120 mm crops."""
    samples_source = Path(stage4a["samples"]["path"])
    samples_path = Path(run_dir) / "sample_points_raw_itk.csv"
    shutil.copy2(str(samples_source), str(samples_path))
    if sha256_file(samples_path) != stage4a["samples"]["sha256"]:
        raise Stage4Error("Frozen Stage 4A sample-point copy changed")
    samples = read_csv_rows(samples_path)
    old_containment = {
        (row["scan_key"], row["group_name"], row["mask_name"]): row
        for row in read_csv_rows(stage4a["containment"]["path"])
    }
    plans = {(plan["scan_key"], plan["group_name"]): plan for plan in candidate_plans}
    grouped = {}
    for sample in samples:
        key = (sample["scan_key"], sample["group_name"], sample["mask_name"])
        grouped.setdefault(key, []).append(sample)
    containment = []
    for key, rows in sorted(grouped.items()):
        if (key[0], key[1]) not in plans or key not in old_containment:
            raise Stage4Error("Stage 4A sample group is incompatible: {}".format(key))
        plan = plans[(key[0], key[1])]
        start = [float(value) for value in plan["crop_start_xyz"]]
        end = [float(value) for value in plan["crop_end_xyz"]]
        outside = 0
        for row in rows:
            point = [float(row["raw_x"]), float(row["raw_y"]), float(row["raw_z"])]
            outside += 0 if all(start[i] <= point[i] < end[i] for i in range(3)) else 1
        previous = old_containment[key]
        containment.append({
            "scan_key": key[0],
            "session": previous["session"],
            "group_name": key[1],
            "mask_name": key[2],
            "sample_count": len(rows),
            "outside_selected_crop": outside,
            "mask_voxels": previous["mask_voxels"],
            "status": "PASS" if outside == 0 else "FAIL",
        })
    if not samples or any(row["outside_selected_crop"] for row in containment):
        raise Stage4Error("Frozen Stage 4A samples are incomplete or outside the 120 mm crop")
    return samples, containment, list(stage4a["payload"].get("mask_identities", []))


def homogeneous_transform(affine, point_xyz):
    import numpy as np
    vector = np.asarray(list(point_xyz) + [1.0], dtype=float)
    return np.asarray(affine, dtype=float).dot(vector)[:3]


def coordinate_rows(samples, plans):
    import numpy as np

    rows = []
    by_key = {(p["scan_key"], p["group_name"], int(round(p["margin_mm"]))): p for p in plans}
    margins = sorted({key[2] for key in by_key})
    if len(margins) != 2:
        raise Stage4Error("Coordinate validation requires exactly two margins")
    for sample in samples:
        # CSV-backed Stage 4B samples are strings; convert explicitly so the
        # continuous-transform path is identical to newly sampled Stage 4A data.
        raw = [float(sample["raw_x"]), float(sample["raw_y"]), float(sample["raw_z"])]
        for margin in margins:
            plan = by_key[(sample["scan_key"], sample["group_name"], margin)]
            model = homogeneous_transform(plan["raw_to_model_continuous_affine"], raw)
            back = homogeneous_transform(plan["model_to_raw_continuous_affine"], model.tolist())
            voxel_error = float(np.max(np.abs(back - np.asarray(raw))))
            physical_error = float(
                np.linalg.norm(
                    homogeneous_transform(plan["source_ct"]["affine"], back.tolist())
                    - homogeneous_transform(plan["source_ct"]["affine"], raw)
                )
            )
            padded = np.asarray(plan["padded_shape_xyz"], dtype=float)
            inside = bool(np.all(model >= -0.5) and np.all(model <= padded - 0.5))
            rows.append({
                "point_id": sample["point_id"],
                "scan_key": sample["scan_key"],
                "group_name": sample["group_name"],
                "margin_mm": margin,
                "model_x": float(model[0]),
                "model_y": float(model[1]),
                "model_z": float(model[2]),
                "max_raw_voxel_roundtrip_error": voxel_error,
                "physical_roundtrip_error_mm": physical_error,
                "inside_model_grid": inside,
            })
    return rows


def make_plan_qc(candidate_plan, reference_plan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    volume = np.asanyarray(nib.load(candidate_plan["source_ct"]["path"]).dataobj)
    union_start = np.asarray(candidate_plan["mask_union_start_xyz"], dtype=int)
    union_end = np.asarray(candidate_plan["mask_union_end_xyz"], dtype=int)
    center = ((union_start + union_end - 1) // 2).astype(int)
    views = (
        (volume[:, :, center[2]].T, (0, 1), "axial"),
        (volume[:, center[1], :].T, (0, 2), "coronal"),
        (volume[center[0], :, :].T, (1, 2), "sagittal"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (image, dims, title) in zip(axes, views):
        axis.imshow(image, cmap="gray", vmin=-1024, vmax=500, origin="lower")
        for plan, color in ((reference_plan, "yellow"), (candidate_plan, "red")):
            label = "{} mm".format(int(round(plan["margin_mm"])))
            start = plan["crop_start_xyz"]; end = plan["crop_end_xyz"]
            rectangle = plt.Rectangle(
                (start[dims[0]], start[dims[1]]),
                end[dims[0]] - start[dims[0]],
                end[dims[1]] - start[dims[1]],
                fill=False, edgecolor=color, linewidth=1.2, label=label,
            )
            axis.add_patch(rectangle)
        axis.set_title(title); axis.axis("off")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("{} / {}".format(candidate_plan["scan_key"], candidate_plan["group_name"]))
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=140); plt.close(fig)
    del volume


def _run_directory(output_root, run_id=None, resume=None, prefix=RUN_PREFIX):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if resume:
        run_dir = Path(resume).resolve()
        if not run_dir.is_dir() or run_dir.parent != output_root:
            raise Stage4Error("Invalid Stage 4 resume directory")
        return run_dir, True
    name = run_id or (prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if not name.startswith(prefix):
        raise Stage4Error("Invalid Stage 4 run id")
    run_dir = output_root / name
    if run_dir.exists():
        raise Stage4Error("Refusing to overwrite Stage 4 run: {}".format(run_dir))
    run_dir.mkdir(parents=True)
    return run_dir, False


def run_prepare(args):
    from tools.quadra import coordinate_preserving_crop as stage2
    from tools.quadra import optimization_baseline as baseline

    repository = validate_repository(Path(args.repository_root))
    storage_root = Path(args.storage_root).resolve()
    baseline.validate_locked_contract(
        Path(args.baseline_manifest),
        repository_root=Path(args.repository_root),
        storage_root=storage_root,
        required_profile="preprocess",
    )
    stage3_identity, stage3_checkpoint, selection = validate_stage3_checkpoint(args.stage3_checkpoint)
    stage3_plan_path, stage3_plan = stage3_plan_from_checkpoint(stage3_checkpoint)
    resolution = bool(args.stage4a_checkpoint)
    stage4a = validate_stage4a_checkpoint(args.stage4a_checkpoint) if resolution else None
    validation_id = RESOLUTION_VALIDATION_ID if resolution else VALIDATION_ID
    selected_margin = RESOLUTION_SELECTED_MARGIN_MM if resolution else SELECTED_MARGIN_MM
    reference_margin = RESOLUTION_REFERENCE_MARGIN_MM if resolution else REFERENCE_MARGIN_MM
    run_prefix = RESOLUTION_RUN_PREFIX if resolution else RUN_PREFIX
    accepted_inputs = {
        "baseline_manifest": file_identity(args.baseline_manifest),
        "stage1_checkpoint": file_identity(args.stage1_checkpoint),
        "stage2_checkpoint": file_identity(args.stage2_checkpoint),
    }
    for key, observed in accepted_inputs.items():
        if observed != stage3_plan.get(key):
            raise Stage4Error("{} differs from the artifact accepted by Stage 3".format(key))
    pair_100 = select_pair_plans(stage3_plan)
    candidate_plans = (
        [derive_margin_plan(plan, selected_margin) for plan in pair_100]
        if resolution else pair_100
    )
    reference_plans = [derive_margin_plan(plan, reference_margin) for plan in pair_100]
    all_plans = candidate_plans + reference_plans
    output_root = Path(args.output_root or storage_root / "runs/memory_optimization")
    run_dir, resuming = _run_directory(
        output_root, args.run_id, args.resume_run_directory, prefix=run_prefix
    )
    manifest_path = run_dir / "stage4_manifest.json"
    log_path = run_dir / "stage4.log"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "validation_id": validation_id,
        "status": "preparing",
        "created_at": utc_now(),
        "repository": repository,
        "baseline_manifest": accepted_inputs["baseline_manifest"],
        "stage1_checkpoint": accepted_inputs["stage1_checkpoint"],
        "stage2_checkpoint": accepted_inputs["stage2_checkpoint"],
        "stage3_checkpoint": stage3_identity,
        "stage3_plan": file_identity(stage3_plan_path),
        "stage3_selection": stage3_checkpoint["selected_configuration"],
        "source_stage4a": stage4a["checkpoint"] if resolution else None,
        "selected_candidate": selection["preferred"]["candidate_id"],
        "fallback_candidate": selection["fallback"]["candidate_id"],
        "subject_id": EXPECTED_SUBJECT,
        "settings": {
            "selected_margin_mm": selected_margin,
            "reference_margin_mm": reference_margin,
            "groups": {key: list(value) for key, value in stage3.GROUPS.items()},
            "max_points_per_mask": MAX_POINTS_PER_MASK,
            "seed": stage3.SEED,
            "cosine_median_min": COSINE_MEDIAN_MIN,
            "cosine_p01_min": COSINE_P01_MIN,
            "vram_ceiling_mib": stage3.VRAM_CEILING_MIB,
            "precision_policy": "fp32_then_complete_amp_only_for_candidate_cuda_oom",
            "frozen_sample_source": "stage4a" if resolution else "stage1_masks",
        },
        "scope": {
            "dense_embedding_extraction": True,
            "matching": False,
            "cycle_error": False,
            "full_fp16": False,
            "saved_embeddings": False,
        },
    }
    if resuming:
        existing = load_json(manifest_path)
        if existing.get("status") != "preparing":
            raise Stage4Error("Completed Stage 4 preparation is immutable")
        for key in ("validation_id", "repository", "baseline_manifest", "stage1_checkpoint", "stage2_checkpoint", "stage3_checkpoint", "stage3_plan", "source_stage4a", "settings", "scope"):
            if existing.get(key) != contract.get(key):
                raise Stage4Error("Stage 4 resume contract changed: {}".format(key))
        contract = existing
    else:
        atomic_json(manifest_path, contract, refuse=True)

    emit(
        "{} prepare: validating {}/{} mm plans with {} samples".format(
            "Stage 4B" if resolution else "Stage 4",
            int(selected_margin), int(reference_margin),
            "frozen Stage 4A" if resolution else "newly frozen foreground",
        ),
        log_path,
    )
    plan_dir = run_dir / "plans"; plan_dir.mkdir(exist_ok=True)
    qc_dir = run_dir / "qc"; qc_dir.mkdir(exist_ok=True)
    if resolution:
        samples, containment, mask_identities = frozen_stage4a_samples(
            stage4a, candidate_plans, run_dir
        )
    else:
        samples, containment, mask_identities = prepare_samples(
            candidate_plans, stage3_plan, run_dir
        )
        write_csv(run_dir / "sample_points_raw_itk.csv", samples)
    write_csv(run_dir / "containment_summary.csv", containment)
    coordinate = coordinate_rows(samples, all_plans)
    write_csv(run_dir / "coordinate_roundtrip.csv", coordinate)
    max_voxel = max(row["max_raw_voxel_roundtrip_error"] for row in coordinate)
    max_mm = max(row["physical_roundtrip_error_mm"] for row in coordinate)
    if max_voxel > ROUNDTRIP_VOXEL_ATOL or max_mm > ROUNDTRIP_PHYSICAL_ATOL_MM or any(not row["inside_model_grid"] for row in coordinate):
        raise Stage4Error("Stage 4 coordinate validation failed")

    cpu_rows = []
    plan_refs = []
    for plan in sorted(all_plans, key=lambda item: (item["session"], item["group_name"], item["margin_mm"])):
        plan_name = "{}-{}-m{:03d}.json".format(plan["scan_key"], plan["group_name"], int(plan["margin_mm"]))
        plan_path = plan_dir / plan_name
        if not plan_path.exists():
            atomic_json(plan_path, plan, refuse=True)
        plan_refs.append(dict(file_identity(plan_path), scan_key=plan["scan_key"], group_name=plan["group_name"], margin_mm=plan["margin_mm"]))
        emit("CPU validating {} / {} / {} mm".format(plan["scan_key"], plan["group_name"], int(plan["margin_mm"])), log_path)
        prepared = stage2.prepare_scan_from_plan(Path(plan["source_ct"]["path"]), plan)
        cpu_rows.append({
            "scan_key": plan["scan_key"], "session": plan["session"],
            "group_name": plan["group_name"], "margin_mm": plan["margin_mm"],
            "tensor_shape_ncdhw": json.dumps(list(prepared.tensor_shape_ncdhw)),
            "dtype": str(prepared.data_zyx.dtype), "finite": bool(__import__("numpy").isfinite(prepared.data_zyx).all()),
            "status": "PASS",
        })
        if int(round(plan["margin_mm"])) == int(round(selected_margin)):
            reference = next(
                p for p in reference_plans
                if p["scan_key"] == plan["scan_key"]
                and p["group_name"] == plan["group_name"]
            )
            make_plan_qc(plan, reference, qc_dir / "{}-{}.png".format(plan["scan_key"], plan["group_name"]))
        del prepared; gc.collect()
    write_csv(run_dir / "cpu_plan_validation.csv", cpu_rows)
    pair_plan = {
        "schema_version": SCHEMA_VERSION,
        "validation_id": validation_id,
        "subject_id": EXPECTED_SUBJECT,
        "plans": plan_refs,
        "samples": file_identity(run_dir / "sample_points_raw_itk.csv"),
        "mask_identities": mask_identities,
        "source_stage4a": stage4a["checkpoint"] if resolution else None,
        "smallest_selected_plan": min(candidate_plans, key=lambda item: (item["padded_2mm_voxels"], item["scan_key"], item["group_name"]))["scan_key"] + "/" + min(candidate_plans, key=lambda item: (item["padded_2mm_voxels"], item["scan_key"], item["group_name"]))["group_name"],
    }
    atomic_json(run_dir / "pair_plan.json", pair_plan, refuse=True)
    contract.update({
        "status": "prepared", "prepared_at": utc_now(),
        "pair_plan": file_identity(run_dir / "pair_plan.json"),
        "sample_count": len(samples), "mask_count": len(containment),
        "maximum_roundtrip_voxel_error": max_voxel,
        "maximum_roundtrip_physical_error_mm": max_mm,
        "outputs": {
            "samples": file_identity(run_dir / "sample_points_raw_itk.csv"),
            "containment": file_identity(run_dir / "containment_summary.csv"),
            "coordinates": file_identity(run_dir / "coordinate_roundtrip.csv"),
            "cpu_validation": file_identity(run_dir / "cpu_plan_validation.csv"),
        },
    })
    atomic_json(manifest_path, contract)
    emit("{} prepare PASS".format("Stage 4B" if resolution else "Stage 4"), log_path)
    emit("Run directory: {}".format(run_dir), log_path)
    return run_dir


def _read_samples(path, scan_key, group_name):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scan_key"] == scan_key and row["group_name"] == group_name:
                rows.append({
                    "point_id": int(row["point_id"]), "mask_name": row["mask_name"],
                    "raw_xyz": [float(row["raw_x"]), float(row["raw_y"]), float(row["raw_z"])],
                })
    if not rows:
        raise Stage4Error("No frozen samples for {}/{}".format(scan_key, group_name))
    rows.sort(key=lambda item: item["point_id"])
    return rows


def model_to_feature_xyz(model_xyz, input_shape_zyx, output_shape_zyx):
    import numpy as np
    model = np.asarray(model_xyz, dtype=float)
    input_xyz = np.asarray(input_shape_zyx[::-1], dtype=float)
    output_xyz = np.asarray(output_shape_zyx[::-1], dtype=float)
    return (model + 0.5) * output_xyz / input_xyz - 0.5


def sample_feature_descriptors(feature, plan, sample_rows):
    import numpy as np
    import torch

    raw_to_model = np.asarray(plan["raw_to_model_continuous_affine"], dtype=float)
    raw = np.asarray([row["raw_xyz"] for row in sample_rows], dtype=float)
    homogeneous = np.column_stack((raw, np.ones(len(raw), dtype=float)))
    model_xyz = homogeneous.dot(raw_to_model.T)[:, :3]
    feature_xyz = model_to_feature_xyz(model_xyz, plan["model_tensor_shape_zyx"], list(feature.shape[2:]))
    output_xyz = np.asarray(list(feature.shape[2:])[::-1], dtype=int)
    if np.any(feature_xyz < -0.5 - 1e-6) or np.any(feature_xyz > output_xyz - 0.5 + 1e-6):
        raise Stage4Error("Frozen raw point maps outside the feature grid")
    clipped = np.minimum(np.maximum(feature_xyz, 0.0), output_xyz.astype(float) - 1.0)
    lower = np.floor(clipped).astype(np.int64)
    upper = np.minimum(lower + 1, output_xyz - 1)
    weights = clipped - lower
    descriptors = torch.zeros((len(sample_rows), int(feature.shape[1])), dtype=torch.float32, device=feature.device)
    for x_side in (0, 1):
        for y_side in (0, 1):
            for z_side in (0, 1):
                index = np.column_stack((
                    upper[:, 0] if x_side else lower[:, 0],
                    upper[:, 1] if y_side else lower[:, 1],
                    upper[:, 2] if z_side else lower[:, 2],
                ))
                weight = (
                    (weights[:, 0] if x_side else 1.0 - weights[:, 0])
                    * (weights[:, 1] if y_side else 1.0 - weights[:, 1])
                    * (weights[:, 2] if z_side else 1.0 - weights[:, 2])
                )
                ix = torch.as_tensor(index[:, 0], dtype=torch.long, device=feature.device)
                iy = torch.as_tensor(index[:, 1], dtype=torch.long, device=feature.device)
                iz = torch.as_tensor(index[:, 2], dtype=torch.long, device=feature.device)
                w = torch.as_tensor(weight, dtype=torch.float32, device=feature.device)[:, None]
                descriptors += feature[0, :, iz, iy, ix].transpose(0, 1).float() * w
    norms_before = torch.norm(descriptors, p=2, dim=1)
    if not bool(torch.isfinite(descriptors).all().item()) or bool((norms_before <= 0).any().item()):
        raise Stage4Error("Sampled descriptors are non-finite or zero")
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    return descriptors.cpu(), [float(value) for value in norms_before.cpu().tolist()]


def _extract_once(model, data, plan, sample_rows, precision, capture_memory=True):
    import numpy as np
    import torch

    tensor = torch.from_numpy(data)[None, None].cuda(non_blocking=False)
    if precision == "amp":
        input_dtype = str(tensor.dtype)
    else:
        input_dtype = str(tensor.dtype)
    torch.cuda.synchronize()
    if capture_memory:
        torch.cuda.reset_peak_memory_stats()
    sampler = stage3.NvidiaProcessSampler(os.getpid()); sampler.start()
    started = time.time()
    with torch.no_grad():
        if precision == "amp":
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model.extract_feat(tensor)
        else:
            outputs = model.extract_feat(tensor)
    torch.cuda.synchronize(); forward_seconds = time.time() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    sampler.stop(); process_peak = sampler.maximum
    expected = stage3._expected_feature_shapes([int(value) for value in tensor.shape])
    descriptors = {}; norm_records = {}; feature_contract = []
    for name, feature in zip(("fine", "coarse", "semantic"), outputs):
        shape = [int(value) for value in feature.shape]
        grid_samples = stage3._sample_descriptors((feature,))[0]
        if shape != expected[name] or grid_samples["dtype"] != "torch.float16" or not grid_samples["finite"]:
            raise Stage4Error("{} output contract failed".format(name))
        if any(abs(value - 1.0) > 0.03 for value in grid_samples["norms"]):
            raise Stage4Error("{} grid descriptors are not normalized".format(name))
        sampled, norms = sample_feature_descriptors(feature, plan, sample_rows)
        descriptors[name] = sampled
        norm_records[name] = norms
        feature_contract.append({
            "name": name, "shape": shape, "dtype": str(feature.dtype),
            "grid_sample_norms": grid_samples["norms"], "finite": True,
        })
    del outputs, tensor
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    return {
        "descriptors": descriptors,
        "interpolated_norms_before_normalization": norm_records,
        "features": feature_contract,
        "input_dtype": input_dtype,
        "forward_seconds": forward_seconds,
        "memory": {
            "torch_peak_allocated_bytes": peak_allocated,
            "torch_peak_reserved_bytes": peak_reserved,
            "process_gpu_peak_mib": process_peak,
        },
    }


def cosine_rows(reference, candidate, sample_rows, comparison):
    import numpy as np

    rows = []
    summary = []
    for feature in ("fine", "coarse", "semantic"):
        first = reference[feature].numpy(); second = candidate[feature].numpy()
        cosine = np.sum(first * second, axis=1)
        for sample, value in zip(sample_rows, cosine):
            rows.append({
                "comparison": comparison, "feature": feature,
                "point_id": sample["point_id"], "mask_name": sample["mask_name"],
                "cosine": float(value),
            })
        summary.append({
            "comparison": comparison, "feature": feature,
            "count": len(cosine), "minimum_cosine": float(np.min(cosine)),
            "p01_cosine": float(np.percentile(cosine, 1)),
            "median_cosine": float(np.median(cosine)),
            "mean_cosine": float(np.mean(cosine)),
            "passed": bool(np.median(cosine) >= COSINE_MEDIAN_MIN and np.percentile(cosine, 1) >= COSINE_P01_MIN),
        })
    return rows, summary


def worker_signature(
    candidate_identity, reference_identity, sample_identity, precision,
    repeatability, validation_id=VALIDATION_ID,
):
    payload = {
        "validation_id": validation_id, "candidate_plan": candidate_identity,
        "reference_plan": reference_identity, "samples": sample_identity,
        "precision": precision, "repeatability": bool(repeatability),
        "checkpoint_sha256": stage3.EXPECTED_CHECKPOINT_SHA256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_worker(args):
    result_path = Path(args.result_path)
    started = time.time(); result = {
        "schema_version": SCHEMA_VERSION, "validation_id": args.validation_id,
        "status": "running", "started_at": utc_now(), "precision": args.precision,
        "worker_signature": args.worker_signature, "pid": os.getpid(),
    }
    try:
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise Stage4Error("CUDA is unavailable")
        torch.manual_seed(stage3.SEED); np.random.seed(stage3.SEED)
        torch.backends.cudnn.benchmark = stage3.CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = stage3.CUDNN_DETERMINISTIC
        candidate_plan = load_json(args.plan100); reference_plan = load_json(args.plan120)
        candidate_margin = int(round(candidate_plan["margin_mm"]))
        reference_margin = int(round(reference_plan["margin_mm"]))
        if candidate_margin >= reference_margin:
            raise Stage4Error("Candidate margin must be smaller than reference margin")
        samples = _read_samples(
            args.samples, candidate_plan["scan_key"], candidate_plan["group_name"]
        )
        config = Path(args.config); checkpoint = Path(args.checkpoint)
        if sha256_file(config) != stage3.EXPECTED_CONFIG_SHA256 or sha256_file(checkpoint) != stage3.EXPECTED_CHECKPOINT_SHA256:
            raise Stage4Error("Config or checkpoint hash mismatch")
        model_started = time.time()
        model, hook_present = stage3._load_model(config, checkpoint, args.precision)
        torch.cuda.synchronize(); model_seconds = time.time() - model_started
        model_dtype = str(next(model.parameters()).dtype)
        if model_dtype != "torch.float32" or not hook_present:
            raise Stage4Error("Stage 4 model precision contract failed")
        extractions = {}
        descriptors = {}
        for margin, plan in (
            (candidate_margin, candidate_plan),
            (reference_margin, reference_plan),
        ):
            try:
                prep_started = time.time(); data = stage3._legacy_prepare(plan)
                prep_seconds = time.time() - prep_started
                extraction = _extract_once(model, data, plan, samples, args.precision)
                extraction["preprocessing_seconds"] = prep_seconds
                descriptors[margin] = extraction.pop("descriptors")
                extractions[str(margin)] = extraction
                del data; gc.collect()
            except RuntimeError as exc:
                message = str(exc)
                if "out of memory" in message.lower() and "cuda" in message.lower():
                    result.update({
                        "status": "failed", "failure_classification": "cuda_oom",
                        "failed_margin_mm": margin, "error": message[-4000:],
                    })
                    atomic_json(result_path, dict(result, completed_at=utc_now(), wall_time_seconds=time.time() - started))
                    return 3
                raise
        boundary_comparison = "{}mm_vs_{}mm".format(
            candidate_margin, reference_margin
        )
        point_rows, boundary_summary = cosine_rows(
            descriptors[candidate_margin], descriptors[reference_margin],
            samples, boundary_comparison,
        )
        repeatability_summary = []
        repeatability_rows = []
        if args.repeatability:
            data = stage3._legacy_prepare(candidate_plan)
            repeated = _extract_once(
                model, data, candidate_plan, samples, args.precision,
                capture_memory=False,
            )
            repeatability_rows, repeatability_summary = cosine_rows(
                descriptors[candidate_margin], repeated["descriptors"], samples,
                "{}mm_repeatability".format(candidate_margin),
            )
            del data; gc.collect()
        memory_peaks = []
        for extraction in extractions.values():
            reserved = float(extraction["memory"]["torch_peak_reserved_bytes"]) / 1048576.0
            process_peak = extraction["memory"]["process_gpu_peak_mib"]
            memory_peaks.append(max([value for value in (reserved, process_peak) if value is not None]))
        result.update({
            "status": "success", "failure_classification": None,
            "scan_key": candidate_plan["scan_key"],
            "session": candidate_plan["session"],
            "group_name": candidate_plan["group_name"],
            "sample_count": len(samples),
            "candidate_margin_mm": candidate_margin,
            "reference_margin_mm": reference_margin,
            "model_dtype": model_dtype, "precision_contract": {
                "training_fp16_hook_present": hook_present,
                "training_fp16_hook_ignored": True,
                "autocast_enabled": args.precision == "amp",
                "full_fp16_used": False,
                "returned_embedding_dtype": "torch.float16",
                "passed": True,
            },
            "model_load_seconds": model_seconds, "extractions": extractions,
            "boundary_summary": boundary_summary,
            "repeatability_summary": repeatability_summary,
            "point_cosines": point_rows + repeatability_rows,
            "measured_peak_mib": max(memory_peaks),
            "cpu_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        })
    except RuntimeError as exc:
        message = str(exc); oom = "out of memory" in message.lower() and "cuda" in message.lower()
        result.update({"status": "failed", "failure_classification": "cuda_oom" if oom else "model_error", "error": message[-4000:]})
    except Exception as exc:
        result.update({"status": "failed", "failure_classification": "environment_error", "error": repr(exc)})
    result["completed_at"] = utc_now(); result["wall_time_seconds"] = time.time() - started
    atomic_json(result_path, result)
    return 0 if result["status"] == "success" else 3


def _worker_command(args, candidate_plan, reference_plan, result_path, signature, repeatability):
    command = [
        sys.executable, "-m", "tools.quadra.organ_group_numerical_validation", "_worker",
        "--plan100", str(candidate_plan), "--plan120", str(reference_plan),
        "--samples", str(Path(args.run_directory) / "sample_points_raw_itk.csv"),
        "--precision", args.precision, "--config", args.config,
        "--checkpoint", args.checkpoint, "--result-path", str(result_path),
        "--worker-signature", signature,
        "--validation-id", args.validation_id,
    ]
    if repeatability:
        command.append("--repeatability")
    return command


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
        classification = "timeout" if returncode is None else ("process_kill" if returncode < 0 else "process_crash")
        atomic_json(result_path, {
            "schema_version": SCHEMA_VERSION,
            "validation_id": command[command.index("--validation-id") + 1],
            "status": "failed", "failure_classification": classification,
            "worker_signature": command[command.index("--worker-signature") + 1],
            "returncode": returncode, "completed_at": utc_now(),
        })
    return load_json(result_path)


def _validate_prepared_run(run_dir, storage_root, profile):
    manifest_path = Path(run_dir) / "stage4_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") not in ("prepared", "benchmarked")
        or manifest.get("validation_id") not in (VALIDATION_ID, RESOLUTION_VALIDATION_ID)
    ):
        raise Stage4Error("Stage 4 preparation is incomplete")
    repository = validate_repository(PROJECT_ROOT)
    if repository["execution_commit"] != manifest["repository"]["execution_commit"]:
        raise Stage4Error("Repository commit differs from Stage 4 preparation")
    validate_stage3_checkpoint(manifest["stage3_checkpoint"]["path"])
    validate_resolution_source(manifest)
    profile_record = stage3.read_profile_fingerprint(storage_root, profile)
    baseline = stage3.require_model_contract(manifest["baseline_manifest"]["path"])
    stage3.require_gpu_matches_baseline(baseline, profile_record)
    return manifest_path, manifest, profile_record


def _plan_paths(run_dir, manifest):
    plans = {}
    for path in sorted((Path(run_dir) / "plans").glob("*.json")):
        plan = load_json(path)
        plans[(plan["scan_key"], plan["group_name"], int(round(plan["margin_mm"])))] = path
    selected = int(round(manifest["settings"]["selected_margin_mm"]))
    reference = int(round(manifest["settings"]["reference_margin_mm"]))
    if len(plans) != 16 or {key[2] for key in plans} != {selected, reference}:
        raise Stage4Error(
            "Expected 16 frozen {}/{} mm plans".format(selected, reference)
        )
    return plans


def run_precision(args, manifest, precision):
    run_dir = Path(args.run_directory)
    plans = _plan_paths(run_dir, manifest)
    selected_margin = int(round(manifest["settings"]["selected_margin_mm"]))
    reference_margin = int(round(manifest["settings"]["reference_margin_mm"]))
    results_dir = run_dir / "worker_results" / precision; results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "worker_logs" / precision; logs_dir.mkdir(parents=True, exist_ok=True)
    samples_identity = file_identity(run_dir / "sample_points_raw_itk.csv")
    keys = sorted({(key[0], key[1]) for key in plans})
    selected_paths = [
        plans[(scan_key, group, selected_margin)] for scan_key, group in keys
    ]
    smallest = min(selected_paths, key=lambda path: load_json(path)["padded_2mm_voxels"])
    smallest_plan = load_json(smallest)
    results = []
    for scan_key, group in keys:
        candidate_plan = plans[(scan_key, group, selected_margin)]
        reference_plan = plans[(scan_key, group, reference_margin)]
        repeatability = scan_key == smallest_plan["scan_key"] and group == smallest_plan["group_name"]
        signature = worker_signature(
            file_identity(candidate_plan), file_identity(reference_plan),
            samples_identity, precision, repeatability,
            validation_id=manifest["validation_id"],
        )
        result_path = results_dir / "{}-{}.json".format(scan_key, group)
        if result_path.exists():
            result = load_json(result_path)
            if result.get("worker_signature") != signature:
                raise Stage4Error("Incompatible resumable worker result: {}".format(result_path))
        else:
            label = "Stage 4B" if manifest["validation_id"] == RESOLUTION_VALIDATION_ID else "Stage 4"
            emit("{} {}: {} / {}".format(label, precision, scan_key, group), run_dir / "stage4.log")
            command_args = argparse.Namespace(**vars(args)); command_args.precision = precision
            command_args.validation_id = manifest["validation_id"]
            result = launch_worker(
                _worker_command(
                    command_args, candidate_plan, reference_plan, result_path,
                    signature, repeatability,
                ),
                result_path, logs_dir / "{}-{}.log".format(scan_key, group), args.timeout_seconds,
            )
        results.append(result)
        if result.get("status") != "success":
            break
    return results


def flatten_results(results, precision):
    memory_rows = []; summary_rows = []; point_rows = []
    for result in results:
        memory_rows.append({
            "precision": precision, "scan_key": result.get("scan_key", ""),
            "session": result.get("session", ""), "group_name": result.get("group_name", ""),
            "status": result.get("status"), "failure_classification": result.get("failure_classification") or "",
            "failed_margin_mm": result.get("failed_margin_mm", ""),
            "measured_peak_mib": result.get("measured_peak_mib", ""),
            "wall_time_seconds": result.get("wall_time_seconds", ""),
        })
        for row in result.get("boundary_summary", []) + result.get("repeatability_summary", []):
            summary_rows.append(dict(row, precision=precision, scan_key=result["scan_key"], session=result["session"], group_name=result["group_name"]))
        for row in result.get("point_cosines", []):
            point_rows.append(dict(row, precision=precision, scan_key=result["scan_key"], session=result["session"], group_name=result["group_name"]))
    return memory_rows, summary_rows, point_rows


def render_report(manifest, precision, results, summaries):
    selected_margin = int(round(manifest["settings"]["selected_margin_mm"]))
    reference_margin = int(round(manifest["settings"]["reference_margin_mm"]))
    stage_label = (
        "Stage 4B" if manifest["validation_id"] == RESOLUTION_VALIDATION_ID
        else "Stage 4"
    )
    lines = [
        "# {} organ-group numerical validation".format(stage_label), "", "## Decision scope", "",
        "Dense UAE-S extraction for all four organ groups in the largest Test/Retest pair. No matching or cycle error was run.", "",
        "Selected precision: `{}`. UAE-S explicitly returned FP16 embeddings; convolution/model precision followed the selected {} mode.".format(precision, precision.upper()), "",
        "## Boundary sensitivity", "",
        "| Scan | Group | Feature | Comparison | Median | P01 | Pass |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in summaries:
        lines.append("| {scan_key} | {group_name} | {feature} | {comparison} | {median_cosine:.6f} | {p01_cosine:.6f} | {passed} |".format(**row))
    lines.extend([
        "",
        "A failed {}-versus-{} mm gate blocks selection; it does not silently enlarge the production crop.".format(
            selected_margin, reference_margin
        ),
        "",
    ])
    return "\n".join(lines)


def run_benchmark(args):
    run_dir = Path(args.run_directory).resolve()
    manifest_path, manifest, profile = _validate_prepared_run(run_dir, Path(args.storage_root), "uae")
    if (run_dir / "checkpoint_summary.json").exists():
        raise Stage4Error("Stage 4 already has a final checkpoint")
    stage3.require_idle_gpu()
    fp32 = run_precision(args, manifest, "fp32")
    selected_precision = "fp32"
    results = fp32
    failure = next((result for result in fp32 if result.get("status") != "success"), None)
    selected_margin = int(round(manifest["settings"]["selected_margin_mm"]))
    stage_label = "Stage 4B" if manifest["validation_id"] == RESOLUTION_VALIDATION_ID else "Stage 4"
    if failure is not None:
        if (
            failure.get("failure_classification") == "cuda_oom"
            and int(failure.get("failed_margin_mm", -1)) == selected_margin
        ):
            emit(
                "Selected {} mm FP32 produced CUDA OOM; running the complete AMP fallback".format(
                    selected_margin
                ),
                run_dir / "stage4.log",
            )
            stage3.require_idle_gpu()
            results = run_precision(args, manifest, "amp")
            selected_precision = "amp"
        else:
            emit("{} benchmark BLOCKED by a non-fallback failure".format(stage_label), run_dir / "stage4.log")
    memory_rows, summaries, point_rows = flatten_results(results, selected_precision)
    write_csv(run_dir / "memory_runtime.csv", memory_rows)
    if summaries:
        write_csv(run_dir / "descriptor_boundary_summary.csv", summaries)
    if point_rows:
        write_csv(run_dir / "descriptor_point_cosines.csv", point_rows)
    atomic_text(run_dir / "validation_report.md", render_report(manifest, selected_precision, results, summaries))
    manifest.update({
        "status": "benchmarked", "benchmarked_at": utc_now(),
        "selected_precision_for_review": selected_precision,
        "uae_profile": profile,
        "worker_result_count": len(results),
        "outputs": dict(manifest.get("outputs", {}),
            memory_runtime=file_identity(run_dir / "memory_runtime.csv"),
            report=file_identity(run_dir / "validation_report.md"),
            descriptor_summary=file_identity(run_dir / "descriptor_boundary_summary.csv") if summaries else None,
            point_cosines=file_identity(run_dir / "descriptor_point_cosines.csv") if point_rows else None,
        ),
    })
    atomic_json(manifest_path, manifest)
    emit("{} benchmark complete; inspect evidence, then run select".format(stage_label), run_dir / "stage4.log")
    return run_dir


def evaluate_selection(manifest, results, summaries):
    failures = []
    if len(results) != 8:
        failures.append("incomplete_worker_count")
    for result in results:
        if result.get("status") != "success":
            failures.append("{}:{}".format(result.get("scan_key", "unknown"), result.get("failure_classification", "failed")))
            continue
        if result.get("precision_contract", {}).get("passed") is not True:
            failures.append("precision_contract")
        if result.get("measured_peak_mib", float("inf")) > stage3.VRAM_CEILING_MIB:
            failures.append("memory_ceiling")
        selected_margin = int(round(manifest["settings"]["selected_margin_mm"]))
        reference_margin = int(round(manifest["settings"]["reference_margin_mm"]))
        if set(result.get("extractions", {})) != {
            str(selected_margin), str(reference_margin)
        }:
            failures.append("missing_margin_extraction")
    expected_boundary = 8 * 3
    selected_margin = int(round(manifest["settings"]["selected_margin_mm"]))
    reference_margin = int(round(manifest["settings"]["reference_margin_mm"]))
    boundary_label = "{}mm_vs_{}mm".format(selected_margin, reference_margin)
    repeatability_label = "{}mm_repeatability".format(selected_margin)
    boundary = [row for row in summaries if row["comparison"] == boundary_label]
    repeatability = [row for row in summaries if row["comparison"] == repeatability_label]
    if len(boundary) != expected_boundary or any(not row["passed"] for row in boundary):
        failures.append("boundary_sensitivity")
    if len(repeatability) != 3 or any(not row["passed"] for row in repeatability):
        failures.append("repeatability")
    if manifest.get("maximum_roundtrip_voxel_error", float("inf")) > ROUNDTRIP_VOXEL_ATOL:
        failures.append("raw_coordinate_roundtrip")
    if manifest.get("maximum_roundtrip_physical_error_mm", float("inf")) > ROUNDTRIP_PHYSICAL_ATOL_MM:
        failures.append("physical_coordinate_roundtrip")
    return sorted(set(failures))


def forbidden_full_volume_outputs(run_dir):
    forbidden = (".nii", ".nii.gz", ".npy", ".npz", ".mmap", ".pt", ".pth")
    return [
        str(path)
        for path in Path(run_dir).rglob("*")
        if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in forbidden)
    ]


def run_select(args):
    run_dir = Path(args.run_directory).resolve()
    manifest_path = run_dir / "stage4_manifest.json"; manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "benchmarked"
        or manifest.get("validation_id") not in (VALIDATION_ID, RESOLUTION_VALIDATION_ID)
    ):
        raise Stage4Error("Stage 4 benchmark is incomplete")
    validate_repository(PROJECT_ROOT)
    validate_resolution_source(manifest)
    selected_path = run_dir / "selected_stage4_configuration.json"
    checkpoint_path = run_dir / "checkpoint_summary.json"
    if selected_path.exists() or checkpoint_path.exists():
        raise Stage4Error("Stage 4 selection is immutable")
    precision = manifest["selected_precision_for_review"]
    result_paths = sorted((run_dir / "worker_results" / precision).glob("*.json"))
    results = [load_json(path) for path in result_paths]
    summaries = []
    for result in results:
        for row in result.get("boundary_summary", []) + result.get("repeatability_summary", []):
            summaries.append(dict(row, scan_key=result.get("scan_key"), group_name=result.get("group_name"), session=result.get("session")))
    failures = evaluate_selection(manifest, results, summaries)
    if forbidden_full_volume_outputs(run_dir):
        failures = sorted(set(failures + ["forbidden_full_volume_output_retained"]))
    status = "PASS" if not failures else "BLOCKED"
    selected_margin = int(round(manifest["settings"]["selected_margin_mm"]))
    reference_margin = int(round(manifest["settings"]["reference_margin_mm"]))
    resolution = manifest["validation_id"] == RESOLUTION_VALIDATION_ID
    stage_label = "Stage 4B" if resolution else "Stage 4"
    selection = {
        "schema_version": SCHEMA_VERSION,
        "validation_id": manifest["validation_id"],
        "status": status, "created_at": utc_now(),
        "selected_spatial_configuration": (
            "organ_group_{}mm".format(selected_margin) if status == "PASS" else None
        ),
        "selected_precision": precision if status == "PASS" else None,
        "fallback_used": precision == "amp",
        "model_compute_dtype": "torch.float32",
        "returned_embedding_dtype": "torch.float16",
        "subject_id": EXPECTED_SUBJECT,
        "groups": list(stage3.GROUPS),
        "boundary_reference_margin_mm": reference_margin,
        "source_stage4a": manifest.get("source_stage4a"),
        "failures": failures,
        "limitations": [
            "Validation covers the largest Stage 3 Test/Retest pair, not the full cohort.",
            "Organ-group crops impose an anatomical search prior.",
            "Dense matching memory and cycle-error stability were not tested in Stage 4.",
        ],
    }
    atomic_json(selected_path, selection, refuse=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "stage": 4, "status": status,
        "substage": "B" if resolution else "A",
        "created_at": utc_now(), "validation_id": manifest["validation_id"],
        "selected_configuration": file_identity(selected_path),
        "selected_precision": selection["selected_precision"],
        "selected_margin_mm": selected_margin if status == "PASS" else None,
        "gates": {
            "stage0_to_stage3_contracts_validated": True,
            "all_four_groups_both_sessions_completed": len(results) == 8 and all(r.get("status") == "success" for r in results),
            "foreground_samples_contained": all(row["outside_selected_crop"] == "0" for row in csv.DictReader(open(run_dir / "containment_summary.csv"))),
            "coordinate_roundtrip_passed": manifest.get("maximum_roundtrip_voxel_error", float("inf")) <= ROUNDTRIP_VOXEL_ATOL and manifest.get("maximum_roundtrip_physical_error_mm", float("inf")) <= ROUNDTRIP_PHYSICAL_ATOL_MM,
            "output_geometry_precision_and_finiteness_passed": "precision_contract" not in failures and "missing_margin_extraction" not in failures,
            "candidate_vs_reference_boundary_gate_passed": "boundary_sensitivity" not in failures,
            "{}mm_vs_{}mm_boundary_gate_passed".format(
                selected_margin, reference_margin
            ): "boundary_sensitivity" not in failures,
            "same_crop_repeatability_passed": "repeatability" not in failures,
            "memory_headroom_passed": "memory_ceiling" not in failures,
            "matching_or_cycle_error_run": False,
            "embeddings_or_prepared_volumes_retained": bool(forbidden_full_volume_outputs(run_dir)),
            "full_fp16_used": False,
        },
        "next_stage": "exact_global_matching" if status == "PASS" else "resolve_stage4_blocker",
    }
    atomic_json(checkpoint_path, checkpoint, refuse=True)
    manifest.update({
        "status": "selected" if status == "PASS" else "blocked",
        "completed_at": utc_now(), "selected_configuration": file_identity(selected_path),
        "checkpoint_summary": file_identity(checkpoint_path),
    })
    atomic_json(manifest_path, manifest)
    print("{} {}".format(stage_label, status), flush=True)
    print("Selected configuration: {}".format(selection["selected_spatial_configuration"]), flush=True)
    print("Selected precision: {}".format(selection["selected_precision"]), flush=True)
    print("Checkpoint: {}".format(checkpoint_path), flush=True)
    return status == "PASS"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    prepare = sub.add_parser(
        "prepare",
        help="Prepare Stage 4A, or Stage 4B when --stage4a-checkpoint is supplied.",
    )
    prepare.add_argument("--baseline-manifest", default=str(stage3.EXPECTED_BASELINE))
    prepare.add_argument("--stage1-checkpoint", default=str(stage3.EXPECTED_STAGE1))
    prepare.add_argument("--stage2-checkpoint", default=str(stage3.EXPECTED_STAGE2))
    prepare.add_argument("--stage3-checkpoint", default=str(EXPECTED_STAGE3))
    prepare.add_argument(
        "--stage4a-checkpoint", default=None,
        help="Create isolated Stage 4B 120/150 mm evidence from the blocked Stage 4A checkpoint.",
    )
    prepare.add_argument("--storage-root", default="/workspace/quadra")
    prepare.add_argument("--repository-root", default=str(PROJECT_ROOT))
    prepare.add_argument("--output-root", default=None); prepare.add_argument("--run-id", default=None)
    prepare.add_argument("--resume-run-directory", default=None)
    benchmark = sub.add_parser(
        "benchmark", help="Run fresh-process dense candidate/reference descriptor validation."
    )
    benchmark.add_argument("--run-directory", required=True)
    benchmark.add_argument("--storage-root", default="/workspace/quadra")
    benchmark.add_argument("--config", default="configs/samv2/samv2_NIHLN.py")
    benchmark.add_argument("--checkpoint", default="checkpoints/SAMv2_iter_20000.pth")
    benchmark.add_argument("--timeout-seconds", type=int, default=WORKER_TIMEOUT_SECONDS)
    select = sub.add_parser(
        "select", help="Freeze the candidate margin only after every Stage 4 gate passes."
    )
    select.add_argument("--run-directory", required=True)
    worker = sub.add_parser("_worker")
    worker.add_argument("--plan100", required=True); worker.add_argument("--plan120", required=True)
    worker.add_argument("--samples", required=True); worker.add_argument("--precision", choices=PRECISION_ORDER, required=True)
    worker.add_argument("--config", required=True); worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--result-path", required=True); worker.add_argument("--worker-signature", required=True)
    worker.add_argument(
        "--validation-id", choices=(VALIDATION_ID, RESOLUTION_VALIDATION_ID),
        default=VALIDATION_ID,
    )
    worker.add_argument("--repeatability", action="store_true")
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
    except Stage4Error as exc:
        parser.error("Stage 4 failed: {}".format(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
