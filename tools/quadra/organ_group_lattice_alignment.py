#!/usr/bin/env python
"""Stage 5R global-lattice-aligned organ-group match sensitivity.

Stage 5R preserves the immutable BLOCKED Stage 5 evidence and changes only the
crop-grid construction.  Every organ-group envelope is an outward, stride-
snapped subregion of one full-image 2 mm lattice.  A four-cell factorial then
separates query-context, target-context, and combined margin sensitivity while
searching one identical aligned-100 valid target domain.

The module is intentionally compatible with Python 3.7 for the pinned UAE-S
container.
"""

from __future__ import print_function

import argparse
import gc
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
from tools.quadra import organ_group_match_sensitivity as stage5  # noqa: E402
from tools.quadra import organ_group_numerical_validation as stage4  # noqa: E402


SCHEMA_VERSION = 1
RESOLUTION_ID = "quadra-organ-group-global-lattice-v1"
RUN_PREFIX = "stage5r-lattice-alignment-"
EXPECTED_SUBJECT = "quadra_hc_030"
MARGINS_MM = (100, 120)
STRIDE_XYZ = (16, 16, 4)
SPACING_XYZ_MM = (2.0, 2.0, 2.0)
CONFIGURATIONS = {
    "A": (100, 100),
    "B": (120, 100),
    "C": (100, 120),
    "D": (120, 120),
}
CONTRASTS = ("A_vs_B", "A_vs_C", "A_vs_D")
WORKER_TIMEOUT_SECONDS = 60 * 60
QUERY_BATCH_SIZE = stage5.QUERY_BATCH_SIZE
MATCH_CHUNK_XYZ = stage5.MATCH_CHUNK_XYZ
VRAM_CEILING_MIB = stage5.VRAM_CEILING_MIB


class Stage5RError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise Stage5RError("Cannot read JSON {}: {}".format(path, exc))
    if not isinstance(value, dict):
        raise Stage5RError("Expected a JSON object: {}".format(path))
    return value


def file_identity(path):
    return stage5.file_identity(Path(path))


def atomic_json(path, value, refuse=False):
    try:
        stage5.atomic_json(path, value, refuse=refuse)
    except stage5.Stage5Error as exc:
        raise Stage5RError(str(exc))


def atomic_text(path, value):
    stage5.atomic_text(path, value)


def write_csv(path, rows, fieldnames=None):
    stage5.write_csv(path, rows, fieldnames=fieldnames)


def read_csv(path):
    return stage5.read_csv(path)


def apply_affine(points_xyz, affine):
    return stage5.apply_affine(points_xyz, affine)


def _array(value, dtype=float):
    import numpy as np
    return np.asarray(value, dtype=dtype)


def validate_stage5_blocked_checkpoint(path):
    """Validate and return immutable Stage 5 BLOCKED lineage."""
    identity = file_identity(path)
    checkpoint = load_json(path)
    gates = checkpoint.get("gates", {})
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("stage") != 5
        or checkpoint.get("validation_id") != stage5.VALIDATION_ID
        or checkpoint.get("status") != "BLOCKED"
        or gates.get("stage4a_and_stage4b_blocked_evidence_preserved") is not True
        or gates.get("bounded_dense_streamed_equivalence_passed") is not True
        or gates.get("all_eight_group_margin_workers_completed") is not True
        or gates.get("global_nn_all_queries_completed") is not True
        or gates.get("global_nn_crop_sensitivity_passed") is not False
        or gates.get("no_120mm_match_outside_100mm_crop") is not True
        or gates.get("cohort_authorized") is not False
        or checkpoint.get("fixed_point_status") != "PROVISIONAL_CONCERN"
    ):
        raise Stage5RError("Stage 5 checkpoint is not the accepted BLOCKED evidence")
    selection_ref = checkpoint.get("selected_workflow")
    if not isinstance(selection_ref, dict) or file_identity(selection_ref.get("path")) != selection_ref:
        raise Stage5RError("Stage 5 selection evidence changed")
    selection = load_json(selection_ref["path"])
    if selection.get("status") != "BLOCKED" or selection.get("selected_workflow") is not None:
        raise Stage5RError("Stage 5 BLOCKED selection contract changed")
    run_dir = Path(path).resolve().parent
    manifest_path = run_dir / "stage5_manifest.json"
    manifest_identity = file_identity(manifest_path)
    manifest = load_json(manifest_path)
    if (
        manifest.get("validation_id") != stage5.VALIDATION_ID
        or manifest.get("status") != "blocked"
        or manifest.get("subject_id") != EXPECTED_SUBJECT
    ):
        raise Stage5RError("Stage 5 manifest is not finalized BLOCKED evidence")
    stage5._validate_frozen_outputs(manifest)
    if int(manifest.get("global_query_count", -1)) != 2482:
        raise Stage5RError("Stage 5 frozen global query denominator changed")
    return {
        "checkpoint": identity,
        "checkpoint_payload": checkpoint,
        "selection": selection_ref,
        "manifest": manifest_identity,
        "manifest_payload": manifest,
        "run_directory": str(run_dir),
    }


def full_reference_lattice(source_ct):
    """Return the unpadded full-image TorchIO-compatible 2 mm lattice."""
    import numpy as np
    from tools.quadra import body_envelope_audit as stage1

    raw_shape = np.asarray(source_ct["native_shape_xyz"], dtype=np.int64)
    raw_affine = np.asarray(source_ct["affine"], dtype=np.float64)
    native_spacing = np.linalg.norm(raw_affine[:3, :3], axis=0)
    target_shape = stage1.torchio_target_shape(
        raw_shape, native_spacing, SPACING_XYZ_MM
    )
    affine, _ = stage1.resampled_and_padded_affines(
        raw_affine, raw_shape, target_shape, [0, 0, 0], SPACING_XYZ_MM
    )
    return target_shape, affine


def outward_stride_snap(start_xyz, stop_xyz, stride_xyz=STRIDE_XYZ):
    """Snap a half-open continuous grid interval outward to stride lines."""
    import numpy as np

    start = np.asarray(start_xyz, dtype=np.float64)
    stop = np.asarray(stop_xyz, dtype=np.float64)
    stride = np.asarray(stride_xyz, dtype=np.int64)
    if start.shape != (3,) or stop.shape != (3,) or np.any(stop <= start):
        raise Stage5RError("Invalid continuous lattice interval")
    lower = np.floor(start).astype(np.int64)
    upper = np.ceil(stop).astype(np.int64)
    snapped_lower = np.floor_divide(lower, stride) * stride
    snapped_upper = np.ceil(upper.astype(np.float64) / stride).astype(np.int64) * stride
    if np.any(snapped_upper <= snapped_lower) or np.any((snapped_upper - snapped_lower) % stride):
        raise Stage5RError("Outward stride snapping failed")
    return snapped_lower, snapped_upper


def aligned_plan_from_union(old_plan, margin_mm):
    """Derive one margin from the frozen mask union on a global 2 mm lattice."""
    import numpy as np

    if int(round(float(old_plan["margin_mm"]))) not in MARGINS_MM:
        raise Stage5RError("Unexpected source-plan margin")
    source = old_plan["source_ct"]
    raw_affine = np.asarray(source["affine"], dtype=np.float64)
    union_start = np.asarray(old_plan["mask_union_start_xyz"], dtype=np.float64)
    union_stop = np.asarray(old_plan["mask_union_end_xyz"], dtype=np.float64)
    if np.any(union_stop <= union_start):
        raise Stage5RError("Invalid frozen mask-union bounds")
    global_shape, global_affine = full_reference_lattice(source)
    inverse_global = np.linalg.inv(global_affine)

    # Convert raw voxel-cell boundaries, not only voxel centres, into global
    # lattice cell-boundary coordinates.  The +0.5 changes centre indices into
    # half-open cell coordinates before the physical margin is applied.
    raw_boundary_low = union_start - 0.5
    raw_boundary_high = union_stop - 0.5
    corners = np.asarray(
        [
            [x, y, z]
            for x in (raw_boundary_low[0], raw_boundary_high[0])
            for y in (raw_boundary_low[1], raw_boundary_high[1])
            for z in (raw_boundary_low[2], raw_boundary_high[2])
        ],
        dtype=np.float64,
    )
    physical = apply_affine(corners, raw_affine)
    global_centres = apply_affine(physical, inverse_global)
    global_boundaries = global_centres + 0.5
    margin_grid = float(margin_mm) / np.asarray(SPACING_XYZ_MM, dtype=np.float64)
    continuous_start = global_boundaries.min(axis=0) - margin_grid
    continuous_stop = global_boundaries.max(axis=0) + margin_grid
    snapped_start, snapped_stop = outward_stride_snap(continuous_start, continuous_stop)

    valid_global_start = np.maximum(snapped_start, np.zeros(3, dtype=np.int64))
    valid_global_stop = np.minimum(snapped_stop, global_shape)
    if np.any(valid_global_stop <= valid_global_start):
        raise Stage5RError("Aligned region has no acquired-FOV intersection")
    valid_local = np.stack(
        (valid_global_start - snapped_start, valid_global_stop - snapped_start)
    )
    shape = snapped_stop - snapped_start
    model_affine = global_affine.copy()
    model_affine[:3, 3] = apply_affine(snapped_start, global_affine)
    raw_to_model = np.linalg.inv(model_affine).dot(raw_affine)
    plan = dict(old_plan)
    # Remove the independently cropped/resampled geometry inherited from
    # Stage 5.  Leaving these fields beside the aligned affine would create an
    # ambiguous plan that a future caller could interpret incorrectly.
    for obsolete in (
        "crop_start_xyz", "crop_end_xyz", "crop_shape_xyz",
        "target_shape_xyz", "padding_lower_xyz", "padding_upper_xyz",
        "native_crop_affine", "resampled_2mm_affine",
        "raw_to_crop_continuous_affine", "crop_to_raw_continuous_affine",
        "crop_to_model_continuous_affine", "model_to_crop_continuous_affine",
    ):
        plan.pop(obsolete, None)
    plan.update(
        {
            "alignment_id": RESOLUTION_ID,
            "strategy": "organ_group_global_lattice",
            "axis_policy": "global_lattice_xyz",
            "margin_mm": float(margin_mm),
            "global_lattice_affine": global_affine.tolist(),
            "global_lattice_shape_xyz": global_shape.tolist(),
            "continuous_envelope_start_global_xyz": continuous_start.tolist(),
            "continuous_envelope_stop_global_xyz": continuous_stop.tolist(),
            "global_grid_start_xyz": snapped_start.tolist(),
            "global_grid_stop_xyz": snapped_stop.tolist(),
            "valid_global_box_xyz": [valid_global_start.tolist(), valid_global_stop.tolist()],
            "valid_model_box_xyz": valid_local.tolist(),
            "padded_shape_xyz": shape.tolist(),
            "target_shape_xyz": shape.tolist(),
            "padding_lower_xyz": [0, 0, 0],
            "padding_upper_xyz": [0, 0, 0],
            "model_tensor_shape_zyx": shape[::-1].tolist(),
            "padded_2mm_voxels": int(np.prod(shape, dtype=np.int64)),
            "padded_2mm_affine": model_affine.tolist(),
            "raw_to_model_continuous_affine": raw_to_model.tolist(),
            "model_to_raw_continuous_affine": np.linalg.inv(raw_to_model).tolist(),
            "fov_extension_lower_xyz": np.maximum(-snapped_start, 0).tolist(),
            "fov_extension_upper_xyz": np.maximum(snapped_stop - global_shape, 0).tolist(),
            "normalization_padding_hu": -1024.0,
            "normalization_padding_value": -50.0,
            "spacing_xyz_mm": list(SPACING_XYZ_MM),
        }
    )
    return plan


def assert_pair_alignment(plan100, plan120):
    """Require containment and identical global fine/coarse phase."""
    import numpy as np

    if plan100["scan_key"] != plan120["scan_key"] or plan100["group_name"] != plan120["group_name"]:
        raise Stage5RError("Aligned plan pair identity mismatch")
    start100 = np.asarray(plan100["global_grid_start_xyz"], dtype=np.int64)
    stop100 = np.asarray(plan100["global_grid_stop_xyz"], dtype=np.int64)
    start120 = np.asarray(plan120["global_grid_start_xyz"], dtype=np.int64)
    stop120 = np.asarray(plan120["global_grid_stop_xyz"], dtype=np.int64)
    stride = np.asarray(STRIDE_XYZ, dtype=np.int64)
    if np.any(start120 > start100) or np.any(stop120 < stop100):
        raise Stage5RError("Aligned 100 mm region is not contained by 120 mm")
    if np.any((start100 - start120) % stride):
        raise Stage5RError("Aligned margins do not share the model stride phase")
    affine100 = np.asarray(plan100["padded_2mm_affine"], dtype=np.float64)
    affine120 = np.asarray(plan120["padded_2mm_affine"], dtype=np.float64)
    mapped_origin = apply_affine(start100 - start120, affine120)
    if not np.allclose(mapped_origin, affine100[:3, 3], atol=1e-5, rtol=0.0):
        raise Stage5RError("Aligned margins do not share the global physical lattice")
    return True


def admissible_box_for_target(aligned100, target_plan):
    """Express the aligned-100 valid global box in a target plan's local grid."""
    import numpy as np

    global_box = np.asarray(aligned100["valid_global_box_xyz"], dtype=np.int64)
    target_start = np.asarray(target_plan["global_grid_start_xyz"], dtype=np.int64)
    local = global_box - target_start[None, :]
    shape = np.asarray(target_plan["padded_shape_xyz"], dtype=np.int64)
    if np.any(local[0] < 0) or np.any(local[1] > shape) or np.any(local[1] <= local[0]):
        raise Stage5RError("Shared aligned-100 target box is outside the target plan")
    return local.tolist()


def validate_raw_model_roundtrip(plan, points_raw):
    import numpy as np

    points = np.asarray(points_raw, dtype=np.float64)
    model = apply_affine(points, plan["raw_to_model_continuous_affine"])
    returned = apply_affine(model, plan["model_to_raw_continuous_affine"])
    return float(np.max(np.abs(returned - points)))


def _stage5_plan_lookup(evidence):
    manifest = evidence["manifest_payload"]
    lookup = stage5._frozen_plan_paths(evidence["run_directory"], manifest)
    result = {}
    for key, path in lookup.items():
        plan = load_json(path)
        plan["source_stage5_plan"] = file_identity(path)
        result[key] = plan
    return result


def _run_directory(output_root, run_id=None, resume=None):
    if resume:
        path = Path(resume).resolve()
        if not path.is_dir():
            raise Stage5RError("Resume directory is missing: {}".format(path))
        return path, True
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_root) / (run_id or RUN_PREFIX + timestamp)
    if path.exists():
        raise Stage5RError("Refusing to overwrite Stage 5R directory: {}".format(path))
    path.mkdir(parents=True)
    return path, False


def _make_qc(prepared, plan, path):
    stage3.make_qc(prepared, dict(plan, strategy="aligned_global_lattice"), path)


def run_prepare(args):
    from tools.quadra import optimization_baseline as baseline
    from tools.quadra import coordinate_preserving_crop as stage2

    repository = stage5.validate_repository(Path(args.repository_root))
    evidence = validate_stage5_blocked_checkpoint(args.stage5_checkpoint)
    old_manifest = evidence["manifest_payload"]
    baseline.validate_locked_contract(
        Path(old_manifest["baseline_manifest"]["path"]),
        repository_root=Path(args.repository_root),
        storage_root=Path(args.storage_root),
        required_profile="preprocess",
    )
    old_plans = _stage5_plan_lookup(evidence)
    output_root = Path(
        args.output_root or Path(args.storage_root) / "runs/memory_optimization"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir, resuming = _run_directory(output_root, args.run_id, args.resume_run_directory)
    manifest_path = run_dir / "stage5r_manifest.json"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "resolution_id": RESOLUTION_ID,
        "status": "preparing",
        "created_at": utc_now(),
        "repository": repository,
        "baseline_manifest": old_manifest["baseline_manifest"],
        "stage5_blocked_checkpoint": evidence["checkpoint"],
        "stage5_blocked_manifest": evidence["manifest"],
        "stage5_blocked_selection": evidence["selection"],
        "subject_id": EXPECTED_SUBJECT,
        "settings": {
            "margins_mm": list(MARGINS_MM),
            "spacing_xyz_mm": list(SPACING_XYZ_MM),
            "global_stride_xyz": list(STRIDE_XYZ),
            "precision": "fp32",
            "returned_embedding_dtype": "fp16",
            "similarity_dtype": "fp32",
            "configurations": {key: list(value) for key, value in CONFIGURATIONS.items()},
            "contrasts": list(CONTRASTS),
            "shared_target_domain": "aligned_100_valid_nonpadding",
            "query_batch_size": QUERY_BATCH_SIZE,
            "match_chunk_xyz": list(MATCH_CHUNK_XYZ),
            "vram_ceiling_mib": VRAM_CEILING_MIB,
        },
        "scope": {
            "global_nn": True,
            "all_2482_stage5_queries": True,
            "fixed_point_rerun": False,
            "fixed_point_status_preserved": "PROVISIONAL_CONCERN",
            "largest_pair_pilot": False,
            "cohort_authorized": False,
            "embeddings_saved": False,
        },
    }
    if resuming:
        existing = load_json(manifest_path)
        if existing.get("status") != "preparing":
            raise Stage5RError("Completed Stage 5R preparation is immutable")
        for key in (
            "resolution_id", "repository", "baseline_manifest",
            "stage5_blocked_checkpoint", "stage5_blocked_manifest",
            "stage5_blocked_selection", "subject_id", "settings", "scope",
        ):
            if existing.get(key) != contract.get(key):
                raise Stage5RError("Stage 5R resume contract changed: {}".format(key))
        contract = existing
    else:
        atomic_json(manifest_path, contract, refuse=True)

    plans_dir = run_dir / "plans"
    plans_dir.mkdir(exist_ok=True)
    plan_records = []
    aligned = {}
    for session in ("test", "retest"):
        for group in stage3.GROUPS:
            source_plan = old_plans[(session, group, 100)]
            for margin in MARGINS_MM:
                plan = aligned_plan_from_union(source_plan, margin)
                key = (session, group, margin)
                aligned[key] = plan
                target = plans_dir / "{}-{}-m{:03d}-aligned.json".format(
                    plan["scan_key"], group, margin
                )
                if target.exists():
                    if load_json(target) != plan:
                        raise Stage5RError("Existing aligned plan differs: {}".format(target))
                else:
                    atomic_json(target, plan, refuse=True)
                plan_records.append(
                    dict(
                        file_identity(target),
                        session=session,
                        group_name=group,
                        margin_mm=margin,
                    )
                )
            assert_pair_alignment(aligned[(session, group, 100)], aligned[(session, group, 120)])

    old_queries = old_manifest["outputs"]["global_queries"]
    if file_identity(old_queries["path"]) != old_queries:
        raise Stage5RError("Frozen Stage 5 query CSV changed")
    queries = read_csv(old_queries["path"])
    if len(queries) != 2482:
        raise Stage5RError("Expected exactly 2,482 frozen Stage 5 queries")
    query_target = run_dir / "global_nn_queries_raw_itk.csv"
    write_csv(query_target, queries)
    query_identity = file_identity(query_target)

    coordinate_rows = []
    for row in queries:
        raw = [float(row[key]) for key in ("raw_x", "raw_y", "raw_z")]
        for margin in MARGINS_MM:
            plan = aligned[(row["source_session"], row["group_name"], margin)]
            error = validate_raw_model_roundtrip(plan, [raw])
            if error > 1e-6:
                raise Stage5RError("Raw/model round trip failed for {}".format(row["query_id"]))
            model = apply_affine(raw, plan["raw_to_model_continuous_affine"])
            shape = _array(plan["padded_shape_xyz"], dtype=float)
            if (_array(model) < -1e-6).any() or (_array(model) > shape - 1 + 1e-6).any():
                raise Stage5RError("Frozen query lies outside aligned plan")
        coordinate_rows.append(
            {"query_id": row["query_id"], "max_raw_roundtrip_error": max(
                validate_raw_model_roundtrip(
                    aligned[(row["source_session"], row["group_name"], margin)], [raw]
                ) for margin in MARGINS_MM
            )}
        )
    write_csv(run_dir / "coordinate_roundtrip.csv", coordinate_rows)

    # Realize all 16 plans sequentially.  This validates FOV extension and
    # -1024 -> -50 padding without retaining a prepared volume.
    qc_dir = run_dir / "qc"
    preparation_rows = []
    for key in sorted(aligned):
        plan = aligned[key]
        prepared = stage2.prepare_scan_on_global_lattice(
            Path(plan["source_ct"]["path"]), plan
        )
        if prepared.metadata["normalized_outside_fov_padding_max_error"] > 1e-6:
            raise Stage5RError("Aligned padding validation failed")
        qc_path = qc_dir / "{}-{}-m{:03d}.png".format(plan["scan_key"], plan["group_name"], key[2])
        _make_qc(prepared, plan, qc_path)
        preparation_rows.append(
            {
                "session": key[0], "group_name": key[1], "margin_mm": key[2],
                "scan_key": plan["scan_key"],
                "shape_xyz": json.dumps(plan["padded_shape_xyz"]),
                "valid_box_xyz": json.dumps(plan["valid_model_box_xyz"]),
                "fov_extension_lower_xyz": json.dumps(plan["fov_extension_lower_xyz"]),
                "fov_extension_upper_xyz": json.dumps(plan["fov_extension_upper_xyz"]),
                "qc": str(qc_path), "status": "PASS",
            }
        )
        del prepared
        gc.collect()
    write_csv(run_dir / "aligned_spatial_plans.csv", preparation_rows)

    contract.update(
        {
            "status": "prepared",
            "prepared_at": utc_now(),
            "plans": plan_records,
            "global_query_count": len(queries),
            "outputs": {
                "global_queries": query_identity,
                "coordinate_roundtrip": file_identity(run_dir / "coordinate_roundtrip.csv"),
                "spatial_plans": file_identity(run_dir / "aligned_spatial_plans.csv"),
            },
        }
    )
    atomic_json(manifest_path, contract)
    print("Stage 5R prepare PASS", flush=True)
    print("Run directory: {}".format(run_dir), flush=True)
    return run_dir


class InMemoryUaesCache(stage5.InMemoryUaesCache):
    pass


def _extract_cache(model, plan):
    import numpy as np
    import torch

    started = time.time()
    data = _legacy_prepare_aligned(plan)
    preprocessing = time.time() - started
    tensor = torch.from_numpy(data)[None, None].cuda(non_blocking=False)
    del data
    torch.cuda.reset_peak_memory_stats()
    forward_started = time.time()
    with torch.no_grad():
        outputs = model.extract_feat(tensor)
    torch.cuda.synchronize()
    forward_seconds = time.time() - forward_started
    expected = stage3._expected_feature_shapes([int(value) for value in tensor.shape])
    arrays = []
    records = []
    for name, value in zip(("fine", "coarse", "semantic"), outputs):
        record = {
            "name": name, "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype), "finite": bool(torch.isfinite(value).all().item()),
        }
        if record["shape"] != expected[name] or record["dtype"] != "torch.float16" or not record["finite"]:
            raise Stage5RError("UAE-S output contract failed for {}".format(name))
        arrays.append(value[0].detach().cpu().numpy())
        records.append(record)
    extraction = {
        "preprocessing_seconds": preprocessing,
        "forward_seconds": forward_seconds,
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "features": records,
    }
    del tensor, outputs
    torch.cuda.empty_cache()
    return InMemoryUaesCache(
        arrays[0], arrays[1], arrays[2], plan["padded_shape_xyz"],
        cache_dir="{}/{}mm-aligned".format(plan["scan_key"], int(plan["margin_mm"])),
    ), extraction


def _legacy_prepare_aligned(plan):
    """Python-3.7-compatible realization of a frozen global-lattice plan."""
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk

    source = plan["source_ct"]
    path = Path(source["path"])
    if file_identity(path)["sha256"] != source["sha256"]:
        raise Stage5RError("Source CT hash changed: {}".format(path))
    image_nib = nib.load(str(path))
    raw_affine = np.asarray(source["affine"], dtype=np.float64)
    if tuple(image_nib.shape[:3]) != tuple(source["native_shape_xyz"]) or not np.allclose(
        image_nib.affine, raw_affine, atol=1e-5, rtol=0.0
    ):
        raise Stage5RError("Source CT geometry changed: {}".format(plan["scan_key"]))
    model_affine = np.asarray(plan["padded_2mm_affine"], dtype=np.float64)
    ras_lps = np.diag([-1.0, -1.0, 1.0, 1.0]).dot(model_affine)
    spacing = np.linalg.norm(ras_lps[:3, :3], axis=0)
    direction = ras_lps[:3, :3] / spacing
    image = sitk.ReadImage(str(path))
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(value) for value in plan["padded_shape_xyz"]])
    resampler.SetOutputSpacing(tuple(float(value) for value in spacing))
    resampler.SetOutputOrigin(tuple(float(value) for value in ras_lps[:3, 3]))
    resampler.SetOutputDirection(tuple(float(value) for value in direction.reshape(-1)))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    prepared = resampler.Execute(image)
    actual_spacing = np.asarray(prepared.GetSpacing(), dtype=np.float64)
    actual_direction = np.asarray(prepared.GetDirection(), dtype=np.float64).reshape(3, 3)
    actual_lps = np.eye(4, dtype=np.float64)
    actual_lps[:3, :3] = actual_direction * actual_spacing
    actual_lps[:3, 3] = np.asarray(prepared.GetOrigin(), dtype=np.float64)
    actual_ras = np.diag([-1.0, -1.0, 1.0, 1.0]).dot(actual_lps)
    if not np.allclose(actual_ras, model_affine, atol=1e-5, rtol=0.0):
        raise Stage5RError("Prepared global-lattice affine mismatch")
    data = sitk.GetArrayFromImage(prepared).astype(np.float32, copy=False)
    if data.shape != tuple(plan["model_tensor_shape_zyx"]):
        raise Stage5RError("Prepared global-lattice shape mismatch")
    valid = np.asarray(plan["valid_model_box_xyz"], dtype=np.int64)[:, ::-1]
    for axis in range(3):
        for start, stop in ((0, int(valid[0, axis])), (int(valid[1, axis]), data.shape[axis])):
            if stop <= start:
                continue
            slices = [slice(None)] * 3
            slices[axis] = slice(start, stop)
            if float(np.max(np.abs(data[tuple(slices)] + 1024.0))) > 1e-6:
                raise Stage5RError("Prepared out-of-FOV padding is not -1024 HU")
    if not np.isfinite(data).all():
        raise Stage5RError("Prepared CT contains non-finite values")
    np.clip(data, -1024.0, 3071.0, out=data)
    data -= np.float32(-1024.0)
    data *= np.float32(255.0 / 4095.0)
    data -= np.float32(50.0)
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    return data


def _raw_to_model_index(raw_xyz, plan):
    import numpy as np

    model = apply_affine(raw_xyz, plan["raw_to_model_continuous_affine"])
    index = np.rint(model).astype(np.int64)
    shape = np.asarray(plan["padded_shape_xyz"], dtype=np.int64)
    if np.any(index < 0) or np.any(index >= shape):
        raise Stage5RError("Raw query rounds outside aligned model grid")
    return index


def _model_to_raw(model_xyz, plan):
    return apply_affine(model_xyz, plan["model_to_raw_continuous_affine"])


def _physical(raw_xyz, plan):
    return apply_affine(raw_xyz, plan["source_ct"]["affine"])


def _descriptor_cosines(cache100, cache120, plan100, plan120, rows):
    import numpy as np
    import torch
    from tools.quadra.streaming_cycle_error import extract_uaes_query_descriptors

    raw = np.asarray(
        [[float(row[key]) for key in ("raw_x", "raw_y", "raw_z")] for row in rows],
        dtype=np.float64,
    )
    points100 = np.stack([_raw_to_model_index(point, plan100) for point in raw])
    points120 = np.stack([_raw_to_model_index(point, plan120) for point in raw])
    left = extract_uaes_query_descriptors(cache100, points100, torch.device("cuda:0"))[:3]
    right = extract_uaes_query_descriptors(cache120, points120, torch.device("cuda:0"))[:3]
    records = []
    for index, row in enumerate(rows):
        per_level = []
        for left_level, right_level in zip(left, right):
            per_level.append(float(torch.sum(left_level[index] * right_level[index]).item()))
        records.append(
            {
                "query_id": row["query_id"], "source_session": row["source_session"],
                "group_name": row["group_name"], "fine_cosine": per_level[0],
                "coarse_cosine": per_level[1], "semantic_cosine": per_level[2],
                "mean_cosine": float(np.mean(per_level)),
            }
        )
    del left, right
    torch.cuda.empty_cache()
    return records


def _factorial_records(
    config_id, source_session, group, source_margin, target_margin,
    source_cache, target_cache, source_plan, target_plan,
    source_plan100, target_plan100, rows,
):
    import numpy as np
    from tools.quadra.streaming_cycle_error import stream_global_match_uaes

    query_raw = np.asarray(
        [[float(row[key]) for key in ("raw_x", "raw_y", "raw_z")] for row in rows],
        dtype=np.float64,
    )
    query_model = np.stack([_raw_to_model_index(point, source_plan) for point in query_raw])
    forward_box = admissible_box_for_target(target_plan100, target_plan)
    matched_model, score_forward, profile_forward = stream_global_match_uaes(
        source_cache, target_cache, query_model, QUERY_BATCH_SIZE,
        MATCH_CHUNK_XYZ, output_space="native",
        admissible_target_box_xyz=forward_box,
    )
    backward_box = admissible_box_for_target(source_plan100, source_plan)
    returned_model, score_backward, profile_backward = stream_global_match_uaes(
        target_cache, source_cache, matched_model, QUERY_BATCH_SIZE,
        MATCH_CHUNK_XYZ, output_space="native",
        admissible_target_box_xyz=backward_box,
    )
    records = []
    for index, row in enumerate(rows):
        matched_raw = _model_to_raw(matched_model[index], target_plan)
        returned_raw = _model_to_raw(returned_model[index], source_plan)
        matched_physical = _physical(matched_raw, target_plan)
        returned_physical = _physical(returned_raw, source_plan)
        query_physical = _physical(query_raw[index], source_plan)
        matched_inside = bool(
            np.all(matched_model[index] >= np.asarray(forward_box[0], dtype=np.int64))
            and np.all(matched_model[index] < np.asarray(forward_box[1], dtype=np.int64))
        )
        returned_inside = bool(
            np.all(returned_model[index] >= np.asarray(backward_box[0], dtype=np.int64))
            and np.all(returned_model[index] < np.asarray(backward_box[1], dtype=np.int64))
        )
        if not matched_inside or not returned_inside:
            raise Stage5RError("Restricted matcher returned a point outside its admissible box")
        records.append(
            {
                "query_id": row["query_id"], "point_id": int(row["point_id"]),
                "configuration": config_id,
                "source_session": source_session,
                "target_session": "retest" if source_session == "test" else "test",
                "group_name": group, "mask_name": row["mask_name"],
                "source_margin_mm": source_margin, "target_margin_mm": target_margin,
                "query_raw_xyz": query_raw[index].tolist(),
                "query_model_xyz": query_model[index].tolist(),
                "matched_model_xyz": matched_model[index].tolist(),
                "returned_model_xyz": returned_model[index].tolist(),
                "matched_raw_xyz": matched_raw.tolist(),
                "returned_raw_xyz": returned_raw.tolist(),
                "matched_physical_xyz": matched_physical.tolist(),
                "returned_physical_xyz": returned_physical.tolist(),
                "cycle_error_mm": float(np.linalg.norm(query_physical - returned_physical)),
                "score_forward": float(score_forward[index]),
                "score_backward": float(score_backward[index]),
                "forward_admissible_box_xyz": forward_box,
                "backward_admissible_box_xyz": backward_box,
                "matched_inside_shared_domain": matched_inside,
                "returned_inside_shared_domain": returned_inside,
                "status": "success",
            }
        )
    return records, {"forward": profile_forward, "backward": profile_backward}


def _worker_signature(group, source_margin, plan_refs, queries, config, checkpoint):
    return stage5.sha256_payload(
        {
            "resolution_id": RESOLUTION_ID, "group": group,
            "source_margin_mm": int(source_margin),
            "plans": plan_refs, "queries": queries,
            "config": config, "checkpoint": checkpoint,
            "configurations": CONFIGURATIONS,
            "query_batch_size": QUERY_BATCH_SIZE,
            "match_chunk_xyz": MATCH_CHUNK_XYZ,
        }
    )


def configurations_for_source_margin(source_margin):
    margin = int(source_margin)
    if margin == 100:
        return ("A", "C")
    if margin == 120:
        return ("B", "D")
    raise Stage5RError("Worker source margin must be 100 or 120 mm")


def run_worker(args):
    result_path = Path(args.result_path)
    result = {
        "schema_version": SCHEMA_VERSION, "resolution_id": RESOLUTION_ID,
        "kind": "factorial_group", "status": "running",
        "started_at": utc_now(), "worker_signature": args.worker_signature,
        "pid": os.getpid(),
    }
    sampler = None
    started = time.time()
    try:
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise Stage5RError("CUDA is unavailable")
        torch.manual_seed(stage3.SEED)
        np.random.seed(stage3.SEED)
        torch.backends.cudnn.benchmark = stage3.CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = stage3.CUDNN_DETERMINISTIC
        plans = {}
        for session, margin, path in (
            ("test", 100, args.test_100_plan), ("test", 120, args.test_120_plan),
            ("retest", 100, args.retest_100_plan), ("retest", 120, args.retest_120_plan),
        ):
            plans[(session, margin)] = load_json(path)
        groups = {plan["group_name"] for plan in plans.values()}
        if len(groups) != 1:
            raise Stage5RError("Factorial worker plans disagree on group")
        group = next(iter(groups))
        source_margin_filter = int(args.source_margin)
        worker_configurations = configurations_for_source_margin(source_margin_filter)
        assert_pair_alignment(plans[("test", 100)], plans[("test", 120)])
        assert_pair_alignment(plans[("retest", 100)], plans[("retest", 120)])
        if stage4.sha256_file(Path(args.config)) != stage3.EXPECTED_CONFIG_SHA256:
            raise Stage5RError("Config hash mismatch")
        if stage4.sha256_file(Path(args.checkpoint)) != stage3.EXPECTED_CHECKPOINT_SHA256:
            raise Stage5RError("Checkpoint hash mismatch")
        model_started = time.time()
        model, hook_present = stage3._load_model(Path(args.config), Path(args.checkpoint), "fp32")
        torch.cuda.synchronize()
        if str(next(model.parameters()).dtype) != "torch.float32" or not hook_present:
            raise Stage5RError("FP32 model precision contract failed")
        model_seconds = time.time() - model_started
        sampler = stage3.NvidiaProcessSampler(os.getpid())
        sampler.start()
        caches = {}
        extractions = {}
        for key in (("test", 100), ("test", 120), ("retest", 100), ("retest", 120)):
            caches[key], extractions["{}-{}".format(*key)] = _extract_cache(model, plans[key])
        queries = [row for row in read_csv(args.queries) if row["group_name"] == group]
        descriptor_records = []
        if source_margin_filter == 100:
            for session in ("test", "retest"):
                subset = [row for row in queries if row["source_session"] == session]
                descriptor_records.extend(
                    _descriptor_cosines(
                        caches[(session, 100)], caches[(session, 120)],
                        plans[(session, 100)], plans[(session, 120)], subset,
                    )
                )
        records = []
        profiles = {}
        for config_id in worker_configurations:
            source_margin, target_margin = CONFIGURATIONS[config_id]
            for source_session, target_session in (("test", "retest"), ("retest", "test")):
                subset = [row for row in queries if row["source_session"] == source_session]
                generated, profile = _factorial_records(
                    config_id, source_session, group, source_margin, target_margin,
                    caches[(source_session, source_margin)],
                    caches[(target_session, target_margin)],
                    plans[(source_session, source_margin)],
                    plans[(target_session, target_margin)],
                    plans[(source_session, 100)], plans[(target_session, 100)], subset,
                )
                records.extend(generated)
                profiles["{}-{}".format(config_id, source_session)] = profile
        sampler.stop()
        process_peak = sampler.maximum
        sampler = None
        peaks = [
            float(item["torch_peak_reserved_bytes"]) / 1048576.0
            for item in extractions.values()
        ]
        for profile in profiles.values():
            for direction in ("forward", "backward"):
                value = profile[direction].get("peak_gpu_memory_bytes")
                if value is not None:
                    peaks.append(float(value) / 1048576.0)
        if process_peak is not None:
            peaks.append(float(process_peak))
        result.update(
            {
                "status": "success", "failure_classification": None,
                "group_name": group, "source_margin_mm": source_margin_filter,
                "configurations": list(worker_configurations),
                "query_count": len(queries),
                "factorial_record_count": len(records),
                "model_dtype": "torch.float32",
                "returned_embedding_dtype": "torch.float16",
                "similarity_dtype": "torch.float32",
                "factorial_records": records,
                "descriptor_cosines": descriptor_records,
                "profiles": profiles, "extractions": extractions,
                "model_load_seconds": model_seconds,
                "measured_peak_mib": max(peaks),
                "process_gpu_peak_mib": process_peak,
                "cpu_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
        )
        del caches, model
        gc.collect()
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        message = str(exc)
        oom = "out of memory" in message.lower() and "cuda" in message.lower()
        result.update(
            {"status": "failed", "failure_classification": "cuda_oom" if oom else "model_error", "error": message[-4000:]}
        )
    except Exception as exc:
        result.update(
            {"status": "failed", "failure_classification": "environment_error", "error": repr(exc)}
        )
    finally:
        if sampler is not None:
            sampler.stop()
        result["wall_time_seconds"] = time.time() - started
        result["completed_at"] = utc_now()
        atomic_json(result_path, result)
    return 0 if result.get("status") == "success" else 3


def run_equivalence_worker(args):
    result_path = Path(args.result_path)
    result = {
        "schema_version": SCHEMA_VERSION, "resolution_id": RESOLUTION_ID,
        "kind": "restricted_matcher_equivalence", "status": "running",
        "started_at": utc_now(), "worker_signature": args.worker_signature,
    }
    try:
        import numpy as np
        import torch
        from tools.quadra.streaming_cycle_error import stream_global_match_uaes

        plan = load_json(args.plan)
        model, _ = stage3._load_model(Path(args.config), Path(args.checkpoint), "fp32")
        cache, extraction = _extract_cache(model, plan)
        valid = np.asarray(plan["valid_model_box_xyz"], dtype=np.int64)
        stop = np.minimum(valid[0] + np.asarray([16, 16, 8]), valid[1])
        box = np.stack((valid[0], stop))
        extent = box[1] - box[0]
        points = np.asarray(
            [box[0], box[0] + (extent - 1) // 2, box[1] - 1], dtype=np.int64
        )
        dense_points, dense_scores, dense_profile = stream_global_match_uaes(
            cache, cache, points, len(points), extent.tolist(),
            output_space="native", admissible_target_box_xyz=box.tolist(),
        )
        chunked_points, chunked_scores, chunked_profile = stream_global_match_uaes(
            cache, cache, points, 2, [4, 4, 4],
            output_space="native", admissible_target_box_xyz=box.tolist(),
        )
        agreement = float(np.mean(np.all(dense_points == chunked_points, axis=1)))
        score_difference = float(np.max(np.abs(dense_scores - chunked_scores)))
        result.update(
            {
                "status": "success", "failure_classification": None,
                "argmax_agreement_rate": agreement,
                "max_score_abs_difference": score_difference,
                "passed": agreement == 1.0 and score_difference <= stage5.MATCH_SCORE_ATOL,
                "admissible_target_box_xyz": box.tolist(),
                "searched_locations_expected": int(np.prod(extent)),
                "dense_profile": dense_profile, "chunked_profile": chunked_profile,
                "extraction": extraction,
            }
        )
        del cache, model
        gc.collect()
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        message = str(exc)
        oom = "out of memory" in message.lower() and "cuda" in message.lower()
        result.update(
            {"status": "failed", "failure_classification": "cuda_oom" if oom else "model_error", "error": message[-4000:]}
        )
    except Exception as exc:
        result.update(
            {"status": "failed", "failure_classification": "environment_error", "error": repr(exc)}
        )
    result["completed_at"] = utc_now()
    atomic_json(result_path, result)
    return 0 if result.get("status") == "success" and result.get("passed") else 3


def _load_prepared(run_dir, storage_root, profile=None):
    manifest_path = Path(run_dir) / "stage5r_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("resolution_id") != RESOLUTION_ID or manifest.get("status") not in (
        "prepared", "benchmarked"
    ):
        raise Stage5RError("Stage 5R preparation is incomplete")
    repository = stage5.validate_repository(PROJECT_ROOT)
    if repository["execution_commit"] != manifest["repository"]["execution_commit"]:
        raise Stage5RError("Repository commit differs from Stage 5R preparation")
    validate_stage5_blocked_checkpoint(manifest["stage5_blocked_checkpoint"]["path"])
    profile_record = None
    if profile is not None:
        profile_record = stage3.read_profile_fingerprint(Path(storage_root), profile)
        baseline = stage3.require_model_contract(manifest["baseline_manifest"]["path"])
        stage3.require_gpu_matches_baseline(baseline, profile_record)
    return manifest_path, manifest, profile_record


def _frozen_plan_paths(run_dir, manifest):
    lookup = {}
    expected = {Path(item["path"]).resolve(): item for item in manifest.get("plans", [])}
    for path in sorted((Path(run_dir) / "plans").glob("*.json")):
        identity = file_identity(path)
        if path.resolve() not in expected or any(
            expected[path.resolve()].get(key) != identity[key] for key in ("path", "bytes", "sha256")
        ):
            raise Stage5RError("A frozen Stage 5R plan changed: {}".format(path))
        plan = load_json(path)
        lookup[(plan["session"], plan["group_name"], int(plan["margin_mm"]))] = path
    if len(lookup) != 16 or len(expected) != 16:
        raise Stage5RError("Expected exactly 16 frozen Stage 5R plans")
    return lookup


def _launch(command, result_path, log_path, timeout):
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = None
    if not Path(result_path).is_file():
        classification = "timeout" if returncode is None else (
            "process_kill" if returncode < 0 else "process_crash"
        )
        atomic_json(
            result_path,
            {"schema_version": SCHEMA_VERSION, "resolution_id": RESOLUTION_ID,
             "status": "failed", "failure_classification": classification,
             "returncode": returncode, "completed_at": utc_now()},
        )
    return load_json(result_path)


def run_benchmark(args):
    run_dir = Path(args.run_directory).resolve()
    manifest_path, manifest, profile = _load_prepared(run_dir, args.storage_root, "uae")
    if (run_dir / "checkpoint_summary.json").exists():
        raise Stage5RError("Stage 5R already has a final checkpoint")
    stage3.require_idle_gpu()
    queries = manifest["outputs"]["global_queries"]
    if file_identity(queries["path"]) != queries:
        raise Stage5RError("Frozen Stage 5R queries changed")
    plans = _frozen_plan_paths(run_dir, manifest)
    result_dir = run_dir / "worker_results"
    logs_dir = run_dir / "worker_logs"
    result_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    config = file_identity(args.config)
    checkpoint = file_identity(args.checkpoint)

    smallest = min(
        (path for key, path in plans.items() if key[2] == 100),
        key=lambda path: load_json(path)["padded_2mm_voxels"],
    )
    equivalence_path = result_dir / "bounded_restricted_equivalence.json"
    equivalence_signature = stage5.sha256_payload(
        {"resolution_id": RESOLUTION_ID, "kind": "equivalence",
         "plan": file_identity(smallest), "config": config, "checkpoint": checkpoint}
    )
    if equivalence_path.exists():
        equivalence = load_json(equivalence_path)
        if equivalence.get("worker_signature") != equivalence_signature:
            raise Stage5RError("Incompatible bounded-equivalence resume result")
    else:
        command = [
            sys.executable, "-m", "tools.quadra.organ_group_lattice_alignment",
            "_equivalence_worker", "--plan", str(smallest), "--config", str(args.config),
            "--checkpoint", str(args.checkpoint), "--result-path", str(equivalence_path),
            "--worker-signature", equivalence_signature,
        ]
        equivalence = _launch(
            command, equivalence_path, logs_dir / "bounded_restricted_equivalence.log",
            args.timeout_seconds,
        )

    results = []
    if equivalence.get("status") == "success" and equivalence.get("passed") is True:
        for group in stage3.GROUPS:
            for source_margin in MARGINS_MM:
                plan_paths = [
                    plans[(session, group, margin)]
                    for session in ("test", "retest") for margin in MARGINS_MM
                ]
                signature = _worker_signature(
                    group, source_margin,
                    [file_identity(path) for path in plan_paths], queries,
                    config, checkpoint,
                )
                result_path = result_dir / "{}-source-m{:03d}-factorial.json".format(
                    group, source_margin
                )
                if result_path.exists():
                    result = load_json(result_path)
                    if result.get("worker_signature") != signature:
                        raise Stage5RError("Incompatible resumable worker: {}".format(result_path))
                else:
                    stage3.require_idle_gpu()
                    command = [
                        sys.executable, "-m", "tools.quadra.organ_group_lattice_alignment",
                        "_worker", "--source-margin", str(source_margin),
                        "--test-100-plan", str(plans[("test", group, 100)]),
                        "--test-120-plan", str(plans[("test", group, 120)]),
                        "--retest-100-plan", str(plans[("retest", group, 100)]),
                        "--retest-120-plan", str(plans[("retest", group, 120)]),
                        "--queries", queries["path"], "--config", str(args.config),
                        "--checkpoint", str(args.checkpoint), "--result-path", str(result_path),
                        "--worker-signature", signature,
                    ]
                    result = _launch(
                        command, result_path,
                        logs_dir / "{}-source-m{:03d}-factorial.log".format(group, source_margin),
                        args.timeout_seconds,
                    )
                results.append(result)
                if result.get("status") != "success":
                    break
            if results and results[-1].get("status") != "success":
                break
    memory_rows = [
        {
            "group_name": item.get("group_name", ""),
            "source_margin_mm": item.get("source_margin_mm", ""),
            "status": item.get("status", ""),
            "failure_classification": item.get("failure_classification") or "",
            "measured_peak_mib": item.get("measured_peak_mib", ""),
            "process_gpu_peak_mib": item.get("process_gpu_peak_mib", ""),
            "wall_time_seconds": item.get("wall_time_seconds", ""),
        }
        for item in results
    ]
    write_csv(run_dir / "memory_profile.csv", memory_rows)
    manifest.update(
        {
            "status": "benchmarked", "benchmarked_at": utc_now(),
            "uae_profile": profile,
            "equivalence_result": file_identity(equivalence_path),
            "worker_result_count": len(results),
            "worker_results": [file_identity(path) for path in sorted(result_dir.glob("*-factorial.json"))],
            "outputs": dict(manifest["outputs"], memory_profile=file_identity(run_dir / "memory_profile.csv")),
        }
    )
    atomic_json(manifest_path, manifest)
    print("Stage 5R benchmark complete; select is profile-neutral", flush=True)
    print("Run directory: {}".format(run_dir), flush=True)
    return run_dir


def _distance(left, right):
    import numpy as np
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def comparison_rows(results, contrast):
    reference_id, candidate_id = contrast.split("_vs_")
    lookup = {}
    for result in results:
        for row in result.get("factorial_records", []):
            lookup[(row["query_id"], row["configuration"])] = row
    rows = []
    for query_id in sorted({key[0] for key in lookup}):
        reference = lookup.get((query_id, reference_id))
        candidate = lookup.get((query_id, candidate_id))
        if reference is None or candidate is None:
            continue
        paired = reference.get("status") == "success" and candidate.get("status") == "success"
        rows.append(
            {
                "contrast": contrast, "query_id": query_id,
                "point_id": reference["point_id"],
                "source_session": reference["source_session"],
                "target_session": reference["target_session"],
                "group_name": reference["group_name"], "mask_name": reference["mask_name"],
                "paired_success": paired,
                "forward_displacement_mm": _distance(reference["matched_physical_xyz"], candidate["matched_physical_xyz"]) if paired else None,
                "backward_displacement_mm": _distance(reference["returned_physical_xyz"], candidate["returned_physical_xyz"]) if paired else None,
                "cycle_error_A_mm": reference.get("cycle_error_mm"),
                "cycle_error_candidate_mm": candidate.get("cycle_error_mm"),
                "cycle_error_abs_delta_mm": abs(float(reference["cycle_error_mm"]) - float(candidate["cycle_error_mm"])) if paired else None,
                "score_forward_A": reference.get("score_forward"),
                "score_forward_candidate": candidate.get("score_forward"),
                "score_backward_A": reference.get("score_backward"),
                "score_backward_candidate": candidate.get("score_backward"),
                "match_outside_shared_domain": not bool(
                    reference.get("matched_inside_shared_domain", False)
                    and reference.get("returned_inside_shared_domain", False)
                    and candidate.get("matched_inside_shared_domain", False)
                    and candidate.get("returned_inside_shared_domain", False)
                ),
            }
        )
    return rows


def summarize_contrast(rows):
    import numpy as np

    summaries = []
    scopes = [("ALL_GROUPS", rows)] + [
        (group, [row for row in rows if row["group_name"] == group]) for group in stage3.GROUPS
    ]
    for label, subset in scopes:
        paired = [row for row in subset if row["paired_success"]]
        directional = np.asarray(
            [value for row in paired for value in (
                float(row["forward_displacement_mm"]), float(row["backward_displacement_mm"])
            )], dtype=np.float64,
        )
        cycle = np.asarray(
            [float(row["cycle_error_abs_delta_mm"]) for row in paired], dtype=np.float64
        )
        complete = len(subset) > 0 and len(paired) == len(subset)
        within = float(np.mean(directional <= 2.0)) if len(directional) else 0.0
        summary = {
            "contrast": rows[0]["contrast"] if rows else "", "scope": label,
            "queries": len(subset), "paired_successes": len(paired), "complete": complete,
            "directional_within_2mm_rate": within,
            "displacement_median_mm": stage5.percentile(directional, 50),
            "displacement_p95_mm": stage5.percentile(directional, 95),
            "cycle_delta_median_mm": stage5.percentile(cycle, 50),
            "cycle_delta_p95_mm": stage5.percentile(cycle, 95),
            "outside_shared_domain": sum(bool(row["match_outside_shared_domain"]) for row in subset),
        }
        summary["passed"] = bool(
            complete
            and within >= stage5.WITHIN_2MM_RATE_MIN
            and summary["displacement_median_mm"] <= stage5.DISPLACEMENT_MEDIAN_MAX_MM
            and summary["displacement_p95_mm"] <= stage5.DISPLACEMENT_P95_MAX_MM
            and summary["cycle_delta_median_mm"] <= stage5.CYCLE_DELTA_MEDIAN_MAX_MM
            and summary["cycle_delta_p95_mm"] <= stage5.CYCLE_DELTA_P95_MAX_MM
            and summary["outside_shared_domain"] == 0
        )
        summaries.append(summary)
    return summaries


def _legacy_stage5_summary(evidence):
    path = Path(evidence["run_directory"]) / "global_nn_summary.csv"
    if not path.is_file():
        raise Stage5RError("Legacy Stage 5 summary is missing")
    return read_csv(path), file_identity(path)


def forbidden_outputs(run_dir):
    return stage5.forbidden_outputs(run_dir)


def make_figures(run_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = Path(run_dir) / "figures"
    figures.mkdir(exist_ok=True)
    outputs = {}
    for contrast in CONTRASTS:
        subset = [row for row in rows if row["contrast"] == contrast and row["paired_success"]]
        displacement = [
            float(value) for row in subset
            for value in (row["forward_displacement_mm"], row["backward_displacement_mm"])
        ]
        cycle = [float(row["cycle_error_abs_delta_mm"]) for row in subset]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(displacement, bins=35)
        axes[0].set_xlabel("Correspondence displacement (mm)")
        axes[1].hist(cycle, bins=35)
        axes[1].set_xlabel("Absolute cycle-error difference (mm)")
        for axis in axes:
            axis.set_ylabel("Count")
            axis.grid(alpha=0.2)
        fig.suptitle(contrast.replace("_", " "))
        fig.tight_layout()
        path = figures / "{}.png".format(contrast.lower())
        fig.savefig(str(path), dpi=160)
        plt.close(fig)
        outputs[path.name] = file_identity(path)
    return outputs


def render_report(status, summaries, technical_failures, legacy_rows, descriptor_summary):
    lines = [
        "# Stage 5R global-lattice alignment", "", "## Outcome", "",
        "Stage 5R status: **{}**.".format(status), "",
        "Stage 5 remains immutable BLOCKED evidence. Stage 5R changes only grid construction and searches the same aligned-100 valid target domain in A-D.", "",
        "## Factorial sensitivity", "",
        "| Contrast | Scope | Queries | <=2 mm | Median displacement | P95 displacement | Median cycle delta | P95 cycle delta | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {contrast} | {scope} | {queries} | {directional_within_2mm_rate:.4f} | {displacement_median_mm:.4f} | {displacement_p95_mm:.4f} | {cycle_delta_median_mm:.4f} | {cycle_delta_p95_mm:.4f} | {passed} |".format(**row)
        )
    lines.extend(
        ["", "## Previous unaligned Stage 5", "",
         "The previous 100-120 mm summary is retained for descriptive comparison only; it is not recomputed or reclassified.", "",
         "| Scope | <=2 mm | Median displacement | P95 displacement | Median cycle delta | P95 cycle delta |",
         "|---|---:|---:|---:|---:|---:|"]
    )
    for row in legacy_rows:
        lines.append(
            "| {scope} | {directional_within_2mm_rate} | {displacement_median_mm} | {displacement_p95_mm} | {cycle_delta_median_mm} | {cycle_delta_p95_mm} |".format(**row)
        )
    lines.extend(
        ["", "## Descriptor cosine diagnostics", "",
         "These values are descriptive only and are not used as Stage 5R selection gates.", "",
         "Rows: **{rows}**; mean cosine: **{mean:.6f}**; minimum cosine: **{minimum:.6f}**.".format(**descriptor_summary), "",
         "## Fixed point", "", "No fixed-point matching was rerun. Its status remains **PROVISIONAL_CONCERN**.", "",
         "## Technical failures", "", ", ".join(technical_failures) if technical_failures else "None.", "",
         "## Interpretation", "",
         "PASS freezes only `organ_group_aligned_100mm_fp32_global_nn` for a later largest-pair pilot. It does not authorize fixed-point or cohort analysis."]
    )
    return "\n".join(lines)


def run_select(args):
    run_dir = Path(args.run_directory).resolve()
    # Deliberately profile-neutral: reporting must not trigger a pod image edit.
    manifest_path, manifest, _ = _load_prepared(run_dir, args.storage_root, profile=None)
    if manifest.get("status") != "benchmarked":
        raise Stage5RError("Stage 5R benchmark is incomplete")
    selected_path = run_dir / "selected_stage5r_workflow.json"
    checkpoint_path = run_dir / "checkpoint_summary.json"
    if selected_path.exists() or checkpoint_path.exists():
        raise Stage5RError("Stage 5R selection is immutable")
    evidence = validate_stage5_blocked_checkpoint(manifest["stage5_blocked_checkpoint"]["path"])
    equivalence = load_json(manifest["equivalence_result"]["path"])
    results = []
    technical_failures = []
    if file_identity(manifest["equivalence_result"]["path"]) != manifest["equivalence_result"]:
        technical_failures.append("equivalence_evidence_changed")
    if equivalence.get("status") != "success" or equivalence.get("passed") is not True:
        technical_failures.append("restricted_dense_chunked_equivalence")
    for item in manifest.get("worker_results", []):
        if file_identity(item["path"]) != item:
            technical_failures.append("worker_evidence_changed")
            continue
        results.append(load_json(item["path"]))
    if len(results) != 8:
        technical_failures.append("incomplete_worker_count")
    for item in results:
        if item.get("status") != "success":
            technical_failures.append("{}:{}".format(item.get("group_name", "unknown"), item.get("failure_classification", "failed")))
        if float(item.get("measured_peak_mib", float("inf"))) > VRAM_CEILING_MIB:
            technical_failures.append("memory_ceiling")
        if int(item.get("factorial_record_count", -1)) != 2 * int(item.get("query_count", -2)):
            technical_failures.append("factorial_denominator")
        if (
            item.get("model_dtype") != "torch.float32"
            or item.get("returned_embedding_dtype") != "torch.float16"
            or item.get("similarity_dtype") != "torch.float32"
        ):
            technical_failures.append("precision_contract")
        for extraction in item.get("extractions", {}).values():
            features = extraction.get("features", [])
            if len(features) != 3 or any(
                feature.get("dtype") != "torch.float16" or feature.get("finite") is not True
                for feature in features
            ):
                technical_failures.append("embedding_contract")
        for record in item.get("factorial_records", []):
            if (
                record.get("matched_inside_shared_domain") is not True
                or record.get("returned_inside_shared_domain") is not True
            ):
                technical_failures.append("admissible_domain_violation")
    retained = forbidden_outputs(run_dir)
    if retained:
        technical_failures.append("forbidden_full_volume_output_retained")

    factorial_records = [
        record for item in results for record in item.get("factorial_records", [])
    ]
    factorial_keys = {
        (record.get("query_id"), record.get("configuration"))
        for record in factorial_records
    }
    if len(factorial_records) != 4 * 2482 or len(factorial_keys) != 4 * 2482:
        technical_failures.append("factorial_global_denominator_or_duplicate")

    all_rows = []
    all_summaries = []
    for contrast in CONTRASTS:
        rows = comparison_rows(results, contrast)
        if len(rows) != 2482:
            technical_failures.append("{}:query_denominator".format(contrast))
        summaries = summarize_contrast(rows) if rows else []
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        write_csv(run_dir / "{}_sensitivity.csv".format(contrast), rows)
    if len(all_summaries) != len(CONTRASTS) * 5:
        technical_failures.append("summary_denominator")
    write_csv(run_dir / "factorial_sensitivity.csv", all_rows)
    write_csv(run_dir / "factorial_summary.csv", all_summaries)
    descriptor_rows = [row for item in results for row in item.get("descriptor_cosines", [])]
    if len(descriptor_rows) != 2482 or len({row.get("query_id") for row in descriptor_rows}) != 2482:
        technical_failures.append("descriptor_diagnostic_denominator")
    write_csv(run_dir / "descriptor_cosine_diagnostics.csv", descriptor_rows)
    descriptor_values = [float(row["mean_cosine"]) for row in descriptor_rows]
    descriptor_summary = {
        "rows": len(descriptor_values),
        "mean": sum(descriptor_values) / len(descriptor_values) if descriptor_values else float("nan"),
        "minimum": min(descriptor_values) if descriptor_values else float("nan"),
    }
    scientific_passed = bool(
        not technical_failures and all(row["passed"] for row in all_summaries)
    )
    status = "INCOMPLETE" if technical_failures else ("PASS" if scientific_passed else "BLOCKED")
    legacy_rows, legacy_identity = _legacy_stage5_summary(evidence)
    figures = make_figures(run_dir, all_rows)
    report_path = run_dir / "lattice_alignment_report.md"
    atomic_text(
        report_path,
        render_report(status, all_summaries, technical_failures, legacy_rows, descriptor_summary),
    )
    selection = {
        "schema_version": SCHEMA_VERSION, "resolution_id": RESOLUTION_ID,
        "status": status, "created_at": utc_now(),
        "selected_workflow": "organ_group_aligned_100mm_fp32_global_nn" if status == "PASS" else None,
        "precision": "fp32" if status == "PASS" else None,
        "matching_mode": "global_nn" if status == "PASS" else None,
        "fixed_point_status": "PROVISIONAL_CONCERN",
        "cohort_authorized": False,
        "largest_pair_pilot_authorized": status == "PASS",
        "technical_failures": sorted(set(technical_failures)),
        "limitations": [
            "Validation is limited to subject quadra_hc_030 and frozen Stage 5 queries.",
            "Organ-group crops impose an anatomical search prior.",
            "Fixed-point was not rerun and remains PROVISIONAL_CONCERN.",
            "No cohort analysis is authorized.",
        ],
    }
    atomic_json(selected_path, selection, refuse=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "stage": 5, "substage": "R",
        "status": status, "created_at": utc_now(), "resolution_id": RESOLUTION_ID,
        "selected_workflow": file_identity(selected_path),
        "stage5_blocked_checkpoint": manifest["stage5_blocked_checkpoint"],
        "gates": {
            "stage5_blocked_evidence_preserved": True,
            "all_eight_workers_and_four_factorial_configurations_completed_for_2482_queries": len(results) == 8 and not any("denominator" in item for item in technical_failures),
            "restricted_dense_chunked_equivalence_passed": equivalence.get("passed") is True,
            "all_three_contrasts_passed_pooled_and_per_group": scientific_passed,
            "zero_matches_outside_shared_admissible_domain": all(not row["match_outside_shared_domain"] for row in all_rows),
            "memory_headroom_passed": "memory_ceiling" not in technical_failures,
            "fixed_point_rerun": False,
            "fixed_point_status_preserved": "PROVISIONAL_CONCERN",
            "embeddings_or_prepared_volumes_retained": bool(retained),
            "cohort_authorized": False,
        },
        "next_stage": "largest_pair_aligned_global_nn_pilot" if status == "PASS" else (
            "reconsider_broader_context" if status == "BLOCKED" else "resolve_technical_failure"
        ),
        "outputs": {
            "factorial_sensitivity": file_identity(run_dir / "factorial_sensitivity.csv"),
            "factorial_summary": file_identity(run_dir / "factorial_summary.csv"),
            "descriptor_cosines": file_identity(run_dir / "descriptor_cosine_diagnostics.csv"),
            "memory_profile": manifest["outputs"]["memory_profile"],
            "legacy_stage5_summary": legacy_identity,
            "report": file_identity(report_path),
            "figures": figures,
        },
    }
    atomic_json(checkpoint_path, checkpoint, refuse=True)
    manifest.update(
        {"status": status.lower(), "completed_at": utc_now(),
         "selection": file_identity(selected_path), "checkpoint": file_identity(checkpoint_path)}
    )
    atomic_json(manifest_path, manifest)
    print("Stage 5R {}".format(status), flush=True)
    print("Selected workflow: {}".format(selection["selected_workflow"]), flush=True)
    print("Checkpoint: {}".format(checkpoint_path), flush=True)
    return status == "PASS"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    prepare = sub.add_parser("prepare", help="Freeze Stage 5 lineage and build aligned 100/120 mm plans.")
    prepare.add_argument("--stage5-checkpoint", required=True)
    prepare.add_argument("--storage-root", default="/workspace/quadra")
    prepare.add_argument("--repository-root", default=str(PROJECT_ROOT))
    prepare.add_argument("--output-root", default=None)
    prepare.add_argument("--run-id", default=None)
    prepare.add_argument("--resume-run-directory", default=None)
    benchmark = sub.add_parser("benchmark", help="Run restricted equivalence and four group factorial workers.")
    benchmark.add_argument("--run-directory", required=True)
    benchmark.add_argument("--storage-root", default="/workspace/quadra")
    benchmark.add_argument("--config", default="configs/samv2/samv2_NIHLN.py")
    benchmark.add_argument("--checkpoint", default="checkpoints/SAMv2_iter_20000.pth")
    benchmark.add_argument("--timeout-seconds", type=int, default=WORKER_TIMEOUT_SECONDS)
    select = sub.add_parser("select", help="Apply all three contrast gates without requiring an environment profile.")
    select.add_argument("--run-directory", required=True)
    select.add_argument("--storage-root", default="/workspace/quadra")
    worker = sub.add_parser("_worker")
    worker.add_argument("--source-margin", required=True, type=int)
    worker.add_argument("--test-100-plan", required=True)
    worker.add_argument("--test-120-plan", required=True)
    worker.add_argument("--retest-100-plan", required=True)
    worker.add_argument("--retest-120-plan", required=True)
    worker.add_argument("--queries", required=True)
    worker.add_argument("--config", required=True)
    worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--result-path", required=True)
    worker.add_argument("--worker-signature", required=True)
    equivalence = sub.add_parser("_equivalence_worker")
    equivalence.add_argument("--plan", required=True)
    equivalence.add_argument("--config", required=True)
    equivalence.add_argument("--checkpoint", required=True)
    equivalence.add_argument("--result-path", required=True)
    equivalence.add_argument("--worker-signature", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "prepare":
            run_prepare(args)
        elif args.command == "benchmark":
            run_benchmark(args)
        elif args.command == "select":
            return 0 if run_select(args) else 3
        elif args.command == "_worker":
            return run_worker(args)
        elif args.command == "_equivalence_worker":
            return run_equivalence_worker(args)
    except (Stage5RError, stage5.Stage5Error) as exc:
        parser.error("Stage 5R failed: {}".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
