#!/usr/bin/env python3
"""Audit and freeze conservative body-envelope crops for Quadra UAE-S.

The ``audit`` phase reads the 56 accepted Stage-5 CT/mask sets, but it never
writes cropped images, resamples a volume, loads UAE-S, or launches CUDA work.
The ``select`` phase freezes one eligible candidate after human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra import optimization_baseline as baseline  # noqa: E402
from tools.quadra.environment import canonical_layout  # noqa: E402


SCHEMA_VERSION = 1
AUDIT_ID = "quadra-body-envelope-v1"
EXPECTED_BASELINE_PATH = Path(
    "/workspace/quadra/runs/memory_optimization/"
    "stage0-20260731T085944Z/baseline_manifest.json"
)
EXPECTED_BASELINE_SHA256 = (
    "c331396b41a5cd03c039700b37c79e54fc920944c3e8ff304c2b0175c94d4a47"
)
COHORT_MANIFEST_NAME = "totalsegmentator_cohort.json"
BODY_THRESHOLD_HU = -800.0
MIN_COMPONENT_VOLUME_ML = 10.0
AXIS_POLICIES = ("xy", "xyz")
MARGINS_MM = (0.0, 10.0, 20.0, 30.0, 40.0, 60.0)
TARGET_SPACING_XYZ_MM = (2.0, 2.0, 2.0)
MODEL_STRIDE_XYZ = (16, 16, 4)
MIN_ARTIFICIAL_CLEARANCE_MM = 20.0
EXPECTED_SCANS = 56
EXPECTED_SUBJECTS = 28
EXPECTED_MASKS = 2208
EXPECTED_MASKS_PER_SCAN_BY_SEX = {"F": 39, "M": 40}


class BodyEnvelopeAuditError(RuntimeError):
    """Raised when Stage 1 cannot produce defensible audit evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("stage1-audit-%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise BodyEnvelopeAuditError(f"Required file is missing: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def _is_within(path: Path, root: Path) -> bool:
    resolved = Path(path).resolve()
    resolved_root = Path(root).resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def atomic_replace_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_create_json(path: Path, value: Any) -> None:
    path = Path(path)
    if path.exists():
        raise BodyEnvelopeAuditError(f"Refusing to overwrite existing file: {path}")
    atomic_replace_json(path, value)


def atomic_replace_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def emit(message: str, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {message}\n")


def load_json(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BodyEnvelopeAuditError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BodyEnvelopeAuditError(f"Expected a JSON object: {path}")
    return value


def verify_baseline_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    expected_path = EXPECTED_BASELINE_PATH.resolve()
    if resolved != expected_path:
        raise BodyEnvelopeAuditError(
            "Stage 1 requires the accepted Stage 0 manifest at "
            f"{expected_path}; found {resolved}"
        )
    identity = file_identity(resolved)
    if identity["sha256"] != EXPECTED_BASELINE_SHA256:
        raise BodyEnvelopeAuditError(
            "Stage 1 requires the accepted Stage 0 manifest: "
            f"{EXPECTED_BASELINE_SHA256}; found {identity['sha256']}"
        )
    return identity


def current_repository_record(repository_root: Path) -> dict[str, Any]:
    record = baseline.inspect_repository(repository_root)
    return {
        "path": record["path"],
        "branch": record["branch"],
        "execution_commit": record["execution_commit"],
        "clean": record["clean"],
    }


def audit_settings() -> dict[str, Any]:
    return {
        "body_threshold_hu": BODY_THRESHOLD_HU,
        "component_connectivity": 26,
        "minimum_component_volume_ml": MIN_COMPONENT_VOLUME_ML,
        "component_policy": "retain_every_component_at_or_above_threshold",
        "aggressive_bed_removal": False,
        "axis_policies": list(AXIS_POLICIES),
        "margins_mm": list(MARGINS_MM),
        "target_spacing_xyz_mm": list(TARGET_SPACING_XYZ_MM),
        "model_stride_xyz": list(MODEL_STRIDE_XYZ),
        "padding_policy": "symmetric_lower_floor_upper_remainder",
        "padding_value_hu": -1024.0,
        "minimum_artificial_mask_clearance_mm": MIN_ARTIFICIAL_CLEARANCE_MM,
        "crop_bounds": "raw_itk_voxel_half_open",
    }


def candidate_id(axis_policy: str, margin_mm: float) -> str:
    if axis_policy not in AXIS_POLICIES:
        raise BodyEnvelopeAuditError(f"Unsupported axis policy: {axis_policy}")
    if float(margin_mm) not in MARGINS_MM:
        raise BodyEnvelopeAuditError(f"Unsupported margin: {margin_mm}")
    return f"{axis_policy}_m{int(round(float(margin_mm))):03d}"


def half_open_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask.ndim != 3:
        raise BodyEnvelopeAuditError(f"Expected a 3D mask, found shape {mask.shape}")
    occupied_axes = [
        np.flatnonzero(
            np.any(
                mask,
                axis=tuple(other for other in range(3) if other != axis),
            )
        )
        for axis in range(3)
    ]
    if any(not occupied.size for occupied in occupied_axes):
        raise BodyEnvelopeAuditError("Cannot calculate bounds for an empty mask")
    start = np.array([occupied[0] for occupied in occupied_axes], dtype=np.int64)
    end = np.array([occupied[-1] + 1 for occupied in occupied_axes], dtype=np.int64)
    return start, end


def build_conservative_foreground(
    volume_xyz: np.ndarray,
    spacing_xyz_mm: Sequence[float],
    *,
    threshold_hu: float = BODY_THRESHOLD_HU,
    minimum_component_volume_ml: float = MIN_COMPONENT_VOLUME_ML,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    if volume_xyz.ndim != 3:
        raise BodyEnvelopeAuditError(f"CT is not 3D: {volume_xyz.shape}")
    if not np.isfinite(volume_xyz).all():
        raise BodyEnvelopeAuditError("CT contains non-finite voxel values")
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise BodyEnvelopeAuditError(f"Invalid CT spacing: {spacing.tolist()}")
    voxel_volume_mm3 = float(np.prod(spacing))
    minimum_voxels = max(
        1,
        int(math.ceil(minimum_component_volume_ml * 1000.0 / voxel_volume_mm3)),
    )
    thresholded = np.asarray(volume_xyz > threshold_hu, dtype=bool)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    labels, component_count = ndimage.label(thresholded, structure=structure)
    counts = np.bincount(labels.reshape(-1))
    keep = counts >= minimum_voxels
    if keep.size:
        keep[0] = False
    foreground = keep[labels]
    retained_labels = np.flatnonzero(keep)
    if not retained_labels.size:
        raise BodyEnvelopeAuditError(
            "No conservative foreground component survived the 10 mL filter"
        )
    return foreground, {
        "threshold_hu": float(threshold_hu),
        "voxel_volume_mm3": voxel_volume_mm3,
        "minimum_component_voxels": minimum_voxels,
        "components_detected": int(component_count),
        "components_retained": int(retained_labels.size),
        "retained_foreground_voxels": int(np.count_nonzero(foreground)),
        "retained_component_volumes_ml": sorted(
            (float(counts[label] * voxel_volume_mm3 / 1000.0) for label in retained_labels),
            reverse=True,
        ),
    }


def expand_bounds(
    base_start_xyz: Sequence[int],
    base_end_xyz: Sequence[int],
    native_shape_xyz: Sequence[int],
    spacing_xyz_mm: Sequence[float],
    *,
    axis_policy: str,
    margin_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(base_start_xyz, dtype=np.int64).copy()
    end = np.asarray(base_end_xyz, dtype=np.int64).copy()
    shape = np.asarray(native_shape_xyz, dtype=np.int64)
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    if axis_policy not in AXIS_POLICIES:
        raise BodyEnvelopeAuditError(f"Unsupported axis policy: {axis_policy}")
    cropped_axes = (0, 1) if axis_policy == "xy" else (0, 1, 2)
    margin_voxels = np.ceil(float(margin_mm) / spacing).astype(np.int64)
    for axis in range(3):
        if axis in cropped_axes:
            start[axis] = max(0, int(start[axis] - margin_voxels[axis]))
            end[axis] = min(int(shape[axis]), int(end[axis] + margin_voxels[axis]))
        else:
            start[axis] = 0
            end[axis] = shape[axis]
    if np.any(start < 0) or np.any(end > shape) or np.any(end <= start):
        raise BodyEnvelopeAuditError(
            f"Invalid crop bounds: start={start.tolist()}, end={end.tolist()}"
        )
    return start, end


def torchio_target_shape(
    native_shape_xyz: Sequence[int],
    native_spacing_xyz_mm: Sequence[float],
    target_spacing_xyz_mm: Sequence[float] = TARGET_SPACING_XYZ_MM,
) -> np.ndarray:
    shape = np.asarray(native_shape_xyz, dtype=float)
    spacing = np.asarray(native_spacing_xyz_mm, dtype=float)
    target = np.asarray(target_spacing_xyz_mm, dtype=float)
    if shape.shape != (3,) or spacing.shape != (3,) or target.shape != (3,):
        raise BodyEnvelopeAuditError("Shapes and spacings must contain exactly three values")
    if np.any(shape <= 0) or np.any(spacing <= 0) or np.any(target <= 0):
        raise BodyEnvelopeAuditError("Shapes and spacings must be positive")
    return np.ceil(shape * spacing / target - 1e-9).astype(np.int64)


def symmetric_stride_padding(
    shape_xyz: Sequence[int],
    stride_xyz: Sequence[int] = MODEL_STRIDE_XYZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.asarray(shape_xyz, dtype=np.int64)
    stride = np.asarray(stride_xyz, dtype=np.int64)
    if np.any(shape <= 0) or np.any(stride <= 0):
        raise BodyEnvelopeAuditError("Shape and stride must be positive")
    total = (-shape) % stride
    lower = total // 2
    upper = total - lower
    padded = shape + total
    return lower, upper, padded


def crop_affine(raw_affine: np.ndarray, crop_start_xyz: Sequence[int]) -> np.ndarray:
    affine = np.asarray(raw_affine, dtype=float)
    start = np.asarray(crop_start_xyz, dtype=float)
    result = affine.copy()
    result[:3, 3] = (
        affine @ np.array([start[0], start[1], start[2], 1.0])
    )[:3]
    return result


def resampled_and_padded_affines(
    native_crop_affine: np.ndarray,
    native_crop_shape_xyz: Sequence[int],
    target_shape_xyz: Sequence[int],
    lower_padding_xyz: Sequence[int],
    target_spacing_xyz_mm: Sequence[float] = TARGET_SPACING_XYZ_MM,
) -> tuple[np.ndarray, np.ndarray]:
    affine = np.asarray(native_crop_affine, dtype=float)
    native_shape = np.asarray(native_crop_shape_xyz, dtype=np.int64)
    target_shape = np.asarray(target_shape_xyz, dtype=np.int64)
    target_spacing = np.asarray(target_spacing_xyz_mm, dtype=float)
    if affine.shape != (4, 4):
        raise BodyEnvelopeAuditError("The native crop affine must be 4 by 4")
    if native_shape.shape != (3,) or target_shape.shape != (3,):
        raise BodyEnvelopeAuditError("Native and target shapes must have three values")
    if np.any(native_shape <= 0) or np.any(target_shape <= 0):
        raise BodyEnvelopeAuditError("Native and target shapes must be positive")
    native_spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(native_spacing <= 0) or np.any(target_spacing <= 0):
        raise BodyEnvelopeAuditError("Native and target spacings must be positive")

    # This reproduces TorchIO Resample.get_reference_image for a spacing target:
    # new_size = ceil(old_size * old_spacing / new_spacing), and the new origin
    # is the old image's physical point at continuous index
    # 0.5 * (new_spacing / old_spacing - 1).  It is deliberately not
    # nibabel.rescale_affine, which preserves a different centre voxel when an
    # even-sized dimension is resampled.
    expected_target_shape = torchio_target_shape(
        native_shape,
        native_spacing,
        target_spacing,
    )
    if not np.array_equal(target_shape, expected_target_shape):
        raise BodyEnvelopeAuditError(
            "Target shape does not match the TorchIO spacing convention"
        )
    direction = affine[:3, :3] / native_spacing
    resampled = np.eye(4, dtype=float)
    resampled[:3, :3] = direction * target_spacing
    resampled[:3, 3] = affine[:3, 3] + direction @ (
        0.5 * (target_spacing - native_spacing)
    )
    lower = np.asarray(lower_padding_xyz, dtype=float)
    padded = resampled.copy()
    padded[:3, 3] = (
        resampled @ np.array([-lower[0], -lower[1], -lower[2], 1.0])
    )[:3]
    return resampled, padded


def coordinate_roundtrip(
    raw_affine: np.ndarray,
    crop_affine_value: np.ndarray,
    model_affine: np.ndarray,
    points_raw_xyz: Iterable[Sequence[float]],
) -> dict[str, Any]:
    raw_affine = np.asarray(raw_affine, dtype=float)
    crop_affine_value = np.asarray(crop_affine_value, dtype=float)
    model_affine = np.asarray(model_affine, dtype=float)
    raw_to_crop = np.linalg.inv(crop_affine_value) @ raw_affine
    crop_to_raw = np.linalg.inv(raw_to_crop)
    crop_to_model = np.linalg.inv(model_affine) @ crop_affine_value
    model_to_crop = np.linalg.inv(crop_to_model)
    raw_to_model = crop_to_model @ raw_to_crop
    model_to_raw = np.linalg.inv(raw_to_model)
    max_voxel_error = 0.0
    max_physical_error_mm = 0.0
    count = 0
    for point in points_raw_xyz:
        raw = np.array([float(point[0]), float(point[1]), float(point[2]), 1.0])
        crop = raw_to_crop @ raw
        model = crop_to_model @ crop
        recovered_crop = model_to_crop @ model
        recovered = crop_to_raw @ recovered_crop
        max_voxel_error = max(max_voxel_error, float(np.max(np.abs(recovered[:3] - raw[:3]))))
        physical_before = raw_affine @ raw
        physical_after = raw_affine @ recovered
        max_physical_error_mm = max(
            max_physical_error_mm,
            float(np.linalg.norm(physical_after[:3] - physical_before[:3])),
        )
        count += 1
    return {
        "points_checked": count,
        "max_raw_voxel_roundtrip_error": max_voxel_error,
        "max_physical_roundtrip_error_mm": max_physical_error_mm,
        "raw_to_crop_continuous_affine": raw_to_crop.tolist(),
        "crop_to_raw_continuous_affine": crop_to_raw.tolist(),
        "crop_to_model_continuous_affine": crop_to_model.tolist(),
        "model_to_crop_continuous_affine": model_to_crop.tolist(),
        "raw_to_model_continuous_affine": raw_to_model.tolist(),
        "model_to_raw_continuous_affine": model_to_raw.tolist(),
        "passed": max_voxel_error <= 1e-6 and max_physical_error_mm <= 1e-5,
    }


def mask_candidate_metrics(
    mask: np.ndarray,
    mask_start_xyz: Sequence[int],
    mask_end_xyz: Sequence[int],
    crop_start_xyz: Sequence[int],
    crop_end_xyz: Sequence[int],
    native_shape_xyz: Sequence[int],
    spacing_xyz_mm: Sequence[float],
    *,
    mask_voxel_count: int | None = None,
) -> dict[str, Any]:
    mask_start = np.asarray(mask_start_xyz, dtype=np.int64)
    mask_end = np.asarray(mask_end_xyz, dtype=np.int64)
    crop_start = np.asarray(crop_start_xyz, dtype=np.int64)
    crop_end = np.asarray(crop_end_xyz, dtype=np.int64)
    shape = np.asarray(native_shape_xyz, dtype=np.int64)
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    total = (
        int(mask_voxel_count)
        if mask_voxel_count is not None
        else int(np.count_nonzero(mask))
    )
    bounding_box_inside = bool(
        np.all(mask_start >= crop_start) and np.all(mask_end <= crop_end)
    )
    if bounding_box_inside:
        inside = total
    else:
        inside = int(
            np.count_nonzero(
                mask[
                    crop_start[0] : crop_end[0],
                    crop_start[1] : crop_end[1],
                    crop_start[2] : crop_end[2],
                ]
            )
        )
    artificial = []
    clearance_values = []
    for axis, label in enumerate(("x", "y", "z")):
        lower_artificial = bool(crop_start[axis] > 0)
        upper_artificial = bool(crop_end[axis] < shape[axis])
        lower_clearance = float((mask_start[axis] - crop_start[axis]) * spacing[axis])
        upper_clearance = float((crop_end[axis] - mask_end[axis]) * spacing[axis])
        artificial.append(
            {
                "axis": label,
                "lower_is_artificial": lower_artificial,
                "upper_is_artificial": upper_artificial,
                "lower_clearance_mm": lower_clearance if lower_artificial else None,
                "upper_clearance_mm": upper_clearance if upper_artificial else None,
            }
        )
        if lower_artificial:
            clearance_values.append(lower_clearance)
        if upper_artificial:
            clearance_values.append(upper_clearance)
    return {
        "mask_voxels": total,
        "inside_crop_voxels": inside,
        "outside_crop_voxels": total - inside,
        "minimum_artificial_clearance_mm": (
            min(clearance_values) if clearance_values else None
        ),
        "boundary_clearances": artificial,
    }


def _geometry_matches(reference, candidate, *, atol: float = 1e-5) -> bool:
    return tuple(reference.shape[:3]) == tuple(candidate.shape[:3]) and np.allclose(
        reference.affine, candidate.affine, rtol=0.0, atol=atol
    )


def _scan_paths(
    scan: dict[str, Any], ct_root: Path, mask_root: Path
) -> tuple[Path, Path]:
    subject = str(scan["subject_id"]).lower()
    session = str(scan["session"]).lower()
    if session not in {"test", "retest"}:
        raise BodyEnvelopeAuditError(f"Unexpected session: {session}")
    number = subject.rsplit("_", 1)[-1]
    ct_path = ct_root / f"QUADRA_HC_{number}" / f"{session}_CT-AC.nii.gz"
    masks = mask_root / subject / session / "masks"
    return ct_path, masks


def load_cohort(
    cohort_manifest_path: Path, ct_root: Path, mask_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(cohort_manifest_path)
    scans = manifest.get("scans")
    if not isinstance(scans, list) or len(scans) != EXPECTED_SCANS:
        raise BodyEnvelopeAuditError(
            f"Expected {EXPECTED_SCANS} cohort scans, found {0 if not isinstance(scans, list) else len(scans)}"
        )
    normalized = []
    seen = set()
    mask_count = 0
    for scan in scans:
        subject = str(scan.get("subject_id", "")).lower()
        session = str(scan.get("session", "")).lower()
        sex = str(scan.get("sex", "")).upper()
        key = f"{subject}-{session}"
        if key in seen:
            raise BodyEnvelopeAuditError(f"Duplicate cohort scan: {key}")
        seen.add(key)
        expected_masks = list(scan.get("expected_masks", []))
        if sex not in EXPECTED_MASKS_PER_SCAN_BY_SEX:
            raise BodyEnvelopeAuditError(f"Missing or invalid sex for {key}: {sex!r}")
        expected_mask_count = EXPECTED_MASKS_PER_SCAN_BY_SEX[sex]
        if len(expected_masks) != expected_mask_count:
            raise BodyEnvelopeAuditError(
                f"Expected {expected_mask_count} masks for {sex} scan {key}, "
                f"found {len(expected_masks)}"
            )
        ct_path, masks = _scan_paths(scan, ct_root, mask_root)
        if not ct_path.is_file():
            raise BodyEnvelopeAuditError(f"Missing cohort CT: {ct_path}")
        if not masks.is_dir():
            raise BodyEnvelopeAuditError(f"Missing cohort mask directory: {masks}")
        missing = [name for name in expected_masks if not (masks / f"{name}.nii.gz").is_file()]
        if missing:
            raise BodyEnvelopeAuditError(f"Missing masks for {key}: {missing}")
        normalized.append(
            {
                "key": key,
                "subject_id": subject,
                "session": session,
                "sex": sex,
                "expected_masks": expected_masks,
                "expected_input_sha256": scan.get("input_sha256"),
                "ct_path": str(ct_path.resolve()),
                "mask_directory": str(masks.resolve()),
            }
        )
        mask_count += len(expected_masks)
    subjects = {scan["subject_id"] for scan in normalized}
    if len(subjects) != EXPECTED_SUBJECTS or mask_count != EXPECTED_MASKS:
        raise BodyEnvelopeAuditError(
            f"Unexpected cohort denominators: subjects={len(subjects)}, masks={mask_count}"
        )
    normalized.sort(key=lambda item: (item["subject_id"], item["session"]))
    return manifest, normalized


def _corners(start: np.ndarray, end: np.ndarray) -> list[list[float]]:
    high = end - 1
    return [
        [float(x), float(y), float(z)]
        for x in (start[0], high[0])
        for y in (start[1], high[1])
        for z in (start[2], high[2])
    ]


def _scan_signature(scan: dict[str, Any], run_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "baseline_sha256": run_contract["baseline_manifest"]["sha256"],
        "cohort_manifest_sha256": run_contract["cohort_manifest"]["sha256"],
        "settings": run_contract["settings"],
        "scan": {
            "subject_id": scan["subject_id"],
            "session": scan["session"],
            "sex": scan["sex"],
            "expected_input_sha256": scan["expected_input_sha256"],
            "expected_masks": scan["expected_masks"],
        },
    }


def process_scan(
    scan: dict[str, Any],
    run_contract: dict[str, Any],
    preview_path: Path | None = None,
) -> dict[str, Any]:
    import nibabel as nib
    from scipy import ndimage

    ct_path = Path(scan["ct_path"])
    observed_ct_hash = sha256_file(ct_path)
    if observed_ct_hash != scan["expected_input_sha256"]:
        raise BodyEnvelopeAuditError(
            f"CT checksum mismatch for {scan['key']}: {observed_ct_hash}"
        )
    ct_image = nib.load(str(ct_path))
    if len(ct_image.shape) != 3:
        raise BodyEnvelopeAuditError(f"CT is not 3D: {ct_path}")
    spatial_unit, _ = ct_image.header.get_xyzt_units()
    if spatial_unit not in {"mm", None, "unknown"}:
        raise BodyEnvelopeAuditError(f"CT spatial unit is not millimetres: {spatial_unit}")
    ct = np.asanyarray(ct_image.dataobj)
    native_shape = np.asarray(ct.shape, dtype=np.int64)
    spacing = np.asarray(nib.affines.voxel_sizes(ct_image.affine), dtype=float)
    foreground, component_stats = build_conservative_foreground(ct, spacing)
    body_start, body_end = half_open_bounds(foreground)
    uncropped_target_shape = torchio_target_shape(native_shape, spacing)
    _, _, uncropped_padded_shape = symmetric_stride_padding(uncropped_target_shape)
    uncropped_padded_voxels = int(
        np.prod(uncropped_padded_shape, dtype=np.int64)
    )

    # Candidate geometry depends only on the CT.  Build it before opening masks,
    # then stream one mask at a time across every candidate.  This prevents 39
    # full-resolution boolean masks from being duplicated in RAM.
    candidates = []
    for axis_policy in AXIS_POLICIES:
        for margin_mm in MARGINS_MM:
            crop_start, crop_end = expand_bounds(
                body_start,
                body_end,
                native_shape,
                spacing,
                axis_policy=axis_policy,
                margin_mm=margin_mm,
            )
            crop_shape = crop_end - crop_start
            target_shape = torchio_target_shape(crop_shape, spacing)
            lower_pad, upper_pad, padded_shape = symmetric_stride_padding(target_shape)
            native_crop_affine = crop_affine(ct_image.affine, crop_start)
            resampled_affine, padded_affine = resampled_and_padded_affines(
                native_crop_affine,
                crop_shape,
                target_shape,
                lower_pad,
            )
            native_voxels = int(np.prod(native_shape, dtype=np.int64))
            crop_voxels = int(np.prod(crop_shape, dtype=np.int64))
            padded_voxels = int(np.prod(padded_shape, dtype=np.int64))
            native_retained_fraction = crop_voxels / native_voxels
            padded_retained_fraction = padded_voxels / uncropped_padded_voxels
            stride_compatible = bool(
                np.all(padded_shape % np.asarray(MODEL_STRIDE_XYZ, dtype=np.int64) == 0)
            )
            candidates.append(
                {
                    "candidate_id": candidate_id(axis_policy, margin_mm),
                    "axis_policy": axis_policy,
                    "margin_mm": float(margin_mm),
                    "crop_start_xyz": crop_start.tolist(),
                    "crop_end_xyz": crop_end.tolist(),
                    "crop_shape_xyz": crop_shape.tolist(),
                    "target_shape_xyz": target_shape.tolist(),
                    "padding_lower_xyz": lower_pad.tolist(),
                    "padding_upper_xyz": upper_pad.tolist(),
                    "padded_shape_xyz": padded_shape.tolist(),
                    "model_tensor_shape_zyx": padded_shape[::-1].tolist(),
                    "native_voxels": native_voxels,
                    "native_crop_voxels": crop_voxels,
                    "padded_2mm_voxels": padded_voxels,
                    "uncropped_target_shape_xyz": uncropped_target_shape.tolist(),
                    "uncropped_padded_shape_xyz": uncropped_padded_shape.tolist(),
                    "uncropped_padded_2mm_voxels": uncropped_padded_voxels,
                    "native_voxel_retained_fraction": native_retained_fraction,
                    "native_voxel_reduction_fraction": 1.0 - native_retained_fraction,
                    "padded_2mm_voxel_retained_fraction": padded_retained_fraction,
                    "padded_2mm_voxel_reduction_fraction": 1.0
                    - padded_retained_fraction,
                    "clipped_mask_voxels": 0,
                    "clipped_mask_names": [],
                    "minimum_artificial_mask_clearance_mm": None,
                    "artificial_boundaries": {
                        "lower_xyz": (crop_start > 0).tolist(),
                        "upper_xyz": (crop_end < native_shape).tolist(),
                    },
                    "stride_compatible": stride_compatible,
                    "coordinate_roundtrip": None,
                    "native_crop_affine": native_crop_affine.tolist(),
                    "resampled_2mm_affine": resampled_affine.tolist(),
                    "padded_2mm_affine": padded_affine.tolist(),
                    "masks": [],
                    "eligible_for_scan": False,
                }
            )

    mask_directory = Path(scan["mask_directory"])
    centroids = []
    for name in scan["expected_masks"]:
        path = mask_directory / f"{name}.nii.gz"
        image = nib.load(str(path))
        if not _geometry_matches(ct_image, image):
            raise BodyEnvelopeAuditError(f"CT-mask geometry mismatch: {path}")
        data = np.asanyarray(image.dataobj)
        if data.ndim != 3 or not np.isfinite(data).all():
            raise BodyEnvelopeAuditError(f"Invalid mask data: {path}")
        nonzero = data != 0
        if not np.any(nonzero):
            raise BodyEnvelopeAuditError(f"Expected mask is empty: {path}")
        start, end = half_open_bounds(nonzero)
        centroid = [float(value) for value in ndimage.center_of_mass(nonzero)]
        centroids.append(centroid)
        voxel_count = int(np.count_nonzero(nonzero))
        for candidate in candidates:
            metrics = mask_candidate_metrics(
                nonzero,
                start,
                end,
                candidate["crop_start_xyz"],
                candidate["crop_end_xyz"],
                native_shape,
                spacing,
                mask_voxel_count=voxel_count,
            )
            outside = int(metrics["outside_crop_voxels"])
            if outside:
                candidate["clipped_mask_names"].append(name)
                candidate["clipped_mask_voxels"] += outside
            clearance = metrics["minimum_artificial_clearance_mm"]
            current_clearance = candidate["minimum_artificial_mask_clearance_mm"]
            if clearance is not None and (
                current_clearance is None or float(clearance) < float(current_clearance)
            ):
                candidate["minimum_artificial_mask_clearance_mm"] = float(clearance)
            candidate["masks"].append(
                {
                    "mask_name": name,
                    "mask_voxels": voxel_count,
                    "mask_start_xyz": start.tolist(),
                    "mask_end_xyz": end.tolist(),
                    "centroid_raw_xyz": centroid,
                    **metrics,
                }
            )
        del data, nonzero

    for candidate in candidates:
        crop_start = np.asarray(candidate["crop_start_xyz"], dtype=np.int64)
        crop_end = np.asarray(candidate["crop_end_xyz"], dtype=np.int64)
        roundtrip = coordinate_roundtrip(
            ct_image.affine,
            np.asarray(candidate["native_crop_affine"], dtype=float),
            np.asarray(candidate["padded_2mm_affine"], dtype=float),
            _corners(crop_start, crop_end) + centroids,
        )
        candidate["coordinate_roundtrip"] = roundtrip
        minimum_clearance = candidate["minimum_artificial_mask_clearance_mm"]
        candidate["eligible_for_scan"] = bool(
            candidate["clipped_mask_voxels"] == 0
            and roundtrip["passed"]
            and candidate["stride_compatible"]
            and (
                minimum_clearance is None
                or float(minimum_clearance) + 1e-6
                >= MIN_ARTIFICIAL_CLEARANCE_MM
            )
        )

    if preview_path is not None:
        make_preview(ct, body_start, body_end, preview_path, scan["key"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "scan_signature": _scan_signature(scan, run_contract),
        "subject_id": scan["subject_id"],
        "session": scan["session"],
        "sex": scan["sex"],
        "scan_key": scan["key"],
        "ct": {
            "path": str(ct_path.resolve()),
            "bytes": int(ct_path.stat().st_size),
            "sha256": observed_ct_hash,
            "native_shape_xyz": native_shape.tolist(),
            "spacing_xyz_mm": spacing.tolist(),
            "affine": np.asarray(ct_image.affine).tolist(),
            "spatial_unit": spatial_unit,
        },
        "body_envelope": {
            "start_xyz": body_start.tolist(),
            "end_xyz": body_end.tolist(),
            "shape_xyz": (body_end - body_start).tolist(),
            **component_stats,
        },
        "mask_count": len(centroids),
        "mask_geometry_verified": True,
        "candidates": candidates,
    }
    if preview_path is not None:
        result["preview"] = file_identity(preview_path)
    return result


def make_preview(
    ct_xyz: np.ndarray,
    start_xyz: Sequence[int],
    end_xyz: Sequence[int],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    start = np.asarray(start_xyz, dtype=int)
    end = np.asarray(end_xyz, dtype=int)
    centers = ((start + end - 1) // 2).astype(int)
    arrays = [
        ct_xyz[:, :, centers[2]].T,
        ct_xyz[:, centers[1], :].T,
        ct_xyz[centers[0], :, :].T,
    ]
    rectangles = [
        (start[0], start[1], end[0] - start[0], end[1] - start[1]),
        (start[0], start[2], end[0] - start[0], end[2] - start[2]),
        (start[1], start[2], end[1] - start[1], end[2] - start[2]),
    ]
    labels = ("axial", "coronal", "sagittal")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for axis, image, rectangle, label in zip(axes, arrays, rectangles, labels):
        axis.imshow(np.clip(image, -1000, 500), cmap="gray", vmin=-1000, vmax=500, origin="lower")
        axis.add_patch(Rectangle(rectangle[:2], rectangle[2], rectangle[3], fill=False, edgecolor="red", linewidth=1))
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _flatten_scan_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scan_rows = []
    mask_rows = []
    for result in results:
        for candidate in result["candidates"]:
            common = {
                "subject_id": result["subject_id"],
                "session": result["session"],
                "scan_key": result["scan_key"],
                "candidate_id": candidate["candidate_id"],
                "axis_policy": candidate["axis_policy"],
                "margin_mm": candidate["margin_mm"],
            }
            scan_rows.append(
                {
                    **common,
                    "native_shape_xyz": json.dumps(result["ct"]["native_shape_xyz"]),
                    "spacing_xyz_mm": json.dumps(result["ct"]["spacing_xyz_mm"]),
                    "crop_start_xyz": json.dumps(candidate["crop_start_xyz"]),
                    "crop_end_xyz": json.dumps(candidate["crop_end_xyz"]),
                    "crop_shape_xyz": json.dumps(candidate["crop_shape_xyz"]),
                    "target_shape_xyz": json.dumps(candidate["target_shape_xyz"]),
                    "padding_lower_xyz": json.dumps(candidate["padding_lower_xyz"]),
                    "padding_upper_xyz": json.dumps(candidate["padding_upper_xyz"]),
                    "padded_shape_xyz": json.dumps(candidate["padded_shape_xyz"]),
                    "native_voxels": candidate["native_voxels"],
                    "native_crop_voxels": candidate["native_crop_voxels"],
                    "padded_2mm_voxels": candidate["padded_2mm_voxels"],
                    "uncropped_target_shape_xyz": json.dumps(
                        candidate["uncropped_target_shape_xyz"]
                    ),
                    "uncropped_padded_shape_xyz": json.dumps(
                        candidate["uncropped_padded_shape_xyz"]
                    ),
                    "uncropped_padded_2mm_voxels": candidate[
                        "uncropped_padded_2mm_voxels"
                    ],
                    "native_voxel_retained_fraction": candidate[
                        "native_voxel_retained_fraction"
                    ],
                    "native_voxel_reduction_fraction": candidate["native_voxel_reduction_fraction"],
                    "padded_2mm_voxel_retained_fraction": candidate[
                        "padded_2mm_voxel_retained_fraction"
                    ],
                    "padded_2mm_voxel_reduction_fraction": candidate[
                        "padded_2mm_voxel_reduction_fraction"
                    ],
                    "clipped_mask_voxels": candidate["clipped_mask_voxels"],
                    "clipped_mask_count": len(candidate["clipped_mask_names"]),
                    "minimum_artificial_mask_clearance_mm": candidate["minimum_artificial_mask_clearance_mm"],
                    "stride_compatible": candidate["stride_compatible"],
                    "coordinate_roundtrip_passed": candidate["coordinate_roundtrip"]["passed"],
                    "eligible_for_scan": candidate["eligible_for_scan"],
                }
            )
            for mask in candidate["masks"]:
                mask_rows.append(
                    {
                        **common,
                        "mask_name": mask["mask_name"],
                        "mask_voxels": mask["mask_voxels"],
                        "outside_crop_voxels": mask["outside_crop_voxels"],
                        "minimum_artificial_clearance_mm": mask["minimum_artificial_clearance_mm"],
                        "mask_start_xyz": json.dumps(mask["mask_start_xyz"]),
                        "mask_end_xyz": json.dumps(mask["mask_end_xyz"]),
                        "boundary_clearances": json.dumps(mask["boundary_clearances"], sort_keys=True),
                    }
                )
    return scan_rows, mask_rows


def summarize_candidates(scan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scan_rows:
        grouped.setdefault(row["candidate_id"], []).append(row)
    summaries = []
    for identifier, rows in sorted(grouped.items()):
        if len(rows) != EXPECTED_SCANS:
            raise BodyEnvelopeAuditError(
                f"Candidate {identifier} has {len(rows)} rows instead of {EXPECTED_SCANS}"
            )
        padded = np.asarray([row["padded_2mm_voxels"] for row in rows], dtype=np.int64)
        reduction = np.asarray(
            [row["native_voxel_reduction_fraction"] for row in rows], dtype=float
        )
        clearances = [
            float(row["minimum_artificial_mask_clearance_mm"])
            for row in rows
            if row["minimum_artificial_mask_clearance_mm"] is not None
        ]
        clipped_voxels = int(sum(int(row["clipped_mask_voxels"]) for row in rows))
        coordinate_pass = all(bool(row["coordinate_roundtrip_passed"]) for row in rows)
        stride_pass = all(bool(row["stride_compatible"]) for row in rows)
        minimum_clearance = min(clearances) if clearances else None
        clearance_pass = minimum_clearance is None or minimum_clearance + 1e-6 >= MIN_ARTIFICIAL_CLEARANCE_MM
        eligible = clipped_voxels == 0 and coordinate_pass and stride_pass and clearance_pass
        largest = max(rows, key=lambda row: int(row["padded_2mm_voxels"]))
        summaries.append(
            {
                "candidate_id": identifier,
                "axis_policy": rows[0]["axis_policy"],
                "margin_mm": float(rows[0]["margin_mm"]),
                "scans": len(rows),
                "total_clipped_mask_voxels": clipped_voxels,
                "scans_with_clipping": sum(int(row["clipped_mask_voxels"] > 0) for row in rows),
                "coordinate_roundtrip_passed": coordinate_pass,
                "stride_compatible": stride_pass,
                "minimum_artificial_mask_clearance_mm": minimum_clearance,
                "clearance_gate_passed": clearance_pass,
                "eligible": eligible,
                "largest_padded_2mm_voxels": int(padded.max()),
                "p95_padded_2mm_voxels": float(np.percentile(padded, 95)),
                "median_padded_2mm_voxels": float(np.median(padded)),
                "median_native_voxel_reduction_fraction": float(np.median(reduction)),
                "largest_scan_key": largest["scan_key"],
                "largest_padded_shape_xyz": largest["padded_shape_xyz"],
            }
        )
    eligible = [item for item in summaries if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            item["largest_padded_2mm_voxels"],
            item["p95_padded_2mm_voxels"],
            -float(item["minimum_artificial_mask_clearance_mm"] or float("inf")),
            0 if item["axis_policy"] == "xy" else 1,
        )
    )
    ranks = {item["candidate_id"]: index + 1 for index, item in enumerate(eligible)}
    for item in summaries:
        item["eligible_rank"] = ranks.get(item["candidate_id"])
    summaries.sort(
        key=lambda item: (
            item["eligible_rank"] is None,
            item["eligible_rank"] or 999,
            item["candidate_id"],
        )
    )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise BodyEnvelopeAuditError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def render_report(
    run_directory: Path,
    summaries: list[dict[str, Any]],
    scan_rows: list[dict[str, Any]],
    recommended: str | None,
) -> str:
    lines = [
        "# Quadra Stage 1 Body-Envelope Audit",
        "",
        "## Result",
        "",
        f"- Technical audit status: **PASS**",
        f"- Scans: `{EXPECTED_SCANS}`",
        f"- Expected masks evaluated: `{EXPECTED_MASKS}` per candidate",
        f"- Recommended candidate for review: `{recommended or 'none'}`",
        "- Selection frozen: **No**",
        "",
        "The recommendation is not a scientific result and must be reviewed before the `select` command is run.",
        "",
        "## Definitions",
        "",
        "- **Body envelope:** the half-open raw-ITK bounding box of CT voxels "
        "above −800 HU after discarding only connected components smaller than 10 mL.",
        "- **XY crop:** crops surrounding air in x and y while retaining the complete acquired z range.",
        "- **XYZ crop:** crops the body envelope in all three axes.",
        "- **Margin:** physical space added on each cropped side before clamping to the original CT field of view.",
        "- **Stride padding:** symmetric background padding that makes the 2 mm shape divisible by `(16,16,4)`.",
        "",
        "## Candidate summary",
        "",
        "| Rank | Candidate | Eligible | Clipped voxels | Minimum clearance (mm) "
        "| Largest 2 mm voxels | P95 voxels | Median native reduction | Largest scan |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        clearance = item["minimum_artificial_mask_clearance_mm"]
        lines.append(
            (
                "| {rank} | `{identifier}` | {eligible} | {clipped} | {clearance} "
                "| {largest} | {p95:.0f} | {reduction:.1%} | `{scan}` |"
            ).format(
                rank=item["eligible_rank"] or "—",
                identifier=item["candidate_id"],
                eligible="yes" if item["eligible"] else "no",
                clipped=item["total_clipped_mask_voxels"],
                clearance="—" if clearance is None else f"{clearance:.1f}",
                largest=item["largest_padded_2mm_voxels"],
                p95=item["p95_padded_2mm_voxels"],
                reduction=item["median_native_voxel_reduction_fraction"],
                scan=item["largest_scan_key"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- TotalSegmentator masks test the 39 female-specific or 40 male-specific "
            "expected structures per scan; "
            "they do not enumerate every anatomical structure.",
            "- The retained-component policy may keep the CT table. This can reduce "
            "memory savings but avoids aggressive anatomy removal.",
            "- The 20 mm artificial-boundary clearance is a provisional engineering guard, not a clinical threshold.",
            "- This audit estimates voxel reduction only. It does not measure UAE-S "
            "GPU memory or numerical equivalence.",
            "",
            "## Review files",
            "",
            "- `candidate_summary.csv` contains the ranking inputs.",
            "- `scan_audit.csv` contains every scan/candidate geometry.",
            "- `mask_clearance.csv` contains every mask/candidate safety measurement.",
            "- `body_envelope_overview.png` provides the visual crop-envelope audit.",
            "- `qc/largest_padded_scan.png` and `qc/worst_clearance_scan.png` "
            "preserve the detailed three-plane review cases.",
            "",
        ]
    )
    return "\n".join(lines)


def make_overview(preview_directory: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(preview_directory.glob("*.png"))
    if len(paths) != EXPECTED_SCANS:
        raise BodyEnvelopeAuditError(
            f"Expected {EXPECTED_SCANS} preview images, found {len(paths)}"
        )
    columns = 4
    rows = int(math.ceil(len(paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 3.0))
    flat = np.asarray(axes).reshape(-1)
    for axis, path in zip(flat, paths):
        axis.imshow(plt.imread(path))
        axis.set_title(path.stem, fontsize=7)
        axis.axis("off")
    for axis in flat[len(paths) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def preserve_review_previews(
    run_directory: Path,
    scan_rows: list[dict[str, Any]],
    candidate_for_review: str,
) -> dict[str, Any]:
    rows = [row for row in scan_rows if row["candidate_id"] == candidate_for_review]
    if len(rows) != EXPECTED_SCANS:
        raise BodyEnvelopeAuditError(
            f"Review candidate {candidate_for_review} has {len(rows)} scans"
        )
    largest = max(rows, key=lambda row: int(row["padded_2mm_voxels"]))
    rows_with_clearance = [
        row
        for row in rows
        if row["minimum_artificial_mask_clearance_mm"] is not None
    ]
    worst = (
        min(
            rows_with_clearance,
            key=lambda row: float(row["minimum_artificial_mask_clearance_mm"]),
        )
        if rows_with_clearance
        else largest
    )
    qc_directory = run_directory / "qc"
    qc_directory.mkdir(exist_ok=True)
    cases = {
        "candidate_id": candidate_for_review,
        "largest_padded_scan": largest,
        "worst_clearance_scan": worst,
    }
    for label, row in (
        ("largest_padded_scan", largest),
        ("worst_clearance_scan", worst),
    ):
        destination = qc_directory / f"{label}.png"
        result = load_json(run_directory / "scan_results" / f"{row['scan_key']}.json")
        candidate = _candidate_from_result(result, candidate_for_review)
        import nibabel as nib

        ct = np.asanyarray(nib.load(result["ct"]["path"]).dataobj)
        make_preview(
            ct,
            candidate["crop_start_xyz"],
            candidate["crop_end_xyz"],
            destination,
            f"{row['scan_key']} — {candidate_for_review}",
        )
        del ct
    atomic_replace_json(run_directory / "review_cases.json", cases)
    return cases


def _run_directory(
    storage_root: Path,
    output_root: Path,
    run_id: str | None,
    resume_run_directory: Path | None,
) -> tuple[Path, bool]:
    if resume_run_directory is not None:
        directory = Path(resume_run_directory).resolve()
        if not _is_within(directory, storage_root):
            raise BodyEnvelopeAuditError("Resume directory escapes the storage root")
        if not directory.is_dir():
            raise BodyEnvelopeAuditError(f"Resume directory does not exist: {directory}")
        return directory, True
    directory = Path(output_root).resolve() / (run_id or default_run_id())
    if not _is_within(directory, storage_root):
        raise BodyEnvelopeAuditError("Stage 1 output must remain inside the storage root")
    try:
        directory.mkdir(parents=True)
    except FileExistsError as exc:
        raise BodyEnvelopeAuditError(f"Refusing to reuse existing audit directory: {directory}") from exc
    return directory, False


def validate_resume_contract(
    existing: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Require an incomplete, otherwise identical audit before resuming it."""
    if existing.get("status") != "in_progress":
        raise BodyEnvelopeAuditError(
            "Refusing to resume an audit that is not in progress; "
            "completed audit evidence is immutable"
        )
    for key in (
        "schema_version",
        "audit_id",
        "baseline_manifest",
        "repository",
        "cohort_manifest",
        "settings",
        "denominators",
        "scientific_computation",
    ):
        if existing.get(key) != expected.get(key):
            raise BodyEnvelopeAuditError(f"Resume contract mismatch for {key}")


def run_audit(args: argparse.Namespace) -> Path:
    storage_root = Path(args.storage_root).resolve()
    baseline_path = Path(args.baseline_manifest).resolve()
    baseline_identity = verify_baseline_identity(baseline_path)
    baseline.validate_locked_contract(
        baseline_path,
        repository_root=Path(args.repository_root),
        storage_root=storage_root,
        required_profile="preprocess",
    )
    repository = current_repository_record(Path(args.repository_root))
    layout = canonical_layout(storage_root)
    cohort_path = Path(args.cohort_manifest or (layout["manifests"] / COHORT_MANIFEST_NAME)).resolve()
    cohort_identity = file_identity(cohort_path)
    _, scans = load_cohort(
        cohort_path,
        Path(layout["whole_body_ct"]),
        Path(layout["totalsegmentator_outputs"]),
    )
    run_directory, resuming = _run_directory(
        storage_root,
        Path(args.output_root or (storage_root / "runs/memory_optimization")),
        args.run_id,
        args.resume_run_directory,
    )
    contract_path = run_directory / "audit_manifest.json"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "in_progress",
        "created_at": utc_now(),
        "baseline_manifest": baseline_identity,
        "repository": repository,
        "cohort_manifest": cohort_identity,
        "settings": audit_settings(),
        "denominators": {
            "subjects": EXPECTED_SUBJECTS,
            "scans": EXPECTED_SCANS,
            "masks_per_scan_by_sex": dict(EXPECTED_MASKS_PER_SCAN_BY_SEX),
            "masks": EXPECTED_MASKS,
            "candidates_per_scan": len(AXIS_POLICIES) * len(MARGINS_MM),
        },
        "scientific_computation": {
            "cropped_images_written": False,
            "ct_resampling_performed": False,
            "uae_model_loaded": False,
            "cuda_used": False,
            "segmentation_run": False,
            "cycle_error_generated": False,
        },
    }
    if resuming:
        existing = load_json(contract_path)
        validate_resume_contract(existing, contract)
        contract = existing
    else:
        atomic_create_json(contract_path, contract)

    result_directory = run_directory / "scan_results"
    preview_directory = run_directory / "previews"
    log_path = run_directory / "stage1-audit.log"
    result_directory.mkdir(exist_ok=True)
    preview_directory.mkdir(exist_ok=True)
    emit(
        f"Starting Stage 1 audit with {len(scans)} scans; resume={resuming}",
        log_path,
    )
    results = []
    for index, scan in enumerate(scans, start=1):
        result_path = result_directory / f"{scan['key']}.json"
        preview_path = preview_directory / f"{scan['key']}.png"
        signature = _scan_signature(scan, contract)
        if result_path.is_file():
            result = load_json(result_path)
            if result.get("status") != "passed" or result.get("scan_signature") != signature:
                raise BodyEnvelopeAuditError(f"Incompatible resumable result: {result_path}")
            if not preview_path.is_file():
                raise BodyEnvelopeAuditError(f"Resumable result lacks preview: {preview_path}")
            if result.get("preview") != file_identity(preview_path):
                raise BodyEnvelopeAuditError(
                    f"Resumable preview changed after scan completion: {preview_path}"
                )
            emit(f"[{index}/{len(scans)}] Reusing {scan['key']}", log_path)
        else:
            emit(f"[{index}/{len(scans)}] Auditing {scan['key']}", log_path)
            result = process_scan(scan, contract, preview_path=preview_path)
            atomic_create_json(result_path, result)
        results.append(result)

    scan_rows, mask_rows = _flatten_scan_rows(results)
    summaries = summarize_candidates(scan_rows)
    recommended = next(
        (item["candidate_id"] for item in summaries if item["eligible"]), None
    )
    write_csv(run_directory / "scan_audit.csv", scan_rows)
    write_csv(run_directory / "mask_clearance.csv", mask_rows)
    write_csv(run_directory / "candidate_summary.csv", summaries)
    make_overview(preview_directory, run_directory / "body_envelope_overview.png")
    review_candidate = recommended or summaries[0]["candidate_id"]
    preserve_review_previews(run_directory, scan_rows, review_candidate)
    report = render_report(run_directory, summaries, scan_rows, recommended)
    atomic_replace_text(run_directory / "body_envelope_audit_report.md", report)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "PASS",
        "completed_at": utc_now(),
        "run_directory": str(run_directory),
        "baseline_manifest": baseline_identity,
        "scans_completed": len(results),
        "mask_candidate_rows": len(mask_rows),
        "candidate_count": len(summaries),
        "eligible_candidates": [item["candidate_id"] for item in summaries if item["eligible"]],
        "recommended_candidate_for_review": recommended,
        "selection_frozen": False,
        "next_action": "review_then_run_select",
    }
    atomic_replace_json(run_directory / "audit_summary.json", summary)
    emit("Stage 1 audit PASS — review required before selection", log_path)
    emit(f"Run directory: {run_directory}", log_path)
    emit(f"Recommended candidate: {recommended or 'none'}", log_path)
    contract.update(
        {
            "status": "passed",
            "completed_at": utc_now(),
            "outputs": {
                name: file_identity(run_directory / name)
                for name in (
                    "scan_audit.csv",
                    "mask_clearance.csv",
                    "candidate_summary.csv",
                    "audit_summary.json",
                    "body_envelope_audit_report.md",
                    "body_envelope_overview.png",
                    "review_cases.json",
                    "qc/largest_padded_scan.png",
                    "qc/worst_clearance_scan.png",
                    "stage1-audit.log",
                )
            },
        }
    )
    atomic_replace_json(contract_path, contract)
    return run_directory


def _candidate_from_result(result: dict[str, Any], identifier: str) -> dict[str, Any]:
    matches = [item for item in result["candidates"] if item["candidate_id"] == identifier]
    if len(matches) != 1:
        raise BodyEnvelopeAuditError(
            f"Expected one {identifier} candidate for {result.get('scan_key')}, found {len(matches)}"
        )
    return matches[0]


def verify_audit_outputs(run_directory: Path, audit_manifest: dict[str, Any]) -> None:
    outputs = audit_manifest.get("outputs")
    required = {
        "scan_audit.csv",
        "mask_clearance.csv",
        "candidate_summary.csv",
        "audit_summary.json",
        "body_envelope_audit_report.md",
        "body_envelope_overview.png",
        "review_cases.json",
        "qc/largest_padded_scan.png",
        "qc/worst_clearance_scan.png",
        "stage1-audit.log",
    }
    if not isinstance(outputs, dict) or set(outputs) != required:
        raise BodyEnvelopeAuditError("Audit output manifest is incomplete")
    for name in sorted(required):
        observed = file_identity(run_directory / name)
        expected = outputs[name]
        if observed["bytes"] != expected.get("bytes") or observed["sha256"] != expected.get("sha256"):
            raise BodyEnvelopeAuditError(f"Audit output changed after completion: {name}")


def run_select(args: argparse.Namespace) -> Path:
    run_directory = Path(args.audit_run_directory).resolve()
    audit_manifest_path = run_directory / "audit_manifest.json"
    audit_manifest = load_json(audit_manifest_path)
    if audit_manifest.get("status") != "passed":
        raise BodyEnvelopeAuditError("Audit is incomplete and cannot be selected")
    storage_root = Path(args.storage_root).resolve()
    if not _is_within(run_directory, storage_root):
        raise BodyEnvelopeAuditError("Audit run directory escapes the storage root")
    verify_audit_outputs(run_directory, audit_manifest)
    baseline_identity = audit_manifest["baseline_manifest"]
    if baseline_identity["sha256"] != EXPECTED_BASELINE_SHA256:
        raise BodyEnvelopeAuditError("Audit references the wrong Stage 0 manifest")
    baseline_path = Path(baseline_identity["path"])
    verify_baseline_identity(baseline_path)
    baseline.validate_locked_contract(
        baseline_path,
        repository_root=Path(args.repository_root),
        storage_root=storage_root,
        required_profile="preprocess",
    )
    summary = load_json(run_directory / "audit_summary.json")
    if summary.get("status") != "PASS" or summary.get("selection_frozen"):
        raise BodyEnvelopeAuditError("Audit summary is not ready for a new selection")
    expected_mask_rows = EXPECTED_MASKS * len(AXIS_POLICIES) * len(MARGINS_MM)
    if summary.get("scans_completed") != EXPECTED_SCANS or summary.get("mask_candidate_rows") != expected_mask_rows:
        raise BodyEnvelopeAuditError("Audit denominators are incomplete")
    with (run_directory / "candidate_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    matches = [row for row in candidates if row["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        raise BodyEnvelopeAuditError(f"Unknown candidate: {args.candidate_id}")
    if matches[0]["eligible"].lower() not in {"true", "1", "yes"}:
        raise BodyEnvelopeAuditError(f"Candidate is not eligible: {args.candidate_id}")
    result_paths = sorted((run_directory / "scan_results").glob("*.json"))
    if len(result_paths) != EXPECTED_SCANS:
        raise BodyEnvelopeAuditError(
            f"Expected {EXPECTED_SCANS} scan results, found {len(result_paths)}"
        )
    plans = []
    original_fov_limitations = []
    for path in result_paths:
        result = load_json(path)
        candidate = _candidate_from_result(result, args.candidate_id)
        cropped_axes = (0, 1) if candidate["axis_policy"] == "xy" else (0, 1, 2)
        lower_artificial = candidate["artificial_boundaries"]["lower_xyz"]
        upper_artificial = candidate["artificial_boundaries"]["upper_xyz"]
        scan_limitations = []
        for axis in cropped_axes:
            axis_name = ("x", "y", "z")[axis]
            if not lower_artificial[axis]:
                scan_limitations.append(f"{axis_name}_lower_original_fov")
            if not upper_artificial[axis]:
                scan_limitations.append(f"{axis_name}_upper_original_fov")
        plan = {
            "subject_id": result["subject_id"],
            "session": result["session"],
            "scan_key": result["scan_key"],
            "axis_policy": candidate["axis_policy"],
            "margin_mm": candidate["margin_mm"],
            "source_ct": result["ct"],
            "body_envelope": result["body_envelope"],
            "crop_start_xyz": candidate["crop_start_xyz"],
            "crop_end_xyz": candidate["crop_end_xyz"],
            "crop_shape_xyz": candidate["crop_shape_xyz"],
            "target_shape_xyz": candidate["target_shape_xyz"],
            "padding_lower_xyz": candidate["padding_lower_xyz"],
            "padding_upper_xyz": candidate["padding_upper_xyz"],
            "padded_shape_xyz": candidate["padded_shape_xyz"],
            "model_tensor_shape_zyx": candidate["model_tensor_shape_zyx"],
            "padded_2mm_voxels": candidate["padded_2mm_voxels"],
            "native_voxel_reduction_fraction": candidate[
                "native_voxel_reduction_fraction"
            ],
            "padded_2mm_voxel_reduction_fraction": candidate[
                "padded_2mm_voxel_reduction_fraction"
            ],
            "artificial_boundaries": candidate["artificial_boundaries"],
            "minimum_artificial_mask_clearance_mm": candidate[
                "minimum_artificial_mask_clearance_mm"
            ],
            "native_crop_affine": candidate["native_crop_affine"],
            "resampled_2mm_affine": candidate["resampled_2mm_affine"],
            "padded_2mm_affine": candidate["padded_2mm_affine"],
            "raw_to_crop_continuous_affine": candidate["coordinate_roundtrip"][
                "raw_to_crop_continuous_affine"
            ],
            "crop_to_raw_continuous_affine": candidate["coordinate_roundtrip"][
                "crop_to_raw_continuous_affine"
            ],
            "crop_to_model_continuous_affine": candidate["coordinate_roundtrip"][
                "crop_to_model_continuous_affine"
            ],
            "model_to_crop_continuous_affine": candidate["coordinate_roundtrip"][
                "model_to_crop_continuous_affine"
            ],
            "raw_to_model_continuous_affine": candidate["coordinate_roundtrip"][
                "raw_to_model_continuous_affine"
            ],
            "model_to_raw_continuous_affine": candidate["coordinate_roundtrip"][
                "model_to_raw_continuous_affine"
            ],
            "original_fov_limitations": scan_limitations,
        }
        plans.append(plan)
        if scan_limitations:
            original_fov_limitations.append(
                {"scan_key": result["scan_key"], "boundaries": scan_limitations}
            )
    largest_scan = max(plans, key=lambda item: int(item["padded_2mm_voxels"]))
    subjects: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        subjects.setdefault(plan["subject_id"], []).append(plan)
    pairs = []
    for subject, subject_plans in subjects.items():
        if len(subject_plans) != 2 or {
            item["session"] for item in subject_plans
        } != {"test", "retest"}:
            raise BodyEnvelopeAuditError(f"Incomplete Test/Retest pair: {subject}")
        values = [int(item["padded_2mm_voxels"]) for item in subject_plans]
        pairs.append(
            {
                "subject_id": subject,
                "peak_scan_padded_2mm_voxels": max(values),
                "pair_total_padded_2mm_voxels": sum(values),
                "scans": sorted(subject_plans, key=lambda item: item["session"]),
            }
        )
    largest_pair = max(
        pairs,
        key=lambda item: (
            item["peak_scan_padded_2mm_voxels"],
            item["pair_total_padded_2mm_voxels"],
        ),
    )
    selected = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "selected",
        "selected_at": utc_now(),
        "candidate_id": args.candidate_id,
        "review_rationale": args.review_rationale,
        "baseline_manifest": baseline_identity,
        "audit_manifest": file_identity(audit_manifest_path),
        "settings": audit_manifest["settings"],
        "candidate_summary": matches[0],
        "scan_plans": plans,
        "original_fov_limitations": original_fov_limitations,
        "largest_single_scan": largest_scan,
        "largest_test_retest_pair": largest_pair,
        "selection_policy": {
            "largest_pair_primary_key": "maximum_single_scan_padded_2mm_voxels",
            "largest_pair_tiebreak": "pair_total_padded_2mm_voxels",
            "human_review_required": True,
        },
    }
    selected_path = run_directory / "selected_body_envelope.json"
    atomic_create_json(selected_path, selected)
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "stage": 1,
        "status": "PASS",
        "created_at": utc_now(),
        "audit_id": AUDIT_ID,
        "selected_candidate": args.candidate_id,
        "selected_body_envelope": file_identity(selected_path),
        "largest_single_scan": largest_scan["scan_key"],
        "largest_test_retest_pair": largest_pair["subject_id"],
        "gates": {
            "stage0_contract_validated": True,
            "all_56_scans_completed": True,
            "all_2208_masks_evaluated_per_candidate": True,
            "zero_mask_voxels_clipped": True,
            "minimum_artificial_clearance_passed": True,
            "coordinate_roundtrip_passed": True,
            "stride_compatibility_passed": True,
            "human_review_recorded": True,
            "cropped_images_or_scientific_results_generated": False,
        },
        "next_stage": "coordinate_preserving_crop_implementation",
    }
    atomic_create_json(run_directory / "checkpoint_summary.json", checkpoint)
    print("Stage 1 selection PASS", flush=True)
    print(f"Selected candidate: {args.candidate_id}", flush=True)
    print(f"Largest Test/Retest pair: {largest_pair['subject_id']}", flush=True)
    print(f"Checkpoint: {run_directory / 'checkpoint_summary.json'}", flush=True)
    return selected_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit every crop candidate; do not freeze a selection.")
    audit.add_argument("--baseline-manifest", type=Path, default=EXPECTED_BASELINE_PATH)
    audit.add_argument("--storage-root", type=Path, default=Path("/workspace/quadra"))
    audit.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    audit.add_argument("--cohort-manifest", type=Path, default=None)
    audit.add_argument("--output-root", type=Path, default=None)
    audit.add_argument("--run-id", default=None)
    audit.add_argument("--resume-run-directory", type=Path, default=None)

    select = subparsers.add_parser("select", help="Freeze one eligible candidate after review.")
    select.add_argument("--audit-run-directory", type=Path, required=True)
    select.add_argument("--candidate-id", required=True)
    select.add_argument("--review-rationale", required=True)
    select.add_argument("--storage-root", type=Path, default=Path("/workspace/quadra"))
    select.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            if args.run_id and args.resume_run_directory:
                raise BodyEnvelopeAuditError(
                    "Use either --run-id or --resume-run-directory, not both"
                )
            run_audit(args)
        else:
            run_select(args)
    except (BodyEnvelopeAuditError, baseline.OptimizationBaselineError) as exc:
        parser.exit(2, f"Stage 1 failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
