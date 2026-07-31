#!/usr/bin/env python3
"""Prepare and validate frozen, coordinate-preserving Quadra body crops.

Stage 2 consumes the immutable Stage 1 ``xy_m010`` plans.  It never detects a
new body envelope, loads UAE-S, allocates CUDA tensors, or writes a prepared 3D
volume.  The reusable preparation function returns one normalized CPU array in
``ZYX`` order together with explicit raw-ITK/model ``XYZ`` transforms.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra import body_envelope_audit as stage1  # noqa: E402
from tools.quadra import optimization_baseline as baseline  # noqa: E402


SCHEMA_VERSION = 1
PREPARATION_ID = "quadra-coordinate-preserving-crop-v1"
EXPECTED_STAGE1_CHECKPOINT_PATH = Path(
    "/workspace/quadra/runs/memory_optimization/"
    "stage1-audit-20260731T110726Z/checkpoint_summary.json"
)
EXPECTED_SELECTED_SHA256 = (
    "ec3df0f1c9ed3148058a850ac4591a2dd5a84e029786a5c6c52db140783b5b6c"
)
EXPECTED_STAGE1_EXECUTION_COMMIT = (
    "6489e14080e4dd30f32581978d9df2cd926eeb97"
)
EXPECTED_CANDIDATE_ID = "xy_m010"
EXPECTED_LARGEST_PAIR = "quadra_hc_044"
EXPECTED_SCAN_COUNT = 56
TARGET_SPACING_XYZ_MM = (2.0, 2.0, 2.0)
MODEL_STRIDE_XYZ = (16, 16, 4)
PADDING_HU = -1024.0
CT_MIN_HU = -1024.0
CT_MAX_HU = 3071.0
INTENSITY_OUTPUT_MIN = 0.0
INTENSITY_OUTPUT_MAX = 255.0
INTENSITY_BIAS = 50.0
MIN_CLEARANCE_MM = 20.0
AFFINE_ATOL = 1e-5
ROUNDTRIP_VOXEL_ATOL = 1e-6
ROUNDTRIP_PHYSICAL_ATOL_MM = 1e-5
RUN_ID_PATTERN = re.compile(r"stage2-crop-[A-Za-z0-9._-]+")
RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


class CoordinatePreservingCropError(RuntimeError):
    """Raised when Stage 2 cannot safely prepare or validate a frozen crop."""


@dataclass
class PreparedVolume:
    """One normalized CPU model input and its coordinate contract.

    ``data_zyx`` is deliberately not a torch tensor.  A later stage can create
    an NCDHW view with ``data_zyx[None, None]`` without changing the spatial
    policy frozen here.
    """

    data_zyx: np.ndarray
    raw_affine_ras: np.ndarray
    model_affine_ras: np.ndarray
    raw_to_model_xyz: np.ndarray
    model_to_raw_xyz: np.ndarray
    metadata: dict[str, Any]

    @property
    def tensor_shape_ncdhw(self) -> tuple[int, int, int, int, int]:
        z, y, x = self.data_zyx.shape
        return (1, 1, z, y, x)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("stage2-crop-%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise CoordinatePreservingCropError(f"Required file is missing: {resolved}")
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


def load_json(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CoordinatePreservingCropError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CoordinatePreservingCropError(f"Expected a JSON object: {path}")
    return value


def atomic_replace_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_create_json(path: Path, value: Any) -> None:
    if Path(path).exists():
        raise CoordinatePreservingCropError(f"Refusing to overwrite existing file: {path}")
    atomic_replace_json(path, value)


def atomic_replace_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise CoordinatePreservingCropError(f"Refusing to write an empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def emit(message: str, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {message}\n")


def parse_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise CoordinatePreservingCropError(f"{label} is not a strict boolean: {value!r}")


def parse_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CoordinatePreservingCropError(f"{label} is not an integer: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CoordinatePreservingCropError(f"{label} is not an integer: {value!r}") from exc
    if isinstance(value, float) and not float(value).is_integer():
        raise CoordinatePreservingCropError(f"{label} is not an integer: {value!r}")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise CoordinatePreservingCropError(f"{label} is not a canonical integer: {value!r}")
    return parsed


def parse_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CoordinatePreservingCropError(f"{label} is not numeric: {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CoordinatePreservingCropError(f"{label} is not numeric: {value!r}") from exc
    if not np.isfinite(parsed):
        raise CoordinatePreservingCropError(f"{label} is not finite: {value!r}")
    return parsed


def typed_candidate_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CoordinatePreservingCropError("Stage 1 candidate summary is not an object")
    required = {
        "candidate_id",
        "axis_policy",
        "margin_mm",
        "scans",
        "total_clipped_mask_voxels",
        "coordinate_roundtrip_passed",
        "stride_compatible",
        "minimum_artificial_mask_clearance_mm",
        "clearance_gate_passed",
        "eligible",
    }
    missing = sorted(required - set(value))
    if missing:
        raise CoordinatePreservingCropError(
            f"Stage 1 candidate summary is incomplete: {', '.join(missing)}"
        )
    result = dict(value)
    result["margin_mm"] = parse_float(value["margin_mm"], "margin_mm")
    result["scans"] = parse_int(value["scans"], "scans")
    result["total_clipped_mask_voxels"] = parse_int(
        value["total_clipped_mask_voxels"], "total_clipped_mask_voxels"
    )
    result["minimum_artificial_mask_clearance_mm"] = parse_float(
        value["minimum_artificial_mask_clearance_mm"],
        "minimum_artificial_mask_clearance_mm",
    )
    for key in (
        "coordinate_roundtrip_passed",
        "stride_compatible",
        "clearance_gate_passed",
        "eligible",
    ):
        result[key] = parse_bool(value[key], key)
    return result


def _as_vector(value: Any, label: str, *, dtype: Any = float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (3,):
        raise CoordinatePreservingCropError(f"{label} must contain three values")
    if not np.isfinite(array.astype(float)).all():
        raise CoordinatePreservingCropError(f"{label} contains non-finite values")
    return array


def _as_affine(value: Any, label: str) -> np.ndarray:
    affine = np.asarray(value, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise CoordinatePreservingCropError(f"{label} must be a finite 4 by 4 matrix")
    if not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise CoordinatePreservingCropError(f"{label} has an invalid homogeneous row")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
        raise CoordinatePreservingCropError(f"{label} is singular")
    return affine


def _validate_scan_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise CoordinatePreservingCropError("A Stage 1 scan plan is not an object")
    required = {
        "subject_id",
        "session",
        "scan_key",
        "axis_policy",
        "margin_mm",
        "source_ct",
        "crop_start_xyz",
        "crop_end_xyz",
        "crop_shape_xyz",
        "target_shape_xyz",
        "padding_lower_xyz",
        "padding_upper_xyz",
        "padded_shape_xyz",
        "model_tensor_shape_zyx",
        "minimum_artificial_mask_clearance_mm",
        "native_crop_affine",
        "resampled_2mm_affine",
        "padded_2mm_affine",
        "raw_to_model_continuous_affine",
        "model_to_raw_continuous_affine",
    }
    missing = sorted(required - set(plan))
    if missing:
        raise CoordinatePreservingCropError(
            f"Stage 1 scan plan is incomplete: {', '.join(missing)}"
        )
    if plan["axis_policy"] != "xy" or parse_float(plan["margin_mm"], "plan margin") != 10.0:
        raise CoordinatePreservingCropError(f"Unexpected crop policy in {plan['scan_key']}")
    if plan["session"] not in {"test", "retest"}:
        raise CoordinatePreservingCropError(f"Invalid session in {plan['scan_key']}")
    source = plan["source_ct"]
    if not isinstance(source, dict) or not {"path", "bytes", "sha256", "native_shape_xyz", "affine"}.issubset(source):
        raise CoordinatePreservingCropError(f"Incomplete source CT identity in {plan['scan_key']}")
    start = _as_vector(plan["crop_start_xyz"], "crop_start_xyz", dtype=np.int64)
    end = _as_vector(plan["crop_end_xyz"], "crop_end_xyz", dtype=np.int64)
    crop_shape = _as_vector(plan["crop_shape_xyz"], "crop_shape_xyz", dtype=np.int64)
    native_shape = _as_vector(source["native_shape_xyz"], "native_shape_xyz", dtype=np.int64)
    target_shape = _as_vector(plan["target_shape_xyz"], "target_shape_xyz", dtype=np.int64)
    lower = _as_vector(plan["padding_lower_xyz"], "padding_lower_xyz", dtype=np.int64)
    upper = _as_vector(plan["padding_upper_xyz"], "padding_upper_xyz", dtype=np.int64)
    padded = _as_vector(plan["padded_shape_xyz"], "padded_shape_xyz", dtype=np.int64)
    tensor_zyx = _as_vector(plan["model_tensor_shape_zyx"], "model_tensor_shape_zyx", dtype=np.int64)
    if np.any(start < 0) or np.any(end > native_shape) or np.any(end <= start):
        raise CoordinatePreservingCropError(f"Invalid half-open crop bounds in {plan['scan_key']}")
    if not np.array_equal(end - start, crop_shape):
        raise CoordinatePreservingCropError(f"Crop shape mismatch in {plan['scan_key']}")
    if np.any(lower < 0) or np.any(upper < 0) or not np.array_equal(target_shape + lower + upper, padded):
        raise CoordinatePreservingCropError(f"Padding shape mismatch in {plan['scan_key']}")
    if not np.array_equal(tensor_zyx, padded[::-1]):
        raise CoordinatePreservingCropError(f"XYZ/ZYX shape mismatch in {plan['scan_key']}")
    if np.any(padded % np.asarray(MODEL_STRIDE_XYZ, dtype=np.int64)):
        raise CoordinatePreservingCropError(f"Stride-incompatible plan in {plan['scan_key']}")
    clearance = parse_float(
        plan["minimum_artificial_mask_clearance_mm"],
        "minimum_artificial_mask_clearance_mm",
    )
    if clearance + 1e-6 < MIN_CLEARANCE_MM:
        raise CoordinatePreservingCropError(f"Insufficient mask clearance in {plan['scan_key']}")
    raw_affine = _as_affine(source["affine"], "source_ct.affine")
    model_affine = _as_affine(plan["padded_2mm_affine"], "padded_2mm_affine")
    raw_to_model = _as_affine(
        plan["raw_to_model_continuous_affine"], "raw_to_model_continuous_affine"
    )
    model_to_raw = _as_affine(
        plan["model_to_raw_continuous_affine"], "model_to_raw_continuous_affine"
    )
    calculated = np.linalg.inv(model_affine) @ raw_affine
    if not np.allclose(raw_to_model, calculated, atol=AFFINE_ATOL, rtol=0.0):
        raise CoordinatePreservingCropError(f"Raw-to-model transform mismatch in {plan['scan_key']}")
    if not np.allclose(model_to_raw, np.linalg.inv(raw_to_model), atol=AFFINE_ATOL, rtol=0.0):
        raise CoordinatePreservingCropError(f"Model-to-raw transform mismatch in {plan['scan_key']}")
    return plan


def validate_selected_payload(selected: dict[str, Any]) -> dict[str, Any]:
    if selected.get("schema_version") != stage1.SCHEMA_VERSION:
        raise CoordinatePreservingCropError("Unsupported Stage 1 selected-manifest schema")
    if selected.get("status") != "selected" or selected.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        raise CoordinatePreservingCropError("Stage 1 did not select xy_m010")
    summary = typed_candidate_summary(selected.get("candidate_summary"))
    if summary["candidate_id"] != EXPECTED_CANDIDATE_ID or summary["axis_policy"] != "xy":
        raise CoordinatePreservingCropError("Stage 1 candidate summary identifies the wrong crop")
    if summary["margin_mm"] != 10.0 or summary["scans"] != EXPECTED_SCAN_COUNT:
        raise CoordinatePreservingCropError("Stage 1 candidate summary has unexpected settings")
    if (
        not summary["eligible"]
        or not summary["coordinate_roundtrip_passed"]
        or not summary["stride_compatible"]
        or not summary["clearance_gate_passed"]
        or summary["total_clipped_mask_voxels"] != 0
        or summary["minimum_artificial_mask_clearance_mm"] + 1e-6 < MIN_CLEARANCE_MM
    ):
        raise CoordinatePreservingCropError("Stage 1 candidate failed a required safety gate")
    plans = selected.get("scan_plans")
    if not isinstance(plans, list) or len(plans) != EXPECTED_SCAN_COUNT:
        raise CoordinatePreservingCropError(
            f"Stage 1 must contain {EXPECTED_SCAN_COUNT} scan plans"
        )
    keys = []
    for plan in plans:
        _validate_scan_plan(plan)
        keys.append(plan["scan_key"])
    if len(set(keys)) != len(keys):
        raise CoordinatePreservingCropError("Stage 1 contains duplicate scan plans")
    pair = selected.get("largest_test_retest_pair")
    if not isinstance(pair, dict) or pair.get("subject_id") != EXPECTED_LARGEST_PAIR:
        raise CoordinatePreservingCropError("Stage 1 identifies the wrong largest pair")
    pair_scans = pair.get("scans")
    if not isinstance(pair_scans, list) or len(pair_scans) != 2:
        raise CoordinatePreservingCropError("Largest Test/Retest pair is incomplete")
    if {item.get("session") for item in pair_scans} != {"test", "retest"}:
        raise CoordinatePreservingCropError("Largest pair does not contain Test and Retest")
    plan_lookup = {item["scan_key"]: item for item in plans}
    for item in pair_scans:
        if item.get("scan_key") not in plan_lookup or item != plan_lookup[item["scan_key"]]:
            raise CoordinatePreservingCropError("Largest-pair plan differs from the frozen scan plan")
    return {"selected": selected, "candidate_summary": summary, "pair_scans": pair_scans}


def _identity_matches(record: Any, observed: dict[str, Any], label: str) -> None:
    if not isinstance(record, dict):
        raise CoordinatePreservingCropError(f"Missing {label} identity")
    for key in ("path", "bytes", "sha256"):
        if record.get(key) != observed[key]:
            raise CoordinatePreservingCropError(f"{label} identity mismatch for {key}")


def _require_git_ancestor(repository_root: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "merge-base", "--is-ancestor", commit, "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CoordinatePreservingCropError(
            f"Current repository does not descend from Stage 1 commit {commit}"
        )


def validate_stage1_contract(
    checkpoint_path: Path,
    *,
    repository_root: Path,
    storage_root: Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path != EXPECTED_STAGE1_CHECKPOINT_PATH.resolve():
        raise CoordinatePreservingCropError(
            f"Stage 2 requires the accepted Stage 1 checkpoint at {EXPECTED_STAGE1_CHECKPOINT_PATH}"
        )
    checkpoint_identity = file_identity(checkpoint_path)
    checkpoint = load_json(checkpoint_path)
    required_gates = {
        "stage0_contract_validated",
        "all_56_scans_completed",
        "all_2208_masks_evaluated_per_candidate",
        "zero_mask_voxels_clipped",
        "minimum_artificial_clearance_passed",
        "coordinate_roundtrip_passed",
        "stride_compatibility_passed",
        "human_review_recorded",
    }
    gates = checkpoint.get("gates")
    if (
        checkpoint.get("stage") != 1
        or checkpoint.get("status") != "PASS"
        or checkpoint.get("selected_candidate") != EXPECTED_CANDIDATE_ID
        or checkpoint.get("largest_test_retest_pair") != EXPECTED_LARGEST_PAIR
        or not isinstance(gates, dict)
        or any(gates.get(key) is not True for key in required_gates)
        or gates.get("cropped_images_or_scientific_results_generated") is not False
    ):
        raise CoordinatePreservingCropError("Stage 1 checkpoint failed a required gate")
    selected_record = checkpoint.get("selected_body_envelope")
    selected_path = Path(str(selected_record.get("path", ""))).resolve() if isinstance(selected_record, dict) else Path()
    if not _is_within(selected_path, storage_root):
        raise CoordinatePreservingCropError("Selected Stage 1 manifest escapes the storage root")
    selected_identity = file_identity(selected_path)
    _identity_matches(selected_record, selected_identity, "selected-body-envelope")
    if selected_identity["sha256"] != EXPECTED_SELECTED_SHA256:
        raise CoordinatePreservingCropError(
            "Selected Stage 1 manifest hash differs from the human-reviewed artifact"
        )
    selected = load_json(selected_path)
    validated = validate_selected_payload(selected)
    audit_record = selected.get("audit_manifest")
    audit_path = Path(str(audit_record.get("path", ""))).resolve() if isinstance(audit_record, dict) else Path()
    if not _is_within(audit_path, storage_root):
        raise CoordinatePreservingCropError("Stage 1 audit manifest escapes the storage root")
    audit_identity = file_identity(audit_path)
    _identity_matches(audit_record, audit_identity, "Stage 1 audit manifest")
    audit_manifest = load_json(audit_path)
    audit_commit = audit_manifest.get("repository", {}).get("execution_commit")
    if audit_commit != EXPECTED_STAGE1_EXECUTION_COMMIT:
        raise CoordinatePreservingCropError("Stage 1 audit used an unexpected repository commit")
    _require_git_ancestor(repository_root, EXPECTED_STAGE1_EXECUTION_COMMIT)
    return {
        **validated,
        "checkpoint": checkpoint,
        "checkpoint_identity": checkpoint_identity,
        "selected_identity": selected_identity,
        "audit_identity": audit_identity,
        "stage1_execution_commit": audit_commit,
    }


def ras_affine_to_sitk_geometry(
    affine_ras: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    affine = RAS_TO_LPS @ _as_affine(affine_ras, "RAS affine")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction = affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise CoordinatePreservingCropError("Image direction is not orthonormal")
    return (
        tuple(float(value) for value in spacing),
        tuple(float(value) for value in affine[:3, 3]),
        tuple(float(value) for value in direction.reshape(-1)),
    )


def sitk_geometry_to_ras_affine(image: Any) -> np.ndarray:
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, :3] = direction * spacing
    affine_lps[:3, 3] = origin
    return RAS_TO_LPS @ affine_lps


def apply_affine_xyz(points_xyz: Any, affine: Any) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    single = points.shape == (3,)
    if single:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise CoordinatePreservingCropError("XYZ points must have shape (3,) or (N,3)")
    matrix = _as_affine(affine, "point transform")
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=float)], axis=1)
    transformed = (matrix @ homogeneous.T).T[:, :3]
    return transformed[0] if single else transformed


def xyz_to_zyx(points_xyz: Any) -> np.ndarray:
    points = np.asarray(points_xyz)
    if points.shape[-1:] != (3,):
        raise CoordinatePreservingCropError("XYZ coordinates must end in three values")
    return points[..., ::-1]


def zyx_to_xyz(points_zyx: Any) -> np.ndarray:
    points = np.asarray(points_zyx)
    if points.shape[-1:] != (3,):
        raise CoordinatePreservingCropError("ZYX coordinates must end in three values")
    return points[..., ::-1]


def nearest_raw_indices(points_raw_xyz: Any, raw_shape_xyz: Sequence[int]) -> np.ndarray:
    points = np.asarray(points_raw_xyz, dtype=float)
    single = points.shape == (3,)
    if single:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise CoordinatePreservingCropError("Raw coordinates must have shape (3,) or (N,3)")
    shape = _as_vector(raw_shape_xyz, "raw_shape_xyz", dtype=np.int64)
    rounded = np.rint(points).astype(np.int64)
    if np.any(rounded < 0) or np.any(rounded >= shape):
        raise CoordinatePreservingCropError("Rounded raw coordinate is outside the source image")
    return rounded[0] if single else rounded


def normalize_ct_inplace(data: np.ndarray) -> np.ndarray:
    if not isinstance(data, np.ndarray) or data.dtype != np.float32:
        raise CoordinatePreservingCropError("CT normalization requires a float32 NumPy array")
    if not np.isfinite(data).all():
        raise CoordinatePreservingCropError("CT contains non-finite values")
    np.clip(data, CT_MIN_HU, CT_MAX_HU, out=data)
    data -= np.float32(CT_MIN_HU)
    data *= np.float32(
        (INTENSITY_OUTPUT_MAX - INTENSITY_OUTPUT_MIN) / (CT_MAX_HU - CT_MIN_HU)
    )
    data += np.float32(INTENSITY_OUTPUT_MIN - INTENSITY_BIAS)
    return data


def _padding_max_error(
    data_zyx: np.ndarray,
    lower_xyz: Sequence[int],
    upper_xyz: Sequence[int],
    expected: float,
) -> tuple[float, int]:
    lower = np.asarray(lower_xyz, dtype=np.int64)[::-1]
    upper = np.asarray(upper_xyz, dtype=np.int64)[::-1]
    maximum = 0.0
    count = 0
    for axis in range(3):
        if lower[axis] > 0:
            slices = [slice(None)] * 3
            slices[axis] = slice(0, int(lower[axis]))
            values = data_zyx[tuple(slices)]
            maximum = max(maximum, float(np.max(np.abs(values - expected))))
            count += int(values.size)
        if upper[axis] > 0:
            slices = [slice(None)] * 3
            slices[axis] = slice(data_zyx.shape[axis] - int(upper[axis]), None)
            values = data_zyx[tuple(slices)]
            maximum = max(maximum, float(np.max(np.abs(values - expected))))
            count += int(values.size)
    return maximum, count


def _validate_source_identity(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    expected = plan["source_ct"]
    observed = file_identity(path)
    for key in ("path", "bytes", "sha256"):
        if observed[key] != expected[key]:
            raise CoordinatePreservingCropError(
                f"Source CT identity mismatch for {plan['scan_key']}:{key}"
            )
    return observed


def prepare_scan_from_plan(path: Path, scan_plan: dict[str, Any]) -> PreparedVolume:
    """Prepare one Stage 1 crop as a normalized CPU ``ZYX`` array."""

    import nibabel as nib
    import SimpleITK as sitk

    plan = _validate_scan_plan(scan_plan)
    path = Path(path).resolve()
    source_identity = _validate_source_identity(path, plan)
    source = plan["source_ct"]
    raw_shape = _as_vector(source["native_shape_xyz"], "native_shape_xyz", dtype=np.int64)
    raw_affine = _as_affine(source["affine"], "source affine")
    nib_image = nib.load(str(path))
    if tuple(int(value) for value in nib_image.shape[:3]) != tuple(raw_shape):
        raise CoordinatePreservingCropError(f"Nibabel source shape mismatch in {plan['scan_key']}")
    if not np.allclose(nib_image.affine, raw_affine, atol=AFFINE_ATOL, rtol=0.0):
        raise CoordinatePreservingCropError(f"Nibabel source affine mismatch in {plan['scan_key']}")

    started = time.perf_counter()
    image = sitk.ReadImage(str(path))
    if tuple(int(value) for value in image.GetSize()) != tuple(raw_shape):
        raise CoordinatePreservingCropError(f"SimpleITK source shape mismatch in {plan['scan_key']}")
    actual_raw_affine = sitk_geometry_to_ras_affine(image)
    if not np.allclose(actual_raw_affine, raw_affine, atol=AFFINE_ATOL, rtol=0.0):
        raise CoordinatePreservingCropError(f"SimpleITK source affine mismatch in {plan['scan_key']}")

    start = _as_vector(plan["crop_start_xyz"], "crop_start_xyz", dtype=np.int64)
    end = _as_vector(plan["crop_end_xyz"], "crop_end_xyz", dtype=np.int64)
    crop_shape = end - start
    cropped = sitk.RegionOfInterest(
        image,
        size=[int(value) for value in crop_shape],
        index=[int(value) for value in start],
    )
    if tuple(int(value) for value in cropped.GetSize()) != tuple(crop_shape):
        raise CoordinatePreservingCropError(f"Actual crop shape mismatch in {plan['scan_key']}")
    actual_crop_affine = sitk_geometry_to_ras_affine(cropped)
    expected_crop_affine = _as_affine(plan["native_crop_affine"], "native_crop_affine")
    if not np.allclose(actual_crop_affine, expected_crop_affine, atol=AFFINE_ATOL, rtol=0.0):
        raise CoordinatePreservingCropError(f"Actual crop affine mismatch in {plan['scan_key']}")

    target_shape = _as_vector(plan["target_shape_xyz"], "target_shape_xyz", dtype=np.int64)
    expected_resampled_affine = _as_affine(
        plan["resampled_2mm_affine"], "resampled_2mm_affine"
    )
    spacing, origin, direction = ras_affine_to_sitk_geometry(expected_resampled_affine)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(value) for value in target_shape])
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(PADDING_HU))
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampled = resampler.Execute(cropped)
    actual_resampled_affine = sitk_geometry_to_ras_affine(resampled)
    if tuple(int(value) for value in resampled.GetSize()) != tuple(target_shape):
        raise CoordinatePreservingCropError(f"Actual target shape mismatch in {plan['scan_key']}")
    if not np.allclose(
        actual_resampled_affine, expected_resampled_affine, atol=AFFINE_ATOL, rtol=0.0
    ):
        raise CoordinatePreservingCropError(f"Actual resampled affine mismatch in {plan['scan_key']}")

    lower = _as_vector(plan["padding_lower_xyz"], "padding_lower_xyz", dtype=np.int64)
    upper = _as_vector(plan["padding_upper_xyz"], "padding_upper_xyz", dtype=np.int64)
    padded = sitk.ConstantPad(
        resampled,
        [int(value) for value in lower],
        [int(value) for value in upper],
        float(PADDING_HU),
    )
    expected_model_affine = _as_affine(plan["padded_2mm_affine"], "padded_2mm_affine")
    actual_model_affine = sitk_geometry_to_ras_affine(padded)
    padded_shape = _as_vector(plan["padded_shape_xyz"], "padded_shape_xyz", dtype=np.int64)
    if tuple(int(value) for value in padded.GetSize()) != tuple(padded_shape):
        raise CoordinatePreservingCropError(f"Actual padded shape mismatch in {plan['scan_key']}")
    if np.any(padded_shape % np.asarray(MODEL_STRIDE_XYZ, dtype=np.int64)):
        raise CoordinatePreservingCropError(f"Actual shape is not stride-compatible in {plan['scan_key']}")
    if not np.allclose(actual_model_affine, expected_model_affine, atol=AFFINE_ATOL, rtol=0.0):
        raise CoordinatePreservingCropError(f"Actual padded affine mismatch in {plan['scan_key']}")

    data_zyx = sitk.GetArrayFromImage(padded)
    if data_zyx.dtype != np.float32:
        data_zyx = data_zyx.astype(np.float32, copy=False)
    if data_zyx.shape != tuple(int(value) for value in padded_shape[::-1]):
        raise CoordinatePreservingCropError(f"Actual model array order mismatch in {plan['scan_key']}")
    if not data_zyx.flags.c_contiguous:
        data_zyx = np.ascontiguousarray(data_zyx)
    hu_min = float(data_zyx.min())
    hu_max = float(data_zyx.max())
    hu_padding_error, padded_value_count = _padding_max_error(data_zyx, lower, upper, PADDING_HU)
    padding_expected = bool(np.any(lower > 0) or np.any(upper > 0))
    if (padding_expected and padded_value_count <= 0) or hu_padding_error > 1e-6:
        raise CoordinatePreservingCropError(f"HU padding validation failed in {plan['scan_key']}")
    normalize_ct_inplace(data_zyx)
    normalized_padding_error, _ = _padding_max_error(data_zyx, lower, upper, -50.0)
    if (
        normalized_padding_error > 1e-6
        or not np.isfinite(data_zyx).all()
        or float(data_zyx.min()) < -50.0 - 1e-5
        or float(data_zyx.max()) > 205.0 + 1e-5
    ):
        raise CoordinatePreservingCropError(f"Normalized padding validation failed in {plan['scan_key']}")

    raw_to_model = np.linalg.inv(actual_model_affine) @ actual_raw_affine
    model_to_raw = np.linalg.inv(raw_to_model)
    expected_raw_to_model = _as_affine(
        plan["raw_to_model_continuous_affine"], "raw_to_model_continuous_affine"
    )
    expected_model_to_raw = _as_affine(
        plan["model_to_raw_continuous_affine"], "model_to_raw_continuous_affine"
    )
    if not np.allclose(raw_to_model, expected_raw_to_model, atol=AFFINE_ATOL, rtol=0.0):
        raise CoordinatePreservingCropError(f"Actual raw-to-model transform mismatch in {plan['scan_key']}")
    if not np.allclose(model_to_raw, expected_model_to_raw, atol=AFFINE_ATOL, rtol=0.0):
        raise CoordinatePreservingCropError(f"Actual model-to-raw transform mismatch in {plan['scan_key']}")

    metadata = {
        "scan_key": plan["scan_key"],
        "subject_id": plan["subject_id"],
        "session": plan["session"],
        "source_ct": source_identity,
        "raw_shape_xyz": raw_shape.tolist(),
        "crop_start_xyz": start.tolist(),
        "crop_end_xyz": end.tolist(),
        "crop_shape_xyz": crop_shape.tolist(),
        "target_shape_xyz": target_shape.tolist(),
        "padding_lower_xyz": lower.tolist(),
        "padding_upper_xyz": upper.tolist(),
        "padded_shape_xyz": padded_shape.tolist(),
        "data_shape_zyx": list(data_zyx.shape),
        "tensor_shape_ncdhw": [1, 1, *data_zyx.shape],
        "dtype": str(data_zyx.dtype),
        "raw_affine_ras": actual_raw_affine.tolist(),
        "crop_affine_ras": actual_crop_affine.tolist(),
        "resampled_affine_ras": actual_resampled_affine.tolist(),
        "model_affine_ras": actual_model_affine.tolist(),
        "raw_to_model_xyz": raw_to_model.tolist(),
        "model_to_raw_xyz": model_to_raw.tolist(),
        "hu_min_before_normalization": hu_min,
        "hu_max_before_normalization": hu_max,
        "normalized_min": float(data_zyx.min()),
        "normalized_max": float(data_zyx.max()),
        "padded_value_count_with_overlap": padded_value_count,
        "hu_padding_max_error": hu_padding_error,
        "normalized_padding_max_error": normalized_padding_error,
        "seconds": float(time.perf_counter() - started),
        "cuda_used": False,
        "model_loaded": False,
    }
    return PreparedVolume(
        data_zyx=data_zyx,
        raw_affine_ras=actual_raw_affine,
        model_affine_ras=actual_model_affine,
        raw_to_model_xyz=raw_to_model,
        model_to_raw_xyz=model_to_raw,
        metadata=metadata,
    )


def deterministic_raw_points(scan_plan: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    start = _as_vector(scan_plan["crop_start_xyz"], "crop_start_xyz", dtype=float)
    end_inclusive = _as_vector(scan_plan["crop_end_xyz"], "crop_end_xyz", dtype=float) - 1.0
    points: list[tuple[str, np.ndarray]] = []
    for x_index, x in enumerate((start[0], end_inclusive[0])):
        for y_index, y in enumerate((start[1], end_inclusive[1])):
            for z_index, z in enumerate((start[2], end_inclusive[2])):
                points.append((f"corner_{x_index}{y_index}{z_index}", np.array([x, y, z])))
    extent = end_inclusive - start
    for name, fraction in (("quarter", 0.25), ("centre", 0.5), ("three_quarter", 0.75)):
        points.append((name, start + fraction * extent))
    return points


def coordinate_check_rows(prepared: PreparedVolume, scan_plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw_shape = _as_vector(
        scan_plan["source_ct"]["native_shape_xyz"], "native_shape_xyz", dtype=np.int64
    )
    model_shape = _as_vector(scan_plan["padded_shape_xyz"], "padded_shape_xyz", dtype=float)
    rows = []
    for name, raw in deterministic_raw_points(scan_plan):
        model = apply_affine_xyz(raw, prepared.raw_to_model_xyz)
        recovered = apply_affine_xyz(model, prepared.model_to_raw_xyz)
        raw_error = float(np.max(np.abs(recovered - raw)))
        physical_before = apply_affine_xyz(raw, prepared.raw_affine_ras)
        physical_after = apply_affine_xyz(recovered, prepared.raw_affine_ras)
        physical_error = float(np.linalg.norm(physical_after - physical_before))
        if np.any(model < -1e-6) or np.any(model > model_shape - 1.0 + 1e-6):
            raise CoordinatePreservingCropError(
                f"Deterministic point {name} maps outside {scan_plan['scan_key']}"
            )
        rounded = nearest_raw_indices(recovered, raw_shape)
        rows.append(
            {
                "scan_key": scan_plan["scan_key"],
                "point_name": name,
                "raw_x": float(raw[0]),
                "raw_y": float(raw[1]),
                "raw_z": float(raw[2]),
                "model_x": float(model[0]),
                "model_y": float(model[1]),
                "model_z": float(model[2]),
                "recovered_raw_x": float(recovered[0]),
                "recovered_raw_y": float(recovered[1]),
                "recovered_raw_z": float(recovered[2]),
                "nearest_raw_x": int(rounded[0]),
                "nearest_raw_y": int(rounded[1]),
                "nearest_raw_z": int(rounded[2]),
                "max_raw_voxel_error": raw_error,
                "physical_error_mm": physical_error,
            }
        )
    return rows


def make_qc(prepared: PreparedVolume, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volume = prepared.data_zyx
    z, y, x = (size // 2 for size in volume.shape)
    views = (
        (volume[z], "axial"),
        (volume[:, y, :], "coronal"),
        (volume[:, :, x], "sagittal"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (image, title) in zip(axes, views):
        axis.imshow(image, cmap="gray", vmin=-50, vmax=80, origin="lower")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{prepared.metadata['scan_key']} — prepared xy_m010")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _scan_signature(selected_identity: dict[str, Any], plan: dict[str, Any]) -> str:
    value = {
        "preparation_id": PREPARATION_ID,
        "selected_sha256": selected_identity["sha256"],
        "scan_plan": plan,
        "normalization": {
            "input_hu": [CT_MIN_HU, CT_MAX_HU],
            "output": [INTENSITY_OUTPUT_MIN, INTENSITY_OUTPUT_MAX],
            "bias": INTENSITY_BIAS,
        },
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_directory(
    storage_root: Path,
    output_root: Path,
    run_id: str | None,
    resume_run_directory: Path | None,
) -> tuple[Path, bool]:
    storage_root = Path(storage_root).resolve()
    output_root = Path(output_root).resolve()
    if not _is_within(output_root, storage_root):
        raise CoordinatePreservingCropError("Stage 2 output root escapes the storage root")
    if resume_run_directory is not None:
        run_directory = Path(resume_run_directory).resolve()
        if not _is_within(run_directory, output_root) or not run_directory.is_dir():
            raise CoordinatePreservingCropError("Resume directory is invalid or outside output root")
        return run_directory, True
    identifier = run_id or default_run_id()
    if not RUN_ID_PATTERN.fullmatch(identifier):
        raise CoordinatePreservingCropError("Stage 2 run id is invalid")
    run_directory = output_root / identifier
    if run_directory.exists():
        raise CoordinatePreservingCropError(f"Refusing to overwrite run directory: {run_directory}")
    run_directory.mkdir(parents=True)
    return run_directory, False


def _contract_core(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: contract[key]
        for key in (
            "schema_version",
            "preparation_id",
            "baseline_manifest",
            "stage1_checkpoint",
            "selected_body_envelope",
            "repository",
            "settings",
            "largest_pair",
            "scientific_computation",
        )
    }


def validate_resume_contract(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    if existing.get("status") == "passed" or "checkpoint_summary" in existing.get("outputs", {}):
        raise CoordinatePreservingCropError("Completed Stage 2 evidence is immutable")
    if _contract_core(existing) != _contract_core(expected):
        raise CoordinatePreservingCropError("Stage 2 resume contract changed")


def _scan_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result["metadata"]
    checks = result["checks"]
    return {
        "scan_key": metadata["scan_key"],
        "subject_id": metadata["subject_id"],
        "session": metadata["session"],
        "raw_shape_xyz": json.dumps(metadata["raw_shape_xyz"]),
        "crop_shape_xyz": json.dumps(metadata["crop_shape_xyz"]),
        "target_shape_xyz": json.dumps(metadata["target_shape_xyz"]),
        "padded_shape_xyz": json.dumps(metadata["padded_shape_xyz"]),
        "data_shape_zyx": json.dumps(metadata["data_shape_zyx"]),
        "tensor_shape_ncdhw": json.dumps(metadata["tensor_shape_ncdhw"]),
        "dtype": metadata["dtype"],
        "normalized_min": metadata["normalized_min"],
        "normalized_max": metadata["normalized_max"],
        "hu_padding_max_error": metadata["hu_padding_max_error"],
        "normalized_padding_max_error": metadata["normalized_padding_max_error"],
        "max_raw_voxel_roundtrip_error": checks["max_raw_voxel_roundtrip_error"],
        "max_physical_roundtrip_error_mm": checks["max_physical_roundtrip_error_mm"],
        "seconds": metadata["seconds"],
        "status": result["status"],
    }


def render_report(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Stage 2 coordinate-preserving crop validation",
        "",
        "## Outcome",
        "",
        "Both scans in the frozen largest Test/Retest pair passed the CPU geometry, padding, intensity, and coordinate checks.",
        "",
        "## Scans",
        "",
        "| Scan | Padded XYZ | Normalized range | Max voxel round trip | Max physical error (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        metadata = result["metadata"]
        checks = result["checks"]
        lines.append(
            "| {scan} | `{shape}` | {minimum:.6g} to {maximum:.6g} | {voxel:.3g} | {physical:.3g} |".format(
                scan=metadata["scan_key"],
                shape=metadata["padded_shape_xyz"],
                minimum=metadata["normalized_min"],
                maximum=metadata["normalized_max"],
                voxel=checks["max_raw_voxel_roundtrip_error"],
                physical=checks["max_physical_roundtrip_error_mm"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- The body envelope was not recalculated; every crop used the immutable Stage 1 `xy_m010` plan.",
            "- Arrays were processed sequentially and discarded after each scan result and QC image were written.",
            "- No UAE-S model, CUDA operation, embedding, matching, segmentation, or cycle-error computation ran.",
            "- No prepared 3D image or array was retained.",
            "",
            "Stage 3 must reuse this preparation interface and must separately verify compatibility with the legacy UAE loader before model benchmarking.",
            "",
        ]
    )
    return "\n".join(lines)


def _assert_no_full_volume_outputs(run_directory: Path) -> None:
    forbidden = {".nii", ".npy", ".npz"}
    for path in run_directory.rglob("*"):
        lower = path.name.lower()
        if path.is_file() and (path.suffix.lower() in forbidden or lower.endswith(".nii.gz")):
            raise CoordinatePreservingCropError(f"Forbidden full-volume output exists: {path}")


def run_validate(args: argparse.Namespace) -> Path:
    storage_root = Path(args.storage_root).resolve()
    repository_root = Path(args.repository_root).resolve()
    baseline_path = Path(args.baseline_manifest).resolve()
    stage1.verify_baseline_identity(baseline_path)
    baseline.validate_locked_contract(
        baseline_path,
        repository_root=repository_root,
        storage_root=storage_root,
        required_profile="preprocess",
    )
    stage1_contract = validate_stage1_contract(
        Path(args.stage1_checkpoint),
        repository_root=repository_root,
        storage_root=storage_root,
    )
    repository = baseline.inspect_repository(
        repository_root,
        expected_base_commit=EXPECTED_STAGE1_EXECUTION_COMMIT,
        expected_branch=baseline.EXPECTED_BRANCH,
    )
    output_root = Path(
        args.output_root or (storage_root / "runs/memory_optimization")
    ).resolve()
    if not _is_within(output_root, storage_root):
        raise CoordinatePreservingCropError("Stage 2 output root escapes the storage root")
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory, resuming = _run_directory(
        storage_root,
        output_root,
        args.run_id,
        args.resume_run_directory,
    )
    plans = sorted(
        stage1_contract["pair_scans"],
        key=lambda item: 0 if item["session"] == "test" else 1,
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "preparation_id": PREPARATION_ID,
        "status": "in_progress",
        "created_at": utc_now(),
        "baseline_manifest": file_identity(baseline_path),
        "stage1_checkpoint": stage1_contract["checkpoint_identity"],
        "selected_body_envelope": stage1_contract["selected_identity"],
        "repository": {
            "path": repository["path"],
            "branch": repository["branch"],
            "execution_commit": repository["execution_commit"],
            "clean": repository["clean"],
            "stage1_ancestor": EXPECTED_STAGE1_EXECUTION_COMMIT,
        },
        "settings": {
            "candidate_id": EXPECTED_CANDIDATE_ID,
            "target_spacing_xyz_mm": list(TARGET_SPACING_XYZ_MM),
            "model_stride_xyz": list(MODEL_STRIDE_XYZ),
            "crop_bounds": "raw_itk_voxel_half_open",
            "ct_interpolation": "SimpleITK_linear",
            "padding_policy": "symmetric_lower_floor_upper_remainder",
            "padding_hu": PADDING_HU,
            "array_order": "ZYX",
            "future_tensor_order": "NCDHW",
            "array_dtype": "float32",
            "normalization": {
                "input_hu": [CT_MIN_HU, CT_MAX_HU],
                "output": [INTENSITY_OUTPUT_MIN, INTENSITY_OUTPUT_MAX],
                "bias": INTENSITY_BIAS,
            },
            "inverse_coordinate_policy": "continuous_until_final_export",
            "integer_export_rounding": "numpy_rint_no_clipping",
        },
        "largest_pair": {
            "subject_id": EXPECTED_LARGEST_PAIR,
            "scan_keys": [plan["scan_key"] for plan in plans],
        },
        "scientific_computation": {
            "body_envelope_recalculated": False,
            "uae_model_loaded": False,
            "cuda_used": False,
            "embedding_generated": False,
            "matching_run": False,
            "segmentation_run": False,
            "cycle_error_generated": False,
            "full_3d_outputs_retained": False,
        },
    }
    manifest_path = run_directory / "stage2_manifest.json"
    if resuming:
        existing = load_json(manifest_path)
        if (run_directory / "checkpoint_summary.json").exists():
            raise CoordinatePreservingCropError(
                "A Stage 2 checkpoint already exists; completed evidence is immutable"
            )
        validate_resume_contract(existing, contract)
        contract = existing
    else:
        atomic_create_json(manifest_path, contract)

    log_path = run_directory / "stage2.log"
    result_directory = run_directory / "scan_results"
    qc_directory = run_directory / "qc"
    result_directory.mkdir(exist_ok=True)
    qc_directory.mkdir(exist_ok=True)
    emit(f"Starting Stage 2 validation; resume={resuming}", log_path)
    results = []
    coordinate_rows: list[dict[str, Any]] = []
    for index, plan in enumerate(plans, start=1):
        result_path = result_directory / f"{plan['scan_key']}.json"
        qc_path = qc_directory / f"{plan['scan_key']}.png"
        signature = _scan_signature(stage1_contract["selected_identity"], plan)
        if result_path.is_file():
            result = load_json(result_path)
            if result.get("status") != "PASS" or result.get("scan_signature") != signature:
                raise CoordinatePreservingCropError(f"Incompatible resumable result: {result_path}")
            _identity_matches(result.get("qc"), file_identity(qc_path), "resumable QC")
            emit(f"[{index}/2] Reusing {plan['scan_key']}", log_path)
        else:
            emit(f"[{index}/2] Preparing {plan['scan_key']}", log_path)
            prepared = prepare_scan_from_plan(Path(plan["source_ct"]["path"]), plan)
            rows = coordinate_check_rows(prepared, plan)
            max_voxel = max(row["max_raw_voxel_error"] for row in rows)
            max_physical = max(row["physical_error_mm"] for row in rows)
            if max_voxel > ROUNDTRIP_VOXEL_ATOL or max_physical > ROUNDTRIP_PHYSICAL_ATOL_MM:
                raise CoordinatePreservingCropError(
                    f"Coordinate round trip failed in {plan['scan_key']}"
                )
            make_qc(prepared, qc_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "preparation_id": PREPARATION_ID,
                "status": "PASS",
                "scan_signature": signature,
                "metadata": prepared.metadata,
                "coordinate_checks": rows,
                "checks": {
                    "source_identity_passed": True,
                    "source_geometry_passed": True,
                    "crop_geometry_passed": True,
                    "resampled_geometry_passed": True,
                    "padded_geometry_passed": True,
                    "stride_compatibility_passed": True,
                    "hu_padding_passed": True,
                    "normalized_padding_passed": True,
                    "finite_float32_passed": True,
                    "transform_contract_passed": True,
                    "max_raw_voxel_roundtrip_error": max_voxel,
                    "max_physical_roundtrip_error_mm": max_physical,
                    "prepared_volume_retained": False,
                },
                "qc": file_identity(qc_path),
            }
            atomic_create_json(result_path, result)
            del prepared
            gc.collect()
        results.append(result)
        coordinate_rows.extend(result["coordinate_checks"])

    scan_rows = [_scan_summary_row(result) for result in results]
    write_csv(run_directory / "scan_preparation_summary.csv", scan_rows)
    write_csv(run_directory / "coordinate_roundtrip.csv", coordinate_rows)
    atomic_replace_text(run_directory / "prepared_geometry_report.md", render_report(results))
    _assert_no_full_volume_outputs(run_directory)
    final_repository = baseline.inspect_repository(
        repository_root,
        expected_base_commit=EXPECTED_STAGE1_EXECUTION_COMMIT,
        expected_branch=baseline.EXPECTED_BRANCH,
    )
    if final_repository["execution_commit"] != repository["execution_commit"]:
        raise CoordinatePreservingCropError(
            "Repository commit changed while Stage 2 was running"
        )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "stage": 2,
        "status": "PASS",
        "created_at": utc_now(),
        "preparation_id": PREPARATION_ID,
        "selected_candidate": EXPECTED_CANDIDATE_ID,
        "largest_test_retest_pair": EXPECTED_LARGEST_PAIR,
        "selected_body_envelope": stage1_contract["selected_identity"],
        "gates": {
            "stage0_contract_validated": True,
            "stage1_selection_validated": True,
            "stage1_types_strictly_parsed": True,
            "largest_pair_test_passed": True,
            "largest_pair_retest_passed": True,
            "source_identity_and_geometry_passed": True,
            "crop_resample_padding_geometry_passed": True,
            "stride_compatibility_passed": True,
            "intensity_and_padding_passed": True,
            "coordinate_roundtrip_passed": True,
            "continuous_inverse_coordinates_preserved": True,
            "repository_remained_clean": True,
            "model_or_cuda_computation_launched": False,
            "scientific_results_generated": False,
            "full_prepared_volumes_retained": False,
        },
        "next_stage": "uae_loader_compatibility_and_memory_feasibility",
    }
    checkpoint_path = run_directory / "checkpoint_summary.json"
    atomic_create_json(checkpoint_path, checkpoint)
    emit("Stage 2 PASS", log_path)
    emit(f"Run directory: {run_directory}", log_path)
    emit(f"Largest Test/Retest pair: {EXPECTED_LARGEST_PAIR}", log_path)
    emit(f"Checkpoint: {checkpoint_path}", log_path)
    outputs = {
        name: file_identity(run_directory / name)
        for name in (
            "scan_preparation_summary.csv",
            "coordinate_roundtrip.csv",
            "prepared_geometry_report.md",
            "stage2.log",
            "checkpoint_summary.json",
        )
    }
    outputs["scan_results"] = {
        result["metadata"]["scan_key"]: file_identity(
            result_directory / f"{result['metadata']['scan_key']}.json"
        )
        for result in results
    }
    outputs["qc"] = {
        result["metadata"]["scan_key"]: file_identity(
            qc_directory / f"{result['metadata']['scan_key']}.png"
        )
        for result in results
    }
    contract.update({"status": "passed", "completed_at": utc_now(), "outputs": outputs})
    atomic_replace_json(manifest_path, contract)
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="Prepare and validate the frozen largest Test/Retest pair."
    )
    validate.add_argument(
        "--baseline-manifest", type=Path, default=stage1.EXPECTED_BASELINE_PATH
    )
    validate.add_argument(
        "--stage1-checkpoint", type=Path, default=EXPECTED_STAGE1_CHECKPOINT_PATH
    )
    validate.add_argument("--storage-root", type=Path, default=Path("/workspace/quadra"))
    validate.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    validate.add_argument("--output-root", type=Path, default=None)
    validate.add_argument("--run-id", default=None)
    validate.add_argument("--resume-run-directory", type=Path, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.run_id and args.resume_run_directory:
            raise CoordinatePreservingCropError(
                "Use either --run-id or --resume-run-directory, not both"
            )
        run_validate(args)
    except (
        CoordinatePreservingCropError,
        stage1.BodyEnvelopeAuditError,
        baseline.OptimizationBaselineError,
    ) as exc:
        parser.exit(2, f"Stage 2 failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
