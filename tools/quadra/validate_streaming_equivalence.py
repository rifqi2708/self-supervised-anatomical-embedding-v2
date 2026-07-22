#!/usr/bin/env python3
"""Validate Quadra tiled embeddings and exhaustive streamed matching.

The command separates numerical matching correctness from encoder tile-context
sensitivity. Dense inference is restricted to deterministic organ-centred
crops; the full 2 mm subject is evaluated only with memory-bounded tiles.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra.coord_space_utils import (  # noqa: E402
    COORD_SPACE_RAW_ITK,
    COORD_SPACE_SAM,
    build_sam_to_raw_transform,
    transform_point_xyz,
)
from tools.quadra.streaming_cycle_error import (  # noqa: E402
    DEFAULT_CHECKPOINT_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MATCH_CHUNK_XYZ,
    DEFAULT_QUERY_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_SUBJECT,
    EmbeddingCache,
    build_embedding_cache,
    canonical_subject_id,
    extract_query_descriptors,
    file_identity,
    load_complete_manifest,
    model_module_and_device,
    resolve_subject_pair,
    sample_subject_points,
    stream_global_match,
    unwrap_model_input,
    utc_now,
    write_json,
)
from tools.quadra.streaming_embedding import (  # noqa: E402
    COARSE_STRIDE_XYZ,
    FINE_STRIDE_XYZ,
    TilePlan,
    build_tile_plan,
    iter_tile_locations,
)


DEFAULT_OUTPUT_ROOT = "data/quadra_output/streaming_validation"
DEFAULT_CACHE_ROOT = "data/quadra_streaming_validation_cache"
DEFAULT_DENSE_CROP_SIZE_XYZ = (128, 128, 64)
DEFAULT_BASELINE_TILE_SIZE_XYZ = (128, 128, 64)
DEFAULT_BASELINE_HALO_XYZ = (32, 32, 16)
DEFAULT_EXPANDED_TILE_SIZE_XYZ = (160, 160, 80)
DEFAULT_EXPANDED_HALO_XYZ = (48, 48, 24)
DEFAULT_CROP_POINTS_PER_ORGAN = 20
DEFAULT_ORGANS = ("bladder", "colon", "kidney", "liver", "lungs")
MATCH_SCORE_TOLERANCE = 1e-5
DESCRIPTOR_MEDIAN_COSINE_MIN = 0.99
DESCRIPTOR_P01_COSINE_MIN = 0.95
DESCRIPTOR_SEAM_DROP_MAX = 0.01
CROP_MATCH_WITHIN_MM = 2.0
CROP_MATCH_RATE_MIN = 0.95
CYCLE_MEDIAN_DELTA_MAX_MM = 1.0
CYCLE_P95_DELTA_MAX_MM = 4.0
FULL_MATCH_MEDIAN_MAX_MM = 2.0
FULL_MATCH_P95_MAX_MM = 4.0


@dataclass
class ArrayEmbeddingCache:
    """In-memory cache with the interface required by production matching."""

    fine: np.ndarray
    coarse: np.ndarray
    native_shape_xyz_value: tuple[int, int, int]
    norm_ratio_xyz_value: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def feature_shape_xyz(self, level: str) -> tuple[int, int, int]:
        array = self.fine if level == "fine" else self.coarse
        return int(array.shape[3]), int(array.shape[2]), int(array.shape[1])

    @property
    def native_shape_xyz(self) -> tuple[int, int, int]:
        return self.native_shape_xyz_value

    @property
    def norm_ratio_xyz(self) -> np.ndarray:
        return np.asarray(self.norm_ratio_xyz_value, dtype=np.float64)

    def valid_array(self, level: str):
        return self.fine if level == "fine" else self.coarse


def _triple(values: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three x,y,z values")
    result = tuple(int(value) for value in values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def percentile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, quantile))


def deterministic_crop_start(
    mask_zyx: np.ndarray,
    crop_size_xyz: Sequence[int],
) -> tuple[int, int, int]:
    """Return a clamped crop start centred on the median positive mask voxel."""
    mask = np.asarray(mask_zyx)
    if mask.ndim != 3:
        raise ValueError(f"mask_zyx must be 3D, got shape {mask.shape}")
    positive_zyx = np.argwhere(mask > 0)
    if positive_zyx.size == 0:
        raise ValueError("Cannot centre a crop on an empty mask")
    crop_xyz = np.asarray(_triple(crop_size_xyz, "crop_size_xyz"), dtype=np.int64)
    shape_xyz = np.asarray((mask.shape[2], mask.shape[1], mask.shape[0]), dtype=np.int64)
    if np.any(crop_xyz > shape_xyz):
        raise ValueError(f"Crop {tuple(crop_xyz)} exceeds volume {tuple(shape_xyz)}")
    centre_xyz = np.median(positive_zyx[:, ::-1], axis=0)
    start = np.floor(centre_xyz - crop_xyz / 2.0).astype(np.int64)
    start = np.clip(start, 0, shape_xyz - crop_xyz)
    return tuple(int(value) for value in start)


def crop_slices_zyx(start_xyz: Sequence[int], size_xyz: Sequence[int]):
    start_x, start_y, start_z = _triple_or_zero(start_xyz, "start_xyz")
    size_x, size_y, size_z = _triple(size_xyz, "size_xyz")
    return (
        slice(start_z, start_z + size_z),
        slice(start_y, start_y + size_y),
        slice(start_x, start_x + size_x),
    )


def _triple_or_zero(values: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three x,y,z values")
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} values must be non-negative, got {result}")
    return result


def internal_seam_distance(
    points_xyz: np.ndarray,
    shape_xyz: Sequence[int],
    core_size_xyz: Sequence[int],
) -> np.ndarray:
    """Distance in grid units to the nearest internal retained-core boundary."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    shape = np.asarray(_triple(shape_xyz, "shape_xyz"), dtype=np.int64)
    core = np.asarray(_triple(core_size_xyz, "core_size_xyz"), dtype=np.int64)
    per_axis = []
    for axis in range(3):
        boundaries = np.arange(core[axis], shape[axis], core[axis], dtype=np.float64)
        if boundaries.size == 0:
            per_axis.append(np.full(len(points), np.inf, dtype=np.float64))
        else:
            per_axis.append(np.min(np.abs(points[:, axis, None] - boundaries[None, :]), axis=1))
    return np.min(np.stack(per_axis, axis=1), axis=1)


def seam_distance_mm_for_native_points(
    points_xyz: np.ndarray,
    norm_ratio_xyz: Sequence[float],
    resampled_shape_xyz: Sequence[int],
    core_size_xyz: Sequence[int],
) -> np.ndarray:
    points_resampled = np.asarray(points_xyz, dtype=np.float64) * np.asarray(norm_ratio_xyz, dtype=np.float64)
    return internal_seam_distance(points_resampled, resampled_shape_xyz, core_size_xyz) * 2.0


def descriptor_cosine_and_error(reference: np.ndarray, candidate: np.ndarray):
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError(f"Descriptor arrays must share [c,z,y,x] shape, got {reference.shape}, {candidate.shape}")
    dot = np.sum(reference * candidate, axis=0, dtype=np.float32)
    norm_ref = np.sqrt(np.sum(reference * reference, axis=0, dtype=np.float32))
    norm_candidate = np.sqrt(np.sum(candidate * candidate, axis=0, dtype=np.float32))
    denominator = np.maximum(norm_ref * norm_candidate, 1e-12)
    cosine = np.clip(dot / denominator, -1.0, 1.0)
    l2 = np.sqrt(np.sum((reference - candidate) ** 2, axis=0, dtype=np.float32))
    mean_abs = np.mean(np.abs(reference - candidate), axis=0, dtype=np.float32)
    return cosine, l2, mean_abs, norm_ref, norm_candidate


def feature_region_masks(
    shape_xyz: Sequence[int],
    core_size_xyz: Sequence[int],
    margin_xyz: Sequence[int],
) -> dict[str, np.ndarray]:
    shape_x, shape_y, shape_z = _triple(shape_xyz, "shape_xyz")
    core_x, core_y, core_z = _triple(core_size_xyz, "core_size_xyz")
    margin_x, margin_y, margin_z = _triple_or_zero(margin_xyz, "margin_xyz")
    z, y, x = np.meshgrid(
        np.arange(shape_z), np.arange(shape_y), np.arange(shape_x), indexing="ij"
    )
    points = np.stack((x.ravel() + 0.5, y.ravel() + 0.5, z.ravel() + 0.5), axis=1)
    distance = internal_seam_distance(points, (shape_x, shape_y, shape_z), (core_x, core_y, core_z)).reshape(
        shape_z, shape_y, shape_x
    )
    eligible = (
        (x + 0.5 >= margin_x)
        & (x + 0.5 < shape_x - margin_x)
        & (y + 0.5 >= margin_y)
        & (y + 0.5 < shape_y - margin_y)
        & (z + 0.5 >= margin_z)
        & (z + 0.5 < shape_z - margin_z)
    )
    seam = eligible & (distance <= 1.0)
    interior = eligible & ~seam
    return {"all": eligible, "seam": seam, "interior": interior}


def descriptor_summary_rows(
    reference: np.ndarray,
    candidate: np.ndarray,
    plan: TilePlan,
    level: str,
    metadata: dict[str, object],
) -> tuple[list[dict[str, object]], np.ndarray]:
    stride = FINE_STRIDE_XYZ if level == "fine" else COARSE_STRIDE_XYZ
    shape_xyz = (reference.shape[3], reference.shape[2], reference.shape[1])
    core_feature = tuple(plan.core_size_xyz[axis] // stride[axis] for axis in range(3))
    margin_feature = tuple(plan.halo_xyz[axis] // stride[axis] for axis in range(3))
    regions = feature_region_masks(shape_xyz, core_feature, margin_feature)
    cosine, l2, mean_abs, norm_ref, norm_candidate = descriptor_cosine_and_error(reference, candidate)
    rows = []
    for region_name, region_mask in regions.items():
        selected_cosine = cosine[region_mask]
        selected_l2 = l2[region_mask]
        selected_abs = mean_abs[region_mask]
        selected_ref_norm = norm_ref[region_mask]
        selected_candidate_norm = norm_candidate[region_mask]
        rows.append(
            {
                **metadata,
                "level": level,
                "region": region_name,
                "count": int(selected_cosine.size),
                "cosine_median": percentile(selected_cosine, 50),
                "cosine_p01": percentile(selected_cosine, 1),
                "cosine_p05": percentile(selected_cosine, 5),
                "l2_median": percentile(selected_l2, 50),
                "l2_p95": percentile(selected_l2, 95),
                "mean_abs_median": percentile(selected_abs, 50),
                "reference_norm_median": percentile(selected_ref_norm, 50),
                "candidate_norm_median": percentile(selected_candidate_norm, 50),
            }
        )
    return rows, 1.0 - cosine


def select_mask_points_in_crop(
    mask_crop_zyx: np.ndarray,
    halo_xyz: Sequence[int],
    count: int,
    seed: int,
) -> np.ndarray:
    halo_x, halo_y, halo_z = _triple(halo_xyz, "halo_xyz")
    eligible = np.asarray(mask_crop_zyx) > 0
    interior = np.zeros_like(eligible, dtype=bool)
    z_stop = eligible.shape[0] - halo_z
    y_stop = eligible.shape[1] - halo_y
    x_stop = eligible.shape[2] - halo_x
    interior[halo_z:z_stop, halo_y:y_stop, halo_x:x_stop] = True
    candidates_zyx = np.argwhere(eligible & interior)
    if len(candidates_zyx) < count:
        raise ValueError(f"Only {len(candidates_zyx)} interior mask voxels are available; need {count}")
    rng = np.random.default_rng(int(seed))
    selected = candidates_zyx[rng.choice(len(candidates_zyx), size=count, replace=False)]
    return selected[:, ::-1].astype(np.int64)


def dense_global_match(
    query_cache: ArrayEmbeddingCache,
    target_cache: ArrayEmbeddingCache,
    query_points_xyz: np.ndarray,
    query_batch_size: int,
    device,
):
    """Dense reference matcher using the entire target similarity volume."""
    import torch
    import torch.nn.functional as torch_f

    device = torch.device(device)
    points = np.asarray(query_points_xyz, dtype=np.int64)
    query_fine, query_coarse, _ = extract_query_descriptors(query_cache, points, device)
    target_fine = torch.from_numpy(np.asarray(target_cache.valid_array("fine"), dtype=np.float32)).to(device)
    target_coarse = torch.from_numpy(np.asarray(target_cache.valid_array("coarse"), dtype=np.float32)).unsqueeze(0).to(device)
    fine_shape_zyx = tuple(int(value) for value in target_fine.shape[1:])
    coarse_at_fine = torch_f.interpolate(target_coarse, fine_shape_zyx, mode="trilinear", align_corners=False)
    coarse_at_fine = torch_f.normalize(coarse_at_fine, dim=1)[0]
    fine_flat = target_fine.reshape(target_fine.shape[0], -1)
    coarse_flat = coarse_at_fine.reshape(coarse_at_fine.shape[0], -1)
    native_shape_xyz = target_cache.native_shape_xyz
    native_size_zyx = (native_shape_xyz[2], native_shape_xyz[1], native_shape_xyz[0])
    best_points = np.zeros((len(points), 3), dtype=np.int64)
    best_scores = np.full(len(points), -np.inf, dtype=np.float32)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(points), int(query_batch_size)):
        stop = min(start + int(query_batch_size), len(points))
        sim_fine = torch.matmul(query_fine[start:stop], fine_flat)
        sim_coarse = torch.matmul(query_coarse[start:stop], coarse_flat)
        sim = ((sim_fine + sim_coarse) * 0.5).reshape(stop - start, 1, *fine_shape_zyx)
        sim_native = torch_f.interpolate(sim, native_size_zyx, mode="trilinear", align_corners=False).reshape(
            stop - start, -1
        )
        values, indices = torch.max(sim_native, dim=1)
        flat = indices.detach().cpu().numpy().astype(np.int64)
        x = flat % native_shape_xyz[0]
        y = (flat // native_shape_xyz[0]) % native_shape_xyz[1]
        z = flat // (native_shape_xyz[0] * native_shape_xyz[1])
        best_points[start:stop] = np.stack((x, y, z), axis=1)
        best_scores[start:stop] = values.detach().cpu().numpy().astype(np.float32)
        del sim_fine, sim_coarse, sim, sim_native, values, indices
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return best_points, best_scores, {"seconds": time.time() - started, "peak_gpu_memory_bytes": peak}


def extract_dense_embeddings(volume, model) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import torch

    module, device = model_module_and_device(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    with torch.no_grad():
        fine, coarse = module.extract_feat(volume.to(device=device, non_blocking=True))
    profile = {
        "seconds": float(time.time() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    fine_np = fine[0].detach().cpu().float().numpy()
    coarse_np = coarse[0].detach().cpu().float().numpy()
    del fine, coarse
    torch.cuda.empty_cache()
    return fine_np, coarse_np, profile


def extract_tiled_embeddings(
    volume,
    model,
    tile_size_xyz: Sequence[int],
    halo_xyz: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, TilePlan, dict[str, object]]:
    import torch
    import torch.nn.functional as torch_f

    module, device = model_module_and_device(model)
    shape_xyz = (int(volume.shape[4]), int(volume.shape[3]), int(volume.shape[2]))
    plan = build_tile_plan(shape_xyz, tile_size_xyz=tile_size_xyz, halo_xyz=halo_xyz)
    tail_xyz = tuple(plan.grid_shape_xyz[axis] * plan.core_size_xyz[axis] - shape_xyz[axis] for axis in range(3))
    halo_x, halo_y, halo_z = plan.halo_xyz
    padded = torch_f.pad(
        volume,
        (
            halo_x,
            halo_x + tail_xyz[0],
            halo_y,
            halo_y + tail_xyz[1],
            halo_z,
            halo_z + tail_xyz[2],
        ),
        mode="constant",
        value=0.0,
    )
    fine_map = None
    coarse_map = None
    peak = 0
    started = time.time()
    for location in iter_tile_locations(plan):
        tile = padded[(slice(None), slice(None), *location.padded_input_slices_zyx)].to(
            device=device, non_blocking=True
        )
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            fine, coarse = module.extract_feat(tile)
        peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        if fine_map is None:
            fine_map = np.empty((int(fine.shape[1]), *reversed(plan.stored_fine_shape_xyz)), dtype=np.float32)
            coarse_map = np.empty((int(coarse.shape[1]), *reversed(plan.stored_coarse_shape_xyz)), dtype=np.float32)
        fine_map[(slice(None), *location.fine_destination_slices_zyx)] = (
            fine[(0, slice(None), *location.fine_source_slices_zyx)].detach().cpu().float().numpy()
        )
        coarse_map[(slice(None), *location.coarse_destination_slices_zyx)] = (
            coarse[(0, slice(None), *location.coarse_source_slices_zyx)].detach().cpu().float().numpy()
        )
        del tile, fine, coarse
    valid_fine = tuple(reversed(plan.valid_fine_shape_xyz))
    valid_coarse = tuple(reversed(plan.valid_coarse_shape_xyz))
    fine_map = fine_map[:, : valid_fine[0], : valid_fine[1], : valid_fine[2]]
    coarse_map = coarse_map[:, : valid_coarse[0], : valid_coarse[1], : valid_coarse[2]]
    del padded
    torch.cuda.empty_cache()
    return fine_map, coarse_map, plan, {
        "seconds": float(time.time() - started),
        "peak_gpu_memory_bytes": int(peak),
        "tile_plan": plan.to_dict(),
    }


def save_discrepancy_heatmap(error_zyx: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = np.asarray(error_zyx, dtype=np.float32)
    projections = (
        (np.max(error, axis=0), "max over z"),
        (np.max(error, axis=1), "max over y"),
        (np.max(error, axis=2), "max over x"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    vmax = float(np.percentile(error, 99)) if error.size else 1.0
    vmax = max(vmax, 1e-6)
    image = None
    for axis, (projection, label) in zip(axes, projections):
        image = axis.imshow(projection, cmap="magma", vmin=0.0, vmax=vmax, origin="lower")
        axis.set_title(label)
        axis.set_axis_off()
    figure.suptitle(title)
    figure.colorbar(image, ax=axes, shrink=0.8, label="1 - cosine similarity")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replace_phase_rows(path: Path, phase: str, new_rows: list[dict[str, object]]) -> None:
    retained = [row for row in read_csv(path) if row.get("phase") != phase]
    write_csv(path, retained + new_rows)


def _number(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def gate(name: str, passed: bool | None, value, threshold: str, detail: str = "") -> dict[str, object]:
    return {
        "name": name,
        "status": "blocked" if passed is None else ("pass" if passed else "fail"),
        "value": value,
        "threshold": threshold,
        "detail": detail,
    }


def evaluate_matcher_gates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return [gate("streaming_matcher_equivalence", None, None, "100% coordinates and score delta <= 1e-5")]
    coordinate_rate = float(np.mean([str(row["coordinate_match"]).lower() in ("true", "1") for row in rows]))
    max_score = max(_number(row, "score_abs_diff") for row in rows)
    return [
        gate("streaming_matcher_coordinates", coordinate_rate == 1.0, coordinate_rate, "1.0"),
        gate("streaming_matcher_scores", max_score <= MATCH_SCORE_TOLERANCE, max_score, "<= 1e-5"),
    ]


def evaluate_descriptor_gates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    relevant = [
        row
        for row in rows
        if row.get("phase") == "crop"
        and row.get("comparison") == "dense_vs_tiled_fp16"
        and row.get("region") == "all"
    ]
    if not relevant:
        return [gate("crop_descriptor_equivalence", None, None, "descriptor metrics available")]
    worst_median = min(_number(row, "cosine_median") for row in relevant)
    worst_p01 = min(_number(row, "cosine_p01") for row in relevant)
    seam_lookup = {}
    interior_lookup = {}
    for row in rows:
        if row.get("phase") != "crop" or row.get("comparison") != "dense_vs_tiled_fp16":
            continue
        key = (row.get("timepoint"), row.get("organ"), row.get("level"))
        if row.get("region") == "seam":
            seam_lookup[key] = _number(row, "cosine_median")
        elif row.get("region") == "interior":
            interior_lookup[key] = _number(row, "cosine_median")
    drops = [interior_lookup[key] - value for key, value in seam_lookup.items() if key in interior_lookup]
    worst_drop = max(drops) if drops else float("nan")
    return [
        gate("descriptor_median_cosine", worst_median >= DESCRIPTOR_MEDIAN_COSINE_MIN, worst_median, ">= 0.99"),
        gate("descriptor_p01_cosine", worst_p01 >= DESCRIPTOR_P01_COSINE_MIN, worst_p01, ">= 0.95"),
        gate("descriptor_seam_drop", worst_drop <= DESCRIPTOR_SEAM_DROP_MAX, worst_drop, "<= 0.01"),
    ]


def evaluate_correspondence_gates(rows: list[dict[str, object]], phase: str) -> list[dict[str, object]]:
    relevant = [row for row in rows if row.get("phase") == phase]
    if phase == "crop":
        relevant = [
            row
            for row in relevant
            if row.get("comparison") in (None, "", "dense_vs_baseline_tiled")
        ]
    if not relevant:
        return [gate(f"{phase}_correspondence", None, None, "correspondence metrics available")]
    forward = np.asarray([_number(row, "forward_displacement_mm") for row in relevant])
    backward = np.asarray([_number(row, "backward_displacement_mm") for row in relevant])
    cycle_delta = np.asarray([_number(row, "cycle_error_abs_delta_mm") for row in relevant])
    both = np.concatenate((forward, backward))
    if phase == "crop":
        within_rate = float(np.mean(both <= CROP_MATCH_WITHIN_MM))
        return [
            gate("crop_correspondence_within_2mm", within_rate >= CROP_MATCH_RATE_MIN, within_rate, ">= 0.95"),
            gate(
                "crop_cycle_delta_median",
                percentile(cycle_delta, 50) <= CYCLE_MEDIAN_DELTA_MAX_MM,
                percentile(cycle_delta, 50),
                "<= 1 mm",
            ),
            gate(
                "crop_cycle_delta_p95",
                percentile(cycle_delta, 95) <= CYCLE_P95_DELTA_MAX_MM,
                percentile(cycle_delta, 95),
                "<= 4 mm",
            ),
        ]
    gates = [
        gate(
            "full_halo_correspondence_median",
            percentile(both, 50) <= FULL_MATCH_MEDIAN_MAX_MM,
            percentile(both, 50),
            "<= 2 mm",
        ),
        gate(
            "full_halo_correspondence_p95",
            percentile(both, 95) <= FULL_MATCH_P95_MAX_MM,
            percentile(both, 95),
            "<= 4 mm",
        ),
        gate(
            "full_halo_cycle_delta_median",
            percentile(cycle_delta, 50) <= CYCLE_MEDIAN_DELTA_MAX_MM,
            percentile(cycle_delta, 50),
            "<= 1 mm",
        ),
        gate(
            "full_halo_cycle_delta_p95",
            percentile(cycle_delta, 95) <= CYCLE_P95_DELTA_MAX_MM,
            percentile(cycle_delta, 95),
            "<= 4 mm",
        ),
    ]
    for organ in sorted({str(row.get("organ", "unknown")) for row in relevant}):
        organ_rows = [row for row in relevant if str(row.get("organ", "unknown")) == organ]
        organ_forward = np.asarray([_number(row, "forward_displacement_mm") for row in organ_rows])
        organ_backward = np.asarray([_number(row, "backward_displacement_mm") for row in organ_rows])
        organ_both = np.concatenate((organ_forward, organ_backward))
        organ_cycle = np.asarray([_number(row, "cycle_error_abs_delta_mm") for row in organ_rows])
        gates.extend(
            [
                gate(
                    f"full_{organ}_correspondence_median",
                    percentile(organ_both, 50) <= FULL_MATCH_MEDIAN_MAX_MM,
                    percentile(organ_both, 50),
                    "<= 2 mm",
                ),
                gate(
                    f"full_{organ}_correspondence_p95",
                    percentile(organ_both, 95) <= FULL_MATCH_P95_MAX_MM,
                    percentile(organ_both, 95),
                    "<= 4 mm",
                ),
                gate(
                    f"full_{organ}_cycle_delta_median",
                    percentile(organ_cycle, 50) <= CYCLE_MEDIAN_DELTA_MAX_MM,
                    percentile(organ_cycle, 50),
                    "<= 1 mm",
                ),
                gate(
                    f"full_{organ}_cycle_delta_p95",
                    percentile(organ_cycle, 95) <= CYCLE_P95_DELTA_MAX_MM,
                    percentile(organ_cycle, 95),
                    "<= 4 mm",
                ),
            ]
        )
    return gates


def build_validation_summary(output_dir: Path, phase_status: dict[str, object]) -> dict[str, object]:
    matcher_rows = read_csv(output_dir / "matcher_equivalence.csv")
    descriptor_rows = read_csv(output_dir / "descriptor_summary.csv")
    correspondence_rows = read_csv(output_dir / "correspondence_comparison.csv")
    gates = []
    gates.extend(evaluate_matcher_gates(matcher_rows))
    gates.extend(evaluate_descriptor_gates(descriptor_rows))
    gates.extend(evaluate_correspondence_gates(correspondence_rows, "crop"))
    gates.extend(evaluate_correspondence_gates(correspondence_rows, "full"))
    phase_failed = any(
        isinstance(value, dict) and str(value.get("status", "")).startswith("failed")
        for value in phase_status.values()
    )
    if phase_failed or any(item["status"] == "fail" for item in gates):
        overall = "fail"
    elif any(item["status"] == "blocked" for item in gates):
        overall = "blocked"
    else:
        overall = "pass"
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "overall_status": overall,
        "phase_status": phase_status,
        "gates": gates,
        "threshold_note": "Engineering tolerances are pre-specified diagnostics, not biological or clinical criteria.",
    }


def render_report(summary: dict[str, object], manifest: dict[str, object]) -> str:
    lines = [
        "# Quadra streaming-equivalence validation",
        "",
        f"- Subject: `{manifest.get('subject_id', 'unknown')}`",
        f"- Checkpoint role: `{manifest.get('checkpoint_role', 'unknown')}`",
        f"- Spacing: `{manifest.get('norm_spacing_xyz', [2.0, 2.0, 2.0])}` mm",
        f"- Overall engineering status: **{str(summary.get('overall_status', 'unknown')).upper()}**",
        "",
        "## Acceptance gates",
        "",
        "| Gate | Status | Value | Threshold |",
        "|---|---:|---:|---|",
    ]
    for item in summary.get("gates", []):
        value = item.get("value")
        value_text = f"{value:.6g}" if isinstance(value, (float, int)) and math.isfinite(float(value)) else str(value)
        lines.append(f"| {item['name']} | {str(item['status']).upper()} | {value_text} | {item['threshold']} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This is an engineering validation on one subject. A pass supports numerical streaming correctness and",
            "low sensitivity to the tested tile context; it does not establish clinical correspondence accuracy.",
            "The original `SAM.pth` result must be repeated with the Quadra fine-tuned checkpoint before scientific reporting.",
            "",
            "Review `descriptor_summary.csv`, `correspondence_comparison.csv`, and the discrepancy figures before accepting",
            "the generated status; the continuous measurements are more informative than the provisional thresholds alone.",
            "",
        ]
    )
    return "\n".join(lines)


def strip_nifti_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def load_preprocessed_case(pair: dict[str, object], timepoint: str, organs: Sequence[str], is_mri: bool):
    """Load one normalized image tensor and aligned processed masks."""
    from tools.utils import read_image

    image_path = Path(pair[timepoint])
    image_info, batch, _ = read_image(str(image_path), norm_spacing=(2.0, 2.0, 2.0), is_MRI=is_mri)
    volume = unwrap_model_input(batch)
    mask_parent = Path(pair["mask_dir"]).parent
    mask_dir = mask_parent / strip_nifti_suffix(image_path.name)
    masks = {}
    for organ in organs:
        mask_path = mask_dir / f"{organ}.nii.gz"
        if not mask_path.is_file():
            alternate = mask_dir / f"{organ}.nii"
            if alternate.is_file():
                mask_path = alternate
            else:
                raise FileNotFoundError(f"Required {timepoint} mask not found: {mask_path}")
        mask_info, mask_batch, _ = read_image(
            str(image_path), mask_path=str(mask_path), norm_spacing=(2.0, 2.0, 2.0), is_MRI=is_mri
        )
        mask_volume = unwrap_model_input(mask_batch)
        if tuple(mask_volume.shape) != tuple(volume.shape):
            raise RuntimeError(f"Mask preprocessing changed image shape for {mask_path}")
        # Both paths apply the same deterministic image preprocessing. Use only
        # the first image tensor so dense and tiled inference share exact bytes.
        max_difference = float(torch_max_abs_difference(volume, mask_volume))
        if max_difference > 1e-6:
            raise RuntimeError(f"Image preprocessing differed by {max_difference} while loading {mask_path}")
        mask_yxz = np.asarray(mask_info["processed_mask"])
        mask_zyx = np.transpose(mask_yxz, (2, 0, 1))
        if tuple(mask_zyx.shape) != tuple(volume.shape[2:]):
            raise RuntimeError(
                f"Processed mask shape {mask_zyx.shape} does not match image tensor {tuple(volume.shape[2:])}"
            )
        masks[organ] = mask_zyx > 0
        del mask_batch, mask_volume
    return image_path, image_info, volume, masks


def torch_max_abs_difference(left, right) -> float:
    import torch

    with torch.no_grad():
        return float(torch.max(torch.abs(left - right)).item())


def crop_tensor(volume, start_xyz: Sequence[int], size_xyz: Sequence[int]):
    slices = crop_slices_zyx(start_xyz, size_xyz)
    return volume[(slice(None), slice(None), *slices)].contiguous()


def point_columns(prefix: str, point_xyz: Sequence[int]) -> dict[str, int]:
    point = np.asarray(point_xyz, dtype=np.int64)
    return {f"{prefix}_x": int(point[0]), f"{prefix}_y": int(point[1]), f"{prefix}_z": int(point[2])}


def matcher_comparison_rows(
    organ: str,
    direction: str,
    queries: np.ndarray,
    dense_points: np.ndarray,
    dense_scores: np.ndarray,
    streamed_points: np.ndarray,
    streamed_scores: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for index in range(len(queries)):
        rows.append(
            {
                "phase": "crop",
                "organ": organ,
                "direction": direction,
                "query_index": index,
                **point_columns("query", queries[index]),
                **point_columns("dense", dense_points[index]),
                **point_columns("streamed", streamed_points[index]),
                "coordinate_match": bool(np.array_equal(dense_points[index], streamed_points[index])),
                "dense_score": float(dense_scores[index]),
                "streamed_score": float(streamed_scores[index]),
                "score_abs_diff": float(abs(float(dense_scores[index]) - float(streamed_scores[index]))),
            }
        )
    return rows


def crop_correspondence_rows(
    comparison: str,
    organ: str,
    queries: np.ndarray,
    dense_forward: np.ndarray,
    dense_backward: np.ndarray,
    dense_forward_scores: np.ndarray,
    dense_backward_scores: np.ndarray,
    tiled_forward: np.ndarray,
    tiled_backward: np.ndarray,
    tiled_forward_scores: np.ndarray,
    tiled_backward_scores: np.ndarray,
    core_size_xyz: Sequence[int],
    crop_shape_xyz: Sequence[int],
) -> list[dict[str, object]]:
    rows = []
    dense_cycle = np.linalg.norm((dense_backward - queries).astype(np.float64) * 2.0, axis=1)
    tiled_cycle = np.linalg.norm((tiled_backward - queries).astype(np.float64) * 2.0, axis=1)
    forward_displacement = np.linalg.norm((tiled_forward - dense_forward).astype(np.float64) * 2.0, axis=1)
    backward_displacement = np.linalg.norm((tiled_backward - dense_backward).astype(np.float64) * 2.0, axis=1)
    query_seam = internal_seam_distance(queries, crop_shape_xyz, core_size_xyz) * 2.0
    target_seam = internal_seam_distance(tiled_forward, crop_shape_xyz, core_size_xyz) * 2.0
    for index in range(len(queries)):
        rows.append(
            {
                "phase": "crop",
                "comparison": comparison,
                "organ": organ,
                "query_index": index,
                **point_columns("query", queries[index]),
                **point_columns("reference_forward", dense_forward[index]),
                **point_columns("candidate_forward", tiled_forward[index]),
                **point_columns("reference_backward", dense_backward[index]),
                **point_columns("candidate_backward", tiled_backward[index]),
                "forward_displacement_mm": float(forward_displacement[index]),
                "backward_displacement_mm": float(backward_displacement[index]),
                "reference_cycle_error_mm": float(dense_cycle[index]),
                "candidate_cycle_error_mm": float(tiled_cycle[index]),
                "cycle_error_abs_delta_mm": float(abs(dense_cycle[index] - tiled_cycle[index])),
                "forward_score_abs_diff": float(abs(dense_forward_scores[index] - tiled_forward_scores[index])),
                "backward_score_abs_diff": float(abs(dense_backward_scores[index] - tiled_backward_scores[index])),
                "query_seam_distance_mm": float(query_seam[index]),
                "candidate_target_seam_distance_mm": float(target_seam[index]),
            }
        )
    return rows


def descriptor_summary_rows_chunked(
    reference,
    candidate,
    plan: TilePlan,
    level: str,
    metadata: dict[str, object],
    z_chunk: int = 4,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Calculate full-volume metrics without materialising channels in FP32."""
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(f"Descriptor cache shapes differ: {reference.shape}, {candidate.shape}")
    stride = FINE_STRIDE_XYZ if level == "fine" else COARSE_STRIDE_XYZ
    shape_xyz = (int(reference.shape[3]), int(reference.shape[2]), int(reference.shape[1]))
    core_feature = tuple(plan.core_size_xyz[axis] // stride[axis] for axis in range(3))
    margin_feature = tuple(plan.halo_xyz[axis] // stride[axis] for axis in range(3))
    regions = feature_region_masks(shape_xyz, core_feature, margin_feature)
    metric_maps = {
        name: np.empty((reference.shape[1], reference.shape[2], reference.shape[3]), dtype=np.float32)
        for name in ("cosine", "l2", "mean_abs", "norm_ref", "norm_candidate")
    }
    for z_start in range(0, reference.shape[1], int(z_chunk)):
        z_stop = min(z_start + int(z_chunk), reference.shape[1])
        values = descriptor_cosine_and_error(
            np.asarray(reference[:, z_start:z_stop], dtype=np.float32),
            np.asarray(candidate[:, z_start:z_stop], dtype=np.float32),
        )
        for name, value in zip(metric_maps, values):
            metric_maps[name][z_start:z_stop] = value
    rows = []
    for region_name, region_mask in regions.items():
        rows.append(
            {
                **metadata,
                "level": level,
                "region": region_name,
                "count": int(np.count_nonzero(region_mask)),
                "cosine_median": percentile(metric_maps["cosine"][region_mask], 50),
                "cosine_p01": percentile(metric_maps["cosine"][region_mask], 1),
                "cosine_p05": percentile(metric_maps["cosine"][region_mask], 5),
                "l2_median": percentile(metric_maps["l2"][region_mask], 50),
                "l2_p95": percentile(metric_maps["l2"][region_mask], 95),
                "mean_abs_median": percentile(metric_maps["mean_abs"][region_mask], 50),
                "reference_norm_median": percentile(metric_maps["norm_ref"][region_mask], 50),
                "candidate_norm_median": percentile(metric_maps["norm_candidate"][region_mask], 50),
            }
        )
    return rows, 1.0 - metric_maps["cosine"]


def run_crop_validation(args, pair, model, output_dir: Path):
    import torch

    organs = tuple(args.organs)
    cases = {}
    for timepoint in ("test", "retest"):
        cases[timepoint] = load_preprocessed_case(pair, timepoint, organs, args.is_mri)
    matcher_rows = []
    descriptor_rows = []
    correspondence_rows = []
    frozen_rows = []
    profiles = []
    device = model_module_and_device(model)[1]
    figures_dir = output_dir / "figures"

    for organ_index, organ in enumerate(organs):
        embeddings = {}
        crop_starts = {}
        masks_crop = {}
        for timepoint in ("test", "retest"):
            _, _, volume, masks = cases[timepoint]
            start_xyz = deterministic_crop_start(masks[organ], args.dense_crop_size)
            crop_starts[timepoint] = start_xyz
            crop = crop_tensor(volume, start_xyz, args.dense_crop_size)
            mask_crop = masks[organ][crop_slices_zyx(start_xyz, args.dense_crop_size)]
            masks_crop[timepoint] = mask_crop
            dense_fine, dense_coarse, dense_profile = extract_dense_embeddings(crop, model)
            tiled_fine, tiled_coarse, plan, tiled_profile = extract_tiled_embeddings(
                crop, model, args.baseline_tile_size, args.baseline_halo
            )
            expanded_fine, expanded_coarse, expanded_plan, expanded_profile = extract_tiled_embeddings(
                crop, model, args.expanded_tile_size, args.expanded_halo
            )
            profiles.extend(
                [
                    {"phase": "crop", "timepoint": timepoint, "organ": organ, "method": "dense", **dense_profile},
                    {
                        "phase": "crop",
                        "timepoint": timepoint,
                        "organ": organ,
                        "method": "baseline_tiled",
                        **tiled_profile,
                    },
                    {
                        "phase": "crop",
                        "timepoint": timepoint,
                        "organ": organ,
                        "method": "expanded_tiled",
                        **expanded_profile,
                    },
                ]
            )
            for level, dense_array, tiled_array, expanded_array in (
                ("fine", dense_fine, tiled_fine, expanded_fine),
                ("coarse", dense_coarse, tiled_coarse, expanded_coarse),
            ):
                base_metadata = {"phase": "crop", "timepoint": timepoint, "organ": organ}
                rows, _ = descriptor_summary_rows(
                    dense_array,
                    tiled_array,
                    plan,
                    level,
                    {**base_metadata, "comparison": "dense_vs_tiled_fp32"},
                )
                descriptor_rows.extend(rows)
                rows, error_map = descriptor_summary_rows(
                    dense_array,
                    tiled_array.astype(np.float16).astype(np.float32),
                    plan,
                    level,
                    {**base_metadata, "comparison": "dense_vs_tiled_fp16"},
                )
                descriptor_rows.extend(rows)
                save_discrepancy_heatmap(
                    error_map,
                    figures_dir / f"crop_{timepoint}_{organ}_{level}_dense_vs_tiled.png",
                    f"{timepoint} {organ} {level}: dense vs baseline tiled FP16",
                )
                rows, _ = descriptor_summary_rows(
                    tiled_array.astype(np.float16).astype(np.float32),
                    expanded_array.astype(np.float16).astype(np.float32),
                    plan,
                    level,
                    {**base_metadata, "comparison": "baseline_vs_expanded_fp16"},
                )
                descriptor_rows.extend(rows)
                rows, expanded_error_map = descriptor_summary_rows(
                    dense_array,
                    expanded_array.astype(np.float16).astype(np.float32),
                    plan,
                    level,
                    {**base_metadata, "comparison": "dense_vs_expanded_fp16"},
                )
                descriptor_rows.extend(rows)
                save_discrepancy_heatmap(
                    expanded_error_map,
                    figures_dir / f"crop_{timepoint}_{organ}_{level}_dense_vs_expanded.png",
                    f"{timepoint} {organ} {level}: dense vs expanded tiled FP16",
                )
            shape_xyz = tuple(int(value) for value in args.dense_crop_size)
            embeddings[timepoint] = {
                "dense": ArrayEmbeddingCache(dense_fine, dense_coarse, shape_xyz),
                "tiled": ArrayEmbeddingCache(
                    tiled_fine.astype(np.float16), tiled_coarse.astype(np.float16), shape_xyz
                ),
                "expanded": ArrayEmbeddingCache(
                    expanded_fine.astype(np.float16), expanded_coarse.astype(np.float16), shape_xyz
                ),
                "plan": plan,
                "expanded_plan": expanded_plan,
            }
            del crop, dense_fine, dense_coarse, tiled_fine, tiled_coarse, expanded_fine, expanded_coarse
            torch.cuda.empty_cache()

        queries = select_mask_points_in_crop(
            masks_crop["test"], args.baseline_halo, args.crop_points_per_organ, args.seed + organ_index
        )
        for query_index, query in enumerate(queries):
            global_query = query + np.asarray(crop_starts["test"], dtype=np.int64)
            frozen_rows.append(
                {
                    "phase": "crop",
                    "query_index": query_index,
                    "organ": organ,
                    "coord_space": "resampled_crop_voxel",
                    **point_columns("point", query),
                    **point_columns("point_resampled_global", global_query),
                }
            )

        dense_forward, dense_forward_scores, _ = dense_global_match(
            embeddings["test"]["dense"], embeddings["retest"]["dense"], queries, args.query_batch_size, device
        )
        streamed_same_forward, streamed_same_forward_scores, _ = stream_global_match(
            embeddings["test"]["dense"],
            embeddings["retest"]["dense"],
            queries,
            args.query_batch_size,
            args.match_chunk_size,
            device=device,
        )
        matcher_rows.extend(
            matcher_comparison_rows(
                organ,
                "test_to_retest",
                queries,
                dense_forward,
                dense_forward_scores,
                streamed_same_forward,
                streamed_same_forward_scores,
            )
        )
        dense_backward, dense_backward_scores, _ = dense_global_match(
            embeddings["retest"]["dense"],
            embeddings["test"]["dense"],
            dense_forward,
            args.query_batch_size,
            device,
        )
        streamed_same_backward, streamed_same_backward_scores, _ = stream_global_match(
            embeddings["retest"]["dense"],
            embeddings["test"]["dense"],
            dense_forward,
            args.query_batch_size,
            args.match_chunk_size,
            device=device,
        )
        matcher_rows.extend(
            matcher_comparison_rows(
                organ,
                "retest_to_test",
                dense_forward,
                dense_backward,
                dense_backward_scores,
                streamed_same_backward,
                streamed_same_backward_scores,
            )
        )
        tiled_forward, tiled_forward_scores, _ = stream_global_match(
            embeddings["test"]["tiled"],
            embeddings["retest"]["tiled"],
            queries,
            args.query_batch_size,
            args.match_chunk_size,
            device=device,
        )
        tiled_backward, tiled_backward_scores, _ = stream_global_match(
            embeddings["retest"]["tiled"],
            embeddings["test"]["tiled"],
            tiled_forward,
            args.query_batch_size,
            args.match_chunk_size,
            device=device,
        )
        correspondence_rows.extend(
            crop_correspondence_rows(
                "dense_vs_baseline_tiled",
                organ,
                queries,
                dense_forward,
                dense_backward,
                dense_forward_scores,
                dense_backward_scores,
                tiled_forward,
                tiled_backward,
                tiled_forward_scores,
                tiled_backward_scores,
                embeddings["test"]["plan"].core_size_xyz,
                args.dense_crop_size,
            )
        )
        expanded_forward, expanded_forward_scores, _ = stream_global_match(
            embeddings["test"]["expanded"],
            embeddings["retest"]["expanded"],
            queries,
            args.query_batch_size,
            args.match_chunk_size,
            device=device,
        )
        expanded_backward, expanded_backward_scores, _ = stream_global_match(
            embeddings["retest"]["expanded"],
            embeddings["test"]["expanded"],
            expanded_forward,
            args.query_batch_size,
            args.match_chunk_size,
            device=device,
        )
        correspondence_rows.extend(
            crop_correspondence_rows(
                "dense_vs_expanded_tiled",
                organ,
                queries,
                dense_forward,
                dense_backward,
                dense_forward_scores,
                dense_backward_scores,
                expanded_forward,
                expanded_backward,
                expanded_forward_scores,
                expanded_backward_scores,
                embeddings["test"]["plan"].core_size_xyz,
                args.dense_crop_size,
            )
        )
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()

    replace_phase_rows(output_dir / "matcher_equivalence.csv", "crop", matcher_rows)
    replace_phase_rows(output_dir / "descriptor_summary.csv", "crop", descriptor_rows)
    replace_phase_rows(output_dir / "correspondence_comparison.csv", "crop", correspondence_rows)
    replace_phase_rows(output_dir / "frozen_query_points.csv", "crop", frozen_rows)
    return {
        "status": "complete",
        "organs": list(organs),
        "crop_size_xyz": list(args.dense_crop_size),
        "queries": len(frozen_rows),
        "profiles": profiles,
    }


def sam_points_to_raw_and_physical(image_path: Path, points_xyz: np.ndarray):
    import SimpleITK as sitk

    transform, raw_shape = build_sam_to_raw_transform(str(image_path))
    image = sitk.ReadImage(str(image_path))
    raw = np.stack(
        [transform_point_xyz(point, transform, raw_shape) for point in np.asarray(points_xyz, dtype=np.int64)], axis=0
    ).astype(np.int64)
    physical = np.stack(
        [image.TransformIndexToPhysicalPoint(tuple(int(value) for value in point)) for point in raw], axis=0
    ).astype(np.float64)
    return raw, physical


def full_correspondence_rows(
    records: list[dict[str, object]],
    pair: dict[str, object],
    baseline_forward: np.ndarray,
    baseline_backward: np.ndarray,
    baseline_forward_scores: np.ndarray,
    baseline_backward_scores: np.ndarray,
    expanded_forward: np.ndarray,
    expanded_backward: np.ndarray,
    expanded_forward_scores: np.ndarray,
    expanded_backward_scores: np.ndarray,
    baseline_test_cache: EmbeddingCache,
    baseline_retest_cache: EmbeddingCache,
) -> list[dict[str, object]]:
    queries = np.stack([np.asarray(record["pt1_sam"], dtype=np.int64) for record in records], axis=0)
    _, query_physical = sam_points_to_raw_and_physical(Path(pair["test"]), queries)
    _, baseline_forward_physical = sam_points_to_raw_and_physical(Path(pair["retest"]), baseline_forward)
    _, expanded_forward_physical = sam_points_to_raw_and_physical(Path(pair["retest"]), expanded_forward)
    _, baseline_backward_physical = sam_points_to_raw_and_physical(Path(pair["test"]), baseline_backward)
    _, expanded_backward_physical = sam_points_to_raw_and_physical(Path(pair["test"]), expanded_backward)
    forward_displacement = np.linalg.norm(expanded_forward_physical - baseline_forward_physical, axis=1)
    backward_displacement = np.linalg.norm(expanded_backward_physical - baseline_backward_physical, axis=1)
    baseline_cycle = np.linalg.norm(baseline_backward_physical - query_physical, axis=1)
    expanded_cycle = np.linalg.norm(expanded_backward_physical - query_physical, axis=1)
    test_plan = build_tile_plan(
        tuple(int(value) for value in baseline_test_cache.manifest["tile_plan"]["volume_shape_xyz"]),
        baseline_test_cache.manifest["tile_plan"]["tile_size_xyz"],
        baseline_test_cache.manifest["tile_plan"]["halo_xyz"],
    )
    retest_plan = build_tile_plan(
        tuple(int(value) for value in baseline_retest_cache.manifest["tile_plan"]["volume_shape_xyz"]),
        baseline_retest_cache.manifest["tile_plan"]["tile_size_xyz"],
        baseline_retest_cache.manifest["tile_plan"]["halo_xyz"],
    )
    query_seam = seam_distance_mm_for_native_points(
        queries, baseline_test_cache.norm_ratio_xyz, test_plan.volume_shape_xyz, test_plan.core_size_xyz
    )
    target_seam = seam_distance_mm_for_native_points(
        expanded_forward,
        baseline_retest_cache.norm_ratio_xyz,
        retest_plan.volume_shape_xyz,
        retest_plan.core_size_xyz,
    )
    rows = []
    for index, record in enumerate(records):
        organ = Path(str(record["mask_name"])).name.replace(".nii.gz", "").replace(".nii", "")
        rows.append(
            {
                "phase": "full",
                "comparison": "baseline_vs_expanded_tiled",
                "organ": organ,
                "query_index": index,
                **point_columns("query", queries[index]),
                **point_columns("reference_forward", baseline_forward[index]),
                **point_columns("candidate_forward", expanded_forward[index]),
                **point_columns("reference_backward", baseline_backward[index]),
                **point_columns("candidate_backward", expanded_backward[index]),
                "forward_displacement_mm": float(forward_displacement[index]),
                "backward_displacement_mm": float(backward_displacement[index]),
                "reference_cycle_error_mm": float(baseline_cycle[index]),
                "candidate_cycle_error_mm": float(expanded_cycle[index]),
                "cycle_error_abs_delta_mm": float(abs(baseline_cycle[index] - expanded_cycle[index])),
                "forward_score_abs_diff": float(abs(baseline_forward_scores[index] - expanded_forward_scores[index])),
                "backward_score_abs_diff": float(
                    abs(baseline_backward_scores[index] - expanded_backward_scores[index])
                ),
                "query_seam_distance_mm": float(query_seam[index]),
                "candidate_target_seam_distance_mm": float(target_seam[index]),
            }
        )
    return rows


def frozen_full_query_rows(records: list[dict[str, object]], test_image: Path) -> list[dict[str, object]]:
    sam_points = np.stack([np.asarray(record["pt1_sam"], dtype=np.int64) for record in records], axis=0)
    raw_points, _ = sam_points_to_raw_and_physical(test_image, sam_points)
    rows = []
    for index, (record, sam_point, raw_point) in enumerate(zip(records, sam_points, raw_points)):
        organ = Path(str(record["mask_name"])).name.replace(".nii.gz", "").replace(".nii", "")
        rows.append(
            {
                "phase": "full",
                "query_index": index,
                "organ": organ,
                "coord_space": COORD_SPACE_SAM,
                **point_columns("point", sam_point),
                **point_columns("point_raw_itk", raw_point),
                "raw_coord_space": COORD_SPACE_RAW_ITK,
            }
        )
    return rows


def build_named_cache_pair(
    args,
    pair: dict[str, object],
    model,
    config_identity: dict[str, object],
    checkpoint_identity: dict[str, object],
    cache_namespace: str,
    tile_size_xyz: Sequence[int],
    halo_xyz: Sequence[int],
):
    checkpoint_key = f"{Path(args.checkpoint_file).stem}_{checkpoint_identity['sha256'][:12]}_2mm"
    root = Path(args.cache_root).resolve() / checkpoint_key / pair["subject_id"] / cache_namespace
    manifests = {}
    for timepoint in ("test", "retest"):
        manifests[timepoint] = build_embedding_cache(
            Path(pair[timepoint]),
            root / timepoint,
            model,
            config_identity,
            checkpoint_identity,
            tile_size_xyz,
            halo_xyz,
            args.overwrite_cache,
            args.is_mri,
        )
    return root, manifests


def is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def run_full_validation(args, pair, model, output_dir: Path, config_identity, checkpoint_identity):
    import torch

    baseline_root, baseline_manifests = build_named_cache_pair(
        args,
        pair,
        model,
        config_identity,
        checkpoint_identity,
        "tile128_halo32",
        args.baseline_tile_size,
        args.baseline_halo,
    )
    try:
        expanded_root, expanded_manifests = build_named_cache_pair(
            args,
            pair,
            model,
            config_identity,
            checkpoint_identity,
            "tile160_halo48",
            args.expanded_tile_size,
            args.expanded_halo,
        )
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "blocked_full_expanded_oom",
            "error": str(exc),
            "fallback": "crop_baseline_vs_expanded",
            "baseline_cache_root": str(baseline_root),
        }

    records = sample_subject_points(
        pair, args.num_points, args.seed, set(args.organs) if args.organs else None, args.is_mri
    )
    queries = np.stack([np.asarray(record["pt1_sam"], dtype=np.int64) for record in records], axis=0)
    replace_phase_rows(
        output_dir / "frozen_query_points.csv", "full", frozen_full_query_rows(records, Path(pair["test"]))
    )
    baseline_test = EmbeddingCache(baseline_root / "test")
    baseline_retest = EmbeddingCache(baseline_root / "retest")
    expanded_test = EmbeddingCache(expanded_root / "test")
    expanded_retest = EmbeddingCache(expanded_root / "retest")

    baseline_forward, baseline_forward_scores, baseline_forward_profile = stream_global_match(
        baseline_test,
        baseline_retest,
        queries,
        args.query_batch_size,
        args.match_chunk_size,
    )
    baseline_backward, baseline_backward_scores, baseline_backward_profile = stream_global_match(
        baseline_retest,
        baseline_test,
        baseline_forward,
        args.query_batch_size,
        args.match_chunk_size,
    )
    expanded_forward, expanded_forward_scores, expanded_forward_profile = stream_global_match(
        expanded_test,
        expanded_retest,
        queries,
        args.query_batch_size,
        args.match_chunk_size,
    )
    expanded_backward, expanded_backward_scores, expanded_backward_profile = stream_global_match(
        expanded_retest,
        expanded_test,
        expanded_forward,
        args.query_batch_size,
        args.match_chunk_size,
    )
    correspondence_rows = full_correspondence_rows(
        records,
        pair,
        baseline_forward,
        baseline_backward,
        baseline_forward_scores,
        baseline_backward_scores,
        expanded_forward,
        expanded_backward,
        expanded_forward_scores,
        expanded_backward_scores,
        baseline_test,
        baseline_retest,
    )
    replace_phase_rows(output_dir / "correspondence_comparison.csv", "full", correspondence_rows)

    descriptor_rows = []
    figures_dir = output_dir / "figures"
    for timepoint, baseline_cache, expanded_cache in (
        ("test", baseline_test, expanded_test),
        ("retest", baseline_retest, expanded_retest),
    ):
        plan_payload = baseline_cache.manifest["tile_plan"]
        plan = build_tile_plan(
            plan_payload["volume_shape_xyz"], plan_payload["tile_size_xyz"], plan_payload["halo_xyz"]
        )
        for level in ("fine", "coarse"):
            rows, error_map = descriptor_summary_rows_chunked(
                baseline_cache.valid_array(level),
                expanded_cache.valid_array(level),
                plan,
                level,
                {
                    "phase": "full",
                    "timepoint": timepoint,
                    "organ": "ALL",
                    "comparison": "baseline_vs_expanded_fp16",
                },
            )
            descriptor_rows.extend(rows)
            save_discrepancy_heatmap(
                error_map,
                figures_dir / f"full_{timepoint}_{level}_baseline_vs_expanded.png",
                f"full {timepoint} {level}: baseline vs expanded halo",
            )
            del error_map
            gc.collect()
    replace_phase_rows(output_dir / "descriptor_summary.csv", "full", descriptor_rows)
    return {
        "status": "complete",
        "queries": len(records),
        "baseline_cache_root": str(baseline_root),
        "expanded_cache_root": str(expanded_root),
        "baseline_manifests": baseline_manifests,
        "expanded_manifests": expanded_manifests,
        "matching_profiles": {
            "baseline_forward": baseline_forward_profile,
            "baseline_backward": baseline_backward_profile,
            "expanded_forward": expanded_forward_profile,
            "expanded_backward": expanded_backward_profile,
        },
    }


def parse_phases(values: Sequence[str]) -> tuple[str, ...]:
    phases = []
    for value in values:
        for item in value.split(","):
            normalized = item.strip().lower()
            if normalized == "all":
                normalized_values = ("crop", "full")
            else:
                normalized_values = (normalized,)
            for phase in normalized_values:
                if phase not in ("crop", "full"):
                    raise ValueError(f"Unknown phase {phase!r}; expected crop, full, or all")
                if phase not in phases:
                    phases.append(phase)
    return tuple(phases)


def run(args) -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run validation in the configured RunPod GPU environment.")
    from tools.interfaces import init

    os.chdir(PROJECT_ROOT)
    subject_id = canonical_subject_id(args.subject)
    dataset_root = Path(args.dataset_root).resolve()
    pair = resolve_subject_pair(dataset_root, subject_id)
    config_identity = file_identity(args.config_file)
    checkpoint_identity = file_identity(args.checkpoint_file)
    checkpoint_role = "base_sam_engineering" if Path(args.checkpoint_file).name == "SAM.pth" else "user_supplied"
    if args.run_dir:
        output_dir = Path(args.run_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root).resolve() / f"{subject_id}_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = output_dir / "validation_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        prior_checkpoint = manifest.get("checkpoint", {}).get("sha256")
        if prior_checkpoint and prior_checkpoint != checkpoint_identity["sha256"]:
            raise RuntimeError("--run-dir checkpoint differs from the existing validation manifest")
    manifest.update(
        {
            "schema_version": 1,
            "created_at": manifest.get("created_at", utc_now()),
            "updated_at": utc_now(),
            "subject_id": subject_id,
            "dataset_root": str(dataset_root),
            "test_image": str(pair["test"]),
            "retest_image": str(pair["retest"]),
            "config": config_identity,
            "checkpoint": checkpoint_identity,
            "checkpoint_role": checkpoint_role,
            "result_status": "engineering_trial" if checkpoint_role == "base_sam_engineering" else "model_validation",
            "norm_spacing_xyz": [2.0, 2.0, 2.0],
            "phases_requested": list(args.phases),
            "dense_crop_size_xyz": list(args.dense_crop_size),
            "baseline_tile_size_xyz": list(args.baseline_tile_size),
            "baseline_halo_xyz": list(args.baseline_halo),
            "expanded_tile_size_xyz": list(args.expanded_tile_size),
            "expanded_halo_xyz": list(args.expanded_halo),
            "match_chunk_xyz": list(args.match_chunk_size),
            "query_batch_size": int(args.query_batch_size),
            "num_points_per_organ_full": int(args.num_points),
            "num_points_per_organ_crop": int(args.crop_points_per_organ),
            "seed": int(args.seed),
            "organs": list(args.organs),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "thresholds": {
                "match_score_abs_diff_max": MATCH_SCORE_TOLERANCE,
                "descriptor_median_cosine_min": DESCRIPTOR_MEDIAN_COSINE_MIN,
                "descriptor_p01_cosine_min": DESCRIPTOR_P01_COSINE_MIN,
                "descriptor_seam_drop_max": DESCRIPTOR_SEAM_DROP_MAX,
                "crop_match_within_mm": CROP_MATCH_WITHIN_MM,
                "crop_match_rate_min": CROP_MATCH_RATE_MIN,
                "cycle_median_delta_max_mm": CYCLE_MEDIAN_DELTA_MAX_MM,
                "cycle_p95_delta_max_mm": CYCLE_P95_DELTA_MAX_MM,
                "full_match_median_max_mm": FULL_MATCH_MEDIAN_MAX_MM,
                "full_match_p95_max_mm": FULL_MATCH_P95_MAX_MM,
            },
            "output_dir": str(output_dir),
        }
    )
    write_json(manifest_path, manifest)

    model = init(str(Path(args.config_file).resolve()), str(Path(args.checkpoint_file).resolve()))
    phase_status = dict(manifest.get("phase_status", {}))
    run_error = None
    try:
        if "crop" in args.phases:
            print("Running crop-level dense/tiled and matcher-equivalence validation")
            try:
                phase_status["crop"] = run_crop_validation(args, pair, model, output_dir)
            except Exception as exc:
                phase_status["crop"] = {"status": "failed", "error": str(exc)}
                run_error = exc
            manifest["phase_status"] = phase_status
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
        if "full" in args.phases and run_error is None:
            print("Running full-subject baseline/expanded-halo validation")
            try:
                phase_status["full"] = run_full_validation(
                    args, pair, model, output_dir, config_identity, checkpoint_identity
                )
                if phase_status["full"].get("status") == "blocked_full_expanded_oom" and "crop" not in args.phases:
                    print("Expanded full-subject tile exceeded GPU memory; running the required organ-crop fallback")
                    phase_status["crop"] = run_crop_validation(args, pair, model, output_dir)
            except Exception as exc:
                phase_status["full"] = {"status": "failed", "error": str(exc)}
                run_error = exc
            manifest["phase_status"] = phase_status
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = build_validation_summary(output_dir, phase_status)
    write_json(output_dir / "validation_summary.json", summary)
    (output_dir / "validation_report.md").write_text(render_report(summary, manifest), encoding="utf-8")
    print(f"Validation output: {output_dir}")
    print(f"Overall engineering status: {summary['overall_status']}")
    if run_error is not None:
        raise run_error
    return output_dir


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", help="Reuse an existing output directory to resume selected phases.")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--phases", nargs="+", default=("all",), help="crop, full, or all; comma-separated accepted.")
    parser.add_argument("--dense-crop-size", nargs=3, type=int, default=DEFAULT_DENSE_CROP_SIZE_XYZ)
    parser.add_argument("--baseline-tile-size", nargs=3, type=int, default=DEFAULT_BASELINE_TILE_SIZE_XYZ)
    parser.add_argument("--baseline-halo", nargs=3, type=int, default=DEFAULT_BASELINE_HALO_XYZ)
    parser.add_argument("--expanded-tile-size", nargs=3, type=int, default=DEFAULT_EXPANDED_TILE_SIZE_XYZ)
    parser.add_argument("--expanded-halo", nargs=3, type=int, default=DEFAULT_EXPANDED_HALO_XYZ)
    parser.add_argument("--match-chunk-size", nargs=3, type=int, default=DEFAULT_MATCH_CHUNK_XYZ)
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--crop-points-per-organ", type=int, default=DEFAULT_CROP_POINTS_PER_ORGAN)
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--organs", nargs="+", default=DEFAULT_ORGANS)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--is-mri", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.phases = parse_phases(args.phases)
    except ValueError as exc:
        parser.error(str(exc))
    for name in (
        "dense_crop_size",
        "baseline_tile_size",
        "baseline_halo",
        "expanded_tile_size",
        "expanded_halo",
        "match_chunk_size",
    ):
        setattr(args, name, _triple(getattr(args, name), name))
    if args.query_batch_size < 1 or args.crop_points_per_organ < 1 or args.num_points < 1:
        parser.error("query and point counts must be positive")
    args.organs = tuple(str(value).strip().lower() for value in args.organs)
    if len(set(args.organs)) != len(args.organs):
        parser.error("--organs must not contain duplicates")
    baseline_plan = build_tile_plan(args.dense_crop_size, args.baseline_tile_size, args.baseline_halo)
    expanded_plan = build_tile_plan(args.dense_crop_size, args.expanded_tile_size, args.expanded_halo)
    if baseline_plan.core_size_xyz != expanded_plan.core_size_xyz:
        parser.error(
            "Baseline and expanded tile configurations must retain the same core; "
            f"got {baseline_plan.core_size_xyz} and {expanded_plan.core_size_xyz}"
        )
    return args


def main(argv: Iterable[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
