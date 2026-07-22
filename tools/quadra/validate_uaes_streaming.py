#!/usr/bin/env python3
"""Validate UAE-S tiled descriptors and streamed matching on bounded crops."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra.streaming_cycle_error import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    canonical_subject_id,
    extract_uaes_query_descriptors,
    file_identity,
    model_module_and_device,
    resolve_subject_pair,
    stream_global_match_uaes,
    utc_now,
    write_json,
)
from tools.quadra.streaming_embedding import (  # noqa: E402
    COARSE_STRIDE_XYZ,
    FINE_STRIDE_XYZ,
    build_tile_plan,
    iter_tile_locations,
)
from tools.quadra.uaes_matching import FixedPointSettings, fixed_point_match_batch  # noqa: E402
from tools.quadra.validate_streaming_equivalence import (  # noqa: E402
    crop_tensor,
    descriptor_cosine_and_error,
    deterministic_crop_start,
    feature_region_masks,
    load_preprocessed_case,
    save_discrepancy_heatmap,
    select_mask_points_in_crop,
)


DEFAULT_CONFIG = "configs/samv2/samv2_NIHLN.py"
DEFAULT_CHECKPOINT = "checkpoints/SAMv2_iter_20000.pth"
DEFAULT_OUTPUT_ROOT = "data/quadra_output/uaes_streaming_validation"
DEFAULT_ORGANS = ("bladder", "colon", "kidney", "liver", "lungs")
DEFAULT_CROP_XYZ = (128, 128, 64)
DEFAULT_TILE_XYZ = (160, 160, 80)
DEFAULT_HALO_XYZ = (48, 48, 24)
DEFAULT_CHUNK_XYZ = (32, 32, 16)


@dataclass
class ArrayUaesCache:
    fine: np.ndarray
    coarse: np.ndarray
    semantic: np.ndarray
    native_shape_xyz_value: tuple[int, int, int]
    norm_ratio_xyz_value: tuple[float, float, float] = (1.0, 1.0, 1.0)
    cache_dir: str = "bounded_in_memory"

    def feature_shape_xyz(self, level: str) -> tuple[int, int, int]:
        array = getattr(self, level)
        return int(array.shape[3]), int(array.shape[2]), int(array.shape[1])

    @property
    def native_shape_xyz(self) -> tuple[int, int, int]:
        return self.native_shape_xyz_value

    @property
    def norm_ratio_xyz(self) -> np.ndarray:
        return np.asarray(self.norm_ratio_xyz_value, dtype=np.float64)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "native_spacing_xyz": [2.0, 2.0, 2.0],
            "norm_ratio_xyz": list(self.norm_ratio_xyz_value),
        }

    def valid_array(self, level: str):
        return getattr(self, level)


def _triple(values, name):
    if len(values) != 3:
        raise ValueError(f"{name} requires three x,y,z values")
    result = tuple(int(value) for value in values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


def extract_dense_uaes(volume, model):
    import torch

    module, device = model_module_and_device(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    with torch.no_grad():
        fine, coarse, semantic = module.extract_feat(volume.to(device=device, non_blocking=True))
    arrays = tuple(value[0].detach().cpu().float().numpy() for value in (fine, coarse, semantic))
    profile = {
        "seconds": float(time.time() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del fine, coarse, semantic
    torch.cuda.empty_cache()
    return (*arrays, profile)


def extract_tiled_uaes(volume, model, tile_xyz, halo_xyz):
    import torch
    import torch.nn.functional as torch_f

    module, device = model_module_and_device(model)
    shape_xyz = (int(volume.shape[4]), int(volume.shape[3]), int(volume.shape[2]))
    plan = build_tile_plan(shape_xyz, tile_size_xyz=tile_xyz, halo_xyz=halo_xyz)
    tail = tuple(plan.grid_shape_xyz[axis] * plan.core_size_xyz[axis] - shape_xyz[axis] for axis in range(3))
    hx, hy, hz = plan.halo_xyz
    padded = torch_f.pad(
        volume,
        (hx, hx + tail[0], hy, hy + tail[1], hz, hz + tail[2]),
        mode="constant",
        value=0.0,
    )
    maps = None
    peak = 0
    started = time.time()
    for location in iter_tile_locations(plan):
        tile = padded[(slice(None), slice(None), *location.padded_input_slices_zyx)].to(device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            fine, coarse, semantic = module.extract_feat(tile)
        peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        if maps is None:
            maps = [
                np.empty((int(fine.shape[1]), *reversed(plan.stored_fine_shape_xyz)), dtype=np.float32),
                np.empty((int(coarse.shape[1]), *reversed(plan.stored_coarse_shape_xyz)), dtype=np.float32),
                np.empty((int(semantic.shape[1]), *reversed(plan.stored_fine_shape_xyz)), dtype=np.float32),
            ]
        maps[0][(slice(None), *location.fine_destination_slices_zyx)] = (
            fine[(0, slice(None), *location.fine_source_slices_zyx)].detach().cpu().float().numpy()
        )
        maps[1][(slice(None), *location.coarse_destination_slices_zyx)] = (
            coarse[(0, slice(None), *location.coarse_source_slices_zyx)].detach().cpu().float().numpy()
        )
        maps[2][(slice(None), *location.fine_destination_slices_zyx)] = (
            semantic[(0, slice(None), *location.fine_source_slices_zyx)].detach().cpu().float().numpy()
        )
        del tile, fine, coarse, semantic
    valid_fine = tuple(reversed(plan.valid_fine_shape_xyz))
    valid_coarse = tuple(reversed(plan.valid_coarse_shape_xyz))
    maps[0] = maps[0][:, : valid_fine[0], : valid_fine[1], : valid_fine[2]]
    maps[1] = maps[1][:, : valid_coarse[0], : valid_coarse[1], : valid_coarse[2]]
    maps[2] = maps[2][:, : valid_fine[0], : valid_fine[1], : valid_fine[2]]
    del padded
    torch.cuda.empty_cache()
    return (*maps, plan, {"seconds": time.time() - started, "peak_gpu_memory_bytes": peak})


def dense_match_uaes(
    query_cache,
    target_cache,
    query_points_xyz,
    query_batch_size,
    match_chunk_xyz=None,
    device=None,
    output_space="native",
):
    """Official-formula dense reference on a bounded in-memory embedding."""
    import torch
    import torch.nn.functional as torch_f

    device = torch.device(device or "cuda:0")
    points = np.asarray(query_points_xyz, dtype=np.int64)
    q_fine, q_coarse, q_semantic, _ = extract_uaes_query_descriptors(
        query_cache, points, device, points_are_fine=output_space == "fine"
    )
    target_fine = torch.from_numpy(np.asarray(target_cache.fine, dtype=np.float32)).to(device)
    target_semantic = torch.from_numpy(np.asarray(target_cache.semantic, dtype=np.float32)).to(device)
    target_coarse = torch.from_numpy(np.asarray(target_cache.coarse, dtype=np.float32)).unsqueeze(0).to(device)
    fine_shape_zyx = tuple(int(value) for value in target_fine.shape[1:])
    coarse = torch_f.interpolate(target_coarse, fine_shape_zyx, mode="trilinear", align_corners=False)
    coarse = torch_f.normalize(coarse, dim=1)[0]
    flat = [value.reshape(value.shape[0], -1) for value in (target_fine, coarse, target_semantic)]
    output_shape_xyz = target_cache.native_shape_xyz if output_space == "native" else target_cache.feature_shape_xyz("fine")
    output_shape_zyx = tuple(reversed(output_shape_xyz))
    best_points = np.zeros((len(points), 3), dtype=np.int64)
    best_scores = np.full(len(points), -np.inf, dtype=np.float32)
    started = time.time()
    for start in range(0, len(points), int(query_batch_size)):
        stop = min(start + int(query_batch_size), len(points))
        sim = (
            torch.matmul(q_fine[start:stop], flat[0])
            + torch.matmul(q_coarse[start:stop], flat[1])
            + torch.matmul(q_semantic[start:stop], flat[2])
        ) / 3.0
        sim = sim.reshape(stop - start, 1, *fine_shape_zyx)
        if output_space == "native":
            sim = torch_f.interpolate(sim, output_shape_zyx, mode="trilinear", align_corners=False)
        values, indices = torch.max(sim.reshape(stop - start, -1), dim=1)
        indices = indices.detach().cpu().numpy().astype(np.int64)
        sx, sy, _ = output_shape_xyz
        best_points[start:stop] = np.stack(
            (indices % sx, (indices // sx) % sy, indices // (sx * sy)), axis=1
        )
        best_scores[start:stop] = values.detach().cpu().numpy().astype(np.float32)
    return best_points, best_scores, {
        "seconds": time.time() - started,
        "output_space": output_space,
        "reference": "official_dense_formula",
    }


def _descriptor_rows(reference, candidate, plan, level, organ, timepoint):
    stride = FINE_STRIDE_XYZ if level in ("fine", "semantic") else COARSE_STRIDE_XYZ
    shape_xyz = (reference.shape[3], reference.shape[2], reference.shape[1])
    core = tuple(plan.core_size_xyz[axis] // stride[axis] for axis in range(3))
    halo = tuple(plan.halo_xyz[axis] // stride[axis] for axis in range(3))
    regions = feature_region_masks(shape_xyz, core, halo)
    cosine, l2, mean_abs, _, _ = descriptor_cosine_and_error(reference, candidate)
    rows = []
    for region, mask in regions.items():
        values = cosine[mask]
        rows.append(
            {
                "organ": organ,
                "timepoint": timepoint,
                "feature": level,
                "region": region,
                "voxel_count": int(values.size),
                "median_cosine": float(np.median(values)),
                "p01_cosine": float(np.percentile(values, 1)),
                "median_l2": float(np.median(l2[mask])),
                "median_abs": float(np.median(mean_abs[mask])),
            }
        )
    return rows, 1.0 - cosine


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    import torch
    from tools.interfaces import init

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UAE-S validation")
    os.chdir(PROJECT_ROOT)
    subject = canonical_subject_id(args.subject)
    pair = resolve_subject_pair(Path(args.dataset_root).resolve(), subject)
    run_dir = Path(args.output_root).resolve() / f"{subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)
    figures = run_dir / "figures"
    model = init(str(Path(args.config_file).resolve()), str(Path(args.checkpoint_file).resolve()))
    cases = {name: load_preprocessed_case(pair, name, args.organs, False) for name in ("test", "retest")}
    descriptor_rows = []
    matcher_rows = []
    fixed_rows = []
    profiles = []
    settings = FixedPointSettings()
    device = model_module_and_device(model)[1]
    for organ_index, organ in enumerate(args.organs):
        caches = {}
        masks = {}
        for timepoint in ("test", "retest"):
            _, _, volume, case_masks = cases[timepoint]
            start = deterministic_crop_start(case_masks[organ], args.crop_size)
            crop = crop_tensor(volume, start, args.crop_size)
            masks[timepoint] = case_masks[organ][
                start[2] : start[2] + args.crop_size[2],
                start[1] : start[1] + args.crop_size[1],
                start[0] : start[0] + args.crop_size[0],
            ]
            dense_f, dense_c, dense_s, dense_profile = extract_dense_uaes(crop, model)
            tiled_f, tiled_c, tiled_s, plan, tiled_profile = extract_tiled_uaes(
                crop, model, args.tile_size, args.halo
            )
            profiles.extend(
                [
                    {"organ": organ, "timepoint": timepoint, "method": "dense", **dense_profile},
                    {"organ": organ, "timepoint": timepoint, "method": "tiled", **tiled_profile},
                ]
            )
            for level, dense, tiled in (
                ("fine", dense_f, tiled_f),
                ("coarse", dense_c, tiled_c),
                ("semantic", dense_s, tiled_s),
            ):
                rows, error = _descriptor_rows(
                    dense, tiled.astype(np.float16).astype(np.float32), plan, level, organ, timepoint
                )
                descriptor_rows.extend(rows)
                if level == "semantic":
                    save_discrepancy_heatmap(
                        error,
                        figures / f"{organ}_{timepoint}_semantic.png",
                        f"{organ} {timepoint}: dense vs tiled semantic",
                    )
            shape = tuple(args.crop_size)
            caches[timepoint] = {
                "dense": ArrayUaesCache(dense_f, dense_c, dense_s, shape),
                "tiled": ArrayUaesCache(
                    tiled_f.astype(np.float16), tiled_c.astype(np.float16), tiled_s.astype(np.float16), shape
                ),
            }
        queries = select_mask_points_in_crop(
            masks["test"], args.halo, args.points_per_organ, args.seed + organ_index
        )
        dense_points, dense_scores, _ = dense_match_uaes(
            caches["test"]["dense"], caches["retest"]["dense"], queries,
            args.query_batch_size, args.match_chunk_size, device, "native"
        )
        streamed_points, streamed_scores, _ = stream_global_match_uaes(
            caches["test"]["dense"], caches["retest"]["dense"], queries,
            args.query_batch_size, args.match_chunk_size, device, "native"
        )
        for index in range(len(queries)):
            matcher_rows.append(
                {
                    "organ": organ,
                    "query_index": index,
                    "coordinate_equal": bool(np.array_equal(dense_points[index], streamed_points[index])),
                    "coordinate_distance_voxels": float(np.linalg.norm(dense_points[index] - streamed_points[index])),
                    "score_abs_difference": float(abs(dense_scores[index] - streamed_scores[index])),
                }
            )
        fixed_queries = queries[: args.fixed_points_per_organ]
        dense_fixed, dense_profile = fixed_point_match_batch(
            caches["test"]["dense"], caches["retest"]["dense"], fixed_queries, settings,
            args.query_batch_size, args.match_chunk_size, device=device, match_function=dense_match_uaes
        )
        streamed_fixed, streamed_profile = fixed_point_match_batch(
            caches["test"]["dense"], caches["retest"]["dense"], fixed_queries, settings,
            args.query_batch_size, args.match_chunk_size, device=device
        )
        internal_equal = [
            left["matched_fine_sha256"] == right["matched_fine_sha256"]
            for left, right in zip(dense_profile["iterations"], streamed_profile["iterations"])
        ]
        for index, (dense_result, stream_result) in enumerate(zip(dense_fixed, streamed_fixed)):
            coordinates = None
            if dense_result["point_xyz"] is not None and stream_result["point_xyz"] is not None:
                coordinates = float(np.linalg.norm(dense_result["point_xyz"] - stream_result["point_xyz"]))
            fixed_rows.append(
                {
                    "organ": organ,
                    "query_index": index,
                    "dense_status": dense_result["status"],
                    "streamed_status": stream_result["status"],
                    "all_internal_coordinates_equal": bool(all(internal_equal)),
                    "final_coordinate_distance_voxels": coordinates,
                    "dense_failure_reason": dense_result.get("failure_reason"),
                    "streamed_failure_reason": stream_result.get("failure_reason"),
                }
            )

    _write_csv(run_dir / "descriptor_summary.csv", descriptor_rows)
    _write_csv(run_dir / "matcher_equivalence.csv", matcher_rows)
    _write_csv(run_dir / "fixed_point_equivalence.csv", fixed_rows)
    semantic = [row for row in descriptor_rows if row["feature"] == "semantic"]
    seam_by_case = {}
    for row in semantic:
        seam_by_case.setdefault((row["organ"], row["timepoint"]), {})[row["region"]] = row
    seam_drops = [
        regions["interior"]["median_cosine"] - regions["seam"]["median_cosine"]
        for regions in seam_by_case.values()
        if "interior" in regions and "seam" in regions
    ]
    gates = {
        "global_argmax_exact": all(row["coordinate_equal"] for row in matcher_rows),
        "fixed_internal_argmax_exact": all(row["all_internal_coordinates_equal"] for row in fixed_rows),
        "fixed_final_within_one_native_voxel": all(
            row["final_coordinate_distance_voxels"] is None
            or row["final_coordinate_distance_voxels"] <= 1.0
            for row in fixed_rows
        ),
        "semantic_median_cosine": min(row["median_cosine"] for row in semantic if row["region"] == "all") >= 0.99,
        "semantic_p01_cosine": min(row["p01_cosine"] for row in semantic if row["region"] == "all") >= 0.95,
        "semantic_seam_drop": max(seam_drops) <= 0.01,
    }
    summary = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "subject_id": subject,
        "spacing_xyz_mm": [2.0, 2.0, 2.0],
        "crop_size_xyz": list(args.crop_size),
        "tile_size_xyz": list(args.tile_size),
        "halo_xyz": list(args.halo),
        "core_xyz": [args.tile_size[i] - 2 * args.halo[i] for i in range(3)],
        "config": file_identity(args.config_file),
        "checkpoint": file_identity(args.checkpoint_file),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "profiles": profiles,
    }
    write_json(run_dir / "validation_summary.json", summary)
    print(f"Completed UAE-S streaming validation: {run_dir}")
    print(json.dumps(gates, indent=2))
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="quadra_hc_021")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--organs", nargs="+", default=list(DEFAULT_ORGANS))
    parser.add_argument("--crop-size", nargs=3, type=int, default=DEFAULT_CROP_XYZ)
    parser.add_argument("--tile-size", nargs=3, type=int, default=DEFAULT_TILE_XYZ)
    parser.add_argument("--halo", nargs=3, type=int, default=DEFAULT_HALO_XYZ)
    parser.add_argument("--match-chunk-size", nargs=3, type=int, default=DEFAULT_CHUNK_XYZ)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--points-per-organ", type=int, default=5)
    parser.add_argument("--fixed-points-per-organ", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args(argv)
    for name in ("crop_size", "tile_size", "halo", "match_chunk_size"):
        setattr(args, name, _triple(getattr(args, name), name))
    if any(2 * args.halo[i] >= args.tile_size[i] for i in range(3)):
        parser.error("tile size must exceed twice the halo in every axis")
    return args


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
