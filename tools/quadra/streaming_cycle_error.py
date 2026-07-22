#!/usr/bin/env python3
"""Run 2 mm Quadra UAE cycle matching with tiled embeddings and global search.

The encoder uses overlapping tiles and keeps only each tile's central region.
Matching streams every target location in bounded chunks; chunks control memory
but never restrict the anatomical search range.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
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
from tools.quadra.streaming_embedding import (  # noqa: E402
    COARSE_STRIDE_XYZ,
    FINE_STRIDE_XYZ,
    TilePlan,
    align_corners_false_source_positions,
    build_tile_plan,
    iter_chunks_xyz,
    iter_tile_locations,
    source_bounds_for_output_interval,
)


CACHE_SCHEMA_VERSION = 1
NORM_SPACING_XYZ = (2.0, 2.0, 2.0)
DEFAULT_DATASET_ROOT = "data/quadra_dataset_cropped"
DEFAULT_CACHE_ROOT = "data/quadra_streaming_cache"
DEFAULT_OUTPUT_ROOT = "data/quadra_output/streaming_cycle_error"
DEFAULT_CONFIG_FILE = "configs/sam/sam_NIHLN.py"
DEFAULT_CHECKPOINT_FILE = "checkpoints/SAM.pth"
DEFAULT_SUBJECT = "quadra_hc_021"
DEFAULT_TILE_SIZE_XYZ = (128, 128, 64)
DEFAULT_HALO_XYZ = (32, 32, 16)
DEFAULT_MATCH_CHUNK_XYZ = (64, 64, 32)
DEFAULT_QUERY_BATCH_SIZE = 16
DEFAULT_NUM_POINTS = 100
DEFAULT_SEED = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required file not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def is_nifti_file(name: str) -> bool:
    return name.endswith(".nii") or name.endswith(".nii.gz")


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def canonical_subject_id(subject_id: str) -> str:
    normalized = str(subject_id).strip().lower()
    prefix = "quadra_hc_"
    if normalized.startswith(prefix):
        suffix = normalized[len(prefix) :]
        if suffix.isdigit():
            return f"{prefix}{int(suffix):03d}"
    return normalized


def resolve_subject_pair(dataset_root: Path, subject_id: str) -> dict[str, object]:
    images_root = dataset_root / "images"
    masks_root = dataset_root / "masks"
    image_dir = images_root / subject_id
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Subject image directory not found: {image_dir}")
    names = sorted(path.name for path in image_dir.iterdir() if path.is_file() and is_nifti_file(path.name))
    test = [name for name in names if "_Test_" in name]
    retest = [name for name in names if "_Retest_" in name]
    if len(test) != 1 or len(retest) != 1:
        raise RuntimeError(f"Expected one Test and Retest image in {image_dir}; Test={test}, Retest={retest}")
    mask_dir = masks_root / subject_id / strip_nii_suffix(test[0])
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Test mask directory not found: {mask_dir}")
    masks = sorted(path for path in mask_dir.iterdir() if path.is_file() and is_nifti_file(path.name))
    if not masks:
        raise RuntimeError(f"No NIfTI masks found in: {mask_dir}")
    return {
        "subject_id": subject_id,
        "test": image_dir / test[0],
        "retest": image_dir / retest[0],
        "mask_dir": mask_dir,
        "masks": masks,
    }


def unwrap_model_input(batched_data):
    import torch

    value = batched_data["img"]
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty model image container returned by read_image")
        value = value[0]
    if hasattr(value, "data") and not torch.is_tensor(value):
        value = value.data
        while isinstance(value, (list, tuple)):
            value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected a model input tensor, got {type(value)!r}")
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 1:
        raise ValueError(f"Expected model input shape [1,1,z,y,x], got {tuple(value.shape)}")
    return value.detach().cpu().contiguous()


def model_module_and_device(model):
    module = model.module if hasattr(model, "module") else model
    try:
        device = next(module.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters from which to determine its device") from exc
    return module, device


def expected_feature_shape_zyx(tile_size_xyz: Sequence[int], stride_xyz: Sequence[int]):
    tile_x, tile_y, tile_z = (int(value) for value in tile_size_xyz)
    stride_x, stride_y, stride_z = (int(value) for value in stride_xyz)
    return tile_z // stride_z, tile_y // stride_y, tile_x // stride_x


def cache_signature(
    image_identity: dict[str, object],
    config_identity: dict[str, object],
    checkpoint_identity: dict[str, object],
    plan: TilePlan,
) -> dict[str, object]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_image": image_identity,
        "config": config_identity,
        "checkpoint": checkpoint_identity,
        "norm_spacing_xyz": list(NORM_SPACING_XYZ),
        "tile_plan": plan.to_dict(),
        "embedding_dtype": "float16",
        "stitching": "central_valid_region",
    }


def load_complete_manifest(cache_dir: Path) -> dict[str, object] | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("complete"):
        return None
    return manifest


def validate_cache_signature(manifest: dict[str, object], expected: dict[str, object], cache_dir: Path) -> None:
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"Existing cache is incompatible with this run: {cache_dir}. "
            "Use --overwrite-cache or a different --cache-root."
        )
    for filename in ("fine.npy", "coarse.npy"):
        if not (cache_dir / filename).is_file():
            raise FileNotFoundError(f"Complete cache manifest references missing file: {cache_dir / filename}")


def build_embedding_cache(
    image_path: Path,
    cache_dir: Path,
    model,
    config_identity: dict[str, object],
    checkpoint_identity: dict[str, object],
    tile_size_xyz: Sequence[int],
    halo_xyz: Sequence[int],
    overwrite: bool,
    is_mri: bool,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as torch_f

    from tools.utils import read_image

    image_identity = file_identity(str(image_path))
    image_info, batched_data, norm_ratio = read_image(
        str(image_path), norm_spacing=NORM_SPACING_XYZ, mask_path=None, is_MRI=is_mri
    )
    volume = unwrap_model_input(batched_data)
    volume_shape_xyz = (int(volume.shape[4]), int(volume.shape[3]), int(volume.shape[2]))
    plan = build_tile_plan(volume_shape_xyz, tile_size_xyz=tile_size_xyz, halo_xyz=halo_xyz)
    signature = cache_signature(image_identity, config_identity, checkpoint_identity, plan)

    existing = load_complete_manifest(cache_dir)
    if existing is not None and not overwrite:
        validate_cache_signature(existing, signature, cache_dir)
        print(f"Reusing complete embedding cache: {cache_dir}")
        del volume, batched_data
        gc.collect()
        return existing

    if cache_dir.exists():
        if not overwrite:
            raise RuntimeError(f"Incomplete embedding cache exists: {cache_dir}. Use --overwrite-cache.")
        shutil.rmtree(cache_dir)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.", dir=str(cache_dir.parent)))
    module, device = model_module_and_device(model)
    if device.type != "cuda":
        raise RuntimeError("Tiled embedding generation requires a CUDA model on the RunPod GPU.")

    tile_x, tile_y, tile_z = plan.tile_size_xyz
    halo_x, halo_y, halo_z = plan.halo_xyz
    covered_x = plan.grid_shape_xyz[0] * plan.core_size_xyz[0]
    covered_y = plan.grid_shape_xyz[1] * plan.core_size_xyz[1]
    covered_z = plan.grid_shape_xyz[2] * plan.core_size_xyz[2]
    tail_x = covered_x - plan.volume_shape_xyz[0]
    tail_y = covered_y - plan.volume_shape_xyz[1]
    tail_z = covered_z - plan.volume_shape_xyz[2]
    padded = torch_f.pad(
        volume,
        (halo_x, halo_x + tail_x, halo_y, halo_y + tail_y, halo_z, halo_z + tail_z),
        mode="constant",
        value=0.0,
    )
    expected_padded_zyx = (
        plan.padded_shape_xyz[2],
        plan.padded_shape_xyz[1],
        plan.padded_shape_xyz[0],
    )
    if tuple(padded.shape[2:]) != expected_padded_zyx:
        raise RuntimeError(f"Unexpected padded shape {tuple(padded.shape[2:])}, expected {expected_padded_zyx}")

    fine_map = None
    coarse_map = None
    fine_channels = None
    coarse_channels = None
    peak_bytes = 0
    started = time.time()
    locations = list(iter_tile_locations(plan))

    try:
        for tile_index, location in enumerate(locations, start=1):
            tile = padded[(slice(None), slice(None), *location.padded_input_slices_zyx)].to(
                device=device, non_blocking=True
            )
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                fine, coarse = module.extract_feat(tile)
            peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated(device)))

            fine_expected = expected_feature_shape_zyx(plan.tile_size_xyz, FINE_STRIDE_XYZ)
            coarse_expected = expected_feature_shape_zyx(plan.tile_size_xyz, COARSE_STRIDE_XYZ)
            if tuple(fine.shape[2:]) != fine_expected:
                raise RuntimeError(f"Fine tile shape {tuple(fine.shape[2:])} does not match expected {fine_expected}")
            if tuple(coarse.shape[2:]) != coarse_expected:
                raise RuntimeError(
                    f"Coarse tile shape {tuple(coarse.shape[2:])} does not match expected {coarse_expected}"
                )

            if fine_map is None:
                fine_channels = int(fine.shape[1])
                coarse_channels = int(coarse.shape[1])
                fine_stored_zyx = tuple(reversed(plan.stored_fine_shape_xyz))
                coarse_stored_zyx = tuple(reversed(plan.stored_coarse_shape_xyz))
                fine_map = np.lib.format.open_memmap(
                    temporary_dir / "fine.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(fine_channels, *fine_stored_zyx),
                )
                coarse_map = np.lib.format.open_memmap(
                    temporary_dir / "coarse.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(coarse_channels, *coarse_stored_zyx),
                )

            fine_core = fine[(0, slice(None), *location.fine_source_slices_zyx)].detach().cpu().numpy()
            coarse_core = coarse[(0, slice(None), *location.coarse_source_slices_zyx)].detach().cpu().numpy()
            fine_map[(slice(None), *location.fine_destination_slices_zyx)] = fine_core
            coarse_map[(slice(None), *location.coarse_destination_slices_zyx)] = coarse_core

            del tile, fine, coarse, fine_core, coarse_core
            if tile_index == 1 or tile_index % 25 == 0 or tile_index == len(locations):
                print(f"  embedding tile {tile_index}/{len(locations)} for {image_path.name}")

        fine_map.flush()
        coarse_map.flush()
        manifest = {
            **signature,
            "complete": True,
            "created_at": utc_now(),
            "cache_dir": str(cache_dir.resolve()),
            "native_shape_yxz": [int(value) for value in image_info["img"].shape],
            "native_sam_shape_xyz": [
                int(image_info["img"].shape[1]),
                int(image_info["img"].shape[0]),
                int(image_info["img"].shape[2]),
            ],
            "native_spacing_xyz": [float(value) for value in image_info["spacing"]],
            "native_affine": np.asarray(image_info["affine"], dtype=float).tolist(),
            "resampled_affine": np.asarray(image_info["affine_resampled"], dtype=float).tolist(),
            "norm_ratio_xyz": np.asarray(norm_ratio, dtype=float).tolist(),
            "features": {
                "fine": {
                    "file": "fine.npy",
                    "channels": int(fine_channels),
                    "stride_xyz": list(FINE_STRIDE_XYZ),
                    "stored_shape_xyz": list(plan.stored_fine_shape_xyz),
                    "valid_shape_xyz": list(plan.valid_fine_shape_xyz),
                },
                "coarse": {
                    "file": "coarse.npy",
                    "channels": int(coarse_channels),
                    "stride_xyz": list(COARSE_STRIDE_XYZ),
                    "stored_shape_xyz": list(plan.stored_coarse_shape_xyz),
                    "valid_shape_xyz": list(plan.valid_coarse_shape_xyz),
                },
            },
            "generation_seconds": float(time.time() - started),
            "peak_gpu_memory_bytes": int(peak_bytes),
        }
        write_json(temporary_dir / "manifest.json", manifest)
        os.replace(temporary_dir, cache_dir)
        print(f"Completed embedding cache: {cache_dir}")
        return manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    finally:
        del padded, volume, batched_data
        gc.collect()
        torch.cuda.empty_cache()


class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        manifest = load_complete_manifest(cache_dir)
        if manifest is None:
            raise RuntimeError(f"Embedding cache is missing or incomplete: {cache_dir}")
        self.cache_dir = cache_dir
        self.manifest = manifest
        self.fine = np.load(cache_dir / manifest["features"]["fine"]["file"], mmap_mode="r")
        self.coarse = np.load(cache_dir / manifest["features"]["coarse"]["file"], mmap_mode="r")

    def feature_shape_xyz(self, level: str) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.manifest["features"][level]["valid_shape_xyz"])

    @property
    def native_shape_xyz(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.manifest["native_sam_shape_xyz"])

    @property
    def norm_ratio_xyz(self) -> np.ndarray:
        return np.asarray(self.manifest["norm_ratio_xyz"], dtype=np.float64)

    def valid_array(self, level: str):
        array = self.fine if level == "fine" else self.coarse
        shape_x, shape_y, shape_z = self.feature_shape_xyz(level)
        return array[:, :shape_z, :shape_y, :shape_x]


def normalized_grid_coordinates(source_positions: np.ndarray, source_start: int, source_size: int) -> np.ndarray:
    local = np.asarray(source_positions, dtype=np.float64) - float(source_start)
    if source_size <= 1:
        return np.zeros_like(local, dtype=np.float32)
    return (2.0 * local / float(source_size - 1) - 1.0).astype(np.float32)


def interpolation_grid_for_box(
    output_ranges_xyz,
    input_shape_xyz,
    output_shape_xyz,
    source_bounds_xyz,
    device,
):
    import torch

    positions = []
    for axis in range(3):
        output_start, output_stop = output_ranges_xyz[axis]
        input_size = input_shape_xyz[axis]
        output_size = output_shape_xyz[axis]
        source_start, source_stop = source_bounds_xyz[axis]
        global_positions = align_corners_false_source_positions(
            output_start, output_stop, input_size, output_size
        )
        positions.append(
            torch.as_tensor(
                normalized_grid_coordinates(global_positions, source_start, source_stop - source_start),
                device=device,
            )
        )
    # Omit the newer ``indexing=`` keyword for compatibility with torch 1.9.
    grid_x, grid_y, grid_z = torch.meshgrid(positions[0], positions[1], positions[2])
    return torch.stack((grid_x, grid_y, grid_z), dim=-1).permute(2, 1, 0, 3).unsqueeze(0)


def sample_coarse_for_fine_box(coarse_tensor, fine_shape_xyz, fine_bounds_xyz):
    import torch
    import torch.nn.functional as torch_f

    coarse_shape_xyz = (
        int(coarse_tensor.shape[4]),
        int(coarse_tensor.shape[3]),
        int(coarse_tensor.shape[2]),
    )
    grid = interpolation_grid_for_box(
        output_ranges_xyz=fine_bounds_xyz,
        input_shape_xyz=coarse_shape_xyz,
        output_shape_xyz=fine_shape_xyz,
        source_bounds_xyz=((0, coarse_shape_xyz[0]), (0, coarse_shape_xyz[1]), (0, coarse_shape_xyz[2])),
        device=coarse_tensor.device,
    )
    sampled = torch_f.grid_sample(
        coarse_tensor.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return torch_f.normalize(sampled, dim=1)


def sample_coarse_at_fine_points(coarse_tensor, fine_shape_xyz, fine_points_xyz):
    import torch
    import torch.nn.functional as torch_f

    points = np.asarray(fine_points_xyz, dtype=np.int64)
    coarse_shape_xyz = np.array(
        [coarse_tensor.shape[4], coarse_tensor.shape[3], coarse_tensor.shape[2]], dtype=np.int64
    )
    fine_shape = np.asarray(fine_shape_xyz, dtype=np.int64)
    source = (points.astype(np.float64) + 0.5) * coarse_shape_xyz / fine_shape - 0.5
    source = np.clip(source, 0.0, coarse_shape_xyz - 1.0)
    normalized = np.zeros_like(source, dtype=np.float32)
    for axis in range(3):
        if coarse_shape_xyz[axis] > 1:
            normalized[:, axis] = 2.0 * source[:, axis] / float(coarse_shape_xyz[axis] - 1) - 1.0
    grid = torch.as_tensor(normalized, device=coarse_tensor.device).view(1, len(points), 1, 1, 3)
    sampled = torch_f.grid_sample(
        coarse_tensor.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    sampled = sampled[0, :, :, 0, 0].transpose(0, 1)
    return torch_f.normalize(sampled, dim=1)


def extract_query_descriptors(cache: EmbeddingCache, points_xyz, device):
    import torch

    points = np.asarray(points_xyz, dtype=np.float64)
    fine_shape_xyz = np.asarray(cache.feature_shape_xyz("fine"), dtype=np.int64)
    fine_points = np.floor((points * cache.norm_ratio_xyz) / 2.0).astype(np.int64)
    fine_points = np.clip(fine_points, 0, fine_shape_xyz - 1)
    fine_array = cache.valid_array("fine")
    fine_vectors = np.stack(
        [fine_array[:, point[2], point[1], point[0]] for point in fine_points], axis=0
    ).astype(np.float32, copy=False)
    coarse_array = np.asarray(cache.valid_array("coarse"), dtype=np.float32)
    coarse_tensor = torch.from_numpy(coarse_array).unsqueeze(0).to(device=device)
    coarse_vectors = sample_coarse_at_fine_points(
        coarse_tensor, tuple(int(value) for value in fine_shape_xyz), fine_points
    )
    del coarse_tensor
    return torch.from_numpy(fine_vectors).to(device=device), coarse_vectors, fine_points


def stream_global_match(
    query_cache: EmbeddingCache,
    target_cache: EmbeddingCache,
    query_points_xyz,
    query_batch_size: int,
    match_chunk_xyz: Sequence[int],
    device=None,
):
    import torch
    import torch.nn.functional as torch_f

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Global streaming matching requires a CUDA GPU.")
        device = torch.device("cuda:0")
    else:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested matching device {device}, but CUDA is unavailable.")
    query_points = np.asarray(query_points_xyz, dtype=np.int64)
    query_fine, query_coarse, _ = extract_query_descriptors(query_cache, query_points, device)

    target_fine = target_cache.valid_array("fine")
    target_coarse_np = np.asarray(target_cache.valid_array("coarse"), dtype=np.float32)
    target_coarse = torch.from_numpy(target_coarse_np).unsqueeze(0).to(device=device)
    fine_shape_xyz = target_cache.feature_shape_xyz("fine")
    native_shape_xyz = target_cache.native_shape_xyz

    best_scores = np.full(len(query_points), -np.inf, dtype=np.float32)
    best_points = np.zeros((len(query_points), 3), dtype=np.int64)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    chunks = list(iter_chunks_xyz(native_shape_xyz, match_chunk_xyz))

    for chunk_index, ranges in enumerate(chunks, start=1):
        source_bounds = tuple(
            source_bounds_for_output_interval(
                ranges[axis][0], ranges[axis][1], fine_shape_xyz[axis], native_shape_xyz[axis]
            )
            for axis in range(3)
        )
        x_bounds, y_bounds, z_bounds = source_bounds
        fine_np = np.asarray(
            target_fine[
                :,
                z_bounds[0] : z_bounds[1],
                y_bounds[0] : y_bounds[1],
                x_bounds[0] : x_bounds[1],
            ],
            dtype=np.float32,
        )
        key_fine = torch.from_numpy(fine_np).unsqueeze(0).to(device=device)
        key_coarse = sample_coarse_for_fine_box(target_coarse, fine_shape_xyz, source_bounds)
        fine_flat = key_fine[0].reshape(key_fine.shape[1], -1)
        coarse_flat = key_coarse[0].reshape(key_coarse.shape[1], -1)
        native_grid = interpolation_grid_for_box(
            output_ranges_xyz=ranges,
            input_shape_xyz=fine_shape_xyz,
            output_shape_xyz=native_shape_xyz,
            source_bounds_xyz=source_bounds,
            device=device,
        )
        chunk_x = ranges[0][1] - ranges[0][0]
        chunk_y = ranges[1][1] - ranges[1][0]
        chunk_z = ranges[2][1] - ranges[2][0]

        for batch_start in range(0, len(query_points), query_batch_size):
            batch_stop = min(batch_start + query_batch_size, len(query_points))
            sim_fine = torch.matmul(query_fine[batch_start:batch_stop], fine_flat)
            sim_coarse = torch.matmul(query_coarse[batch_start:batch_stop], coarse_flat)
            sim_source = ((sim_fine + sim_coarse) * 0.5).view(
                batch_stop - batch_start,
                1,
                z_bounds[1] - z_bounds[0],
                y_bounds[1] - y_bounds[0],
                x_bounds[1] - x_bounds[0],
            )
            grid = native_grid.expand(batch_stop - batch_start, -1, -1, -1, -1)
            sim_native = torch_f.grid_sample(
                sim_source, grid, mode="bilinear", padding_mode="border", align_corners=True
            ).reshape(batch_stop - batch_start, -1)
            values, local_indices = torch.max(sim_native, dim=1)
            values_np = values.detach().cpu().numpy().astype(np.float32)
            local_np = local_indices.detach().cpu().numpy().astype(np.int64)
            local_x = local_np % chunk_x
            local_y = (local_np // chunk_x) % chunk_y
            local_z = local_np // (chunk_x * chunk_y)
            candidate_points = np.stack(
                (
                    local_x + ranges[0][0],
                    local_y + ranges[1][0],
                    local_z + ranges[2][0],
                ),
                axis=1,
            )
            current = best_scores[batch_start:batch_stop]
            improved = values_np > current
            current[improved] = values_np[improved]
            best_points[batch_start:batch_stop][improved] = candidate_points[improved]

            del sim_fine, sim_coarse, sim_source, sim_native, values, local_indices, grid

        del key_fine, key_coarse, fine_flat, coarse_flat, native_grid
        if chunk_index == 1 or chunk_index % 25 == 0 or chunk_index == len(chunks):
            print(f"  matching chunk {chunk_index}/{len(chunks)}")

    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    elapsed = float(time.time() - started)
    del query_fine, query_coarse, target_coarse
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_points, best_scores, {"seconds": elapsed, "peak_gpu_memory_bytes": peak_bytes}


def load_origin_mask(image_path: Path, mask_path: Path, is_mri: bool):
    from tools.quadra.rd_cycle_error_helper import validate_origin_mask
    from tools.utils import read_image

    image_info, _, _ = read_image(
        str(image_path), norm_spacing=NORM_SPACING_XYZ, mask_path=str(mask_path), is_MRI=is_mri
    )
    return validate_origin_mask(image_info.get("origin_mask"), image_info["img"], str(mask_path))


def sample_subject_points(pair, num_points: int, seed: int, organs: set[str] | None, is_mri: bool):
    from tools.quadra.rd_cycle_error_helper import sample_random_mask_points, validate_sampled_points_inside_mask

    records = []
    selected_masks = 0
    for mask_index, mask_path in enumerate(pair["masks"]):
        organ = strip_nii_suffix(mask_path.name).lower()
        if organs is not None and organ not in organs:
            continue
        mask = load_origin_mask(pair["test"], mask_path, is_mri=is_mri)
        points = sample_random_mask_points(mask, num_points, seed + mask_index)
        validate_sampled_points_inside_mask(points, mask, f"{pair['subject_id']}:{mask_path.name}")
        for point in points:
            records.append(
                {
                    "subject_id": pair["subject_id"],
                    "mask_name": f"{pair['subject_id']}/{mask_path.name}",
                    "pt1_sam": np.asarray(point, dtype=np.int64),
                }
            )
        selected_masks += 1
    if selected_masks == 0:
        available = [strip_nii_suffix(path.name).lower() for path in pair["masks"]]
        raise ValueError(f"No requested organs found. Available organs: {available}")
    return records


def convert_results_to_raw_itk(internal_results, test_image: Path, retest_image: Path):
    import SimpleITK as sitk

    test_transform, test_shape = build_sam_to_raw_transform(str(test_image))
    retest_transform, retest_shape = build_sam_to_raw_transform(str(retest_image))
    test_itk = sitk.ReadImage(str(test_image))
    output = []
    for record in internal_results:
        pt1 = transform_point_xyz(record["pt1_sam"], test_transform, test_shape)
        pt2 = transform_point_xyz(record["pt2_sam"], retest_transform, retest_shape)
        pt1_back = transform_point_xyz(record["pt1_back_sam"], test_transform, test_shape)
        physical_1 = np.asarray(test_itk.TransformIndexToPhysicalPoint(tuple(int(v) for v in pt1)), dtype=float)
        physical_back = np.asarray(
            test_itk.TransformIndexToPhysicalPoint(tuple(int(v) for v in pt1_back)), dtype=float
        )
        converted = {
            "subject_id": record["subject_id"],
            "mask_name": record["mask_name"],
            "coord_space": COORD_SPACE_RAW_ITK,
            "pt1": pt1,
            "pt2": pt2,
            "pt1_back": pt1_back,
            "voxel_error": float(np.linalg.norm(pt1_back.astype(float) - pt1.astype(float))),
            "mm_error": float(np.linalg.norm(physical_back - physical_1)),
            "score_12": float(record["score_12"]),
            "score_21": float(record["score_21"]),
        }
        output.append(converted)
    return output


def build_sam_cycle_results(internal_results, raw_itk_results):
    """Build a debugging export in SAM display/native-voxel coordinates.

    The millimetre error is copied from the corresponding raw-ITK result because
    it is a physical-space measurement. The voxel error is recomputed from the
    SAM-coordinate Test and cycle-return points.
    """
    if len(internal_results) != len(raw_itk_results):
        raise ValueError("SAM and raw-ITK result counts do not match.")

    output = []
    for internal, raw_itk in zip(internal_results, raw_itk_results):
        pt1 = np.asarray(internal["pt1_sam"], dtype=np.int64)
        pt2 = np.asarray(internal["pt2_sam"], dtype=np.int64)
        pt1_back = np.asarray(internal["pt1_back_sam"], dtype=np.int64)
        output.append(
            {
                "subject_id": internal["subject_id"],
                "mask_name": internal["mask_name"],
                "coord_space": COORD_SPACE_SAM,
                "pt1": pt1,
                "pt2": pt2,
                "pt1_back": pt1_back,
                "voxel_error": float(np.linalg.norm(pt1_back.astype(float) - pt1.astype(float))),
                "mm_error": float(raw_itk["mm_error"]),
                "score_12": float(internal["score_12"]),
                "score_21": float(internal["score_21"]),
            }
        )
    return output


def write_query_points_raw_itk_csv(results, out_path):
    """Write only the raw-ITK Test query points required by registration."""
    fieldnames = [
        "idx",
        "mask_name",
        "subject_id",
        "pt1_x",
        "pt1_y",
        "pt1_z",
        "coord_space",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(results):
            coord_space = str(record.get("coord_space", ""))
            if coord_space != COORD_SPACE_RAW_ITK:
                raise ValueError(
                    f"Query point {idx} has coord_space={coord_space!r}; "
                    f"expected {COORD_SPACE_RAW_ITK!r}."
                )
            pt1 = np.asarray(record["pt1"], dtype=int)
            writer.writerow(
                {
                    "idx": idx,
                    "mask_name": str(record.get("mask_name", "")),
                    "subject_id": str(record.get("subject_id", "")),
                    "pt1_x": int(pt1[0]),
                    "pt1_y": int(pt1[1]),
                    "pt1_z": int(pt1[2]),
                    "coord_space": coord_space,
                }
            )


def summarize_by_mask(results):
    from tools.quadra.rd_cycle_error_helper import compute_summary_stats

    grouped = {}
    for record in results:
        grouped.setdefault(record["mask_name"], []).append(record)
    rows = []
    for mask_name in sorted(grouped):
        voxel_stats, mm_stats = compute_summary_stats(grouped[mask_name])
        rows.append({"mask_name": mask_name, "voxel_stats": voxel_stats, "mm_stats": mm_stats})
    return rows


def run(args) -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command inside the configured RunPod GPU environment.")

    from tools.interfaces import init
    from tools.quadra.rd_cycle_error_helper import (
        compute_summary_stats,
        print_summary,
        write_points_csv_with_mask,
        write_summary_with_mask_labels_csv,
    )

    os.chdir(PROJECT_ROOT)
    dataset_root = Path(args.dataset_root).resolve()
    subject_id = canonical_subject_id(args.subject)
    pair = resolve_subject_pair(dataset_root, subject_id)
    config_identity = file_identity(args.config_file)
    checkpoint_identity = file_identity(args.checkpoint_file)
    checkpoint_name = Path(args.checkpoint_file).name
    checkpoint_role = "base_sam_engineering" if checkpoint_name == "SAM.pth" else "user_supplied"
    if checkpoint_role == "base_sam_engineering":
        print(
            "WARNING: checkpoints/SAM.pth is the original SAM checkpoint. "
            "This run is an engineering trial, not a Quadra fine-tuned result."
        )

    cache_key = f"{Path(args.checkpoint_file).stem}_{checkpoint_identity['sha256'][:12]}_2mm"
    subject_cache_root = Path(args.cache_root).resolve() / cache_key / subject_id
    test_cache_dir = subject_cache_root / "test"
    retest_cache_dir = subject_cache_root / "retest"

    test_existing = load_complete_manifest(test_cache_dir)
    retest_existing = load_complete_manifest(retest_cache_dir)
    model = None
    if args.overwrite_cache or test_existing is None or retest_existing is None:
        model = init(str(Path(args.config_file).resolve()), str(Path(args.checkpoint_file).resolve()))

    test_manifest = build_embedding_cache(
        pair["test"],
        test_cache_dir,
        model,
        config_identity,
        checkpoint_identity,
        args.tile_size,
        args.halo,
        args.overwrite_cache,
        args.is_mri,
    )
    retest_manifest = build_embedding_cache(
        pair["retest"],
        retest_cache_dir,
        model,
        config_identity,
        checkpoint_identity,
        args.tile_size,
        args.halo,
        args.overwrite_cache,
        args.is_mri,
    )
    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    organs = None if not args.organs else {value.lower() for value in args.organs}
    point_records = sample_subject_points(pair, args.num_points, args.seed, organs, args.is_mri)
    query_points = np.stack([record["pt1_sam"] for record in point_records], axis=0)
    print(f"Matching {len(query_points)} points across {len(set(r['mask_name'] for r in point_records))} masks")

    test_cache = EmbeddingCache(test_cache_dir)
    retest_cache = EmbeddingCache(retest_cache_dir)
    pt2, score_12, forward_profile = stream_global_match(
        test_cache,
        retest_cache,
        query_points,
        query_batch_size=args.query_batch_size,
        match_chunk_xyz=args.match_chunk_size,
    )
    pt1_back, score_21, backward_profile = stream_global_match(
        retest_cache,
        test_cache,
        pt2,
        query_batch_size=args.query_batch_size,
        match_chunk_xyz=args.match_chunk_size,
    )

    internal_results = []
    for index, point_record in enumerate(point_records):
        internal_results.append(
            {
                **point_record,
                "pt2_sam": pt2[index],
                "pt1_back_sam": pt1_back[index],
                "score_12": score_12[index],
                "score_21": score_21[index],
            }
        )
    results = convert_results_to_raw_itk(internal_results, pair["test"], pair["retest"])
    sam_results = build_sam_cycle_results(internal_results, results)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).resolve() / f"{subject_id}_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    points_path = output_dir / "cycle_points.csv"
    sam_points_path = output_dir / "cycle_points_sam.csv"
    query_path = output_dir / "query_points_raw_itk.csv"
    summary_path = output_dir / "cycle_summary.csv"
    write_points_csv_with_mask(results, str(points_path))
    write_points_csv_with_mask(sam_results, str(sam_points_path))
    write_query_points_raw_itk_csv(results, str(query_path))
    global_voxel, global_mm = compute_summary_stats(results)
    per_mask_rows = summarize_by_mask(results)
    write_summary_with_mask_labels_csv(
        per_mask_rows,
        str(summary_path),
        global_voxel_stats=global_voxel,
        global_mm_stats=global_mm,
        all_masks_label="ALL_MASKS",
    )
    print_summary(results)

    run_manifest = {
        "schema_version": 2,
        "created_at": utc_now(),
        "subject_id": subject_id,
        "method": "uae_tiled_embedding_exact_global_streaming_match",
        "result_status": "engineering_trial" if checkpoint_role == "base_sam_engineering" else "model_run",
        "checkpoint_role": checkpoint_role,
        "config": config_identity,
        "checkpoint": checkpoint_identity,
        "dataset_root": str(dataset_root),
        "test_image": str(pair["test"]),
        "retest_image": str(pair["retest"]),
        "norm_spacing_xyz": list(NORM_SPACING_XYZ),
        "tile_size_xyz": list(args.tile_size),
        "halo_xyz": list(args.halo),
        "match_chunk_xyz": list(args.match_chunk_size),
        "query_batch_size": int(args.query_batch_size),
        "similarity_compute_dtype": "float32",
        "num_points_per_mask": int(args.num_points),
        "seed": int(args.seed),
        "organs": sorted(organs) if organs is not None else "all",
        "coord_space": COORD_SPACE_RAW_ITK,
        "cycle_error_units": "native_voxels_and_physical_mm",
        "test_cache_manifest": test_manifest,
        "retest_cache_manifest": retest_manifest,
        "forward_matching": forward_profile,
        "backward_matching": backward_profile,
        "outputs": {
            "points": str(points_path),
            "points_raw_itk": str(points_path),
            "points_sam": str(sam_points_path),
            "query_points_for_registration": str(query_path),
            "summary": str(summary_path),
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(f"Streaming cycle output: {output_dir}")
    return output_dir


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT, help="Subject id; quadra_hc_21 is normalized to 021.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--tile-size", nargs=3, type=int, default=DEFAULT_TILE_SIZE_XYZ, metavar=("X", "Y", "Z"))
    parser.add_argument("--halo", nargs=3, type=int, default=DEFAULT_HALO_XYZ, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--match-chunk-size",
        nargs=3,
        type=int,
        default=DEFAULT_MATCH_CHUNK_XYZ,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--organs", nargs="*", default=None, help="Optional organ names; default processes all masks.")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--is-mri", action="store_true")
    args = parser.parse_args(argv)
    for name in ("tile_size", "halo", "match_chunk_size"):
        setattr(args, name, tuple(int(value) for value in getattr(args, name)))
    if args.query_batch_size < 1:
        parser.error("--query-batch-size must be at least 1")
    if args.num_points < 1:
        parser.error("--num-points must be at least 1")
    build_tile_plan((128, 128, 64), tile_size_xyz=args.tile_size, halo_xyz=args.halo)
    return args


def main(argv: Iterable[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
