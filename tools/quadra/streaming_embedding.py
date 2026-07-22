"""Pure geometry helpers for tiled Quadra embedding and streaming matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


FINE_STRIDE_XYZ = (2, 2, 2)
COARSE_STRIDE_XYZ = (16, 16, 4)

# Validated production geometry. The larger encoder input increases anatomical
# context while retaining the same core and therefore the same tile grid as the
# original 128x128x64 / 32x32x16 configuration.
RECOMMENDED_TILE_SIZE_XYZ = (160, 160, 80)
RECOMMENDED_HALO_XYZ = (48, 48, 24)
RECOMMENDED_CORE_SIZE_XYZ = (64, 64, 32)


def _triple(values: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three x,y,z values, got {values!r}")
    triple = tuple(int(value) for value in values)
    if any(value <= 0 for value in triple):
        raise ValueError(f"{name} values must be positive, got {triple}")
    return triple


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def retained_core_size_xyz(
    tile_size_xyz: Sequence[int] = RECOMMENDED_TILE_SIZE_XYZ,
    halo_xyz: Sequence[int] = RECOMMENDED_HALO_XYZ,
) -> tuple[int, int, int]:
    """Return the central region retained from each encoder tile."""
    tile = _triple(tile_size_xyz, "tile_size_xyz")
    halo = _triple(halo_xyz, "halo_xyz")
    core = tuple(tile[axis] - 2 * halo[axis] for axis in range(3))
    if any(value <= 0 for value in core):
        raise ValueError(f"tile_size_xyz must be larger than two halos; tile={tile}, halo={halo}")
    return core


def embedding_geometry_namespace(
    tile_size_xyz: Sequence[int] = RECOMMENDED_TILE_SIZE_XYZ,
    halo_xyz: Sequence[int] = RECOMMENDED_HALO_XYZ,
) -> str:
    """Return a stable cache namespace for one tile/halo/core geometry."""
    tile = _triple(tile_size_xyz, "tile_size_xyz")
    halo = _triple(halo_xyz, "halo_xyz")
    core = retained_core_size_xyz(tile, halo)

    def label(values: Sequence[int]) -> str:
        return "x".join(str(int(value)) for value in values)

    return f"tile{label(tile)}_halo{label(halo)}_core{label(core)}"


@dataclass(frozen=True)
class TilePlan:
    """Regular central-crop tiling plan expressed in x,y,z order."""

    volume_shape_xyz: tuple[int, int, int]
    tile_size_xyz: tuple[int, int, int]
    halo_xyz: tuple[int, int, int]
    core_size_xyz: tuple[int, int, int]
    grid_shape_xyz: tuple[int, int, int]
    padded_shape_xyz: tuple[int, int, int]
    stored_fine_shape_xyz: tuple[int, int, int]
    valid_fine_shape_xyz: tuple[int, int, int]
    stored_coarse_shape_xyz: tuple[int, int, int]
    valid_coarse_shape_xyz: tuple[int, int, int]

    @property
    def tile_count(self) -> int:
        return int(np.prod(self.grid_shape_xyz))

    def to_dict(self) -> dict[str, object]:
        return {
            "volume_shape_xyz": list(self.volume_shape_xyz),
            "tile_size_xyz": list(self.tile_size_xyz),
            "halo_xyz": list(self.halo_xyz),
            "core_size_xyz": list(self.core_size_xyz),
            "grid_shape_xyz": list(self.grid_shape_xyz),
            "padded_shape_xyz": list(self.padded_shape_xyz),
            "stored_fine_shape_xyz": list(self.stored_fine_shape_xyz),
            "valid_fine_shape_xyz": list(self.valid_fine_shape_xyz),
            "stored_coarse_shape_xyz": list(self.stored_coarse_shape_xyz),
            "valid_coarse_shape_xyz": list(self.valid_coarse_shape_xyz),
            "tile_count": self.tile_count,
        }


@dataclass(frozen=True)
class TileLocation:
    grid_index_xyz: tuple[int, int, int]
    padded_input_slices_zyx: tuple[slice, slice, slice]
    fine_source_slices_zyx: tuple[slice, slice, slice]
    fine_destination_slices_zyx: tuple[slice, slice, slice]
    coarse_source_slices_zyx: tuple[slice, slice, slice]
    coarse_destination_slices_zyx: tuple[slice, slice, slice]


def build_tile_plan(
    volume_shape_xyz: Sequence[int],
    tile_size_xyz: Sequence[int] = RECOMMENDED_TILE_SIZE_XYZ,
    halo_xyz: Sequence[int] = RECOMMENDED_HALO_XYZ,
) -> TilePlan:
    volume = _triple(volume_shape_xyz, "volume_shape_xyz")
    tile = _triple(tile_size_xyz, "tile_size_xyz")
    halo = _triple(halo_xyz, "halo_xyz")
    core = retained_core_size_xyz(tile, halo)

    for axis, axis_name in enumerate("xyz"):
        alignment = COARSE_STRIDE_XYZ[axis]
        for label, values in (("tile", tile), ("halo", halo), ("core", core)):
            if values[axis] % alignment != 0:
                raise ValueError(
                    f"{label} {axis_name}={values[axis]} must be divisible by the model alignment "
                    f"stride {alignment}; tile={tile}, halo={halo}, core={core}"
                )

    grid = tuple(ceil_div(volume[axis], core[axis]) for axis in range(3))
    covered_core = tuple(grid[axis] * core[axis] for axis in range(3))
    padded = tuple(covered_core[axis] + 2 * halo[axis] for axis in range(3))

    def feature_shapes(stride_xyz: tuple[int, int, int]):
        stored = tuple(covered_core[axis] // stride_xyz[axis] for axis in range(3))
        valid = tuple(ceil_div(volume[axis], stride_xyz[axis]) for axis in range(3))
        return stored, valid

    stored_fine, valid_fine = feature_shapes(FINE_STRIDE_XYZ)
    stored_coarse, valid_coarse = feature_shapes(COARSE_STRIDE_XYZ)
    return TilePlan(
        volume_shape_xyz=volume,
        tile_size_xyz=tile,
        halo_xyz=halo,
        core_size_xyz=core,
        grid_shape_xyz=grid,
        padded_shape_xyz=padded,
        stored_fine_shape_xyz=stored_fine,
        valid_fine_shape_xyz=valid_fine,
        stored_coarse_shape_xyz=stored_coarse,
        valid_coarse_shape_xyz=valid_coarse,
    )


def iter_tile_locations(plan: TilePlan) -> Iterator[TileLocation]:
    tile_x, tile_y, tile_z = plan.tile_size_xyz
    halo_x, halo_y, halo_z = plan.halo_xyz
    core_x, core_y, core_z = plan.core_size_xyz
    grid_x, grid_y, grid_z = plan.grid_shape_xyz

    for grid_z_index in range(grid_z):
        for grid_y_index in range(grid_y):
            for grid_x_index in range(grid_x):
                start_x = grid_x_index * core_x
                start_y = grid_y_index * core_y
                start_z = grid_z_index * core_z

                def feature_slices(stride_xyz: tuple[int, int, int]):
                    stride_x, stride_y, stride_z = stride_xyz
                    source = (
                        slice(halo_z // stride_z, (halo_z + core_z) // stride_z),
                        slice(halo_y // stride_y, (halo_y + core_y) // stride_y),
                        slice(halo_x // stride_x, (halo_x + core_x) // stride_x),
                    )
                    destination = (
                        slice(start_z // stride_z, (start_z + core_z) // stride_z),
                        slice(start_y // stride_y, (start_y + core_y) // stride_y),
                        slice(start_x // stride_x, (start_x + core_x) // stride_x),
                    )
                    return source, destination

                fine_source, fine_destination = feature_slices(FINE_STRIDE_XYZ)
                coarse_source, coarse_destination = feature_slices(COARSE_STRIDE_XYZ)
                yield TileLocation(
                    grid_index_xyz=(grid_x_index, grid_y_index, grid_z_index),
                    padded_input_slices_zyx=(
                        slice(start_z, start_z + tile_z),
                        slice(start_y, start_y + tile_y),
                        slice(start_x, start_x + tile_x),
                    ),
                    fine_source_slices_zyx=fine_source,
                    fine_destination_slices_zyx=fine_destination,
                    coarse_source_slices_zyx=coarse_source,
                    coarse_destination_slices_zyx=coarse_destination,
                )


def iter_chunks_xyz(shape_xyz: Sequence[int], chunk_size_xyz: Sequence[int]):
    shape_x, shape_y, shape_z = _triple(shape_xyz, "shape_xyz")
    chunk_x, chunk_y, chunk_z = _triple(chunk_size_xyz, "chunk_size_xyz")
    for z_start in range(0, shape_z, chunk_z):
        z_stop = min(z_start + chunk_z, shape_z)
        for y_start in range(0, shape_y, chunk_y):
            y_stop = min(y_start + chunk_y, shape_y)
            for x_start in range(0, shape_x, chunk_x):
                x_stop = min(x_start + chunk_x, shape_x)
                yield (x_start, x_stop), (y_start, y_stop), (z_start, z_stop)


def align_corners_false_source_positions(
    output_start: int,
    output_stop: int,
    input_size: int,
    output_size: int,
) -> np.ndarray:
    """Return PyTorch interpolate source coordinates for an output interval."""
    output_indices = np.arange(int(output_start), int(output_stop), dtype=np.float64)
    positions = (output_indices + 0.5) * float(input_size) / float(output_size) - 0.5
    return np.clip(positions, 0.0, float(input_size - 1))


def source_bounds_for_output_interval(
    output_start: int,
    output_stop: int,
    input_size: int,
    output_size: int,
) -> tuple[int, int]:
    positions = align_corners_false_source_positions(output_start, output_stop, input_size, output_size)
    source_start = int(math.floor(float(positions.min())))
    source_stop = int(math.ceil(float(positions.max()))) + 1
    return source_start, min(source_stop, int(input_size))


def flattened_zyx_index(point_xyz: Sequence[int], shape_xyz: Sequence[int]) -> int:
    x, y, z = (int(value) for value in point_xyz)
    shape_x, shape_y, shape_z = _triple(shape_xyz, "shape_xyz")
    if x < 0 or y < 0 or z < 0 or x >= shape_x or y >= shape_y or z >= shape_z:
        raise ValueError(f"Point {(x, y, z)} is outside shape {(shape_x, shape_y, shape_z)}")
    return (z * shape_y + y) * shape_x + x
