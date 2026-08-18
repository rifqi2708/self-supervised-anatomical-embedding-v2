"""CPU-only 3D interior keypoint pilot for one Quadra organ.

This command deliberately stops after candidate detection, deterministic point
selection, and visual review. It does not import UAE, create embeddings, match
points, or calculate cycle error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np


DEFAULT_SCALES_MM = (2.0, 2.5, 3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Detect and visualize CPU-only 3D Harris-Laplacian query points "
            "inside one native-grid Quadra organ mask."
        )
    )
    parser.add_argument("--ct", required=True, help="Native-grid Test CT NIfTI.")
    parser.add_argument("--mask", required=True, help="Aligned binary organ mask NIfTI.")
    parser.add_argument("--organ", default="liver")
    parser.add_argument("--subject", default="quadra_hc_021")
    parser.add_argument("--timepoint", default="test", choices=("test", "retest"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--crop-padding-mm", type=float, default=10.0)
    parser.add_argument("--interior-margin-mm", type=float, default=5.0)
    parser.add_argument("--suppression-radius-mm", type=float, default=3.0)
    parser.add_argument(
        "--scales-mm",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES_MM),
        help="Physical Harris-Laplacian scales in millimetres.",
    )
    parser.add_argument(
        "--harris-quantile",
        type=float,
        default=0.5,
        help="Per-scale positive Harris-response quantile used for raw candidates.",
    )
    parser.add_argument("--harris-k", type=float, default=0.005)
    parser.add_argument("--max-candidates-per-scale", type=int, default=3000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_parameters(args):
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if args.window_width <= 0:
        raise ValueError("--window-width must be positive")
    if args.crop_padding_mm < 0 or args.interior_margin_mm < 0:
        raise ValueError("Crop padding and interior margin must be non-negative")
    if args.suppression_radius_mm <= 0:
        raise ValueError("--suppression-radius-mm must be positive")
    scales = np.asarray(args.scales_mm, dtype=np.float64)
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all():
        raise ValueError("--scales-mm must contain finite values")
    if np.any(scales <= 0) or np.any(np.diff(scales) <= 0):
        raise ValueError("--scales-mm must be positive and strictly increasing")
    if not 0.0 < args.harris_quantile < 1.0:
        raise ValueError("--harris-quantile must be between zero and one")
    if args.harris_k <= 0:
        raise ValueError("--harris-k must be positive")
    if args.max_candidates_per_scale <= 0:
        raise ValueError("--max-candidates-per-scale must be positive")


def mask_bounding_box(mask):
    foreground = np.argwhere(np.asarray(mask, dtype=bool))
    if not foreground.size:
        raise ValueError("Organ mask is empty")
    return foreground.min(axis=0), foreground.max(axis=0) + 1


def padded_crop_slices(mask, spacing_xyz_mm, padding_mm):
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise ValueError("Expected three positive finite voxel spacings")
    minimum, maximum = mask_bounding_box(mask)
    padding_voxels = np.ceil(float(padding_mm) / spacing).astype(np.int64)
    start = np.maximum(minimum - padding_voxels, 0)
    stop = np.minimum(maximum + padding_voxels, np.asarray(mask.shape, dtype=np.int64))
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop)), start


def window_ct(volume_hu, center_hu, width_hu):
    values = np.asarray(volume_hu, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Expected a three-dimensional CT crop")
    if not np.isfinite(values).all():
        raise ValueError("CT crop contains non-finite values")
    lower = float(center_hu) - float(width_hu) / 2.0
    normalized = (np.clip(values, lower, lower + float(width_hu)) - lower) / float(width_hu)
    return np.asarray(normalized, dtype=np.float32)


def boundary_distances_mm(mask, spacing_xyz_mm):
    from scipy.ndimage import distance_transform_edt

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3 or not binary.any():
        raise ValueError("Expected a non-empty three-dimensional mask")
    return np.asarray(
        distance_transform_edt(binary, sampling=np.asarray(spacing_xyz_mm, dtype=np.float64)),
        dtype=np.float32,
    )


def harris_response_3d(image, spacing_xyz_mm, scale_mm, harris_k):
    """Return a physical-scale 3D Harris response.

    The structure tensor uses Gaussian derivatives at ``scale_mm`` and an
    integration scale 1.5 times larger. The 3D response is
    ``det(M) - k * trace(M)^3``.
    """

    from scipy.ndimage import gaussian_filter

    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    derivative_sigma = np.maximum(float(scale_mm) / spacing, 0.5)
    integration_sigma = np.maximum(1.5 * float(scale_mm) / spacing, 0.5)
    smoothed = gaussian_filter(
        np.asarray(image, dtype=np.float32), derivative_sigma, mode="nearest"
    )
    gradients = np.gradient(smoothed, *spacing, edge_order=1)
    gx, gy, gz = [np.asarray(component, dtype=np.float32) for component in gradients]
    a = gaussian_filter(gx * gx, integration_sigma, mode="nearest")
    d = gaussian_filter(gy * gy, integration_sigma, mode="nearest")
    f = gaussian_filter(gz * gz, integration_sigma, mode="nearest")
    b = gaussian_filter(gx * gy, integration_sigma, mode="nearest")
    c = gaussian_filter(gx * gz, integration_sigma, mode="nearest")
    e = gaussian_filter(gy * gz, integration_sigma, mode="nearest")
    determinant = a * d * f + 2.0 * b * c * e - a * e * e - d * c * c - f * b * b
    trace = a + d + f
    response = determinant - float(harris_k) * trace * trace * trace
    return np.asarray(np.maximum(response, 0.0), dtype=np.float32)


def normalized_laplacian_3d(image, spacing_xyz_mm, scale_mm):
    from scipy.ndimage import gaussian_laplace

    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    sigma = np.maximum(float(scale_mm) / spacing, 0.5)
    response = np.abs(
        gaussian_laplace(np.asarray(image, dtype=np.float32), sigma=sigma, mode="nearest")
    )
    return np.asarray(response * float(scale_mm) ** 2, dtype=np.float32)


def local_maxima_above_quantile(response, eligible_mask, spacing_xyz_mm, radius_mm, quantile):
    from scipy.ndimage import maximum_filter

    values = np.asarray(response, dtype=np.float32)
    eligible = np.asarray(eligible_mask, dtype=bool)
    positive = values[eligible & (values > 0)]
    if not positive.size:
        return np.empty((0, 3), dtype=np.int64), math.nan
    threshold = float(np.quantile(positive, float(quantile)))
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    radius_voxels = np.maximum(np.ceil(float(radius_mm) / spacing).astype(np.int64), 1)
    size = tuple(int(2 * radius + 1) for radius in radius_voxels)
    maxima = maximum_filter(values, size=size, mode="nearest")
    coordinates = np.argwhere(eligible & (values >= threshold) & (values == maxima))
    return coordinates.astype(np.int64, copy=False), threshold


def detect_harris_laplacian_candidates(
    image,
    organ_mask,
    spacing_xyz_mm,
    scales_mm,
    harris_k=0.005,
    harris_quantile=0.985,
    local_radius_mm=2.0,
    max_candidates_per_scale=3000,
):
    """Detect candidates and retain Laplacian scale maxima.

    Results are returned in crop voxel coordinates. Harris responses are
    normalized within scale before scores from different scales are compared.
    """

    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(organ_mask, dtype=bool)
    scale_values = np.asarray(scales_mm, dtype=np.float64)
    records = []
    thresholds = []
    for scale_index, scale_mm in enumerate(scale_values):
        response = harris_response_3d(image, spacing_xyz_mm, scale_mm, harris_k)
        coordinates, threshold = local_maxima_above_quantile(
            response,
            mask,
            spacing_xyz_mm,
            local_radius_mm,
            harris_quantile,
        )
        thresholds.append(threshold)
        if not len(coordinates):
            continue
        values = response[tuple(coordinates.T)]
        order = np.argsort(-values, kind="stable")[: int(max_candidates_per_scale)]
        coordinates = coordinates[order]
        values = values[order]
        reference = float(np.quantile(response[mask & (response > 0)], 0.999))
        reference = max(reference, np.finfo(np.float32).tiny)
        for coordinate, value in zip(coordinates, values):
            records.append(
                {
                    "crop_xyz": coordinate,
                    "scale_index": int(scale_index),
                    "scale_mm": float(scale_mm),
                    "harris_raw": float(value),
                    "harris_normalized": float(value / reference),
                }
            )
        del response

    if not records:
        return [], thresholds

    all_coordinates = np.asarray([record["crop_xyz"] for record in records], dtype=np.int64)
    laplacian_by_scale = np.empty((len(records), len(scale_values)), dtype=np.float32)
    laplacian_references = []
    for scale_index, scale_mm in enumerate(scale_values):
        laplacian = normalized_laplacian_3d(image, spacing_xyz_mm, scale_mm)
        laplacian_by_scale[:, scale_index] = laplacian[tuple(all_coordinates.T)]
        positive = laplacian[mask & (laplacian > 0)]
        laplacian_references.append(
            max(float(np.quantile(positive, 0.999)), np.finfo(np.float32).tiny)
        )
        del laplacian

    accepted = []
    for row_index, record in enumerate(records):
        scale_index = record["scale_index"]
        own_value = float(laplacian_by_scale[row_index, scale_index])
        left = max(scale_index - 1, 0)
        right = min(scale_index + 2, len(scale_values))
        if own_value + np.finfo(np.float32).eps < float(
            np.max(laplacian_by_scale[row_index, left:right])
        ):
            continue
        record = dict(record)
        record["laplacian_raw"] = own_value
        record["laplacian_normalized"] = own_value / laplacian_references[scale_index]
        record["detector_score"] = (
            record["harris_normalized"] * record["laplacian_normalized"]
        )
        accepted.append(record)
    return accepted, thresholds


def greedy_radius_suppression(points_xyz, scores, spacing_xyz_mm, radius_mm):
    points = np.asarray(points_xyz, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected points with shape [N, 3]")
    if score_values.shape != (len(points),):
        raise ValueError("Score/point count mismatch")
    if not len(points):
        return np.empty((0,), dtype=np.int64)
    physical = points * np.asarray(spacing_xyz_mm, dtype=np.float64)[None, :]
    order = np.argsort(-score_values, kind="stable")
    retained = []
    radius_squared = float(radius_mm) ** 2
    for index in order:
        if retained:
            delta = physical[np.asarray(retained)] - physical[index]
            if np.any(np.sum(delta * delta, axis=1) < radius_squared):
                continue
        retained.append(int(index))
    return np.asarray(retained, dtype=np.int64)


def farthest_point_sample(points_xyz, scores, spacing_xyz_mm, count):
    """Select a deterministic spatial quota, seeded by the strongest candidate."""

    points = np.asarray(points_xyz, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if count <= 0:
        raise ValueError("Requested point count must be positive")
    if len(points) < count:
        raise ValueError(
            "Only {} eligible candidates remain; cannot select {} without replacement".format(
                len(points), count
            )
        )
    physical = points * np.asarray(spacing_xyz_mm, dtype=np.float64)[None, :]
    selected = [int(np.argmax(score_values))]
    delta = physical - physical[selected[0]]
    minimum_squared = np.sum(delta * delta, axis=1)
    minimum_squared[selected[0]] = -1.0
    while len(selected) < count:
        next_index = int(np.argmax(minimum_squared))
        selected.append(next_index)
        delta = physical - physical[next_index]
        minimum_squared = np.minimum(minimum_squared, np.sum(delta * delta, axis=1))
        minimum_squared[np.asarray(selected, dtype=np.int64)] = -1.0
    return np.asarray(selected, dtype=np.int64)


def _atomic_destination(path, overwrite=False):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite existing result: {}".format(destination))
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    return destination, temporary


def write_csv_atomic(path, rows, fieldnames, overwrite=False):
    destination, temporary = _atomic_destination(path, overwrite=overwrite)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(destination))


def write_json_atomic(path, payload, overwrite=False):
    destination, temporary = _atomic_destination(path, overwrite=overwrite)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(destination))


def _mask_boundary_2d(mask):
    from scipy.ndimage import binary_erosion

    binary = np.asarray(mask, dtype=bool)
    return binary & ~binary_erosion(binary)


def _save_figure_atomic(figure, path, overwrite=False):
    destination, temporary = _atomic_destination(path, overwrite=overwrite)
    figure.savefig(str(temporary), dpi=180, bbox_inches="tight", format="png")
    os.replace(str(temporary), str(destination))


def plot_axial_montage(
    image,
    mask,
    selected_crop_xyz,
    distances_mm,
    crop_start_xyz,
    spacing_xyz_mm,
    output_path,
    title,
    overwrite=False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(selected_crop_xyz, dtype=np.int64)
    occupied = np.unique(points[:, 2])
    quantile_indices = np.rint(np.linspace(0, len(occupied) - 1, min(6, len(occupied)))).astype(np.int64)
    slices = occupied[quantile_indices]
    slab_half_width_mm = 5.0
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    marker = None
    for axis, z_index in zip(axes.flat, slices):
        axis.imshow(image[:, :, z_index].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        boundary = _mask_boundary_2d(mask[:, :, z_index].T)
        axis.contour(boundary, levels=[0.5], colors=["#00d5ff"], linewidths=0.8)
        on_slice = np.abs(points[:, 2] - z_index) * spacing[2] <= slab_half_width_mm
        marker = axis.scatter(
            points[on_slice, 0],
            points[on_slice, 1],
            c=np.asarray(distances_mm)[on_slice],
            cmap="plasma",
            vmin=float(np.min(distances_mm)),
            vmax=float(np.max(distances_mm)),
            s=30,
            edgecolors="white",
            linewidths=0.5,
        )
        axis.set_title(
            "native z={} | {} points in +/-{} mm slab".format(
                int(z_index + crop_start_xyz[2]), int(on_slice.sum()), int(slab_half_width_mm)
            )
        )
        axis.set_axis_off()
    for axis in axes.flat[len(slices) :]:
        axis.set_axis_off()
    figure.suptitle(title)
    if marker is not None:
        figure.colorbar(marker, ax=axes, shrink=0.75, label="distance inside liver boundary (mm)")
    _save_figure_atomic(figure, output_path, overwrite=overwrite)
    plt.close(figure)


def plot_orthogonal_views(
    image,
    mask,
    selected_crop_xyz,
    distances_mm,
    crop_start_xyz,
    spacing_xyz_mm,
    output_path,
    title,
    overwrite=False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(selected_crop_xyz, dtype=np.int64)
    centres = np.rint(np.median(points, axis=0)).astype(np.int64)
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    slab_half_width_mm = 5.0
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    panels = [
        (
            image[:, :, centres[2]].T,
            mask[:, :, centres[2]].T,
            points[:, [0, 1]],
            np.abs(points[:, 2] - centres[2]) * spacing[2] <= slab_half_width_mm,
            "axial z",
            2,
        ),
        (
            image[:, centres[1], :].T,
            mask[:, centres[1], :].T,
            points[:, [0, 2]],
            np.abs(points[:, 1] - centres[1]) * spacing[1] <= slab_half_width_mm,
            "coronal y",
            1,
        ),
        (
            image[centres[0], :, :].T,
            mask[centres[0], :, :].T,
            points[:, [1, 2]],
            np.abs(points[:, 0] - centres[0]) * spacing[0] <= slab_half_width_mm,
            "sagittal x",
            0,
        ),
    ]
    marker = None
    for axis, (slice_image, slice_mask, projected, on_slice, label, coordinate_axis) in zip(axes, panels):
        axis.imshow(slice_image, cmap="gray", origin="lower", vmin=0, vmax=1)
        axis.contour(_mask_boundary_2d(slice_mask), levels=[0.5], colors=["#00d5ff"], linewidths=1.0)
        marker = axis.scatter(
            projected[on_slice, 0],
            projected[on_slice, 1],
            c=np.asarray(distances_mm)[on_slice],
            cmap="plasma",
            vmin=float(np.min(distances_mm)),
            vmax=float(np.max(distances_mm)),
            s=36,
            edgecolors="white",
            linewidths=0.6,
        )
        native_coordinate = int(centres[coordinate_axis] + crop_start_xyz[coordinate_axis])
        axis.set_title(
            "{}={} | {} points in +/-{} mm slab".format(
                label, native_coordinate, int(on_slice.sum()), int(slab_half_width_mm)
            )
        )
        axis.set_axis_off()
    figure.suptitle(title)
    if marker is not None:
        figure.colorbar(marker, ax=axes, shrink=0.7, label="distance inside liver boundary (mm)")
    _save_figure_atomic(figure, output_path, overwrite=overwrite)
    plt.close(figure)


def plot_boundary_distance_histogram(raw_distances, selected_distances, margin_mm, output_path, overwrite=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.hist(raw_distances, bins=30, alpha=0.45, label="raw Harris-Laplacian candidates")
    axis.hist(selected_distances, bins=20, alpha=0.75, label="final FPS query points")
    axis.axvline(float(margin_mm), color="black", linestyle="--", label="interior margin")
    axis.set_xlabel("distance inside liver boundary (mm)")
    axis.set_ylabel("point count")
    axis.set_title("Liver candidate location relative to organ boundary")
    axis.legend()
    _save_figure_atomic(figure, output_path, overwrite=overwrite)
    plt.close(figure)


def _candidate_rows(records, crop_start, affine, boundary_distance, selected_indices, args):
    selected_set = set(int(index) for index in selected_indices)
    rows = []
    for index, record in enumerate(records):
        crop_xyz = np.asarray(record["crop_xyz"], dtype=np.int64)
        raw_xyz = crop_xyz + np.asarray(crop_start, dtype=np.int64)
        ras_xyz = np.asarray(affine, dtype=np.float64).dot(np.r_[raw_xyz, 1.0])[:3]
        rows.append(
            {
                "candidate_id": int(index),
                "subject_id": args.subject,
                "timepoint": args.timepoint,
                "organ": args.organ,
                "raw_x_voxel": int(raw_xyz[0]),
                "raw_y_voxel": int(raw_xyz[1]),
                "raw_z_voxel": int(raw_xyz[2]),
                "physical_ras_x_mm": float(ras_xyz[0]),
                "physical_ras_y_mm": float(ras_xyz[1]),
                "physical_ras_z_mm": float(ras_xyz[2]),
                "scale_mm": float(record["scale_mm"]),
                "harris_score": float(record["harris_normalized"]),
                "laplacian_score": float(record["laplacian_normalized"]),
                "detector_score": float(record["detector_score"]),
                "boundary_distance_mm": float(boundary_distance[tuple(crop_xyz)]),
                "passes_interior_margin": bool(
                    boundary_distance[tuple(crop_xyz)] >= args.interior_margin_mm
                ),
                "selected_query_point": index in selected_set,
                "coordinate_space": "native_nifti_voxel_xyz",
            }
        )
    return rows


CSV_FIELDS = [
    "candidate_id",
    "subject_id",
    "timepoint",
    "organ",
    "raw_x_voxel",
    "raw_y_voxel",
    "raw_z_voxel",
    "physical_ras_x_mm",
    "physical_ras_y_mm",
    "physical_ras_z_mm",
    "scale_mm",
    "harris_score",
    "laplacian_score",
    "detector_score",
    "boundary_distance_mm",
    "passes_interior_margin",
    "selected_query_point",
    "coordinate_space",
]


def run(args):
    validate_parameters(args)
    try:
        import nibabel as nib
        import scipy
    except ImportError as exc:
        raise RuntimeError("nibabel and scipy are required for this CPU pilot") from exc

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Output directory is not empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    ct_path = Path(args.ct).resolve()
    mask_path = Path(args.mask).resolve()
    ct = nib.load(str(ct_path))
    mask_image = nib.load(str(mask_path))
    if ct.ndim != 3 or mask_image.ndim != 3:
        raise ValueError("CT and mask must both be three-dimensional")
    if ct.shape != mask_image.shape:
        raise ValueError("Mask shape {} does not match CT {}".format(mask_image.shape, ct.shape))
    if not np.allclose(ct.affine, mask_image.affine, rtol=0.0, atol=1e-5):
        raise ValueError("Mask affine does not match CT affine")
    spacing = np.asarray(ct.header.get_zooms()[:3], dtype=np.float64)
    mask_full = np.asarray(mask_image.dataobj) > 0
    crop_slices, crop_start = padded_crop_slices(mask_full, spacing, args.crop_padding_mm)
    mask_crop = np.asarray(mask_full[crop_slices], dtype=bool)
    ct_crop_hu = np.asarray(ct.dataobj[crop_slices], dtype=np.float32)
    image = window_ct(ct_crop_hu, args.window_center, args.window_width)
    boundary_distance = boundary_distances_mm(mask_crop, spacing)

    records, harris_thresholds = detect_harris_laplacian_candidates(
        image,
        mask_crop,
        spacing,
        args.scales_mm,
        harris_k=args.harris_k,
        harris_quantile=args.harris_quantile,
        local_radius_mm=2.0,
        max_candidates_per_scale=args.max_candidates_per_scale,
    )
    if not records:
        raise RuntimeError("The detector produced no liver candidates")
    coordinates = np.asarray([record["crop_xyz"] for record in records], dtype=np.int64)
    scores = np.asarray([record["detector_score"] for record in records], dtype=np.float64)
    raw_distances = boundary_distance[tuple(coordinates.T)]
    interior_indices = np.flatnonzero(raw_distances >= float(args.interior_margin_mm))
    if not len(interior_indices):
        raise RuntimeError("No candidates pass the interior-margin gate")
    suppressed_local = greedy_radius_suppression(
        coordinates[interior_indices],
        scores[interior_indices],
        spacing,
        args.suppression_radius_mm,
    )
    suppressed_indices = interior_indices[suppressed_local]
    fps_local = farthest_point_sample(
        coordinates[suppressed_indices],
        scores[suppressed_indices],
        spacing,
        args.num_points,
    )
    selected_indices = suppressed_indices[fps_local]
    selected_coordinates = coordinates[selected_indices]
    selected_distances = boundary_distance[tuple(selected_coordinates.T)]

    rows = _candidate_rows(
        records,
        crop_start,
        ct.affine,
        boundary_distance,
        selected_indices,
        args,
    )
    query_rows = [dict(rows[int(index)]) for index in selected_indices]
    for point_id, row in enumerate(query_rows):
        row["candidate_id"] = point_id

    write_csv_atomic(output_dir / "raw_candidates.csv", rows, CSV_FIELDS, overwrite=args.overwrite)
    write_csv_atomic(
        output_dir / "query_points.csv", query_rows, CSV_FIELDS, overwrite=args.overwrite
    )
    title = (
        "{} {} {} | 3D Harris-Laplacian + {} mm interior + FPS (n={})".format(
            args.subject,
            args.timepoint,
            args.organ,
            args.interior_margin_mm,
            args.num_points,
        )
    )
    plot_axial_montage(
        image,
        mask_crop,
        selected_coordinates,
        selected_distances,
        crop_start,
        spacing,
        output_dir / "query_points_axial_montage.png",
        title,
        overwrite=args.overwrite,
    )
    plot_orthogonal_views(
        image,
        mask_crop,
        selected_coordinates,
        selected_distances,
        crop_start,
        spacing,
        output_dir / "query_points_orthogonal.png",
        title,
        overwrite=args.overwrite,
    )
    plot_boundary_distance_histogram(
        raw_distances,
        selected_distances,
        args.interior_margin_mm,
        output_dir / "boundary_distance_histogram.png",
        overwrite=args.overwrite,
    )

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "subject021_test_liver_query_generation_and_visualization_only",
        "explicitly_not_run": [
            "uae_model_loading",
            "embedding_generation",
            "point_matching",
            "cycle_error",
        ],
        "subject_id": args.subject,
        "timepoint": args.timepoint,
        "organ": args.organ,
        "ct_path": str(ct_path),
        "mask_path": str(mask_path),
        "native_shape_xyz": [int(value) for value in ct.shape],
        "spacing_xyz_mm": [float(value) for value in spacing],
        "orientation_codes": [str(value) for value in nib.aff2axcodes(ct.affine)],
        "crop_start_xyz": [int(value) for value in crop_start],
        "crop_shape_xyz": [int(value) for value in image.shape],
        "window": {"center_hu": args.window_center, "width_hu": args.window_width},
        "detector": {
            "name": "3d_harris_laplacian",
            "parameter_status": "exploratory_tuned_on_subject021_test_liver",
            "scales_mm": [float(value) for value in args.scales_mm],
            "harris_k": args.harris_k,
            "harris_quantile": args.harris_quantile,
            "harris_thresholds": [
                None if not np.isfinite(value) else float(value) for value in harris_thresholds
            ],
        },
        "selection": {
            "interior_margin_mm": args.interior_margin_mm,
            "suppression_radius_mm": args.suppression_radius_mm,
            "method": "deterministic_farthest_point_sampling",
            "requested_query_points": args.num_points,
        },
        "counts": {
            "raw_candidates": len(records),
            "passing_interior_margin": int(len(interior_indices)),
            "after_radius_suppression": int(len(suppressed_indices)),
            "selected_query_points": int(len(selected_indices)),
        },
        "boundary_distance_mm": {
            "raw_median": float(np.median(raw_distances)),
            "raw_p05": float(np.percentile(raw_distances, 5)),
            "selected_min": float(np.min(selected_distances)),
            "selected_median": float(np.median(selected_distances)),
            "selected_p95": float(np.percentile(selected_distances, 95)),
        },
        "runtime_seconds": float(elapsed),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "outputs": {
            "query_points_csv": "query_points.csv",
            "raw_candidates_csv": "raw_candidates.csv",
            "axial_montage_png": "query_points_axial_montage.png",
            "orthogonal_png": "query_points_orthogonal.png",
            "boundary_histogram_png": "boundary_distance_histogram.png",
        },
        "interpretation_limit": (
            "This gate assesses detector location and spatial coverage only. It does not "
            "show that the points improve UAE correspondence accuracy. Detector defaults "
            "were selected after inspecting subject 021 liver candidate supply and are not "
            "an independently validated universal policy."
        ),
    }
    write_json_atomic(output_dir / "summary.json", summary, overwrite=args.overwrite)
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
