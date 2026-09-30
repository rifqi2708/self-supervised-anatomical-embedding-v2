"""Resumable subject-level orchestration for the interior-keypoint pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np

try:
    from tools.quadra.interior_keypoint_gate import (
        CSV_FIELDS,
        _save_figure_atomic,
        boundary_distances_mm,
        detect_harris_laplacian_candidates,
        draw_gap_crosshair,
        farthest_point_sample,
        greedy_radius_suppression,
        padded_crop_slices,
        plot_axial_montage,
        validate_parameters,
        window_ct,
        write_csv_atomic,
        write_json_atomic,
    )
    from tools.quadra.superpoint_adapter import sha256_file
except ModuleNotFoundError:  # direct-path entrypoint from repository root
    from interior_keypoint_gate import (
        CSV_FIELDS,
        _save_figure_atomic,
        boundary_distances_mm,
        detect_harris_laplacian_candidates,
        draw_gap_crosshair,
        farthest_point_sample,
        greedy_radius_suppression,
        padded_crop_slices,
        plot_axial_montage,
        validate_parameters,
        window_ct,
        write_csv_atomic,
        write_json_atomic,
    )
    from superpoint_adapter import sha256_file


SCHEMA_VERSION = 2
WINDOW_PRESETS = {
    "soft_tissue": {"center_hu": 40.0, "width_hu": 400.0},
    "lung": {"center_hu": -600.0, "width_hu": 1500.0},
    "bone": {"center_hu": 500.0, "width_hu": 2000.0},
    "brain": {"center_hu": 40.0, "width_hu": 80.0},
}
BONE_EXACT = {"skull", "sacrum", "hip_left", "hip_right"}
BONE_PREFIXES = ("vertebrae_", "rib_")
BATCH_CSV_FIELDS = CSV_FIELDS[:-2] + [
    "survives_suppression",
    "selection_tier",
    "selected_query_point",
    "reviewed_point",
    "coordinate_space",
]
ORGAN_SUMMARY_FIELDS = [
    "organ",
    "status",
    "window_category",
    "window_center_hu",
    "window_width_hu",
    "mask_voxels",
    "raw_candidates",
    "after_suppression",
    "strict_collectable",
    "relaxed_only_collectable",
    "total_collectable",
    "selected_strict",
    "selected_relaxed",
    "selected_total",
    "reviewed_points",
    "runtime_seconds",
    "notes",
]
REVIEW_FIELDS = [
    "subject_id",
    "timepoint",
    "organ",
    "point_id",
    "raw_x_voxel",
    "raw_y_voxel",
    "raw_z_voxel",
    "detector_score",
    "selection_tier",
    "contact_sheet",
    "distinctiveness_label",
    "adjacent_slice_persistence",
    "reviewer_notes",
]


class OrganGeometryError(ValueError):
    pass


def organ_name_from_path(path):
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def window_for_organ(organ):
    name = str(organ)
    if name == "brain":
        category = "brain"
    elif name.startswith("lung_"):
        category = "lung"
    elif name in BONE_EXACT or name.startswith(BONE_PREFIXES):
        category = "bone"
    else:
        category = "soft_tissue"
    return category, dict(WINDOW_PRESETS[category])


def discover_masks(mask_dir, organs):
    directory = Path(mask_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError("Mask directory not found: {}".format(directory))
    paths = sorted(
        path for path in directory.glob("*.nii.gz") if not path.name.startswith("._")
    )
    if not paths:
        raise FileNotFoundError("No .nii.gz masks found under {}".format(directory))
    requested = list(organs or ["all"])
    if requested == ["all"]:
        return paths
    if "all" in requested:
        raise ValueError("'all' cannot be combined with explicit organ names")
    by_name = {organ_name_from_path(path): path for path in paths}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise FileNotFoundError("Requested masks not found: {}".format(", ".join(missing)))
    return [by_name[name] for name in sorted(set(requested))]


def _stable_argsort(values, descending=False):
    array = np.asarray(values)
    return np.argsort(-array if descending else array, kind="stable")


def _fps_with_seeds(points_xyz, scores, spacing_xyz_mm, count, seed_indices=()):
    points = np.asarray(points_xyz, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    seeds = [int(index) for index in seed_indices]
    if count <= 0 or not len(points):
        return np.empty((0,), dtype=np.int64)
    if len(seeds) > count:
        return np.asarray(seeds[:count], dtype=np.int64)
    selected = list(dict.fromkeys(seeds))
    physical = points * np.asarray(spacing_xyz_mm, dtype=np.float64)[None, :]
    if selected:
        minimum_squared = np.full(len(points), np.inf, dtype=np.float64)
        for index in selected:
            delta = physical - physical[index]
            minimum_squared = np.minimum(minimum_squared, np.sum(delta * delta, axis=1))
    else:
        first = int(np.argmax(scores))
        selected.append(first)
        delta = physical - physical[first]
        minimum_squared = np.sum(delta * delta, axis=1)
    minimum_squared[np.asarray(selected, dtype=np.int64)] = -1.0
    target = min(int(count), len(points))
    while len(selected) < target:
        next_index = int(np.argmax(minimum_squared))
        selected.append(next_index)
        delta = physical - physical[next_index]
        minimum_squared = np.minimum(minimum_squared, np.sum(delta * delta, axis=1))
        minimum_squared[np.asarray(selected, dtype=np.int64)] = -1.0
    return np.asarray(selected, dtype=np.int64)


def strict_first_relaxed_selection(
    points_xyz,
    scores,
    boundary_distances,
    spacing_xyz_mm,
    suppression_radius_mm,
    interior_margin_mm,
    quota,
):
    points = np.asarray(points_xyz, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    distances = np.asarray(boundary_distances, dtype=np.float64)
    if not len(points):
        return {
            "suppressed_indices": np.empty((0,), dtype=np.int64),
            "strict_indices": np.empty((0,), dtype=np.int64),
            "relaxed_indices": np.empty((0,), dtype=np.int64),
            "selected_indices": np.empty((0,), dtype=np.int64),
            "selection_tiers": {},
            "status": "NO_CANDIDATES",
        }
    suppressed = greedy_radius_suppression(
        points, score_values, spacing_xyz_mm, suppression_radius_mm
    )
    strict = suppressed[distances[suppressed] >= float(interior_margin_mm)]
    relaxed = suppressed[distances[suppressed] < float(interior_margin_mm)]
    target = min(int(quota), len(suppressed))
    if len(strict) >= target:
        local = farthest_point_sample(
            points[strict], score_values[strict], spacing_xyz_mm, target
        )
        selected = strict[local]
    else:
        strict_order = _stable_argsort(score_values[strict], descending=True)
        selected_strict = strict[strict_order]
        selected = list(int(index) for index in selected_strict)
        deficit = target - len(selected)
        if deficit and len(relaxed):
            combined = np.concatenate([strict, relaxed])
            seed_count = len(strict)
            seed_positions = list(range(seed_count))
            chosen_positions = _fps_with_seeds(
                points[combined],
                score_values[combined],
                spacing_xyz_mm,
                seed_count + deficit,
                seed_indices=seed_positions,
            )
            chosen_relaxed = [
                int(combined[position])
                for position in chosen_positions
                if position >= seed_count
            ][:deficit]
            selected.extend(chosen_relaxed)
        selected = np.asarray(selected, dtype=np.int64)
    tiers = {
        int(index): (
            "strict_5mm"
            if distances[index] >= float(interior_margin_mm)
            else "relaxed_in_mask"
        )
        for index in selected
    }
    strict_selected = sum(value == "strict_5mm" for value in tiers.values())
    relaxed_selected = sum(value == "relaxed_in_mask" for value in tiers.values())
    if not len(selected):
        status = "NO_CANDIDATES"
    elif len(selected) < int(quota):
        status = "PARTIAL_SUPPLY"
    elif relaxed_selected:
        status = "FULL_QUOTA_WITH_RELAXED"
    else:
        status = "FULL_QUOTA_STRICT"
    return {
        "suppressed_indices": suppressed,
        "strict_indices": strict,
        "relaxed_indices": relaxed,
        "selected_indices": selected,
        "selection_tiers": tiers,
        "status": status,
        "selected_strict": strict_selected,
        "selected_relaxed": relaxed_selected,
    }


def choose_review_indices(points_xyz, scores, limit, spacing_xyz_mm):
    points = np.asarray(points_xyz, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if len(points) <= int(limit):
        return np.arange(len(points), dtype=np.int64)
    chosen = []
    for index in _stable_argsort(score_values, descending=True)[:5]:
        chosen.append(int(index))
    for index in _stable_argsort(score_values, descending=False):
        if index not in chosen:
            chosen.append(int(index))
        if len(chosen) == 10:
            break
    remaining = np.asarray(
        [index for index in range(len(points)) if index not in set(chosen)], dtype=np.int64
    )
    spatial_needed = min(10, int(limit) - len(chosen), len(remaining))
    if spatial_needed:
        spatial_local = farthest_point_sample(
            points[remaining], score_values[remaining], spacing_xyz_mm, spatial_needed
        )
        chosen.extend(int(index) for index in remaining[spatial_local])
    if len(chosen) < int(limit):
        for index in _stable_argsort(score_values, descending=True):
            if int(index) not in chosen:
                chosen.append(int(index))
            if len(chosen) == int(limit):
                break
    return np.asarray(chosen[: int(limit)], dtype=np.int64)


def _save_figure_atomic(figure, destination):
    path = Path(destination)
    if path.exists():
        raise FileExistsError("Refusing to overwrite figure: {}".format(path))
    temporary = path.with_name(path.name + ".tmp")
    figure.savefig(str(temporary), dpi=180, bbox_inches="tight", format="png")
    os.replace(str(temporary), str(path))


def _window_slice(slice_hu, center_hu, width_hu):
    values = np.asarray(slice_hu, dtype=np.float32)
    lower = float(center_hu) - float(width_hu) / 2.0
    return np.asarray(
        (np.clip(values, lower, lower + float(width_hu)) - lower) / float(width_hu),
        dtype=np.float32,
    )


def write_review_contact_sheets(
    ct_image,
    query_rows,
    review_row_indices,
    window,
    spacing_xyz_mm,
    output_dir,
    organ,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    half_xy = np.maximum(np.ceil(15.0 / spacing[:2]).astype(np.int64), 2)
    selected_rows = [query_rows[int(index)] for index in review_row_indices]
    outputs = []
    rows_per_page = 10
    for page_start in range(0, len(selected_rows), rows_per_page):
        page_rows = selected_rows[page_start : page_start + rows_per_page]
        figure, axes = plt.subplots(
            len(page_rows),
            4,
            figsize=(10.5, max(2.0 * len(page_rows), 2.4)),
            squeeze=False,
            gridspec_kw={"width_ratios": [0.72, 1.0, 1.0, 1.0]},
        )
        for row_index, row in enumerate(page_rows):
            x = int(row["raw_x_voxel"])
            y = int(row["raw_y_voxel"])
            z = int(row["raw_z_voxel"])
            x0 = max(0, x - int(half_xy[0]))
            x1 = min(ct_image.shape[0], x + int(half_xy[0]) + 1)
            y0 = max(0, y - int(half_xy[1]))
            y1 = min(ct_image.shape[1], y + int(half_xy[1]) + 1)
            metadata_axis = axes[row_index, 0]
            metadata_axis.set_axis_off()
            metadata_axis.text(
                0.98,
                0.5,
                "P{:03d}\nscore={:.3g}\n{}".format(
                    int(row["point_id"]),
                    float(row["detector_score"]),
                    row["selection_tier"],
                ),
                ha="right",
                va="center",
                fontsize=7,
                transform=metadata_axis.transAxes,
            )
            for column, offset in enumerate((-1, 0, 1)):
                axis = axes[row_index, column + 1]
                slice_z = min(max(z + offset, 0), ct_image.shape[2] - 1)
                patch = np.asarray(ct_image.dataobj[x0:x1, y0:y1, slice_z], dtype=np.float32)
                displayed = _window_slice(
                    patch, window["center_hu"], window["width_hu"]
                ).T
                axis.imshow(displayed, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
                draw_gap_crosshair(axis, float(x - x0), float(y - y0))
                axis.set_title("z={}".format(slice_z), fontsize=8)
                axis.set_axis_off()
        figure.suptitle(
            "{} exact-slice review | {} window {}/{} HU".format(
                organ,
                window["category"],
                int(window["center_hu"]),
                int(window["width_hu"]),
            ),
            fontsize=11,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        page_number = page_start // rows_per_page + 1
        destination = Path(output_dir) / "review_contact_sheet_{:02d}.png".format(page_number)
        _save_figure_atomic(figure, destination)
        plt.close(figure)
        outputs.append(destination.name)
    return outputs


def write_mask_only_overview(image, mask, crop_start_xyz, output_path, title):
    """Show representative mask slices when the detector finds no candidates."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occupied = np.flatnonzero(np.any(mask, axis=(0, 1)))
    if not len(occupied):
        return None
    quantile_indices = np.rint(
        np.linspace(0, len(occupied) - 1, min(6, len(occupied)))
    ).astype(np.int64)
    slices = occupied[quantile_indices]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, z_index in zip(axes.flat, slices):
        axis.imshow(image[:, :, z_index].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axis.contour(
            np.asarray(mask[:, :, z_index], dtype=np.uint8).T,
            levels=[0.5],
            colors=["#00d5ff"],
            linewidths=0.8,
        )
        axis.set_title(
            "native z={} | no candidates".format(int(z_index + crop_start_xyz[2]))
        )
        axis.set_axis_off()
    for axis in axes.flat[len(slices) :]:
        axis.set_axis_off()
    figure.suptitle(title)
    _save_figure_atomic(figure, output_path)
    plt.close(figure)
    return Path(output_path).name


def _candidate_rows(
    records,
    crop_start,
    affine,
    boundary_distance,
    selection,
    review_candidate_indices,
    subject,
    timepoint,
    organ,
    interior_margin_mm,
):
    suppressed = set(int(index) for index in selection["suppressed_indices"])
    selected = set(int(index) for index in selection["selected_indices"])
    reviewed = set(int(index) for index in review_candidate_indices)
    rows = []
    for index, record in enumerate(records):
        crop_xyz = np.asarray(record["crop_xyz"], dtype=np.int64)
        raw_xyz = crop_xyz + np.asarray(crop_start, dtype=np.int64)
        ras_xyz = np.asarray(affine, dtype=np.float64).dot(np.r_[raw_xyz, 1.0])[:3]
        rows.append(
            {
                "candidate_id": index,
                "subject_id": subject,
                "timepoint": timepoint,
                "organ": organ,
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
                    boundary_distance[tuple(crop_xyz)] >= float(interior_margin_mm)
                ),
                "survives_suppression": index in suppressed,
                "selection_tier": selection["selection_tiers"].get(index, ""),
                "selected_query_point": index in selected,
                "reviewed_point": index in reviewed,
                "coordinate_space": "native_nifti_voxel_xyz",
            }
        )
    return rows


def _empty_summary(organ, mask_path, window, status, notes, runtime_seconds=0.0):
    return {
        "schema_version": SCHEMA_VERSION,
        "organ": organ,
        "mask_path": str(mask_path),
        "status": status,
        "window": window,
        "mask_voxels": 0,
        "counts": {
            "raw_candidates": 0,
            "after_suppression": 0,
            "strict_collectable": 0,
            "relaxed_only_collectable": 0,
            "total_collectable": 0,
            "selected_strict": 0,
            "selected_relaxed": 0,
            "selected_total": 0,
            "reviewed_points": 0,
        },
        "runtime_seconds": float(runtime_seconds),
        "notes": notes,
    }


def process_organ(ct_image, mask_path, organ_dir, args):
    import nibabel as nib

    started = time.perf_counter()
    organ = organ_name_from_path(mask_path)
    category, preset = window_for_organ(organ)
    window = {"category": category, **preset}
    mask_image = nib.load(str(mask_path))
    if mask_image.ndim != 3:
        raise OrganGeometryError("Mask is not three-dimensional")
    if mask_image.shape != ct_image.shape:
        raise OrganGeometryError(
            "Mask shape {} does not match CT {}".format(mask_image.shape, ct_image.shape)
        )
    if not np.allclose(mask_image.affine, ct_image.affine, rtol=0.0, atol=1e-5):
        raise OrganGeometryError("Mask affine does not match CT affine")
    values = np.asarray(mask_image.dataobj)
    if not np.isfinite(values).all():
        raise OrganGeometryError("Mask contains non-finite values")
    unique = np.unique(values)
    if not set(float(value) for value in unique).issubset({0.0, 1.0}):
        raise OrganGeometryError("Mask is not binary: {}".format(unique.tolist()))
    mask_full = values > 0
    mask_voxels = int(mask_full.sum())
    if not mask_voxels:
        summary = _empty_summary(
            organ, mask_path, window, "EMPTY_MASK", "Mask contains no foreground voxels"
        )
        write_csv_atomic(organ_dir / "raw_candidates.csv", [], BATCH_CSV_FIELDS)
        write_csv_atomic(
            organ_dir / "query_points.csv", [], ["point_id"] + BATCH_CSV_FIELDS[1:]
        )
        return summary, [], []

    spacing = np.asarray(ct_image.header.get_zooms()[:3], dtype=np.float64)
    crop_slices, crop_start = padded_crop_slices(mask_full, spacing, args.crop_padding_mm)
    mask_crop = np.asarray(mask_full[crop_slices], dtype=bool)
    ct_crop = np.asarray(ct_image.dataobj[crop_slices], dtype=np.float32)
    image = window_ct(ct_crop, window["center_hu"], window["width_hu"])
    distances = boundary_distances_mm(mask_crop, spacing)
    records, thresholds = detect_harris_laplacian_candidates(
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
        overview_name = write_mask_only_overview(
            image,
            mask_crop,
            crop_start,
            organ_dir / "whole_organ_overview.png",
            "{} {} {} | {} window | no candidates".format(
                args.subject, args.timepoint, organ, category
            ),
        )
        summary = _empty_summary(
            organ,
            mask_path,
            window,
            "NO_CANDIDATES",
            "No Harris-Laplacian candidate survived scale selection",
            time.perf_counter() - started,
        )
        summary["mask_voxels"] = mask_voxels
        summary["crop_start_xyz"] = [int(value) for value in crop_start]
        summary["crop_shape_xyz"] = [int(value) for value in image.shape]
        summary["harris_thresholds"] = [
            None if not np.isfinite(value) else float(value) for value in thresholds
        ]
        summary["outputs"] = {
            "raw_candidates": "raw_candidates.csv",
            "query_points": "query_points.csv",
            "whole_organ_overview": overview_name,
            "review_contact_sheets": [],
        }
        write_csv_atomic(organ_dir / "raw_candidates.csv", [], BATCH_CSV_FIELDS)
        write_csv_atomic(
            organ_dir / "query_points.csv", [], ["point_id"] + BATCH_CSV_FIELDS[1:]
        )
        return summary, [], []

    coordinates = np.asarray([record["crop_xyz"] for record in records], dtype=np.int64)
    scores = np.asarray([record["detector_score"] for record in records], dtype=np.float64)
    candidate_distances = distances[tuple(coordinates.T)]
    selection = strict_first_relaxed_selection(
        coordinates,
        scores,
        candidate_distances,
        spacing,
        args.suppression_radius_mm,
        args.interior_margin_mm,
        args.num_points,
    )
    selected_candidate_indices = selection["selected_indices"]
    selected_coordinates = coordinates[selected_candidate_indices]
    selected_scores = scores[selected_candidate_indices]
    review_local = choose_review_indices(
        selected_coordinates, selected_scores, args.review_points, spacing
    )
    review_candidate_indices = selected_candidate_indices[review_local]
    rows = _candidate_rows(
        records,
        crop_start,
        ct_image.affine,
        distances,
        selection,
        review_candidate_indices,
        args.subject,
        args.timepoint,
        organ,
        args.interior_margin_mm,
    )
    query_rows = [dict(rows[int(index)]) for index in selected_candidate_indices]
    for point_id, row in enumerate(query_rows):
        row.pop("candidate_id")
        row["point_id"] = point_id
    review_candidate_set = set(int(index) for index in review_candidate_indices)
    review_query_indices = np.asarray(
        [
            index
            for index, candidate_index in enumerate(selected_candidate_indices)
            if int(candidate_index) in review_candidate_set
        ],
        dtype=np.int64,
    )
    review_rows = [query_rows[int(index)] for index in review_query_indices]

    write_csv_atomic(organ_dir / "raw_candidates.csv", rows, BATCH_CSV_FIELDS)
    query_fields = ["point_id"] + BATCH_CSV_FIELDS[1:]
    write_csv_atomic(organ_dir / "query_points.csv", query_rows, query_fields)
    overview_title = (
        "{} {} {} | {} window | selected {}/{}".format(
            args.subject, args.timepoint, organ, category, len(query_rows), args.num_points
        )
    )
    if len(selected_coordinates):
        selected_distances = candidate_distances[selected_candidate_indices]
        plot_axial_montage(
            image,
            mask_crop,
            selected_coordinates,
            selected_distances,
            crop_start,
            spacing,
            organ_dir / "whole_organ_overview.png",
            overview_title,
        )
        contact_sheets = write_review_contact_sheets(
            ct_image,
            review_rows,
            np.arange(len(review_rows), dtype=np.int64),
            window,
            spacing,
            organ_dir,
            organ,
        )
    else:
        contact_sheets = []
    elapsed = time.perf_counter() - started
    counts = {
        "raw_candidates": len(records),
        "after_suppression": int(len(selection["suppressed_indices"])),
        "strict_collectable": int(len(selection["strict_indices"])),
        "relaxed_only_collectable": int(len(selection["relaxed_indices"])),
        "total_collectable": int(len(selection["suppressed_indices"])),
        "selected_strict": int(selection.get("selected_strict", 0)),
        "selected_relaxed": int(selection.get("selected_relaxed", 0)),
        "selected_total": int(len(selected_candidate_indices)),
        "reviewed_points": int(len(review_rows)),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "organ": organ,
        "mask_path": str(mask_path),
        "status": selection["status"],
        "window": window,
        "mask_voxels": mask_voxels,
        "crop_start_xyz": [int(value) for value in crop_start],
        "crop_shape_xyz": [int(value) for value in image.shape],
        "harris_thresholds": [
            None if not np.isfinite(value) else float(value) for value in thresholds
        ],
        "counts": counts,
        "runtime_seconds": float(elapsed),
        "notes": "",
        "outputs": {
            "raw_candidates": "raw_candidates.csv",
            "query_points": "query_points.csv",
            "whole_organ_overview": "whole_organ_overview.png",
            "review_contact_sheets": contact_sheets,
        },
    }
    return summary, query_rows, review_rows


def _summary_row(summary):
    counts = summary["counts"]
    window = summary["window"]
    return {
        "organ": summary["organ"],
        "status": summary["status"],
        "window_category": window["category"],
        "window_center_hu": window["center_hu"],
        "window_width_hu": window["width_hu"],
        "mask_voxels": summary["mask_voxels"],
        "raw_candidates": counts["raw_candidates"],
        "after_suppression": counts["after_suppression"],
        "strict_collectable": counts["strict_collectable"],
        "relaxed_only_collectable": counts["relaxed_only_collectable"],
        "total_collectable": counts["total_collectable"],
        "selected_strict": counts["selected_strict"],
        "selected_relaxed": counts["selected_relaxed"],
        "selected_total": counts["selected_total"],
        "reviewed_points": counts["reviewed_points"],
        "runtime_seconds": summary["runtime_seconds"],
        "notes": summary.get("notes", ""),
    }


def _review_template_rows(subject, timepoint, organ, review_rows, contact_sheets):
    rows = []
    for index, row in enumerate(review_rows):
        page = min(index // 10, max(len(contact_sheets) - 1, 0))
        rows.append(
            {
                "subject_id": subject,
                "timepoint": timepoint,
                "organ": organ,
                "point_id": row["point_id"],
                "raw_x_voxel": row["raw_x_voxel"],
                "raw_y_voxel": row["raw_y_voxel"],
                "raw_z_voxel": row["raw_z_voxel"],
                "detector_score": row["detector_score"],
                "selection_tier": row["selection_tier"],
                "contact_sheet": (
                    "{}/{}".format(organ, contact_sheets[page]) if contact_sheets else ""
                ),
                "distinctiveness_label": "",
                "adjacent_slice_persistence": "",
                "reviewer_notes": "",
            }
        )
    return rows


def _write_report(output_dir, rows, manifest):
    destination = Path(output_dir) / "pilot_report.md"
    temporary = destination.with_name(destination.name + ".tmp")
    lines = [
        "# Subject-021 Multi-Organ Harris-Laplacian Pilot",
        "",
        "Status: **{}**".format(manifest["status"]),
        "",
        "This CPU-only pilot generated and visualized Test query points. It did not load UAE, generate embeddings, match points, or calculate cycle error.",
        "",
        "| Organ | Window | Mask voxels | Raw | Suppressed | Strict | Relaxed-only | Selected | Status | Review |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        review_link = ""
        organ_dir = Path(output_dir) / row["organ"]
        sheets = sorted(organ_dir.glob("review_contact_sheet_*.png"))
        if sheets:
            review_link = "[contact sheet]({}/{})".format(row["organ"], sheets[0].name)
        elif (organ_dir / "whole_organ_overview.png").exists():
            review_link = "[mask overview]({}/whole_organ_overview.png)".format(
                row["organ"]
            )
        lines.append(
            "| {organ} | {window_category} | {mask_voxels} | {raw_candidates} | {after_suppression} | {strict_collectable} | {relaxed_only_collectable} | {selected_total} | {status} | {review} |".format(
                review=review_link, **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Candidate supply and visual appearance do not demonstrate improved UAE correspondence accuracy.",
            "- The detector parameters were previously tuned on subject-021 liver and are not organ-general validation.",
            "- Relaxed points are inside the organ mask but less than 5 mm from its boundary and are labelled separately.",
            "- Manual distinctiveness labels remain pending in `visual_review_template.csv`.",
        ]
    )
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(destination))


def _signature_payload(args, ct_path, ct_image, mask_paths):
    masks = []
    for path in mask_paths:
        masks.append(
            {
                "organ": organ_name_from_path(path),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "ct": {
            "path": str(ct_path),
            "size_bytes": ct_path.stat().st_size,
            "sha256": sha256_file(ct_path),
            "geometry": {
                "shape_xyz": [int(value) for value in ct_image.shape],
                "spacing_xyz_mm": [
                    float(value) for value in ct_image.header.get_zooms()[:3]
                ],
                "affine_ras": np.asarray(ct_image.affine, dtype=np.float64).tolist(),
            },
        },
        "masks": masks,
        "parameters": {
            "subject": args.subject,
            "timepoint": args.timepoint,
            "num_points": args.num_points,
            "review_points": args.review_points,
            "window_policy": args.window_policy,
            "selection_policy": args.selection_policy,
            "crop_padding_mm": args.crop_padding_mm,
            "interior_margin_mm": args.interior_margin_mm,
            "suppression_radius_mm": args.suppression_radius_mm,
            "scales_mm": [float(value) for value in args.scales_mm],
            "harris_quantile": args.harris_quantile,
            "harris_k": args.harris_k,
            "max_candidates_per_scale": args.max_candidates_per_scale,
        },
    }


def _signature(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_ct_image(ct_image):
    if ct_image.ndim != 3:
        raise ValueError("CT must be three-dimensional")
    values = np.asanyarray(ct_image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError("CT contains non-finite values")


def run_batch(args):
    validate_parameters(args)
    if args.timepoint != "test":
        raise ValueError("The multi-organ pilot is intentionally limited to Test")
    if args.window_policy != "fixed-categories":
        raise ValueError("Only fixed-categories is supported")
    if args.selection_policy != "strict-first-relaxed":
        raise ValueError("Only strict-first-relaxed is supported")
    try:
        import nibabel as nib
        import scipy
    except ImportError as exc:
        raise RuntimeError("nibabel and scipy are required for this CPU pilot") from exc

    ct_path = Path(args.ct).resolve()
    mask_paths = discover_masks(args.mask_dir, args.organs)
    ct_image = nib.load(str(ct_path))
    validate_ct_image(ct_image)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    payload = _signature_payload(args, ct_path, ct_image, mask_paths)
    signature = _signature(payload)
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Existing batch run requires --resume: {}".format(output_dir))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            raise RuntimeError("Resume signature mismatch; use a new output directory")
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError("Output directory is not empty: {}".format(output_dir))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "in_progress",
            "scope": "subject021_test_all_organs_query_generation_and_visual_review_only",
            "signature": signature,
            "inputs_and_parameters": payload,
            "organs": {},
            "explicitly_not_run": [
                "retest_processing",
                "uae_model_loading",
                "embedding_generation",
                "point_matching",
                "cycle_error",
            ],
        }
        write_json_atomic(manifest_path, manifest)

    all_summaries = []
    all_review_template_rows = []
    for position, mask_path in enumerate(mask_paths, start=1):
        organ = organ_name_from_path(mask_path)
        final_dir = output_dir / organ
        existing = manifest["organs"].get(organ)
        if existing and existing.get("complete") and final_dir.is_dir():
            summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
            all_summaries.append(summary)
            query_rows = []
            query_path = final_dir / "query_points.csv"
            if query_path.exists():
                with query_path.open(newline="", encoding="utf-8") as handle:
                    query_rows = list(csv.DictReader(handle))
            reviewed = [row for row in query_rows if row.get("reviewed_point") == "True"]
            sheets = summary.get("outputs", {}).get("review_contact_sheets", [])
            all_review_template_rows.extend(
                _review_template_rows(args.subject, args.timepoint, organ, reviewed, sheets)
            )
            print("[{}/{}] {}: resumed {}".format(position, len(mask_paths), organ, summary["status"]))
            continue

        temporary_dir = output_dir / ".{}.tmp".format(organ)
        if temporary_dir.exists():
            if not args.resume:
                raise FileExistsError("Temporary organ output exists: {}".format(temporary_dir))
            shutil.rmtree(str(temporary_dir))
        temporary_dir.mkdir()
        try:
            summary, query_rows, review_rows = process_organ(
                ct_image, mask_path, temporary_dir, args
            )
        except OrganGeometryError as exc:
            category, preset = window_for_organ(organ)
            summary = _empty_summary(
                organ,
                mask_path,
                {"category": category, **preset},
                "GEOMETRY_FAILURE",
                str(exc),
            )
            query_rows = []
            review_rows = []
            write_csv_atomic(temporary_dir / "raw_candidates.csv", [], BATCH_CSV_FIELDS)
            write_csv_atomic(temporary_dir / "query_points.csv", [], ["point_id"] + BATCH_CSV_FIELDS[1:])
        write_json_atomic(temporary_dir / "summary.json", summary)
        if final_dir.exists():
            raise FileExistsError("Refusing to replace existing organ output: {}".format(final_dir))
        os.replace(str(temporary_dir), str(final_dir))
        manifest["organs"][organ] = {"complete": True, "status": summary["status"]}
        write_json_atomic(manifest_path, manifest, overwrite=True)
        all_summaries.append(summary)
        sheets = summary.get("outputs", {}).get("review_contact_sheets", [])
        all_review_template_rows.extend(
            _review_template_rows(args.subject, args.timepoint, organ, review_rows, sheets)
        )
        print(
            "[{}/{}] {}: {} selected={} raw={}".format(
                position,
                len(mask_paths),
                organ,
                summary["status"],
                summary["counts"]["selected_total"],
                summary["counts"]["raw_candidates"],
            ),
            flush=True,
        )

    rows = [_summary_row(summary) for summary in sorted(all_summaries, key=lambda item: item["organ"])]
    write_csv_atomic(
        output_dir / "organ_summary.csv", rows, ORGAN_SUMMARY_FIELDS, overwrite=args.resume
    )
    write_csv_atomic(
        output_dir / "visual_review_template.csv",
        all_review_template_rows,
        REVIEW_FIELDS,
        overwrite=args.resume,
    )
    failure_statuses = {"GEOMETRY_FAILURE"}
    manifest["status"] = (
        "complete_with_organ_failures"
        if any(row["status"] in failure_statuses for row in rows)
        else "complete"
    )
    manifest["completed_organ_count"] = len(rows)
    manifest["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "nibabel": nib.__version__,
        "platform": platform.platform(),
        "device": "cpu",
    }
    write_json_atomic(manifest_path, manifest, overwrite=True)
    _write_report(output_dir, rows, manifest)
    return manifest
