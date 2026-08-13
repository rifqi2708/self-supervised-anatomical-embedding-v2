"""Bounded SuperPoint threshold sensitivity for sparse Quadra organs.

The gate evaluates bladder and combined kidneys only. It does not apply FPS,
process Retest, or run UAE. Confidence-greedy 3D suppression is reported as a
candidate-supply sensitivity analysis rather than a final selection policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    import sys

    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from tools.quadra.superpoint_adapter import (  # noqa: E402
    candidate_mask_membership,
    load_axial_ct_slice,
    load_superpoint_model,
    native_xy_to_model_yx,
    run_superpoint_on_slice,
    window_and_normalize_ct,
    write_comparison_overlay_png_atomic,
    write_json_atomic,
)
from tools.quadra.superpoint_full_volume_gate import _write_csv_atomic  # noqa: E402
from tools.quadra.superpoint_representative_gate import (  # noqa: E402
    ORGAN_GROUPS,
    load_group_union,
)


SCHEMA_VERSION = 1
FOCUS_ORGANS = ("bladder", "kidneys")
DEFAULT_THRESHOLDS = (0.005, 0.002, 0.001)
DEFAULT_RADII_MM = (3.0, 5.0, 10.0)
WINDOW_CENTER_HU = 40.0
WINDOW_WIDTH_HU = 400.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed SuperPoint detection thresholds on bladder- and "
            "kidney-containing Test slices. Reports raw supply and 3/5/10 mm "
            "3D suppression sensitivity without FPS or UAE."
        )
    )
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--superpoint-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS)
    )
    parser.add_argument(
        "--dedup-radii-mm", type=float, nargs="+", default=list(DEFAULT_RADII_MM)
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def validate_gate_parameters(thresholds, radii_mm):
    thresholds = tuple(float(value) for value in thresholds)
    radii_mm = tuple(float(value) for value in radii_mm)
    if not thresholds or not radii_mm:
        raise ValueError("At least one threshold and one deduplication radius are required")
    if any(not np.isfinite(value) or value <= 0 or value >= 1 for value in thresholds):
        raise ValueError("Thresholds must be finite values strictly between 0 and 1")
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("Thresholds must be unique and provided from highest to lowest")
    if any(not np.isfinite(value) or value <= 0 for value in radii_mm):
        raise ValueError("Deduplication radii must be positive finite millimetres")
    if tuple(sorted(set(radii_mm))) != radii_mm:
        raise ValueError("Deduplication radii must be unique and strictly increasing")
    return thresholds, radii_mm


def union_slice_indices(*index_groups):
    values = sorted({int(value) for group in index_groups for value in group})
    if not values:
        raise ValueError("At least one non-empty focus-organ slice is required")
    return values


def set_detection_threshold(model, threshold):
    if not hasattr(model, "conf") or not hasattr(model.conf, "detection_threshold"):
        raise AttributeError("SuperPoint model does not expose conf.detection_threshold")
    model.conf.detection_threshold = float(threshold)
    observed = float(model.conf.detection_threshold)
    if observed != float(threshold):
        raise RuntimeError(
            "Failed to set SuperPoint detection threshold: expected {}, observed {}".format(
                threshold, observed
            )
        )


def greedy_deduplicate_3d(points_xyz_voxel, scores, spacing_xyz_mm, radius_mm):
    """Return deterministic confidence-greedy physical-space keep indices."""

    points = np.asarray(points_xyz_voxel, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected candidate coordinates with shape [N, 3]")
    if values.shape != (len(points),):
        raise ValueError("Candidate score count does not match coordinate count")
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three positive finite values")
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("Deduplication radius must be positive and finite")
    if np.any(~np.isfinite(points)) or np.any(~np.isfinite(values)):
        raise ValueError("Candidate coordinates and scores must be finite")

    physical = points * spacing
    order = np.lexsort((np.arange(len(values), dtype=np.int64), -values))
    kept = []
    kept_physical = []
    squared_radius = float(radius_mm) ** 2
    for index in order:
        point = physical[index]
        if kept_physical:
            distances_squared = np.sum((np.asarray(kept_physical) - point) ** 2, axis=1)
            if np.any(distances_squared < squared_radius):
                continue
        kept.append(int(index))
        kept_physical.append(point)
    return np.asarray(kept, dtype=np.int64)


def validate_nested_detections(previous, current, previous_threshold, current_threshold):
    """Verify lowering the threshold retains all prior coordinates and scores."""

    if current_threshold >= previous_threshold:
        raise ValueError("Current threshold must be lower than the previous threshold")
    previous_map = {
        (float(xy[0]), float(xy[1])): float(score)
        for xy, score in zip(previous["keypoints_xy"], previous["scores"])
    }
    current_map = {
        (float(xy[0]), float(xy[1])): float(score)
        for xy, score in zip(current["keypoints_xy"], current["scores"])
    }
    missing = set(previous_map).difference(current_map)
    if missing:
        raise RuntimeError(
            "Lower threshold removed {} prior keypoint coordinates".format(len(missing))
        )
    for coordinate, score in previous_map.items():
        if not np.isclose(score, current_map[coordinate], rtol=0.0, atol=1e-6):
            raise RuntimeError("Keypoint score changed across threshold runs")


def _threshold_label(value):
    return "{:.6f}".format(float(value)).rstrip("0").rstrip(".")


def _write_coverage_png_atomic(path, slice_rows, thresholds):
    from PIL import Image, ImageDraw

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite PNG: {}".format(destination))
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    width, row_height, margin = 1200, 100, 70
    rows_total = len(FOCUS_ORGANS) * len(thresholds)
    image = Image.new("RGB", (width, margin + rows_total * row_height + 55), "white")
    draw = ImageDraw.Draw(image)
    colors = {"bladder": "#b91c1c", "kidneys": "#7c3aed"}
    draw.text((margin, 18), "Raw in-mask candidates by slice and detection threshold", fill="black")
    plot_left, plot_right = 220, width - 45
    minimum_slice = min(int(row["slice_index"]) for row in slice_rows)
    maximum_slice = max(int(row["slice_index"]) for row in slice_rows)
    row_index = 0
    for organ in FOCUS_ORGANS:
        for threshold in thresholds:
            selected = [
                row for row in slice_rows if float(row["threshold"]) == float(threshold)
            ]
            counts = [int(row["inside_{}_count".format(organ)]) for row in selected]
            slices = [int(row["slice_index"]) for row in selected]
            top = margin + row_index * row_height
            baseline = top + 64
            draw.text(
                (20, top + 20),
                "{}  threshold={}".format(organ, _threshold_label(threshold)),
                fill="black",
            )
            draw.line((plot_left, baseline, plot_right, baseline), fill="#777777", width=1)
            maximum_count = max(counts) if counts else 0
            for slice_index, count in zip(slices, counts):
                fraction = (slice_index - minimum_slice) / max(1, maximum_slice - minimum_slice)
                x = plot_left + int((plot_right - plot_left) * fraction)
                bar_height = int(50 * count / max(1, maximum_count))
                if bar_height:
                    draw.line((x, baseline, x, baseline - bar_height), fill=colors[organ], width=2)
            draw.text(
                (plot_right - 190, top + 3),
                "total={}  max/slice={}".format(sum(counts), maximum_count),
                fill=colors[organ],
            )
            row_index += 1
    draw.text(
        (margin, image.height - 35),
        "Counts are before 3D suppression and FPS; only Test focus-organ slices were run.",
        fill="#555555",
    )
    image.save(temporary, format="PNG")
    os.replace(str(temporary), str(destination))


def main(argv=None):
    args = parse_args(argv)
    thresholds, radii_mm = validate_gate_parameters(args.thresholds, args.dedup_radii_mm)
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to reuse threshold output: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for the threshold gate") from exc

    started = time.perf_counter()
    ct_path = Path(args.ct).resolve()
    ct_image = nib.load(str(ct_path))
    if ct_image.ndim != 3:
        raise ValueError("Expected a 3D CT, got shape {}".format(ct_image.shape))
    _, ct_metadata = load_axial_ct_slice(ct_path, 0)
    group_data = {
        organ: load_group_union(args.mask_dir, ORGAN_GROUPS[organ], ct_path)
        for organ in FOCUS_ORGANS
    }
    slice_indices = union_slice_indices(
        *(group_data[organ]["nonempty_slice_indices"] for organ in FOCUS_ORGANS)
    )
    model, provenance = load_superpoint_model(
        args.superpoint_root, args.checkpoint, device=args.device
    )
    loaded_threshold = provenance["model_configuration"]["detection_threshold"]
    if not np.isclose(loaded_threshold, DEFAULT_THRESHOLDS[0], rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Pinned model default threshold changed: expected {}, observed {}".format(
                DEFAULT_THRESHOLDS[0], loaded_threshold
            )
        )

    candidate_rows = []
    slice_rows = []
    point_id = 0
    completed_runs = 0
    total_runs = len(slice_indices) * len(thresholds)
    previous_by_slice = {}
    overlay_written = set()
    total_inference_seconds = 0.0
    peak_gpu_bytes = 0
    for threshold in thresholds:
        set_detection_threshold(model, threshold)
        for slice_index in slice_indices:
            slice_started = time.perf_counter()
            slice_hu = np.asarray(ct_image.dataobj[:, :, slice_index], dtype=np.float32)
            model_image = native_xy_to_model_yx(
                window_and_normalize_ct(
                    slice_hu, center=WINDOW_CENTER_HU, width=WINDOW_WIDTH_HU
                )
            )
            prediction = run_superpoint_on_slice(model, model_image, provenance["device"])
            if len(prediction["scores"]) and np.min(prediction["scores"]) < threshold - 1e-7:
                raise RuntimeError("SuperPoint returned a score below the active threshold")
            if slice_index in previous_by_slice:
                validate_nested_detections(
                    previous_by_slice[slice_index],
                    prediction,
                    previous_by_slice[slice_index]["threshold"],
                    threshold,
                )
            previous_by_slice[slice_index] = {
                "keypoints_xy": prediction["keypoints_xy"],
                "scores": prediction["scores"],
                "threshold": threshold,
            }

            keypoints = prediction["keypoints_xy"]
            scores = prediction["scores"]
            memberships = {
                organ: candidate_mask_membership(
                    native_xy_to_model_yx(group_data[organ]["mask_xyz"][:, :, slice_index]),
                    keypoints,
                )
                for organ in FOCUS_ORGANS
            }
            membership_count = sum(
                (memberships[organ].astype(np.int16) for organ in FOCUS_ORGANS),
                np.zeros(len(keypoints), dtype=np.int16),
            )
            if np.any(membership_count > 1):
                raise RuntimeError("Bladder and kidney masks overlap at a detected candidate")
            for candidate_index, (xy, score) in enumerate(zip(keypoints, scores)):
                organ = ""
                for name in FOCUS_ORGANS:
                    if memberships[name][candidate_index]:
                        organ = name
                candidate_rows.append(
                    {
                        "point_id": point_id,
                        "threshold": threshold,
                        "raw_x_voxel": float(xy[0]),
                        "raw_y_voxel": float(xy[1]),
                        "raw_z_voxel": float(slice_index),
                        "score": float(score),
                        "inside_bladder": bool(memberships["bladder"][candidate_index]),
                        "inside_kidneys": bool(memberships["kidneys"][candidate_index]),
                        "exclusive_organ_group": organ,
                        "coord_space": "native_nifti_voxel_xyz",
                    }
                )
                point_id += 1
            slice_rows.append(
                {
                    "threshold": threshold,
                    "slice_index": slice_index,
                    "candidate_count": int(len(keypoints)),
                    "inside_bladder_count": int(memberships["bladder"].sum()),
                    "inside_kidneys_count": int(memberships["kidneys"].sum()),
                    "inference_seconds": float(prediction["runtime_seconds"]),
                    "total_slice_seconds": float(time.perf_counter() - slice_started),
                }
            )

            for organ in FOCUS_ORGANS:
                overlay_key = (organ, threshold)
                if (
                    overlay_key not in overlay_written
                    and slice_index == group_data[organ]["slice_index"]
                ):
                    mask_yx = native_xy_to_model_yx(
                        group_data[organ]["mask_xyz"][:, :, slice_index]
                    )
                    write_comparison_overlay_png_atomic(
                        output_dir
                        / "{}_threshold_{}_z{:04d}.png".format(
                            organ, _threshold_label(threshold).replace(".", "p"), slice_index
                        ),
                        model_image,
                        keypoints,
                        keypoints[memberships[organ]],
                        [{"name": organ, "mask_yx": mask_yx}],
                        "{} | threshold={} | z={} | L/W=40/400".format(
                            organ, _threshold_label(threshold), slice_index
                        ),
                    )
                    overlay_written.add(overlay_key)

            total_inference_seconds += float(prediction["runtime_seconds"])
            peak_gpu_bytes = max(
                peak_gpu_bytes, int(prediction["peak_gpu_memory_bytes"] or 0)
            )
            completed_runs += 1
            if (
                completed_runs == 1
                or completed_runs % args.progress_every == 0
                or completed_runs == total_runs
            ):
                print(
                    "[{}/{}] threshold={} z={}: candidates={}, bladder={}, kidneys={}".format(
                        completed_runs,
                        total_runs,
                        _threshold_label(threshold),
                        slice_index,
                        len(keypoints),
                        int(memberships["bladder"].sum()),
                        int(memberships["kidneys"].sum()),
                    ),
                    flush=True,
                )

    aggregate_rows = []
    for threshold in thresholds:
        threshold_candidates = [
            row for row in candidate_rows if float(row["threshold"]) == threshold
        ]
        threshold_slices = [row for row in slice_rows if float(row["threshold"]) == threshold]
        for organ in FOCUS_ORGANS:
            eligible = [
                row for row in threshold_candidates if row["exclusive_organ_group"] == organ
            ]
            coordinates = np.asarray(
                [
                    [row["raw_x_voxel"], row["raw_y_voxel"], row["raw_z_voxel"]]
                    for row in eligible
                ],
                dtype=np.float64,
            ).reshape((-1, 3))
            scores = np.asarray([row["score"] for row in eligible], dtype=np.float64)
            retained_counts = {}
            for radius in radii_mm:
                kept = greedy_deduplicate_3d(
                    coordinates, scores, ct_metadata["spacing_xyz_mm"], radius
                )
                retained_counts[radius] = int(len(kept))
            active_slices = sorted(
                {
                    int(row["raw_z_voxel"])
                    for row in eligible
                }
            )
            aggregate = {
                "organ_group": organ,
                "threshold": threshold,
                "evaluated_slice_count": len(threshold_slices),
                "all_candidate_count": len(threshold_candidates),
                "inside_count": len(eligible),
                "slices_with_inside_count": len(active_slices),
                "inside_slice_min": min(active_slices) if active_slices else None,
                "inside_slice_max": max(active_slices) if active_slices else None,
                "inside_score_median": float(np.median(scores)) if len(scores) else None,
                "inside_score_min": float(np.min(scores)) if len(scores) else None,
            }
            for radius in radii_mm:
                aggregate["retained_{}mm".format(_threshold_label(radius))] = retained_counts[
                    radius
                ]
            aggregate_rows.append(aggregate)

    candidate_fields = list(candidate_rows[0])
    _write_csv_atomic(
        output_dir / "threshold_candidates.csv", candidate_fields, candidate_rows
    )
    _write_csv_atomic(
        output_dir / "threshold_slice_summary.csv", list(slice_rows[0]), slice_rows
    )
    _write_csv_atomic(
        output_dir / "threshold_aggregate_summary.csv",
        list(aggregate_rows[0]),
        aggregate_rows,
    )
    _write_coverage_png_atomic(
        output_dir / "threshold_candidate_coverage.png", slice_rows, thresholds
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "bounded_sparse_organ_superpoint_threshold_sensitivity",
        "test_volume_processed": True,
        "retest_volume_processed": False,
        "whole_volume_processed": False,
        "focus_organs": list(FOCUS_ORGANS),
        "focus_slice_policy": "union of non-empty native Test mask slices",
        "thresholds": list(thresholds),
        "deduplication_radii_mm": list(radii_mm),
        "deduplication_policy": (
            "confidence-descending greedy suppression in native physical space; "
            "distance strictly below radius is suppressed"
        ),
        "deduplication_used_for_final_selection": False,
        "farthest_point_sampling_applied": False,
        "uae_matching_run": False,
        "window": {"center_hu": WINDOW_CENTER_HU, "width_hu": WINDOW_WIDTH_HU},
        "ct_path": str(ct_path),
        "ct_shape_xyz": [int(value) for value in ct_image.shape],
        "ct_spacing_xyz_mm": ct_metadata["spacing_xyz_mm"],
        "ct_orientation_codes": ct_metadata["orientation_codes"],
        "mask_dir": str(Path(args.mask_dir).resolve()),
        "organ_groups": {
            organ: {
                "component_masks": ORGAN_GROUPS[organ],
                "nonempty_slice_count": len(group_data[organ]["nonempty_slice_indices"]),
                "nonempty_slice_min": min(group_data[organ]["nonempty_slice_indices"]),
                "nonempty_slice_max": max(group_data[organ]["nonempty_slice_indices"]),
                "representative_slice": group_data[organ]["slice_index"],
            }
            for organ in FOCUS_ORGANS
        },
        "focus_slice_count": len(slice_indices),
        "focus_slice_min": min(slice_indices),
        "focus_slice_max": max(slice_indices),
        "planned_slice_runs": total_runs,
        "completed_slice_runs": completed_runs,
        "model": provenance,
        "active_model_threshold_after_run": float(model.conf.detection_threshold),
        "total_exported_candidate_rows": len(candidate_rows),
        "peak_gpu_memory_bytes": peak_gpu_bytes or None,
        "total_inference_seconds": total_inference_seconds,
        "total_runtime_seconds": float(time.perf_counter() - started),
        "aggregate_summaries": aggregate_rows,
        "outputs": {
            "candidate_csv": str(output_dir / "threshold_candidates.csv"),
            "slice_summary_csv": str(output_dir / "threshold_slice_summary.csv"),
            "aggregate_summary_csv": str(output_dir / "threshold_aggregate_summary.csv"),
            "coverage_png": str(output_dir / "threshold_candidate_coverage.png"),
        },
    }
    write_json_atomic(output_dir / "threshold_gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
