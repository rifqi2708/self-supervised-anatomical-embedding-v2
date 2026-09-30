"""Survey raw SuperPoint candidates on bounded slices across each organ extent."""

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
    inside_point_boundary_distances_mm,
    load_axial_ct_slice,
    load_superpoint_model,
    native_xy_to_model_yx,
    run_superpoint_on_slice,
    window_and_normalize_ct,
    write_comparison_overlay_png_atomic,
    write_json_atomic,
)
from tools.quadra.superpoint_representative_gate import (  # noqa: E402
    ORGAN_GROUPS,
    WINDOW_CASES,
    load_group_union,
)


SCHEMA_VERSION = 1
DEFAULT_SLICE_COUNT = 7


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded multi-slice survey of raw SuperPoint candidates across five "
            "organ extents. This does not process every CT slice, deduplicate points, "
            "apply FPS, or run UAE matching."
        )
    )
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--superpoint-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slice-count", type=int, default=DEFAULT_SLICE_COUNT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def choose_extent_quantile_slices(nonempty_slice_indices, count=DEFAULT_SLICE_COUNT):
    """Choose deterministic quantiles from actual non-empty organ slices."""

    indices = np.asarray(nonempty_slice_indices, dtype=np.int64)
    if indices.ndim != 1 or not indices.size:
        raise ValueError("At least one non-empty organ slice is required")
    if count <= 0:
        raise ValueError("Slice count must be positive")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Non-empty slice indices must be strictly increasing")
    selected_count = min(int(count), int(indices.size))
    positions = np.rint(np.linspace(0, indices.size - 1, selected_count)).astype(int)
    return indices[np.unique(positions)].astype(int).tolist()


def _optional_summary(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_window_cases(cases):
    grouped = {}
    for case in cases:
        key = (case["organ_group"], case["window"]["name"])
        grouped.setdefault(key, []).append(case)
    summaries = []
    for (organ_group, window_name), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["slice_index"])
        inside_counts = np.asarray([row["inside_count"] for row in rows], dtype=float)
        scores = [score for row in rows for score in row["inside_scores"]]
        summaries.append(
            {
                "organ_group": organ_group,
                "window_name": window_name,
                "sampled_slice_count": len(rows),
                "sampled_slice_indices": [row["slice_index"] for row in rows],
                "candidate_count": int(sum(row["candidate_count"] for row in rows)),
                "inside_count": int(inside_counts.sum()),
                "slices_with_inside_count": int((inside_counts > 0).sum()),
                "inside_per_slice": _optional_summary(inside_counts),
                "inside_score": _optional_summary(scores),
                "raw_candidates_are_not_3d_deduplicated": True,
            }
        )
    return summaries


def _write_csv_atomic(path, fieldnames, rows):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite CSV: {}".format(destination))
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("Refusing to replace temporary CSV: {}".format(temporary))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(destination))


def _write_montage_atomic(path, image_paths):
    from PIL import Image, ImageDraw

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite montage: {}".format(destination))
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    panels = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            panel = image.convert("RGB")
            panel.thumbnail((768, 408))
            panels.append(panel.copy())
    gap = 8
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    montage = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(montage)
    top = 0
    for panel in panels:
        left = (width - panel.width) // 2
        montage.paste(panel, (left, top))
        top += panel.height
        if top < height:
            draw.rectangle((0, top, width, top + gap - 1), fill=(35, 35, 35))
            top += gap
    montage.save(temporary, format="PNG")
    os.replace(str(temporary), str(destination))


def main(argv=None):
    args = parse_args(argv)
    if args.slice_count <= 0:
        raise ValueError("--slice-count must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to reuse multi-slice output: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    started = time.perf_counter()
    group_data = {
        group: load_group_union(args.mask_dir, names, args.ct)
        for group, names in ORGAN_GROUPS.items()
    }
    for group in group_data.values():
        group["sampled_slice_indices"] = choose_extent_quantile_slices(
            group["nonempty_slice_indices"], args.slice_count
        )

    model, provenance = load_superpoint_model(
        args.superpoint_root, args.checkpoint, device=args.device
    )
    cases = []
    point_rows = []
    montage_inputs = {}
    point_id = 0
    for organ_group, window_name, center, width in WINDOW_CASES:
        group = group_data[organ_group]
        montage_key = (organ_group, window_name)
        montage_inputs[montage_key] = []
        for sampled_rank, slice_index in enumerate(group["sampled_slice_indices"]):
            slice_hu, ct_metadata = load_axial_ct_slice(args.ct, slice_index)
            model_image = native_xy_to_model_yx(
                window_and_normalize_ct(slice_hu, center=center, width=width)
            )
            prediction = run_superpoint_on_slice(model, model_image, provenance["device"])
            keypoints = prediction["keypoints_xy"]
            scores = prediction["scores"]
            mask_yx = native_xy_to_model_yx(group["mask_xyz"][:, :, slice_index])
            membership = candidate_mask_membership(mask_yx, keypoints)
            inside_scores = scores[membership]
            distances = inside_point_boundary_distances_mm(
                mask_yx, keypoints, ct_metadata["spacing_xyz_mm"][:2]
            )
            mask_fraction = float(mask_yx.mean())
            inside_fraction = float(membership.mean()) if len(membership) else 0.0
            stem = "{}_{}_q{:02d}_z{:04d}".format(
                organ_group, window_name, sampled_rank, slice_index
            )
            overlay_path = output_dir / (stem + "_comparison.png")
            write_comparison_overlay_png_atomic(
                overlay_path,
                model_image,
                keypoints,
                keypoints[membership],
                [{"name": organ_group, "mask_yx": mask_yx}],
                "{} | q={} | z={} | L/W={}/{}".format(
                    organ_group, sampled_rank, slice_index, center, width
                ),
            )
            montage_inputs[montage_key].append(overlay_path)
            case = {
                "organ_group": organ_group,
                "window": {"name": window_name, "center_hu": center, "width_hu": width},
                "sampled_rank": sampled_rank,
                "slice_index": int(slice_index),
                "mask_area_pixels": int(mask_yx.sum()),
                "mask_area_percent": 100.0 * mask_fraction,
                "candidate_count": int(len(keypoints)),
                "inside_count": int(membership.sum()),
                "inside_percent": 100.0 * inside_fraction,
                "density_enrichment": inside_fraction / mask_fraction if mask_fraction else None,
                "inside_scores": [float(value) for value in inside_scores],
                "inside_score": _optional_summary(inside_scores),
                "inside_boundary_distance_mm": _optional_summary(distances),
                "runtime_seconds": float(prediction["runtime_seconds"]),
                "comparison_overlay_png": str(overlay_path),
            }
            cases.append(case)
            for xy, score, inside in zip(keypoints, scores, membership):
                point_rows.append(
                    {
                        "point_id": point_id,
                        "organ_group": organ_group,
                        "window_name": window_name,
                        "sampled_rank": sampled_rank,
                        "raw_x_voxel": float(xy[0]),
                        "raw_y_voxel": float(xy[1]),
                        "raw_z_voxel": float(slice_index),
                        "score": float(score),
                        "inside_group_mask": bool(inside),
                        "coord_space": "native_nifti_voxel_xyz",
                    }
                )
                point_id += 1
            print(
                "{} {} q{} z{}: candidates={}, inside={}".format(
                    organ_group, window_name, sampled_rank, slice_index,
                    len(keypoints), membership.sum()
                ),
                flush=True,
            )

    for (organ_group, window_name), image_paths in montage_inputs.items():
        _write_montage_atomic(
            output_dir / ("{}_{}_montage.png".format(organ_group, window_name)), image_paths
        )

    summary_rows = summarize_window_cases(cases)
    _write_csv_atomic(
        output_dir / "multislice_candidates.csv",
        [
            "point_id", "organ_group", "window_name", "sampled_rank",
            "raw_x_voxel", "raw_y_voxel", "raw_z_voxel", "score",
            "inside_group_mask", "coord_space",
        ],
        point_rows,
    )
    _write_csv_atomic(
        output_dir / "multislice_case_summary.csv",
        [
            "organ_group", "window_name", "sampled_rank", "slice_index",
            "mask_area_pixels", "mask_area_percent", "candidate_count", "inside_count",
            "inside_percent", "density_enrichment", "inside_score_median",
            "inside_boundary_distance_median_mm", "inside_boundary_distance_p95_mm",
            "runtime_seconds",
        ],
        [
            {
                "organ_group": case["organ_group"],
                "window_name": case["window"]["name"],
                "sampled_rank": case["sampled_rank"],
                "slice_index": case["slice_index"],
                "mask_area_pixels": case["mask_area_pixels"],
                "mask_area_percent": case["mask_area_percent"],
                "candidate_count": case["candidate_count"],
                "inside_count": case["inside_count"],
                "inside_percent": case["inside_percent"],
                "density_enrichment": case["density_enrichment"],
                "inside_score_median": case["inside_score"]["median"] if case["inside_score"] else "",
                "inside_boundary_distance_median_mm": case["inside_boundary_distance_mm"]["median"] if case["inside_boundary_distance_mm"] else "",
                "inside_boundary_distance_p95_mm": case["inside_boundary_distance_mm"]["p95"] if case["inside_boundary_distance_mm"] else "",
                "runtime_seconds": case["runtime_seconds"],
            }
            for case in cases
        ],
    )
    _write_csv_atomic(
        output_dir / "multislice_aggregate_summary.csv",
        [
            "organ_group", "window_name", "sampled_slice_count", "sampled_slice_indices",
            "candidate_count", "inside_count", "slices_with_inside_count",
            "inside_per_slice_median", "inside_per_slice_min", "inside_per_slice_max",
            "inside_score_median", "raw_candidates_are_not_3d_deduplicated",
        ],
        [
            {
                "organ_group": row["organ_group"],
                "window_name": row["window_name"],
                "sampled_slice_count": row["sampled_slice_count"],
                "sampled_slice_indices": ";".join(map(str, row["sampled_slice_indices"])),
                "candidate_count": row["candidate_count"],
                "inside_count": row["inside_count"],
                "slices_with_inside_count": row["slices_with_inside_count"],
                "inside_per_slice_median": row["inside_per_slice"]["median"],
                "inside_per_slice_min": row["inside_per_slice"]["min"],
                "inside_per_slice_max": row["inside_per_slice"]["max"],
                "inside_score_median": row["inside_score"]["median"] if row["inside_score"] else "",
                "raw_candidates_are_not_3d_deduplicated": True,
            }
            for row in summary_rows
        ],
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "bounded_multislice_raw_candidate_survey",
        "whole_volume_superpoint_processed": False,
        "deduplication_applied": False,
        "farthest_point_sampling_applied": False,
        "uae_matching_run": False,
        "slice_selection": "quantiles over sorted non-empty union-mask slice indices",
        "requested_slice_count_per_case": int(args.slice_count),
        "ct_path": str(Path(args.ct).resolve()),
        "mask_dir": str(Path(args.mask_dir).resolve()),
        "model": provenance,
        "organ_groups": {
            name: {
                "component_masks": ORGAN_GROUPS[name],
                "nonempty_slice_count": len(group["nonempty_slice_indices"]),
                "nonempty_slice_min": min(group["nonempty_slice_indices"]),
                "nonempty_slice_max": max(group["nonempty_slice_indices"]),
                "sampled_slice_indices": group["sampled_slice_indices"],
            }
            for name, group in group_data.items()
        },
        "cases": cases,
        "aggregate_summaries": summary_rows,
        "total_runtime_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(output_dir / "multislice_gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
