"""Survey raw SuperPoint candidates across the complete Quadra Test CT.

This gate characterizes detector behaviour only. It intentionally does not
deduplicate candidates in 3D, apply farthest-point sampling, process Retest, or
run UAE matching.
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
    write_json_atomic,
)
from tools.quadra.superpoint_representative_gate import (  # noqa: E402
    ORGAN_GROUPS,
    load_group_union,
)


SCHEMA_VERSION = 1
WINDOWS = {
    "soft_tissue": {"center_hu": 40.0, "width_hu": 400.0},
    "lung": {"center_hu": -600.0, "width_hu": 1500.0},
}
PRIMARY_WINDOW = {
    "bladder": "soft_tissue",
    "colon": "soft_tissue",
    "kidneys": "soft_tissue",
    "liver": "soft_tissue",
    "lungs": "lung",
}
CANDIDATE_FIELDS = [
    "point_id",
    "window_name",
    "raw_x_voxel",
    "raw_y_voxel",
    "raw_z_voxel",
    "score",
    "inside_bladder",
    "inside_colon",
    "inside_kidneys",
    "inside_liver",
    "inside_lungs",
    "inside_group_count",
    "exclusive_organ_group",
    "ambiguous_membership",
    "coord_space",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run SuperPoint over the complete Test CT: one soft-tissue pass over "
            "every axial slice and one lung-window pass over lung-containing slices. "
            "Exports raw candidates without 3D deduplication or FPS."
        )
    )
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--superpoint-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def build_slice_plan(depth, lung_slice_indices):
    """Return the fixed full-volume window plan without duplicate slice work."""

    if int(depth) <= 0:
        raise ValueError("CT depth must be positive")
    lung_indices = [int(value) for value in lung_slice_indices]
    if any(value < 0 or value >= int(depth) for value in lung_indices):
        raise ValueError("Lung slice index is outside CT bounds")
    if any(right <= left for left, right in zip(lung_indices, lung_indices[1:])):
        raise ValueError("Lung slice indices must be strictly increasing")
    return [("soft_tissue", list(range(int(depth)))), ("lung", lung_indices)]


def classify_candidates(group_masks_yx, keypoints_xy):
    """Label candidates against all organ groups while preserving ambiguity."""

    names = list(ORGAN_GROUPS)
    memberships = {
        name: candidate_mask_membership(group_masks_yx[name], keypoints_xy)
        for name in names
    }
    count = np.zeros(len(keypoints_xy), dtype=np.int16)
    for name in names:
        count += memberships[name].astype(np.int16)
    exclusive = np.full(len(keypoints_xy), "", dtype=object)
    for name in names:
        exclusive[(count == 1) & memberships[name]] = name
    return memberships, count, exclusive


def summarize_supply(slice_rows, inside_scores, exclusive_scores):
    """Aggregate raw and exclusive candidate supply by organ and window."""

    summaries = []
    for organ in ORGAN_GROUPS:
        for window_name in WINDOWS:
            rows = [row for row in slice_rows if row["window_name"] == window_name]
            inside_column = "inside_{}_count".format(organ)
            exclusive_column = "exclusive_{}_count".format(organ)
            ambiguous_column = "ambiguous_inside_{}_count".format(organ)
            nonzero_slices = [
                int(row["slice_index"]) for row in rows if int(row[exclusive_column]) > 0
            ]
            scores_inside = inside_scores[(organ, window_name)]
            scores_exclusive = exclusive_scores[(organ, window_name)]
            summaries.append(
                {
                    "organ_group": organ,
                    "window_name": window_name,
                    "primary_window_for_organ": PRIMARY_WINDOW[organ] == window_name,
                    "evaluated_slice_count": len(rows),
                    "all_candidate_count": int(sum(row["candidate_count"] for row in rows)),
                    "inside_count": int(sum(row[inside_column] for row in rows)),
                    "exclusive_count": int(sum(row[exclusive_column] for row in rows)),
                    "ambiguous_inside_count": int(sum(row[ambiguous_column] for row in rows)),
                    "slices_with_exclusive_count": len(nonzero_slices),
                    "exclusive_slice_min": min(nonzero_slices) if nonzero_slices else None,
                    "exclusive_slice_max": max(nonzero_slices) if nonzero_slices else None,
                    "inside_score_median": (
                        float(np.median(scores_inside)) if scores_inside else None
                    ),
                    "exclusive_score_median": (
                        float(np.median(scores_exclusive)) if scores_exclusive else None
                    ),
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


def _write_coverage_png_atomic(path, slice_rows):
    """Write a compact z-coverage diagnostic using only Pillow."""

    from PIL import Image, ImageDraw

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite PNG: {}".format(destination))
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    width, row_height, margin = 1200, 115, 75
    height = margin + row_height * len(ORGAN_GROUPS) + 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        "bladder": "#b91c1c",
        "colon": "#c2410c",
        "kidneys": "#7c3aed",
        "liver": "#0284c7",
        "lungs": "#059669",
    }
    draw.text((margin, 18), "Raw exclusive SuperPoint candidates by native CT slice", fill="black")
    plot_left, plot_right = 190, width - 45
    full_depth_max = max(int(row["slice_index"]) for row in slice_rows)
    for row_index, organ in enumerate(ORGAN_GROUPS):
        window_name = PRIMARY_WINDOW[organ]
        rows = [row for row in slice_rows if row["window_name"] == window_name]
        counts = [int(row["exclusive_{}_count".format(organ)]) for row in rows]
        slices = [int(row["slice_index"]) for row in rows]
        top = margin + row_index * row_height
        baseline = top + 74
        draw.text((20, top + 22), "{} ({})".format(organ, window_name), fill="black")
        draw.line((plot_left, baseline, plot_right, baseline), fill="#777777", width=1)
        maximum = max(counts) if counts else 0
        for slice_index, count in zip(slices, counts):
            x = plot_left + int(
                (plot_right - plot_left) * slice_index / max(1, full_depth_max)
            )
            bar_height = int(58 * count / max(1, maximum))
            if bar_height:
                draw.line((x, baseline, x, baseline - bar_height), fill=colors[organ], width=2)
        draw.text(
            (plot_right - 195, top + 6),
            "total={}  max/slice={}".format(sum(counts), maximum),
            fill=colors[organ],
        )
    draw.text(
        (margin, height - 35),
        "No 3D deduplication or farthest-point sampling has been applied.",
        fill="#555555",
    )
    image.save(temporary, format="PNG")
    os.replace(str(temporary), str(destination))


def main(argv=None):
    args = parse_args(argv)
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to reuse full-volume output: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for the full-volume gate") from exc

    started = time.perf_counter()
    ct_path = Path(args.ct).resolve()
    ct_image = nib.load(str(ct_path))
    if ct_image.ndim != 3:
        raise ValueError("Expected a 3D CT, got shape {}".format(ct_image.shape))
    _, ct_metadata = load_axial_ct_slice(ct_path, 0)
    group_data = {
        group: load_group_union(args.mask_dir, names, ct_path)
        for group, names in ORGAN_GROUPS.items()
    }
    plan = build_slice_plan(ct_image.shape[2], group_data["lungs"]["nonempty_slice_indices"])
    model, provenance = load_superpoint_model(
        args.superpoint_root, args.checkpoint, device=args.device
    )

    candidate_path = output_dir / "full_volume_candidates.csv"
    candidate_temporary = candidate_path.with_name(candidate_path.name + ".tmp")
    if candidate_temporary.exists():
        raise FileExistsError(
            "Refusing to replace interrupted candidate CSV: {}".format(candidate_temporary)
        )
    slice_rows = []
    inside_scores = {(organ, window): [] for organ in ORGAN_GROUPS for window in WINDOWS}
    exclusive_scores = {(organ, window): [] for organ in ORGAN_GROUPS for window in WINDOWS}
    total_runs = sum(len(indices) for _, indices in plan)
    run_index = 0
    point_id = 0
    total_inference_seconds = 0.0
    peak_gpu_bytes = 0
    with candidate_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for window_name, slice_indices in plan:
            window = WINDOWS[window_name]
            for slice_index in slice_indices:
                slice_started = time.perf_counter()
                slice_hu = np.asarray(ct_image.dataobj[:, :, int(slice_index)], dtype=np.float32)
                model_image = native_xy_to_model_yx(
                    window_and_normalize_ct(
                        slice_hu,
                        center=window["center_hu"],
                        width=window["width_hu"],
                    )
                )
                prediction = run_superpoint_on_slice(model, model_image, provenance["device"])
                keypoints = prediction["keypoints_xy"]
                scores = prediction["scores"]
                group_masks_yx = {
                    organ: native_xy_to_model_yx(
                        group_data[organ]["mask_xyz"][:, :, int(slice_index)]
                    )
                    for organ in ORGAN_GROUPS
                }
                memberships, membership_count, exclusive = classify_candidates(
                    group_masks_yx, keypoints
                )
                for candidate_index, (xy, score) in enumerate(zip(keypoints, scores)):
                    writer.writerow(
                        {
                            "point_id": point_id,
                            "window_name": window_name,
                            "raw_x_voxel": float(xy[0]),
                            "raw_y_voxel": float(xy[1]),
                            "raw_z_voxel": float(slice_index),
                            "score": float(score),
                            **{
                                "inside_{}".format(organ): bool(
                                    memberships[organ][candidate_index]
                                )
                                for organ in ORGAN_GROUPS
                            },
                            "inside_group_count": int(membership_count[candidate_index]),
                            "exclusive_organ_group": exclusive[candidate_index],
                            "ambiguous_membership": bool(membership_count[candidate_index] > 1),
                            "coord_space": "native_nifti_voxel_xyz",
                        }
                    )
                    point_id += 1

                slice_row = {
                    "window_name": window_name,
                    "slice_index": int(slice_index),
                    "candidate_count": int(len(keypoints)),
                    "outside_all_groups_count": int((membership_count == 0).sum()),
                    "ambiguous_count": int((membership_count > 1).sum()),
                    "inference_seconds": float(prediction["runtime_seconds"]),
                    "total_slice_seconds": float(time.perf_counter() - slice_started),
                }
                for organ in ORGAN_GROUPS:
                    inside = memberships[organ]
                    exclusive_organ = exclusive == organ
                    ambiguous_inside = inside & (membership_count > 1)
                    slice_row["inside_{}_count".format(organ)] = int(inside.sum())
                    slice_row["exclusive_{}_count".format(organ)] = int(exclusive_organ.sum())
                    slice_row["ambiguous_inside_{}_count".format(organ)] = int(
                        ambiguous_inside.sum()
                    )
                    inside_scores[(organ, window_name)].extend(
                        float(value) for value in scores[inside]
                    )
                    exclusive_scores[(organ, window_name)].extend(
                        float(value) for value in scores[exclusive_organ]
                    )
                slice_rows.append(slice_row)
                total_inference_seconds += float(prediction["runtime_seconds"])
                peak_gpu_bytes = max(
                    peak_gpu_bytes, int(prediction["peak_gpu_memory_bytes"] or 0)
                )
                run_index += 1
                if run_index == 1 or run_index % args.progress_every == 0 or run_index == total_runs:
                    print(
                        "[{}/{}] {} z={}: candidates={}, exclusive={}".format(
                            run_index,
                            total_runs,
                            window_name,
                            slice_index,
                            len(keypoints),
                            int((membership_count == 1).sum()),
                        ),
                        flush=True,
                    )
    os.replace(str(candidate_temporary), str(candidate_path))

    slice_fields = [
        "window_name", "slice_index", "candidate_count", "outside_all_groups_count",
        "ambiguous_count", "inference_seconds", "total_slice_seconds",
    ]
    for organ in ORGAN_GROUPS:
        slice_fields.extend(
            [
                "inside_{}_count".format(organ),
                "exclusive_{}_count".format(organ),
                "ambiguous_inside_{}_count".format(organ),
            ]
        )
    _write_csv_atomic(output_dir / "full_volume_slice_summary.csv", slice_fields, slice_rows)
    aggregate_rows = summarize_supply(slice_rows, inside_scores, exclusive_scores)
    _write_csv_atomic(
        output_dir / "full_volume_aggregate_summary.csv",
        list(aggregate_rows[0]),
        aggregate_rows,
    )
    _write_coverage_png_atomic(output_dir / "full_volume_candidate_coverage.png", slice_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "complete_test_volume_raw_superpoint_candidate_survey",
        "test_volume_processed": True,
        "retest_volume_processed": False,
        "whole_test_volume_soft_tissue_processed": True,
        "lung_window_restricted_to_lung_mask_slices": True,
        "deduplication_applied": False,
        "farthest_point_sampling_applied": False,
        "uae_matching_run": False,
        "ambiguous_candidates_retained_and_labelled": True,
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
            }
            for organ in ORGAN_GROUPS
        },
        "windows": WINDOWS,
        "primary_window_by_organ": PRIMARY_WINDOW,
        "planned_slice_runs": total_runs,
        "completed_slice_runs": len(slice_rows),
        "total_raw_candidate_count": point_id,
        "model": provenance,
        "peak_gpu_memory_bytes": peak_gpu_bytes or None,
        "total_inference_seconds": total_inference_seconds,
        "total_runtime_seconds": float(time.perf_counter() - started),
        "outputs": {
            "candidate_csv": str(candidate_path),
            "slice_summary_csv": str(output_dir / "full_volume_slice_summary.csv"),
            "aggregate_summary_csv": str(output_dir / "full_volume_aggregate_summary.csv"),
            "coverage_png": str(output_dir / "full_volume_candidate_coverage.png"),
        },
        "aggregate_summaries": aggregate_rows,
    }
    write_json_atomic(output_dir / "full_volume_gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
