"""Run a bounded representative-slice SuperPoint gate for five organ groups."""

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
    choose_max_area_slice,
    inside_point_boundary_distances_mm,
    load_axial_ct_slice,
    load_superpoint_model,
    native_xy_to_model_yx,
    run_superpoint_on_slice,
    window_and_normalize_ct,
    write_comparison_overlay_png_atomic,
    write_json_atomic,
)


SCHEMA_VERSION = 1
ORGAN_GROUPS = {
    "bladder": ["urinary_bladder"],
    "colon": ["colon"],
    "kidneys": ["kidney_left", "kidney_right"],
    "liver": ["liver"],
    "lungs": [
        "lung_lower_lobe_left",
        "lung_lower_lobe_right",
        "lung_middle_lobe_right",
        "lung_upper_lobe_left",
        "lung_upper_lobe_right",
    ],
}
WINDOW_CASES = [
    ("bladder", "soft_tissue", 40.0, 400.0),
    ("colon", "soft_tissue", 40.0, 400.0),
    ("kidneys", "soft_tissue", 40.0, 400.0),
    ("liver", "soft_tissue", 40.0, 400.0),
    ("lungs", "soft_tissue", 40.0, 400.0),
    ("lungs", "lung", -600.0, 1500.0),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run six bounded SuperPoint cases on deterministic maximum-area organ slices; "
            "this command does not process every CT slice."
        )
    )
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--superpoint-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def load_group_union(mask_dir, names, ct_path):
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for representative-slice selection") from exc

    ct_image = nib.load(str(Path(ct_path).resolve()))
    union = np.zeros(ct_image.shape, dtype=bool)
    paths = []
    for name in names:
        path = Path(mask_dir).resolve() / (name + ".nii.gz")
        if not path.is_file():
            raise FileNotFoundError("Required organ mask not found: {}".format(path))
        image = nib.load(str(path))
        if image.shape != ct_image.shape:
            raise ValueError("Mask shape mismatch: {}".format(path))
        if not np.allclose(image.affine, ct_image.affine, rtol=0.0, atol=1e-5):
            raise ValueError("Mask affine mismatch: {}".format(path))
        values = np.asarray(image.dataobj)
        if not np.isfinite(values).all():
            raise ValueError("Mask contains non-finite values: {}".format(path))
        union |= values > 0
        paths.append(str(path))
    areas = union.sum(axis=(0, 1))
    slice_index = choose_max_area_slice(areas)
    return {
        "slice_index": slice_index,
        "mask_xy": union[:, :, slice_index],
        "mask_xyz": union,
        "nonempty_slice_indices": np.flatnonzero(areas > 0).astype(int).tolist(),
        "max_area_pixels": int(areas[slice_index]),
        "mask_paths": paths,
    }


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


def _write_case_csv(path, keypoints, scores, membership, organ_group, window_name, slice_index):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite case CSV: {}".format(destination))
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("Refusing to replace temporary case CSV: {}".format(temporary))
    fieldnames = [
        "point_id",
        "organ_group",
        "window_name",
        "raw_x_voxel",
        "raw_y_voxel",
        "raw_z_voxel",
        "score",
        "inside_group_mask",
        "coord_space",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point_id, (xy, score, inside) in enumerate(zip(keypoints, scores, membership)):
            writer.writerow(
                {
                    "point_id": point_id,
                    "organ_group": organ_group,
                    "window_name": window_name,
                    "raw_x_voxel": float(xy[0]),
                    "raw_y_voxel": float(xy[1]),
                    "raw_z_voxel": float(slice_index),
                    "score": float(score),
                    "inside_group_mask": bool(inside),
                    "coord_space": "native_nifti_voxel_xyz",
                }
            )
    os.replace(str(temporary), str(destination))


def _write_summary_csv(path, cases):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    fieldnames = [
        "organ_group", "window_name", "window_center_hu", "window_width_hu",
        "slice_index", "mask_area_pixels", "mask_area_percent", "candidate_count",
        "inside_count", "inside_percent", "density_enrichment", "inside_score_median",
        "inside_boundary_distance_median_mm", "inside_boundary_distance_p95_mm",
        "runtime_seconds",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            distance = case["inside_boundary_distance_mm"]
            score = case["inside_score"]
            writer.writerow(
                {
                    "organ_group": case["organ_group"],
                    "window_name": case["window"]["name"],
                    "window_center_hu": case["window"]["center_hu"],
                    "window_width_hu": case["window"]["width_hu"],
                    "slice_index": case["slice_index"],
                    "mask_area_pixels": case["mask_area_pixels"],
                    "mask_area_percent": case["mask_area_percent"],
                    "candidate_count": case["candidate_count"],
                    "inside_count": case["inside_count"],
                    "inside_percent": case["inside_percent"],
                    "density_enrichment": case["density_enrichment"],
                    "inside_score_median": score["median"] if score else "",
                    "inside_boundary_distance_median_mm": distance["median"] if distance else "",
                    "inside_boundary_distance_p95_mm": distance["p95"] if distance else "",
                    "runtime_seconds": case["runtime_seconds"],
                }
            )
    os.replace(str(temporary), str(destination))


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to reuse representative-gate output: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    started = time.perf_counter()
    group_data = {
        group: load_group_union(args.mask_dir, names, args.ct)
        for group, names in ORGAN_GROUPS.items()
    }
    model, provenance = load_superpoint_model(
        args.superpoint_root, args.checkpoint, device=args.device
    )
    cases = []
    for organ_group, window_name, center, width in WINDOW_CASES:
        group = group_data[organ_group]
        slice_hu, ct_metadata = load_axial_ct_slice(args.ct, group["slice_index"])
        model_image = native_xy_to_model_yx(
            window_and_normalize_ct(slice_hu, center=center, width=width)
        )
        prediction = run_superpoint_on_slice(model, model_image, provenance["device"])
        keypoints = prediction["keypoints_xy"]
        scores = prediction["scores"]
        mask_yx = native_xy_to_model_yx(group["mask_xy"])
        membership = candidate_mask_membership(mask_yx, keypoints)
        inside_scores = scores[membership]
        distances = inside_point_boundary_distances_mm(
            mask_yx, keypoints, ct_metadata["spacing_xyz_mm"][:2]
        )
        mask_fraction = float(mask_yx.mean())
        inside_fraction = float(membership.mean()) if len(membership) else 0.0
        stem = "{}_{}".format(organ_group, window_name)
        case = {
            "organ_group": organ_group,
            "component_masks": ORGAN_GROUPS[organ_group],
            "mask_paths": group["mask_paths"],
            "slice_index": int(group["slice_index"]),
            "window": {"name": window_name, "center_hu": center, "width_hu": width},
            "mask_area_pixels": int(group["max_area_pixels"]),
            "mask_area_percent": 100.0 * mask_fraction,
            "candidate_count": int(len(keypoints)),
            "inside_count": int(membership.sum()),
            "inside_percent": 100.0 * inside_fraction,
            "density_enrichment": inside_fraction / mask_fraction,
            "all_score": _optional_summary(scores),
            "inside_score": _optional_summary(inside_scores),
            "inside_boundary_distance_mm": _optional_summary(distances),
            "runtime_seconds": float(prediction["runtime_seconds"]),
            "points_csv": str(output_dir / (stem + "_points.csv")),
            "comparison_overlay_png": str(output_dir / (stem + "_comparison.png")),
        }
        _write_case_csv(
            case["points_csv"], keypoints, scores, membership,
            organ_group, window_name, group["slice_index"]
        )
        write_comparison_overlay_png_atomic(
            case["comparison_overlay_png"], model_image, keypoints, keypoints[membership],
            [{"name": organ_group, "mask_yx": mask_yx}],
            "{} | z={} | L/W={}/{}".format(organ_group, group["slice_index"], center, width),
        )
        cases.append(case)
        print(
            "{} {}: z={}, candidates={}, inside={}".format(
                organ_group, window_name, group["slice_index"], len(keypoints), membership.sum()
            ), flush=True,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "bounded_representative_organ_slice_gate",
        "whole_volume_superpoint_processed": False,
        "selection": "middle slice among maximum union-mask area ties",
        "ct_path": str(Path(args.ct).resolve()),
        "mask_dir": str(Path(args.mask_dir).resolve()),
        "model": provenance,
        "cases": cases,
        "total_runtime_seconds": float(time.perf_counter() - started),
    }
    _write_summary_csv(output_dir / "representative_gate_summary.csv", cases)
    write_json_atomic(output_dir / "representative_gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
