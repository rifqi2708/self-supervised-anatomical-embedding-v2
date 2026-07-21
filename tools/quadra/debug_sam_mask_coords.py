#!/usr/bin/env python3
"""Debug Quadra SAM cycle-error point coordinates against raw NIfTI masks."""

import argparse
import csv
import os
from collections import defaultdict

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def sam_origin_mask_from_nifti(mask_path):
    """Emulate tools/utils.py:read_image(..., mask_path=...) before resampling."""
    img = nib.load(mask_path)
    canonical = nib.as_closest_canonical(img)
    axcodes = nib.aff2axcodes(canonical.affine)
    if tuple(axcodes) != ("R", "A", "S"):
        raise ValueError(f"Expected canonical RAS orientation, got {axcodes} for {mask_path}")

    data_xyz = np.asanyarray(canonical.dataobj)
    data_lps_xyz = np.flip(data_xyz, axis=(0, 1))
    return np.transpose(data_lps_xyz, (1, 0, 2))


def point_xyz_to_sam_index_yxz(point_xyz):
    point_xyz = np.asarray(point_xyz, dtype=int)
    return np.array([point_xyz[1], point_xyz[0], point_xyz[2]], dtype=int)


def load_subject_test_mask_paths(dataset_root, subject_id):
    subject_mask_root = os.path.join(dataset_root, "masks", subject_id)
    if not os.path.isdir(subject_mask_root):
        raise FileNotFoundError(f"Subject mask directory not found: {subject_mask_root}")

    test_dirs = [name for name in sorted(os.listdir(subject_mask_root)) if "_Test_" in name]
    if len(test_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one Test mask directory under {subject_mask_root}, found: {test_dirs}"
        )

    test_dir = os.path.join(subject_mask_root, test_dirs[0])
    mask_paths = {}
    for name in sorted(os.listdir(test_dir)):
        if name.endswith(".nii.gz") or name.endswith(".nii"):
            mask_paths[name] = os.path.join(test_dir, name)
    if not mask_paths:
        raise RuntimeError(f"No NIfTI masks found under {test_dir}")
    return test_dir, mask_paths


def load_subject_masks(dataset_root, subject_id):
    test_dir, mask_paths = load_subject_test_mask_paths(dataset_root, subject_id)
    sam_masks = {}
    raw_masks = {}
    meta = {}
    for mask_name, mask_path in mask_paths.items():
        sam_masks[mask_name] = sam_origin_mask_from_nifti(mask_path)
        raw_masks[mask_name] = sitk.ReadImage(mask_path)
        meta[mask_name] = {
            "path": mask_path,
            "axcodes": nib.aff2axcodes(nib.load(mask_path).affine),
            "size_xyz": raw_masks[mask_name].GetSize(),
        }
    return test_dir, sam_masks, raw_masks, meta


def mask_value_sam(sam_mask_yxz, point_xyz):
    point_yxz = point_xyz_to_sam_index_yxz(point_xyz)
    return int(sam_mask_yxz[point_yxz[0], point_yxz[1], point_yxz[2]])


def mask_value_raw(raw_mask, point_xyz):
    x, y, z = map(int, point_xyz)
    return int(raw_mask.GetPixel(x, y, z))


def classify_membership(point_xyz, sam_masks, raw_masks):
    sam_positive = [name for name, mask in sam_masks.items() if mask_value_sam(mask, point_xyz) > 0]
    raw_positive = [name for name, mask in raw_masks.items() if mask_value_raw(mask, point_xyz) > 0]
    return sam_positive, raw_positive


def load_subject_rows(csv_path, subject_id):
    rows = []
    with open(csv_path, newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            mask_name = str(row.get("mask_name", "")).strip()
            if not mask_name.startswith(subject_id + "/"):
                continue
            rows.append(
                {
                    "mask_name": mask_name.split("/", 1)[1],
                    "point_xyz": np.array(
                        [int(row["pt1_x"]), int(row["pt1_y"]), int(row["pt1_z"])],
                        dtype=int,
                    ),
                }
            )
    return rows


def compute_validity(rows, sam_masks, raw_masks):
    per_mask = defaultdict(lambda: {"count": 0, "sam_inside": 0, "raw_inside": 0})
    any_counts = {"count": 0, "sam_inside_any": 0, "raw_inside_any": 0}

    for row in rows:
        mask_name = row["mask_name"]
        point_xyz = row["point_xyz"]

        sam_inside = mask_value_sam(sam_masks[mask_name], point_xyz) > 0
        raw_inside = mask_value_raw(raw_masks[mask_name], point_xyz) > 0

        per_mask[mask_name]["count"] += 1
        per_mask[mask_name]["sam_inside"] += int(sam_inside)
        per_mask[mask_name]["raw_inside"] += int(raw_inside)

        any_counts["count"] += 1
        any_counts["sam_inside_any"] += int(
            any(mask_value_sam(mask, point_xyz) > 0 for mask in sam_masks.values())
        )
        any_counts["raw_inside_any"] += int(
            any(mask_value_raw(mask, point_xyz) > 0 for mask in raw_masks.values())
        )

    return per_mask, any_counts


def print_point_report(subject_id, label_mask_name, point_xyz, sam_masks, raw_masks):
    print(f"Subject: {subject_id}")
    print(f"Point xyz: {point_xyz.tolist()}")
    print(f"Labeled mask: {label_mask_name}")

    label_sam = mask_value_sam(sam_masks[label_mask_name], point_xyz)
    label_raw = mask_value_raw(raw_masks[label_mask_name], point_xyz)
    print(f"SAM sampling mask value in {label_mask_name}: {label_sam}")
    print(f"Raw NIfTI mask value in {label_mask_name}: {label_raw}")

    sam_positive, raw_positive = classify_membership(point_xyz, sam_masks, raw_masks)
    print(f"SAM says point is inside: {sam_positive or ['neither']}")
    print(f"Raw NIfTI says point is inside: {raw_positive or ['neither']}")


def print_validity_report(rows, per_mask, any_counts):
    if not rows:
        print("No CSV rows found for subject.")
        return

    print("")
    print("Per-mask CSV validity")
    for mask_name in sorted(per_mask):
        stats = per_mask[mask_name]
        print(
            f"{mask_name}: "
            f"SAM {stats['sam_inside']}/{stats['count']} | "
            f"raw {stats['raw_inside']}/{stats['count']}"
        )

    print("")
    print(
        "Any-mask validity: "
        f"SAM {any_counts['sam_inside_any']}/{any_counts['count']} | "
        f"raw {any_counts['raw_inside_any']}/{any_counts['count']}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/quadra_dataset_cropped",
        help="Quadra dataset root containing images/ and masks/.",
    )
    parser.add_argument("--subject", required=True, help="Subject id, for example quadra_hc_026.")
    parser.add_argument("--mask", required=True, help="Mask filename, for example bladder.nii.gz.")
    parser.add_argument(
        "--point",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Point stored in the SAM CSV as x y z.",
    )
    parser.add_argument(
        "--csv",
        help="Optional cycle-points CSV. When provided, compute per-mask validity for this subject.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    point_xyz = np.array(args.point, dtype=int)

    _, sam_masks, raw_masks, meta = load_subject_masks(args.dataset_root, args.subject)
    if args.mask not in sam_masks:
        raise FileNotFoundError(f"Mask {args.mask} not found for subject {args.subject}")

    print(f"Mask metadata for {args.mask}: {meta[args.mask]}")
    print_point_report(args.subject, args.mask, point_xyz, sam_masks, raw_masks)

    if args.csv:
        rows = load_subject_rows(args.csv, args.subject)
        per_mask, any_counts = compute_validity(rows, sam_masks, raw_masks)
        print_validity_report(rows, per_mask, any_counts)


if __name__ == "__main__":
    main()
