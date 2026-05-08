#!/usr/bin/env python3
"""Re-export a SAM cycle-error CSV with point columns converted to raw NIfTI voxel coordinates."""

import argparse
import csv
import os

import numpy as np

try:
    from coord_space_utils import (
        COORD_GROUPS,
        COORD_SPACE_RAW_ITK,
        build_sam_to_raw_transform,
        resolve_subject_images,
        transform_point_xyz,
    )
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "coord_space_utils":
        raise
    from tools.coord_space_utils import (
        COORD_GROUPS,
        COORD_SPACE_RAW_ITK,
        build_sam_to_raw_transform,
        resolve_subject_images,
        transform_point_xyz,
    )


def parse_subject_id(row):
    subject_id = str(row.get("subject_id", "")).strip()
    if subject_id:
        return subject_id

    mask_name = str(row.get("mask_name", "")).strip()
    if "/" in mask_name:
        return mask_name.split("/", 1)[0]
    raise ValueError(f"Unable to resolve subject_id from row: {row}")


def has_coord_group(row, prefix):
    keys = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z")
    return all(str(row.get(key, "")).strip() != "" for key in keys)


def read_point_xyz(row, prefix):
    return np.array(
        [int(row[f"{prefix}_x"]), int(row[f"{prefix}_y"]), int(row[f"{prefix}_z"])],
        dtype=int,
    )


def write_point_xyz(row, prefix, point_xyz):
    row[f"{prefix}_x"] = str(int(point_xyz[0]))
    row[f"{prefix}_y"] = str(int(point_xyz[1]))
    row[f"{prefix}_z"] = str(int(point_xyz[2]))


def insert_backup_fieldnames(fieldnames):
    new_fieldnames = list(fieldnames)
    insert_at = 3 if "subject_id" in new_fieldnames else len(new_fieldnames)
    backup_fields = []
    for prefix, _image_role in COORD_GROUPS:
        backup_fields.extend([f"{prefix}_sam_x", f"{prefix}_sam_y", f"{prefix}_sam_z"])
    backup_fields.append("coord_space")
    for offset, field in enumerate(backup_fields):
        if field not in new_fieldnames:
            new_fieldnames.insert(insert_at + offset, field)
    return new_fieldnames


def convert_rows(rows, dataset_root):
    subject_image_cache = {}
    transform_cache = {}
    converted_rows = []

    for row_idx, row in enumerate(rows, start=2):
        subject_id = parse_subject_id(row)
        if subject_id not in subject_image_cache:
            subject_image_cache[subject_id] = resolve_subject_images(dataset_root, subject_id)

        converted = dict(row)
        converted["coord_space"] = COORD_SPACE_RAW_ITK

        for prefix, image_role in COORD_GROUPS:
            if not has_coord_group(row, prefix):
                continue

            image_path = subject_image_cache[subject_id][image_role]
            if image_path not in transform_cache:
                affine, shape = build_sam_to_raw_transform(image_path)
                transform_cache[image_path] = {
                    "affine": affine,
                    "shape": shape,
                }

            point_sam_xyz = read_point_xyz(row, prefix)
            converted[f"{prefix}_sam_x"] = str(int(point_sam_xyz[0]))
            converted[f"{prefix}_sam_y"] = str(int(point_sam_xyz[1]))
            converted[f"{prefix}_sam_z"] = str(int(point_sam_xyz[2]))

            try:
                point_raw_xyz = transform_point_xyz(
                    point_sam_xyz,
                    transform_cache[image_path]["affine"],
                    transform_cache[image_path]["shape"],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to transform {prefix} in row {row_idx} for subject {subject_id}: {exc}"
                ) from exc

            write_point_xyz(converted, prefix, point_raw_xyz)

        converted_rows.append(converted)

    return converted_rows


def derive_output_path(input_csv):
    root, ext = os.path.splitext(input_csv)
    return f"{root}_raw_coords{ext}"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        required=True,
        help="SAM-space cycle CSV, for example data/ori_sam_result.csv",
    )
    parser.add_argument(
        "--output-csv",
        help="Output CSV path. Defaults to '<input>_raw_coords.csv'.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/quadra_dataset_cropped",
        help="Dataset root containing images/ and masks/.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_csv = args.output_csv or derive_output_path(args.input_csv)

    with open(args.input_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"No CSV header found in {args.input_csv}")

    converted_rows = convert_rows(rows, args.dataset_root)
    output_fieldnames = insert_backup_fieldnames(fieldnames)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(converted_rows)

    print(f"Wrote raw-coordinate CSV: {output_csv}")
    print(f"Rows converted: {len(converted_rows)}")


if __name__ == "__main__":
    main()
