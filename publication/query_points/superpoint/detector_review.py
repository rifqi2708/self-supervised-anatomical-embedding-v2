"""Reusable mask grouping and atomic CSV support for SuperPoint review runners.

Inputs: native CT/mask paths, registry group names and output rows.
Outputs: validated group unions or a new CSV; refuses overwrite.
Dependencies: NumPy, nibabel (loaded on demand), SuperPoint geometry helpers.
Role: method helper. Usage: query_points/superpoint/README.md.
"""
import csv
import os
from pathlib import Path
import numpy as np
from publication.query_points.superpoint.superpoint_adapter import choose_max_area_slice

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
