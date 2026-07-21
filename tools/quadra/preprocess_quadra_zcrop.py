import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


def is_mask_file(name):
    return name.endswith(".nii.gz") or name.endswith(".nii")


def strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def list_image_cases(images_root):
    cases = []
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Images root not found: {images_root}")
    for subject in sorted(os.listdir(images_root)):
        subject_dir = os.path.join(images_root, subject)
        if not os.path.isdir(subject_dir):
            continue
        for name in sorted(os.listdir(subject_dir)):
            if not is_mask_file(name):
                continue
            image_path = os.path.join(subject_dir, name)
            if os.path.isfile(image_path):
                cases.append((subject, strip_nii_suffix(name), image_path))
    if not cases:
        raise RuntimeError(f"No image files found under: {images_root}")
    return cases


def list_case_masks(mask_dir):
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    mask_files = []
    for name in sorted(os.listdir(mask_dir)):
        path = os.path.join(mask_dir, name)
        if os.path.isfile(path) and is_mask_file(name):
            mask_files.append((name, path))
    if not mask_files:
        raise RuntimeError(f"No .nii/.nii.gz mask files found in: {mask_dir}")
    return mask_files


def crop_z_sitk(image, z_min, z_max):
    x, y, z = image.GetSize()
    if z_min < 0 or z_max >= z or z_max < z_min:
        raise ValueError(f"Invalid crop bounds [{z_min}, {z_max}] for size z={z}")
    start = [0, 0, int(z_min)]
    size = [int(x), int(y), int(z_max - z_min + 1)]
    return sitk.Extract(image, size, start)


def compute_case_plan(subject, case_name, image_path, mask_dir, z_margin):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    if image_array.ndim != 3:
        raise ValueError(f"Expected 3D image for {image_path}, got shape {image_array.shape}")

    mask_files = list_case_masks(mask_dir)
    union_mask = np.zeros_like(image_array, dtype=bool)
    for mask_name, mask_path in mask_files:
        mask_image = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_image)
        if tuple(mask_array.shape) != tuple(image_array.shape):
            raise ValueError(
                f"Mask/image shape mismatch for {subject}/{case_name}: "
                f"{mask_name} shape={mask_array.shape}, image shape={image_array.shape}"
            )
        union_mask |= mask_array != 0

    z_indices = np.where(np.any(union_mask, axis=(1, 2)))[0]
    if z_indices.size == 0:
        raise RuntimeError(f"Combined mask is empty for {subject}/{case_name}")

    z_min_raw = int(z_indices.min())
    z_max_raw = int(z_indices.max())
    z_min = max(0, z_min_raw - int(z_margin))
    z_max = min(int(image_array.shape[0]) - 1, z_max_raw + int(z_margin))

    return {
        "subject": subject,
        "case": case_name,
        "image_path": image_path,
        "mask_dir": mask_dir,
        "mask_files": mask_files,
        "image": image,
        "z": {
            "original_depth": int(image_array.shape[0]),
            "z_min_raw": z_min_raw,
            "z_max_raw": z_max_raw,
            "z_min": z_min,
            "z_max": z_max,
            "cropped_depth": int(z_max - z_min + 1),
            "margin": int(z_margin),
        },
    }


def write_case_outputs(case_plan, input_root, output_root, overwrite):
    subject = case_plan["subject"]
    case_name = case_plan["case"]
    z_min = case_plan["z"]["z_min"]
    z_max = case_plan["z"]["z_max"]

    rel_image = os.path.relpath(case_plan["image_path"], input_root)
    out_image = os.path.join(output_root, rel_image)
    out_mask_dir = os.path.join(output_root, "masks", subject, case_name)
    out_mask_paths = []

    os.makedirs(os.path.dirname(out_image), exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    if os.path.exists(out_image) and not overwrite:
        raise FileExistsError(f"Output image exists (use --overwrite to replace): {out_image}")

    cropped_image = crop_z_sitk(case_plan["image"], z_min, z_max)
    sitk.WriteImage(cropped_image, out_image)

    for mask_name, mask_path in case_plan["mask_files"]:
        out_mask_path = os.path.join(out_mask_dir, mask_name)
        if os.path.exists(out_mask_path) and not overwrite:
            raise FileExistsError(f"Output mask exists (use --overwrite to replace): {out_mask_path}")
        mask_image = sitk.ReadImage(mask_path)
        cropped_mask = crop_z_sitk(mask_image, z_min, z_max)
        sitk.WriteImage(cropped_mask, out_mask_path)
        out_mask_paths.append(out_mask_path)

    return out_image, out_mask_dir, out_mask_paths


def make_manifest_record(case_plan, input_root, output_root, out_image, out_mask_dir):
    subject = case_plan["subject"]
    case_name = case_plan["case"]
    case_id = f"{subject}/{case_name}"

    rel_source_image = os.path.relpath(case_plan["image_path"], input_root)
    rel_source_mask_dir = os.path.relpath(case_plan["mask_dir"], input_root)
    rel_cropped_image = os.path.relpath(out_image, output_root)
    rel_cropped_mask_dir = os.path.relpath(out_mask_dir, output_root)

    return {
        "case_id": case_id,
        "subject": subject,
        "case": case_name,
        "source_image": rel_source_image,
        "source_image_abs": os.path.abspath(case_plan["image_path"]),
        "source_mask_dir": rel_source_mask_dir,
        "source_mask_dir_abs": os.path.abspath(case_plan["mask_dir"]),
        "cropped_image": rel_cropped_image,
        "cropped_image_abs": os.path.abspath(out_image),
        "cropped_mask_dir": rel_cropped_mask_dir,
        "cropped_mask_dir_abs": os.path.abspath(out_mask_dir),
        "mask_files": [name for name, _ in case_plan["mask_files"]],
        "z": case_plan["z"],
    }


def run_preprocess(input_root, output_root, z_margin=5, overwrite=False, dry_run=False):
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    images_root = os.path.join(input_root, "images")
    masks_root = os.path.join(input_root, "masks")

    cases = list_image_cases(images_root)
    print(f"Found {len(cases)} image cases.")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Z margin: {z_margin}")
    print(f"Dry run: {dry_run}")

    manifest_cases = []
    for idx, (subject, case_name, image_path) in enumerate(cases, start=1):
        case_id = f"{subject}/{case_name}"
        mask_dir = os.path.join(masks_root, subject, case_name)
        print(f"[{idx:03d}/{len(cases):03d}] Planning {case_id}")
        case_plan = compute_case_plan(subject, case_name, image_path, mask_dir, z_margin)
        z = case_plan["z"]
        print(
            f"  z_raw=[{z['z_min_raw']},{z['z_max_raw']}], "
            f"z_crop=[{z['z_min']},{z['z_max']}], depth {z['original_depth']} -> {z['cropped_depth']}"
        )

        if dry_run:
            out_image = os.path.join(output_root, os.path.relpath(image_path, input_root))
            out_mask_dir = os.path.join(output_root, "masks", subject, case_name)
        else:
            out_image, out_mask_dir, _ = write_case_outputs(case_plan, input_root, output_root, overwrite)

        manifest_cases.append(make_manifest_record(case_plan, input_root, output_root, out_image, out_mask_dir))

    if dry_run:
        print("Dry run complete. No files were written.")
        return

    os.makedirs(output_root, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset": "quadra",
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": input_root,
        "output_root": output_root,
        "z_margin": int(z_margin),
        "num_cases": len(manifest_cases),
        "cases": manifest_cases,
    }
    manifest_path = os.path.join(output_root, "boundaries_z.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop QUADRA images/masks in Z using union-of-masks boundaries."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/quadra_dataset",
        help="Input dataset root containing images/ and masks/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/quadra_dataset_cropped",
        help="Output dataset root for cropped images/masks and manifest.",
    )
    parser.add_argument(
        "--z-margin",
        type=int,
        default=5,
        help="Extra slices added to both z_min and z_max (clipped to image range).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned crop bounds; do not write files.",
    )
    args = parser.parse_args()
    if args.z_margin < 0:
        parser.error("--z-margin must be >= 0")
    return args


def main():
    args = parse_args()
    run_preprocess(
        input_root=args.input_root,
        output_root=args.output_root,
        z_margin=args.z_margin,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
