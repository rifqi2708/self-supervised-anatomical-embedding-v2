import os

import nibabel as nib
import numpy as np


COORD_SPACE_RAW_ITK = "raw_itk_voxel"
COORD_GROUPS = (
    ("pt1", "test"),
    ("pt2", "retest"),
    ("pt1_back", "test"),
)


def is_nifti_file(name):
    return isinstance(name, str) and (name.endswith(".nii.gz") or name.endswith(".nii"))


def resolve_subject_images(dataset_root, subject_id):
    image_dir = os.path.join(dataset_root, "images", subject_id)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = [name for name in sorted(os.listdir(image_dir)) if is_nifti_file(name)]
    test_files = [name for name in image_files if "_Test_" in name]
    retest_files = [name for name in image_files if "_Retest_" in name]
    if len(test_files) != 1 or len(retest_files) != 1:
        raise RuntimeError(
            f"Expected one Test and one Retest image in {image_dir}, "
            f"found Test={test_files}, Retest={retest_files}"
        )

    return {
        "test": os.path.join(image_dir, test_files[0]),
        "retest": os.path.join(image_dir, retest_files[0]),
    }


def build_sam_to_raw_transform(nifti_path):
    raw_img = nib.load(nifti_path)
    raw_shape = tuple(int(v) for v in raw_img.shape[:3])
    canonical = nib.as_closest_canonical(raw_img)
    ras_shape = tuple(int(v) for v in canonical.shape[:3])

    raw_ornt = nib.orientations.io_orientation(raw_img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    raw_to_ras_ornt = nib.orientations.ornt_transform(raw_ornt, ras_ornt)
    ras_to_raw_aff = nib.orientations.inv_ornt_aff(raw_to_ras_ornt, raw_shape)

    sam_to_ras_aff = np.array(
        [
            [-1.0, 0.0, 0.0, ras_shape[0] - 1],
            [0.0, -1.0, 0.0, ras_shape[1] - 1],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return ras_to_raw_aff @ sam_to_ras_aff, raw_shape


def transform_point_xyz(point_xyz, sam_to_raw_aff, raw_shape):
    point_h = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=float)
    raw_point = sam_to_raw_aff @ point_h
    raw_point_xyz = np.rint(raw_point[:3]).astype(int)

    if not np.allclose(raw_point[:3], raw_point_xyz, atol=1e-6):
        raise ValueError(f"Non-integer transformed coordinate: {point_xyz} -> {raw_point[:3].tolist()}")
    if np.any(raw_point_xyz < 0) or np.any(raw_point_xyz >= np.array(raw_shape, dtype=int)):
        raise ValueError(f"Transformed point out of bounds: {point_xyz} -> {raw_point_xyz.tolist()}")
    return raw_point_xyz
