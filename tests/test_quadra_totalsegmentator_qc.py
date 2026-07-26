import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from tools.quadra.totalsegmentator.core import WorkflowError
from tools.quadra.totalsegmentator.qc import (
    derive_spinal_cord_segments,
    validate_mask,
)


def save_mask(path: Path, shape, affine, coordinates):
    data = np.zeros(shape, dtype=np.uint8)
    for coordinate in coordinates:
        data[coordinate] = 1
    nib.save(nib.Nifti1Image(data, affine), path)


class TotalSegmentatorQCTests(unittest.TestCase):
    def _derive(self, root: Path, affine: np.ndarray, reverse: bool = False):
        shape = (5, 5, 20)
        path = lambda name: root / f"{name}.nii.gz"
        cord = [(2, 2, z) for z in range(2, 20)]
        positions = {"c1": 18, "c7": 15, "t1": 13, "t12": 5, "l1": 3}
        if reverse:
            positions = {name: shape[2] - 1 - z for name, z in positions.items()}
            cord = [(2, 2, shape[2] - 1 - z) for z in range(2, 20)]
        save_mask(path("cord"), shape, affine, cord)
        for name, z in positions.items():
            save_mask(path(name), shape, affine, [(1, 1, z), (2, 2, z)])
        result = derive_spinal_cord_segments(
            path("cord"),
            path("c1"),
            path("c7"),
            path("t1"),
            path("t12"),
            path("l1"),
            path("cervical"),
            path("thoracic"),
        )
        cervical = np.asanyarray(nib.load(path("cervical")).dataobj).astype(bool)
        thoracic = np.asanyarray(nib.load(path("thoracic")).dataobj).astype(bool)
        source = np.asanyarray(nib.load(path("cord")).dataobj).astype(bool)
        self.assertTrue(cervical.any())
        self.assertTrue(thoracic.any())
        self.assertFalse(np.any(cervical & thoracic))
        self.assertFalse(np.any(cervical & ~source))
        self.assertFalse(np.any(thoracic & ~source))
        self.assertTrue(result["provisional"])

    def test_derivation_handles_positive_superior_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            self._derive(Path(directory), np.eye(4))

    def test_derivation_handles_negative_voxel_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            affine = np.diag([1.0, 1.0, -1.0, 1.0])
            affine[2, 3] = 19.0
            self._derive(Path(directory), affine, reverse=True)

    def test_empty_landmark_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (3, 3, 10)
            affine = np.eye(4)
            save_mask(root / "cord.nii.gz", shape, affine, [(1, 1, z) for z in range(10)])
            for name, z in (("c1", 9), ("c7", 7), ("t1", 6), ("t12", 2)):
                save_mask(root / f"{name}.nii.gz", shape, affine, [(1, 1, z)])
            save_mask(root / "l1.nii.gz", shape, affine, [])
            with self.assertRaisesRegex(WorkflowError, "empty"):
                derive_spinal_cord_segments(
                    root / "cord.nii.gz",
                    root / "c1.nii.gz",
                    root / "c7.nii.gz",
                    root / "t1.nii.gz",
                    root / "t12.nii.gz",
                    root / "l1.nii.gz",
                    root / "cervical.nii.gz",
                    root / "thoracic.nii.gz",
                )

    def test_mask_geometry_binary_and_nonempty_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)), reference)
            valid = root / "valid.nii.gz"
            save_mask(valid, (3, 3, 3), np.eye(4), [(1, 1, 1)])
            self.assertEqual(validate_mask(valid, reference)["voxel_count"], 1)

            nonbinary = root / "nonbinary.nii.gz"
            data = np.zeros((3, 3, 3), dtype=np.uint8)
            data[1, 1, 1] = 2
            nib.save(nib.Nifti1Image(data, np.eye(4)), nonbinary)
            with self.assertRaisesRegex(WorkflowError, "binary"):
                validate_mask(nonbinary, reference)

            wrong_affine = root / "wrong_affine.nii.gz"
            shifted = np.eye(4)
            shifted[0, 3] = 2
            save_mask(wrong_affine, (3, 3, 3), shifted, [(1, 1, 1)])
            with self.assertRaisesRegex(WorkflowError, "affine"):
                validate_mask(wrong_affine, reference)


if __name__ == "__main__":
    unittest.main()
