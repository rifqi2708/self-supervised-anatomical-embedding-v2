import hashlib
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.quadra.superpoint_adapter import (
    ensure_stride_compatible,
    mask_boundary,
    model_keypoints_to_raw_voxels,
    native_xy_to_model_yx,
    sha256_file,
    validate_superpoint_assets,
    window_and_normalize_ct,
    write_keypoints_csv_atomic,
    write_json_atomic,
    write_overlay_png_atomic,
)
from tools.quadra.superpoint_smoke import parse_args, summarize_prediction


class SuperPointPreprocessingTests(unittest.TestCase):
    def test_fixed_hu_window_maps_to_unit_interval(self):
        values = np.array([[-200.0, -160.0, 40.0, 240.0, 300.0]], dtype=np.float32)
        result = window_and_normalize_ct(values, center=40.0, width=400.0)
        np.testing.assert_allclose(result, [[0.0, 0.0, 0.5, 1.0, 1.0]])
        self.assertEqual(result.dtype, np.float32)

    def test_nonfinite_slice_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            window_and_normalize_ct(np.array([[np.nan]], dtype=np.float32))

    def test_nonpositive_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "width > 0"):
            window_and_normalize_ct(np.zeros((8, 8)), width=0)

    def test_stride_compatible_shape_is_unchanged(self):
        image = np.zeros((512, 512), dtype=np.float32)
        self.assertEqual(ensure_stride_compatible(image), (512, 512))

    def test_stride_incompatible_shape_requires_explicit_policy(self):
        with self.assertRaisesRegex(ValueError, "explicit padding policy"):
            ensure_stride_compatible(np.zeros((511, 512), dtype=np.float32))

    def test_native_slice_is_transposed_to_model_row_column_order(self):
        native_xy = np.array([[1, 2, 3], [4, 5, 6]])
        model_yx = native_xy_to_model_yx(native_xy)
        np.testing.assert_array_equal(model_yx, [[1, 4], [2, 5], [3, 6]])
        self.assertTrue(model_yx.flags.c_contiguous)

    def test_model_pixels_map_directly_to_native_voxels_after_transpose(self):
        keypoints = np.array([[17.0, 23.0], [41.0, 59.0]], dtype=np.float32)
        raw = model_keypoints_to_raw_voxels(keypoints, slice_index=265)
        np.testing.assert_array_equal(raw, [[17.0, 23.0, 265.0], [41.0, 59.0, 265.0]])

    def test_mask_boundary_excludes_four_neighbour_interior(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        boundary = mask_boundary(mask)
        self.assertEqual(int(boundary.sum()), 8)
        self.assertFalse(boundary[2, 2])


class SuperPointProvenanceTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pth"
            path.write_bytes(b"known checkpoint")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"known checkpoint").hexdigest())

    @mock.patch("tools.quadra.superpoint_adapter._git_output")
    def test_verified_assets_return_provenance(self, git_output):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "SuperPoint"
            root.mkdir()
            (root / "superpoint_pytorch.py").write_text("# test\n", encoding="utf-8")
            checkpoint = root / "weights.pth"
            checkpoint.write_bytes(b"weights")
            expected_sha = hashlib.sha256(b"weights").hexdigest()
            git_output.side_effect = ["expected-commit", "", "https://example.test/SuperPoint.git"]

            result = validate_superpoint_assets(
                root,
                checkpoint,
                expected_commit="expected-commit",
                expected_checkpoint_sha256=expected_sha,
            )

        self.assertEqual(result["commit"], "expected-commit")
        self.assertEqual(result["checkpoint_sha256"], expected_sha)
        self.assertEqual(result["repository_origin"], "https://example.test/SuperPoint.git")

    @mock.patch("tools.quadra.superpoint_adapter._git_output")
    def test_dirty_external_repository_is_rejected(self, git_output):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "SuperPoint"
            root.mkdir()
            (root / "superpoint_pytorch.py").write_text("# test\n", encoding="utf-8")
            checkpoint = root / "weights.pth"
            checkpoint.write_bytes(b"weights")
            git_output.side_effect = ["expected-commit", " M superpoint_pytorch.py"]
            with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
                validate_superpoint_assets(
                    root,
                    checkpoint,
                    expected_commit="expected-commit",
                    expected_checkpoint_sha256=hashlib.sha256(b"weights").hexdigest(),
                )

    def test_atomic_json_refuses_existing_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result.json"
            write_json_atomic(destination, {"status": "first"})
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_json_atomic(destination, {"status": "second"})

    def test_atomic_json_refuses_existing_temporary_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result.json"
            destination.with_name("result.json.tmp").write_text("interrupted\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "temporary result"):
                write_json_atomic(destination, {"status": "new"})

    def test_keypoint_csv_preserves_model_and_native_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "points.csv"
            write_keypoints_csv_atomic(
                destination,
                np.array([[17.0, 23.0]], dtype=np.float32),
                np.array([0.75], dtype=np.float32),
                slice_index=265,
            )
            with destination.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_x_pixel"], "17.0")
        self.assertEqual(rows[0]["raw_x_voxel"], "17.0")
        self.assertEqual(rows[0]["raw_z_voxel"], "265.0")
        self.assertEqual(rows[0]["coord_space"], "native_nifti_voxel_xyz")

    def test_overlay_png_is_created_at_native_resolution_plus_header(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "overlay.png"
            mask = np.zeros((16, 16), dtype=bool)
            mask[4:12, 4:12] = True
            write_overlay_png_atomic(
                destination,
                np.zeros((16, 16), dtype=np.float32),
                np.array([[8.0, 8.0]], dtype=np.float32),
                [{"name": "organ", "mask_yx": mask}],
                "test overlay",
            )
            with Image.open(destination) as image:
                self.assertEqual(image.width, 16)
                self.assertGreater(image.height, 16)


class SuperPointSmokeInterfaceTests(unittest.TestCase):
    def test_slice_index_and_output_are_required(self):
        args = parse_args(
            [
                "--ct",
                "test.nii.gz",
                "--slice-index",
                "265",
                "--superpoint-root",
                "SuperPoint",
                "--checkpoint",
                "weights.pth",
                "--output-json",
                "smoke.json",
            ]
        )
        self.assertEqual(args.slice_index, 265)
        self.assertEqual(args.window_center, 40.0)
        self.assertEqual(args.window_width, 400.0)
        self.assertIsNone(args.output_keypoints_csv)

    def test_prediction_summary_retains_denominators_and_bounds(self):
        prediction = {
            "keypoints_xy": np.array([[2.0, 3.0], [8.0, 9.0]], dtype=np.float32),
            "scores": np.array([0.1, 0.3], dtype=np.float32),
            "descriptors": np.ones((2, 256), dtype=np.float32),
            "runtime_seconds": 0.25,
            "peak_gpu_memory_bytes": 1024,
        }
        result = summarize_prediction(prediction)
        self.assertEqual(result["keypoint_count"], 2)
        self.assertEqual(result["descriptor_shape"], [2, 256])
        self.assertEqual(result["coordinate_bounds_xy"]["min"], [2.0, 3.0])
        self.assertEqual(result["coordinate_bounds_xy"]["max"], [8.0, 9.0])
        self.assertTrue(result["outputs_finite"])


if __name__ == "__main__":
    unittest.main()
