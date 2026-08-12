import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.quadra.superpoint_adapter import (
    ensure_stride_compatible,
    sha256_file,
    validate_superpoint_assets,
    window_and_normalize_ct,
    write_json_atomic,
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
