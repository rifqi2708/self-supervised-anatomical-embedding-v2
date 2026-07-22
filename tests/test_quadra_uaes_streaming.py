import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.quadra.streaming_cycle_error import EmbeddingCache, stream_global_match_uaes, write_json
from tools.quadra.streaming_cycle_error_cohort import build_subject_command, parse_args as parse_cohort_args
from tools.quadra.streaming_cycle_error_uaes import parse_args
from tools.quadra.streaming_embedding import build_tile_plan
from tools.quadra.uaes_matching import (
    fine_to_native,
    local_anchor_grid,
    native_to_fine,
    robust_affine_predict,
)
from tools.quadra.validate_uaes_streaming import _descriptor_rows


def make_cache(root: Path, name: str, fine: np.ndarray, semantic: np.ndarray, coarse=None):
    cache_dir = root / name
    cache_dir.mkdir()
    if coarse is None:
        coarse = fine
    np.save(cache_dir / "fine.npy", fine.astype(np.float16))
    np.save(cache_dir / "coarse.npy", coarse.astype(np.float16))
    np.save(cache_dir / "semantic.npy", semantic.astype(np.float16))
    shape_xyz = [fine.shape[3], fine.shape[2], fine.shape[1]]
    write_json(
        cache_dir / "manifest.json",
        {
            "complete": True,
            "model_profile": "uae_s",
            "native_sam_shape_xyz": shape_xyz,
            "native_spacing_xyz": [4.0, 4.0, 4.0],
            "norm_ratio_xyz": [2.0, 2.0, 2.0],
            "features": {
                "fine": {"file": "fine.npy", "valid_shape_xyz": shape_xyz},
                "coarse": {"file": "coarse.npy", "valid_shape_xyz": shape_xyz},
                "semantic": {"file": "semantic.npy", "valid_shape_xyz": shape_xyz},
            },
        },
    )
    return EmbeddingCache(cache_dir)


class UaesCliTests(unittest.TestCase):
    def test_uaes_entrypoint_defaults(self):
        args = parse_args([])
        self.assertEqual(args.config_file, "configs/samv2/samv2_NIHLN.py")
        self.assertEqual(args.checkpoint_file, "checkpoints/SAMv2_iter_20000.pth")
        self.assertEqual(args.matching_modes, ("global_nn", "fixed_point"))
        self.assertEqual(args.fixed_point_margin, (2, 2, 2))
        self.assertEqual(args.query_batch_size, 64)
        self.assertFalse(args.keep_cache)

    def test_cohort_uaes_profile_forwards_all_matching_settings(self):
        args = parse_cohort_args(["--model-profile", "uae_s", "--num-points", "7"])
        command = build_subject_command(args, "quadra_hc_021")
        self.assertIn("tools.quadra.streaming_cycle_error_uaes", command)
        self.assertIn("--matching-modes", command)
        self.assertIn("fixed_point", command)
        self.assertEqual(command[command.index("--num-points") + 1], "7")
        self.assertEqual(args.config_file, "configs/samv2/samv2_NIHLN.py")
        self.assertEqual(args.query_batch_size, 64)

    def test_cohort_preserves_explicit_uaes_query_batch(self):
        args = parse_cohort_args(["--model-profile", "uae_s", "--query-batch-size", "16"])
        self.assertEqual(args.query_batch_size, 16)


class UaesCacheAndMatcherTests(unittest.TestCase):
    def test_empty_descriptor_region_is_exported_as_nan(self):
        plan = build_tile_plan((128, 128, 64), (160, 160, 80), (48, 48, 24))
        values = np.ones((2, 8, 16, 16), dtype=np.float32)
        rows, _ = _descriptor_rows(values, values, plan, "coarse", "colon", "test")
        empty = [row for row in rows if row["voxel_count"] == 0]
        self.assertTrue(empty)
        self.assertTrue(np.isnan(empty[0]["p01_cosine"]))

    def test_three_feature_cache_closes_idempotently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            values = np.ones((2, 1, 1, 2), dtype=np.float32)
            cache = make_cache(Path(temp_dir), "cache", values, values)
            self.assertEqual(cache.valid_array("semantic").shape, values.shape)
            cache.close()
            cache.close()
            with self.assertRaises(RuntimeError):
                cache.valid_array("semantic")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is available on RunPod, not local Python")
    def test_streamed_semantic_optimum_across_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = np.zeros((2, 1, 1, 4), dtype=np.float32)
            target[:, 0, 0, 0] = [1, 0]
            target[:, 0, 0, 1] = [0, 1]
            target[:, 0, 0, 2] = [-1, 0]
            target[:, 0, 0, 3] = [0, -1]
            query = target.copy()
            query[:, 0, 0, 0] = [0, -1]
            query_cache = make_cache(root, "query", query, query)
            target_cache = make_cache(root, "target", target, target)
            points, scores, profile = stream_global_match_uaes(
                query_cache,
                target_cache,
                [[0, 0, 0]],
                query_batch_size=1,
                match_chunk_xyz=(2, 1, 1),
                device="cpu",
                output_space="fine",
            )
            np.testing.assert_array_equal(points[0], [3, 0, 0])
            self.assertAlmostEqual(float(scores[0]), 1.0, places=5)
            self.assertEqual(profile["output_space"], "fine")
            query_cache.close()
            target_cache.close()

    def test_native_fine_coordinate_round_trip_matches_official_formula(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            values = np.ones((1, 2, 2, 2), dtype=np.float32)
            cache = make_cache(Path(temp_dir), "cache", values, values)
            fine = native_to_fine(cache, [[1, 1, 1]])
            native = fine_to_native(cache, fine)
            np.testing.assert_array_equal(fine, [[1, 1, 1]])
            np.testing.assert_array_equal(native, [[1, 1, 1]])
            cache.close()


class FixedPointGeometryTests(unittest.TestCase):
    def test_anchor_grid_clips_and_deduplicates_edges(self):
        points = local_anchor_grid((0, 0, 0), (4, 4, 4), (2, 2, 2))
        self.assertTrue(np.all(points >= 0))
        self.assertTrue(np.all(points < 4))
        self.assertEqual(len(points), len(np.unique(points, axis=0)))

    def test_robust_affine_recovers_translation(self):
        source = np.array(
            [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 10, 10]],
            dtype=float,
        )
        translation = np.array([5, 7, 9])
        target = source + translation
        predicted, profile = robust_affine_predict(source, target, [2, 3, 4], [100, 100, 100])
        np.testing.assert_array_less(np.abs(predicted - np.array([7, 10, 13])), np.full(3, 2))
        self.assertEqual(profile["affine_mode"], "3d")
        self.assertEqual(profile["prediction_method"], "official_shift_corrected_anchor_mean")

    def test_robust_affine_matches_official_shift_and_average_order(self):
        returned = np.array(
            [[1, 2, 3], [5, 2, 3], [1, 6, 3], [1, 2, 7], [5, 6, 7]],
            dtype=float,
        )
        original = np.array([2, 3, 4], dtype=float)
        linear = np.diag([2.0, 3.0, 4.0])
        translation = np.array([10.0, 20.0, 30.0])
        target = returned @ linear.T + translation
        predicted, _ = robust_affine_predict(
            returned,
            target,
            original,
            [200, 200, 200],
        )
        expected = (original @ linear.T + translation).astype(np.int64)
        # The official implementation truncates each corrected float anchor
        # before averaging, so numerical fit noise can move a coordinate down
        # by one voxel even for a mathematically exact affine transform.
        np.testing.assert_array_less(np.abs(predicted - expected), np.full(3, 2))

    def test_degenerate_affine_is_reported(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        with self.assertRaisesRegex(ValueError, "degenerate_affine_geometry"):
            robust_affine_predict(source, source, [0, 0, 0], [10, 10, 10])


if __name__ == "__main__":
    unittest.main()
