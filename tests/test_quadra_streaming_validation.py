import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.quadra.streaming_cycle_error import stream_global_match
from tools.quadra.streaming_embedding import build_tile_plan
from tools.quadra.validate_streaming_equivalence import (
    ArrayEmbeddingCache,
    build_validation_summary,
    dense_global_match,
    descriptor_summary_rows,
    deterministic_crop_start,
    evaluate_descriptor_gates,
    feature_region_masks,
    internal_seam_distance,
    render_report,
    write_csv,
)


class QuadraStreamingValidationTests(unittest.TestCase):
    def test_deterministic_crop_is_centered_and_clamped(self):
        mask = np.zeros((20, 30, 40), dtype=np.uint8)
        mask[8:12, 18:22, 33:37] = 1

        start = deterministic_crop_start(mask, (16, 12, 8))

        self.assertEqual(start, (24, 13, 5))

    def test_internal_seam_distance_ignores_external_boundaries(self):
        points = np.array([[0, 0, 0], [31, 20, 10], [32, 20, 10], [48, 32, 10]])

        distance = internal_seam_distance(points, (64, 64, 32), (32, 32, 16))

        np.testing.assert_allclose(distance, [16, 1, 0, 0])

    def test_feature_regions_separate_seams_and_interiors(self):
        masks = feature_region_masks((64, 64, 32), (32, 32, 16), (4, 4, 2))

        self.assertGreater(np.count_nonzero(masks["seam"]), 0)
        self.assertGreater(np.count_nonzero(masks["interior"]), 0)
        self.assertFalse(np.any(masks["seam"] & masks["interior"]))
        np.testing.assert_array_equal(masks["all"], masks["seam"] | masks["interior"])

    def test_identical_descriptors_pass_engineering_gates(self):
        rng = np.random.default_rng(7)
        descriptors = rng.normal(size=(8, 32, 64, 64)).astype(np.float32)
        plan = build_tile_plan((128, 128, 64), (128, 128, 64), (32, 32, 16))

        rows, error = descriptor_summary_rows(
            descriptors,
            descriptors.copy(),
            plan,
            "fine",
            {
                "phase": "crop",
                "timepoint": "test",
                "organ": "colon",
                "comparison": "dense_vs_tiled_fp16",
            },
        )
        gates = evaluate_descriptor_gates(rows)

        self.assertLessEqual(float(np.max(np.abs(error))), 2e-7)
        self.assertTrue(all(item["status"] == "pass" for item in gates))

    def test_summary_and_report_preserve_failed_gate(self):
        matcher_rows = [
            {
                "phase": "crop",
                "coordinate_match": False,
                "score_abs_diff": 0.0,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_csv(output / "matcher_equivalence.csv", matcher_rows)
            summary = build_validation_summary(output, {"crop": {"status": "complete"}})
            report = render_report(
                summary,
                {
                    "subject_id": "quadra_hc_021",
                    "checkpoint_role": "base_sam_engineering",
                    "norm_spacing_xyz": [2.0, 2.0, 2.0],
                },
            )

        self.assertEqual(summary["overall_status"], "fail")
        self.assertIn("STREAMING_MATCHER_COORDINATES", report.upper())
        self.assertIn("**FAIL**", report)

    def test_streaming_match_finds_optimum_beyond_chunk_boundary(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is unavailable in the local CPU test environment")

        query_fine = np.zeros((2, 1, 2, 4), dtype=np.float32)
        query_fine[0, :, :, :] = 1.0
        target_fine = np.zeros_like(query_fine)
        target_fine[1, :, :, :] = 1.0
        target_fine[:, 0, 1, 3] = (1.0, 0.0)
        coarse = np.zeros((2, 1, 1, 1), dtype=np.float32)
        query_cache = ArrayEmbeddingCache(query_fine, coarse, (8, 4, 2))
        target_cache = ArrayEmbeddingCache(target_fine, coarse, (8, 4, 2))
        queries = np.array([[0, 0, 0]], dtype=np.int64)

        dense_points, dense_scores, _ = dense_global_match(query_cache, target_cache, queries, 1, "cpu")
        streamed_points, streamed_scores, _ = stream_global_match(
            query_cache,
            target_cache,
            queries,
            query_batch_size=1,
            match_chunk_xyz=(4, 4, 2),
            device="cpu",
        )

        np.testing.assert_array_equal(streamed_points, dense_points)
        np.testing.assert_allclose(streamed_scores, dense_scores, atol=1e-6)
        self.assertGreaterEqual(int(streamed_points[0, 0]), 4)


if __name__ == "__main__":
    unittest.main()
