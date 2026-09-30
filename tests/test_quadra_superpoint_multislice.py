import unittest

from tools.quadra.superpoint_multislice_gate import (
    choose_extent_quantile_slices,
    summarize_window_cases,
)


class SuperPointMultisliceTests(unittest.TestCase):
    def test_quantiles_use_actual_nonempty_slices(self):
        selected = choose_extent_quantile_slices([2, 3, 7, 8, 20, 21, 30], count=4)
        self.assertEqual(selected, [2, 7, 20, 30])

    def test_quantiles_use_all_slices_when_extent_is_short(self):
        self.assertEqual(choose_extent_quantile_slices([4, 9], count=7), [4, 9])

    def test_invalid_slice_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            choose_extent_quantile_slices([], count=7)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            choose_extent_quantile_slices([2, 2, 3], count=2)

    def test_aggregate_keeps_raw_denominators(self):
        cases = [
            {
                "organ_group": "kidneys",
                "window": {"name": "soft_tissue"},
                "slice_index": 4,
                "candidate_count": 10,
                "inside_count": 0,
                "inside_scores": [],
            },
            {
                "organ_group": "kidneys",
                "window": {"name": "soft_tissue"},
                "slice_index": 8,
                "candidate_count": 12,
                "inside_count": 3,
                "inside_scores": [0.1, 0.2, 0.3],
            },
        ]
        result = summarize_window_cases(cases)[0]
        self.assertEqual(result["candidate_count"], 22)
        self.assertEqual(result["inside_count"], 3)
        self.assertEqual(result["slices_with_inside_count"], 1)
        self.assertEqual(result["inside_per_slice"]["median"], 1.5)
        self.assertTrue(result["raw_candidates_are_not_3d_deduplicated"])


if __name__ == "__main__":
    unittest.main()
