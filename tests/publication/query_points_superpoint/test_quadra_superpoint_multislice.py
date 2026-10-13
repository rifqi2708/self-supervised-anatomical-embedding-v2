"""Central test collection; stage inferred from inspected imports: query_points/superpoint.

Purpose: Central test collection; stage inferred from inspected imports: query_points/superpoint.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: publication, unittest.
Role: automated test.
Usage: test discovery from the repository root.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    __package__ = 'tests.publication.query_points_superpoint'
import unittest

from publication.investigations.superpoint.superpoint_multislice_gate import (
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
