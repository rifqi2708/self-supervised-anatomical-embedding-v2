import unittest

import numpy as np

from tools.quadra.superpoint_full_volume_gate import (
    build_slice_plan,
    classify_candidates,
    summarize_supply,
)


class SuperPointFullVolumeTests(unittest.TestCase):
    def test_slice_plan_runs_soft_tissue_once_everywhere_and_lung_only_in_extent(self):
        plan = build_slice_plan(5, [1, 3, 4])
        self.assertEqual(plan[0], ("soft_tissue", [0, 1, 2, 3, 4]))
        self.assertEqual(plan[1], ("lung", [1, 3, 4]))

    def test_invalid_lung_slice_plan_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            build_slice_plan(5, [1, 5])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_slice_plan(5, [1, 1])

    def test_candidate_membership_preserves_outside_exclusive_and_ambiguous(self):
        masks = {}
        for organ in ("bladder", "colon", "kidneys", "liver", "lungs"):
            masks[organ] = np.zeros((8, 8), dtype=bool)
        masks["bladder"][1, 1] = True
        masks["colon"][2, 2] = True
        masks["liver"][2, 2] = True
        points = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)

        memberships, count, exclusive = classify_candidates(masks, points)

        np.testing.assert_array_equal(count, [0, 1, 2])
        self.assertFalse(memberships["bladder"][0])
        self.assertEqual(exclusive.tolist(), ["", "bladder", ""])

    def test_supply_summary_keeps_raw_and_exclusive_denominators(self):
        organs = ("bladder", "colon", "kidneys", "liver", "lungs")
        windows = ("soft_tissue", "lung")
        rows = []
        for window in windows:
            for slice_index in (0, 1):
                row = {
                    "window_name": window,
                    "slice_index": slice_index,
                    "candidate_count": 10,
                }
                for organ in organs:
                    row["inside_{}_count".format(organ)] = 0
                    row["exclusive_{}_count".format(organ)] = 0
                    row["ambiguous_inside_{}_count".format(organ)] = 0
                rows.append(row)
        rows[0]["inside_bladder_count"] = 3
        rows[0]["exclusive_bladder_count"] = 2
        rows[0]["ambiguous_inside_bladder_count"] = 1
        scores = {(organ, window): [] for organ in organs for window in windows}
        scores[("bladder", "soft_tissue")] = [0.1, 0.2, 0.3]
        exclusive_scores = {key: [] for key in scores}
        exclusive_scores[("bladder", "soft_tissue")] = [0.2, 0.3]

        summary = summarize_supply(rows, scores, exclusive_scores)
        bladder_soft = next(
            row
            for row in summary
            if row["organ_group"] == "bladder" and row["window_name"] == "soft_tissue"
        )
        self.assertEqual(bladder_soft["all_candidate_count"], 20)
        self.assertEqual(bladder_soft["inside_count"], 3)
        self.assertEqual(bladder_soft["exclusive_count"], 2)
        self.assertEqual(bladder_soft["ambiguous_inside_count"], 1)
        self.assertTrue(bladder_soft["primary_window_for_organ"])
        self.assertAlmostEqual(bladder_soft["inside_score_median"], 0.2)


if __name__ == "__main__":
    unittest.main()
