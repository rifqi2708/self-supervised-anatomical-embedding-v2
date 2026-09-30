import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.quadra.interior_keypoint_batch import (
    choose_review_indices,
    run_batch,
    strict_first_relaxed_selection,
    window_for_organ,
)
from tools.quadra.interior_keypoint_gate import draw_gap_crosshair, parse_args


class WindowPolicyTests(unittest.TestCase):
    def test_four_fixed_categories(self):
        self.assertEqual(window_for_organ("liver")[0], "soft_tissue")
        self.assertEqual(window_for_organ("optic_nerve_left")[0], "soft_tissue")
        self.assertEqual(window_for_organ("lung_lower_lobe_left")[0], "lung")
        self.assertEqual(window_for_organ("vertebrae_C4")[0], "bone")
        self.assertEqual(window_for_organ("rib_left_6")[0], "bone")
        self.assertEqual(window_for_organ("skull")[0], "bone")
        self.assertEqual(window_for_organ("brain")[0], "brain")

    def test_preset_values_are_fixed(self):
        self.assertEqual(window_for_organ("pancreas")[1], {"center_hu": 40.0, "width_hu": 400.0})
        self.assertEqual(window_for_organ("lung_upper_lobe_right")[1], {"center_hu": -600.0, "width_hu": 1500.0})
        self.assertEqual(window_for_organ("hip_left")[1], {"center_hu": 500.0, "width_hu": 2000.0})
        self.assertEqual(window_for_organ("brain")[1], {"center_hu": 40.0, "width_hu": 80.0})


class StrictFirstSelectionTests(unittest.TestCase):
    def _points(self, count):
        return np.stack([np.arange(count) * 10, np.zeros(count), np.zeros(count)], axis=1)

    def test_zero_candidates_is_valid(self):
        result = strict_first_relaxed_selection(
            np.empty((0, 3)), np.empty((0,)), np.empty((0,)), (1, 1, 1), 3, 5, 100
        )
        self.assertEqual(result["status"], "NO_CANDIDATES")
        self.assertEqual(len(result["selected_indices"]), 0)

    def test_optic_nerve_scale_supply_is_partial_not_error(self):
        count = 70
        result = strict_first_relaxed_selection(
            self._points(count),
            np.linspace(1.0, 0.1, count),
            np.ones(count),
            (1, 1, 1),
            3,
            5,
            100,
        )
        self.assertEqual(result["status"], "PARTIAL_SUPPLY")
        self.assertEqual(len(result["selected_indices"]), 70)
        self.assertEqual(result["selected_strict"], 0)
        self.assertEqual(result["selected_relaxed"], 70)

    def test_exact_strict_quota(self):
        count = 100
        result = strict_first_relaxed_selection(
            self._points(count), np.ones(count), np.full(count, 6.0), (1, 1, 1), 3, 5, 100
        )
        self.assertEqual(result["status"], "FULL_QUOTA_STRICT")
        self.assertEqual(len(np.unique(result["selected_indices"])), 100)

    def test_excess_strict_supply_is_capped(self):
        count = 120
        result = strict_first_relaxed_selection(
            self._points(count), np.linspace(1, 0, count), np.full(count, 6.0), (1, 1, 1), 3, 5, 100
        )
        self.assertEqual(result["status"], "FULL_QUOTA_STRICT")
        self.assertEqual(len(result["selected_indices"]), 100)

    def test_relaxed_points_fill_strict_shortfall(self):
        count = 120
        distances = np.r_[np.full(60, 6.0), np.full(60, 2.0)]
        result = strict_first_relaxed_selection(
            self._points(count), np.linspace(1, 0, count), distances, (1, 1, 1), 3, 5, 100
        )
        self.assertEqual(result["status"], "FULL_QUOTA_WITH_RELAXED")
        self.assertEqual(result["selected_strict"], 60)
        self.assertEqual(result["selected_relaxed"], 40)


class ReviewSelectionAndMarkerTests(unittest.TestCase):
    def test_review_selection_is_deterministic_and_capped(self):
        points = np.stack([np.arange(100), np.zeros(100), np.zeros(100)], axis=1)
        scores = np.linspace(0.0, 1.0, 100)
        first = choose_review_indices(points, scores, 20, (1, 1, 1))
        second = choose_review_indices(points, scores, 20, (1, 1, 1))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(np.unique(first)), 20)
        self.assertIn(0, first)
        self.assertIn(99, first)

    def test_review_selection_returns_all_small_supply(self):
        points = np.zeros((7, 3))
        selected = choose_review_indices(points, np.arange(7), 20, (1, 1, 1))
        np.testing.assert_array_equal(selected, np.arange(7))

    def test_gap_crosshair_does_not_include_centre_in_line_data(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        draw_gap_crosshair(axis, 10.0, 20.0, inner=2.0, outer=7.0)
        self.assertEqual(len(axis.lines), 4)
        for line in axis.lines:
            coordinates = list(zip(line.get_xdata(), line.get_ydata()))
            self.assertNotIn((10.0, 20.0), coordinates)
        plt.close(figure)


class ResumeCompatibilityTests(unittest.TestCase):
    def _write_nifti(self, path, data):
        import nibabel as nib

        nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4)), str(path))

    def test_resume_reuses_compatible_outputs_and_rejects_changed_quota(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ct = root / "ct.nii.gz"
            masks = root / "masks"
            masks.mkdir()
            output = root / "output"
            self._write_nifti(ct, np.zeros((12, 12, 12), dtype=np.float32))
            mask = np.zeros((12, 12, 12), dtype=np.uint8)
            mask[4:8, 4:8, 4:8] = 1
            self._write_nifti(masks / "optic_nerve_left.nii.gz", mask)
            base = [
                "--ct", str(ct),
                "--mask-dir", str(masks),
                "--organs", "all",
                "--output-dir", str(output),
                "--num-points", "10",
                "--review-points", "5",
            ]
            first = run_batch(parse_args(base))
            self.assertEqual(first["status"], "complete")
            resumed = run_batch(parse_args(base + ["--resume"]))
            self.assertEqual(resumed["status"], "complete")
            with (output / "organ_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertTrue(
                (output / "optic_nerve_left" / "whole_organ_overview.png").is_file()
            )
            changed = list(base)
            changed[changed.index("10")] = "9"
            with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
                run_batch(parse_args(changed + ["--resume"]))


if __name__ == "__main__":
    unittest.main()
