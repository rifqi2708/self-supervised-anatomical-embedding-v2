import unittest

import numpy as np

from tools.quadra.superpoint_threshold_gate import (
    greedy_deduplicate_3d,
    set_detection_threshold,
    union_slice_indices,
    validate_gate_parameters,
    validate_nested_detections,
)


class _Configuration:
    detection_threshold = 0.005


class _Model:
    conf = _Configuration()


class SuperPointThresholdGateTests(unittest.TestCase):
    def test_parameters_require_descending_thresholds_and_increasing_radii(self):
        self.assertEqual(
            validate_gate_parameters([0.005, 0.002, 0.001], [3, 5, 10]),
            ((0.005, 0.002, 0.001), (3.0, 5.0, 10.0)),
        )
        with self.assertRaisesRegex(ValueError, "highest to lowest"):
            validate_gate_parameters([0.001, 0.005], [3, 5])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_gate_parameters([0.005], [5, 3])

    def test_union_slice_indices_is_sorted_and_unique(self):
        self.assertEqual(union_slice_indices([4, 2, 3], [3, 8]), [2, 3, 4, 8])

    def test_model_threshold_is_explicitly_mutated(self):
        model = _Model()
        set_detection_threshold(model, 0.002)
        self.assertEqual(model.conf.detection_threshold, 0.002)

    def test_physical_deduplication_uses_spacing_and_confidence(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [10.0, 0.0, 0.0],
            ]
        )
        scores = np.array([0.8, 0.9, 0.7])
        kept = greedy_deduplicate_3d(points, scores, [1.0, 1.0, 2.0], radius_mm=3.0)
        self.assertEqual(kept.tolist(), [1, 2])

    def test_exact_radius_is_retained(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        scores = np.array([0.8, 0.7])
        kept = greedy_deduplicate_3d(points, scores, [1.0, 1.0, 1.0], radius_mm=3.0)
        self.assertEqual(kept.tolist(), [0, 1])

    def test_lower_threshold_must_retain_prior_points(self):
        previous = {
            "keypoints_xy": np.array([[1.0, 2.0]]),
            "scores": np.array([0.9]),
        }
        current = {
            "keypoints_xy": np.array([[1.0, 2.0], [5.0, 6.0]]),
            "scores": np.array([0.9, 0.1]),
        }
        validate_nested_detections(previous, current, 0.005, 0.002)
        with self.assertRaisesRegex(RuntimeError, "removed"):
            validate_nested_detections(previous, {"keypoints_xy": np.empty((0, 2)), "scores": np.empty(0)}, 0.005, 0.002)


if __name__ == "__main__":
    unittest.main()
