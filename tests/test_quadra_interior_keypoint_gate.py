import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.quadra.interior_keypoint_gate import (
    CSV_FIELDS,
    boundary_distances_mm,
    farthest_point_sample,
    greedy_radius_suppression,
    mask_bounding_box,
    padded_crop_slices,
    validate_parameters,
    window_ct,
    write_csv_atomic,
)


class InteriorKeypointGeometryTests(unittest.TestCase):
    def test_mask_bounding_box_is_half_open(self):
        mask = np.zeros((9, 10, 11), dtype=bool)
        mask[2:6, 3:8, 4:9] = True
        minimum, maximum = mask_bounding_box(mask)
        np.testing.assert_array_equal(minimum, [2, 3, 4])
        np.testing.assert_array_equal(maximum, [6, 8, 9])

    def test_physical_crop_padding_is_clamped_to_image(self):
        mask = np.zeros((10, 12, 14), dtype=bool)
        mask[1:3, 5:7, 11:14] = True
        slices, start = padded_crop_slices(mask, (2.0, 1.0, 4.0), padding_mm=4.0)
        self.assertEqual(slices, (slice(0, 5), slice(1, 11), slice(10, 14)))
        np.testing.assert_array_equal(start, [0, 1, 10])

    def test_boundary_distance_uses_physical_spacing(self):
        mask = np.zeros((7, 7, 7), dtype=bool)
        mask[1:6, 1:6, 1:6] = True
        distance = boundary_distances_mm(mask, (2.0, 3.0, 4.0))
        self.assertAlmostEqual(float(distance[3, 3, 3]), 6.0)

    def test_ct_window_is_float32_and_bounded(self):
        volume = np.array([[[-300.0, -160.0, 40.0, 240.0, 500.0]]])
        result = window_ct(volume, center_hu=40.0, width_hu=400.0)
        np.testing.assert_allclose(result, [[[0.0, 0.0, 0.5, 1.0, 1.0]]])
        self.assertEqual(result.dtype, np.float32)


class InteriorKeypointSelectionTests(unittest.TestCase):
    def test_radius_suppression_is_physical_and_score_ordered(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]], dtype=float)
        scores = np.array([0.9, 1.0, 0.8])
        retained = greedy_radius_suppression(points, scores, (2.0, 1.0, 1.0), 3.0)
        np.testing.assert_array_equal(retained, [1, 2])

    def test_fps_is_deterministic_and_spatially_distributed(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
        scores = np.array([0.2, 1.0, 0.3, 0.4])
        selected = farthest_point_sample(points, scores, (1.0, 1.0, 1.0), 3)
        np.testing.assert_array_equal(selected, [1, 3, 2])

    def test_fps_refuses_silent_replacement_when_supply_is_low(self):
        with self.assertRaisesRegex(ValueError, "cannot select 3"):
            farthest_point_sample(
                np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
                np.array([1.0, 0.5]),
                (1.0, 1.0, 1.0),
                3,
            )

    def test_csv_writer_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "points.csv"
            row = {field: "" for field in CSV_FIELDS}
            write_csv_atomic(destination, [row], CSV_FIELDS)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_csv_atomic(destination, [row], CSV_FIELDS)


if __name__ == "__main__":
    unittest.main()
