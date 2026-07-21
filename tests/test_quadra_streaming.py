import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.quadra.streaming_cycle_error import (
    build_sam_cycle_results,
    canonical_subject_id,
    write_query_points_raw_itk_csv,
)
from tools.quadra.streaming_embedding import (
    align_corners_false_source_positions,
    build_tile_plan,
    flattened_zyx_index,
    iter_chunks_xyz,
    iter_tile_locations,
    source_bounds_for_output_interval,
)


class QuadraStreamingGeometryTests(unittest.TestCase):
    def test_quadra_021_default_tile_plan(self):
        plan = build_tile_plan((390, 390, 301), (128, 128, 64), (32, 32, 16))
        self.assertEqual(plan.core_size_xyz, (64, 64, 32))
        self.assertEqual(plan.grid_shape_xyz, (7, 7, 10))
        self.assertEqual(plan.tile_count, 490)
        self.assertEqual(plan.valid_fine_shape_xyz, (195, 195, 151))
        self.assertEqual(plan.valid_coarse_shape_xyz, (25, 25, 76))

    def test_tile_destinations_cover_stored_features_once(self):
        plan = build_tile_plan((130, 129, 65), (128, 128, 64), (32, 32, 16))
        fine_shape_zyx = tuple(reversed(plan.stored_fine_shape_xyz))
        coarse_shape_zyx = tuple(reversed(plan.stored_coarse_shape_xyz))
        fine_coverage = np.zeros(fine_shape_zyx, dtype=np.uint8)
        coarse_coverage = np.zeros(coarse_shape_zyx, dtype=np.uint8)
        for location in iter_tile_locations(plan):
            fine_coverage[location.fine_destination_slices_zyx] += 1
            coarse_coverage[location.coarse_destination_slices_zyx] += 1
        self.assertTrue(np.all(fine_coverage == 1))
        self.assertTrue(np.all(coarse_coverage == 1))

    def test_native_chunks_cover_odd_shape_once(self):
        shape = (512, 512, 301)
        coverage = np.zeros((shape[2], shape[1], shape[0]), dtype=np.uint8)
        for x_range, y_range, z_range in iter_chunks_xyz(shape, (64, 64, 32)):
            coverage[
                z_range[0] : z_range[1],
                y_range[0] : y_range[1],
                x_range[0] : x_range[1],
            ] += 1
        self.assertTrue(np.all(coverage == 1))

    def test_interpolation_bounds_include_all_required_neighbors(self):
        positions = align_corners_false_source_positions(64, 128, 195, 512)
        source_start, source_stop = source_bounds_for_output_interval(64, 128, 195, 512)
        self.assertLessEqual(source_start, int(np.floor(positions.min())))
        self.assertGreaterEqual(source_stop - 1, int(np.ceil(positions.max())))

    def test_subject_alias_is_zero_padded(self):
        self.assertEqual(canonical_subject_id("quadra_hc_21"), "quadra_hc_021")
        self.assertEqual(canonical_subject_id("quadra_hc_021"), "quadra_hc_021")

    def test_flattened_index_uses_zyx_iteration_order(self):
        self.assertEqual(flattened_zyx_index((0, 0, 0), (4, 3, 2)), 0)
        self.assertEqual(flattened_zyx_index((3, 2, 1), (4, 3, 2)), 23)

    def test_sam_cycle_export_uses_internal_coordinates(self):
        internal = [
            {
                "subject_id": "quadra_hc_021",
                "mask_name": "quadra_hc_021/colon.nii.gz",
                "pt1_sam": np.array([10, 20, 30]),
                "pt2_sam": np.array([40, 50, 60]),
                "pt1_back_sam": np.array([13, 24, 30]),
                "score_12": 0.8,
                "score_21": 0.7,
            }
        ]
        raw_itk = [{"mm_error": 6.5}]

        result = build_sam_cycle_results(internal, raw_itk)[0]

        self.assertEqual(result["coord_space"], "sam_display_voxel")
        np.testing.assert_array_equal(result["pt1"], [10, 20, 30])
        np.testing.assert_array_equal(result["pt2"], [40, 50, 60])
        np.testing.assert_array_equal(result["pt1_back"], [13, 24, 30])
        self.assertEqual(result["voxel_error"], 5.0)
        self.assertEqual(result["mm_error"], 6.5)

    def test_registration_query_export_is_minimal_raw_itk_schema(self):
        results = [
            {
                "subject_id": "quadra_hc_021",
                "mask_name": "quadra_hc_021/colon.nii.gz",
                "coord_space": "raw_itk_voxel",
                "pt1": np.array([1, 2, 3]),
                "pt2": np.array([4, 5, 6]),
                "pt1_back": np.array([7, 8, 9]),
                "mm_error": 10.0,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "query_points_raw_itk.csv"
            write_query_points_raw_itk_csv(results, output_path)
            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            list(rows[0]),
            ["idx", "mask_name", "subject_id", "pt1_x", "pt1_y", "pt1_z", "coord_space"],
        )
        self.assertEqual(rows[0]["pt1_x"], "1")
        self.assertNotIn("pt2_x", rows[0])
        self.assertNotIn("mm_error", rows[0])


if __name__ == "__main__":
    unittest.main()
