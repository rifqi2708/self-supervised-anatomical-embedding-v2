import unittest

import numpy as np

from tools.quadra.streaming_cycle_error import canonical_subject_id
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


if __name__ == "__main__":
    unittest.main()
