import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.quadra.streaming_cycle_error import (
    DEFAULT_HALO_XYZ,
    DEFAULT_TILE_SIZE_XYZ,
    CLEANUP_FAILURE_EXIT_CODE,
    EmbeddingCache,
    CacheCleanupError,
    build_sam_cycle_results,
    canonical_subject_id,
    delete_subject_cache_safely,
    main,
    parse_args,
    validate_completed_outputs,
    write_query_points_raw_itk_csv,
    write_json,
)
from tools.quadra.rd_cycle_error_helper import (
    compute_summary_stats,
    write_points_csv_with_mask,
    write_summary_with_mask_labels_csv,
)
from tools.quadra.streaming_embedding import (
    RECOMMENDED_CORE_SIZE_XYZ,
    RECOMMENDED_HALO_XYZ,
    RECOMMENDED_TILE_SIZE_XYZ,
    align_corners_false_source_positions,
    build_tile_plan,
    embedding_geometry_namespace,
    flattened_zyx_index,
    iter_chunks_xyz,
    iter_tile_locations,
    source_bounds_for_output_interval,
)


class QuadraStreamingGeometryTests(unittest.TestCase):
    def test_quadra_021_baseline_tile_plan(self):
        plan = build_tile_plan((390, 390, 301), (128, 128, 64), (32, 32, 16))
        self.assertEqual(plan.core_size_xyz, (64, 64, 32))
        self.assertEqual(plan.grid_shape_xyz, (7, 7, 10))
        self.assertEqual(plan.tile_count, 490)
        self.assertEqual(plan.valid_fine_shape_xyz, (195, 195, 151))
        self.assertEqual(plan.valid_coarse_shape_xyz, (25, 25, 76))

    def test_validated_expanded_geometry_is_the_production_default(self):
        plan = build_tile_plan((390, 390, 301))
        args = parse_args([])

        self.assertEqual(RECOMMENDED_TILE_SIZE_XYZ, (160, 160, 80))
        self.assertEqual(RECOMMENDED_HALO_XYZ, (48, 48, 24))
        self.assertEqual(RECOMMENDED_CORE_SIZE_XYZ, (64, 64, 32))
        self.assertEqual(DEFAULT_TILE_SIZE_XYZ, RECOMMENDED_TILE_SIZE_XYZ)
        self.assertEqual(DEFAULT_HALO_XYZ, RECOMMENDED_HALO_XYZ)
        self.assertEqual(args.tile_size, RECOMMENDED_TILE_SIZE_XYZ)
        self.assertEqual(args.halo, RECOMMENDED_HALO_XYZ)
        self.assertEqual(plan.tile_size_xyz, RECOMMENDED_TILE_SIZE_XYZ)
        self.assertEqual(plan.halo_xyz, RECOMMENDED_HALO_XYZ)
        self.assertEqual(plan.core_size_xyz, RECOMMENDED_CORE_SIZE_XYZ)
        self.assertEqual(plan.tile_count, 490)
        self.assertFalse(args.keep_cache)

    def test_keep_cache_is_explicit_opt_out(self):
        self.assertTrue(parse_args(["--keep-cache"]).keep_cache)

    def test_embedding_cache_namespace_changes_with_geometry(self):
        expanded = embedding_geometry_namespace()
        baseline = embedding_geometry_namespace((128, 128, 64), (32, 32, 16))

        self.assertEqual(expanded, "tile160x160x80_halo48x48x24_core64x64x32")
        self.assertEqual(baseline, "tile128x128x64_halo32x32x16_core64x64x32")
        self.assertNotEqual(expanded, baseline)

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


class QuadraCacheCleanupTests(unittest.TestCase):
    def _make_cache_tree(self, root: Path, subject_id: str = "quadra_hc_021"):
        subject_root = root / "namespace" / subject_id
        manifests = []
        for timepoint in ("test", "retest"):
            cache_dir = subject_root / timepoint
            cache_dir.mkdir(parents=True)
            (cache_dir / "fine.npy").write_bytes(b"fine")
            (cache_dir / "coarse.npy").write_bytes(b"coarse")
            manifest = {
                "cache_dir": str(cache_dir.resolve()),
                "source_image": {"path": f"/dataset/{subject_id}/{timepoint}.nii.gz"},
            }
            write_json(cache_dir / "manifest.json", {"complete": True, **manifest})
            manifests.append(manifest)
        return subject_root, manifests

    def test_safe_deletion_removes_only_requested_subject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            subject_root, manifests = self._make_cache_tree(root)
            sibling_root, _ = self._make_cache_tree(root, "quadra_hc_022")

            result = delete_subject_cache_safely(root, subject_root, "quadra_hc_021", manifests)

            self.assertFalse(subject_root.exists())
            self.assertTrue(sibling_root.exists())
            self.assertTrue((root / "namespace").is_dir())
            self.assertGreater(result["bytes_freed"], 0)

    def test_safe_deletion_rejects_mismatched_subject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            subject_root, manifests = self._make_cache_tree(root)
            with self.assertRaises(CacheCleanupError):
                delete_subject_cache_safely(root, subject_root, "quadra_hc_022", manifests)
            self.assertTrue(subject_root.exists())

    def test_safe_deletion_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "cache"
            outside, manifests = self._make_cache_tree(temp / "outside")
            (root / "namespace").mkdir(parents=True)
            link = root / "namespace" / "quadra_hc_021"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(CacheCleanupError):
                delete_subject_cache_safely(root, link, "quadra_hc_021", manifests)
            self.assertTrue(outside.exists())

    def test_embedding_cache_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            np.save(cache_dir / "fine.npy", np.zeros((2, 2, 2, 2), dtype=np.float16))
            np.save(cache_dir / "coarse.npy", np.zeros((2, 1, 1, 1), dtype=np.float16))
            write_json(
                cache_dir / "manifest.json",
                {
                    "complete": True,
                    "native_sam_shape_xyz": [4, 4, 4],
                    "norm_ratio_xyz": [1, 1, 1],
                    "features": {
                        "fine": {"file": "fine.npy", "valid_shape_xyz": [2, 2, 2]},
                        "coarse": {"file": "coarse.npy", "valid_shape_xyz": [1, 1, 1]},
                    },
                },
            )
            cache = EmbeddingCache(cache_dir)
            cache.close()
            cache.close()
            self.assertTrue(cache._closed)
            with self.assertRaises(RuntimeError):
                cache.valid_array("fine")

    def test_cleanup_error_uses_distinct_nonzero_status(self):
        with mock.patch("tools.quadra.streaming_cycle_error.run", side_effect=CacheCleanupError("blocked")):
            self.assertEqual(main([]), CLEANUP_FAILURE_EXIT_CODE)


class QuadraOutputValidationTests(unittest.TestCase):
    def _write_valid_run(self, output_dir: Path):
        subject_id = "quadra_hc_021"
        masks = [f"{subject_id}/bladder.nii.gz", f"{subject_id}/colon.nii.gz"]
        results = []
        sam_results = []
        for idx, mask_name in enumerate(masks):
            base = {
                "subject_id": subject_id,
                "mask_name": mask_name,
                "pt1": np.array([idx + 1, 2, 3]),
                "pt2": np.array([4, 5, 6]),
                "pt1_back": np.array([idx + 1, 2, 3]),
                "voxel_error": 0.0,
                "mm_error": 0.0,
                "score_12": 0.9,
                "score_21": 0.8,
            }
            results.append({**base, "coord_space": "raw_itk_voxel"})
            sam_results.append({**base, "coord_space": "sam_display_voxel"})
        points = output_dir / "cycle_points.csv"
        sam = output_dir / "cycle_points_sam.csv"
        query = output_dir / "query_points_raw_itk.csv"
        summary = output_dir / "cycle_summary.csv"
        write_points_csv_with_mask(results, points)
        write_points_csv_with_mask(sam_results, sam)
        write_query_points_raw_itk_csv(results, query)
        per_mask = []
        for result in results:
            voxel, mm = compute_summary_stats([result])
            per_mask.append({"mask_name": result["mask_name"], "voxel_stats": voxel, "mm_stats": mm})
        voxel, mm = compute_summary_stats(results)
        write_summary_with_mask_labels_csv(per_mask, summary, voxel, mm)
        cache_manifest = {
            "complete": True,
            "cache_dir": "/cache/namespace/quadra_hc_021/test",
            "source_image": {"path": "/dataset/quadra_hc_021/test.nii.gz", "sha256": "image"},
            "config": {"path": "/config.py", "sha256": "a"},
            "checkpoint": {"path": "/SAM.pth", "sha256": "b"},
            "norm_spacing_xyz": [2.0, 2.0, 2.0],
            "tile_plan": {
                "tile_size_xyz": [160, 160, 80],
                "halo_xyz": [48, 48, 24],
                "core_size_xyz": [64, 64, 32],
            },
        }
        write_json(
            output_dir / "run_manifest.json",
            {
                "schema_version": 3,
                "completed": True,
                "subject_id": subject_id,
                "config": {"path": "/config.py", "sha256": "a"},
                "checkpoint": {"path": "/SAM.pth", "sha256": "b"},
                "norm_spacing_xyz": [2.0, 2.0, 2.0],
                "tile_size_xyz": [160, 160, 80],
                "halo_xyz": [48, 48, 24],
                "retained_core_size_xyz": [64, 64, 32],
                "cache_policy": "delete_on_success",
                "cache_cleanup": {
                    "status": "scheduled",
                    "subject_cache_root": "/cache/namespace/quadra_hc_021",
                    "deleted_paths": [],
                    "bytes_before_cleanup": None,
                    "bytes_freed": 0,
                    "completed_at": None,
                    "error": None,
                },
                "test_cache_manifest": cache_manifest,
                "retest_cache_manifest": {**cache_manifest, "cache_dir": "/cache/namespace/quadra_hc_021/retest"},
                "outputs": {
                    "points_raw_itk": str(points),
                    "points_sam": str(sam),
                    "query_points_for_registration": str(query),
                    "summary": str(summary),
                },
            },
        )
        return masks

    def test_completed_outputs_validate_before_cleanup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            masks = self._write_valid_run(output_dir)
            result = validate_completed_outputs(output_dir, 2, "quadra_hc_021", masks)
            self.assertEqual(result["point_count"], 2)

    def test_incomplete_query_output_blocks_cleanup_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            masks = self._write_valid_run(output_dir)
            query = output_dir / "query_points_raw_itk.csv"
            with query.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
            query.write_text("".join(lines[:-1]), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_completed_outputs(output_dir, 2, "quadra_hc_021", masks)


if __name__ == "__main__":
    unittest.main()
