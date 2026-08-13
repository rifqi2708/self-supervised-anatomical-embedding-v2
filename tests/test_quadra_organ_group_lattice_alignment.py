import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

from tools.quadra import organ_group_lattice_alignment as stage5r
from tools.quadra import coordinate_preserving_crop as crop


def oblique_affine(spacing=(1.5, 2.5, 3.0), angle_degrees=17.0):
    angle = np.deg2rad(angle_degrees)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = rotation @ np.diag(spacing)
    affine[:3, 3] = [10.0, -30.0, 7.0]
    return affine


def old_plan(session="test", group="abdomen", margin=100):
    affine = oblique_affine()
    return {
        "scan_key": "quadra_hc_030-{}".format(session),
        "subject_id": "quadra_hc_030",
        "session": session,
        "group_name": group,
        "margin_mm": margin,
        "source_ct": {
            "path": "/tmp/not-read.nii.gz",
            "bytes": 1,
            "sha256": "0" * 64,
            "native_shape_xyz": [101, 83, 57],
            "affine": affine.tolist(),
        },
        "included_masks": ["liver"],
        "mask_union_start_xyz": [2, 10, 5],
        "mask_union_end_xyz": [70, 60, 50],
    }


class GlobalLatticeGeometryTests(unittest.TestCase):
    def test_outward_snap_supports_negative_coordinates(self):
        start, stop = stage5r.outward_stride_snap(
            [-3.2, 15.9, -0.1], [20.1, 33.0, 7.9]
        )
        np.testing.assert_array_equal(start, [-16, 0, -4])
        np.testing.assert_array_equal(stop, [32, 48, 8])
        np.testing.assert_array_equal((stop - start) % [16, 16, 4], [0, 0, 0])

    def test_aligned_100_is_contained_and_phase_matched(self):
        source = old_plan()
        plan100 = stage5r.aligned_plan_from_union(source, 100)
        plan120 = stage5r.aligned_plan_from_union(source, 120)
        self.assertTrue(stage5r.assert_pair_alignment(plan100, plan120))
        self.assertNotIn("crop_start_xyz", plan100)
        self.assertNotIn("resampled_2mm_affine", plan100)
        self.assertEqual(plan100["target_shape_xyz"], plan100["padded_shape_xyz"])
        self.assertEqual(plan100["padding_lower_xyz"], [0, 0, 0])
        start100 = np.asarray(plan100["global_grid_start_xyz"])
        start120 = np.asarray(plan120["global_grid_start_xyz"])
        np.testing.assert_array_equal((start100 - start120) % [16, 16, 4], [0, 0, 0])
        self.assertTrue(np.all(np.asarray(plan120["global_grid_start_xyz"]) <= start100))
        self.assertTrue(np.all(np.asarray(plan120["global_grid_stop_xyz"]) >= plan100["global_grid_stop_xyz"]))

    def test_fov_extension_and_valid_box_are_explicit(self):
        plan = stage5r.aligned_plan_from_union(old_plan(), 120)
        start = np.asarray(plan["global_grid_start_xyz"])
        stop = np.asarray(plan["global_grid_stop_xyz"])
        global_shape = np.asarray(plan["global_lattice_shape_xyz"])
        np.testing.assert_array_equal(plan["fov_extension_lower_xyz"], np.maximum(-start, 0))
        np.testing.assert_array_equal(plan["fov_extension_upper_xyz"], np.maximum(stop - global_shape, 0))
        valid = np.asarray(plan["valid_model_box_xyz"])
        shape = np.asarray(plan["padded_shape_xyz"])
        self.assertTrue(np.all(valid[0] >= 0))
        self.assertTrue(np.all(valid[1] <= shape))

    def test_raw_model_roundtrip_is_subvoxel_accurate(self):
        plan = stage5r.aligned_plan_from_union(old_plan(), 100)
        points = [[2.25, 10.5, 5.75], [69.5, 59.25, 49.0]]
        self.assertLessEqual(stage5r.validate_raw_model_roundtrip(plan, points), 1e-10)

    def test_shared_admissible_box_maps_to_both_local_grids(self):
        source = old_plan()
        plan100 = stage5r.aligned_plan_from_union(source, 100)
        plan120 = stage5r.aligned_plan_from_union(source, 120)
        box100 = np.asarray(stage5r.admissible_box_for_target(plan100, plan100))
        box120 = np.asarray(stage5r.admissible_box_for_target(plan100, plan120))
        global100 = box100 + np.asarray(plan100["global_grid_start_xyz"])
        global120 = box120 + np.asarray(plan120["global_grid_start_xyz"])
        np.testing.assert_array_equal(global100, global120)
        np.testing.assert_array_equal(box100, plan100["valid_model_box_xyz"])

    def test_real_preparation_fills_extended_fov_and_normalizes_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (21, 19, 13)
            affine = oblique_affine()
            data = np.full(shape, 100.0, dtype=np.float32)
            path = root / "test_CT-AC.nii.gz"
            image = nib.Nifti1Image(data, affine)
            image.header.set_xyzt_units("mm")
            nib.save(image, str(path))
            loaded = nib.load(str(path))
            plan = old_plan()
            plan["source_ct"] = {
                **crop.file_identity(path),
                "native_shape_xyz": list(shape),
                "affine": np.asarray(loaded.affine, dtype=float).tolist(),
            }
            plan["mask_union_start_xyz"] = [5, 5, 3]
            plan["mask_union_end_xyz"] = [16, 14, 10]
            aligned = stage5r.aligned_plan_from_union(plan, 100)
            prepared = crop.prepare_scan_on_global_lattice(path, aligned)
            self.assertEqual(prepared.data_zyx.dtype, np.float32)
            self.assertEqual(prepared.data_zyx.shape, tuple(aligned["model_tensor_shape_zyx"]))
            self.assertGreater(prepared.metadata["outside_fov_value_count_with_overlap"], 0)
            self.assertEqual(prepared.metadata["hu_outside_fov_padding_max_error"], 0.0)
            self.assertEqual(prepared.metadata["normalized_outside_fov_padding_max_error"], 0.0)
            valid = np.asarray(aligned["valid_model_box_xyz"], dtype=int)[:, ::-1]
            if valid[0, 0] > 0:
                self.assertTrue(np.all(prepared.data_zyx[: valid[0, 0]] == -50.0))


class FactorialAndGateTests(unittest.TestCase):
    def test_eight_worker_split_covers_each_factorial_cell_once(self):
        self.assertEqual(stage5r.configurations_for_source_margin(100), ("A", "C"))
        self.assertEqual(stage5r.configurations_for_source_margin(120), ("B", "D"))
        self.assertEqual(
            set(stage5r.configurations_for_source_margin(100))
            | set(stage5r.configurations_for_source_margin(120)),
            {"A", "B", "C", "D"},
        )
        with self.assertRaises(stage5r.Stage5RError):
            stage5r.configurations_for_source_margin(110)

    def _results(self, displacement=1.0, cycle_delta=0.5):
        results = []
        point = 0
        for group in stage5r.stage3.GROUPS:
            records = []
            for _ in range(10):
                point += 1
                for configuration in ("A", "B", "C", "D"):
                    value = 0.0 if configuration == "A" else displacement
                    cycle = 1.0 if configuration == "A" else 1.0 + cycle_delta
                    records.append(
                        {
                            "query_id": str(point), "point_id": point,
                            "configuration": configuration,
                            "source_session": "test", "target_session": "retest",
                            "group_name": group, "mask_name": "mask", "status": "success",
                            "matched_physical_xyz": [value, 0, 0],
                            "returned_physical_xyz": [value, 0, 0],
                            "cycle_error_mm": cycle,
                            "score_forward": 1.0,
                            "matched_inside_shared_domain": True,
                            "returned_inside_shared_domain": True,
                        }
                    )
            results.append({"group_name": group, "factorial_records": records})
        return results

    def test_factorial_contrasts_pair_against_A(self):
        results = self._results()
        for contrast in stage5r.CONTRASTS:
            rows = stage5r.comparison_rows(results, contrast)
            self.assertEqual(len(rows), 40)
            self.assertTrue(all(row["paired_success"] for row in rows))
            self.assertTrue(all(row["contrast"] == contrast for row in rows))

    def test_within_two_mm_rate_is_required_per_group(self):
        rows = stage5r.comparison_rows(self._results(), "A_vs_B")
        summaries = stage5r.summarize_contrast(rows)
        self.assertTrue(all(row["passed"] for row in summaries))
        abdomen = [row for row in rows if row["group_name"] == "abdomen"]
        for row in abdomen:
            row["forward_displacement_mm"] = 3.0
            row["backward_displacement_mm"] = 3.0
        summaries = stage5r.summarize_contrast(rows)
        abdomen_summary = next(row for row in summaries if row["scope"] == "abdomen")
        self.assertFalse(abdomen_summary["passed"])

    def test_forbidden_full_volume_outputs_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compact.json").write_text("{}")
            self.assertEqual(stage5r.forbidden_outputs(root), [])
            (root / "features.npy").write_bytes(b"forbidden")
            self.assertEqual(len(stage5r.forbidden_outputs(root)), 1)


class CompatibilityTests(unittest.TestCase):
    def test_module_parses_with_python37_grammar(self):
        source = Path(stage5r.__file__).read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 7))

    def test_select_is_profile_neutral(self):
        source = Path(stage5r.__file__).read_text(encoding="utf-8")
        start = source.index("def run_select")
        stop = source.index("def build_parser")
        selection_source = source[start:stop]
        self.assertIn("profile=None", selection_source)
        self.assertNotIn('read_profile_fingerprint', selection_source)

    def test_run_directory_refuses_overwrite_and_accepts_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            created, resumed = stage5r._run_directory(root, run_id="stage5r-lattice-alignment-test")
            self.assertFalse(resumed)
            with self.assertRaises(stage5r.Stage5RError):
                stage5r._run_directory(root, run_id="stage5r-lattice-alignment-test")
            observed, resumed = stage5r._run_directory(root, resume=created)
            self.assertTrue(resumed)
            self.assertEqual(observed.resolve(), created.resolve())

    def test_nonblocked_stage5_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_summary.json"
            path.write_text('{"stage": 5, "status": "PASS"}\n')
            with self.assertRaisesRegex(stage5r.Stage5RError, "BLOCKED"):
                stage5r.validate_stage5_blocked_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
