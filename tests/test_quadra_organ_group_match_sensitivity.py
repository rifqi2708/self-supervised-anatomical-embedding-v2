import argparse
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.quadra import memory_configuration_screen as stage3
from tools.quadra import organ_group_match_sensitivity as stage5
from tools.quadra import organ_group_workflow_decision as stage4c


def identity(path):
    return stage5.file_identity(path)


def simple_plan(session, group, margin):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    return {
        "subject_id": "quadra_hc_030",
        "session": session,
        "scan_key": "quadra_hc_030-{}".format(session),
        "group_name": group,
        "margin_mm": margin,
        "padded_shape_xyz": [64, 64, 32],
        "padded_2mm_voxels": 64 * 64 * 32,
        "raw_to_model_continuous_affine": np.eye(4).tolist(),
        "model_to_raw_continuous_affine": np.eye(4).tolist(),
        "source_ct": {"affine": affine.tolist()},
    }


class Stage4CDecisionTests(unittest.TestCase):
    def test_acknowledgement_is_mandatory(self):
        args = argparse.Namespace(
            accept_known_context_sensitivity=False,
            review_rationale=stage4c.FROZEN_RATIONALE,
        )
        with self.assertRaisesRegex(stage4c.Stage4CError, "acknowledgement"):
            stage4c.run_accept(args)

    def test_stage4c_writes_provisional_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage4a_checkpoint = root / "stage4a.json"
            stage4b_checkpoint = root / "stage4b.json"
            stage4a_checkpoint.write_text("{}\n")
            stage4b_checkpoint.write_text("{}\n")
            stage4a = {"checkpoint": identity(stage4a_checkpoint)}
            stage4b = {"checkpoint": identity(stage4b_checkpoint)}
            args = argparse.Namespace(
                accept_known_context_sensitivity=True,
                review_rationale=stage4c.FROZEN_RATIONALE,
                repository_root=str(root),
                stage4a_checkpoint=str(stage4a_checkpoint),
                stage4b_checkpoint=str(stage4b_checkpoint),
                output_root=str(root / "outputs"),
                storage_root=str(root),
                run_id="decision",
            )
            with mock.patch.object(stage4c, "validate_repository", return_value={"clean": True}), \
                 mock.patch.object(stage4c, "validate_stage4a", return_value=stage4a), \
                 mock.patch.object(stage4c, "validate_stage4b", return_value=stage4b):
                run_dir = stage4c.run_accept(args)
            decision = json.loads((run_dir / "limitation_acceptance.json").read_text())
            checkpoint = json.loads((run_dir / "checkpoint_summary.json").read_text())
            self.assertEqual(decision["status"], "PROVISIONAL_ACCEPTANCE")
            self.assertEqual(checkpoint["status"], "PROVISIONAL")
            self.assertFalse(checkpoint["gates"]["descriptor_boundary_invariance_established"])
            self.assertFalse(checkpoint["gates"]["production_workflow_frozen"])
            self.assertEqual(identity(stage4a_checkpoint), stage4a["checkpoint"])
            self.assertEqual(identity(stage4b_checkpoint), stage4b["checkpoint"])

    def test_decision_directory_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            stage4c.atomic_json(path, {"status": "first"}, refuse=True)
            with self.assertRaisesRegex(stage4c.Stage4CError, "overwrite"):
                stage4c.atomic_json(path, {"status": "second"}, refuse=True)


class QueryAndGeometryTests(unittest.TestCase):
    def test_affine_and_bounds_preserve_continuous_coordinates(self):
        plan = simple_plan("test", "abdomen", 100)
        raw = np.asarray([12.25, 18.5, 7.75])
        inside, model = stage5.point_inside_model(raw, plan)
        self.assertTrue(inside)
        np.testing.assert_allclose(model, raw)
        np.testing.assert_allclose(stage5.apply_affine(model, plan["model_to_raw_continuous_affine"]), raw)
        self.assertFalse(stage5.point_inside_model([-0.1, 2, 2], plan)[0])

    def test_sentinel_selection_is_deterministic_and_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            point_id = 0
            for session in ("test", "retest"):
                for group in stage3.GROUPS:
                    for xyz in ((2, 2, 2), (31, 31, 15), (50, 40, 20)):
                        point_id += 1
                        samples.append({
                            "point_id": point_id,
                            "scan_key": "quadra_hc_030-{}".format(session),
                            "subject_id": "quadra_hc_030",
                            "session": session,
                            "group_name": group,
                            "mask_name": "{}_mask".format(group),
                            "raw_x": xyz[0], "raw_y": xyz[1], "raw_z": xyz[2],
                            "coord_space": "raw_itk_voxel",
                        })
            samples_path = root / "samples.csv"
            stage5.write_csv(samples_path, samples)
            stage4a = {"samples": identity(samples_path)}
            plans = {
                (session, group, margin): ({}, simple_plan(session, group, margin))
                for session in ("test", "retest")
                for group in stage3.GROUPS
                for margin in (100, 120)
            }
            first_dir = root / "first"; first_dir.mkdir()
            second_dir = root / "second"; second_dir.mkdir()
            frozen_first, sentinels_first = stage5.freeze_queries(stage4a, plans, first_dir)
            frozen_second, sentinels_second = stage5.freeze_queries(stage4a, plans, second_dir)
            self.assertEqual(frozen_first, frozen_second)
            self.assertEqual(sentinels_first, sentinels_second)
            self.assertEqual(len(sentinels_first), 8)
            for group in stage3.GROUPS:
                group_rows = [row for row in sentinels_first if row["group_name"] == group]
                self.assertEqual({row["sentinel_role"] for row in group_rows}, {"group_centre", "minimum_100mm_clearance"})
                self.assertEqual(len({row["query_id"] for row in group_rows}), 2)

    def test_model_index_rounding_is_deferred_and_bounds_checked(self):
        plan = simple_plan("test", "abdomen", 100)
        np.testing.assert_array_equal(stage5._raw_to_model_index([1.49, 2.5, 3.51], plan), [1, 2, 4])
        with self.assertRaisesRegex(stage5.Stage5Error, "outside"):
            stage5._raw_to_model_index([100, 2, 2], plan)


class SensitivityGateTests(unittest.TestCase):
    def comparison_rows(self, displacement=1.0, cycle_delta=0.5):
        rows = []
        point = 0
        for group in stage3.GROUPS:
            for _ in range(10):
                point += 1
                rows.append({
                    "query_id": str(point), "point_id": point,
                    "source_session": "test", "target_session": "retest",
                    "group_name": group, "mask_name": "mask",
                    "paired_success": True,
                    "forward_displacement_mm": displacement,
                    "backward_displacement_mm": displacement,
                    "cycle_error_abs_delta_mm": cycle_delta,
                })
        return rows

    def test_existing_project_thresholds_pass_and_fail(self):
        summaries = stage5.summarize_global(self.comparison_rows())
        self.assertEqual(len(summaries), 5)
        self.assertTrue(all(row["passed"] for row in summaries))
        rows = self.comparison_rows()
        for row in rows[:3]:
            row["forward_displacement_mm"] = 20.0
            row["backward_displacement_mm"] = 20.0
        summaries = stage5.summarize_global(rows)
        pooled = next(row for row in summaries if row["scope"] == "ALL_GROUPS")
        self.assertFalse(pooled["passed"])

    def test_comparison_preserves_failed_fixed_point_denominator(self):
        results = []
        for margin, status in ((100, "failed"), (120, "failed")):
            results.append({
                "fixed_records": [{
                    "query_id": "q", "point_id": 1, "sentinel_role": "group_centre",
                    "source_session": "test", "target_session": "retest",
                    "group_name": "abdomen", "mask_name": "liver", "margin_mm": margin,
                    "status": status, "cycle_error_mm": None,
                    "stable_anchor_count_forward": 1,
                }]
            })
        rows = stage5._comparison_rows(results, "fixed_point")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["paired_success"])
        self.assertEqual(rows[0]["status_100mm"], "failed")
        self.assertEqual(rows[0]["status_120mm"], "failed")

    def test_forbidden_full_volume_outputs_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compact.json").write_text("{}")
            self.assertEqual(stage5.forbidden_outputs(root), [])
            (root / "embedding.npy").write_bytes(b"forbidden")
            self.assertEqual(len(stage5.forbidden_outputs(root)), 1)

    def test_frozen_query_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_path = root / "global.csv"
            sentinel_path = root / "sentinel.csv"
            global_path.write_text("a\n1\n")
            sentinel_path.write_text("a\n1\n")
            manifest = {"outputs": {"global_queries": identity(global_path), "fixed_point_sentinels": identity(sentinel_path)}}
            stage5._validate_frozen_outputs(manifest)
            global_path.write_text("a\n2\n")
            with self.assertRaisesRegex(stage5.Stage5Error, "changed"):
                stage5._validate_frozen_outputs(manifest)


class CompatibilityTests(unittest.TestCase):
    def test_new_modules_parse_with_python37_grammar(self):
        for module in (stage4c, stage5):
            source = Path(module.__file__).read_text(encoding="utf-8")
            ast.parse(source, feature_version=(3, 7))

    def test_in_memory_cache_exposes_production_interface(self):
        fine = np.zeros((128, 4, 5, 6), dtype=np.float16)
        coarse = np.zeros((128, 2, 3, 3), dtype=np.float16)
        semantic = np.zeros((128, 4, 5, 6), dtype=np.float16)
        cache = stage5.InMemoryUaesCache(fine, coarse, semantic, (12, 10, 8))
        self.assertEqual(cache.feature_shape_xyz("fine"), (6, 5, 4))
        self.assertEqual(cache.native_shape_xyz, (12, 10, 8))
        self.assertEqual(cache.manifest["native_spacing_xyz"], [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
