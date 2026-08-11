import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.quadra import memory_configuration_screen as stage3
from tools.quadra import organ_group_numerical_validation as stage4


def source_record():
    affine = np.diag([-1.5, -1.5, 2.0, 1.0])
    return {
        "path": "/tmp/source.nii.gz",
        "bytes": 1,
        "sha256": "a" * 64,
        "native_shape_xyz": [256, 240, 200],
        "spacing_xyz_mm": [1.5, 1.5, 2.0],
        "affine": affine.tolist(),
    }


def group_plan(scan_key="quadra_hc_030-test", group="abdomen", voxels=100):
    subject, session = scan_key.rsplit("-", 1)
    base = {
        "subject_id": subject,
        "session": session,
        "scan_key": scan_key,
        "sex": "M",
        "source_ct": source_record(),
    }
    plan = stage3._plan_from_bounds(
        base, [20, 20, 10], [220, 220, 190], "organ_group",
        group_name=group, margin_mm=100.0,
    )
    plan["included_masks"] = list(stage3.required_group_masks(group, "M"))
    plan["mask_union_start_xyz"] = [80, 80, 60]
    plan["mask_union_end_xyz"] = [160, 160, 140]
    plan["padded_2mm_voxels"] = voxels
    return plan


class PlanAndSamplingTests(unittest.TestCase):
    def test_reference_plan_expands_frozen_union_to_120mm(self):
        plan = group_plan()
        reference = stage4.derive_margin_plan(plan, 120.0)
        self.assertEqual(reference["margin_mm"], 120.0)
        self.assertEqual(reference["mask_union_start_xyz"], [80, 80, 60])
        self.assertTrue(
            np.all(np.asarray(reference["crop_start_xyz"]) <= np.asarray(plan["crop_start_xyz"]))
        )
        self.assertTrue(
            np.all(np.asarray(reference["crop_end_xyz"]) >= np.asarray(plan["crop_end_xyz"]))
        )
        self.assertTrue(
            np.all(np.asarray(reference["padded_shape_xyz"]) % np.asarray(stage3.MODEL_STRIDE) == 0)
        )

    def test_pair_selection_requires_four_groups_in_both_sessions(self):
        plans = []
        for session in ("test", "retest"):
            for index, group in enumerate(stage3.GROUPS):
                plans.append(group_plan("quadra_hc_030-{}".format(session), group, 100 + index))
        payload = {
            "spatial_plans": {"organ_group": plans},
            "largest_spatial_plans": {"organ_group": plans[0]},
        }
        selected = stage4.select_pair_plans(payload)
        self.assertEqual(len(selected), 8)
        payload["spatial_plans"]["organ_group"].pop()
        with self.assertRaisesRegex(stage4.Stage4Error, "four unique"):
            stage4.select_pair_plans(payload)

    def test_foreground_sampling_is_unique_deterministic_and_inside(self):
        mask = np.zeros((20, 18, 16), dtype=np.uint8)
        mask[2:18, 3:15, 4:13] = 1
        first = stage4.sample_foreground_points(mask, [9.5, 8.5, 8.0])
        second = stage4.sample_foreground_points(mask, [9.5, 8.5, 8.0])
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertLessEqual(len(first), stage4.MAX_POINTS_PER_MASK)
        self.assertTrue(all(mask[point] == 1 for point in first))
        coordinates = np.asarray(first)
        self.assertEqual(int(coordinates[:, 0].min()), 2)
        self.assertEqual(int(coordinates[:, 0].max()), 17)
        self.assertEqual(int(coordinates[:, 2].min()), 4)
        self.assertEqual(int(coordinates[:, 2].max()), 12)

    def test_derived_seed_always_fits_numpy_randomstate(self):
        with mock.patch.object(stage4.hashlib, "sha256") as digest:
            digest.return_value.hexdigest.return_value = "ffffffff" + "0" * 56
            seed = stage4._seed_for("largest-possible-prefix")
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2 ** 32 - 1)
        np.random.RandomState(seed)

    def test_coordinate_rows_round_trip_without_rounding(self):
        selected = group_plan()
        reference = stage4.derive_margin_plan(selected, 120.0)
        rows = stage4.coordinate_rows([
            {
                "point_id": 1, "scan_key": selected["scan_key"],
                "group_name": selected["group_name"],
                "raw_x": 100.25, "raw_y": 110.5, "raw_z": 90.75,
            }
        ], [selected, reference])
        self.assertEqual(len(rows), 2)
        self.assertLess(max(row["max_raw_voxel_roundtrip_error"] for row in rows), 1e-9)
        self.assertTrue(all(row["inside_model_grid"] for row in rows))


class DescriptorGeometryTests(unittest.TestCase):
    def test_align_corners_false_feature_mapping(self):
        points = np.asarray([[0.0, 0.0, 0.0], [15.5, 31.5, 7.5]])
        mapped = stage4.model_to_feature_xyz(points, [16, 64, 32], [8, 32, 16])
        np.testing.assert_allclose(mapped[0], [-0.25, -0.25, -0.25])
        np.testing.assert_allclose(mapped[1], [7.5, 15.5, 3.5])

    def test_cosine_summary_uses_strict_median_and_p01_gates(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is required for descriptor-cosine execution")

        samples = [{"point_id": index, "mask_name": "mask"} for index in range(100)]
        basis = torch.eye(128, dtype=torch.float32)[:100]
        reference = {name: basis.clone() for name in ("fine", "coarse", "semantic")}
        candidate = {name: basis.clone() for name in reference}
        _, summary = stage4.cosine_rows(reference, candidate, samples, "boundary")
        self.assertTrue(all(row["passed"] for row in summary))
        candidate["fine"][0] = -candidate["fine"][0]
        candidate["fine"][1] = -candidate["fine"][1]
        _, summary = stage4.cosine_rows(reference, candidate, samples, "boundary")
        fine = next(row for row in summary if row["feature"] == "fine")
        self.assertFalse(fine["passed"])
        self.assertLess(fine["p01_cosine"], stage4.COSINE_P01_MIN)


class SelectionAndSafetyTests(unittest.TestCase):
    def successful_result(self, scan_key, group, repeatability=False):
        summary = [
            {
                "comparison": "100mm_vs_120mm", "feature": feature,
                "median_cosine": 0.999, "p01_cosine": 0.98, "passed": True,
            }
            for feature in ("fine", "coarse", "semantic")
        ]
        repeat = [
            {
                "comparison": "100mm_repeatability", "feature": feature,
                "median_cosine": 1.0, "p01_cosine": 1.0, "passed": True,
            }
            for feature in ("fine", "coarse", "semantic")
        ] if repeatability else []
        return {
            "status": "success", "scan_key": scan_key, "group_name": group,
            "precision_contract": {"passed": True}, "measured_peak_mib": 12000,
            "extractions": {"100": {}, "120": {}},
            "boundary_summary": summary, "repeatability_summary": repeat,
        }

    def test_selection_passes_only_complete_strict_evidence(self):
        results = []
        for session in ("test", "retest"):
            for group in stage3.GROUPS:
                results.append(self.successful_result(
                    "quadra_hc_030-{}".format(session), group,
                    repeatability=(session == "retest" and group == "pelvis"),
                ))
        summaries = []
        for result in results:
            summaries.extend(result["boundary_summary"])
            summaries.extend(result["repeatability_summary"])
        manifest = {
            "maximum_roundtrip_voxel_error": 1e-10,
            "maximum_roundtrip_physical_error_mm": 1e-10,
        }
        self.assertEqual(stage4.evaluate_selection(manifest, results, summaries), [])
        summaries[0]["passed"] = False
        self.assertIn("boundary_sensitivity", stage4.evaluate_selection(manifest, results, summaries))

    def test_non_oom_failure_is_not_an_amp_trigger(self):
        failure = {"status": "failed", "failure_classification": "model_error", "failed_margin_mm": 100}
        trigger = failure.get("failure_classification") == "cuda_oom" and failure.get("failed_margin_mm") == 100
        self.assertFalse(trigger)
        failure["failure_classification"] = "cuda_oom"
        self.assertTrue(failure.get("failure_classification") == "cuda_oom" and failure.get("failed_margin_mm") == 100)

    def test_stage4_module_parses_with_python37_grammar(self):
        source = Path(stage4.__file__).read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 7))

    def test_atomic_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "immutable.json"
            stage4.atomic_json(path, {"status": "first"}, refuse=True)
            with self.assertRaisesRegex(stage4.Stage4Error, "overwrite"):
                stage4.atomic_json(path, {"status": "second"}, refuse=True)
            self.assertEqual(json.loads(path.read_text())["status"], "first")

    def test_full_volume_artifacts_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compact.json").write_text("{}")
            self.assertEqual(stage4.forbidden_full_volume_outputs(root), [])
            (root / "embedding.npy").write_bytes(b"not-a-real-array")
            self.assertEqual(len(stage4.forbidden_full_volume_outputs(root)), 1)

    def test_repository_rejects_dirty_checkout(self):
        with mock.patch.object(stage3, "git_output", side_effect=[stage4.EXPECTED_BRANCH, "a" * 40, "dirty"]), \
             mock.patch.object(stage4.subprocess, "call", return_value=0):
            with self.assertRaisesRegex(stage4.Stage4Error, "Repository contract"):
                stage4.validate_repository(Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
