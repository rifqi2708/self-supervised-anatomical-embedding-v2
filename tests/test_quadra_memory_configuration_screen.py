import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.quadra import body_envelope_audit as stage1
from tools.quadra import memory_configuration_screen as screen


def base_plan():
    affine = np.diag([-1.5, -2.0, 2.5, 1.0])
    return {
        "subject_id": "quadra_hc_044",
        "session": "test",
        "scan_key": "quadra_hc_044-test",
        "source_ct": {
            "path": "/tmp/test.nii.gz",
            "bytes": 1,
            "sha256": "a" * 64,
            "native_shape_xyz": [512, 512, 531],
            "spacing_xyz_mm": [1.5, 2.0, 2.5],
            "affine": affine.tolist(),
        },
        "minimum_artificial_mask_clearance_mm": 30.0,
    }


class SpatialPlanTests(unittest.TestCase):
    def test_whole_body_plan_preserves_bounds_and_stride(self):
        plan = screen._plan_from_bounds(
            base_plan(), [0, 0, 0], [512, 512, 531], "whole_body"
        )
        self.assertEqual(plan["crop_start_xyz"], [0, 0, 0])
        self.assertEqual(plan["crop_end_xyz"], [512, 512, 531])
        self.assertTrue(
            np.all(np.asarray(plan["padded_shape_xyz"]) % np.asarray(screen.MODEL_STRIDE) == 0)
        )
        np.testing.assert_allclose(
            np.asarray(plan["model_to_raw_continuous_affine"])
            @ np.asarray(plan["raw_to_model_continuous_affine"]),
            np.eye(4),
            atol=1e-10,
        )

    def test_organ_union_uses_100_mm_xyz_margin_and_clamps(self):
        start, end = stage1.expand_bounds(
            [20, 100, 5], [500, 300, 520], [512, 512, 531], [1.5, 2.0, 2.5],
            axis_policy="xyz", margin_mm=screen.ORGAN_MARGIN_MM,
        )
        self.assertEqual(start.tolist(), [0, 50, 0])
        self.assertEqual(end.tolist(), [512, 350, 531])

    def test_group_membership_and_sex_specific_prostate(self):
        female = screen.required_group_masks("pelvis", "F")
        male = screen.required_group_masks("pelvis", "M")
        self.assertNotIn("prostate", female)
        self.assertIn("prostate", male)
        self.assertEqual(len(screen.required_group_masks("head_neck", "F")), 12)
        self.assertEqual(len(screen.required_group_masks("thorax", "M")), 13)
        self.assertEqual(len(screen.required_group_masks("abdomen", "F")), 10)

    def test_largest_case_selection_is_by_padded_voxels(self):
        plans = [dict(base_plan(), padded_2mm_voxels=100),
                 dict(base_plan(), scan_key="quadra_hc_045-test", padded_2mm_voxels=200)]
        largest = max(plans, key=lambda p: (p["padded_2mm_voxels"], p["scan_key"]))
        self.assertEqual(largest["scan_key"], "quadra_hc_045-test")


class PrecisionAndControllerTests(unittest.TestCase):
    def test_expected_uae_feature_strides(self):
        expected = screen._expected_feature_shapes([1, 1, 532, 368, 400])
        self.assertEqual(expected["fine"], [1, 128, 266, 184, 200])
        self.assertEqual(expected["coarse"], [1, 128, 133, 23, 25])
        self.assertEqual(expected["semantic"], expected["fine"])

    def test_full_fp16_is_triggered_only_by_amp_cuda_oom(self):
        self.assertTrue(screen.should_run_full_fp16({"failure_classification": "cuda_oom"}))
        for value in ("timeout", "model_error", "memory_ceiling_exceeded", None):
            self.assertFalse(screen.should_run_full_fp16({"failure_classification": value}))

    def test_process_outcome_classification(self):
        self.assertEqual(screen.classify_missing_worker(None, True), "timeout")
        self.assertEqual(screen.classify_missing_worker(-9), "process_kill")
        self.assertEqual(screen.classify_missing_worker(1), "process_crash")

    def test_worker_signature_changes_with_plan_identity(self):
        first = screen.worker_signature("candidate", "amp", {"sha256": "a" * 64})
        second = screen.worker_signature("candidate", "amp", {"sha256": "b" * 64})
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(screen.Stage3Error, "Incompatible"):
            screen.validate_reusable_worker_result(
                {"worker_signature": first}, second, Path("result.json")
            )

    def test_memory_ceiling_is_inclusive(self):
        result = {
            "candidate_id": "body_envelope_amp", "strategy": "body_envelope",
            "precision": "amp", "status": "success", "failure_classification": None,
            "memory": {
                "torch_peak_allocated_bytes": 1,
                "torch_peak_reserved_bytes": int(screen.VRAM_CEILING_MIB * 1048576),
                "process_gpu_peak_mib": screen.VRAM_CEILING_MIB,
            },
            "timing_seconds": {"forward": 1, "total": 2},
            "precision_contract": {"passed": True},
            "output_contract": {"geometry_passed": True, "finite_passed": True, "normalized_passed": True},
        }
        self.assertTrue(screen.result_row(result)["eligible"])
        result["memory"]["process_gpu_peak_mib"] += 1
        self.assertFalse(screen.result_row(result)["eligible"])

    def test_ranking_prioritizes_coverage_then_precision(self):
        rows = []
        for candidate, spatial, precision, peak in (
            ("whole_body_amp", "whole_body", "amp", 30000),
            ("body_envelope_fp32", "body_envelope", "fp32", 10000),
            ("organ_group_fp32", "organ_group", "fp32", 5000),
        ):
            rows.append({"candidate_id": candidate, "strategy": spatial, "precision": precision,
                         "eligible": True, "measured_peak_mib": peak, "total_seconds": 1,
                         "eligibility_reason": "eligible"})
        selected = screen.select_configurations(rows)
        self.assertEqual(selected["preferred"]["candidate_id"], "whole_body_amp")
        self.assertEqual(selected["fallback"]["candidate_id"], "body_envelope_fp32")

    def test_selection_blocks_without_two_candidates(self):
        selected = screen.select_configurations([
            {"candidate_id": "organ_group_amp", "strategy": "organ_group", "precision": "amp",
             "eligible": True, "measured_peak_mib": 100, "total_seconds": 1,
             "eligibility_reason": "eligible"}
        ])
        self.assertEqual(selected["status"], "BLOCKED")


class CompatibilityAndSafetyTests(unittest.TestCase):
    def test_module_parses_with_python_37_grammar(self):
        source = Path(screen.__file__).read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 7))

    def test_atomic_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            screen.atomic_json(path, {"status": "first"}, refuse=True)
            with self.assertRaisesRegex(screen.Stage3Error, "overwrite"):
                screen.atomic_json(path, {"status": "second"}, refuse=True)
            self.assertEqual(json.loads(path.read_text())["status"], "first")

    def test_repository_contract_rejects_dirty_or_wrong_ancestry(self):
        with mock.patch.object(screen, "git_output", side_effect=[screen.EXPECTED_BRANCH, "a" * 40, "dirty"]), \
             mock.patch.object(screen.subprocess, "call", return_value=0):
            with self.assertRaisesRegex(screen.Stage3Error, "Repository contract"):
                screen.validate_repository(Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
