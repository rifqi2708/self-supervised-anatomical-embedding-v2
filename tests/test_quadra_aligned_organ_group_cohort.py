import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from tools.quadra import aligned_organ_group_cohort as cohort


class RegistryAndSamplingTests(unittest.TestCase):
    def test_registry_is_fixed_and_sex_specific_absence_does_not_renumber(self):
        path = cohort.PROJECT_ROOT / "tools/quadra/totalsegmentator/organs.yaml"
        records = cohort.registry_records(path)
        self.assertEqual(len(records), 40)
        self.assertEqual([item["registry_index"] for item in records], list(range(40)))
        prostate = next(item for item in records if item["filename"] == "prostate")
        cervical = next(item for item in records if item["filename"] == "spinal_cord_cervical")
        self.assertEqual(prostate["group_name"], "pelvis")
        self.assertGreater(cervical["registry_index"], prostate["registry_index"])
        female = [item for item in records if item["filename"] != "prostate"]
        self.assertEqual(cervical["registry_index"], next(item for item in female if item["filename"] == "spinal_cord_cervical")["registry_index"])

    def test_unique_sampling_is_deterministic_and_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.nii.gz"
            data = np.zeros((12, 13, 14), dtype=np.uint8)
            data[1:11, 1:12, 1:13] = 1
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
            _, first, available_first = cohort.sample_unique_mask_points(path, 100, 20260722)
            _, second, available_second = cohort.sample_unique_mask_points(path, 100, 20260722)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(available_first, available_second)
            self.assertEqual(len({tuple(item) for item in first.tolist()}), 100)
            self.assertTrue(np.all(data[tuple(first.T)] == 1))

    def test_small_mask_uses_all_available_voxels_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.nii.gz"
            data = np.zeros((5, 5, 5), dtype=np.uint8)
            data.flat[:99] = 1
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
            _, points, available = cohort.sample_unique_mask_points(path, 100, 1)
            self.assertEqual(available, 99)
            self.assertEqual(len(points), 99)
            self.assertEqual(len({tuple(item) for item in points.tolist()}), 99)


class ResumeAndStatusTests(unittest.TestCase):
    def _manifest(self, root):
        return {
            "contract_signature": "contract",
            "denominators": {"subject_query_counts": {"quadra_hc_021": 1}},
        }

    def _records(self, root):
        test = {"path": str(root / "test.json"), "bytes": 1, "sha256": "a"}
        retest = {"path": str(root / "retest.json"), "bytes": 1, "sha256": "b"}
        return test, retest

    def test_orphan_csv_is_regenerated_but_incompatible_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "group_results/quadra_hc_021"
            result_dir.mkdir(parents=True)
            csv_path = result_dir / "pelvis.csv"
            csv_path.write_text("query_id\nq\n", encoding="utf-8")
            rows = [{"query_id": "q"}]
            test, retest = self._records(root)
            self.assertIsNone(cohort.validate_group_result(root, self._manifest(root), "quadra_hc_021", "pelvis", rows, test, retest))
            self.assertFalse(csv_path.exists())
            csv_path.write_text("query_id\nq\n", encoding="utf-8")
            (result_dir / "pelvis.json").write_text(json.dumps({"status": "success", "group_signature": "wrong"}), encoding="utf-8")
            with self.assertRaises(cohort.CohortError):
                cohort.validate_group_result(root, self._manifest(root), "quadra_hc_021", "pelvis", rows, test, retest)

    def test_status_payload_is_compact_and_reports_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = cohort.initial_status(
                {"run_directory": str(root), "denominators": {"queries": 108431}}
            )
            value["controller_pid"] = 999999999
            cohort.atomic_json(root / "cohort_status.json", value)
            with mock.patch.object(cohort, "gpu_snapshot", return_value={"memory_used_mib": 10}):
                payload = cohort.status_payload(root)
            self.assertFalse(payload["controller_process_alive"])
            self.assertIn("disk_free_gib", payload)
            self.assertEqual(payload["queries_total"], 108431)

    def test_group_signature_changes_with_query_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test, retest = self._records(root)
            manifest = self._manifest(root)
            left = cohort._group_signature(manifest, "quadra_hc_021", "pelvis", [{"query_id": "a"}], test, retest)
            right = cohort._group_signature(manifest, "quadra_hc_021", "pelvis", [{"query_id": "b"}], test, retest)
            self.assertNotEqual(left, right)


class ConsolidationAndCompatibilityTests(unittest.TestCase):
    def test_summary_counts_are_exact(self):
        rows = [
            {"subject_id": "a", "cycle_error_mm": "1.0"},
            {"subject_id": "a", "cycle_error_mm": "3.0"},
            {"subject_id": "b", "cycle_error_mm": "2.0"},
        ]
        result = cohort._summaries(rows, ["subject_id"])
        self.assertEqual([item["count"] for item in result], [2, 1])
        self.assertEqual(result[0]["median_cycle_error_mm"], 2.0)

    def test_python37_syntax_compatibility(self):
        path = cohort.PROJECT_ROOT / "tools/quadra/aligned_organ_group_cohort.py"
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path), feature_version=(3, 7))
        except TypeError:
            # Python 3.7 itself predates this parser convenience argument; a
            # native parse is the stronger compatibility check in that image.
            ast.parse(source, filename=str(path))

    def test_result_schema_has_continuous_and_rounded_coordinates(self):
        fields = set(cohort.RESULT_FIELDS)
        self.assertIn("matched_raw_x", fields)
        self.assertIn("matched_raw_rounded_x", fields)
        self.assertIn("cycle_error_mm", fields)
        self.assertNotIn("embedding_path", fields)


if __name__ == "__main__":
    unittest.main()
