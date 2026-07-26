import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from openpyxl import Workbook

from tools.quadra.totalsegmentator.core import (
    DEFAULT_REGISTRY,
    WorkflowError,
    canonical_subject_id,
    discover_scans,
    expected_mask_names,
    load_registry,
    prepare_manifest,
    read_demographics,
    task_classes,
)


class TotalSegmentatorCoreTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry(DEFAULT_REGISTRY)

    def test_registry_has_expected_sex_specific_counts(self):
        male = expected_mask_names(self.registry, "M")
        female = expected_mask_names(self.registry, "F")
        self.assertEqual(len(male), 40)
        self.assertEqual(len(female), 39)
        self.assertIn("prostate", male)
        self.assertNotIn("prostate", female)
        self.assertEqual(len(male), len(set(male)))

    def test_registry_uses_exactly_two_tasks(self):
        classes = task_classes(self.registry, "M")
        self.assertEqual(set(classes), {"total", "head_glands_cavities"})
        self.assertIn("spinal_cord", classes["total"])
        self.assertIn("vertebrae_T1", classes["total"])
        self.assertIn("optic_nerve_left", classes["head_glands_cavities"])

    def test_subject_normalization(self):
        self.assertEqual(canonical_subject_id(21), "quadra_hc_021")
        self.assertEqual(canonical_subject_id("QUADRA_HC_048"), "quadra_hc_048")

    def _write_csv(self, path: Path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Subject", "Sex"])
            writer.writerows(rows)

    def test_demographics_reject_missing_duplicate_and_invalid_sex(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demographics.csv"
            self._write_csv(path, [(21, "F")])
            with self.assertRaisesRegex(WorkflowError, "Missing demographics"):
                read_demographics(path, ["quadra_hc_021", "quadra_hc_022"])

            self._write_csv(path, [(21, "F"), (21, "F")])
            with self.assertRaisesRegex(WorkflowError, "Duplicate"):
                read_demographics(path, ["quadra_hc_021"])

            self._write_csv(path, [(21, "unknown")])
            with self.assertRaisesRegex(WorkflowError, "Invalid sex"):
                read_demographics(path, ["quadra_hc_021"])

    def test_xlsx_demographics_parser_finds_subject_and_sex_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demographics.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Demographics (All)"
            sheet.append(["Subject", "Sex", "Age [y]"])
            sheet.append([None, None, "secondary header"])
            sheet.append([21, "F", 30])
            sheet.append([22, "M", 31])
            workbook.save(path)
            result = read_demographics(
                path, ["quadra_hc_021", "quadra_hc_022"]
            )
            self.assertEqual(
                result, {"quadra_hc_021": "F", "quadra_hc_022": "M"}
            )

    def test_discovery_rejects_finetuning_subjects(self):
        with self.assertRaisesRegex(WorkflowError, "reserved"):
            discover_scans(Path("/does/not/matter"), 20, 21)

    def test_prepare_small_manifest_hashes_inputs_and_routes_sex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for number in (21, 22):
                subject = dataset / f"QUADRA_HC_{number:03d}"
                subject.mkdir(parents=True)
                for session in ("test", "retest"):
                    image = nib.Nifti1Image(
                        np.zeros((2, 2, 2), dtype=np.int16), np.eye(4)
                    )
                    nib.save(image, subject / f"{session}_CT-AC.nii.gz")
            demographics = root / "demographics.csv"
            self._write_csv(demographics, [(21, "F"), (22, "M")])
            manifest = prepare_manifest(
                dataset,
                demographics,
                DEFAULT_REGISTRY,
                21,
                22,
                "test-run",
            )
            self.assertEqual(manifest["summary"]["subjects"], 2)
            self.assertEqual(manifest["summary"]["scans"], 4)
            self.assertEqual(manifest["summary"]["expected_masks"], 158)
            self.assertTrue(all(len(scan["input_sha256"]) == 64 for scan in manifest["scans"]))
            female_scans = [scan for scan in manifest["scans"] if scan["sex"] == "F"]
            self.assertTrue(all("prostate" not in scan["expected_masks"] for scan in female_scans))


if __name__ == "__main__":
    unittest.main()
