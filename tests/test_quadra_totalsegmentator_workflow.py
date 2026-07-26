import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from tools.quadra.totalsegmentator.cli import main
from tools.quadra.totalsegmentator.core import (
    DEFAULT_REGISTRY,
    expected_mask_names,
    load_registry,
    registry_identity,
    sha256_file,
    task_classes,
)
from tools.quadra.totalsegmentator.workflow import (
    DISK_GUARD_EXIT_CODE,
    SYSTEMIC_FAILURE_EXIT_CODE,
    build_commands,
    completed_scan_is_compatible,
    cohort_status,
    run_cohort,
    run_scan,
    write_status_csv,
)


class TotalSegmentatorWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry(DEFAULT_REGISTRY)

    def _input(self, root: Path) -> Path:
        path = root / "input image.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((5, 5, 20), dtype=np.int16), np.eye(4)),
            path,
        )
        return path

    def _scan(self, input_path: Path, sex: str = "M", subject: str = "quadra_hc_022"):
        return {
            "subject_id": subject,
            "session": "test",
            "sex": sex,
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "input_size_bytes": input_path.stat().st_size,
            "expected_masks": expected_mask_names(self.registry, sex),
            "task_classes": task_classes(self.registry, sex),
        }

    def _manifest(self, scan):
        return {
            "schema_version": 1,
            "run_id": "unit-test",
            "totalsegmentator_version": "2.16.0",
            "registry": registry_identity(DEFAULT_REGISTRY),
            "git_commit": "abc",
            "scans": [scan],
        }

    def _fake_executable(self, root: Path) -> Path:
        path = root / "fake TotalSegmentator"
        script = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

args = sys.argv[1:]
input_path = Path(args[args.index("-i") + 1])
output = Path(args[args.index("-o") + 1])
task = args[args.index("-ta") + 1]
if "--roi_subset" in args:
    classes = args[args.index("--roi_subset") + 1:args.index("--report")]
elif task == "head_glands_cavities":
    classes = ["eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right",
               "parotid_gland_left", "parotid_gland_right"]
else:
    raise SystemExit(f"Unsupported fake task without --roi_subset: {task}")
report = Path(args[args.index("--report") + 1])
image = nib.load(str(input_path))
positions = {"vertebrae_C1": 18, "vertebrae_C7": 15, "vertebrae_T1": 13,
             "vertebrae_T12": 5, "vertebrae_L1": 3}
output.mkdir(parents=True, exist_ok=True)
report.parent.mkdir(parents=True, exist_ok=True)
for name in classes:
    data = np.zeros(image.shape, dtype=np.uint8)
    if name == "spinal_cord":
        data[2, 2, 2:20] = 1
    elif name in positions:
        data[1:3, 1:3, positions[name]] = 1
    else:
        data[2, 2, 10] = 1
    nib.save(nib.Nifti1Image(data, image.affine), output / f"{name}.nii.gz")
report.write_text(json.dumps({"status": "ok", "classes": classes}))
"""
        path.write_text(textwrap.dedent(script), encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_commands_are_task_batched_full_resolution_and_quote_safe(self):
        with tempfile.TemporaryDirectory(prefix="path with spaces ") as directory:
            root = Path(directory)
            scan = self._scan(self._input(root))
            commands = build_commands(
                scan, self.registry, root / "work directory", "Total Segmentator"
            )
            self.assertEqual({row["task"] for row in commands}, {"total", "head_glands_cavities"})
            for row in commands:
                self.assertIn("--report", row["command"])
                self.assertNotIn("--fast", row["command"])
                self.assertNotIn("--fastest", row["command"])
            total = next(row for row in commands if row["task"] == "total")
            head = next(row for row in commands if row["task"] == "head_glands_cavities")
            self.assertIn("--roi_subset", total["command"])
            self.assertNotIn("--roi_subset", head["command"])
            self.assertIn("prostate", total["classes"])

    def test_female_command_never_requests_prostate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = self._scan(self._input(root), "F", "quadra_hc_021")
            classes = {
                value
                for command in build_commands(scan, self.registry, root / "work")
                for value in command["classes"]
            }
            self.assertNotIn("prostate", classes)

    def test_fake_executable_runs_scan_atomically_and_resumes(self):
        with tempfile.TemporaryDirectory(prefix="workflow with spaces ") as directory:
            root = Path(directory)
            input_path = self._input(root)
            scan = self._scan(input_path)
            manifest = self._manifest(scan)
            executable = self._fake_executable(root)
            output_root = root / "outputs with spaces"
            result = run_scan(
                manifest,
                scan,
                DEFAULT_REGISTRY,
                output_root,
                root / "scratch",
                str(executable),
            )
            self.assertEqual(result["status"], "completed")
            output = Path(result["output_directory"])
            self.assertEqual(len(list((output / "masks").glob("*.nii.gz"))), 40)
            self.assertTrue((output / "intermediate" / "spinal_cord.nii.gz").is_file())
            self.assertTrue((output / "intermediate" / "vertebrae_T1.nii.gz").is_file())
            run_manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertTrue(run_manifest["completed"])
            self.assertFalse(run_manifest["qc"]["anatomical_accuracy_assessed"])
            self.assertTrue(
                completed_scan_is_compatible(output, manifest, scan, DEFAULT_REGISTRY)
            )
            resumed = run_scan(
                manifest,
                scan,
                DEFAULT_REGISTRY,
                output_root,
                root / "scratch",
                str(executable),
            )
            self.assertEqual(resumed["status"], "skipped")

    def test_cohort_dry_run_and_failure_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan1 = self._scan(self._input(root))
            scan2 = {**scan1, "session": "retest"}
            manifest = self._manifest(scan1)
            manifest["scans"] = [scan1, scan2]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            code, dry = run_cohort(
                manifest_path,
                DEFAULT_REGISTRY,
                root / "outputs",
                root / "scratch",
                dry_run=True,
            )
            self.assertEqual(code, 0)
            self.assertEqual(dry["status"], "dry_run_complete")
            self.assertEqual(len(dry["scans"]), 2)

            with mock.patch(
                "tools.quadra.totalsegmentator.workflow.run_scan",
                side_effect=[RuntimeError("first failure"), {"status": "completed"}],
            ), mock.patch(
                "tools.quadra.totalsegmentator.workflow.free_disk_gib",
                return_value=100,
            ):
                code, actual = run_cohort(
                    manifest_path,
                    DEFAULT_REGISTRY,
                    root / "outputs",
                    root / "scratch",
                    min_free_gib=20,
                )
            self.assertEqual(code, 1)
            self.assertEqual([row["status"] for row in actual["scans"]], ["failed", "completed"])

    def test_disk_guard_stops_before_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = self._scan(self._input(root))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(self._manifest(scan)), encoding="utf-8")
            with mock.patch(
                "tools.quadra.totalsegmentator.workflow.free_disk_gib",
                return_value=0,
            ), mock.patch(
                "tools.quadra.totalsegmentator.workflow.run_scan"
            ) as runner:
                code, result = run_cohort(
                    manifest_path,
                    DEFAULT_REGISTRY,
                    root / "outputs",
                    root / "scratch",
                    min_free_gib=20,
                )
            self.assertEqual(code, DISK_GUARD_EXIT_CODE)
            self.assertEqual(result["status"], "stopped_low_disk")
            runner.assert_not_called()

    def test_changed_input_stops_cohort_as_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = self._input(root)
            scan = self._scan(input_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(self._manifest(scan)), encoding="utf-8")
            input_path.write_bytes(b"changed after manifest preparation")
            with mock.patch(
                "tools.quadra.totalsegmentator.workflow.free_disk_gib",
                return_value=100,
            ):
                code, result = run_cohort(
                    manifest_path,
                    DEFAULT_REGISTRY,
                    root / "outputs",
                    root / "scratch",
                    min_free_gib=20,
                )
            self.assertEqual(code, SYSTEMIC_FAILURE_EXIT_CODE)
            self.assertEqual(result["status"], "stopped_integrity_error")
            self.assertEqual(result["scans"][0]["status"], "failed_integrity")

    def test_status_reports_failures_and_writes_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = self._scan(self._input(root))
            manifest = self._manifest(scan)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            run_root = root / "outputs" / manifest["run_id"]
            run_root.mkdir(parents=True)
            (run_root / "cohort_status.json").write_text(
                json.dumps(
                    {
                        "scans": [
                            {
                                "subject_id": scan["subject_id"],
                                "session": scan["session"],
                                "status": "failed",
                                "error": "synthetic failure",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = cohort_status(manifest_path, root / "outputs")
            self.assertEqual(result["summary"]["failed"], 1)
            self.assertEqual(result["scans"][0]["detail"], "synthetic failure")
            csv_path = root / "status.csv"
            write_status_csv(csv_path, result)
            self.assertIn("synthetic failure", csv_path.read_text(encoding="utf-8"))

    def test_cli_help(self):
        with self.assertRaises(SystemExit) as exit_context:
            main(["--help"])
        self.assertEqual(exit_context.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
