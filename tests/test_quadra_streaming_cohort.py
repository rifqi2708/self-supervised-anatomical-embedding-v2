import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.quadra.streaming_cycle_error_cohort import (
    DISK_GUARD_EXIT_CODE,
    build_subject_command,
    expand_subject_range,
    expected_manifest_settings,
    manifest_is_compatible,
    parse_args,
    run,
)


class QuadraStreamingCohortTests(unittest.TestCase):
    def test_default_range_expands_to_exactly_28_subjects(self):
        subjects = expand_subject_range(21, 48)
        self.assertEqual(len(subjects), 28)
        self.assertEqual(subjects[0], "quadra_hc_021")
        self.assertEqual(subjects[-1], "quadra_hc_048")

    def test_single_subject_command_forwards_cache_policy_and_settings(self):
        args = parse_args(
            [
                "--keep-cache",
                "--overwrite-cache",
                "--organs",
                "colon",
                "liver",
                "--num-points",
                "7",
            ]
        )
        command = build_subject_command(args, "quadra_hc_021")
        self.assertIn("--keep-cache", command)
        self.assertIn("--overwrite-cache", command)
        self.assertEqual(command[command.index("--num-points") + 1], "7")
        self.assertEqual(command[command.index("--organs") + 1 :], ["colon", "liver", "--overwrite-cache", "--keep-cache"])

    def test_default_command_deletes_cache_on_success(self):
        command = build_subject_command(parse_args([]), "quadra_hc_021")
        self.assertNotIn("--keep-cache", command)

    def test_manifest_compatibility_requires_all_relevant_settings(self):
        args = parse_args([])
        settings = expected_manifest_settings(args)
        manifest = {
            "schema_version": 3,
            "completed": True,
            "subject_id": "quadra_hc_021",
            "cache_cleanup": {"status": "deleted"},
            "config": {"sha256": "config"},
            "checkpoint": {"sha256": "checkpoint"},
            **settings,
        }
        self.assertTrue(
            manifest_is_compatible(manifest, "quadra_hc_021", settings, "config", "checkpoint")
        )
        manifest["halo_xyz"] = [32, 32, 16]
        self.assertFalse(
            manifest_is_compatible(manifest, "quadra_hc_021", settings, "config", "checkpoint")
        )

    def _cohort_args(self, temp: Path, *extra: str):
        config = temp / "config.py"
        checkpoint = temp / "SAM.pth"
        config.write_text("config", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        return parse_args(
            [
                "--subject-start",
                "21",
                "--subject-end",
                "22",
                "--config-file",
                str(config),
                "--checkpoint-file",
                str(checkpoint),
                "--output-root",
                str(temp / "outputs"),
                "--cache-root",
                str(temp / "cache"),
                "--batch-output-root",
                str(temp / "batches"),
                "--min-free-gb",
                "0",
                *extra,
            ]
        )

    def test_ordinary_subject_failures_continue_to_next_subject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._cohort_args(Path(temp_dir))
            with mock.patch(
                "tools.quadra.streaming_cycle_error_cohort.find_compatible_completed_run",
                return_value=None,
            ), mock.patch(
                "tools.quadra.streaming_cycle_error_cohort.run_subprocess_with_log",
                return_value=1,
            ) as runner:
                exit_code, batch_dir = run(args)
            self.assertEqual(exit_code, 1)
            self.assertEqual(runner.call_count, 2)
            manifest = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["failed_subjects"], ["quadra_hc_021", "quadra_hc_022"])
            self.assertEqual(manifest["status"], "completed_with_failures")

    def test_disk_guard_stops_before_launch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._cohort_args(Path(temp_dir), "--min-free-gb", "20")
            with mock.patch(
                "tools.quadra.streaming_cycle_error_cohort.find_compatible_completed_run",
                return_value=None,
            ), mock.patch(
                "tools.quadra.streaming_cycle_error_cohort.free_disk_bytes",
                return_value=0,
            ), mock.patch(
                "tools.quadra.streaming_cycle_error_cohort.run_subprocess_with_log"
            ) as runner:
                exit_code, batch_dir = run(args)
            self.assertEqual(exit_code, DISK_GUARD_EXIT_CODE)
            runner.assert_not_called()
            manifest = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "stopped_low_disk")

    def test_dry_run_records_every_requested_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._cohort_args(Path(temp_dir), "--dry-run")
            with mock.patch(
                "tools.quadra.streaming_cycle_error_cohort.find_compatible_completed_run",
                return_value=None,
            ):
                exit_code, batch_dir = run(args)
            self.assertEqual(exit_code, 0)
            manifest = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "dry_run_complete")
            self.assertEqual(set(manifest["subjects"]), {"quadra_hc_021", "quadra_hc_022"})


if __name__ == "__main__":
    unittest.main()
