import argparse
import io
import json
import os
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tools.quadra import artifact_backup as backup


def make_run(root, relative="runs/cohort/run-001", status="COMPLETE"):
    run = Path(root) / relative
    run.mkdir(parents=True)
    (run / "result.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (run / "run_summary.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    return run


class BackupLayoutTests(unittest.TestCase):
    def test_rejects_broad_roots(self):
        for value in ("/", str(Path.home())):
            with self.assertRaises(backup.BackupError):
                backup.validate_archive_root(Path(value))

    def test_prepare_layout_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            first = backup.prepare_local_layout(root)
            second = backup.prepare_local_layout(root)
            self.assertEqual(first, second)
            self.assertTrue((root / "runs/cohort").is_dir())
            self.assertTrue((root / "reviews/masks").is_dir())

    def test_rejects_symlink_that_escapes_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            backup.prepare_local_layout(root)
            outside = Path(directory) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")
            link = root / "runs/cohort/link.txt"
            link.symlink_to(outside)
            with self.assertRaises(backup.BackupError):
                backup.build_inventory(root)


class InventoryTests(unittest.TestCase):
    def test_inventory_is_sorted_and_checksum_backed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            backup.prepare_local_layout(root)
            run = make_run(root)
            (run / "a.txt").write_text("a", encoding="utf-8")
            inventory = backup.build_inventory(root)
            paths = [item["path"] for item in inventory["entries"]]
            self.assertEqual(paths, sorted(paths))
            result = next(item for item in inventory["entries"] if item["path"].endswith("result.csv"))
            self.assertEqual(result["sha256"], backup.sha256_file(run / "result.csv"))
            self.assertEqual(inventory["run_states"]["runs/cohort/run-001"], "COMPLETE")

    def test_in_progress_run_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            backup.prepare_local_layout(root)
            make_run(root, status="IN_PROGRESS")
            inventory = backup.build_inventory(root)
            self.assertEqual(
                inventory["summary"]["in_progress_runs"],
                ["runs/cohort/run-001"],
            )

    def test_nested_item_status_does_not_mark_archived_run_in_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            backup.prepare_local_layout(root)
            run = root / "runs/archive/provenance"
            run.mkdir(parents=True)
            (run / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "scans": [
                            {"status": "pending", "completed": False},
                            {"status": "valid", "completed": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            inventory = backup.build_inventory(root)
            self.assertEqual(
                inventory["run_states"]["runs/archive/provenance"], "UNKNOWN"
            )
            self.assertEqual(inventory["summary"]["in_progress_runs"], [])

    def test_compare_classifies_missing_changed_and_identical(self):
        remote = {
            "entries": [
                {"path": "a", "sha256": "1", "size": 1, "type": "file"},
                {"path": "b", "sha256": "2", "size": 1, "type": "file"},
            ],
            "excluded_roots": [],
            "summary": {"in_progress_runs": []},
        }
        local = {
            "entries": [
                {"path": "a", "sha256": "1", "size": 1, "type": "file"},
                {"path": "b", "sha256": "3", "size": 1, "type": "file"},
                {"path": "c", "sha256": "4", "size": 1, "type": "file"},
            ]
        }
        result = backup.compare_inventories(remote, local)
        self.assertEqual(result["counts"]["IDENTICAL"], 1)
        self.assertEqual(result["counts"]["CONFLICT"], 1)
        self.assertEqual(result["counts"]["LOCAL_ONLY"], 1)

    def test_previous_inventory_distinguishes_remote_and_local_change(self):
        previous = {
            "entries": [
                {"path": "a", "sha256": "old", "size": 1, "type": "file"},
                {"path": "b", "sha256": "old", "size": 1, "type": "file"},
            ]
        }
        remote = {
            "entries": [
                {"path": "a", "sha256": "new", "size": 1, "type": "file"},
                {"path": "b", "sha256": "old", "size": 1, "type": "file"},
            ],
            "excluded_roots": [],
            "summary": {"in_progress_runs": []},
        }
        local = {
            "entries": [
                {"path": "a", "sha256": "old", "size": 1, "type": "file"},
                {"path": "b", "sha256": "new", "size": 1, "type": "file"},
            ]
        }
        result = backup.compare_inventories(remote, local, previous)
        rows = {item["path"]: item["status"] for item in result["rows"]}
        self.assertEqual(rows["a"], "REMOTE_CHANGED")
        self.assertEqual(rows["b"], "LOCAL_CHANGED")


class PackageTests(unittest.TestCase):
    def test_package_extract_and_atomic_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            remote = base / "remote"
            local = base / "local"
            backup.prepare_local_layout(remote)
            backup.prepare_local_layout(local)
            make_run(remote)
            inventory = backup.build_inventory(remote)
            package = base / "backup.tar.gz"
            backup.create_package(remote, package, inventory)
            extraction = base / "extracted"
            embedded = backup.safe_extract(package, extraction)
            self.assertEqual(backup._entry_map(embedded), backup._entry_map(inventory))
            operations = backup.promote_extracted(extraction, local, "transfer-1")
            self.assertTrue(any(item["status"] == "promoted" for item in operations))
            comparison = backup.compare_inventories(inventory, backup.build_inventory(local))
            self.assertEqual(comparison["counts"].get("REMOTE_ONLY", 0), 0)
            self.assertEqual(comparison["counts"].get("CONFLICT", 0), 0)

    def test_conflicting_run_is_preserved_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            remote = base / "remote"
            local = base / "local"
            backup.prepare_local_layout(remote)
            backup.prepare_local_layout(local)
            make_run(remote)
            local_run = make_run(local)
            (local_run / "result.csv").write_text("different", encoding="utf-8")
            inventory = backup.build_inventory(remote)
            package = base / "backup.tar.gz"
            backup.create_package(remote, package, inventory)
            extraction = base / "extracted"
            backup.safe_extract(package, extraction)
            with self.assertRaises(backup.BackupError):
                backup.promote_extracted(extraction, local, "conflict-1")
            self.assertEqual(
                (local_run / "result.csv").read_text(encoding="utf-8"), "different"
            )
            self.assertTrue(
                (local / "transfer/conflicts/conflict-1/runs/cohort/run-001").is_dir()
            )

    def test_malicious_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "bad.tar.gz"
            with tarfile.open(str(package), "w:gz") as archive:
                info = tarfile.TarInfo("../escape.txt")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(backup.BackupError):
                backup.safe_extract(package, Path(directory) / "extract")


class LocalIntakeTests(unittest.TestCase):
    def test_intake_copies_review_evidence_but_not_mask_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repository = base / "repository"
            repository.mkdir()
            stage5 = repository / "outputs/quadra_totalsegmentator_stage5"
            (stage5 / "evidence").mkdir(parents=True)
            (stage5 / "evidence/report.json").write_text("{}", encoding="utf-8")
            (stage5 / "manual_review_queue.csv").write_text("id\n1\n", encoding="utf-8")
            (stage5 / "backup").mkdir()
            archive = stage5 / "backup/quadra-totalsegmentator-stage5-20260727.tar"
            archive.write_bytes(b"mask archive")
            Path(str(archive) + ".sha256").write_text("abc  archive\n", encoding="utf-8")
            local = base / "local"
            backup.prepare_local_layout(local)
            result = backup.ingest_repository_artifacts(repository, local)
            self.assertTrue(
                (local / "reviews/masks/totalsegmentator_2.16.0_stage5/evidence/report.json").is_file()
            )
            self.assertFalse(
                (local / "reviews/masks/totalsegmentator_2.16.0_stage5/backup").exists()
            )
            self.assertEqual(result["known_assets"]["assets"][0]["copied_into_archive"], False)


class StopSafetyTests(unittest.TestCase):
    def test_safe_stop_blocks_active_process_and_dirty_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "local"
            backup.prepare_local_layout(local)
            make_run(local)
            inventory = backup.build_inventory(local)
            runtime = {
                "active_processes": ["123 python aligned_organ_group_cohort"],
                "repository": {"status_porcelain": " M script.py"},
            }
            args = argparse.Namespace(
                local_root=local,
                ssh_host="root@example",
            )
            output = io.StringIO()
            with mock.patch.object(
                backup, "_run_remote_json", side_effect=[inventory, runtime]
            ), redirect_stdout(output):
                code = backup.command_safe_stop(args)
            self.assertEqual(code, 2)
            result = json.loads(output.getvalue())
            self.assertEqual(result["verdict"], "NOT_SAFE_TO_STOP")
            self.assertTrue(result["repository_dirty"])


class CliTests(unittest.TestCase):
    def test_setup_dispatch_commands_parse(self):
        parser = backup.build_parser()
        args = parser.parse_args(["backup-init", "--local-root", "/tmp/quadra-local"])
        self.assertEqual(args.command, "backup-init")
        args = parser.parse_args(
            [
                "backup-pull",
                "--local-root",
                "/tmp/quadra-local",
                "--transport",
                "local-package",
                "--package-file",
                "/tmp/package.tar.gz",
            ]
        )
        self.assertEqual(args.transport, "local-package")


if __name__ == "__main__":
    unittest.main()
