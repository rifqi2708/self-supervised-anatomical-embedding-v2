import argparse
import gzip
import io
import json
import os
import struct
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tools.quadra import artifact_backup as backup
from tools.quadra import disposable_pod as disposable


class CatalogueTests(unittest.TestCase):
    def test_catalogue_selects_ready_profile_assets(self):
        catalog = disposable.load_catalog(disposable.DEFAULT_CATALOG)
        registration = dict(disposable.required_assets(catalog, "registration", require_ready=True))
        self.assertEqual(set(registration), {"whole_body_ct", "stage5_masks", "experiment_contract"})

    def test_catalogue_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            catalog = json.loads(disposable.DEFAULT_CATALOG.read_text())
            catalog["assets"]["whole_body_ct"]["promote_to"] = "../escape"
            path.write_text(json.dumps(catalog))
            with self.assertRaises(disposable.DisposableError):
                disposable.load_catalog(path)


class ArchiveTests(unittest.TestCase):
    @staticmethod
    def _nifti(path, values, datatype=2, bitpix=8):
        header = bytearray(352)
        struct.pack_into("<I", header, 0, 348)
        struct.pack_into("<8h", header, 40, 3, 2, 2, 2, 1, 1, 1, 1)
        struct.pack_into("<h", header, 70, datatype)
        struct.pack_into("<h", header, 72, bitpix)
        struct.pack_into("<8f", header, 76, 1, 1, 1, 1, 0, 0, 0, 0)
        struct.pack_into("<f", header, 108, 352)
        header[344:348] = b"n+1\x00"
        with gzip.open(str(path), "wb") as handle:
            handle.write(header)
            handle.write(bytes(values))

    def test_zip_and_tar_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            zipped = base / "x.zip"
            with zipfile.ZipFile(str(zipped), "w") as archive:
                archive.writestr("safe/file.txt", "ok")
            self.assertEqual(disposable.validate_archive_members(zipped, "zip"), ["safe/file.txt"])
            tarred = base / "x.tar"
            source = base / "file.txt"
            source.write_text("ok")
            with tarfile.open(str(tarred), "w") as archive:
                archive.add(str(source), arcname="safe/file.txt")
            self.assertEqual(disposable.validate_archive_members(tarred, "tar"), ["safe/file.txt"])

    def test_path_traversal_and_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tar"
            with tarfile.open(str(path), "w") as archive:
                info = tarfile.TarInfo("../bad")
                info.size = 0
                archive.addfile(info)
            with self.assertRaises(disposable.DisposableError):
                disposable.validate_archive_members(path, "tar")

    def test_download_requires_exact_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.tar"
            source = Path(directory) / "x"
            source.write_text("ok")
            with tarfile.open(str(path), "w") as archive:
                archive.add(str(source), arcname="x")
            item = {"bytes": path.stat().st_size, "sha256": disposable.sha256_file(path), "archive_type": "tar"}
            self.assertEqual(disposable.verify_download(path, item)["sha256"], item["sha256"])
            item["bytes"] += 1
            with self.assertRaisesRegex(disposable.DisposableError, "Size mismatch"):
                disposable.verify_download(path, item)

    def test_complete_nifti_payload_validation_detects_mask_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.nii.gz"
            empty = Path(directory) / "empty.nii.gz"
            invalid = Path(directory) / "invalid.nii.gz"
            self._nifti(valid, [0, 1, 0, 1, 0, 0, 0, 0])
            self._nifti(empty, [0] * 8)
            self._nifti(invalid, [0, 1, 2, 0, 0, 0, 0, 0])
            self.assertEqual(disposable.validate_nifti_payload(valid, binary=True)["nonzero"], 2)
            with self.assertRaisesRegex(disposable.DisposableError, "empty"):
                disposable.validate_nifti_payload(empty, binary=True)
            with self.assertRaisesRegex(disposable.DisposableError, "not binary"):
                disposable.validate_nifti_payload(invalid, binary=True)

    def test_promoted_mask_resume_resolves_embedded_checksum_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            extraction = base / "extracted"
            payload = base / "promoted"
            provenance = payload / "_failed/example/logs/example.log"
            provenance.parent.mkdir(parents=True)
            provenance.write_text("verified provenance\n")
            sums = extraction / "logs/checksums.sha256"
            sums.parent.mkdir(parents=True)
            sums.write_text(
                "{}  outputs/payload/_failed/example/logs/example.log\n".format(
                    disposable.sha256_file(provenance)
                )
            )
            item = {
                "payload_subpath": "outputs/payload",
                "checksum_manifest": "logs/checksums.sha256",
                "checksum_entries": 1,
                "expected": {
                    "final_masks": 0,
                    "intermediate_masks": 0,
                    "subjects": 0,
                    "scans": 0,
                },
            }
            result = disposable.validate_extracted_asset(
                "stage5_masks", payload, item,
                extraction_root=extraction, promoted=True,
            )
            self.assertEqual(result["embedded_checksums_verified"], 1)


class ProfileTests(unittest.TestCase):
    def test_selective_ct_payload_keeps_only_021_through_048(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            payload = base / "all"
            payload.mkdir()
            for number in range(1, 49):
                subject = payload / "QUADRA_HC_{:03d}".format(number)
                subject.mkdir()
                (subject / "test_CT-AC.nii.gz").write_bytes(b"test")
                (subject / "retest_CT-AC.nii.gz").write_bytes(b"retest")
            item = json.loads(disposable.DEFAULT_CATALOG.read_text())["assets"]["whole_body_ct"]
            with mock.patch.object(disposable, "validate_extracted_asset") as validate:
                selected = disposable.select_profile_payload("whole_body_ct", payload, item, base)
            self.assertEqual(len(list(selected.glob("QUADRA_HC_*"))), 28)
            self.assertTrue((selected / "QUADRA_HC_021").is_dir())
            self.assertTrue((selected / "QUADRA_HC_048").is_dir())
            self.assertFalse((selected / "QUADRA_HC_020").exists())
            self.assertTrue((payload / "QUADRA_HC_021/test_CT-AC.nii.gz").is_file())
            validate.assert_called_once()

    def test_profile_rejects_wrong_python_image_and_small_gpu(self):
        with mock.patch.object(disposable.sys, "version_info", (3, 7, 0)):
            with self.assertRaisesRegex(disposable.DisposableError, "tag"):
                disposable.validate_profile("uae", "wrong", disposable.EXPECTED_IMAGES["uae"]["digest"], 49140)
            with self.assertRaisesRegex(disposable.DisposableError, "48 GB"):
                disposable.validate_profile("uae", disposable.EXPECTED_IMAGES["uae"]["ref"], disposable.EXPECTED_IMAGES["uae"]["digest"], 20000)

    def test_profile_accepts_decimal_48_gb_gpu_reported_in_mib(self):
        with mock.patch.object(disposable.sys, "version_info", (3, 7, 0)):
            result = disposable.validate_profile(
                "uae",
                disposable.EXPECTED_IMAGES["uae"]["ref"],
                disposable.EXPECTED_IMAGES["uae"]["digest"],
                46068,
            )
        self.assertEqual(result, disposable.EXPECTED_IMAGES["uae"])

    def test_downloader_pin_is_legacy_python_compatible(self):
        self.assertEqual(disposable.gdown_requirement((3, 7, 10)), "gdown==4.7.3")
        self.assertEqual(disposable.gdown_requirement((3, 8, 0)), "gdown==5.2.0")
        self.assertEqual(disposable.gdown_requirement((3, 11, 10)), "gdown==5.2.0")

    def test_activation_template_preserves_shell_parameter_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            repository = Path(directory) / "repo"
            disposable._write_activation(root, repository, "uae")
            content = (root / "runtime/activate.sh").read_text()
            self.assertIn('profile="${1:-}"', content)
            self.assertIn('${profile:-<unset>}', content)
            self.assertIn('${PYTHONPATH:+:${PYTHONPATH}}', content)

    def test_container_storage_detection_compares_devices(self):
        fake_workspace = mock.Mock(stat=lambda: mock.Mock(st_dev=1))
        fake_root = mock.Mock(stat=lambda: mock.Mock(st_dev=1))
        self.assertTrue(disposable.paths_share_device(fake_workspace, fake_root))
        fake_root.stat = lambda: mock.Mock(st_dev=2)
        self.assertFalse(disposable.paths_share_device(fake_workspace, fake_root))

    def test_no_hardcoded_registration_pod_ids_remain(self):
        source = (Path(disposable.__file__).parent / "registration_runtime.py").read_text()
        self.assertNotIn("APPROVED_PODS", source)
        self.assertNotIn("2ohlzqc00kd7sn", source)
        self.assertNotIn("1ngcj5dw1mifiw", source)

    def test_repository_checkpoint_links_are_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "quadra"
            repository = base / "repo"
            models = root / "models/uae/base"
            models.mkdir(parents=True)
            for filename in ("SAM.pth", "SAMv2_iter_20000.pth"):
                (models / filename).write_bytes(filename.encode("ascii"))
            first = disposable._expose_repository_assets(root, repository, "uae")
            second = disposable._expose_repository_assets(root, repository, "uae")
            self.assertEqual(first, second)
            for record in first:
                path = Path(record["path"])
                self.assertTrue(path.is_symlink())
                self.assertTrue(disposable.is_within(path.resolve(), models))


class PackageResultTests(unittest.TestCase):
    def test_result_package_contains_only_the_selected_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            selected = root / "runs/cohort/selected"
            unrelated = root / "runs/cohort/unrelated"
            selected.mkdir(parents=True)
            unrelated.mkdir(parents=True)
            (selected / "result.csv").write_text("ok\n")
            (unrelated / "result.csv").write_text("do-not-package\n")
            args = argparse.Namespace(
                storage_root=root,
                run_directory=selected,
                transfer_id="subject-021",
                repository_root=Path(directory) / "repo",
            )
            with mock.patch.object(backup, "process_status", return_value={"active_processes": []}), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(disposable.command_package_results(args), 0)
            package = root / "staging/result-packages/subject-021.tar.gz"
            with tarfile.open(str(package), "r:gz") as archive:
                names = archive.getnames()
            self.assertIn("quadra/runs/cohort/selected/result.csv", names)
            self.assertNotIn("quadra/runs/cohort/unrelated/result.csv", names)

    def test_result_package_refuses_active_processes_and_interrupted_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            selected = root / "runs/uae/selected"
            selected.mkdir(parents=True)
            args = argparse.Namespace(
                storage_root=root,
                run_directory=selected,
                transfer_id="x",
                repository_root=Path(directory) / "repo",
            )
            with mock.patch.object(backup, "process_status", return_value={"active_processes": ["pid"]}):
                with self.assertRaisesRegex(disposable.DisposableError, "active"):
                    disposable.command_package_results(args)
            snapshot = root / "staging/result-packages/.x-snapshot"
            snapshot.mkdir(parents=True)
            with mock.patch.object(backup, "process_status", return_value={"active_processes": []}):
                with self.assertRaisesRegex(disposable.DisposableError, "Interrupted snapshot"):
                    disposable.command_package_results(args)


class BackupCoverageTests(unittest.TestCase):
    def test_complete_generated_evidence_roots_are_allowlisted(self):
        required = {
            "runs/analysis", "reviews/masks", "reviews/query_points",
            "exports/documents", "exports/presentations",
        }
        self.assertTrue(required.issubset(set(backup.ALLOWLIST)))

    def test_safe_terminate_blocks_unpublished_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "local"
            backup.prepare_local_layout(root)
            remote = backup.build_inventory(root)
            remote["entries"].append({"path": "metadata/manifests/disposable-uae-environment.json",
                                      "sha256": "x", "size": 1, "type": "file"})
            runtime = {"active_processes": [], "repository": {"status_porcelain": ""},
                       "repository_remote_refs_at_commit": ["refs/tags/quadra-disposable-v1"],
                       "unclassified_repository_artifacts": []}
            attestation = Path(directory) / "attestation.json"
            attestation.write_text(json.dumps({"attested_at": "now", "operator": "test",
                                               "all_temporary_drive_links_revoked": True}))
            catalog = json.loads(disposable.DEFAULT_CATALOG.read_text())
            catalog["assets"]["experiment_contract"]["drive_id"] = None
            catalog_path = Path(directory) / "unpublished-catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            args = argparse.Namespace(local_root=root, ssh_host="host", profile="uae",
                                      asset_catalog=catalog_path,
                                      drive_revocation_attestation=attestation,
                                      remote_root=Path("/workspace/quadra"))
            with mock.patch.object(backup, "_run_remote_json", side_effect=[remote, runtime]), redirect_stdout(io.StringIO()):
                self.assertEqual(backup.command_safe_terminate(args), 2)

    def test_safe_terminate_passes_with_live_complete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "local"
            backup.prepare_local_layout(root)
            environment_manifest = root / "metadata/manifests/disposable-uae-environment.json"
            environment_manifest.write_text("{}\n")
            remote = backup.build_inventory(root)
            runtime = {
                "checked_at": backup.utc_now(),
                "active_processes": [],
                "repository": {"status_porcelain": ""},
                "repository_remote_refs_at_commit": ["refs/tags/quadra-disposable-v1"],
                "unclassified_repository_artifacts": [],
            }
            attestation = Path(directory) / "attestation.json"
            attestation.write_text(json.dumps({
                "attested_at": "now", "operator": "test",
                "all_temporary_drive_links_revoked": True,
            }))
            catalog = json.loads(disposable.DEFAULT_CATALOG.read_text())
            catalog["assets"]["experiment_contract"]["drive_id"] = "private-drive-id"
            catalog_path = Path(directory) / "catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            args = argparse.Namespace(
                local_root=root, ssh_host="host", profile="uae",
                asset_catalog=catalog_path,
                drive_revocation_attestation=attestation,
                remote_root=Path("/workspace/quadra"),
            )
            with mock.patch.object(backup, "_run_remote_json", side_effect=[remote, runtime]), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(backup.command_safe_terminate(args), 0)

    def test_safe_terminate_accepts_checksum_transferred_complete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "local"
            backup.prepare_local_layout(root)
            environment_manifest = root / "metadata/manifests/disposable-uae-environment.json"
            environment_manifest.write_text("{}\n")
            remote = backup.build_inventory(root)
            runtime = {
                "checked_at": backup.utc_now(),
                "active_processes": [],
                "repository": {"status_porcelain": ""},
                "repository_remote_refs_at_commit": ["refs/tags/quadra-disposable-v1"],
                "unclassified_repository_artifacts": [],
            }
            remote_inventory_file = Path(directory) / "remote-inventory.json"
            remote_status_file = Path(directory) / "remote-status.json"
            remote_inventory_file.write_text(json.dumps(remote))
            remote_status_file.write_text(json.dumps(runtime))
            attestation = Path(directory) / "attestation.json"
            attestation.write_text(json.dumps({
                "attested_at": "now", "operator": "test",
                "all_temporary_drive_links_revoked": True,
            }))
            catalog = json.loads(disposable.DEFAULT_CATALOG.read_text())
            catalog["assets"]["experiment_contract"]["drive_id"] = "private-drive-id"
            catalog_path = Path(directory) / "catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            args = argparse.Namespace(
                local_root=root, ssh_host=None, profile="uae",
                asset_catalog=catalog_path,
                drive_revocation_attestation=attestation,
                remote_root=Path("/workspace/quadra"),
                remote_inventory_file=remote_inventory_file,
                remote_status_file=remote_status_file,
                max_evidence_age_seconds=300,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(backup.command_safe_terminate(args), 0)
            self.assertEqual(
                json.loads(output.getvalue())["evidence_mode"],
                "operator_transferred_files",
            )
            self.assertTrue(json.loads(output.getvalue())["evidence_fresh"])


class CliTests(unittest.TestCase):
    def test_all_disposable_commands_parse(self):
        parser = disposable.build_parser()
        self.assertEqual(parser.parse_args(["plan", "--profile", "uae"]).command, "plan")
        self.assertEqual(
            parser.parse_args(["status", "--profile", "registration"]).profile,
            "registration",
        )
        self.assertEqual(
            parser.parse_args(["_smoke", "--profile", "uae", "--output", "/tmp/x"]).profile,
            "uae",
        )


if __name__ == "__main__":
    unittest.main()
