import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.quadra import environment


class EnvironmentLayoutTests(unittest.TestCase):
    def test_rejects_broad_storage_roots(self):
        for value in ("/", "/workspace"):
            with self.assertRaises(environment.EnvironmentError):
                environment.validate_storage_root(Path(value))

    def test_resolve_path_precedence(self):
        with mock.patch.dict("os.environ", {"QUADRA_STORAGE_ROOT": "/tmp/quadra"}):
            self.assertEqual(
                environment.resolve_quadra_path(
                    "/explicit", "whole_body_ct", "data/fallback"
                ),
                Path("/explicit"),
            )
            self.assertEqual(
                environment.resolve_quadra_path(
                    None, "whole_body_ct", "data/fallback"
                ),
                Path("/tmp/quadra/datasets/source/whole_body_ct_v1"),
            )
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                environment.resolve_quadra_path(
                    None, "whole_body_ct", "data/fallback"
                ),
                Path("data/fallback"),
            )

    def test_safe_link_is_idempotent_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            target = Path(directory) / "legacy"
            target.mkdir()
            link = root / "datasets/source"
            self.assertEqual(
                environment.ensure_link(link, target, root), "created"
            )
            self.assertEqual(
                environment.ensure_link(link, target, root), "existing"
            )
            other = Path(directory) / "other"
            other.mkdir()
            with self.assertRaises(environment.EnvironmentError):
                environment.ensure_link(link, other, root)

    def test_broken_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            root.mkdir()
            link = root / "broken"
            link.symlink_to(Path(directory) / "missing")
            target = Path(directory) / "target"
            target.mkdir()
            with self.assertRaises(environment.EnvironmentError):
                environment.ensure_link(link, target, root)


class EnvironmentManifestTests(unittest.TestCase):
    def test_manifest_round_trip_and_schema_check(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = environment.canonical_layout(Path(directory) / "quadra")
            environment._prepare_layout(layout)
            value = {
                "schema_version": environment.SCHEMA_VERSION,
                "profiles": {},
            }
            environment.atomic_write_json(
                environment._manifest_path(layout), value
            )
            self.assertEqual(
                environment.load_environment_manifest(layout), value
            )
            value["schema_version"] += 1
            environment.atomic_write_json(
                environment._manifest_path(layout), value
            )
            with self.assertRaises(environment.EnvironmentError):
                environment.load_environment_manifest(layout)

    def test_profile_version_rules(self):
        preprocess = {
            "python": "3.11.8",
            "torch": "2.4.0",
            "mmcv": "unavailable",
            "mmdet": "unavailable",
            "totalsegmentator": "2.16.0",
        }
        self.assertEqual(
            environment.profile_runtime_errors("preprocess", preprocess), []
        )
        uae = {
            "python": "3.7.10",
            "torch": "1.9.0+cu111",
            "mmcv": "1.3.8",
            "mmdet": "2.14.0",
            "totalsegmentator": "unavailable",
        }
        self.assertEqual(environment.profile_runtime_errors("uae", uae), [])
        self.assertTrue(
            environment.profile_runtime_errors("uae", preprocess)
        )

    def test_superpoint_constants_are_pinned(self):
        self.assertEqual(len(environment.SUPERPOINT_COMMIT), 40)
        self.assertEqual(len(environment.SUPERPOINT_CHECKPOINT_SHA256), 64)
        for digest in environment.PROFILE_EXPECTED_DIGESTS.values():
            self.assertTrue(digest.startswith("sha256:"))
            self.assertEqual(len(digest), 71)

    def test_persistent_repository_clone_is_idempotent_and_rejects_dirty_target(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            subprocess.check_call(["git", "-C", str(source), "init", "-b", "feature"])
            subprocess.check_call(
                ["git", "-C", str(source), "config", "user.email", "test@example.invalid"]
            )
            subprocess.check_call(
                ["git", "-C", str(source), "config", "user.name", "Test"]
            )
            (source / "README.md").write_text("test\n", encoding="utf-8")
            subprocess.check_call(["git", "-C", str(source), "add", "README.md"])
            subprocess.check_call(["git", "-C", str(source), "commit", "-m", "test"])
            storage = Path(directory) / "workspace/quadra"
            target, branch, commit = environment.ensure_persistent_repository(
                source, storage
            )
            self.assertEqual(branch, "feature")
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(target), "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                commit,
            )
            environment.ensure_persistent_repository(source, storage)
            (target / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(environment.EnvironmentError):
                environment.ensure_persistent_repository(source, storage)

    def test_activation_script_contains_both_profiles_and_persistent_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = environment.canonical_layout(Path(directory) / "quadra")
            environment._prepare_layout(layout)
            repository = Path(directory) / "repos/uae-quadra-validation"
            script = environment._write_activation_script(layout, repository)
            content = script.read_text(encoding="utf-8")
            self.assertIn("preprocess", content)
            self.assertIn("uae", content)
            self.assertIn("QUADRA_STORAGE_ROOT", content)
            self.assertIn("QUADRA_TOTALSEG_OUTPUT_ROOT", content)


class EnvironmentAssetTests(unittest.TestCase):
    def test_copy_asset_is_atomic_and_rejects_changed_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            destination = Path(directory) / "models/model.bin"
            source.write_bytes(b"model")
            first = environment._copy_verified_asset(source, destination)
            self.assertEqual(first["status"], "copied")
            second = environment._copy_verified_asset(source, destination)
            self.assertEqual(second["status"], "existing")
            destination.write_bytes(b"different")
            with self.assertRaises(environment.EnvironmentError):
                environment._copy_verified_asset(source, destination)

    def test_verify_assets_reports_required_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = environment.canonical_layout(Path(directory) / "quadra")
            environment._prepare_layout(layout)
            result = environment.verify_assets(layout, "uae")
            self.assertFalse(result["ok"])
            self.assertIn("uae_s_checkpoint", result["required"])

    def test_cache_tree_copy_is_idempotent_and_detects_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy"
            destination = Path(directory) / "quadra/cache"
            source.mkdir()
            (source / "weights.bin").write_bytes(b"weights")
            first = environment._copy_tree_once(source, destination)
            self.assertEqual(first["status"], "copied")
            second = environment._copy_tree_once(source, destination)
            self.assertEqual(second["status"], "existing")
            (destination / "extra.bin").write_bytes(b"extra")
            with self.assertRaises(environment.EnvironmentError):
                environment._copy_tree_once(source, destination)


class EnvironmentCliTests(unittest.TestCase):
    def test_help_and_required_profile(self):
        parser = environment.build_parser()
        args = parser.parse_args(
            ["verify-assets", "--storage-root", "/tmp/quadra"]
        )
        self.assertIsNone(args.profile)
        with self.assertRaises(SystemExit):
            parser.parse_args(["bootstrap"])


if __name__ == "__main__":
    unittest.main()
