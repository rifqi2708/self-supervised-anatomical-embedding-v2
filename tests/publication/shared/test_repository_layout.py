"""Verify relocation preserves aliases, source identities and safe CLI entry points."""

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shared.repository import ROOT, source_path, checkout_module


class RepositoryLayoutTests(unittest.TestCase):
    def test_canonical_imports_share_error_classes(self):
        core = importlib.import_module("publication.masks.segmentation.core")
        qc = importlib.import_module("publication.masks.postprocessing.totalsegmentator_qc")
        self.assertIs(core.WorkflowError, qc.WorkflowError)

    def test_obsolete_scaffolding_is_absent(self):
        for relative in ("tools/quadra", "fine_tune", "ai", "misc", "notebooks",
                         "DEEDS_file", ".devcontainer", "shared/compat.py",
                         "tools/interfaces.py", "tools/utils.py", "train_loop.py"):
            with self.subTest(path=relative):
                self.assertFalse((ROOT/relative).exists())
        self.assertTrue((ROOT/"miscellaneous/.devcontainer/devcontainer.json").is_file())

    def test_source_registry_resolves_signature_members_and_pinned_references(self):
        for original in (
            "tools/quadra/reviewed_matching_contract.py",
            "tools/quadra/aligned_organ_group_cohort.py",
            "tools/quadra/memory_configuration_screen.py",
        ):
            self.assertTrue(source_path(original).is_file())
        from publication.uae.reviewed_uae_matching import reference_identity
        self.assertEqual(reference_identity()["commit"],
                         "035bbfc1816e40c122303a5e21accbdc35beac43")

    def test_explicit_frozen_checkout_keeps_legacy_requirements_access(self):
        original = "tools/quadra/environment/requirements-registration.txt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root/original
            legacy.parent.mkdir(parents=True)
            legacy.write_text("legacy frozen requirements\n")
            self.assertEqual(source_path(original, root=root), legacy)
            canonical = root/"publication/operations/requirements-registration.txt"
            canonical.parent.mkdir(parents=True)
            canonical.write_text("current requirements\n")
            self.assertEqual(source_path(original, root=root), canonical)

    def test_checkout_dispatch_selects_canonical_or_explicit_frozen_layout(self):
        self.assertEqual(checkout_module("tools.quadra.environment", ROOT),
                         "publication.operations.environment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member = root/"tools/quadra/environment.py"
            member.parent.mkdir(parents=True)
            member.write_text("# frozen source fixture\n")
            self.assertEqual(checkout_module("tools.quadra.environment", root),
                             "tools.quadra.environment")

    def test_execution_signature_hashes_members_and_fails_on_missing_source(self):
        from publication.shared import reviewed_evidence as evidence
        from unittest import mock
        from publication.uae import aligned_organ_group_cohort as cohort
        self.assertNotEqual(evidence.execution_signature(), cohort.sha256_payload({}))
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(evidence, "source_path", return_value=Path(directory)/"missing.py"):
                with self.assertRaises(FileNotFoundError):
                    evidence.execution_signature()

    def test_canonical_direct_cli_paths_work_outside_repository_cwd(self):
        for path in (
            "publication/operations/environment.py",
            "publication/masks/postprocessing/__main__.py",
            "publication/masks/segmentation/__main__.py",
        ):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run([sys.executable, str(ROOT/path), "--help"],
                                        cwd=cwd, text=True, capture_output=True,
                                        timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
        for module in ("publication.operations.environment", "publication.masks.postprocessing",
                       "publication.masks.segmentation"):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run([sys.executable, "-m", module, "--help"],
                                        cwd=cwd, text=True, capture_output=True,
                                        env=dict(os.environ, PYTHONPATH=str(ROOT)), timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
