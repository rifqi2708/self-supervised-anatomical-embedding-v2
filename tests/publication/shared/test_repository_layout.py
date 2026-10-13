"""Verify relocation preserves aliases, source identities and safe CLI entry points."""

import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shared.repository import ROOT, source_path


class RepositoryLayoutTests(unittest.TestCase):
    def test_legacy_imports_share_one_implementation_and_error_classes(self):
        pairs = (
            ("tools.quadra.environment", "publication.operations.environment"),
            ("tools.quadra.totalsegmentator.core", "publication.masks.segmentation.core"),
            ("tools.quadra.totalsegmentator.qc", "publication.masks.postprocessing.totalsegmentator_qc"),
            ("tools.quadra.mask_review.core", "publication.masks.postprocessing.core"),
            ("tools.quadra.superpoint_adapter", "publication.query_points.superpoint.superpoint_adapter"),
        )
        for old, new in pairs:
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))

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

    def test_old_and_new_direct_cli_paths_work_outside_repository_cwd(self):
        for path in (
            "tools/quadra/environment.py",
            "publication/operations/environment.py",
            "tools/quadra/mask_review/__main__.py",
            "publication/masks/postprocessing/__main__.py",
            "tools/quadra/totalsegmentator/__main__.py",
            "publication/masks/segmentation/__main__.py",
        ):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run([sys.executable, str(ROOT/path), "--help"],
                                        cwd=cwd, text=True, capture_output=True,
                                        timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
