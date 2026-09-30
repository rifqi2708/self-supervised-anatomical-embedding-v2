import csv
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from tools.quadra.mask_review.core import (
    build_resegmentation_review_index,
    load_manifest as load_review_manifest,
)
from tools.quadra.totalsegmentator.core import sha256_file
from tools.quadra.totalsegmentator.resegmentation import (
    ResegmentationIntegrityError,
    build_resegmentation_command,
    prepare_resegmentation_manifest,
    resegmentation_preflight,
    run_correction_case,
    run_segmentation_case,
    stage_selected_inputs,
    verify_source_file,
)


class FlaggedResegmentationTests(unittest.TestCase):
    def _save_mask(self, path: Path, coordinates):
        data = np.zeros((5, 5, 5), dtype=np.uint8)
        for coordinate in coordinates:
            data[coordinate] = 1
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    def _fixture(self, root: Path):
        dataset = root / "dataset"
        ct = dataset / "Image_QUADRA_HC_WB/QUADRA_HC_001/test_CT-AC.nii.gz"
        ct.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4)), ct)
        mask_root = dataset / "Masks_QUADRA_HC_WB/quadra_hc_001/test/masks"
        mask_root.mkdir(parents=True)
        organs = {
            "sacrum": "requires_resegmentation",
            "vertebrae_C1": "requires_resegmentation",
            "rib_left_6": "requires_resegmentation",
            "kidney_left": "requires_resegmentation",
            "hip_left": "requires_correction",
        }
        masks = []
        for organ in organs:
            path = mask_root / f"{organ}.nii.gz"
            coordinates = [(1, 1, 1), (1, 1, 2)]
            if organ == "hip_left":
                coordinates.append((4, 4, 4))
            self._save_mask(path, coordinates)
            masks.append(
                {
                    "organ": organ,
                    "display_name": organ,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )

        review = root / "review"
        review.mkdir()
        decisions = review / "decisions.csv"
        with decisions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["item_id", "subject_id", "session", "organ", "decision", "note"],
            )
            writer.writeheader()
            for organ, decision in organs.items():
                writer.writerow(
                    {
                        "item_id": f"quadra_hc_001|test|{organ}",
                        "subject_id": "quadra_hc_001",
                        "session": "test",
                        "organ": organ,
                        "decision": decision,
                        "note": "synthetic",
                    }
                )
        review_manifest = {
            "schema_version": 1,
            "dataset_root": str(dataset),
            "scans": [
                {
                    "subject_id": "quadra_hc_001",
                    "session": "test",
                    "sex": "F",
                    "ct": {
                        "path": str(ct),
                        "sha256": sha256_file(ct),
                        "size": ct.stat().st_size,
                    },
                    "masks": masks,
                }
            ],
        }
        (review / "review_manifest.json").write_text(
            json.dumps(review_manifest), encoding="utf-8"
        )
        plan = {
            "schema_version": 1,
            "plan_id": "synthetic-plan",
            "totalsegmentator_version": "2.16.0",
            "source_review": {
                "expected_decisions_sha256": sha256_file(decisions),
                "expected_items": 5,
                "expected_flagged": 5,
                "expected_decisions": {
                    "requires_resegmentation": 4,
                    "requires_correction": 1,
                },
            },
            "vertebrae_outputs": ["vertebrae_C1", "vertebrae_C2"],
            "phases": {
                "sacrum_pilot": {
                    "kind": "segmentation",
                    "task": "total",
                    "robust_crop": True,
                    "gated": False,
                    "cases": [
                        {
                            "subject_id": "quadra_hc_001",
                            "sessions": ["test"],
                            "masks": ["sacrum"],
                        }
                    ],
                },
                "vertebrae_pilot": {
                    "kind": "segmentation",
                    "task": "vertebrae_pp_refined",
                    "robust_crop": False,
                    "gated": False,
                    "cases": [
                        {
                            "subject_id": "quadra_hc_001",
                            "sessions": ["test"],
                            "masks": ["vertebrae_C1"],
                        }
                    ],
                },
                "ribs": {
                    "kind": "segmentation",
                    "task": "total",
                    "robust_crop": True,
                    "gated": False,
                    "cases": [
                        {
                            "subject_id": "quadra_hc_001",
                            "sessions": ["test"],
                            "masks": ["rib_left_6"],
                        }
                    ],
                },
                "other_organs": {
                    "kind": "segmentation",
                    "task": "total",
                    "robust_crop": True,
                    "gated": False,
                    "cases": [
                        {
                            "subject_id": "quadra_hc_001",
                            "sessions": ["test"],
                            "masks": ["kidney_left"],
                        }
                    ],
                },
                "corrections": {
                    "kind": "isolated_voxel_correction",
                    "gated": False,
                    "connectivity": 26,
                    "expected_removed_voxels": 1,
                    "cases": [
                        {
                            "subject_id": "quadra_hc_001",
                            "sessions": ["test"],
                            "masks": ["hip_left"],
                        }
                    ],
                },
            },
            "expansions": {},
        }
        plan_path = root / "plan.yaml"
        plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
        manifest = prepare_resegmentation_manifest(review, plan_path)
        return dataset, review, plan_path, manifest

    def _fake_executable(self, root: Path) -> Path:
        path = root / "fake TotalSegmentator"
        path.write_text(
            textwrap.dedent(
                """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

args = sys.argv[1:]
input_path = Path(args[args.index('-i') + 1])
output = Path(args[args.index('-o') + 1])
task = args[args.index('-ta') + 1]
if task == 'total':
    start = args.index('--roi_subset') + 1
    stop = args.index('--robust_crop') if '--robust_crop' in args else args.index('--report')
    classes = args[start:stop]
else:
    classes = ['vertebrae_C1', 'vertebrae_C2']
report = Path(args[args.index('--report') + 1])
image = nib.load(str(input_path))
output.mkdir(parents=True, exist_ok=True)
report.parent.mkdir(parents=True, exist_ok=True)
for name in classes:
    data = np.zeros(image.shape, dtype=np.uint8)
    data[2, 2, 2] = 1
    nib.save(nib.Nifti1Image(data, image.affine), output / f'{name}.nii.gz')
report.write_text(json.dumps({'task': task, 'classes': classes}))
"""
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def test_prepare_freezes_cases_and_builds_safe_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, manifest = self._fixture(root)
            self.assertEqual(manifest["summary"]["cases"], 5)
            self.assertEqual(manifest["summary"]["segmentation_cases"], 4)
            sacrum = next(case for case in manifest["cases"] if case["phase"] == "sacrum_pilot")
            input_path = verify_source_file(dataset, sacrum["input"])
            command = build_resegmentation_command(sacrum, input_path, root / "work")
            self.assertIn("--roi_subset", command)
            self.assertIn("--robust_crop", command)
            self.assertNotIn("--fast", command)
            vertebra = next(
                case for case in manifest["cases"] if case["phase"] == "vertebrae_pilot"
            )
            command = build_resegmentation_command(vertebra, input_path, root / "work2")
            self.assertNotIn("--roi_subset", command)
            self.assertEqual(vertebra["generated_masks"], ["vertebrae_C1", "vertebrae_C2"])

    def test_segmentation_case_is_atomic_validated_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, manifest = self._fixture(root)
            case = next(case for case in manifest["cases"] if case["phase"] == "sacrum_pilot")
            executable = self._fake_executable(root)
            result = run_segmentation_case(
                manifest,
                case,
                dataset,
                root / "outputs",
                root / "scratch",
                str(executable),
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["qc"]["mask_count"], 1)
            resumed = run_segmentation_case(
                manifest,
                case,
                dataset,
                root / "outputs",
                root / "scratch",
                str(executable),
            )
            self.assertEqual(resumed["status"], "skipped")

            manifest_path = root / "execution-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            review_root = root / "pilot-review"
            summary = build_resegmentation_review_index(
                manifest_path,
                dataset,
                root / "outputs",
                review_root,
                ["sacrum_pilot"],
            )
            self.assertEqual(summary["review_items"], 1)
            self.assertEqual(summary["subjects_total"], 1)
            review_manifest = load_review_manifest(review_root)
            self.assertEqual(review_manifest["subject_order"], ["quadra_hc_001"])

    def test_correction_removes_only_one_strictly_isolated_voxel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, manifest = self._fixture(root)
            case = next(case for case in manifest["cases"] if case["phase"] == "corrections")
            result = run_correction_case(
                manifest,
                case,
                dataset,
                root / "outputs",
                root / "scratch",
            )
            self.assertEqual(result["status"], "completed")
            run_manifest = json.loads(
                (Path(result["output_directory"]) / "run_manifest.json").read_text()
            )
            self.assertEqual(run_manifest["removed_voxel_count"], 1)
            corrected = np.asanyarray(
                nib.load(
                    str(Path(result["output_directory"]) / "masks/hip_left.nii.gz")
                ).dataobj
            )
            self.assertEqual(int(corrected.sum()), 2)

    def test_changed_source_is_rejected_and_inputs_stage_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, manifest = self._fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            staged = stage_selected_inputs(manifest_path, dataset, root / "staged")
            self.assertEqual(staged["summary"]["files"], 1)
            staged_relative = Path(manifest["cases"][0]["input"]["relative_path"])
            self.assertTrue((root / "staged" / staged_relative).is_file())
            self.assertEqual(staged["files"][0]["relative_path"], str(staged_relative))
            ct = dataset / "Image_QUADRA_HC_WB/QUADRA_HC_001/test_CT-AC.nii.gz"
            ct.write_bytes(b"changed")
            with self.assertRaises(ResegmentationIntegrityError):
                verify_source_file(dataset, manifest["cases"][0]["input"])

    def test_local_preflight_defers_correction_only_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, manifest = self._fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = resegmentation_preflight(
                manifest_path,
                dataset,
                root / "outputs",
                skip_runtime=True,
                min_free_gib=0,
            )

            self.assertEqual(result["checks"]["inputs"]["scope"], "segmentation_cases")
            self.assertEqual(result["checks"]["inputs"]["files"], 1)
            self.assertEqual(
                result["checks"]["corrections"]["status"],
                "deferred_to_local_execution",
            )


if __name__ == "__main__":
    unittest.main()
