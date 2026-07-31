import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.quadra import body_envelope_audit as audit


class BodyEnvelopeGeometryTests(unittest.TestCase):
    def test_half_open_bounds_include_last_foreground_voxel(self):
        mask = np.zeros((8, 9, 10), dtype=bool)
        mask[2:6, 3:8, 4:10] = True
        start, end = audit.half_open_bounds(mask)
        np.testing.assert_array_equal(start, [2, 3, 4])
        np.testing.assert_array_equal(end, [6, 8, 10])
        self.assertTrue(mask[tuple(end - 1)])

    def test_component_filter_discards_small_noise_and_keeps_detached_regions(self):
        volume = np.full((20, 20, 20), -1000.0, dtype=np.float32)
        volume[2:7, 2:7, 2:7] = 0
        volume[12:16, 12:16, 12:16] = 0
        volume[19, 19, 19] = 0
        foreground, stats = audit.build_conservative_foreground(
            volume,
            (1.0, 1.0, 1.0),
            minimum_component_volume_ml=0.05,
        )
        self.assertTrue(foreground[3, 3, 3])
        self.assertTrue(foreground[13, 13, 13])
        self.assertFalse(foreground[19, 19, 19])
        self.assertEqual(stats["components_retained"], 2)

    def test_margin_is_physical_and_xy_preserves_z(self):
        start, end = audit.expand_bounds(
            (10, 20, 30),
            (90, 100, 110),
            (120, 130, 140),
            (1.0, 2.0, 5.0),
            axis_policy="xy",
            margin_mm=20.0,
        )
        np.testing.assert_array_equal(start, [0, 10, 0])
        np.testing.assert_array_equal(end, [110, 110, 140])

        start_xyz, end_xyz = audit.expand_bounds(
            (10, 20, 30),
            (90, 100, 110),
            (120, 130, 140),
            (1.0, 2.0, 5.0),
            axis_policy="xyz",
            margin_mm=20.0,
        )
        np.testing.assert_array_equal(start_xyz, [0, 10, 26])
        np.testing.assert_array_equal(end_xyz, [110, 110, 114])

    def test_target_shape_and_stride_padding_match_expected_geometry(self):
        target = audit.torchio_target_shape(
            (512, 512, 531),
            (1.5234375, 1.5234375, 2.0),
        )
        np.testing.assert_array_equal(target, [390, 390, 531])
        lower, upper, padded = audit.symmetric_stride_padding(target)
        np.testing.assert_array_equal(padded, [400, 400, 532])
        np.testing.assert_array_equal(lower + upper, padded - target)
        self.assertTrue(np.all(padded % np.array([16, 16, 4]) == 0))

    def test_crop_and_model_coordinate_roundtrip(self):
        raw_affine = np.array(
            [[1.5, 0, 0, 10], [0, 1.5, 0, 20], [0, 0, 2.0, 30], [0, 0, 0, 1]],
            dtype=float,
        )
        crop = audit.crop_affine(raw_affine, (10, 12, 14))
        target_shape = audit.torchio_target_shape((80, 90, 100), (1.5, 1.5, 2.0))
        lower, _, _ = audit.symmetric_stride_padding(target_shape)
        _, padded = audit.resampled_and_padded_affines(
            crop, (80, 90, 100), target_shape, lower
        )
        result = audit.coordinate_roundtrip(
            raw_affine,
            crop,
            padded,
            ([10, 12, 14], [89, 101, 113], [50.25, 55.5, 60.75]),
        )
        self.assertTrue(result["passed"])
        self.assertLessEqual(result["max_raw_voxel_roundtrip_error"], 1e-6)
        np.testing.assert_allclose(
            result["raw_to_crop_continuous_affine"][:3][0],
            [1.0, 0.0, 0.0, -10.0],
        )

    def test_mask_clipping_and_artificial_clearance(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[3:8, 4:9, 5:10] = True
        metrics = audit.mask_candidate_metrics(
            mask,
            (3, 4, 5),
            (8, 9, 10),
            (2, 2, 2),
            (9, 15, 15),
            (20, 20, 20),
            (2.0, 2.0, 2.0),
        )
        self.assertEqual(metrics["outside_crop_voxels"], 0)
        self.assertEqual(metrics["minimum_artificial_clearance_mm"], 2.0)

        clipped = audit.mask_candidate_metrics(
            mask,
            (3, 4, 5),
            (8, 9, 10),
            (5, 2, 2),
            (9, 15, 15),
            (20, 20, 20),
            (2.0, 2.0, 2.0),
        )
        self.assertGreater(clipped["outside_crop_voxels"], 0)

    def test_geometry_matches_torchio_simpleitk_reference_construction(self):
        import SimpleITK as sitk

        native_shape = np.array([16, 18, 20])
        native_spacing = np.array([1.5, 2.5, 3.0])
        target_spacing = np.array([2.0, 2.0, 2.0])
        native_origin = np.array([10.0, 20.0, 30.0])
        affine = np.diag([*native_spacing, 1.0])
        affine[:3, 3] = native_origin

        image = sitk.Image(native_shape.tolist(), sitk.sitkFloat32)
        image.SetSpacing(native_spacing.tolist())
        image.SetOrigin(native_origin.tolist())
        new_origin_index = 0.5 * (target_spacing / native_spacing - 1)
        torchio_reference_origin = np.asarray(
            image.TransformContinuousIndexToPhysicalPoint(new_origin_index.tolist())
        )
        torchio_reference_shape = np.ceil(
            native_shape * native_spacing / target_spacing
        ).astype(int)

        observed_shape = audit.torchio_target_shape(
            native_shape,
            native_spacing,
            target_spacing,
        )
        observed_affine, _ = audit.resampled_and_padded_affines(
            affine,
            native_shape,
            observed_shape,
            (0, 0, 0),
            target_spacing,
        )
        np.testing.assert_array_equal(observed_shape, torchio_reference_shape)
        np.testing.assert_allclose(
            observed_affine[:3, 3],
            torchio_reference_origin,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.linalg.norm(observed_affine[:3, :3], axis=0),
            target_spacing,
            atol=1e-8,
        )

    def test_process_scan_streams_masks_and_builds_all_candidates(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ct_path = root / "test_CT-AC.nii.gz"
            mask_directory = root / "masks"
            mask_directory.mkdir()
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            ct = np.full((24, 24, 24), -1000, dtype=np.int16)
            ct[4:20, 4:20, 4:20] = 0
            ct_image = nib.Nifti1Image(ct, affine)
            ct_image.header.set_xyzt_units("mm")
            nib.save(ct_image, str(ct_path))
            mask = np.zeros_like(ct, dtype=np.uint8)
            mask[8:12, 9:13, 10:14] = 1
            nib.save(nib.Nifti1Image(mask, affine), str(mask_directory / "organ.nii.gz"))
            ct_hash = audit.sha256_file(ct_path)
            contract = {
                "baseline_manifest": {"sha256": "a" * 64},
                "cohort_manifest": {"sha256": "b" * 64},
                "settings": audit.audit_settings(),
            }
            scan = {
                "key": "quadra_hc_021-test",
                "subject_id": "quadra_hc_021",
                "session": "test",
                "sex": "F",
                "expected_masks": ["organ"],
                "expected_input_sha256": ct_hash,
                "ct_path": str(ct_path),
                "mask_directory": str(mask_directory),
            }
            result = audit.process_scan(scan, contract)
            self.assertEqual(result["mask_count"], 1)
            self.assertEqual(len(result["candidates"]), 12)
            self.assertTrue(
                all(candidate["clipped_mask_voxels"] == 0 for candidate in result["candidates"])
            )
            self.assertEqual(result["body_envelope"]["end_xyz"], [20, 20, 20])


class CandidateSummaryTests(unittest.TestCase):
    def _row(self, candidate, scan, voxels, clearance, eligible=True):
        return {
            "candidate_id": candidate,
            "axis_policy": candidate.split("_")[0],
            "margin_mm": float(candidate[-3:]),
            "scan_key": scan,
            "padded_2mm_voxels": voxels,
            "native_voxel_reduction_fraction": 0.25,
            "clipped_mask_voxels": 0,
            "coordinate_roundtrip_passed": True,
            "stride_compatible": True,
            "minimum_artificial_mask_clearance_mm": clearance,
            "eligible_for_scan": eligible,
            "padded_shape_xyz": "[16,16,4]",
        }

    def test_summary_applies_gates_and_ranks_worst_case_first(self):
        rows = []
        for index in range(2):
            rows.append(self._row("xy_m020", f"s{index}", 120 + index, 20.0))
            rows.append(self._row("xyz_m020", f"s{index}", 100 + index, 20.0))
        with mock.patch.object(audit, "EXPECTED_SCANS", 2):
            summaries = audit.summarize_candidates(rows)
        self.assertEqual(summaries[0]["candidate_id"], "xyz_m020")
        self.assertEqual(summaries[0]["eligible_rank"], 1)

        rows[1]["clipped_mask_voxels"] = 1
        with mock.patch.object(audit, "EXPECTED_SCANS", 2):
            summaries = audit.summarize_candidates(rows)
        failed = next(item for item in summaries if item["candidate_id"] == "xyz_m020")
        self.assertFalse(failed["eligible"])

    def test_report_contains_required_interpretation_sections(self):
        summaries = [
            {
                "candidate_id": "xy_m040",
                "eligible_rank": 1,
                "eligible": True,
                "total_clipped_mask_voxels": 0,
                "minimum_artificial_mask_clearance_mm": 40.0,
                "largest_padded_2mm_voxels": 100,
                "p95_padded_2mm_voxels": 90.0,
                "median_native_voxel_reduction_fraction": 0.3,
                "largest_scan_key": "quadra_hc_021-test",
            }
        ]
        report = audit.render_report(Path("/tmp/audit"), summaries, [], "xy_m040")
        for text in (
            "## Definitions",
            "## Candidate summary",
            "## Interpretation limits",
            "Selection frozen: **No**",
        ):
            self.assertIn(text, report)


class ContractAndResumeTests(unittest.TestCase):
    def test_baseline_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "accepted Stage 0"):
                audit.verify_baseline_identity(path)

    def test_atomic_create_and_run_directory_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "quadra"
            root.mkdir()
            output = root / "runs"
            path = root / "value.json"
            audit.atomic_create_json(path, {"value": 1})
            with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "overwrite"):
                audit.atomic_create_json(path, {"value": 2})
            created, resuming = audit._run_directory(
                root, output, "stage1-audit-test", None
            )
            self.assertFalse(resuming)
            with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "reuse"):
                audit._run_directory(root, output, "stage1-audit-test", None)
            resumed, resuming = audit._run_directory(root, output, None, created)
            self.assertTrue(resuming)
            self.assertEqual(resumed, created)

    def test_resume_requires_in_progress_identical_contract(self):
        contract = {
            "schema_version": 1,
            "audit_id": "audit",
            "status": "in_progress",
            "baseline_manifest": {"sha256": "a" * 64},
            "repository": {"execution_commit": "abc"},
            "cohort_manifest": {"sha256": "b" * 64},
            "settings": {"spacing": [2, 2, 2]},
            "denominators": {"scans": 56},
            "scientific_computation": {"uae_model_loaded": False},
        }
        audit.validate_resume_contract(dict(contract), contract)

        completed = dict(contract)
        completed["status"] = "passed"
        with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "immutable"):
            audit.validate_resume_contract(completed, contract)

        changed = dict(contract)
        changed["settings"] = {"spacing": [2.5, 2.5, 2.5]}
        with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "settings"):
            audit.validate_resume_contract(changed, contract)

    def test_cohort_denominators_and_missing_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ct_root = root / "ct"
            mask_root = root / "masks"
            ct = ct_root / "QUADRA_HC_021"
            masks = mask_root / "quadra_hc_021" / "test" / "masks"
            ct.mkdir(parents=True)
            masks.mkdir(parents=True)
            (ct / "test_CT-AC.nii.gz").touch()
            expected_masks = [f"mask_{index}" for index in range(2)]
            for name in expected_masks:
                (masks / f"{name}.nii.gz").touch()
            manifest = root / "cohort.json"
            manifest.write_text(
                json.dumps(
                    {
                        "scans": [
                            {
                                "subject_id": "quadra_hc_021",
                                "session": "test",
                                "sex": "F",
                                "expected_masks": expected_masks,
                                "input_sha256": "x",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.multiple(
                audit,
                EXPECTED_SCANS=1,
                EXPECTED_SUBJECTS=1,
                EXPECTED_MASKS=2,
                EXPECTED_MASKS_PER_SCAN_BY_SEX={"F": 2, "M": 3},
            ):
                _, scans = audit.load_cohort(manifest, ct_root, mask_root)
            self.assertEqual(scans[0]["key"], "quadra_hc_021-test")

            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_value["scans"][0]["sex"] = "M"
            manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
            with mock.patch.multiple(
                audit,
                EXPECTED_SCANS=1,
                EXPECTED_SUBJECTS=1,
                EXPECTED_MASKS=2,
                EXPECTED_MASKS_PER_SCAN_BY_SEX={"F": 2, "M": 3},
            ):
                with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "Expected 3"):
                    audit.load_cohort(manifest, ct_root, mask_root)

            manifest_value["scans"][0]["sex"] = "F"
            manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
            (masks / "mask_1.nii.gz").unlink()
            with mock.patch.multiple(
                audit,
                EXPECTED_SCANS=1,
                EXPECTED_SUBJECTS=1,
                EXPECTED_MASKS=2,
                EXPECTED_MASKS_PER_SCAN_BY_SEX={"F": 2, "M": 3},
            ):
                with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "Missing masks"):
                    audit.load_cohort(manifest, ct_root, mask_root)

    def test_audit_output_verification_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            names = {
                "scan_audit.csv",
                "mask_clearance.csv",
                "candidate_summary.csv",
                "audit_summary.json",
                "body_envelope_audit_report.md",
                "body_envelope_overview.png",
                "review_cases.json",
                "qc/largest_padded_scan.png",
                "qc/worst_clearance_scan.png",
                "stage1-audit.log",
            }
            outputs = {}
            for name in names:
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
                outputs[name] = audit.file_identity(path)
            audit.verify_audit_outputs(run, {"outputs": outputs})
            (run / "scan_audit.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "changed"):
                audit.verify_audit_outputs(run, {"outputs": outputs})

    def test_small_end_to_end_audit_writes_review_artifacts(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / "quadra"
            layout = audit.canonical_layout(storage)
            subject_ct = layout["whole_body_ct"] / "QUADRA_HC_021"
            subject_masks = (
                layout["totalsegmentator_outputs"]
                / "quadra_hc_021"
                / "test"
                / "masks"
            )
            subject_ct.mkdir(parents=True)
            subject_masks.mkdir(parents=True)
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            ct = np.full((24, 24, 24), -1000, dtype=np.int16)
            ct[4:20, 4:20, 4:20] = 0
            ct_image = nib.Nifti1Image(ct, affine)
            ct_image.header.set_xyzt_units("mm")
            ct_path = subject_ct / "test_CT-AC.nii.gz"
            nib.save(ct_image, str(ct_path))
            mask = np.zeros_like(ct, dtype=np.uint8)
            mask[8:12, 8:12, 8:12] = 1
            nib.save(
                nib.Nifti1Image(mask, affine),
                str(subject_masks / "organ.nii.gz"),
            )
            layout["manifests"].mkdir(parents=True)
            cohort_path = layout["manifests"] / audit.COHORT_MANIFEST_NAME
            cohort_path.write_text(
                json.dumps(
                    {
                        "scans": [
                            {
                                "subject_id": "quadra_hc_021",
                                "session": "test",
                                "sex": "F",
                                "expected_masks": ["organ"],
                                "input_sha256": audit.sha256_file(ct_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            baseline_path = root / "baseline.json"
            baseline_path.write_text("{}\n", encoding="utf-8")
            args = argparse.Namespace(
                storage_root=storage,
                baseline_manifest=baseline_path,
                repository_root=root,
                cohort_manifest=cohort_path,
                output_root=storage / "runs/memory_optimization",
                run_id="stage1-audit-test",
                resume_run_directory=None,
            )
            identity = audit.file_identity(baseline_path)
            with mock.patch.object(
                audit, "verify_baseline_identity", return_value=identity
            ), mock.patch.object(
                audit.baseline, "validate_locked_contract"
            ), mock.patch.object(
                audit,
                "current_repository_record",
                return_value={"branch": "test", "clean": True},
            ), mock.patch.multiple(
                audit,
                EXPECTED_SCANS=1,
                EXPECTED_SUBJECTS=1,
                EXPECTED_MASKS=1,
                EXPECTED_MASKS_PER_SCAN_BY_SEX={"F": 1, "M": 2},
                AXIS_POLICIES=("xy",),
                MARGINS_MM=(20.0,),
            ):
                run = audit.run_audit(args)
                summary = audit.load_json(run / "audit_summary.json")
                self.assertEqual(summary["status"], "PASS")
                self.assertFalse(summary["selection_frozen"])
                manifest = audit.load_json(run / "audit_manifest.json")
                audit.verify_audit_outputs(run, manifest)
                self.assertTrue((run / "qc/largest_padded_scan.png").is_file())


class SelectionTests(unittest.TestCase):
    @staticmethod
    def _candidate(identifier, voxels):
        return {
            "candidate_id": identifier,
            "axis_policy": "xy",
            "margin_mm": 20.0,
            "crop_start_xyz": [1, 1, 0],
            "crop_end_xyz": [9, 9, 10],
            "crop_shape_xyz": [8, 8, 10],
            "target_shape_xyz": [8, 8, 10],
            "padding_lower_xyz": [4, 4, 1],
            "padding_upper_xyz": [4, 4, 1],
            "padded_shape_xyz": [16, 16, 12],
            "model_tensor_shape_zyx": [12, 16, 16],
            "padded_2mm_voxels": voxels,
            "native_voxel_reduction_fraction": 0.2,
            "padded_2mm_voxel_reduction_fraction": 0.25,
            "artificial_boundaries": {
                "lower_xyz": [True, True, False],
                "upper_xyz": [True, True, False],
            },
            "minimum_artificial_mask_clearance_mm": 20.0,
            "native_crop_affine": np.eye(4).tolist(),
            "resampled_2mm_affine": np.eye(4).tolist(),
            "padded_2mm_affine": np.eye(4).tolist(),
            "coordinate_roundtrip": {
                "raw_to_crop_continuous_affine": np.eye(4).tolist(),
                "crop_to_raw_continuous_affine": np.eye(4).tolist(),
                "crop_to_model_continuous_affine": np.eye(4).tolist(),
                "model_to_crop_continuous_affine": np.eye(4).tolist(),
                "raw_to_model_continuous_affine": np.eye(4).tolist(),
                "model_to_raw_continuous_affine": np.eye(4).tolist(),
            },
        }

    def test_selection_rejects_ineligible_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "quadra/runs/memory_optimization/stage1-audit-test"
            run.mkdir(parents=True)
            baseline_path = root / "baseline.json"
            baseline_path.write_text("{}\n", encoding="utf-8")
            baseline_hash = audit.sha256_file(baseline_path)
            outputs = {}
            required = {
                "scan_audit.csv": "header\n",
                "mask_clearance.csv": "header\n",
                "audit_summary.json": json.dumps(
                    {
                        "status": "PASS",
                        "selection_frozen": False,
                        "scans_completed": 1,
                        "mask_candidate_rows": 1,
                    }
                ),
                "body_envelope_audit_report.md": "report\n",
                "body_envelope_overview.png": "png\n",
                "review_cases.json": "{}\n",
                "qc/largest_padded_scan.png": "png\n",
                "qc/worst_clearance_scan.png": "png\n",
                "stage1-audit.log": "log\n",
            }
            candidate_path = run / "candidate_summary.csv"
            with candidate_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["candidate_id", "eligible"]
                )
                writer.writeheader()
                writer.writerow({"candidate_id": "xy_m020", "eligible": False})
            required["candidate_summary.csv"] = candidate_path.read_text(encoding="utf-8")
            for name, value in required.items():
                path = run / name
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(value, encoding="utf-8")
                outputs[name] = audit.file_identity(path)
            manifest = {
                "status": "passed",
                "baseline_manifest": {
                    "path": str(baseline_path),
                    "sha256": baseline_hash,
                },
                "outputs": outputs,
            }
            audit.atomic_create_json(run / "audit_manifest.json", manifest)
            args = argparse.Namespace(
                audit_run_directory=run,
                candidate_id="xy_m020",
                review_rationale="reviewed",
                storage_root=root / "quadra",
                repository_root=root,
            )
            with mock.patch.object(audit, "EXPECTED_BASELINE_SHA256", baseline_hash), mock.patch.object(
                audit, "EXPECTED_BASELINE_PATH", baseline_path
            ), mock.patch.object(
                audit.baseline, "validate_locked_contract"
            ), mock.patch.multiple(
                audit,
                EXPECTED_SCANS=1,
                EXPECTED_MASKS=1,
                AXIS_POLICIES=("xy",),
                MARGINS_MM=(20.0,),
            ):
                with self.assertRaisesRegex(audit.BodyEnvelopeAuditError, "not eligible"):
                    audit.run_select(args)

    def test_selection_writes_largest_pair_and_refuses_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / "quadra"
            run = storage / "runs/memory_optimization/stage1-audit-test"
            results = run / "scan_results"
            results.mkdir(parents=True)
            baseline_path = root / "baseline.json"
            baseline_path.write_text("{}\n", encoding="utf-8")
            baseline_hash = audit.sha256_file(baseline_path)
            for session, voxels in (("test", 100), ("retest", 120)):
                result = {
                    "subject_id": "quadra_hc_021",
                    "session": session,
                    "scan_key": f"quadra_hc_021-{session}",
                    "ct": {
                        "path": f"/{session}.nii.gz",
                        "sha256": session,
                    },
                    "body_envelope": {"start_xyz": [1, 1, 0], "end_xyz": [9, 9, 10]},
                    "candidates": [self._candidate("xy_m020", voxels)],
                }
                audit.atomic_create_json(results / f"quadra_hc_021-{session}.json", result)

            candidate_path = run / "candidate_summary.csv"
            with candidate_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["candidate_id", "eligible", "axis_policy", "margin_mm"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate_id": "xy_m020",
                        "eligible": True,
                        "axis_policy": "xy",
                        "margin_mm": 20,
                    }
                )
            required = {
                "scan_audit.csv": "header\n",
                "mask_clearance.csv": "header\n",
                "audit_summary.json": json.dumps(
                    {
                        "status": "PASS",
                        "selection_frozen": False,
                        "scans_completed": 2,
                        "mask_candidate_rows": 2,
                    }
                ),
                "body_envelope_audit_report.md": "report\n",
                "body_envelope_overview.png": "png\n",
                "review_cases.json": "{}\n",
                "qc/largest_padded_scan.png": "png\n",
                "qc/worst_clearance_scan.png": "png\n",
                "stage1-audit.log": "log\n",
            }
            outputs = {"candidate_summary.csv": audit.file_identity(candidate_path)}
            for name, value in required.items():
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value, encoding="utf-8")
                outputs[name] = audit.file_identity(path)
            audit.atomic_create_json(
                run / "audit_manifest.json",
                {
                    "status": "passed",
                    "baseline_manifest": {
                        "path": str(baseline_path),
                        "sha256": baseline_hash,
                    },
                    "settings": audit.audit_settings(),
                    "outputs": outputs,
                },
            )
            args = argparse.Namespace(
                audit_run_directory=run,
                candidate_id="xy_m020",
                review_rationale="Reviewed candidate tables and overlays.",
                storage_root=storage,
                repository_root=root,
            )
            with mock.patch.object(audit, "EXPECTED_BASELINE_SHA256", baseline_hash), mock.patch.object(
                audit, "EXPECTED_BASELINE_PATH", baseline_path
            ), mock.patch.object(
                audit.baseline, "validate_locked_contract"
            ), mock.patch.multiple(
                audit,
                EXPECTED_SCANS=2,
                EXPECTED_MASKS=2,
                AXIS_POLICIES=("xy",),
                MARGINS_MM=(20.0,),
            ):
                selected_path = audit.run_select(args)
                selected = audit.load_json(selected_path)
                self.assertEqual(
                    selected["largest_test_retest_pair"]["subject_id"],
                    "quadra_hc_021",
                )
                self.assertEqual(
                    selected["largest_single_scan"]["scan_key"],
                    "quadra_hc_021-retest",
                )
                checkpoint = audit.load_json(run / "checkpoint_summary.json")
                self.assertEqual(checkpoint["status"], "PASS")
                with self.assertRaisesRegex(
                    audit.BodyEnvelopeAuditError, "overwrite"
                ):
                    audit.run_select(args)


if __name__ == "__main__":
    unittest.main()
