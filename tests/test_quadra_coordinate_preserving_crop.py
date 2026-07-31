import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from tools.quadra import body_envelope_audit as stage1
from tools.quadra import coordinate_preserving_crop as crop


def make_nifti_and_plan(
    root: Path,
    *,
    scan_key: str = "quadra_hc_044-test",
    subject_id: str = "quadra_hc_044",
    session: str = "test",
    shape=(16, 18, 8),
    spacing=(1.5, 2.0, 2.5),
    crop_start=(2, 3, 0),
    crop_end=(14, 17, 8),
):
    root.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = np.diag([-spacing[0], -spacing[1], spacing[2]])
    affine[:3, 3] = [100.0, 200.0, -50.0]
    coordinates = np.indices(shape, dtype=np.float32)
    data = -800.0 + coordinates[0] * 10.0 + coordinates[1] + coordinates[2] * 0.1
    path = root / f"{session}_CT-AC.nii.gz"
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, str(path))
    loaded = nib.load(str(path))
    raw_affine = np.asarray(loaded.affine, dtype=float)
    native_shape = np.asarray(shape, dtype=np.int64)
    start = np.asarray(crop_start, dtype=np.int64)
    end = np.asarray(crop_end, dtype=np.int64)
    crop_shape = end - start
    target_shape = stage1.torchio_target_shape(crop_shape, spacing, (2.0, 2.0, 2.0))
    lower, upper, padded_shape = stage1.symmetric_stride_padding(
        target_shape, crop.MODEL_STRIDE_XYZ
    )
    native_crop_affine = stage1.crop_affine(raw_affine, start)
    resampled_affine, padded_affine = stage1.resampled_and_padded_affines(
        native_crop_affine,
        crop_shape,
        target_shape,
        lower,
        (2.0, 2.0, 2.0),
    )
    raw_to_model = np.linalg.inv(padded_affine) @ raw_affine
    source_identity = crop.file_identity(path)
    source = {
        **source_identity,
        "native_shape_xyz": native_shape.tolist(),
        "spacing_xyz_mm": list(spacing),
        "affine": raw_affine.tolist(),
    }
    plan = {
        "subject_id": subject_id,
        "session": session,
        "scan_key": scan_key,
        "axis_policy": "xy",
        "margin_mm": 10.0,
        "source_ct": source,
        "body_envelope": {"start_xyz": start.tolist(), "end_xyz": end.tolist()},
        "crop_start_xyz": start.tolist(),
        "crop_end_xyz": end.tolist(),
        "crop_shape_xyz": crop_shape.tolist(),
        "target_shape_xyz": target_shape.tolist(),
        "padding_lower_xyz": lower.tolist(),
        "padding_upper_xyz": upper.tolist(),
        "padded_shape_xyz": padded_shape.tolist(),
        "model_tensor_shape_zyx": padded_shape[::-1].tolist(),
        "padded_2mm_voxels": int(np.prod(padded_shape)),
        "native_voxel_reduction_fraction": 0.25,
        "padded_2mm_voxel_reduction_fraction": 0.25,
        "artificial_boundaries": {
            "lower_xyz": [True, True, False],
            "upper_xyz": [True, True, False],
        },
        "minimum_artificial_mask_clearance_mm": 30.0,
        "native_crop_affine": native_crop_affine.tolist(),
        "resampled_2mm_affine": resampled_affine.tolist(),
        "padded_2mm_affine": padded_affine.tolist(),
        "raw_to_crop_continuous_affine": (
            np.linalg.inv(native_crop_affine) @ raw_affine
        ).tolist(),
        "crop_to_raw_continuous_affine": (
            np.linalg.inv(np.linalg.inv(native_crop_affine) @ raw_affine)
        ).tolist(),
        "crop_to_model_continuous_affine": (
            np.linalg.inv(padded_affine) @ native_crop_affine
        ).tolist(),
        "model_to_crop_continuous_affine": (
            np.linalg.inv(np.linalg.inv(padded_affine) @ native_crop_affine)
        ).tolist(),
        "raw_to_model_continuous_affine": raw_to_model.tolist(),
        "model_to_raw_continuous_affine": np.linalg.inv(raw_to_model).tolist(),
        "original_fov_limitations": [],
    }
    return path, plan


def selected_payload(template_plan):
    plans = []
    for number in range(21, 49):
        for session in ("test", "retest"):
            plan = json.loads(json.dumps(template_plan))
            plan["subject_id"] = f"quadra_hc_{number:03d}"
            plan["session"] = session
            plan["scan_key"] = f"quadra_hc_{number:03d}-{session}"
            plans.append(plan)
    lookup = {plan["scan_key"]: plan for plan in plans}
    pair_scans = [lookup["quadra_hc_044-retest"], lookup["quadra_hc_044-test"]]
    return {
        "schema_version": stage1.SCHEMA_VERSION,
        "status": "selected",
        "candidate_id": "xy_m010",
        "candidate_summary": {
            "candidate_id": "xy_m010",
            "axis_policy": "xy",
            "margin_mm": "10.0",
            "scans": "56",
            "total_clipped_mask_voxels": "0",
            "coordinate_roundtrip_passed": "True",
            "stride_compatible": "True",
            "minimum_artificial_mask_clearance_mm": "30.46875",
            "clearance_gate_passed": "True",
            "eligible": "True",
        },
        "scan_plans": plans,
        "largest_single_scan": lookup["quadra_hc_044-test"],
        "largest_test_retest_pair": {
            "subject_id": "quadra_hc_044",
            "scans": pair_scans,
        },
        "audit_manifest": {"path": "/audit.json", "bytes": 1, "sha256": "a" * 64},
    }


class StrictContractTests(unittest.TestCase):
    def test_string_summary_is_converted_to_native_types(self):
        summary = crop.typed_candidate_summary(
            {
                "candidate_id": "xy_m010",
                "axis_policy": "xy",
                "margin_mm": "10.0",
                "scans": "56",
                "total_clipped_mask_voxels": "0",
                "coordinate_roundtrip_passed": "True",
                "stride_compatible": "true",
                "minimum_artificial_mask_clearance_mm": "30.46875",
                "clearance_gate_passed": "1",
                "eligible": "yes",
            }
        )
        self.assertIs(summary["eligible"], True)
        self.assertEqual(summary["margin_mm"], 10.0)
        self.assertEqual(summary["scans"], 56)

    def test_false_string_is_not_truthy(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = make_nifti_and_plan(Path(directory))
            selected = selected_payload(plan)
            selected["candidate_summary"]["eligible"] = "False"
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "safety gate"):
                crop.validate_selected_payload(selected)

    def test_selected_payload_rejects_duplicate_and_wrong_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = make_nifti_and_plan(Path(directory))
            selected = selected_payload(plan)
            selected["scan_plans"][1]["scan_key"] = selected["scan_plans"][0]["scan_key"]
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "duplicate"):
                crop.validate_selected_payload(selected)

            selected = selected_payload(plan)
            selected["largest_test_retest_pair"]["subject_id"] = "quadra_hc_021"
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "largest pair"):
                crop.validate_selected_payload(selected)

    def test_scan_plan_rejects_transform_and_half_open_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = make_nifti_and_plan(Path(directory))
            broken = json.loads(json.dumps(plan))
            broken["crop_end_xyz"][0] -= 1
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "Crop shape"):
                crop._validate_scan_plan(broken)

            broken = json.loads(json.dumps(plan))
            broken["raw_to_model_continuous_affine"][0][3] += 1
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "Raw-to-model"):
                crop._validate_scan_plan(broken)


class GeometryHelperTests(unittest.TestCase):
    def test_ras_lps_geometry_round_trip_with_oblique_direction(self):
        import SimpleITK as sitk

        angle = np.deg2rad(20.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([1.5, 2.0, 2.5])
        affine[:3, 3] = [10.0, -20.0, 30.0]
        spacing, origin, direction = crop.ras_affine_to_sitk_geometry(affine)
        image = sitk.Image([4, 5, 6], sitk.sitkFloat32)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        image.SetDirection(direction)
        np.testing.assert_allclose(crop.sitk_geometry_to_ras_affine(image), affine, atol=1e-8)

    def test_xyz_zyx_order_and_continuous_roundtrip(self):
        points = np.array([[1.25, 2.5, 3.75], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(crop.zyx_to_xyz(crop.xyz_to_zyx(points)), points)
        affine = np.eye(4)
        affine[:3, 3] = [4.5, -2.0, 1.0]
        recovered = crop.apply_affine_xyz(crop.apply_affine_xyz(points, affine), np.linalg.inv(affine))
        np.testing.assert_allclose(recovered, points, atol=1e-12)

    def test_nearest_index_rounds_only_at_export_and_rejects_out_of_bounds(self):
        point = np.array([1.49, 2.5, 3.51])
        np.testing.assert_array_equal(
            crop.nearest_raw_indices(point, [5, 5, 5]), [1, 2, 4]
        )
        with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "outside"):
            crop.nearest_raw_indices([-0.6, 1.0, 1.0], [5, 5, 5])

    def test_intensity_normalization_and_padding_values(self):
        values = np.array([-2000.0, -1024.0, 0.0, 3071.0, 4000.0], dtype=np.float32)
        crop.normalize_ct_inplace(values)
        np.testing.assert_allclose(values[[0, 1]], [-50.0, -50.0], atol=1e-6)
        self.assertAlmostEqual(float(values[3]), 205.0, places=5)
        self.assertAlmostEqual(float(values[4]), 205.0, places=5)

    def test_padding_check_handles_odd_even_and_zero_padding(self):
        data = np.full((8, 16, 16), -1024.0, dtype=np.float32)
        maximum, count = crop._padding_max_error(data, [3, 2, 1], [4, 2, 1], -1024.0)
        self.assertEqual(maximum, 0.0)
        self.assertGreater(count, 0)
        maximum, count = crop._padding_max_error(data, [0, 0, 0], [0, 0, 0], -1024.0)
        self.assertEqual((maximum, count), (0.0, 0))


class PreparedVolumeTests(unittest.TestCase):
    def test_prepare_scan_matches_frozen_geometry_and_discards_no_last_voxel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, plan = make_nifti_and_plan(
                root,
                shape=(18, 18, 8),
                spacing=(2.0, 2.0, 2.0),
                crop_start=(1, 1, 0),
                crop_end=(17, 17, 8),
            )
            image = nib.load(str(path))
            data = np.asanyarray(image.dataobj).copy()
            data[16, 16, 7] = 3071.0
            data[17, 17, 7] = -1024.0
            nib.save(nib.Nifti1Image(data, image.affine, image.header), str(path))
            plan["source_ct"].update(crop.file_identity(path))
            prepared = crop.prepare_scan_from_plan(path, plan)
            self.assertEqual(prepared.data_zyx.dtype, np.float32)
            self.assertTrue(prepared.data_zyx.flags.c_contiguous)
            self.assertEqual(
                prepared.data_zyx.shape,
                tuple(plan["model_tensor_shape_zyx"]),
            )
            self.assertEqual(
                prepared.tensor_shape_ncdhw,
                (1, 1, *plan["model_tensor_shape_zyx"]),
            )
            np.testing.assert_allclose(
                prepared.model_affine_ras,
                plan["padded_2mm_affine"],
                atol=crop.AFFINE_ATOL,
            )
            self.assertEqual(prepared.metadata["crop_shape_xyz"], [16, 16, 8])
            self.assertEqual(prepared.metadata["crop_end_xyz"], [17, 17, 8])
            self.assertAlmostEqual(float(prepared.data_zyx[7, 15, 15]), 205.0, places=5)
            self.assertEqual(prepared.metadata["normalized_padding_max_error"], 0.0)
            self.assertFalse(prepared.metadata["cuda_used"])
            self.assertFalse(prepared.metadata["model_loaded"])

    def test_prepare_rejects_changed_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, plan = make_nifti_and_plan(root)
            plan["source_ct"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "identity mismatch"):
                crop.prepare_scan_from_plan(path, plan)

    def test_identity_comparison_rejects_wrong_selected_hash(self):
        observed = {"path": "/selected.json", "bytes": 10, "sha256": "a" * 64}
        recorded = dict(observed, sha256="b" * 64)
        with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "sha256"):
            crop._identity_matches(recorded, observed, "selected-body-envelope")

    def test_realized_coordinate_rows_pass_and_remain_in_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = make_nifti_and_plan(Path(directory))
            prepared = crop.prepare_scan_from_plan(path, plan)
            rows = crop.coordinate_check_rows(prepared, plan)
            self.assertEqual(len(rows), 11)
            self.assertLessEqual(
                max(row["max_raw_voxel_error"] for row in rows),
                crop.ROUNDTRIP_VOXEL_ATOL,
            )
            self.assertLessEqual(
                max(row["physical_error_mm"] for row in rows),
                crop.ROUNDTRIP_PHYSICAL_ATOL_MM,
            )


class RunSafetyTests(unittest.TestCase):
    def test_run_directory_refuses_overwrite_and_supports_incomplete_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / "quadra"
            output = storage / "runs/memory_optimization"
            output.mkdir(parents=True)
            created, resuming = crop._run_directory(
                storage, output, "stage2-crop-test", None
            )
            self.assertFalse(resuming)
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "overwrite"):
                crop._run_directory(storage, output, "stage2-crop-test", None)
            resumed, resuming = crop._run_directory(storage, output, None, created)
            self.assertTrue(resuming)
            self.assertEqual(resumed, created)

    def test_resume_contract_rejects_changes_and_completion(self):
        base = {
            "schema_version": 1,
            "preparation_id": crop.PREPARATION_ID,
            "status": "in_progress",
            "baseline_manifest": {"sha256": "a"},
            "stage1_checkpoint": {"sha256": "b"},
            "selected_body_envelope": {"sha256": "c"},
            "repository": {"execution_commit": "d"},
            "settings": {"candidate": "xy_m010"},
            "largest_pair": {"subject_id": "quadra_hc_044"},
            "scientific_computation": {"model": False},
        }
        crop.validate_resume_contract(dict(base), dict(base))
        changed = json.loads(json.dumps(base))
        changed["settings"]["candidate"] = "different"
        with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "changed"):
            crop.validate_resume_contract(base, changed)
        completed = dict(base)
        completed["status"] = "passed"
        with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "immutable"):
            crop.validate_resume_contract(completed, base)

    def test_git_ancestry_rejects_unrelated_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            subprocess.check_call(["git", "init", "-b", "main", str(repository)])
            subprocess.check_call(["git", "-C", str(repository), "config", "user.email", "test@example.invalid"])
            subprocess.check_call(["git", "-C", str(repository), "config", "user.name", "Stage Two Test"])
            (repository / "file.txt").write_text("one\n", encoding="utf-8")
            subprocess.check_call(["git", "-C", str(repository), "add", "file.txt"])
            subprocess.check_call(["git", "-C", str(repository), "commit", "-m", "one"])
            commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip()
            crop._require_git_ancestor(repository, commit)
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "does not descend"):
                crop._require_git_ancestor(repository, "0" * 40)

    def test_no_full_volume_outputs_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            crop._assert_no_full_volume_outputs(root)
            (root / "prepared.npy").write_bytes(b"data")
            with self.assertRaisesRegex(crop.CoordinatePreservingCropError, "Forbidden"):
                crop._assert_no_full_volume_outputs(root)

    def test_main_rejects_run_id_and_resume_together(self):
        with mock.patch.object(
            crop,
            "run_validate",
        ) as run_validate, self.assertRaises(SystemExit) as raised:
            crop.main(
                [
                    "validate",
                    "--run-id",
                    "stage2-crop-test",
                    "--resume-run-directory",
                    "/tmp/stage2-crop-test",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        run_validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
