import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

from tools.quadra.mask_review.app import (
    PLANE_COLORS,
    WINDOWS,
    _render_plane,
    _shortcut_event_id,
)
from tools.quadra.mask_review.core import (
    ReviewError,
    build_review_index,
    display_to_voxel,
    export_review_state,
    first_pending_item_id,
    flatten_items,
    latest_checkpoint,
    load_decisions,
    load_manifest,
    mask_bbox,
    plane_projection,
    plane_slice,
    projection_bbox,
    save_decision,
    stepped_slice,
    subject_complete,
    verify_checkpoint,
    verify_identity,
    voxel_to_display,
)
from tools.quadra.totalsegmentator.core import (
    DEFAULT_REGISTRY,
    expected_mask_names,
    load_registry,
    sha256_file,
)


class MaskReviewGeometryTests(unittest.TestCase):
    def test_accept_shortcut_requires_new_event_for_current_item(self):
        payload = {
            "shortcut": "accept_next",
            "item_id": "quadra_hc_001|test|brain",
            "event_id": "event-1",
        }
        self.assertEqual(
            _shortcut_event_id(
                payload,
                current_item_id="quadra_hc_001|test|brain",
                processed_event_id=None,
            ),
            "event-1",
        )
        self.assertIsNone(
            _shortcut_event_id(
                payload,
                current_item_id="quadra_hc_001|test|brain",
                processed_event_id="event-1",
            )
        )
        self.assertIsNone(
            _shortcut_event_id(
                payload,
                current_item_id="quadra_hc_002|test|brain",
                processed_event_id=None,
            )
        )

    def test_click_coordinate_round_trip_for_all_planes(self):
        shape = (30, 40, 50)
        voxel = (11, 22, 33)
        indices = {"axial": voxel[2], "coronal": voxel[1], "sagittal": voxel[0]}
        for plane, index in indices.items():
            display_x, display_y = voxel_to_display(voxel, plane, shape)
            self.assertEqual(
                display_to_voxel(display_x, display_y, plane, index, shape),
                voxel,
            )

    def test_click_coordinate_is_clamped_to_volume(self):
        self.assertEqual(
            display_to_voxel(-100, 1000, "axial", 1000, (3, 4, 5)),
            (0, 0, 4),
        )

    def test_cropped_render_uses_projection_bbox_with_context(self):
        ct = np.zeros((30, 40, 50), dtype=np.int16)
        mask = np.zeros_like(ct, dtype=bool)
        mask[10:13, 18:22, 24:27] = True
        projection = plane_projection(mask, "axial")
        crop_x0, crop_y0, crop_x1, crop_y1, _ = projection_bbox(
            projection, margin=10
        )
        figure = _render_plane(
            ct,
            mask,
            "axial",
            25,
            WINDOWS["soft_tissue"],
            "fill_and_contour",
            0.35,
            boundary_margin=5,
            view_framing="cropped",
            crop_margin=10,
        )
        axis = figure.axes[0]
        self.assertEqual(axis.get_xlim(), (crop_x0 - 0.5, crop_x1 - 0.5))
        self.assertEqual(axis.get_ylim(), (crop_y1 - 0.5, crop_y0 - 0.5))
        plt.close(figure)

    def test_crosshair_lines_use_plane_specific_colors(self):
        ct = np.zeros((30, 40, 50), dtype=np.int16)
        mask = np.zeros_like(ct, dtype=bool)
        mask[10:13, 18:22, 24:27] = True
        expected = {
            "axial": ("sagittal", "coronal"),
            "coronal": ("sagittal", "axial"),
            "sagittal": ("coronal", "axial"),
        }
        indices = {"axial": 25, "coronal": 20, "sagittal": 11}
        for plane, index in indices.items():
            figure = _render_plane(
                ct,
                mask,
                plane,
                index,
                WINDOWS["soft_tissue"],
                "fill_and_contour",
                0.35,
                boundary_margin=5,
                view_framing="cropped",
                crop_margin=10,
                crosshair_voxel=(11, 20, 25),
            )
            vertical_plane, horizontal_plane = expected[plane]
            self.assertEqual(figure.axes[0].lines[0].get_color(), PLANE_COLORS[vertical_plane])
            self.assertEqual(figure.axes[0].lines[1].get_color(), PLANE_COLORS[horizontal_plane])
            plt.close(figure)

    def test_slice_step_is_clamped_to_volume(self):
        self.assertEqual(stepped_slice(4, -1, 9), 3)
        self.assertEqual(stepped_slice(4, 1, 9), 5)
        self.assertEqual(stepped_slice(0, -1, 9), 0)
        self.assertEqual(stepped_slice(9, 1, 9), 9)

    def test_bbox_expands_and_reports_clipped_margin(self):
        mask = np.zeros((8, 9, 10), dtype=bool)
        mask[0:3, 3:6, 4:8] = True
        bbox = mask_bbox(mask, margin=2)
        self.assertEqual(bbox["start"], [0, 3, 4])
        self.assertEqual(bbox["end"], [3, 6, 8])
        self.assertEqual(bbox["expanded_start"], [0, 1, 2])
        self.assertTrue(bbox["touches_volume_boundary"])
        self.assertTrue(bbox["margin_clipped"])
        self.assertEqual(bbox["boundary_axes"], [0])

    def test_plane_orientation_is_shared_by_ct_mask_and_projection(self):
        data = np.zeros((3, 4, 5), dtype=np.int16)
        data[1, 2, 3] = 9
        mask = data > 0
        for plane, index in (("sagittal", 1), ("coronal", 2), ("axial", 3)):
            ct_slice = plane_slice(data, plane, index)
            mask_slice = plane_slice(mask, plane, index)
            self.assertEqual(np.argwhere(ct_slice == 9).tolist(), np.argwhere(mask_slice).tolist())
            projection = plane_projection(mask, plane)
            self.assertTrue(projection.any())
            x0, y0, x1, y1, clipped = projection_bbox(projection, margin=1)
            self.assertLess(x0, x1)
            self.assertLess(y0, y1)
            self.assertIsInstance(clipped, bool)

    def test_empty_mask_is_rejected(self):
        with self.assertRaisesRegex(ReviewError, "empty"):
            mask_bbox(np.zeros((2, 2, 2), dtype=bool))


class MaskReviewWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = cls.root / "QUADRA_HC_WB"
        cls.review = cls.root / "review"
        cls._build_complete_fixture(cls.dataset)
        cls.summary = build_review_index(cls.dataset, cls.review, DEFAULT_REGISTRY)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _link(cls, source: Path, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)

    @classmethod
    def _build_complete_fixture(cls, dataset: Path):
        template_root = cls.root / "templates"
        template_root.mkdir(parents=True)
        affine = np.eye(4)
        ct_template = template_root / "ct.nii.gz"
        mask_template = template_root / "mask.nii.gz"
        nib.save(nib.Nifti1Image(np.arange(60, dtype=np.int16).reshape(3, 4, 5), affine), ct_template)
        mask = np.zeros((3, 4, 5), dtype=np.uint8)
        mask[1, 2, 3] = 1
        nib.save(nib.Nifti1Image(mask, affine), mask_template)
        registry = load_registry(DEFAULT_REGISTRY)
        for number in range(1, 49):
            subject = f"quadra_hc_{number:03d}"
            image_subject = dataset / "Image_QUADRA_HC_WB" / f"QUADRA_HC_{number:03d}"
            if number <= 20:
                mask_subject = (
                    dataset
                    / "Masks_QUADRA_HC_WB_001_020_totalsegmentator_2.16.0_organs_v1"
                    / subject
                )
                sex = "M" if number <= 11 else "F"
                manifest_name = "scan_manifest.json"
            else:
                mask_subject = dataset / "Masks_QUADRA_HC_WB" / subject
                sex = "M" if number <= 32 else "F"
                manifest_name = "run_manifest.json"
            expected = expected_mask_names(registry, sex)
            for session in ("test", "retest"):
                cls._link(ct_template, image_subject / f"{session}_CT-AC.nii.gz")
                scan_root = mask_subject / session
                for organ in expected:
                    cls._link(mask_template, scan_root / "masks" / f"{organ}.nii.gz")
                (scan_root / manifest_name).write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "completed": True,
                            "subject_id": subject,
                            "session": session,
                            "sex": sex,
                        }
                    ),
                    encoding="utf-8",
                )

    def test_complete_cohort_is_indexed_in_frozen_order(self):
        manifest = load_manifest(self.review)
        items = flatten_items(manifest)
        self.assertEqual(manifest["counts"]["subjects"], 48)
        self.assertEqual(manifest["counts"]["scans"], 96)
        self.assertEqual(manifest["counts"]["subjects_001_020_masks"], 1582)
        self.assertEqual(manifest["counts"]["subjects_021_048_masks"], 2208)
        self.assertEqual(len(items), 3790)
        self.assertEqual(items[0]["item_id"], "quadra_hc_001|test|brain")
        self.assertEqual(items[-1]["subject_id"], "quadra_hc_048")
        self.assertEqual(items[-1]["session"], "retest")
        self.assertEqual(
            next(item for item in items if item["subject_id"] == "quadra_hc_030")[
                "source_selection"
            ],
            "subjects_021_048_current_corrected",
        )

    def test_initial_outputs_and_checkpoint_exist(self):
        self.assertEqual(self.summary["review_items"], 3790)
        self.assertEqual(self.summary["counts"]["pending"], 3790)
        self.assertTrue((self.review / "events.jsonl").is_file())
        self.assertTrue((self.review / "decisions.csv").is_file())
        self.assertTrue((self.review / "flagged_cases.csv").is_file())
        self.assertTrue((self.review / "review_summary.json").is_file())
        checkpoints = list((self.review / "checkpoints").glob("*.json"))
        self.assertEqual(len(checkpoints), 1)

    def test_latest_checkpoint_supports_safe_continue(self):
        manifest = load_manifest(self.review)
        state = load_decisions(self.review)
        expected_resume = first_pending_item_id(manifest, state)
        exported = export_review_state(self.review, create_checkpoint=True)
        checkpoint = latest_checkpoint(self.review)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint["checkpoint_id"], exported["checkpoint"]["checkpoint_id"])
        self.assertEqual(checkpoint["resume_item_id"], expected_resume)
        self.assertEqual(checkpoint["counts"], exported["counts"])
        self.assertTrue(verify_checkpoint(self.review, checkpoint)["current"])

        save_decision(
            self.review,
            expected_resume,
            "acceptable",
            "",
            self._view_state(),
        )
        self.assertFalse(verify_checkpoint(self.review, checkpoint)["current"])

    def test_failure_decision_requires_note(self):
        with self.assertRaisesRegex(ReviewError, "note is required"):
            save_decision(
                self.review,
                "quadra_hc_001|test|brain",
                "requires_correction",
                "",
                self._view_state(),
            )

    def test_autosave_resume_and_flag_export(self):
        event = save_decision(
            self.review,
            "quadra_hc_001|test|brain",
            "requires_correction",
            "Superior boundary requires checking.",
            self._view_state(),
        )
        self.assertEqual(event["decision"], "requires_correction")
        resumed = load_decisions(self.review)
        row = resumed["quadra_hc_001|test|brain"]
        self.assertEqual(row["decision"], "requires_correction")
        self.assertEqual(row["axial_slice"], 3)
        with (self.review / "flagged_cases.csv").open(newline="", encoding="utf-8") as handle:
            flagged = list(csv.DictReader(handle))
        self.assertTrue(any(value["item_id"] == row["item_id"] for value in flagged))
        self.assertFalse(subject_complete(resumed, "quadra_hc_001"))

    def test_source_checksum_change_is_detected(self):
        manifest = load_manifest(self.review)
        identity = dict(flatten_items(manifest)[0]["mask"])
        verify_identity(identity)
        identity["sha256"] = "0" * 64
        with self.assertRaisesRegex(ReviewError, "checksum changed"):
            verify_identity(identity)

    def test_build_refuses_to_overwrite_review(self):
        with self.assertRaisesRegex(ReviewError, "Refusing to overwrite"):
            build_review_index(self.dataset, self.review, DEFAULT_REGISTRY)

    @staticmethod
    def _view_state():
        return {
            "axial_slice": 3,
            "coronal_slice": 2,
            "sagittal_slice": 1,
            "window_preset": "soft_tissue",
            "overlay_mode": "fill_and_contour",
            "opacity": 0.35,
            "view_framing": "cropped",
            "crop_margin": 20,
        }


if __name__ == "__main__":
    unittest.main()
