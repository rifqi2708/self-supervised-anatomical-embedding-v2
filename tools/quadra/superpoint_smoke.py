"""Run a bounded, single-slice SuperPoint smoke test on one Quadra CT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    import sys

    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from tools.quadra.superpoint_adapter import (  # noqa: E402
    DEFAULT_WINDOW_CENTER_HU,
    DEFAULT_WINDOW_WIDTH_HU,
    EXPECTED_SUPERPOINT_CHECKPOINT_SHA256,
    EXPECTED_SUPERPOINT_COMMIT,
    ensure_stride_compatible,
    load_aligned_mask_slices,
    load_axial_ct_slice,
    load_superpoint_model,
    native_xy_to_model_yx,
    run_superpoint_on_slice,
    window_and_normalize_ct,
    write_keypoints_csv_atomic,
    write_json_atomic,
    write_overlay_png_atomic,
)


SCHEMA_VERSION = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run SuperPoint on exactly one native-grid axial CT slice. "
            "This command never iterates over the complete volume."
        )
    )
    parser.add_argument("--ct", required=True, help="Input 3D CT NIfTI")
    parser.add_argument("--slice-index", required=True, type=int, help="Native axis-2 slice index")
    parser.add_argument("--superpoint-root", required=True, help="Pinned SuperPoint repository")
    parser.add_argument("--checkpoint", required=True, help="Pinned PyTorch SuperPoint checkpoint")
    parser.add_argument("--output-json", required=True, help="Technical smoke-test summary")
    parser.add_argument("--output-keypoints-csv", help="Exact model-pixel and native-voxel points")
    parser.add_argument("--output-overlay-png", help="Review image with points and mask contours")
    parser.add_argument("--mask-dir", help="Aligned native-grid organ-mask directory")
    parser.add_argument("--window-center", type=float, default=DEFAULT_WINDOW_CENTER_HU)
    parser.add_argument("--window-width", type=float, default=DEFAULT_WINDOW_WIDTH_HU)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def summarize_prediction(prediction):
    keypoints = prediction["keypoints_xy"]
    scores = prediction["scores"]
    descriptors = prediction["descriptors"]
    summary = {
        "keypoint_count": int(keypoints.shape[0]),
        "descriptor_shape": [int(value) for value in descriptors.shape],
        "outputs_finite": bool(
            np.isfinite(keypoints).all()
            and np.isfinite(scores).all()
            and np.isfinite(descriptors).all()
        ),
        "runtime_seconds": float(prediction["runtime_seconds"]),
        "peak_gpu_memory_bytes": prediction["peak_gpu_memory_bytes"],
    }
    if scores.size:
        summary["score"] = {
            "min": float(scores.min()),
            "median": float(np.median(scores)),
            "max": float(scores.max()),
        }
        summary["coordinate_bounds_xy"] = {
            "min": [float(value) for value in keypoints.min(axis=0)],
            "max": [float(value) for value in keypoints.max(axis=0)],
        }
    else:
        summary["score"] = None
        summary["coordinate_bounds_xy"] = None
    return summary


def main(argv=None):
    args = parse_args(argv)
    ct_slice_hu, ct_metadata = load_axial_ct_slice(args.ct, args.slice_index)
    normalized_native_xy = window_and_normalize_ct(
        ct_slice_hu,
        center=args.window_center,
        width=args.window_width,
    )
    model_image_yx = native_xy_to_model_yx(normalized_native_xy)
    ensure_stride_compatible(model_image_yx)
    model, model_provenance = load_superpoint_model(
        args.superpoint_root,
        args.checkpoint,
        device=args.device,
    )
    prediction = run_superpoint_on_slice(model, model_image_yx, model_provenance["device"])

    mask_slices = []
    if args.mask_dir:
        mask_slices = load_aligned_mask_slices(args.mask_dir, args.ct, args.slice_index)
    if args.output_keypoints_csv:
        write_keypoints_csv_atomic(
            args.output_keypoints_csv,
            prediction["keypoints_xy"],
            prediction["scores"],
            args.slice_index,
        )
    if args.output_overlay_png:
        write_overlay_png_atomic(
            args.output_overlay_png,
            model_image_yx,
            prediction["keypoints_xy"],
            mask_slices,
            "SuperPoint: Test CT axial slice z={}".format(args.slice_index),
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "bounded_single_slice_engineering_smoke_test",
        "whole_volume_processed": False,
        "ct": ct_metadata,
        "preprocessing": {
            "window_center_hu": float(args.window_center),
            "window_width_hu": float(args.window_width),
            "window_lower_hu": float(args.window_center - args.window_width / 2.0),
            "window_upper_hu": float(args.window_center + args.window_width / 2.0),
            "normalization_range": [0.0, 1.0],
            "resize_applied": False,
            "padding_applied": False,
            "native_slice_array_order": "nifti_[x,y]",
            "model_image_array_order": "image_[row=y,column=x]",
            "native_xy_to_model_yx_transpose_applied": True,
            "model_input_shape": [1, 1, *[int(value) for value in model_image_yx.shape]],
        },
        "model": model_provenance,
        "expected_provenance": {
            "superpoint_commit": EXPECTED_SUPERPOINT_COMMIT,
            "checkpoint_sha256": EXPECTED_SUPERPOINT_CHECKPOINT_SHA256,
        },
        "prediction": summarize_prediction(prediction),
        "visible_masks": [
            {
                "name": mask["name"],
                "path": mask["path"],
                "foreground_pixel_count": mask["foreground_pixel_count"],
            }
            for mask in mask_slices
        ],
        "outputs": {
            "keypoints_csv": str(Path(args.output_keypoints_csv).resolve())
            if args.output_keypoints_csv
            else None,
            "overlay_png": str(Path(args.output_overlay_png).resolve())
            if args.output_overlay_png
            else None,
        },
    }
    write_json_atomic(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Saved smoke-test summary: {}".format(Path(args.output_json).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
