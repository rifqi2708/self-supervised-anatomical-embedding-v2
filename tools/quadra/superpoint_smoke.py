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
    load_axial_ct_slice,
    load_superpoint_model,
    run_superpoint_on_slice,
    window_and_normalize_ct,
    write_json_atomic,
)


SCHEMA_VERSION = 1


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
    normalized = window_and_normalize_ct(
        ct_slice_hu,
        center=args.window_center,
        width=args.window_width,
    )
    ensure_stride_compatible(normalized)
    model, model_provenance = load_superpoint_model(
        args.superpoint_root,
        args.checkpoint,
        device=args.device,
    )
    prediction = run_superpoint_on_slice(model, normalized, model_provenance["device"])

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
            "model_input_shape": [1, 1, *[int(value) for value in normalized.shape]],
        },
        "model": model_provenance,
        "expected_provenance": {
            "superpoint_commit": EXPECTED_SUPERPOINT_COMMIT,
            "checkpoint_sha256": EXPECTED_SUPERPOINT_CHECKPOINT_SHA256,
        },
        "prediction": summarize_prediction(prediction),
    }
    write_json_atomic(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Saved smoke-test summary: {}".format(Path(args.output_json).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
