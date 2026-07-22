#!/usr/bin/env python3
"""Run tiled UAE-S cycle error with global-NN and fixed-point matching."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra.coord_space_utils import (  # noqa: E402
    COORD_SPACE_RAW_ITK,
    COORD_SPACE_SAM,
    build_sam_to_raw_transform,
    transform_point_xyz,
)
from tools.quadra.streaming_cycle_error import (  # noqa: E402
    CLEANUP_FAILURE_EXIT_CODE,
    DEFAULT_CACHE_ROOT,
    NORM_SPACING_XYZ,
    CacheCleanupError,
    EmbeddingCache,
    add_run_arguments,
    build_embedding_cache,
    canonical_subject_id,
    delete_subject_cache_safely,
    directory_size_bytes,
    embedding_geometry_namespace,
    file_identity,
    normalize_and_validate_args,
    resolve_subject_pair,
    retained_core_size_xyz,
    sample_subject_points,
    stream_global_match_uaes,
    utc_now,
    write_json,
)
from tools.quadra.uaes_matching import FixedPointSettings, fixed_point_match_batch  # noqa: E402

RUN_MANIFEST_SCHEMA_VERSION = 4
DEFAULT_CONFIG_FILE = "configs/samv2/samv2_NIHLN.py"
DEFAULT_CHECKPOINT_FILE = "checkpoints/SAMv2_iter_20000.pth"
DEFAULT_OUTPUT_ROOT = "data/quadra_output/streaming_cycle_error_uaes"
DEFAULT_MATCHING_MODES = ("global_nn", "fixed_point")


def _signature_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _point_json(record):
    return {
        "idx": int(record["idx"]),
        "subject_id": str(record["subject_id"]),
        "mask_name": str(record["mask_name"]),
        "pt1_sam": np.asarray(record["pt1_sam"], dtype=np.int64).tolist(),
    }


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_resumable_run(output_root: Path, subject_id: str, signature_sha256: str):
    for manifest_path in sorted(output_root.glob(f"{subject_id}_*/run_manifest.json"), reverse=True):
        try:
            manifest = _load_json(manifest_path, {})
            if not manifest.get("completed") and manifest.get("run_signature_sha256") == signature_sha256:
                return manifest_path.parent, manifest
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _freeze_queries(run_dir: Path, records, pair):
    frozen_path = run_dir / "progress" / "frozen_queries.json"
    if frozen_path.is_file():
        frozen = _load_json(frozen_path, [])
        if len(frozen) != len(records):
            raise RuntimeError("Frozen-query count is incompatible with the requested run.")
        expected = [_point_json({**record, "idx": index}) for index, record in enumerate(records)]
        if frozen != expected:
            raise RuntimeError("Frozen queries differ from deterministic sampling for this run signature.")
        return frozen
    frozen = [_point_json({**record, "idx": index}) for index, record in enumerate(records)]
    write_json(frozen_path, frozen)
    return frozen


def _write_query_csvs(run_dir: Path, frozen, test_image: Path):
    transform, shape = build_sam_to_raw_transform(str(test_image))
    raw_path = run_dir / "query_points_raw_itk.csv"
    sam_path = run_dir / "query_points_sam.csv"
    fields = ["idx", "mask_name", "subject_id", "pt1_x", "pt1_y", "pt1_z", "coord_space"]
    for path, coord_space in ((raw_path, COORD_SPACE_RAW_ITK), (sam_path, COORD_SPACE_SAM)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in frozen:
                point = np.asarray(record["pt1_sam"], dtype=np.int64)
                if coord_space == COORD_SPACE_RAW_ITK:
                    point = transform_point_xyz(point, transform, shape)
                writer.writerow(
                    {
                        "idx": record["idx"],
                        "mask_name": record["mask_name"],
                        "subject_id": record["subject_id"],
                        "pt1_x": int(point[0]),
                        "pt1_y": int(point[1]),
                        "pt1_z": int(point[2]),
                        "coord_space": coord_space,
                    }
                )
    return raw_path, sam_path


def _global_nn_for_indices(test_cache, retest_cache, frozen, indices, args):
    points = np.stack([frozen[index]["pt1_sam"] for index in indices], axis=0)
    started = time.time()
    pt2, score12, forward_profile = stream_global_match_uaes(
        test_cache,
        retest_cache,
        points,
        args.query_batch_size,
        args.match_chunk_size,
        output_space="native",
    )
    pt1_back, score21, backward_profile = stream_global_match_uaes(
        retest_cache,
        test_cache,
        pt2,
        args.query_batch_size,
        args.match_chunk_size,
        output_space="native",
    )
    seconds_each = (time.time() - started) / max(len(indices), 1)
    records = []
    for local, index in enumerate(indices):
        records.append(
            {
                **frozen[index],
                "pt2_sam": pt2[local].tolist(),
                "pt1_back_sam": pt1_back[local].tolist(),
                "score_12": float(score12[local]),
                "score_21": float(score21[local]),
                "status_12": "success",
                "status_21": "success",
                "failure_reason_12": None,
                "failure_reason_21": None,
                "stable_anchor_count_12": None,
                "stable_anchor_count_21": None,
                "seconds": float(seconds_each),
            }
        )
    return records, {"forward": forward_profile, "backward": backward_profile}


def _fixed_point_for_indices(test_cache, retest_cache, frozen, indices, args, settings):
    points = np.stack([frozen[index]["pt1_sam"] for index in indices], axis=0)
    started = time.time()
    forward, forward_profile = fixed_point_match_batch(
        test_cache,
        retest_cache,
        points,
        settings,
        args.query_batch_size,
        args.match_chunk_size,
    )
    successful = [local for local, result in enumerate(forward) if result["status"] == "success"]
    backward_by_local = {}
    backward_profile = None
    if successful:
        target_points = np.stack([forward[local]["point_xyz"] for local in successful], axis=0)
        backward, backward_profile = fixed_point_match_batch(
            retest_cache,
            test_cache,
            target_points,
            settings,
            args.query_batch_size,
            args.match_chunk_size,
        )
        backward_by_local = {local: result for local, result in zip(successful, backward)}
    seconds_each = (time.time() - started) / max(len(indices), 1)
    records = []
    for local, index in enumerate(indices):
        result12 = forward[local]
        result21 = backward_by_local.get(local)
        if result12["status"] != "success":
            status21 = "not_run"
            failure21 = "forward_match_failed"
        elif result21 is None:
            status21 = "failed"
            failure21 = "backward_result_missing"
        else:
            status21 = result21["status"]
            failure21 = result21.get("failure_reason")
        records.append(
            {
                **frozen[index],
                "pt2_sam": (
                    np.asarray(result12["point_xyz"], dtype=np.int64).tolist()
                    if result12["status"] == "success"
                    else None
                ),
                "pt1_back_sam": (
                    np.asarray(result21["point_xyz"], dtype=np.int64).tolist()
                    if result21 is not None and result21["status"] == "success"
                    else None
                ),
                "score_12": result12.get("score"),
                "score_21": result21.get("score") if result21 is not None else None,
                "status_12": result12["status"],
                "status_21": status21,
                "failure_reason_12": result12.get("failure_reason"),
                "failure_reason_21": failure21,
                "stable_anchor_count_12": result12.get("stable_anchor_count"),
                "stable_anchor_count_21": (
                    result21.get("stable_anchor_count") if result21 is not None else None
                ),
                "seconds": float(seconds_each),
            }
        )
    return records, {"forward": forward_profile, "backward": backward_profile}


def _process_mode_by_organ(mode, test_cache, retest_cache, frozen, args, settings, progress_path):
    progress = _load_json(progress_path, {"records": {}, "profiles": {}})
    grouped = {}
    for record in frozen:
        grouped.setdefault(record["mask_name"], []).append(int(record["idx"]))
    for mask_name in sorted(grouped):
        indices = grouped[mask_name]
        if all(str(index) in progress["records"] for index in indices):
            print(f"Skipping completed {mode} organ: {mask_name}")
            continue
        print(f"Running {mode} for {mask_name} ({len(indices)} points)")
        if mode == "global_nn":
            records, profile = _global_nn_for_indices(test_cache, retest_cache, frozen, indices, args)
        else:
            records, profile = _fixed_point_for_indices(
                test_cache, retest_cache, frozen, indices, args, settings
            )
        for record in records:
            progress["records"][str(record["idx"])] = record
        progress["profiles"][mask_name] = profile
        write_json(progress_path, progress)
    records = [progress["records"][str(index)] for index in range(len(frozen))]
    return records, progress["profiles"]


def _convert_records(records, test_image: Path, retest_image: Path, coord_space: str):
    import SimpleITK as sitk

    test_transform, test_shape = build_sam_to_raw_transform(str(test_image))
    retest_transform, retest_shape = build_sam_to_raw_transform(str(retest_image))
    test_itk = sitk.ReadImage(str(test_image))
    output = []
    for record in records:
        pt1_sam = np.asarray(record["pt1_sam"], dtype=np.int64)
        pt2_sam = None if record["pt2_sam"] is None else np.asarray(record["pt2_sam"], dtype=np.int64)
        back_sam = (
            None if record["pt1_back_sam"] is None else np.asarray(record["pt1_back_sam"], dtype=np.int64)
        )
        if coord_space == COORD_SPACE_RAW_ITK:
            pt1 = transform_point_xyz(pt1_sam, test_transform, test_shape)
            pt2 = None if pt2_sam is None else transform_point_xyz(pt2_sam, retest_transform, retest_shape)
            back = None if back_sam is None else transform_point_xyz(back_sam, test_transform, test_shape)
        else:
            pt1, pt2, back = pt1_sam, pt2_sam, back_sam
        voxel_error = None
        mm_error = None
        if back is not None:
            voxel_error = float(np.linalg.norm(back.astype(float) - pt1.astype(float)))
            raw_pt1 = transform_point_xyz(pt1_sam, test_transform, test_shape)
            raw_back = transform_point_xyz(back_sam, test_transform, test_shape)
            physical_1 = np.asarray(test_itk.TransformIndexToPhysicalPoint(tuple(int(v) for v in raw_pt1)))
            physical_back = np.asarray(test_itk.TransformIndexToPhysicalPoint(tuple(int(v) for v in raw_back)))
            mm_error = float(np.linalg.norm(physical_back - physical_1))
        output.append(
            {
                **record,
                "coord_space": coord_space,
                "pt1": pt1,
                "pt2": pt2,
                "pt1_back": back,
                "voxel_error": voxel_error,
                "mm_error": mm_error,
            }
        )
    return output


RESULT_FIELDS = [
    "idx", "mask_name", "subject_id",
    "pt1_x", "pt1_y", "pt1_z", "pt2_x", "pt2_y", "pt2_z",
    "pt1_back_x", "pt1_back_y", "pt1_back_z",
    "voxel_error", "mm_error", "score_12", "score_21",
    "status_12", "status_21", "failure_reason_12", "failure_reason_21",
    "stable_anchor_count_12", "stable_anchor_count_21", "seconds", "coord_space",
]


def _xyz_fields(prefix, point):
    if point is None:
        return {f"{prefix}_{axis}": "" for axis in "xyz"}
    array = np.asarray(point, dtype=np.int64)
    return {f"{prefix}_{axis}": int(array[index]) for index, axis in enumerate("xyz")}


def _write_results(path: Path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for record in records:
            row = {
                "idx": record["idx"],
                "mask_name": record["mask_name"],
                "subject_id": record["subject_id"],
                **_xyz_fields("pt1", record["pt1"]),
                **_xyz_fields("pt2", record["pt2"]),
                **_xyz_fields("pt1_back", record["pt1_back"]),
                "voxel_error": "" if record["voxel_error"] is None else record["voxel_error"],
                "mm_error": "" if record["mm_error"] is None else record["mm_error"],
                "score_12": "" if record["score_12"] is None else record["score_12"],
                "score_21": "" if record["score_21"] is None else record["score_21"],
                "status_12": record["status_12"],
                "status_21": record["status_21"],
                "failure_reason_12": record["failure_reason_12"] or "",
                "failure_reason_21": record["failure_reason_21"] or "",
                "stable_anchor_count_12": (
                    "" if record["stable_anchor_count_12"] is None else record["stable_anchor_count_12"]
                ),
                "stable_anchor_count_21": (
                    "" if record["stable_anchor_count_21"] is None else record["stable_anchor_count_21"]
                ),
                "seconds": record["seconds"],
                "coord_space": record["coord_space"],
            }
            writer.writerow(row)


def _summary_rows(records):
    groups = {"ALL_MASKS": records}
    for record in records:
        groups.setdefault(record["mask_name"], []).append(record)
    rows = []
    for mask_name, group in sorted(groups.items()):
        successful = [record for record in group if record["mm_error"] is not None]
        values = np.asarray([record["mm_error"] for record in successful], dtype=np.float64)
        rows.append(
            {
                "mask_name": mask_name,
                "requested": len(group),
                "forward_success": sum(record["status_12"] == "success" for record in group),
                "cycle_success": len(successful),
                "failed": len(group) - len(successful),
                "mean_mm": float(values.mean()) if len(values) else None,
                "median_mm": float(np.median(values)) if len(values) else None,
                "p95_mm": float(np.percentile(values, 95)) if len(values) else None,
                "max_mm": float(values.max()) if len(values) else None,
            }
        )
    return rows


def _write_summary(path: Path, rows):
    fields = ["mask_name", "requested", "forward_success", "cycle_success", "failed", "mean_mm", "median_mm", "p95_mm", "max_mm"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _write_comparison(path: Path, global_records, fixed_records, target_spacing_xyz):
    fields = [
        "idx", "mask_name", "subject_id", "global_status", "fixed_status",
        "global_cycle_error_mm", "fixed_cycle_error_mm", "absolute_cycle_error_difference_mm",
        "forward_match_displacement_mm", "fixed_failure_reason", "fixed_stable_anchor_count_12",
        "global_seconds", "fixed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for global_record, fixed_record in zip(global_records, fixed_records):
            cycle_difference = None
            if global_record["mm_error"] is not None and fixed_record["mm_error"] is not None:
                cycle_difference = abs(global_record["mm_error"] - fixed_record["mm_error"])
            displacement = None
            if global_record["pt2"] is not None and fixed_record["pt2"] is not None:
                displacement = float(
                    np.linalg.norm(
                        (
                            np.asarray(global_record["pt2"], dtype=float)
                            - np.asarray(fixed_record["pt2"], dtype=float)
                        )
                        * np.asarray(target_spacing_xyz, dtype=float)
                    )
                )
            writer.writerow(
                {
                    "idx": global_record["idx"],
                    "mask_name": global_record["mask_name"],
                    "subject_id": global_record["subject_id"],
                    "global_status": global_record["status_21"],
                    "fixed_status": fixed_record["status_21"],
                    "global_cycle_error_mm": global_record["mm_error"] if global_record["mm_error"] is not None else "",
                    "fixed_cycle_error_mm": fixed_record["mm_error"] if fixed_record["mm_error"] is not None else "",
                    "absolute_cycle_error_difference_mm": cycle_difference if cycle_difference is not None else "",
                    "forward_match_displacement_mm": displacement if displacement is not None else "",
                    "fixed_failure_reason": fixed_record["failure_reason_12"] or fixed_record["failure_reason_21"] or "",
                    "fixed_stable_anchor_count_12": fixed_record["stable_anchor_count_12"] or "",
                    "global_seconds": global_record["seconds"],
                    "fixed_seconds": fixed_record["seconds"],
                }
            )


def validate_uaes_outputs(run_dir: Path, modes: Sequence[str], expected_count: int):
    manifest = _load_json(run_dir / "run_manifest.json", {})
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("UAE-S run manifest schema is incompatible.")
    for query_name in ("query_points_raw_itk.csv", "query_points_sam.csv"):
        with (run_dir / query_name).open(newline="", encoding="utf-8") as handle:
            if len(list(csv.DictReader(handle))) != expected_count:
                raise RuntimeError(f"{query_name} has an incomplete query denominator.")
    for mode in modes:
        for suffix in ("", "_sam"):
            path = run_dir / f"cycle_points_{mode}{suffix}.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != expected_count:
                raise RuntimeError(f"{path.name} has {len(rows)} rows; expected {expected_count}.")
            if any(row["coord_space"] != (COORD_SPACE_SAM if suffix else COORD_SPACE_RAW_ITK) for row in rows):
                raise RuntimeError(f"{path.name} contains an incompatible coordinate space.")
    if set(modes) == {"global_nn", "fixed_point"}:
        with (run_dir / "matching_comparison.csv").open(newline="", encoding="utf-8") as handle:
            if len(list(csv.DictReader(handle))) != expected_count:
                raise RuntimeError("matching_comparison.csv has an incomplete paired denominator.")
    return {"validated_at": utc_now(), "point_count": expected_count, "modes": list(modes)}


def run(args) -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command inside the configured RunPod GPU environment.")
    from tools.interfaces import init

    os.chdir(PROJECT_ROOT)
    subject_id = canonical_subject_id(args.subject)
    dataset_root = Path(args.dataset_root).resolve()
    pair = resolve_subject_pair(dataset_root, subject_id)
    config = file_identity(args.config_file)
    checkpoint = file_identity(args.checkpoint_file)
    settings = FixedPointSettings(
        margin_xyz=args.fixed_point_margin,
        iterations=args.fixed_point_iterations,
        score_threshold=args.fixed_point_score_threshold,
        max_return_distance_mm=args.fixed_point_max_return_mm,
    )
    organs = None if not args.organs else sorted(value.lower() for value in args.organs)
    signature = {
        "model_profile": "uae_s",
        "matching_modes": list(args.matching_modes),
        "config_sha256": config["sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "dataset_root": str(dataset_root),
        "norm_spacing_xyz": list(NORM_SPACING_XYZ),
        "tile_size_xyz": list(args.tile_size),
        "halo_xyz": list(args.halo),
        "match_chunk_xyz": list(args.match_chunk_size),
        "query_batch_size": args.query_batch_size,
        "num_points_per_mask": args.num_points,
        "seed": args.seed,
        "organs": organs or "all",
        "fixed_point": settings.to_dict(),
    }
    signature_sha = _signature_hash(signature)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resumable = _find_resumable_run(output_root, subject_id, signature_sha)
    if args.resume_run:
        run_dir = Path(args.resume_run).resolve()
        manifest = _load_json(run_dir / "run_manifest.json", {})
        if manifest.get("run_signature_sha256") != signature_sha or manifest.get("completed"):
            raise RuntimeError("--resume-run is complete or incompatible with the requested settings.")
    elif resumable is not None:
        run_dir, manifest = resumable
        print(f"Resuming compatible UAE-S run: {run_dir}")
    else:
        run_dir = output_root / f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "created_at": utc_now(),
            "completed": False,
            "subject_id": subject_id,
            "model_profile": "uae_s",
            "run_signature": signature,
            "run_signature_sha256": signature_sha,
            "phases": {"embedding": "pending", "global_nn": "pending", "fixed_point": "pending", "export": "pending"},
            "error": None,
        }
        write_json(run_dir / "run_manifest.json", manifest)

    geometry = embedding_geometry_namespace(args.tile_size, args.halo)
    cache_key = f"uae_s_{Path(args.checkpoint_file).stem}_{checkpoint['sha256'][:12]}_2mm_{geometry}"
    subject_cache_root = Path(args.cache_root).resolve() / cache_key / subject_id
    test_cache_dir = subject_cache_root / "test"
    retest_cache_dir = subject_cache_root / "retest"
    test_cache = retest_cache = None
    started = time.time()
    try:
        model = None
        if args.overwrite_cache or not (test_cache_dir / "manifest.json").is_file() or not (retest_cache_dir / "manifest.json").is_file():
            model = init(str(Path(args.config_file).resolve()), str(Path(args.checkpoint_file).resolve()))
        test_manifest = build_embedding_cache(
            pair["test"], test_cache_dir, model, config, checkpoint, args.tile_size, args.halo,
            args.overwrite_cache, args.is_mri, model_profile="uae_s"
        )
        retest_manifest = build_embedding_cache(
            pair["retest"], retest_cache_dir, model, config, checkpoint, args.tile_size, args.halo,
            args.overwrite_cache, args.is_mri, model_profile="uae_s"
        )
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        manifest["phases"]["embedding"] = "complete"
        manifest.update({"config": config, "checkpoint": checkpoint, "test_cache_manifest": test_manifest, "retest_cache_manifest": retest_manifest})
        write_json(run_dir / "run_manifest.json", manifest)

        selected = None if organs is None else set(organs)
        sampled = sample_subject_points(pair, args.num_points, args.seed, selected, args.is_mri)
        frozen = _freeze_queries(run_dir, sampled, pair)
        query_raw, query_sam = _write_query_csvs(run_dir, frozen, pair["test"])
        test_cache = EmbeddingCache(test_cache_dir)
        retest_cache = EmbeddingCache(retest_cache_dir)
        method_internal = {}
        profiles = {}
        for mode in args.matching_modes:
            progress_path = run_dir / "progress" / f"{mode}.json"
            records, mode_profiles = _process_mode_by_organ(
                mode, test_cache, retest_cache, frozen, args, settings, progress_path
            )
            method_internal[mode] = records
            profiles[mode] = mode_profiles
            manifest["phases"][mode] = "complete"
            write_json(run_dir / "run_manifest.json", manifest)

        outputs = {"query_points_raw_itk": str(query_raw), "query_points_sam": str(query_sam)}
        converted_raw = {}
        summaries = {}
        for mode, records in method_internal.items():
            raw = _convert_records(records, pair["test"], pair["retest"], COORD_SPACE_RAW_ITK)
            sam = _convert_records(records, pair["test"], pair["retest"], COORD_SPACE_SAM)
            converted_raw[mode] = raw
            raw_path = run_dir / f"cycle_points_{mode}.csv"
            sam_path = run_dir / f"cycle_points_{mode}_sam.csv"
            summary_path = run_dir / f"cycle_summary_{mode}.csv"
            _write_results(raw_path, raw)
            _write_results(sam_path, sam)
            summaries[mode] = _summary_rows(raw)
            _write_summary(summary_path, summaries[mode])
            outputs.update({f"{mode}_raw_itk": str(raw_path), f"{mode}_sam": str(sam_path), f"{mode}_summary": str(summary_path)})
        if set(args.matching_modes) == {"global_nn", "fixed_point"}:
            import SimpleITK as sitk

            comparison_path = run_dir / "matching_comparison.csv"
            target_spacing = sitk.ReadImage(str(pair["retest"])).GetSpacing()
            _write_comparison(
                comparison_path,
                converted_raw["global_nn"],
                converted_raw["fixed_point"],
                target_spacing,
            )
            outputs["matching_comparison"] = str(comparison_path)

        run_summary = {
            "subject_id": subject_id,
            "model_profile": "uae_s",
            "point_count": len(frozen),
            "matching_modes": list(args.matching_modes),
            "summaries": summaries,
            "elapsed_seconds": float(time.time() - started),
            "interpretation_note": "Fixed-point matching uses forward-backward consistency internally; cycle error is a consistency measure, not independent anatomical accuracy.",
        }
        summary_json = run_dir / "run_summary.json"
        write_json(summary_json, run_summary)
        outputs["run_summary"] = str(summary_json)
        cache_policy = "keep" if args.keep_cache else "delete_on_success"
        manifest.update(
            {
                "completed": False,
                "completed_at": None,
                "point_count": len(frozen),
                "matching_modes": list(args.matching_modes),
                "fixed_point": settings.to_dict(),
                "dataset_root": str(dataset_root),
                "match_chunk_xyz": list(args.match_chunk_size),
                "query_batch_size": int(args.query_batch_size),
                "num_points_per_mask": int(args.num_points),
                "seed": int(args.seed),
                "organs": organs or "all",
                "is_mri": bool(args.is_mri),
                "norm_spacing_xyz": list(NORM_SPACING_XYZ),
                "tile_size_xyz": list(args.tile_size),
                "halo_xyz": list(args.halo),
                "retained_core_size_xyz": list(retained_core_size_xyz(args.tile_size, args.halo)),
                "profiles": profiles,
                "outputs": outputs,
                "cache_policy": cache_policy,
                "cache_cleanup": {
                    "status": "retained" if args.keep_cache else "scheduled",
                    "subject_cache_root": str(subject_cache_root),
                    "deleted_paths": [],
                    "bytes_before_cleanup": None,
                    "bytes_freed": 0,
                    "completed_at": None,
                    "error": None,
                },
            }
        )
        manifest["phases"]["export"] = "complete"
        write_json(run_dir / "run_manifest.json", manifest)
        validation = validate_uaes_outputs(run_dir, args.matching_modes, len(frozen))
        manifest["output_validation"] = validation
        test_cache.close()
        retest_cache.close()
        test_cache = retest_cache = None
        if args.keep_cache:
            manifest["cache_cleanup"].update(
                {"bytes_before_cleanup": directory_size_bytes(subject_cache_root), "completed_at": utc_now()}
            )
        else:
            try:
                before = directory_size_bytes(subject_cache_root)
                cleanup = delete_subject_cache_safely(
                    Path(args.cache_root), subject_cache_root, subject_id, (test_manifest, retest_manifest)
                )
                manifest["cache_cleanup"].update(
                    {
                        "status": "deleted",
                        "deleted_paths": cleanup["deleted_paths"],
                        "bytes_before_cleanup": before,
                        "bytes_freed": cleanup["bytes_freed"],
                        "completed_at": utc_now(),
                    }
                )
            except Exception as exc:
                manifest["cache_cleanup"].update({"status": "failed", "error": str(exc), "completed_at": utc_now()})
                write_json(run_dir / "run_manifest.json", manifest)
                raise CacheCleanupError(f"Outputs are complete but UAE-S cache cleanup failed: {exc}") from exc
        manifest["completed"] = True
        manifest["completed_at"] = utc_now()
        manifest["error"] = None
        manifest.pop("failed_at", None)
        write_json(run_dir / "run_manifest.json", manifest)
        print(f"Completed UAE-S run: {run_dir}")
        return run_dir
    except Exception as exc:
        manifest["completed"] = False
        manifest["completed_at"] = None
        manifest["error"] = str(exc)
        manifest["failed_at"] = utc_now()
        write_json(run_dir / "run_manifest.json", manifest)
        raise
    finally:
        if test_cache is not None:
            test_cache.close()
        if retest_cache is not None:
            retest_cache.close()


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.set_defaults(
        config_file=DEFAULT_CONFIG_FILE,
        checkpoint_file=DEFAULT_CHECKPOINT_FILE,
        output_root=DEFAULT_OUTPUT_ROOT,
        cache_root=DEFAULT_CACHE_ROOT,
    )
    parser.add_argument("--matching-modes", nargs="+", choices=DEFAULT_MATCHING_MODES, default=list(DEFAULT_MATCHING_MODES))
    parser.add_argument("--fixed-point-margin", nargs=3, type=int, default=(2, 2, 2), metavar=("X", "Y", "Z"))
    parser.add_argument("--fixed-point-iterations", type=int, default=4)
    parser.add_argument("--fixed-point-score-threshold", type=float, default=0.8)
    parser.add_argument("--fixed-point-max-return-mm", type=float, default=100.0)
    parser.add_argument("--resume-run", default=None)
    args = normalize_and_validate_args(parser, parser.parse_args(argv))
    args.matching_modes = tuple(dict.fromkeys(args.matching_modes))
    args.fixed_point_margin = tuple(int(value) for value in args.fixed_point_margin)
    if args.fixed_point_iterations < 2 or args.fixed_point_iterations % 2:
        parser.error("--fixed-point-iterations must be an even integer of at least 2")
    if any(value < 0 for value in args.fixed_point_margin):
        parser.error("--fixed-point-margin values cannot be negative")
    if not 0.0 <= args.fixed_point_score_threshold <= 1.0:
        parser.error("--fixed-point-score-threshold must be between 0 and 1")
    if args.fixed_point_max_return_mm <= 0:
        parser.error("--fixed-point-max-return-mm must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except CacheCleanupError as exc:
        print(f"CACHE CLEANUP ERROR: {exc}", file=sys.stderr)
        return CLEANUP_FAILURE_EXIT_CODE
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
