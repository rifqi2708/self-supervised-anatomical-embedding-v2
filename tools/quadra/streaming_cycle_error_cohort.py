#!/usr/bin/env python3
"""Run Quadra streaming cycle error sequentially for a resumable subject cohort."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra.streaming_cycle_error import (  # noqa: E402
    CLEANUP_FAILURE_EXIT_CODE,
    NORM_SPACING_XYZ,
    RUN_MANIFEST_SCHEMA_VERSION,
    add_run_arguments,
    canonical_subject_id,
    file_identity,
    normalize_and_validate_args,
    read_csv_rows,
    retained_core_size_xyz,
    utc_now,
    validate_completed_outputs,
    write_json,
)

DEFAULT_SUBJECT_START = 21
DEFAULT_SUBJECT_END = 48
DEFAULT_MIN_FREE_GB = 20.0
DEFAULT_BATCH_OUTPUT_ROOT = "data/quadra_output/streaming_cycle_error_batches"
DISK_GUARD_EXIT_CODE = 4


def shell_join(command: Sequence[str]) -> str:
    """Return a shell-readable command on Python 3.7 and newer."""
    return " ".join(shlex.quote(str(value)) for value in command)


def expand_subject_range(start: int, end: int) -> list[str]:
    if start > end:
        raise ValueError("--subject-start must be less than or equal to --subject-end")
    return [canonical_subject_id(f"quadra_hc_{number}") for number in range(start, end + 1)]


def expected_manifest_settings(args) -> dict[str, object]:
    organs = "all" if not args.organs else sorted(value.lower() for value in args.organs)
    return {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "norm_spacing_xyz": list(NORM_SPACING_XYZ),
        "tile_size_xyz": list(args.tile_size),
        "halo_xyz": list(args.halo),
        "retained_core_size_xyz": list(retained_core_size_xyz(args.tile_size, args.halo)),
        "match_chunk_xyz": list(args.match_chunk_size),
        "query_batch_size": int(args.query_batch_size),
        "num_points_per_mask": int(args.num_points),
        "seed": int(args.seed),
        "organs": organs,
        "is_mri": bool(args.is_mri),
    }


def manifest_is_compatible(
    manifest: dict[str, object],
    subject_id: str,
    settings: dict[str, object],
    config_sha256: str,
    checkpoint_sha256: str,
) -> bool:
    if int(manifest.get("schema_version", -1)) != RUN_MANIFEST_SCHEMA_VERSION:
        return False
    if not manifest.get("completed") or canonical_subject_id(str(manifest.get("subject_id", ""))) != subject_id:
        return False
    if manifest.get("cache_cleanup", {}).get("status") not in {"deleted", "retained"}:
        return False
    if manifest.get("config", {}).get("sha256") != config_sha256:
        return False
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        return False
    return all(manifest.get(key) == value for key, value in settings.items())


def validate_existing_run(run_dir: Path, manifest: dict[str, object]) -> None:
    points_path = Path(str(manifest.get("outputs", {}).get("points_raw_itk", ""))).resolve()
    _, rows = read_csv_rows(points_path)
    mask_names = sorted({row["mask_name"] for row in rows})
    validate_completed_outputs(
        run_dir,
        expected_point_count=int(manifest.get("point_count", -1)),
        subject_id=str(manifest.get("subject_id", "")),
        expected_mask_names=mask_names,
    )


def find_compatible_completed_run(
    output_root: Path,
    subject_id: str,
    settings: dict[str, object],
    config_sha256: str,
    checkpoint_sha256: str,
    not_before_epoch: float | None = None,
) -> tuple[Path, dict[str, object]] | None:
    candidates = sorted(output_root.glob(f"{subject_id}_*/run_manifest.json"), reverse=True)
    for manifest_path in candidates:
        try:
            if not_before_epoch is not None and manifest_path.stat().st_mtime < not_before_epoch:
                continue
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if not manifest_is_compatible(
                manifest, subject_id, settings, config_sha256, checkpoint_sha256
            ):
                continue
            validate_existing_run(manifest_path.parent, manifest)
            return manifest_path.parent, manifest
        except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
            continue
    return None


def build_subject_command(args, subject_id: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.quadra.streaming_cycle_error",
        "--subject",
        subject_id,
        "--dataset-root",
        str(args.dataset_root),
        "--cache-root",
        str(args.cache_root),
        "--output-root",
        str(args.output_root),
        "--config-file",
        str(args.config_file),
        "--checkpoint-file",
        str(args.checkpoint_file),
        "--tile-size",
        *(str(value) for value in args.tile_size),
        "--halo",
        *(str(value) for value in args.halo),
        "--match-chunk-size",
        *(str(value) for value in args.match_chunk_size),
        "--query-batch-size",
        str(args.query_batch_size),
        "--num-points",
        str(args.num_points),
        "--seed",
        str(args.seed),
    ]
    if args.organs:
        command.extend(["--organs", *args.organs])
    if args.overwrite_cache:
        command.append("--overwrite-cache")
    if args.keep_cache:
        command.append("--keep-cache")
    if args.is_mri:
        command.append("--is-mri")
    return command


def nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing parent found for disk check: {path}")
        candidate = candidate.parent
    return candidate


def free_disk_bytes(path: Path) -> int:
    return int(shutil.disk_usage(nearest_existing_parent(path)).free)


def run_subprocess_with_log(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {shell_join(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return int(process.wait())


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, include_subject=False)
    parser.add_argument("--subject-start", type=int, default=DEFAULT_SUBJECT_START)
    parser.add_argument("--subject-end", type=int, default=DEFAULT_SUBJECT_END)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-output-root", default=DEFAULT_BATCH_OUTPUT_ROOT)
    args = normalize_and_validate_args(parser, parser.parse_args(argv))
    if args.min_free_gb < 0:
        parser.error("--min-free-gb cannot be negative")
    try:
        expand_subject_range(args.subject_start, args.subject_end)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run(args) -> tuple[int, Path]:
    started = time.time()
    subjects = expand_subject_range(args.subject_start, args.subject_end)
    config = file_identity(args.config_file)
    checkpoint = file_identity(args.checkpoint_file)
    settings = expected_manifest_settings(args)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.batch_output_root).resolve() / (
        f"quadra_hc_{args.subject_start:03d}_{args.subject_end:03d}_{run_stamp}"
    )
    batch_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = batch_dir / "batch_manifest.json"
    output_root = Path(args.output_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    batch_manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "completed_at": None,
        "status": "dry_run" if args.dry_run else "running",
        "requested_subjects": subjects,
        "config": config,
        "checkpoint": checkpoint,
        "settings": settings,
        "cache_policy": "keep" if args.keep_cache else "delete_on_success",
        "min_free_gb": float(args.min_free_gb),
        "dry_run": bool(args.dry_run),
        "rerun_completed": bool(args.rerun_completed),
        "subjects": {},
        "skipped_subjects": [],
        "completed_subjects": [],
        "failed_subjects": [],
        "elapsed_seconds": 0.0,
    }

    def persist() -> None:
        batch_manifest["elapsed_seconds"] = float(time.time() - started)
        write_json(manifest_path, batch_manifest)

    persist()
    exit_code = 0
    for subject_id in subjects:
        compatible = find_compatible_completed_run(
            output_root,
            subject_id,
            settings,
            str(config["sha256"]),
            str(checkpoint["sha256"]),
        )
        if compatible is not None and not args.rerun_completed:
            run_dir, run_manifest = compatible
            batch_manifest["subjects"][subject_id] = {
                "status": "skipped_compatible_completed",
                "output_dir": str(run_dir),
                "cache_cleanup": run_manifest["cache_cleanup"],
            }
            batch_manifest["skipped_subjects"].append(subject_id)
            print(f"Skipping compatible completed subject: {subject_id} ({run_dir})")
            persist()
            continue

        command = build_subject_command(args, subject_id)
        subject_record = {
            "status": "planned" if args.dry_run else "running",
            "command": command,
            "command_text": shell_join(command),
            "log": str(batch_dir / "logs" / f"{subject_id}.log"),
            "started_at": None if args.dry_run else utc_now(),
            "completed_at": None,
            "return_code": None,
            "output_dir": None,
            "cache_cleanup": None,
            "error": None,
        }
        batch_manifest["subjects"][subject_id] = subject_record
        free_bytes = free_disk_bytes(cache_root)
        subject_record["free_disk_bytes_before"] = free_bytes
        if free_bytes < args.min_free_gb * (1024**3):
            subject_record.update(
                {
                    "status": "blocked_low_disk",
                    "error": f"Free disk is below {args.min_free_gb:g} GB.",
                    "completed_at": utc_now(),
                }
            )
            batch_manifest["status"] = "stopped_low_disk"
            exit_code = DISK_GUARD_EXIT_CODE
            persist()
            break
        if args.dry_run:
            print(f"DRY RUN {subject_id}: {shell_join(command)}")
            persist()
            continue

        persist()
        subprocess_started = time.time()
        return_code = run_subprocess_with_log(command, Path(subject_record["log"]))
        subject_record["return_code"] = return_code
        subject_record["completed_at"] = utc_now()
        completed = find_compatible_completed_run(
            output_root,
            subject_id,
            settings,
            str(config["sha256"]),
            str(checkpoint["sha256"]),
            not_before_epoch=subprocess_started,
        )
        if return_code == 0 and completed is not None:
            run_dir, run_manifest = completed
            subject_record.update(
                {
                    "status": "completed",
                    "output_dir": str(run_dir),
                    "cache_cleanup": run_manifest["cache_cleanup"],
                }
            )
            batch_manifest["completed_subjects"].append(subject_id)
        else:
            subject_record["status"] = "failed"
            subject_record["error"] = (
                f"Single-subject command returned {return_code}; "
                f"compatible complete output found={completed is not None}."
            )
            batch_manifest["failed_subjects"].append(subject_id)
            exit_code = 1
            if completed is not None:
                run_dir, run_manifest = completed
                subject_record["output_dir"] = str(run_dir)
                subject_record["cache_cleanup"] = run_manifest.get("cache_cleanup")
            if return_code == CLEANUP_FAILURE_EXIT_CODE:
                batch_manifest["status"] = "stopped_cleanup_failure"
                exit_code = CLEANUP_FAILURE_EXIT_CODE
                persist()
                break
        persist()

    if batch_manifest["status"] in {"running", "dry_run"}:
        if args.dry_run:
            batch_manifest["status"] = "dry_run_complete"
        elif batch_manifest["failed_subjects"]:
            batch_manifest["status"] = "completed_with_failures"
        else:
            batch_manifest["status"] = "completed"
    batch_manifest["completed_at"] = utc_now()
    persist()
    print(f"Cohort manifest: {manifest_path}")
    return exit_code, batch_dir


def main(argv: Iterable[str] | None = None) -> int:
    try:
        exit_code, _ = run(parse_args(argv))
        return exit_code
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
