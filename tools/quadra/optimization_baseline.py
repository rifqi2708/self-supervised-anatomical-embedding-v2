#!/usr/bin/env python3
"""Capture and validate the immutable Quadra memory-optimization contract."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - exercised by the released Python 3.7 image
    try:
        import importlib_metadata
    except ImportError:  # pragma: no cover - fallback for minimal legacy images
        importlib_metadata = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra.coord_space_utils import COORD_SPACE_RAW_ITK  # noqa: E402
from tools.quadra.environment import (  # noqa: E402
    PROFILE_EXPECTED_DIGESTS,
    PROFILE_IMAGES,
    canonical_layout,
    validate_storage_root,
)


SCHEMA_VERSION = 1
OPTIMIZATION_ID = "quadra-uaes-memory-v1"
EXPECTED_BASE_COMMIT = "f7242febfdb3b2b072aa449f41b2fbd0dcdd7f69"
EXPECTED_BRANCH = "codex/quadra-memory-optimization"
CONFIG_RELATIVE_PATH = Path("configs/samv2/samv2_NIHLN.py")
CHECKPOINT_FILENAME = "SAMv2_iter_20000.pth"
EXPECTED_CHECKPOINT_SHA256 = (
    "a094d5eef867504defdc4c8e1d950835c4eb8aaa19de2027bb1a194781e423e3"
)
NORM_SPACING_XYZ = (2.0, 2.0, 2.0)
OPTIMIZATION_SEED = 20260721
MATCHING_MODES = ("global_nn", "fixed_point")
PRIMARY_ERROR_UNIT = "mm"
EXPECTED_GPU_NAME = "NVIDIA RTX A6000"
VRAM_USAGE_CEILING_FRACTION = 0.80
EXPECTED_COUNTS = {
    "ct_subjects": 48,
    "ct_files": 96,
    "mask_subjects": 28,
    "stage5_scans": 56,
    "final_masks": 2208,
}
RUN_ID_PATTERN = re.compile(r"^stage0-[A-Za-z0-9_.-]+$")


class OptimizationBaselineError(RuntimeError):
    """Raised when Stage 0 cannot establish a trustworthy baseline."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("stage0-%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise OptimizationBaselineError(f"Required file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise OptimizationBaselineError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise OptimizationBaselineError(
            f"Command failed: {' '.join(command)}{suffix}"
        ) from exc


def _git_output(repository_root: Path, *arguments: str) -> str:
    return _run(("git", *arguments), cwd=repository_root).stdout.strip()


def inspect_repository(
    repository_root: Path,
    *,
    expected_base_commit: str = EXPECTED_BASE_COMMIT,
    expected_branch: str = EXPECTED_BRANCH,
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    if not (repository_root / ".git").exists():
        raise OptimizationBaselineError(f"Not a Git repository: {repository_root}")

    branch = _git_output(repository_root, "symbolic-ref", "--short", "HEAD")
    if branch != expected_branch:
        raise OptimizationBaselineError(
            f"Stage 0 requires branch {expected_branch!r}, found {branch!r}"
        )
    execution_commit = _git_output(repository_root, "rev-parse", "HEAD")
    base_result = _run(
        ("git", "merge-base", "--is-ancestor", expected_base_commit, execution_commit),
        cwd=repository_root,
        check=False,
    )
    if base_result.returncode != 0:
        raise OptimizationBaselineError(
            f"Execution commit {execution_commit} does not descend from "
            f"{expected_base_commit}"
        )
    porcelain = _git_output(repository_root, "status", "--porcelain")
    if porcelain:
        raise OptimizationBaselineError(
            "Stage 0 requires a clean repository; commit or remove these changes:\n"
            + porcelain
        )
    return {
        "path": str(repository_root),
        "branch": branch,
        "base_commit": expected_base_commit,
        "execution_commit": execution_commit,
        "clean": True,
    }


def _is_within(path: Path, root: Path) -> bool:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def inspect_assets(storage_root: Path) -> dict[str, Any]:
    layout = canonical_layout(storage_root)
    ct_root = layout["whole_body_ct"]
    mask_root = layout["totalsegmentator_outputs"]
    if not ct_root.is_dir():
        raise OptimizationBaselineError(f"Canonical CT root is missing: {ct_root}")
    if not mask_root.is_dir():
        raise OptimizationBaselineError(f"Canonical mask root is missing: {mask_root}")

    ct_subjects = sorted(path for path in ct_root.glob("QUADRA_HC_*") if path.is_dir())
    ct_files = sorted(ct_root.glob("QUADRA_HC_*/*_CT-AC.nii.gz"))
    mask_subjects = sorted(
        path for path in mask_root.glob("quadra_hc_*") if path.is_dir()
    )
    scan_directories = sorted(
        path
        for path in mask_root.glob("quadra_hc_*/*")
        if path.is_dir() and (path / "masks").is_dir()
    )
    final_masks = sorted(mask_root.glob("quadra_hc_*/*/masks/*.nii.gz"))
    observed = {
        "ct_subjects": len(ct_subjects),
        "ct_files": len(ct_files),
        "mask_subjects": len(mask_subjects),
        "stage5_scans": len(scan_directories),
        "final_masks": len(final_masks),
    }
    if observed != EXPECTED_COUNTS:
        raise OptimizationBaselineError(
            f"Canonical asset counts differ from the accepted environment: "
            f"expected={EXPECTED_COUNTS}, observed={observed}"
        )
    return {
        "whole_body_ct_root": str(ct_root.resolve()),
        "totalsegmentator_mask_root": str(mask_root.resolve()),
        "counts": observed,
        "validation_level": "path_and_count_only_no_nifti_decompression",
    }


def _package_version(distribution: str) -> str:
    if importlib_metadata is not None:
        try:
            return importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise OptimizationBaselineError(
                f"Required preprocessing package is unavailable: {distribution}"
            ) from exc
    try:  # pragma: no cover - only used by a minimal Python 3.7 environment
        import pkg_resources

        return str(pkg_resources.get_distribution(distribution).version)
    except Exception as exc:  # pragma: no cover
        raise OptimizationBaselineError(
            f"Required package is unavailable: {distribution}"
        ) from exc


def _active_storage_root(storage_root: Path) -> None:
    active_storage_root = os.environ.get("QUADRA_STORAGE_ROOT")
    if not active_storage_root:
        raise OptimizationBaselineError(
            "QUADRA_STORAGE_ROOT is unset; first source the persistent activation "
            "script with the required profile"
        )
    if Path(active_storage_root).resolve() != Path(storage_root).resolve():
        raise OptimizationBaselineError(
            f"Active QUADRA_STORAGE_ROOT differs from --storage-root: "
            f"{active_storage_root} != {storage_root}"
        )


def _module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as exc:
        raise OptimizationBaselineError(
            f"Required module is unavailable: {module_name}: {exc}"
        ) from exc
    value = getattr(module, "__version__", None)
    if not value:
        raise OptimizationBaselineError(
            f"Required module has no version: {module_name}"
        )
    return str(value)


def inspect_active_runtime(profile: str, storage_root: Path) -> dict[str, Any]:
    _active_storage_root(storage_root)
    python_version = (
        int(sys.version_info.major),
        int(sys.version_info.minor),
        int(sys.version_info.micro),
    )
    torch_version = _package_version("torch")
    result = {
        "profile": profile,
        "python": ".".join(str(value) for value in python_version),
        "torch": torch_version,
    }
    if profile == "preprocess":
        if python_version[:2] < (3, 10):
            raise OptimizationBaselineError(
                f"Stage 0 requires the preprocessing profile, found Python "
                f"{result['python']}"
            )
        if int(torch_version.split(".", 1)[0]) < 2:
            raise OptimizationBaselineError(
                f"Stage 0 requires preprocessing PyTorch 2+, found {torch_version}"
            )
        totalsegmentator_version = _package_version("TotalSegmentator")
        if totalsegmentator_version != "2.16.0":
            raise OptimizationBaselineError(
                "Stage 0 requires TotalSegmentator 2.16.0, found "
                f"{totalsegmentator_version}"
            )
        result["totalsegmentator"] = totalsegmentator_version
    elif profile == "uae":
        if python_version[:2] != (3, 7):
            raise OptimizationBaselineError(
                f"UAE stages require Python 3.7, found {result['python']}"
            )
        if not torch_version.startswith("1.9"):
            raise OptimizationBaselineError(
                f"UAE stages require PyTorch 1.9, found {torch_version}"
            )
        result["mmcv"] = _module_version("mmcv")
        result["mmdet"] = _module_version("mmdet")
    else:
        raise OptimizationBaselineError(f"Unknown required profile: {profile}")
    return result


def inspect_active_preprocess_runtime(storage_root: Path) -> dict[str, Any]:
    """Backward-compatible named helper for the Stage 0 capture path."""
    return inspect_active_runtime("preprocess", storage_root)


def inspect_environment_profiles(storage_root: Path) -> dict[str, Any]:
    layout = canonical_layout(storage_root)
    manifest_path = layout["manifests"] / "environment.json"
    if not manifest_path.is_file():
        raise OptimizationBaselineError(
            f"Persistent environment manifest is missing: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        environment_manifest = json.load(handle)
    profiles = environment_manifest.get("profiles", {})
    records: dict[str, Any] = {}
    for profile_name in ("preprocess", "uae"):
        if profile_name not in profiles:
            raise OptimizationBaselineError(
                f"Persistent profile is not bootstrapped: {profile_name}"
            )
        profile = profiles[profile_name]
        expected_digest = PROFILE_EXPECTED_DIGESTS[profile_name]
        recorded_expected = profile.get("expected_image_digest")
        recorded_observed = profile.get("image_digest")
        if recorded_expected != expected_digest or recorded_observed != expected_digest:
            raise OptimizationBaselineError(
                f"{profile_name} image digest is incompatible: "
                f"expected={expected_digest}, recorded={recorded_observed}"
            )
        fingerprint_path = Path(profile.get("fingerprint_path", ""))
        fingerprint = file_identity(fingerprint_path)
        records[profile_name] = {
            "image_ref": profile.get("image_ref"),
            "image_digest": recorded_observed,
            "expected_image_ref": PROFILE_IMAGES[profile_name],
            "runtime_errors": profile.get("runtime_errors", []),
            "fingerprint": fingerprint,
        }
        if records[profile_name]["runtime_errors"]:
            raise OptimizationBaselineError(
                f"{profile_name} profile has recorded runtime errors: "
                f"{records[profile_name]['runtime_errors']}"
            )
    return {
        "manifest": file_identity(manifest_path),
        "profiles": records,
    }


def parse_nvidia_smi(output: str) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise OptimizationBaselineError(
            f"Stage 0 expects exactly one visible GPU, found {len(lines)}"
        )
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 5:
        raise OptimizationBaselineError(
            f"Unexpected nvidia-smi response: {lines[0]!r}"
        )
    index, name, driver_version, memory_total_mib, gpu_uuid = fields
    try:
        memory_total_mib_int = int(memory_total_mib)
    except ValueError as exc:
        raise OptimizationBaselineError(
            f"Invalid GPU memory value: {memory_total_mib!r}"
        ) from exc
    if name != EXPECTED_GPU_NAME:
        raise OptimizationBaselineError(
            f"Stage 0 requires {EXPECTED_GPU_NAME}, found {name}"
        )
    return {
        "index": int(index),
        "name": name,
        "driver_version": driver_version,
        "memory_total_mib": memory_total_mib_int,
        "uuid": gpu_uuid,
        "accepted_peak_reserved_mib": int(
            memory_total_mib_int * VRAM_USAGE_CEILING_FRACTION
        ),
        "required_headroom_fraction": 1.0 - VRAM_USAGE_CEILING_FRACTION,
    }


def inspect_gpu() -> dict[str, Any]:
    result = _run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,uuid",
            "--format=csv,noheader,nounits",
        )
    )
    return parse_nvidia_smi(result.stdout)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise OptimizationBaselineError(
            f"Cannot read baseline manifest: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise OptimizationBaselineError(f"Baseline manifest is not an object: {path}")
    return value


def load_baseline_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise OptimizationBaselineError("Unsupported optimization baseline schema")
    if manifest.get("optimization_id") != OPTIMIZATION_ID:
        raise OptimizationBaselineError("Unexpected optimization baseline id")
    if manifest.get("status") != "passed":
        raise OptimizationBaselineError("Optimization baseline has not passed")
    return manifest


def validate_locked_contract(
    baseline_path: Path,
    *,
    repository_root: Path,
    storage_root: Path,
    spacing_xyz: Sequence[float] = NORM_SPACING_XYZ,
    seed: int = OPTIMIZATION_SEED,
    coordinate_space: str = COORD_SPACE_RAW_ITK,
    matching_modes: Sequence[str] = MATCHING_MODES,
    required_profile: str = "preprocess",
) -> dict[str, Any]:
    """Validate current inputs against Stage 0 before a later stage runs."""
    manifest = load_baseline_manifest(baseline_path)
    locked = manifest["scientific_contract"]
    expected_values = {
        "norm_spacing_xyz_mm": [float(value) for value in spacing_xyz],
        "seed": int(seed),
        "coordinate_space": coordinate_space,
        "matching_modes": list(matching_modes),
    }
    for key, expected in expected_values.items():
        if locked.get(key) != expected:
            raise OptimizationBaselineError(
                f"Locked contract mismatch for {key}: "
                f"baseline={locked.get(key)!r}, requested={expected!r}"
            )

    repository = inspect_repository(repository_root)
    baseline_commit = manifest["repository"]["execution_commit"]
    ancestor_result = _run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            baseline_commit,
            repository["execution_commit"],
        ),
        cwd=Path(repository_root),
        check=False,
    )
    if ancestor_result.returncode != 0:
        raise OptimizationBaselineError(
            "Current repository no longer descends from the Stage 0 execution commit"
        )

    layout = canonical_layout(storage_root)
    config = file_identity(Path(repository_root) / CONFIG_RELATIVE_PATH)
    checkpoint = file_identity(layout["uae_models"] / CHECKPOINT_FILENAME)
    if config["sha256"] != manifest["model"]["config"]["sha256"]:
        raise OptimizationBaselineError("UAE-S configuration changed after Stage 0")
    if checkpoint["sha256"] != manifest["model"]["checkpoint"]["sha256"]:
        raise OptimizationBaselineError("UAE-S checkpoint changed after Stage 0")

    environment = inspect_environment_profiles(storage_root)
    baseline_profiles = manifest["environment"]["profiles"]
    if required_profile not in baseline_profiles:
        raise OptimizationBaselineError(
            f"Required profile was not frozen by Stage 0: {required_profile}"
        )
    current_profile = environment["profiles"][required_profile]
    baseline_profile = baseline_profiles[required_profile]
    for key in ("image_ref", "image_digest"):
        if current_profile[key] != baseline_profile[key]:
            raise OptimizationBaselineError(
                f"{required_profile} environment changed for {key}"
            )
    inspect_active_runtime(required_profile, storage_root)
    return manifest


def build_baseline_manifest(
    *,
    repository: dict[str, Any],
    model: dict[str, Any],
    assets: dict[str, Any],
    environment: dict[str, Any],
    active_runtime: dict[str, Any],
    gpu: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "optimization_id": OPTIMIZATION_ID,
        "status": "passed",
        "created_at": utc_now(),
        "repository": repository,
        "model": model,
        "data": assets,
        "environment": {
            **environment,
            "active_runtime": active_runtime,
            "gpu": gpu,
        },
        "scientific_contract": {
            "model_profile": "uae_s",
            "norm_spacing_xyz_mm": list(NORM_SPACING_XYZ),
            "seed": OPTIMIZATION_SEED,
            "coordinate_space": COORD_SPACE_RAW_ITK,
            "primary_error_unit": PRIMARY_ERROR_UNIT,
            "matching_modes": list(MATCHING_MODES),
        },
        "candidate_policy": {
            "spatial_candidates": [
                "uncropped_whole_body",
                "padded_body_envelope",
                "organ_groups",
            ],
            "primary_precision_candidates": ["fp32", "amp"],
            "conditional_precision_candidate": {
                "name": "full_fp16",
                "run_only_when": "amp_failed_due_to_out_of_memory",
            },
            "encoder_tiling_in_scope": False,
        },
        "memory_protocol": {
            "fresh_subprocess_per_candidate": True,
            "cuda_synchronize_before_and_after_measurement": True,
            "reset_peak_memory_stats_before_measurement": True,
            "metrics": [
                "torch_peak_allocated_bytes",
                "torch_peak_reserved_bytes",
                "process_gpu_memory_bytes",
                "wall_time_seconds",
                "failure_classification",
            ],
            "vram_usage_ceiling_fraction": VRAM_USAGE_CEILING_FRACTION,
            "minimum_vram_headroom_fraction": (
                1.0 - VRAM_USAGE_CEILING_FRACTION
            ),
            "oom_policy": "record_and_stop_candidate_without_silent_fallback",
        },
        "scope": {
            "body_crop_audit_subjects": "quadra_hc_021-through-quadra_hc_048",
            "body_crop_audit_scans": 56,
            "cycle_error_pilot": "one_largest_test_retest_pair",
            "full_cohort_cycle_error": False,
            "segmentation": False,
            "scientific_result_generation_in_stage0": False,
            "ct_decompression_in_stage0": False,
            "model_loading_in_stage0": False,
        },
    }


def capture_baseline(
    *,
    repository_root: Path,
    storage_root: Path,
    output_root: Path,
    run_id: str,
) -> Path:
    repository_root = Path(repository_root).resolve()
    storage_root = validate_storage_root(Path(storage_root)).resolve()
    output_root = Path(output_root).resolve()
    if not _is_within(output_root, storage_root):
        raise OptimizationBaselineError(
            f"Output root must stay inside the Quadra storage root: {output_root}"
        )
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise OptimizationBaselineError(
            "Run id must start with 'stage0-' and contain only letters, numbers, "
            "periods, underscores or hyphens"
        )

    repository = inspect_repository(repository_root)
    layout = canonical_layout(storage_root)
    config = file_identity(repository_root / CONFIG_RELATIVE_PATH)
    checkpoint = file_identity(layout["uae_models"] / CHECKPOINT_FILENAME)
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise OptimizationBaselineError(
            "UAE-S checkpoint checksum mismatch: "
            f"{checkpoint['sha256']} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    assets = inspect_assets(storage_root)
    environment = inspect_environment_profiles(storage_root)
    active_runtime = inspect_active_preprocess_runtime(storage_root)
    gpu = inspect_gpu()
    model = {
        "profile": "uae_s",
        "config": config,
        "checkpoint": checkpoint,
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    manifest = build_baseline_manifest(
        repository=repository,
        model=model,
        assets=assets,
        environment=environment,
        active_runtime=active_runtime,
        gpu=gpu,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = output_root / run_id
    try:
        run_directory.mkdir()
    except FileExistsError as exc:
        raise OptimizationBaselineError(
            f"Refusing to reuse existing Stage 0 directory: {run_directory}"
        ) from exc
    manifest_path = run_directory / "baseline_manifest.json"
    atomic_write_json(manifest_path, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": 0,
        "status": "PASS",
        "created_at": utc_now(),
        "optimization_id": OPTIMIZATION_ID,
        "baseline_manifest": file_identity(manifest_path),
        "gates": {
            "accepted_base_ancestor": True,
            "repository_clean": True,
            "config_verified": True,
            "checkpoint_verified": True,
            "canonical_asset_counts_verified": True,
            "preprocess_profile_verified": True,
            "gpu_verified": True,
            "optimization_contract_frozen": True,
            "ct_or_model_computation_launched": False,
        },
        "next_stage": "body_envelope_audit",
    }
    atomic_write_json(run_directory / "checkpoint_summary.json", summary)
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path(os.environ.get("QUADRA_STORAGE_ROOT", "/workspace/quadra")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <storage-root>/runs/memory_optimization.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Persistent repository checkout; primarily useful for testing.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional deterministic stage0-* id; defaults to a UTC timestamp.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = args.output_root
    if output_root is None:
        output_root = args.storage_root / "runs/memory_optimization"
    try:
        run_directory = capture_baseline(
            repository_root=args.repository_root,
            storage_root=args.storage_root,
            output_root=output_root,
            run_id=args.run_id or default_run_id(),
        )
    except OptimizationBaselineError as exc:
        parser.exit(2, f"Stage 0 failed: {exc}\n")
    print("Stage 0 PASS")
    print(f"Run directory: {run_directory}")
    print(f"Baseline manifest: {run_directory / 'baseline_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
