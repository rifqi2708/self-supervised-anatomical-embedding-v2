#!/usr/bin/env python3
"""Aggregate compatible Quadra streaming-validation runs into Markdown.

The command consumes explicit output directories produced by
``validate_streaming_equivalence.py``.  It validates that every run used the
same checkpoint and engineering configuration before pooling raw
correspondence rows.  Per-subject results are retained so pooled measurements
cannot conceal a subject-specific failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REQUIRED_FILES = (
    "validation_manifest.json",
    "validation_summary.json",
    "matcher_equivalence.csv",
    "descriptor_summary.csv",
    "correspondence_comparison.csv",
    "frozen_query_points.csv",
)
COMPATIBILITY_FIELDS = (
    "norm_spacing_xyz",
    "dense_crop_size_xyz",
    "baseline_tile_size_xyz",
    "baseline_halo_xyz",
    "expanded_tile_size_xyz",
    "expanded_halo_xyz",
    "match_chunk_xyz",
    "query_batch_size",
    "num_points_per_organ_full",
    "num_points_per_organ_crop",
    "seed",
    "organs",
)
OUTLIER_THRESHOLDS_MM = (2.0, 4.0, 10.0, 20.0)
SEAM_BAND_MM = 4.0
CROP_COMPARISON = "dense_vs_expanded_tiled"
FULL_COMPARISON = "baseline_vs_expanded_tiled"


@dataclass(frozen=True)
class ValidationRun:
    path: Path
    manifest: dict[str, object]
    summary: dict[str, object]
    matcher_rows: list[dict[str, str]]
    descriptor_rows: list[dict[str, str]]
    correspondence_rows: list[dict[str, str]]
    frozen_rows: list[dict[str, str]]

    @property
    def subject(self) -> str:
        return str(self.manifest["subject_id"])


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid numeric field {key!r} in row {row}") from exc


def percentile(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, quantile))


def status_complete(manifest: dict[str, object], phase: str) -> bool:
    status = manifest.get("phase_status", {}).get(phase, {})
    return isinstance(status, dict) and status.get("status") == "complete"


def validate_run_rows(run: ValidationRun) -> None:
    manifest = run.manifest
    organs = tuple(str(value) for value in manifest["organs"])
    crop_points = int(manifest["num_points_per_organ_crop"])
    full_points = int(manifest["num_points_per_organ_full"])
    expected_crop_queries = len(organs) * crop_points
    expected_full_queries = len(organs) * full_points

    if not status_complete(manifest, "crop"):
        raise ValueError(f"{run.path}: crop phase is not complete")
    full_status = manifest.get("phase_status", {}).get("full", {})
    if not isinstance(full_status, dict) or full_status.get("status") not in (
        "complete",
        "blocked_full_expanded_oom",
    ):
        raise ValueError(f"{run.path}: full phase is neither complete nor an explicit expanded-tile OOM")

    if len(run.matcher_rows) != 2 * expected_crop_queries:
        raise ValueError(
            f"{run.path}: expected {2 * expected_crop_queries} matcher rows, got {len(run.matcher_rows)}"
        )
    crop_correspondence = [row for row in run.correspondence_rows if row.get("phase") == "crop"]
    if len(crop_correspondence) != 2 * expected_crop_queries:
        raise ValueError(
            f"{run.path}: expected {2 * expected_crop_queries} crop correspondence rows, "
            f"got {len(crop_correspondence)}"
        )
    crop_primary = [row for row in crop_correspondence if row.get("comparison") == CROP_COMPARISON]
    if len(crop_primary) != expected_crop_queries:
        raise ValueError(
            f"{run.path}: expected {expected_crop_queries} crop {CROP_COMPARISON!r} rows, "
            f"got {len(crop_primary)}; the run may mix validator schemas and must be rerun"
        )
    crop_frozen = [row for row in run.frozen_rows if row.get("phase") == "crop"]
    if len(crop_frozen) != expected_crop_queries:
        raise ValueError(
            f"{run.path}: expected {expected_crop_queries} crop frozen points, got {len(crop_frozen)}"
        )

    if full_status.get("status") == "complete":
        full_correspondence = [row for row in run.correspondence_rows if row.get("phase") == "full"]
        full_frozen = [row for row in run.frozen_rows if row.get("phase") == "full"]
        if len(full_correspondence) != expected_full_queries:
            raise ValueError(
                f"{run.path}: expected {expected_full_queries} full correspondence rows, "
                f"got {len(full_correspondence)}"
            )
        full_primary = [row for row in full_correspondence if row.get("comparison") == FULL_COMPARISON]
        if len(full_primary) != expected_full_queries:
            raise ValueError(
                f"{run.path}: expected {expected_full_queries} full {FULL_COMPARISON!r} rows, "
                f"got {len(full_primary)}; the run may mix validator schemas and must be rerun"
            )
        if len(full_frozen) != expected_full_queries:
            raise ValueError(
                f"{run.path}: expected {expected_full_queries} full frozen points, got {len(full_frozen)}"
            )

    required_descriptor_comparisons = {
        "dense_vs_tiled_fp16",
        "dense_vs_expanded_fp16",
        "baseline_vs_expanded_fp16",
    }
    present = {row.get("comparison") for row in run.descriptor_rows}
    missing = required_descriptor_comparisons - present
    if missing:
        raise ValueError(f"{run.path}: descriptor comparisons are missing: {sorted(missing)}")


def load_run(path: Path) -> ValidationRun:
    resolved = path.expanduser().resolve()
    for name in REQUIRED_FILES:
        candidate = resolved / name
        if not candidate.is_file():
            raise FileNotFoundError(f"Required validation artifact is missing: {candidate}")
    run = ValidationRun(
        path=resolved,
        manifest=read_json(resolved / "validation_manifest.json"),
        summary=read_json(resolved / "validation_summary.json"),
        matcher_rows=read_csv(resolved / "matcher_equivalence.csv"),
        descriptor_rows=read_csv(resolved / "descriptor_summary.csv"),
        correspondence_rows=read_csv(resolved / "correspondence_comparison.csv"),
        frozen_rows=read_csv(resolved / "frozen_query_points.csv"),
    )
    if not run.manifest.get("subject_id"):
        raise ValueError(f"{resolved}: manifest does not identify a subject")
    validate_run_rows(run)
    return run


def nested_sha(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, dict) or not value.get("sha256"):
        raise ValueError(f"Manifest is missing {key}.sha256")
    return str(value["sha256"])


def validate_compatible_runs(runs: Sequence[ValidationRun]) -> None:
    if not runs:
        raise ValueError("At least one --run-dir is required")
    subjects = [run.subject for run in runs]
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Duplicate subjects are not allowed: {subjects}")
    reference = runs[0].manifest
    for run in runs[1:]:
        for field in COMPATIBILITY_FIELDS:
            if run.manifest.get(field) != reference.get(field):
                raise ValueError(
                    f"Incompatible validation runs: {field} differs between "
                    f"{runs[0].subject} and {run.subject}"
                )
        for identity in ("checkpoint", "config"):
            if nested_sha(run.manifest, identity) != nested_sha(reference, identity):
                raise ValueError(
                    f"Incompatible validation runs: {identity} hash differs between "
                    f"{runs[0].subject} and {run.subject}"
                )


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def correspondence_values(rows: Sequence[dict[str, str]]) -> dict[str, np.ndarray]:
    forward = np.asarray([number(row, "forward_displacement_mm") for row in rows], dtype=np.float64)
    backward = np.asarray([number(row, "backward_displacement_mm") for row in rows], dtype=np.float64)
    cycle = np.asarray([number(row, "cycle_error_abs_delta_mm") for row in rows], dtype=np.float64)
    return {"directional": np.concatenate((forward, backward)), "cycle": cycle}


def correspondence_metrics(rows: Sequence[dict[str, str]]) -> dict[str, float | int]:
    values = correspondence_values(rows)
    directional = values["directional"]
    cycle = values["cycle"]
    metrics: dict[str, float | int] = {
        "queries": len(rows),
        "directions": int(directional.size),
        "within_2mm": float(np.mean(directional <= 2.0)) if directional.size else float("nan"),
        "displacement_median": percentile(directional, 50),
        "displacement_p95": percentile(directional, 95),
        "displacement_max": float(np.max(directional)) if directional.size else float("nan"),
        "cycle_median": percentile(cycle, 50),
        "cycle_p95": percentile(cycle, 95),
        "cycle_max": float(np.max(cycle)) if cycle.size else float("nan"),
    }
    for threshold in OUTLIER_THRESHOLDS_MM:
        label = str(int(threshold))
        metrics[f"displacement_gt_{label}"] = int(np.count_nonzero(directional > threshold))
        metrics[f"cycle_gt_{label}"] = int(np.count_nonzero(cycle > threshold))
    return metrics


def descriptor_metrics(rows: Sequence[dict[str, str]], comparison: str) -> dict[str, float]:
    relevant = [
        row
        for row in rows
        if row.get("phase") == "crop" and row.get("comparison") == comparison
    ]
    all_rows = [row for row in relevant if row.get("region") == "all"]
    if not all_rows:
        raise ValueError(f"No crop descriptor rows found for {comparison}")
    seam = {}
    interior = {}
    for row in relevant:
        key = (row.get("timepoint"), row.get("organ"), row.get("level"))
        if row.get("region") == "seam":
            seam[key] = number(row, "cosine_median")
        elif row.get("region") == "interior":
            interior[key] = number(row, "cosine_median")
    drops = [interior[key] - value for key, value in seam.items() if key in interior]
    return {
        "worst_median_cosine": min(number(row, "cosine_median") for row in all_rows),
        "worst_p01_cosine": min(number(row, "cosine_p01") for row in all_rows),
        "worst_seam_drop": max(drops) if drops else float("nan"),
    }


def matcher_metrics(rows: Sequence[dict[str, str]]) -> dict[str, float | int]:
    differences = np.asarray([number(row, "score_abs_diff") for row in rows], dtype=np.float64)
    exact = sum(bool_value(row.get("coordinate_match")) for row in rows)
    return {
        "comparisons": len(rows),
        "exact": exact,
        "coordinate_rate": exact / len(rows) if rows else float("nan"),
        "score_median": percentile(differences, 50),
        "score_p95": percentile(differences, 95),
        "score_max": float(np.max(differences)) if differences.size else float("nan"),
    }


def seam_metrics(rows: Sequence[dict[str, str]]) -> dict[str, float | int]:
    near_rows = []
    far_rows = []
    for row in rows:
        distance = min(number(row, "query_seam_distance_mm"), number(row, "candidate_target_seam_distance_mm"))
        (near_rows if distance <= SEAM_BAND_MM else far_rows).append(row)

    def outlier_rate(selected: Sequence[dict[str, str]]) -> tuple[int, int, float]:
        outliers = sum(
            max(number(row, "forward_displacement_mm"), number(row, "backward_displacement_mm")) > 4.0
            for row in selected
        )
        return outliers, len(selected), outliers / len(selected) if selected else float("nan")

    near_outliers, near_total, near_rate = outlier_rate(near_rows)
    far_outliers, far_total, far_rate = outlier_rate(far_rows)
    risk_ratio = near_rate / far_rate if far_rate > 0.0 else (float("inf") if near_rate > 0.0 else float("nan"))
    return {
        "near_outliers": near_outliers,
        "near_total": near_total,
        "near_rate": near_rate,
        "far_outliers": far_outliers,
        "far_total": far_total,
        "far_rate": far_rate,
        "risk_ratio": risk_ratio,
    }


def selected_rows(run: ValidationRun, phase: str, comparison: str | None = None, organ: str | None = None):
    rows = [row for row in run.correspondence_rows if row.get("phase") == phase]
    if comparison is not None:
        rows = [row for row in rows if row.get("comparison") == comparison]
    if organ is not None:
        rows = [row for row in rows if row.get("organ") == organ]
    return rows


def run_metrics(run: ValidationRun) -> dict[str, object]:
    crop_rows = selected_rows(run, "crop", CROP_COMPARISON)
    full_rows = selected_rows(run, "full", FULL_COMPARISON)
    return {
        "subject": run.subject,
        "matcher": matcher_metrics(run.matcher_rows),
        "descriptor": descriptor_metrics(run.descriptor_rows, "dense_vs_expanded_fp16"),
        "crop": correspondence_metrics(crop_rows),
        "crop_seam": seam_metrics(crop_rows),
        "full": correspondence_metrics(full_rows) if full_rows else None,
        "full_seam": seam_metrics(full_rows) if full_rows else None,
        "full_status": run.manifest.get("phase_status", {}).get("full", {}).get("status"),
    }


def pooled_rows(runs: Sequence[ValidationRun], phase: str, comparison: str, organ: str | None = None):
    result = []
    for run in runs:
        result.extend(selected_rows(run, phase, comparison, organ))
    return result


def format_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "NA" if math.isnan(float(value)) else "∞"
    return f"{float(value):.{digits}f}"


def format_percent(value: float, digits: int = 1) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{100.0 * float(value):.{digits}f}%"


def pass_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


def max_gpu_memory_gb(run: ValidationRun) -> float:
    peaks = []
    phase_status = run.manifest.get("phase_status", {})
    crop = phase_status.get("crop", {}) if isinstance(phase_status, dict) else {}
    for profile in crop.get("profiles", []) if isinstance(crop, dict) else []:
        if isinstance(profile, dict) and profile.get("peak_gpu_memory_bytes") is not None:
            peaks.append(float(profile["peak_gpu_memory_bytes"]))
    full = phase_status.get("full", {}) if isinstance(phase_status, dict) else {}
    if isinstance(full, dict):
        for group in ("baseline_manifests", "expanded_manifests", "matching_profiles"):
            values = full.get(group, {})
            if isinstance(values, dict):
                for profile in values.values():
                    if isinstance(profile, dict) and profile.get("peak_gpu_memory_bytes") is not None:
                        peaks.append(float(profile["peak_gpu_memory_bytes"]))
    return max(peaks) / (1024.0**3) if peaks else float("nan")


def total_profile_seconds(run: ValidationRun) -> float:
    seconds = []
    phase_status = run.manifest.get("phase_status", {})
    crop = phase_status.get("crop", {}) if isinstance(phase_status, dict) else {}
    for profile in crop.get("profiles", []) if isinstance(crop, dict) else []:
        if isinstance(profile, dict) and profile.get("seconds") is not None:
            seconds.append(float(profile["seconds"]))
    full = phase_status.get("full", {}) if isinstance(phase_status, dict) else {}
    if isinstance(full, dict):
        for group in ("baseline_manifests", "expanded_manifests", "matching_profiles"):
            values = full.get(group, {})
            if isinstance(values, dict):
                for profile in values.values():
                    if not isinstance(profile, dict):
                        continue
                    for key in ("generation_seconds", "seconds"):
                        if profile.get(key) is not None:
                            seconds.append(float(profile[key]))
                            break
    return float(sum(seconds))


def worst_heatmap(runs: Sequence[ValidationRun], phase: str):
    candidates = []
    for run in runs:
        for row in run.descriptor_rows:
            if row.get("phase") != phase or row.get("region") != "all":
                continue
            if phase == "crop" and row.get("comparison") != "dense_vs_expanded_fp16":
                continue
            if phase == "full" and row.get("comparison") != "baseline_vs_expanded_fp16":
                continue
            candidates.append((number(row, "cosine_p01"), run, row))
    return min(candidates, key=lambda value: value[0]) if candidates else None


def copy_selected_heatmaps(runs: Sequence[ValidationRun], output: Path) -> dict[str, str]:
    figure_dir = output.parent / f"{output.stem}_figures"
    links = {}
    for phase in ("crop", "full"):
        selected = worst_heatmap(runs, phase)
        if selected is None:
            continue
        _, run, row = selected
        if phase == "crop":
            source_name = (
                f"crop_{row['timepoint']}_{row['organ']}_{row['level']}_dense_vs_expanded.png"
            )
        else:
            source_name = f"full_{row['timepoint']}_{row['level']}_baseline_vs_expanded.png"
        source = run.path / "figures" / source_name
        if not source.is_file():
            raise FileNotFoundError(f"Selected discrepancy heatmap is missing: {source}")
        figure_dir.mkdir(parents=True, exist_ok=True)
        destination = figure_dir / f"{run.subject}_{source_name}"
        shutil.copy2(source, destination)
        links[phase] = destination.relative_to(output.parent).as_posix()
    return links


def aggregate(runs: Sequence[ValidationRun]) -> dict[str, object]:
    validate_compatible_runs(runs)
    per_subject = [run_metrics(run) for run in runs]
    crop_rows = pooled_rows(runs, "crop", CROP_COMPARISON)
    full_rows = pooled_rows(runs, "full", FULL_COMPARISON)
    organs = tuple(str(value) for value in runs[0].manifest["organs"])
    all_matcher = [row for run in runs for row in run.matcher_rows]
    return {
        "per_subject": per_subject,
        "matcher": matcher_metrics(all_matcher),
        "crop": correspondence_metrics(crop_rows),
        "crop_seam": seam_metrics(crop_rows),
        "full": correspondence_metrics(full_rows) if full_rows else None,
        "full_seam": seam_metrics(full_rows) if full_rows else None,
        "crop_organs": {
            organ: correspondence_metrics(pooled_rows(runs, "crop", CROP_COMPARISON, organ))
            for organ in organs
        },
        "full_organs": {
            organ: correspondence_metrics(pooled_rows(runs, "full", FULL_COMPARISON, organ))
            for organ in organs
            if pooled_rows(runs, "full", FULL_COMPARISON, organ)
        },
    }


def render_report(
    runs: Sequence[ValidationRun],
    aggregate_data: dict[str, object],
    heatmaps: dict[str, str],
    validation_commit: str | None = None,
) -> str:
    manifest = runs[0].manifest
    subjects = [run.subject for run in runs]
    matcher = aggregate_data["matcher"]
    crop = aggregate_data["crop"]
    full = aggregate_data["full"]
    per_subject = aggregate_data["per_subject"]
    exact_pass = matcher["exact"] == matcher["comparisons"]
    crop_pass = (
        crop["within_2mm"] >= 0.95
        and crop["cycle_median"] <= 1.0
        and crop["cycle_p95"] <= 4.0
        and all(
            item["descriptor"]["worst_median_cosine"] >= 0.99
            and item["descriptor"]["worst_p01_cosine"] >= 0.95
            and item["descriptor"]["worst_seam_drop"] <= 0.01
            for item in per_subject
        )
    )
    full_pass = bool(
        full
        and full["displacement_median"] <= 2.0
        and full["displacement_p95"] <= 4.0
        and full["cycle_median"] <= 1.0
        and full["cycle_p95"] <= 4.0
    )
    conclusion = (
        "The tested tile workflow is supported as not materially different from the bounded dense reference "
        "under the prespecified engineering criteria."
        if exact_pass and crop_pass and full_pass
        else "The tested tile workflow is not uniformly supported by all prespecified engineering criteria; "
        "the failed dimensions below must be considered before production use."
    )
    checkpoint = manifest["checkpoint"]
    checkpoint_hash = str(checkpoint["sha256"])
    spacing = tuple(float(value) for value in manifest["norm_spacing_xyz"])
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    full_completed = sum(item["full"] is not None for item in per_subject)

    lines = [
        "# Quadra tiled-embedding and streaming-matching validation",
        "",
        "## Technical summary",
        "",
        f"**{conclusion}** The assessment covers {len(runs)} Test–Retest subjects "
        f"(`{'`, `'.join(subjects)}`) resampled to exact `{spacing[0]:g} mm` isotropic spacing.",
        "",
        f"- Streamed and dense matching on identical embeddings returned the same argmax coordinate in "
        f"**{matcher['exact']}/{matcher['comparisons']} comparisons**: **{pass_text(exact_pass)}**.",
        f"- Dense versus expanded-tile crop correspondence: **{format_percent(crop['within_2mm'])}** of "
        f"directional matches were within 2 mm; cycle-error difference median/p95 was "
        f"**{format_float(crop['cycle_median'])}/{format_float(crop['cycle_p95'])} mm**: "
        f"**{pass_text(crop_pass)}**.",
        f"- Baseline-versus-expanded full-subject halo stability was available for "
        f"**{full_completed}/{len(runs)} subjects**; pooled correspondence displacement median/p95 was "
        f"**{format_float(full['displacement_median'])}/{format_float(full['displacement_p95'])} mm** "
        f"and cycle-error change median/p95 was "
        f"**{format_float(full['cycle_median'])}/{format_float(full['cycle_p95'])} mm**: "
        f"**{pass_text(full_pass)}**."
        if full
        else "- No complete full-subject halo comparison was available.",
        "- Similarity-score differences are reported as a numerical diagnostic only; exact argmax coordinates are "
        "the hard streamed-matcher correctness criterion.",
        "",
        "## Definitions and tested geometry",
        "",
        "All tensor sizes use `(x, y, z)` order in the resampled 2 mm voxel grid.",
        "",
        "- **Dense inference:** one bounded organ-centred crop is passed through the encoder in a single operation, "
        "so the crop has no internal tile boundaries.",
        "- **Tile:** one encoder input window extracted from the resampled volume.",
        "- **Halo:** contextual voxels surrounding the retained centre of a tile. Halo voxels influence the "
        "descriptor but are discarded from that tile's output; the listed halo applies on each side of every axis.",
        "- **Retained core:** the central tile region copied into the global embedding cache. Adjacent cores cover the "
        "subject without averaging descriptors.",
        "- **Seam:** a plane where adjacent retained cores meet. It is not an image discontinuity, but points close to "
        "a seam may have been encoded with different contextual windows.",
        "- **Streamed global matching:** similarity is evaluated over every target voxel in memory-bounded chunks while "
        "retaining the global maximum. Chunk size changes memory scheduling, not the anatomical search range.",
        "",
        "| Configuration | Encoder input | Halo on each side | Retained core |",
        "|---|---:|---:|---:|",
        "| Dense crop | `128×128×64` = `256×256×128 mm` | None | Entire bounded crop |",
        "| Baseline tile | `128×128×64` = `256×256×128 mm` | `32×32×16` = `64×64×32 mm` | `64×64×32` = `128×128×64 mm` |",
        "| Expanded tile | `160×160×80` = `320×320×160 mm` | `48×48×24` = `96×96×48 mm` | `64×64×32` = `128×128×64 mm` |",
        "",
        "## Experimental design",
        "",
        f"The engineering run used `{Path(str(checkpoint['path'])).name}` with SHA-256 "
        f"`{checkpoint_hash}`, configuration SHA-256 `{nested_sha(manifest, 'config')}`, and fixed random seed "
        f"`{manifest['seed']}`. The original SAM checkpoint is used to test implementation behaviour, not to claim "
        "scientific performance of the Quadra fine-tuned model.",
        f"The validator source commit was `{validation_commit}`. This identifier is supplied explicitly during "
        "aggregation because schema-version-1 manifests do not embed a Git commit."
        if validation_commit
        else "The schema-version-1 manifests do not embed a Git commit; no external validator commit was supplied "
        "during aggregation.",
        "",
        f"For every subject, {len(manifest['organs'])} organs "
        f"(`{'`, `'.join(str(value) for value in manifest['organs'])}`) were tested. "
        f"The bounded crop phase used {manifest['num_points_per_organ_crop']} queries per organ "
        f"({len(manifest['organs']) * int(manifest['num_points_per_organ_crop'])} per subject). The full phase used "
        f"{manifest['num_points_per_organ_full']} queries per organ "
        f"({len(manifest['organs']) * int(manifest['num_points_per_organ_full'])} per subject). Forward and backward displacement values "
        "are both included; cycle-error change contributes one value per query.",
        "",
        "The validation separates three questions:",
        "",
        "1. Dense and streamed search consume identical embeddings. Exact argmax coordinates test the streaming implementation.",
        "2. Dense and tiled encoders consume the same normalized organ crop. Descriptor and correspondence changes test tile-context effects.",
        "3. The complete subject is encoded with baseline and expanded halos. Their differences test halo sensitivity while both retain unrestricted global search.",
        "",
        "## Streamed matching is coordinate-exact",
        "",
        f"Across {matcher['comparisons']} forward/backward crop comparisons, the coordinate agreement rate was "
        f"**{format_percent(matcher['coordinate_rate'])}**. Score absolute-difference median/p95/max was "
        f"`{matcher['score_median']:.3e}` / `{matcher['score_p95']:.3e}` / `{matcher['score_max']:.3e}`. "
        "The score differences arise from floating-point operation ordering and do not change the selected coordinate "
        "in this test.",
        "",
        "| Subject | Exact coordinates | Score-difference p95 | Score-difference max |",
        "|---|---:|---:|---:|",
    ]
    for item in per_subject:
        metric = item["matcher"]
        lines.append(
            f"| `{item['subject']}` | {metric['exact']}/{metric['comparisons']} | "
            f"{metric['score_p95']:.3e} | {metric['score_max']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## Expanded tiling remains close to dense crop inference",
            "",
            "The expanded configuration is the primary candidate because it increases anatomical context without "
            "changing the retained core or global search space. Descriptor thresholds are evaluated separately for "
            "each subject; correspondence values below are calculated from raw query rows rather than averaging "
            "subject medians.",
            "",
            "| Subject | Worst median cosine | Worst p01 cosine | Worst seam drop | Directions ≤2 mm | Displacement p95 (mm) | Cycle Δ p95 (mm) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in per_subject:
        descriptor = item["descriptor"]
        metric = item["crop"]
        lines.append(
            f"| `{item['subject']}` | {format_float(descriptor['worst_median_cosine'], 6)} | "
            f"{format_float(descriptor['worst_p01_cosine'], 6)} | "
            f"{format_float(descriptor['worst_seam_drop'], 6)} | {format_percent(metric['within_2mm'])} | "
            f"{format_float(metric['displacement_p95'])} | {format_float(metric['cycle_p95'])} |"
        )
    lines.extend(
        [
            f"| **Pooled** | — | — | — | **{format_percent(crop['within_2mm'])}** | "
            f"**{format_float(crop['displacement_p95'])}** | **{format_float(crop['cycle_p95'])}** |",
            "",
        ]
    )
    if heatmaps.get("crop"):
        lines.extend(
            [
                "The following heatmap is the crop/feature-level case with the lowest first-percentile descriptor "
                "cosine. It shows `1 - cosine similarity`; it is included to localize the worst numerical result, "
                "not as a representative average.",
                "",
                f"![Worst dense-versus-expanded crop descriptor discrepancy]({heatmaps['crop']})",
                "",
            ]
        )

    lines.extend(
        [
            "## Full-subject results are a halo-sensitivity test",
            "",
            "A complete dense 2 mm subject is intentionally not materialized. Therefore this comparison cannot prove "
            "full-volume dense equivalence; it tests whether increasing halo context materially changes the final "
            "correspondences.",
            "",
            "| Subject | Full status | Directions ≤2 mm | Displacement median/p95 (mm) | Cycle Δ median/p95 (mm) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in per_subject:
        metric = item["full"]
        if metric is None:
            lines.append(f"| `{item['subject']}` | `{item['full_status']}` | NA | NA | NA |")
        else:
            lines.append(
                f"| `{item['subject']}` | `{item['full_status']}` | {format_percent(metric['within_2mm'])} | "
                f"{format_float(metric['displacement_median'])}/{format_float(metric['displacement_p95'])} | "
                f"{format_float(metric['cycle_median'])}/{format_float(metric['cycle_p95'])} |"
            )
    if full:
        lines.append(
            f"| **Pooled** | {full_completed}/{len(runs)} complete | **{format_percent(full['within_2mm'])}** | "
            f"**{format_float(full['displacement_median'])}/{format_float(full['displacement_p95'])}** | "
            f"**{format_float(full['cycle_median'])}/{format_float(full['cycle_p95'])}** |"
        )
    lines.append("")
    if heatmaps.get("full"):
        lines.extend(
            [
                "The selected full-subject heatmap is the subject/timepoint/feature level with the lowest "
                "first-percentile descriptor cosine between baseline and expanded halos.",
                "",
                f"![Worst baseline-versus-expanded full-subject descriptor discrepancy]({heatmaps['full']})",
                "",
            ]
        )

    lines.extend(
        [
            "## Organ-level sensitivity and outliers",
            "",
            "Colon is displayed explicitly because large physiological displacement is expected, but it uses the same "
            "sampling and thresholds as every other organ. Isolated model mismatches are retained rather than removed.",
            "",
            "| Organ | Crop directions ≤2 mm | Crop displacement p95 | Crop cycle Δ p95 | Full directions ≤2 mm | Full displacement p95 | Full cycle Δ p95 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for organ, crop_organ in aggregate_data["crop_organs"].items():
        full_organ = aggregate_data["full_organs"].get(organ)
        full_values = (
            f"{format_percent(full_organ['within_2mm'])} | {format_float(full_organ['displacement_p95'])} | "
            f"{format_float(full_organ['cycle_p95'])}"
            if full_organ
            else "NA | NA | NA"
        )
        label = f"**{organ}**" if organ == "colon" else organ
        lines.append(
            f"| {label} | {format_percent(crop_organ['within_2mm'])} | "
            f"{format_float(crop_organ['displacement_p95'])} | {format_float(crop_organ['cycle_p95'])} | "
            f"{full_values} |"
        )

    lines.extend(
        [
            "",
            "Counts below use strict `>` thresholds. Directional counts combine forward and backward comparisons; "
            "cycle counts contain one value per query.",
            "",
            "| Comparison | Denominator | >2 mm | >4 mm | >10 mm | >20 mm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, metric in (("Crop displacement", crop), ("Crop cycle Δ", crop)):
        prefix = "displacement" if "displacement" in label.lower() else "cycle"
        denominator = crop["directions"] if prefix == "displacement" else crop["queries"]
        lines.append(
            f"| {label} | {denominator} | {metric[prefix + '_gt_2']} | {metric[prefix + '_gt_4']} | "
            f"{metric[prefix + '_gt_10']} | {metric[prefix + '_gt_20']} |"
        )
    if full:
        for label, metric in (("Full displacement", full), ("Full cycle Δ", full)):
            prefix = "displacement" if "displacement" in label.lower() else "cycle"
            denominator = full["directions"] if prefix == "displacement" else full["queries"]
            lines.append(
                f"| {label} | {denominator} | {metric[prefix + '_gt_2']} | {metric[prefix + '_gt_4']} | "
                f"{metric[prefix + '_gt_10']} | {metric[prefix + '_gt_20']} |"
            )

    lines.extend(
        [
            "",
            "## Seam proximity does not by itself establish causation",
            "",
            f"A row is labelled near a seam when either its query or candidate target lies within "
            f"{SEAM_BAND_MM:g} mm of a retained-core seam. A correspondence outlier is a row whose larger "
            "forward/backward displacement exceeds 4 mm.",
            "",
            "| Phase | Near-seam outliers | Far-from-seam outliers | Descriptive risk ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for phase, metric in (("Dense vs expanded crop", aggregate_data["crop_seam"]), ("Baseline vs expanded full", aggregate_data["full_seam"])):
        if metric is None:
            continue
        lines.append(
            f"| {phase} | {metric['near_outliers']}/{metric['near_total']} "
            f"({format_percent(metric['near_rate'])}) | {metric['far_outliers']}/{metric['far_total']} "
            f"({format_percent(metric['far_rate'])}) | {format_float(metric['risk_ratio'], 2)} |"
        )
    lines.extend(
        [
            "",
            "These are descriptive rates, not an inferential test. A higher near-seam rate would justify targeted "
            "follow-up only if it recurs across subjects and is not explained by organ composition or difficult anatomy.",
            "",
            "## Computational profile",
            "",
            "Reported seconds are sums of saved encoder and matcher profiles, not wall-clock job duration; cache reuse "
            "can make them differ from terminal elapsed time.",
            "",
            "| Subject | Recorded profile time (min) | Peak allocated GPU memory (GiB) | GPU |",
            "|---|---:|---:|---|",
        ]
    )
    for run in runs:
        gpu = str(run.manifest.get("environment", {}).get("gpu", "unknown"))
        lines.append(
            f"| `{run.subject}` | {format_float(total_profile_seconds(run) / 60.0, 1)} | "
            f"{format_float(max_gpu_memory_gb(run), 3)} | {gpu} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and recommendation",
            "",
            f"**Overall engineering interpretation:** {conclusion}",
            "",
            "Use the expanded `160×160×80` tile with `48×48×24` halo and unchanged `64×64×32` retained core "
            "only if the matcher, dense-crop, and full-halo criteria above pass. If only isolated organ/subject "
            "outliers remain, retain and report them; do not delete or manually correct them. Reconsider the halo or "
            "perform targeted diagnosis if failures recur in at least two subjects, are materially enriched near seams, "
            "or produce a pooled organ-level p95 above the 4 mm engineering target.",
            "",
            "## Limitations and robustness boundaries",
            "",
            "- The dense reference covers bounded organ-centred crops, not the complete 2 mm body volume.",
            "- Five subjects can expose implementation errors and major context sensitivity but do not establish "
            "population-level or clinical robustness.",
            "- The original `SAM.pth` checkpoint tests this implementation. Fine-tuning may change descriptor context "
            "sensitivity, so checkpoint independence is not established.",
            "- Organ masks define the query sampling regions; the validation does not measure anatomical ground-truth "
            "correspondence accuracy.",
            "- Pooled rows within a subject are correlated. Pooled percentages are engineering summaries, not "
            "independent-sample confidence estimates.",
            "- Zero padding at bounded crop/subject edges is part of the deployed tiling implementation and can affect "
            "edge descriptors.",
            "",
            "## Recommended next steps",
            "",
            "1. Use the selected expanded configuration for the 2 mm production trial if the reported criteria pass.",
            "2. Preserve subject-level output directories and checkpoint hashes with subsequent cycle-error results.",
            "3. Repeat a small checkpoint-sensitivity spot check after the Quadra fine-tuned checkpoint becomes available; "
            "do not assume the original and fine-tuned encoders have identical context sensitivity.",
            "4. Expand the subject cohort only if the five-subject results reveal recurrent organ- or seam-associated failures.",
            "",
            "## Evidence inventory",
            "",
            f"Report generated {generated} from the following validator output directories:",
            "",
        ]
    )
    for run in runs:
        lines.append(f"- `{run.path}`")
    lines.extend(
        [
            "",
            "Each directory was required to contain its manifest, frozen points, matcher rows, descriptor summaries, "
            "correspondence rows, validation summary, and discrepancy figures. Compatibility was verified before rows "
            "were pooled.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Completed validation output directory. Repeat once per subject.",
    )
    parser.add_argument("--output", required=True, help="Destination Markdown report path.")
    parser.add_argument(
        "--validation-commit",
        help="Git commit containing the validator source used for the runs (recorded in the report).",
    )
    parser.add_argument(
        "--no-copy-heatmaps",
        action="store_true",
        help="Do not copy the worst crop and full discrepancy heatmaps next to the report.",
    )
    return parser.parse_args(argv)


def run(args) -> Path:
    runs = [load_run(Path(value)) for value in args.run_dir]
    runs.sort(key=lambda value: value.subject)
    validate_compatible_runs(runs)
    data = aggregate(runs)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    heatmaps = {} if args.no_copy_heatmaps else copy_selected_heatmaps(runs, output)
    output.write_text(
        render_report(runs, data, heatmaps, validation_commit=args.validation_commit),
        encoding="utf-8",
    )
    print(f"Validation report: {output}")
    return output


def main(argv: Iterable[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
