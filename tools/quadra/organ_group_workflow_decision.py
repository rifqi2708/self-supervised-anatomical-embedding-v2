#!/usr/bin/env python
"""Record the Stage 4C human acceptance of known crop-context sensitivity.

This command is deliberately evidence-only.  It never reads a CT, imports the
UAE-S model stack, allocates CUDA memory, or rewrites Stage 4A/4B artifacts.
Both numerical-validation checkpoints remain BLOCKED; Stage 4C merely records
the decision to carry the 100 mm FP32 candidate into match-level validation.
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra import memory_configuration_screen as stage3  # noqa: E402
from tools.quadra import organ_group_numerical_validation as stage4  # noqa: E402


SCHEMA_VERSION = 1
DECISION_ID = "quadra-organ-group-context-limitation-v1"
EXPECTED_BRANCH = "codex/quadra-memory-optimization"
EXPECTED_STAGE4A = stage4.EXPECTED_STAGE4A_CHECKPOINT
EXPECTED_STAGE4B = Path(
    "/workspace/quadra/runs/memory_optimization/"
    "stage4b-resolution-20260811T043333Z/checkpoint_summary.json"
)
RUN_PREFIX = "stage4c-limitation-decision-"
FROZEN_RATIONALE = (
    "The 100 mm FP32 configuration is provisionally retained as the least "
    "computationally expensive configuration that completed the required "
    "extraction workflow. Neither the 100-120 mm nor 120-150 mm experiment "
    "established descriptor invariance. Match-level sensitivity must therefore "
    "pass before the workflow is frozen."
)


class Stage4CError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise Stage4CError("Cannot read JSON {}: {}".format(path, exc))
    if not isinstance(value, dict):
        raise Stage4CError("Expected a JSON object: {}".format(path))
    return value


def atomic_json(path, value, refuse=False):
    path = Path(path)
    if refuse and path.exists():
        raise Stage4CError("Refusing to overwrite existing file: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
    os.replace(str(temporary), str(path))


def file_identity(path):
    try:
        return stage4.file_identity(Path(path))
    except Exception as exc:
        raise Stage4CError(str(exc))


def validate_repository(repository=PROJECT_ROOT):
    branch = stage3.git_output(["symbolic-ref", "--short", "HEAD"], repository)
    commit = stage3.git_output(["rev-parse", "HEAD"], repository)
    dirty = stage3.git_output(["status", "--porcelain"], repository)
    ancestor = subprocess.call(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", "e66ebd5", "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0
    if branch != EXPECTED_BRANCH or dirty or not ancestor:
        raise Stage4CError(
            "Repository contract failed: branch={!r}, clean={}, Stage4B ancestor={}".format(
                branch, not bool(dirty), ancestor
            )
        )
    return {
        "path": str(Path(repository).resolve()),
        "branch": branch,
        "execution_commit": commit,
        "clean": True,
    }


def validate_stage4a(path):
    try:
        return stage4.validate_stage4a_checkpoint(Path(path))
    except Exception as exc:
        raise Stage4CError("Stage 4A validation failed: {}".format(exc))


def _identity_equal(record):
    return isinstance(record, dict) and file_identity(record.get("path")) == record


def validate_stage4b(path, stage4a):
    path = Path(path).resolve()
    if path != EXPECTED_STAGE4B.resolve():
        raise Stage4CError("Unexpected Stage 4B checkpoint path: {}".format(path))
    checkpoint_identity = file_identity(path)
    checkpoint = load_json(path)
    gates = checkpoint.get("gates", {})
    selection_ref = checkpoint.get("selected_configuration")
    if (
        checkpoint.get("stage") != 4
        or checkpoint.get("substage") != "B"
        or checkpoint.get("status") != "BLOCKED"
        or checkpoint.get("validation_id") != stage4.RESOLUTION_VALIDATION_ID
        or checkpoint.get("next_stage") != "resolve_stage4_blocker"
        or checkpoint.get("selected_margin_mm") is not None
        or gates.get("120mm_vs_150mm_boundary_gate_passed") is not False
        or gates.get("matching_or_cycle_error_run") is not False
        or gates.get("full_fp16_used") is not False
        or gates.get("embeddings_or_prepared_volumes_retained") is not False
        or not _identity_equal(selection_ref)
    ):
        raise Stage4CError("Stage 4B checkpoint is not the accepted blocked evidence")
    selection = load_json(selection_ref["path"])
    failures = selection.get("failures", [])
    if (
        selection.get("status") != "BLOCKED"
        or selection.get("selected_spatial_configuration") is not None
        or selection.get("selected_precision") is not None
        or "boundary_sensitivity" not in failures
        or "incomplete_worker_count" not in failures
        or not any("model_error" in str(item) for item in failures)
    ):
        raise Stage4CError("Stage 4B did not preserve its boundary and model-error blockers")
    manifest_path = path.parent / "stage4_manifest.json"
    manifest = load_json(manifest_path)
    source = manifest.get("source_stage4a")
    if (
        manifest.get("validation_id") != stage4.RESOLUTION_VALIDATION_ID
        or manifest.get("status") != "blocked"
        or float(manifest.get("settings", {}).get("selected_margin_mm", -1)) != 120.0
        or float(manifest.get("settings", {}).get("reference_margin_mm", -1)) != 150.0
        or source != stage4a["checkpoint"]
    ):
        raise Stage4CError("Stage 4B manifest lineage changed")
    worker_paths = sorted((path.parent / "worker_results" / "fp32").glob("*.json"))
    workers = [load_json(item) for item in worker_paths]
    model_errors = [item for item in workers if item.get("failure_classification") == "model_error"]
    if not model_errors:
        raise Stage4CError("Stage 4B model-error worker evidence is missing")
    if any(item.get("failure_classification") == "cuda_oom" for item in workers):
        raise Stage4CError("Stage 4B model error must not be reclassified as CUDA OOM")
    if not any("CUDNN_STATUS_NOT_SUPPORTED" in str(item.get("error", "")) for item in model_errors):
        raise Stage4CError("Stage 4B cuDNN failure evidence changed")
    return {
        "checkpoint": checkpoint_identity,
        "selection": selection_ref,
        "manifest": file_identity(manifest_path),
        "worker_results": [file_identity(item) for item in worker_paths],
        "failures": list(failures),
    }


def render_report(decision):
    return "\n".join([
        "# Stage 4C organ-group limitation decision",
        "",
        "## Decision",
        "",
        decision["review_rationale"],
        "",
        "The selected candidate is `organ_group_100mm` with FP32 model compute at 2 mm isotropic spacing.",
        "",
        "## Evidence interpretation",
        "",
        "- Stage 4A remains **BLOCKED**: the 100-120 mm descriptor boundary gate failed.",
        "- Stage 4B remains **BLOCKED**: 120-150 mm did not establish invariance and one worker encountered a cuDNN model error.",
        "- Larger margins therefore provided no demonstrated resolution of the context-sensitivity limitation.",
        "- This is a pragmatic engineering decision, not evidence that the tested margins are equivalent.",
        "",
        "## Prohibited conclusions",
        "",
        "Stage 4 did not pass. Crop-context invariance, matching stability, cycle-error stability, anatomical accuracy, and cohort generalisability have not been established.",
        "",
        "## Required next step",
        "",
        "Run match-level 100-120 mm sensitivity validation before freezing a workflow.",
    ])


def run_accept(args):
    if not args.accept_known_context_sensitivity:
        raise Stage4CError("Explicit --accept-known-context-sensitivity acknowledgement is required")
    rationale = " ".join(str(args.review_rationale or "").split())
    if not rationale:
        raise Stage4CError("A non-empty --review-rationale is required")
    repository = validate_repository(Path(args.repository_root))
    stage4a = validate_stage4a(args.stage4a_checkpoint)
    stage4b = validate_stage4b(args.stage4b_checkpoint, stage4a)
    before = {
        "stage4a": file_identity(args.stage4a_checkpoint),
        "stage4b": file_identity(args.stage4b_checkpoint),
    }
    output_root = Path(args.output_root or Path(args.storage_root) / "runs/memory_optimization")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / (args.run_id or RUN_PREFIX + timestamp)
    if run_dir.exists():
        raise Stage4CError("Refusing to overwrite existing Stage 4C directory: {}".format(run_dir))
    run_dir.mkdir(parents=True)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": DECISION_ID,
        "stage": 4,
        "substage": "C",
        "status": "PROVISIONAL_ACCEPTANCE",
        "decision_type": "human_limitation_acceptance",
        "created_at": utc_now(),
        "repository": repository,
        "review_rationale": rationale,
        "sources": {"stage4a": stage4a, "stage4b": stage4b},
        "selected_candidate": {
            "spatial_configuration": "organ_group_100mm",
            "precision": "fp32",
            "spacing_xyz_mm": [2.0, 2.0, 2.0],
            "coordinate_space": "raw_itk_voxel",
            "subject_id": stage4.EXPECTED_SUBJECT,
            "groups": list(stage3.GROUPS),
        },
        "evidence_interpretation": {
            "stage4a_status_preserved": "BLOCKED",
            "stage4b_status_preserved": "BLOCKED",
            "larger_margin_established_invariance": False,
            "matching_or_cycle_error_validated": False,
        },
        "required_next_validation": "match_level_crop_sensitivity",
        "scientific_scope": {
            "workflow_frozen": False,
            "cohort_authorized": False,
            "ct_model_cuda_or_matching_run": False,
        },
    }
    decision_path = run_dir / "limitation_acceptance.json"
    report_path = run_dir / "decision_report.md"
    atomic_json(decision_path, decision, refuse=True)
    atomic_text(report_path, render_report(decision))
    after = {
        "stage4a": file_identity(args.stage4a_checkpoint),
        "stage4b": file_identity(args.stage4b_checkpoint),
    }
    if before != after:
        raise Stage4CError("A source Stage 4 checkpoint changed while writing Stage 4C")
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": DECISION_ID,
        "stage": 4,
        "substage": "C",
        "status": "PROVISIONAL",
        "created_at": utc_now(),
        "limitation_acceptance": file_identity(decision_path),
        "decision_report": file_identity(report_path),
        "source_checkpoints_unchanged": before == after,
        "gates": {
            "stage4a_blocked_status_preserved": True,
            "stage4b_blocked_status_preserved": True,
            "human_limitation_acceptance_recorded": True,
            "100mm_fp32_technical_extraction_gates_passed": True,
            "descriptor_boundary_invariance_established": False,
            "match_level_validation_complete": False,
            "production_workflow_frozen": False,
            "ct_model_cuda_or_matching_run": False,
        },
        "next_stage": "match_level_crop_sensitivity_validation",
    }
    checkpoint_path = run_dir / "checkpoint_summary.json"
    atomic_json(checkpoint_path, checkpoint, refuse=True)
    print("Stage 4C PROVISIONAL_ACCEPTANCE", flush=True)
    print("Run directory: {}".format(run_dir), flush=True)
    print("Checkpoint: {}".format(checkpoint_path), flush=True)
    return run_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    accept = sub.add_parser("accept", help="Record provisional acceptance without altering Stage 4 evidence.")
    accept.add_argument("--stage4a-checkpoint", default=str(EXPECTED_STAGE4A))
    accept.add_argument("--stage4b-checkpoint", default=str(EXPECTED_STAGE4B))
    accept.add_argument("--accept-known-context-sensitivity", action="store_true")
    accept.add_argument("--review-rationale", default=FROZEN_RATIONALE)
    accept.add_argument("--storage-root", default="/workspace/quadra")
    accept.add_argument("--repository-root", default=str(PROJECT_ROOT))
    accept.add_argument("--output-root", default=None)
    accept.add_argument("--run-id", default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "accept":
            run_accept(args)
    except Stage4CError as exc:
        parser.error("Stage 4C failed: {}".format(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
