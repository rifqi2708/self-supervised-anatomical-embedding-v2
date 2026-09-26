"""Command-line interface for the Quadra TotalSegmentator workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from .core import (
    DEFAULT_REGISTRY,
    DEFAULT_RUN_ID,
    WorkflowError,
    atomic_write_json,
    find_scan,
    load_manifest,
    prepare_manifest,
)
from .workflow import (
    DEFAULT_MIN_FREE_GIB,
    cohort_status,
    preflight,
    run_cohort,
    run_scan,
    validate_cohort,
    write_status_csv,
)
from .resegmentation import (
    ALL_PHASE_ORDER,
    DEFAULT_RESEGMENTATION_PLAN,
    SEGMENTATION_PHASE_ORDER,
    load_resegmentation_manifest,
    prepare_resegmentation_manifest,
    resegmentation_preflight,
    resegmentation_status,
    run_case as run_resegmentation_case,
    run_phase as run_resegmentation_phase,
    stage_selected_inputs,
    validate_resegmentation_outputs,
)
from tools.quadra.environment import resolve_quadra_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = resolve_quadra_path(
    None, "whole_body_ct", PROJECT_ROOT / "data" / "QUADRA_HC_WB"
)
DEFAULT_DEMOGRAPHICS = resolve_quadra_path(
    None,
    "manifests",
    PROJECT_ROOT / "data",
) / "Demographics (All).xlsx"
DEFAULT_RUNPOD_ROOT = resolve_quadra_path(
    None, "runs_preprocessing", Path("/workspace/quadra-totalsegmentator")
)


def _common_registry(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)


def _common_execution(parser: argparse.ArgumentParser) -> None:
    _common_registry(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_RUNPOD_ROOT / "outputs"
    )
    parser.add_argument(
        "--scratch-root", type=Path, default=Path("/tmp/quadra-totalsegmentator")
    )
    parser.add_argument("--executable", default="TotalSegmentator")
    parser.add_argument("--device", default="gpu")
    parser.add_argument(
        "--no-resume", action="store_true", help="Do not skip compatible completed scans"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build a validated cohort manifest")
    _common_registry(prepare)
    prepare.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    prepare.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    prepare.add_argument("--subject-start", type=int, default=21)
    prepare.add_argument("--subject-end", type=int, default=48)
    prepare.add_argument("--run-id", default=DEFAULT_RUN_ID)
    prepare.add_argument("--output", type=Path, required=True)

    check = subparsers.add_parser("preflight", help="Validate data, disk, runtime, and GPU")
    _common_registry(check)
    check.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    check.add_argument("--output-root", type=Path, default=DEFAULT_RUNPOD_ROOT / "outputs")
    check.add_argument("--executable", default="TotalSegmentator")
    check.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    check.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Run only local data/storage checks; skip TotalSegmentator and CUDA",
    )

    single = subparsers.add_parser("run-scan", help="Run one manifest scan")
    _common_execution(single)
    single.add_argument("--subject", required=True)
    single.add_argument("--session", choices=("test", "retest"), required=True)

    cohort = subparsers.add_parser("run-cohort", help="Run every scan in a manifest")
    _common_execution(cohort)
    cohort.add_argument("--dry-run", action="store_true")
    cohort.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)

    validate = subparsers.add_parser("validate", help="Run technical mask QC")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--output-root", type=Path, required=True)
    validate.add_argument("--subject")
    validate.add_argument("--session", choices=("test", "retest"))
    validate.add_argument("--json-output", type=Path)

    status = subparsers.add_parser("status", help="Summarize cohort completion")
    status.add_argument("--manifest", type=Path, required=True)
    status.add_argument("--output-root", type=Path, required=True)
    status.add_argument("--json-output", type=Path)
    status.add_argument("--csv-output", type=Path)

    reseg_prepare = subparsers.add_parser(
        "reseg-prepare", help="Freeze the approved flagged-mask execution manifest"
    )
    reseg_prepare.add_argument("--review-root", type=Path, required=True)
    reseg_prepare.add_argument(
        "--plan", type=Path, default=DEFAULT_RESEGMENTATION_PLAN
    )
    reseg_prepare.add_argument("--output", type=Path, required=True)

    reseg_stage = subparsers.add_parser(
        "reseg-stage-inputs", help="Copy only planned GPU CTs into a portable staging tree"
    )
    reseg_stage.add_argument("--manifest", type=Path, required=True)
    reseg_stage.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    reseg_stage.add_argument("--destination", type=Path, required=True)
    reseg_stage.add_argument(
        "--phase",
        action="append",
        choices=SEGMENTATION_PHASE_ORDER,
        help="Stage one or more phases; defaults to every segmentation phase",
    )

    reseg_check = subparsers.add_parser(
        "reseg-preflight", help="Validate the frozen plan, inputs, runtime, tasks, and GPU"
    )
    reseg_check.add_argument("--manifest", type=Path, required=True)
    reseg_check.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    reseg_check.add_argument("--output-root", type=Path, default=DEFAULT_RUNPOD_ROOT)
    reseg_check.add_argument("--executable", default="TotalSegmentator")
    reseg_check.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    reseg_check.add_argument("--skip-runtime", action="store_true")

    def add_reseg_execution(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
        parser.add_argument("--output-root", type=Path, default=DEFAULT_RUNPOD_ROOT)
        parser.add_argument(
            "--scratch-root", type=Path, default=Path("/tmp/quadra-resegmentation")
        )
        parser.add_argument("--executable", default="TotalSegmentator")
        parser.add_argument("--device", default="gpu")
        parser.add_argument("--no-resume", action="store_true")

    reseg_single = subparsers.add_parser(
        "reseg-run-case", help="Run exactly one frozen segmentation or correction case"
    )
    add_reseg_execution(reseg_single)
    reseg_single.add_argument("--case-id", required=True)

    reseg_phase = subparsers.add_parser(
        "reseg-run-phase", help="Run or dry-run one approved phase"
    )
    add_reseg_execution(reseg_phase)
    reseg_phase.add_argument("--phase", choices=ALL_PHASE_ORDER, required=True)
    reseg_phase.add_argument("--dry-run", action="store_true")
    reseg_phase.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)

    reseg_validate = subparsers.add_parser(
        "reseg-validate", help="Run technical QC on re-segmentation derivatives"
    )
    reseg_validate.add_argument("--manifest", type=Path, required=True)
    reseg_validate.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    reseg_validate.add_argument("--output-root", type=Path, default=DEFAULT_RUNPOD_ROOT)
    reseg_validate.add_argument("--phase", choices=ALL_PHASE_ORDER)
    reseg_validate.add_argument("--json-output", type=Path)

    reseg_status_parser = subparsers.add_parser(
        "reseg-status", help="Summarize pending, complete, and incompatible cases"
    )
    reseg_status_parser.add_argument("--manifest", type=Path, required=True)
    reseg_status_parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    reseg_status_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_RUNPOD_ROOT
    )
    reseg_status_parser.add_argument("--json-output", type=Path)
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "prepare":
            manifest = prepare_manifest(
                args.dataset_root,
                args.demographics,
                args.registry,
                args.subject_start,
                args.subject_end,
                args.run_id,
                PROJECT_ROOT,
            )
            atomic_write_json(args.output.expanduser().resolve(), manifest)
            _print_json({"manifest": str(args.output), **manifest["summary"]})
            return 0
        if args.command == "preflight":
            _print_json(
                preflight(
                    args.registry,
                    args.dataset_root,
                    args.output_root,
                    args.executable,
                    args.min_free_gib,
                    args.skip_runtime,
                )
            )
            return 0
        if args.command == "run-scan":
            manifest = load_manifest(args.manifest)
            scan = find_scan(manifest, args.subject, args.session)
            result = run_scan(
                manifest,
                scan,
                args.registry,
                args.output_root,
                args.scratch_root,
                args.executable,
                args.device,
                not args.no_resume,
            )
            _print_json(result)
            return 0
        if args.command == "run-cohort":
            exit_code, result = run_cohort(
                args.manifest,
                args.registry,
                args.output_root,
                args.scratch_root,
                args.executable,
                args.device,
                not args.no_resume,
                args.dry_run,
                args.min_free_gib,
            )
            _print_json(result)
            return exit_code
        if args.command == "validate":
            result = validate_cohort(
                args.manifest,
                args.output_root,
                args.subject,
                args.session,
            )
            if args.json_output:
                atomic_write_json(args.json_output, result)
            _print_json(result["summary"])
            return 0 if result["status"] == "valid" else 1
        if args.command == "status":
            result = cohort_status(args.manifest, args.output_root)
            if args.json_output:
                atomic_write_json(args.json_output, result)
            if args.csv_output:
                write_status_csv(args.csv_output, result)
            _print_json(result)
            return 0
        if args.command == "reseg-prepare":
            result = prepare_resegmentation_manifest(args.review_root, args.plan)
            atomic_write_json(args.output.expanduser().resolve(), result)
            _print_json({"manifest": str(args.output), **result["summary"]})
            return 0
        if args.command == "reseg-stage-inputs":
            result = stage_selected_inputs(
                args.manifest,
                args.dataset_root,
                args.destination,
                args.phase or SEGMENTATION_PHASE_ORDER,
            )
            _print_json({"destination": str(args.destination), **result["summary"]})
            return 0
        if args.command == "reseg-preflight":
            _print_json(
                resegmentation_preflight(
                    args.manifest,
                    args.dataset_root,
                    args.output_root,
                    args.executable,
                    args.min_free_gib,
                    args.skip_runtime,
                )
            )
            return 0
        if args.command == "reseg-run-case":
            manifest = load_resegmentation_manifest(args.manifest)
            selected = [
                case for case in manifest["cases"] if case["case_id"] == args.case_id
            ]
            if len(selected) != 1:
                raise WorkflowError(f"Unknown case ID: {args.case_id}")
            result = run_resegmentation_case(
                manifest,
                selected[0],
                args.dataset_root,
                args.output_root,
                args.scratch_root,
                args.executable,
                args.device,
                not args.no_resume,
            )
            _print_json(result)
            return 0
        if args.command == "reseg-run-phase":
            exit_code, result = run_resegmentation_phase(
                args.manifest,
                args.phase,
                args.dataset_root,
                args.output_root,
                args.scratch_root,
                args.executable,
                args.device,
                not args.no_resume,
                args.dry_run,
                args.min_free_gib,
            )
            _print_json(result)
            return exit_code
        if args.command == "reseg-validate":
            result = validate_resegmentation_outputs(
                args.manifest, args.dataset_root, args.output_root, args.phase
            )
            if args.json_output:
                atomic_write_json(args.json_output, result)
            _print_json(result["summary"])
            return 0 if result["status"] == "valid" else 1
        if args.command == "reseg-status":
            result = resegmentation_status(
                args.manifest, args.dataset_root, args.output_root
            )
            if args.json_output:
                atomic_write_json(args.json_output, result)
            _print_json(result["summary"])
            return 0
    except (WorkflowError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled command: {args.command}")
