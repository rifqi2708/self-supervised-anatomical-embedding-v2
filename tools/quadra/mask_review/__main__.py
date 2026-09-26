"""Command-line interface for the local Quadra mask-review workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from tools.quadra.totalsegmentator.core import DEFAULT_REGISTRY

from .core import (
    ReviewError,
    build_resegmentation_review_index,
    build_review_index,
    export_review_state,
)


def _build_index(args: argparse.Namespace) -> int:
    summary = build_review_index(args.dataset_root, args.review_root, args.registry)
    print(f"Review index ready: {args.review_root}")
    print(f"Review items: {summary['review_items']}")
    print(f"Pending: {summary['counts']['pending']}")
    return 0


def _export(args: argparse.Namespace) -> int:
    summary = export_review_state(args.review_root, create_checkpoint=True)
    print(f"Review state exported: {args.review_root}")
    print(
        f"Completed subjects: {summary['subjects_complete']}/{summary['subjects_total']}"
    )
    print(f"Flagged: {summary['counts']['requires_correction'] + summary['counts']['requires_resegmentation']}")
    return 0


def _build_resegmentation_index(args: argparse.Namespace) -> int:
    summary = build_resegmentation_review_index(
        args.execution_manifest,
        args.dataset_root,
        args.output_root,
        args.review_root,
        args.phase,
        args.registry,
    )
    print(f"Re-segmentation review index ready: {args.review_root}")
    print(f"Review items: {summary['review_items']}")
    print(f"Pending: {summary['counts']['pending']}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ReviewError("The review server must bind to a loopback address")
    review_root = args.review_root.expanduser().resolve()
    if not (review_root / "review_manifest.json").is_file():
        raise ReviewError(f"Review index is missing: {review_root}")
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise ReviewError(
            "Streamlit is not installed. Create the isolated review environment "
            "from tools/quadra/mask_review/requirements-review.txt."
        ) from exc
    app = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--review-root",
        str(review_root),
    ]
    return subprocess.call(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local, read-only anatomical review for Quadra organ masks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="Freeze and validate the review cohort")
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--review-root", type=Path, required=True)
    build.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    build.set_defaults(handler=_build_index)

    build_reseg = subparsers.add_parser(
        "build-reseg-index", help="Freeze completed re-segmentation outputs for review"
    )
    build_reseg.add_argument("--execution-manifest", type=Path, required=True)
    build_reseg.add_argument("--dataset-root", type=Path, required=True)
    build_reseg.add_argument("--output-root", type=Path, required=True)
    build_reseg.add_argument("--review-root", type=Path, required=True)
    build_reseg.add_argument("--phase", action="append")
    build_reseg.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    build_reseg.set_defaults(handler=_build_resegmentation_index)

    serve = subparsers.add_parser("serve", help="Launch the loopback-only review viewer")
    serve.add_argument("--review-root", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8501)
    serve.set_defaults(handler=_serve)

    export = subparsers.add_parser("export", help="Export current queues and a checksum checkpoint")
    export.add_argument("--review-root", type=Path, required=True)
    export.set_defaults(handler=_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ReviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
