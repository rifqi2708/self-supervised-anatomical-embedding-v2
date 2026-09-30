"""Auditable execution for the reviewed Quadra flagged-mask plan.

This module deliberately separates planning, technical execution, and manual
anatomical review.  It never overwrites the source CTs, masks, or review files.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
import yaml

from .core import WorkflowError, atomic_write_json, sha256_file, utc_now
from .qc import validate_mask
from .workflow import DEFAULT_MIN_FREE_GIB, free_disk_gib, shell_join


RESEGMENTATION_SCHEMA_VERSION = 1
DEFAULT_RESEGMENTATION_PLAN = Path(__file__).with_name("resegmentation_plan.yaml")
SEGMENTATION_PHASE_ORDER = (
    "sacrum_pilot",
    "vertebrae_pilot",
    "ribs",
    "other_organs",
)
ALL_PHASE_ORDER = (*SEGMENTATION_PHASE_ORDER, "corrections")


class ResegmentationIntegrityError(WorkflowError):
    """Raised when a frozen plan input has changed or is incompatible."""


def load_resegmentation_plan(path: Path = DEFAULT_RESEGMENTATION_PLAN) -> dict[str, Any]:
    plan_path = path.expanduser().resolve()
    with plan_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise WorkflowError("Unsupported re-segmentation plan schema")
    phases = value.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(ALL_PHASE_ORDER):
        raise WorkflowError("Re-segmentation plan must define the five expected phases")
    for phase_name, phase in phases.items():
        if not isinstance(phase, dict) or not isinstance(phase.get("cases"), list):
            raise WorkflowError(f"Invalid phase configuration: {phase_name}")
        if phase.get("gated"):
            raise WorkflowError(f"Executable phase may not be pre-authorized as gated: {phase_name}")
    return value


def _read_decisions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"item_id", "subject_id", "session", "organ", "decision", "note"}
    if not rows or not required.issubset(rows[0]):
        raise WorkflowError("Review decisions file is empty or missing required columns")
    return rows


def _relative_source_path(path: str, dataset_root: Path) -> str:
    source = Path(path).expanduser().resolve()
    try:
        return str(source.relative_to(dataset_root))
    except ValueError as exc:
        raise WorkflowError(f"Review source escapes its dataset root: {source}") from exc


def _metadata_subset(value: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    return {
        "relative_path": _relative_source_path(str(value["path"]), dataset_root),
        "sha256": str(value["sha256"]),
        "size_bytes": int(value.get("size", value.get("size_bytes", -1))),
    }


def _expand_phase_cases(
    plan: dict[str, Any],
    review_manifest: dict[str, Any],
    decisions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    dataset_root = Path(review_manifest["dataset_root"]).expanduser().resolve()
    scans = {
        (str(scan["subject_id"]), str(scan["session"])): scan
        for scan in review_manifest["scans"]
    }
    decision_rows = {
        (str(row["subject_id"]), str(row["session"]), str(row["organ"])): row
        for row in decisions
    }
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    vertebrae_outputs = [str(value) for value in plan["vertebrae_outputs"]]

    for phase_name in ALL_PHASE_ORDER:
        phase = plan["phases"][phase_name]
        kind = str(phase["kind"])
        for group in phase["cases"]:
            subject = str(group["subject_id"])
            masks = [str(value) for value in group["masks"]]
            for session in group["sessions"]:
                session = str(session)
                scan = scans.get((subject, session))
                if scan is None:
                    raise WorkflowError(f"Review manifest is missing {subject} {session}")
                key = (phase_name, subject, session)
                if key in seen:
                    raise WorkflowError(f"Duplicate planned case: {key}")
                seen.add(key)
                mask_map = {str(mask["organ"]): mask for mask in scan["masks"]}
                missing = sorted(set(masks) - set(mask_map))
                if missing:
                    raise WorkflowError(
                        f"Review manifest is missing masks for {subject} {session}: {missing}"
                    )
                review_rows = []
                source_masks = {}
                for mask_name in masks:
                    row = decision_rows.get((subject, session, mask_name))
                    if row is None:
                        raise WorkflowError(
                            f"Decision is missing for {subject} {session} {mask_name}"
                        )
                    review_rows.append(
                        {
                            "organ": mask_name,
                            "decision": row["decision"],
                            "note": row["note"],
                        }
                    )
                    source_masks[mask_name] = _metadata_subset(mask_map[mask_name], dataset_root)

                if kind == "isolated_voxel_correction":
                    invalid = [
                        row for row in review_rows if row["decision"] != "requires_correction"
                    ]
                    if invalid:
                        raise WorkflowError(
                            f"Correction case is not marked requires_correction: {key}"
                        )
                    generated_masks = masks
                elif phase_name != "vertebrae_pilot":
                    invalid = [
                        row
                        for row in review_rows
                        if row["decision"] != "requires_resegmentation"
                    ]
                    if invalid:
                        raise WorkflowError(
                            f"Segmentation case is not marked requires_resegmentation: {key}"
                        )
                    generated_masks = masks
                else:
                    generated_masks = vertebrae_outputs

                cases.append(
                    {
                        "case_id": f"{phase_name}--{subject}--{session}",
                        "phase": phase_name,
                        "kind": kind,
                        "subject_id": subject,
                        "session": session,
                        "sex": str(scan["sex"]),
                        "task": phase.get("task"),
                        "robust_crop": bool(phase.get("robust_crop", False)),
                        "requested_masks": masks,
                        "generated_masks": generated_masks,
                        "review_items": review_rows,
                        "input": _metadata_subset(scan["ct"], dataset_root),
                        "source_masks": source_masks,
                        "correction": (
                            {
                                "connectivity": int(phase["connectivity"]),
                                "expected_removed_voxels": int(
                                    phase["expected_removed_voxels"]
                                ),
                            }
                            if kind == "isolated_voxel_correction"
                            else None
                        ),
                    }
                )
    return cases


def prepare_resegmentation_manifest(
    review_root: Path,
    plan_path: Path = DEFAULT_RESEGMENTATION_PLAN,
) -> dict[str, Any]:
    review_root = review_root.expanduser().resolve()
    decisions_path = review_root / "decisions.csv"
    review_manifest_path = review_root / "review_manifest.json"
    if not decisions_path.is_file() or not review_manifest_path.is_file():
        raise WorkflowError("Review root must contain decisions.csv and review_manifest.json")

    plan_path = plan_path.expanduser().resolve()
    plan = load_resegmentation_plan(plan_path)
    expected = plan["source_review"]
    actual_decision_hash = sha256_file(decisions_path)
    if actual_decision_hash != expected["expected_decisions_sha256"]:
        raise ResegmentationIntegrityError(
            "Review decisions checksum changed from the approved plan"
        )
    decisions = _read_decisions(decisions_path)
    decision_counts = Counter(row["decision"] for row in decisions)
    if len(decisions) != int(expected["expected_items"]):
        raise ResegmentationIntegrityError("Review item count changed from the approved plan")
    if sum(value != "acceptable" for value in (row["decision"] for row in decisions)) != int(
        expected["expected_flagged"]
    ):
        raise ResegmentationIntegrityError("Flagged-mask count changed from the approved plan")
    if dict(sorted(decision_counts.items())) != dict(
        sorted((str(key), int(value)) for key, value in expected["expected_decisions"].items())
    ):
        raise ResegmentationIntegrityError("Review decision totals changed from the approved plan")

    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review_manifest.get("schema_version") != 1:
        raise WorkflowError("Unsupported review manifest schema")
    cases = _expand_phase_cases(plan, review_manifest, decisions)
    phase_counts = Counter(case["phase"] for case in cases)
    return {
        "schema_version": RESEGMENTATION_SCHEMA_VERSION,
        "plan_id": str(plan["plan_id"]),
        "created_at": utc_now(),
        "totalsegmentator_version": str(plan["totalsegmentator_version"]),
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "source_review": {
            "review_root": str(review_root),
            "decisions_sha256": actual_decision_hash,
            "review_manifest_sha256": sha256_file(review_manifest_path),
            "items": len(decisions),
            "decision_counts": dict(sorted(decision_counts.items())),
        },
        "dataset": {
            "source_root_at_preparation": str(
                Path(review_manifest["dataset_root"]).expanduser().resolve()
            ),
            "portable": True,
        },
        "phase_order": list(ALL_PHASE_ORDER),
        "cases": cases,
        "expansions": plan.get("expansions", {}),
        "summary": {
            "cases": len(cases),
            "segmentation_cases": sum(case["kind"] == "segmentation" for case in cases),
            "correction_cases": sum(
                case["kind"] == "isolated_voxel_correction" for case in cases
            ),
            "phase_cases": dict(sorted(phase_counts.items())),
            "planned_review_items": sum(len(case["requested_masks"]) for case in cases),
        },
        "batch_expansion_authorized": False,
        "anatomical_accuracy_assessed": False,
    }


def load_resegmentation_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_resegmentation_manifest(manifest)
    return manifest


def validate_resegmentation_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != RESEGMENTATION_SCHEMA_VERSION:
        raise ResegmentationIntegrityError("Unsupported re-segmentation manifest schema")
    if manifest.get("batch_expansion_authorized") is not False:
        raise ResegmentationIntegrityError("Batch expansion must remain explicitly unauthorized")
    if manifest.get("phase_order") != list(ALL_PHASE_ORDER):
        raise ResegmentationIntegrityError("Unexpected phase order")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ResegmentationIntegrityError("Manifest contains no cases")
    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ResegmentationIntegrityError("Manifest contains duplicate case IDs")
    for case in cases:
        if case.get("phase") not in ALL_PHASE_ORDER:
            raise ResegmentationIntegrityError(f"Unknown phase: {case.get('phase')}")
        if case.get("kind") not in {"segmentation", "isolated_voxel_correction"}:
            raise ResegmentationIntegrityError(f"Unknown case kind: {case.get('kind')}")


def select_cases(
    manifest: dict[str, Any],
    phase: str | None = None,
    subject: str | None = None,
    session: str | None = None,
) -> list[dict[str, Any]]:
    selected = list(manifest["cases"])
    if phase:
        if phase not in ALL_PHASE_ORDER:
            raise WorkflowError(f"Unknown phase: {phase}")
        selected = [case for case in selected if case["phase"] == phase]
    if subject:
        selected = [case for case in selected if case["subject_id"] == subject]
    if session:
        selected = [case for case in selected if case["session"] == session]
    if not selected:
        raise WorkflowError("No re-segmentation cases match the selection")
    return selected


def _resolve_portable_file(dataset_root: Path, metadata: dict[str, Any]) -> Path:
    root = dataset_root.expanduser().resolve()
    relative = Path(metadata["relative_path"])
    candidates = [root / relative]
    parts = relative.parts
    if "Image_QUADRA_HC_WB" in parts:
        index = parts.index("Image_QUADRA_HC_WB")
        candidates.append(root.joinpath(*parts[index + 1 :]))
    for marker in (
        "Masks_QUADRA_HC_WB",
        "Masks_QUADRA_HC_WB_001_020_totalsegmentator_2.16.0_organs_v1",
    ):
        if marker in parts:
            index = parts.index(marker)
            candidates.append(root.joinpath(*parts[index:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ResegmentationIntegrityError(
        "Required source file is missing; checked: " + ", ".join(str(path) for path in candidates)
    )


def verify_source_file(dataset_root: Path, metadata: dict[str, Any]) -> Path:
    path = _resolve_portable_file(dataset_root, metadata)
    expected_size = int(metadata.get("size_bytes", -1))
    if expected_size >= 0 and path.stat().st_size != expected_size:
        raise ResegmentationIntegrityError(f"Source size changed: {path}")
    if sha256_file(path) != metadata["sha256"]:
        raise ResegmentationIntegrityError(f"Source checksum changed: {path}")
    return path


def case_output_directory(output_root: Path, manifest: dict[str, Any], case: dict[str, Any]) -> Path:
    return (
        output_root.expanduser().resolve()
        / manifest["plan_id"]
        / case["phase"]
        / case["subject_id"]
        / case["session"]
    )


def build_resegmentation_command(
    case: dict[str, Any],
    input_path: Path,
    work_directory: Path,
    executable: str = "TotalSegmentator",
    device: str = "gpu",
) -> list[str]:
    if case["kind"] != "segmentation":
        raise WorkflowError("Correction cases do not have a TotalSegmentator command")
    task = str(case["task"])
    command = [
        executable,
        "-i",
        str(input_path),
        "-o",
        str(work_directory / "task_output"),
        "-ta",
        task,
        "--device",
        device,
    ]
    if task == "total":
        command.extend(["--roi_subset", *case["requested_masks"]])
        if case["robust_crop"]:
            command.append("--robust_crop")
    elif task != "vertebrae_pp_refined":
        raise WorkflowError(f"Unsupported planned task: {task}")
    command.extend(["--report", str(work_directory / "reports" / f"{task}.json")])
    return command


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shell_join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
        process.stdout.close()
    if return_code:
        raise WorkflowError(
            f"Command failed with exit code {return_code}: {shell_join(command)}"
        )


def _publish_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise WorkflowError(f"Destination already exists: {destination}")
    try:
        os.replace(source, destination)
    except OSError:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.publish-", dir=str(destination.parent))
        )
        try:
            shutil.copytree(source, staging, dirs_exist_ok=True)
            os.replace(staging, destination)
            shutil.rmtree(source)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _preserve_failure(source: Path, output_root: Path, manifest: dict[str, Any], case: dict[str, Any]) -> Path:
    failure_root = output_root / manifest["plan_id"] / "_failed"
    failure_root.mkdir(parents=True, exist_ok=True)
    destination = failure_root / f"{case['case_id']}--{source.name.rsplit('-', 1)[-1]}"
    try:
        os.replace(source, destination)
    except OSError:
        shutil.move(str(source), str(destination))
    return destination


def _case_qc(directory: Path, input_path: Path, expected_masks: Iterable[str]) -> dict[str, Any]:
    masks = {}
    for name in expected_masks:
        masks[name] = validate_mask(directory / "masks" / f"{name}.nii.gz", input_path)
    return {
        "status": "valid",
        "checked_at": utc_now(),
        "mask_count": len(masks),
        "masks": masks,
        "anatomical_accuracy_assessed": False,
    }


def completed_case_is_compatible(
    directory: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    input_path: Path,
) -> bool:
    run_manifest_path = directory / "run_manifest.json"
    if not run_manifest_path.is_file():
        return False
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        compatible = (
            run_manifest.get("completed") is True
            and run_manifest.get("plan_id") == manifest["plan_id"]
            and run_manifest.get("plan_sha256") == manifest["plan"]["sha256"]
            and run_manifest.get("case_id") == case["case_id"]
            and run_manifest.get("input_sha256") == case["input"]["sha256"]
            and run_manifest.get("generated_masks") == case["generated_masks"]
            and run_manifest.get("totalsegmentator_version")
            == manifest["totalsegmentator_version"]
        )
        if not compatible:
            return False
        _case_qc(directory, input_path, case["generated_masks"])
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError, WorkflowError):
        return False


def run_segmentation_case(
    manifest: dict[str, Any],
    case: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    scratch_root: Path,
    executable: str = "TotalSegmentator",
    device: str = "gpu",
    resume: bool = True,
) -> dict[str, Any]:
    if case["kind"] != "segmentation":
        raise WorkflowError("Selected case is not a segmentation case")
    input_path = verify_source_file(dataset_root, case["input"])
    destination = case_output_directory(output_root, manifest, case)
    if resume and completed_case_is_compatible(destination, manifest, case, input_path):
        return {"status": "skipped", "output_directory": str(destination)}
    if destination.exists():
        raise WorkflowError(f"Output exists but is incomplete or incompatible: {destination}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{case['case_id']}-", dir=str(scratch_root)))
    command = build_resegmentation_command(case, input_path, work, executable, device)
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "plan_id": manifest["plan_id"],
        "plan_sha256": manifest["plan"]["sha256"],
        "case_id": case["case_id"],
        "phase": case["phase"],
        "subject_id": case["subject_id"],
        "session": case["session"],
        "input_path": str(input_path),
        "input_sha256": case["input"]["sha256"],
        "totalsegmentator_version": manifest["totalsegmentator_version"],
        "task": case["task"],
        "requested_masks": case["requested_masks"],
        "generated_masks": case["generated_masks"],
        "robust_crop": case["robust_crop"],
        "command": command,
        "started_at": utc_now(),
        "completed": False,
        "anatomical_accuracy_assessed": False,
    }
    atomic_write_json(work / "run_manifest.json", run_manifest)
    try:
        (work / "task_output").mkdir(parents=True, exist_ok=True)
        (work / "reports").mkdir(parents=True, exist_ok=True)
        _run_logged(command, work / "logs" / f"{case['task']}.log")
        mask_directory = work / "masks"
        mask_directory.mkdir()
        for name in case["generated_masks"]:
            source = work / "task_output" / f"{name}.nii.gz"
            if not source.is_file():
                raise WorkflowError(f"TotalSegmentator did not produce expected mask: {source}")
            shutil.move(str(source), str(mask_directory / source.name))
        shutil.rmtree(work / "task_output")
        qc = _case_qc(work, input_path, case["generated_masks"])
        run_manifest.update({"completed": True, "completed_at": utc_now(), "qc": qc})
        atomic_write_json(work / "run_manifest.json", run_manifest)
        _publish_directory(work, destination)
        return {"status": "completed", "output_directory": str(destination), "qc": qc}
    except BaseException as exc:
        run_manifest.update(
            {
                "failed_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(work / "run_manifest.json", run_manifest)
        failed = _preserve_failure(work, output_root, manifest, case)
        raise WorkflowError(f"Case failed; evidence preserved at {failed}: {exc}") from exc


def _isolated_voxel_coordinates(data: np.ndarray) -> np.ndarray:
    """Return foreground voxels with no 26-connected foreground neighbour."""
    foreground = data.astype(bool, copy=False)
    neighbours = np.zeros(foreground.shape, dtype=bool)
    shape = foreground.shape
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                source = tuple(
                    slice(max(0, -delta), min(size, size - delta))
                    for delta, size in zip((dx, dy, dz), shape)
                )
                target = tuple(
                    slice(max(0, delta), min(size, size + delta))
                    for delta, size in zip((dx, dy, dz), shape)
                )
                neighbours[target] |= foreground[source]
    return np.argwhere(foreground & ~neighbours)


def run_correction_case(
    manifest: dict[str, Any],
    case: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    scratch_root: Path,
    resume: bool = True,
) -> dict[str, Any]:
    if case["kind"] != "isolated_voxel_correction":
        raise WorkflowError("Selected case is not an isolated-voxel correction")
    if len(case["requested_masks"]) != 1:
        raise WorkflowError("Correction cases must contain exactly one mask")
    input_path = verify_source_file(dataset_root, case["input"])
    name = case["requested_masks"][0]
    source_path = verify_source_file(dataset_root, case["source_masks"][name])
    destination = case_output_directory(output_root, manifest, case)
    if resume and completed_case_is_compatible(destination, manifest, case, input_path):
        return {"status": "skipped", "output_directory": str(destination)}
    if destination.exists():
        raise WorkflowError(f"Output exists but is incomplete or incompatible: {destination}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{case['case_id']}-", dir=str(scratch_root)))
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "plan_id": manifest["plan_id"],
        "plan_sha256": manifest["plan"]["sha256"],
        "case_id": case["case_id"],
        "phase": case["phase"],
        "subject_id": case["subject_id"],
        "session": case["session"],
        "input_path": str(input_path),
        "input_sha256": case["input"]["sha256"],
        "source_mask_path": str(source_path),
        "source_mask_sha256": case["source_masks"][name]["sha256"],
        "totalsegmentator_version": manifest["totalsegmentator_version"],
        "task": None,
        "requested_masks": [name],
        "generated_masks": [name],
        "started_at": utc_now(),
        "completed": False,
        "anatomical_accuracy_assessed": False,
    }
    atomic_write_json(work / "run_manifest.json", run_manifest)
    try:
        image = nib.load(str(source_path))
        data = np.asanyarray(image.dataobj)
        invalid = (data != 0) & (data != 1)
        if np.any(invalid):
            raise WorkflowError(f"Correction source mask is not binary: {source_path}")
        isolated = _isolated_voxel_coordinates(data)
        expected = int(case["correction"]["expected_removed_voxels"])
        if len(isolated) != expected:
            raise WorkflowError(
                f"Expected {expected} isolated voxel(s), found {len(isolated)} in {source_path}"
            )
        corrected = data.astype(np.uint8, copy=True)
        corrected[tuple(isolated.T)] = 0
        masks = work / "masks"
        masks.mkdir()
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        output_path = masks / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(corrected, image.affine, header), str(output_path))
        qc = _case_qc(work, input_path, [name])
        run_manifest.update(
            {
                "completed": True,
                "completed_at": utc_now(),
                "removed_voxel_count": len(isolated),
                "removed_voxel_coordinates": isolated.tolist(),
                "qc": qc,
            }
        )
        atomic_write_json(work / "run_manifest.json", run_manifest)
        _publish_directory(work, destination)
        return {"status": "completed", "output_directory": str(destination), "qc": qc}
    except BaseException as exc:
        run_manifest.update(
            {"failed_at": utc_now(), "error": f"{type(exc).__name__}: {exc}"}
        )
        atomic_write_json(work / "run_manifest.json", run_manifest)
        failed = _preserve_failure(work, output_root, manifest, case)
        raise WorkflowError(f"Correction failed; evidence preserved at {failed}: {exc}") from exc


def run_case(
    manifest: dict[str, Any],
    case: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    scratch_root: Path,
    executable: str = "TotalSegmentator",
    device: str = "gpu",
    resume: bool = True,
) -> dict[str, Any]:
    if case["kind"] == "segmentation":
        return run_segmentation_case(
            manifest,
            case,
            dataset_root,
            output_root,
            scratch_root,
            executable,
            device,
            resume,
        )
    return run_correction_case(
        manifest, case, dataset_root, output_root, scratch_root, resume
    )


def run_phase(
    manifest_path: Path,
    phase: str,
    dataset_root: Path,
    output_root: Path,
    scratch_root: Path,
    executable: str = "TotalSegmentator",
    device: str = "gpu",
    resume: bool = True,
    dry_run: bool = False,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> tuple[int, dict[str, Any]]:
    manifest = load_resegmentation_manifest(manifest_path)
    cases = select_cases(manifest, phase=phase)
    status = {
        "schema_version": 1,
        "plan_id": manifest["plan_id"],
        "phase": phase,
        "started_at": utc_now(),
        "dry_run": dry_run,
        "status": "running",
        "cases": [],
    }
    status_path = output_root / manifest["plan_id"] / f"{phase}-status.json"
    failures = 0
    for case in cases:
        input_path = verify_source_file(dataset_root, case["input"])
        if dry_run:
            row: dict[str, Any] = {
                "case_id": case["case_id"],
                "subject_id": case["subject_id"],
                "session": case["session"],
                "status": "planned",
            }
            if case["kind"] == "segmentation":
                row["command"] = build_resegmentation_command(
                    case,
                    input_path,
                    Path("<scratch>") / case["case_id"],
                    executable,
                    device,
                )
            else:
                row["operation"] = "remove exactly one 26-isolated voxel"
            status["cases"].append(row)
            continue
        if free_disk_gib(output_root) < min_free_gib:
            status.update(
                {
                    "status": "stopped_low_disk",
                    "stopped_at": utc_now(),
                    "minimum_free_gib": min_free_gib,
                    "free_gib": free_disk_gib(output_root),
                }
            )
            atomic_write_json(status_path, status)
            return 4, status
        try:
            result = run_case(
                manifest,
                case,
                dataset_root,
                output_root,
                scratch_root,
                executable,
                device,
                resume,
            )
            status["cases"].append({"case_id": case["case_id"], **result})
        except ResegmentationIntegrityError as exc:
            status["cases"].append(
                {"case_id": case["case_id"], "status": "failed_integrity", "error": str(exc)}
            )
            status.update({"status": "stopped_integrity_error", "stopped_at": utc_now()})
            atomic_write_json(status_path, status)
            return 5, status
        except Exception as exc:
            failures += 1
            status["cases"].append(
                {"case_id": case["case_id"], "status": "failed", "error": str(exc)}
            )
        atomic_write_json(status_path, status)
    status.update(
        {
            "status": (
                "dry_run_complete"
                if dry_run
                else "completed_with_failures"
                if failures
                else "completed"
            ),
            "completed_at": utc_now(),
            "summary": {
                "planned": len(cases),
                "completed": sum(row["status"] == "completed" for row in status["cases"]),
                "skipped": sum(row["status"] == "skipped" for row in status["cases"]),
                "failed": failures,
            },
        }
    )
    if not dry_run:
        atomic_write_json(status_path, status)
    return (1 if failures else 0), status


def _version_from_output(text: str) -> str | None:
    match = re.search(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", text)
    return match.group(1) if match else None


def resegmentation_preflight(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    executable: str = "TotalSegmentator",
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    skip_runtime: bool = False,
) -> dict[str, Any]:
    manifest = load_resegmentation_manifest(manifest_path)
    result: dict[str, Any] = {
        "checked_at": utc_now(),
        "plan_id": manifest["plan_id"],
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "output_root": str(output_root.expanduser().resolve()),
        "free_gib": free_disk_gib(output_root),
        "minimum_free_gib": min_free_gib,
        "checks": {},
        "runtime_skipped": skip_runtime,
    }
    if result["free_gib"] < min_free_gib:
        raise WorkflowError(
            f"Insufficient free disk: {result['free_gib']:.1f} GiB < {min_free_gib:.1f} GiB"
        )
    result["checks"]["storage"] = "ok"
    unique_inputs: dict[str, dict[str, Any]] = {}
    for case in manifest["cases"]:
        # The portable RunPod package intentionally contains only GPU cases.
        # Correction inputs and source masks are verified when those cases run
        # locally against the completed review dataset.
        if case["kind"] != "segmentation":
            continue
        unique_inputs[case["input"]["sha256"]] = case["input"]
    for metadata in unique_inputs.values():
        verify_source_file(dataset_root, metadata)
    result["checks"]["inputs"] = {
        "files": len(unique_inputs),
        "scope": "segmentation_cases",
        "status": "ok",
    }
    result["checks"]["corrections"] = {
        "cases": sum(case["kind"] == "isolated_voxel_correction" for case in manifest["cases"]),
        "status": "deferred_to_local_execution",
    }
    if skip_runtime:
        result["status"] = "local_checks_passed"
        return result

    process = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    version_text = f"{process.stdout}\n{process.stderr}"
    version = _version_from_output(version_text)
    expected_version = manifest["totalsegmentator_version"]
    if process.returncode or version != expected_version:
        raise WorkflowError(
            f"TotalSegmentator version mismatch: expected {expected_version}, found {version!r}"
        )
    result["checks"]["totalsegmentator_version"] = version

    expected_classes: dict[str, set[str]] = {}
    for case in manifest["cases"]:
        if case["kind"] == "segmentation":
            expected_classes.setdefault(case["task"], set()).update(case["generated_masks"])
    for task, classes in expected_classes.items():
        listed = subprocess.run(
            [executable, "--list-classes", task],
            capture_output=True,
            text=True,
            check=False,
        )
        if listed.returncode:
            raise WorkflowError(
                f"Could not list classes for {task}: {(listed.stderr or listed.stdout).strip()}"
            )
        available = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", listed.stdout + listed.stderr))
        missing = sorted(classes - available)
        if missing:
            raise WorkflowError(f"Task {task} is missing classes: {', '.join(missing)}")
        result["checks"][f"task:{task}"] = {"classes": len(classes), "status": "ok"}

    cuda = subprocess.run(
        [
            __import__("sys").executable,
            "-c",
            (
                "import json, torch; print(json.dumps({"
                "'available': torch.cuda.is_available(), 'count': torch.cuda.device_count(), "
                "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
                "'memory_bytes': torch.cuda.get_device_properties(0).total_memory "
                "if torch.cuda.is_available() else None}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if cuda.returncode:
        raise WorkflowError(f"PyTorch CUDA check failed: {cuda.stderr.strip()}")
    cuda_value = json.loads(cuda.stdout)
    if not cuda_value["available"] or int(cuda_value["count"]) < 1:
        raise WorkflowError("CUDA is not available to PyTorch")
    result["checks"]["cuda"] = cuda_value
    result["checks"]["licensed_task"] = {
        "task": "vertebrae_pp_refined",
        "required": True,
        "verified_by_first_smoke_run": True,
    }
    result["status"] = "passed"
    return result


def validate_resegmentation_outputs(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    phase: str | None = None,
) -> dict[str, Any]:
    manifest = load_resegmentation_manifest(manifest_path)
    cases = select_cases(manifest, phase=phase) if phase else list(manifest["cases"])
    rows = []
    for case in cases:
        input_path = verify_source_file(dataset_root, case["input"])
        directory = case_output_directory(output_root, manifest, case)
        try:
            qc = _case_qc(directory, input_path, case["generated_masks"])
            rows.append({"case_id": case["case_id"], "status": "valid", "qc": qc})
        except Exception as exc:
            rows.append(
                {"case_id": case["case_id"], "status": "invalid", "error": str(exc)}
            )
    valid = sum(row["status"] == "valid" for row in rows)
    return {
        "checked_at": utc_now(),
        "plan_id": manifest["plan_id"],
        "phase": phase,
        "status": "valid" if valid == len(rows) else "invalid",
        "summary": {"cases": len(rows), "valid": valid, "invalid": len(rows) - valid},
        "cases": rows,
        "anatomical_accuracy_assessed": False,
    }


def resegmentation_status(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest = load_resegmentation_manifest(manifest_path)
    rows = []
    for case in manifest["cases"]:
        input_path = verify_source_file(dataset_root, case["input"])
        directory = case_output_directory(output_root, manifest, case)
        if completed_case_is_compatible(directory, manifest, case, input_path):
            status = "completed"
        elif directory.exists():
            status = "incompatible"
        else:
            status = "pending"
        rows.append(
            {
                "case_id": case["case_id"],
                "phase": case["phase"],
                "subject_id": case["subject_id"],
                "session": case["session"],
                "status": status,
            }
        )
    counts = Counter(row["status"] for row in rows)
    return {
        "checked_at": utc_now(),
        "plan_id": manifest["plan_id"],
        "summary": dict(sorted(counts.items())),
        "cases": rows,
        "anatomical_accuracy_assessed": False,
    }


def stage_selected_inputs(
    manifest_path: Path,
    dataset_root: Path,
    destination: Path,
    phases: Iterable[str] = SEGMENTATION_PHASE_ORDER,
) -> dict[str, Any]:
    manifest = load_resegmentation_manifest(manifest_path)
    selected_phases = tuple(phases)
    invalid = sorted(set(selected_phases) - set(SEGMENTATION_PHASE_ORDER))
    if invalid:
        raise WorkflowError(f"Only segmentation phases can be staged: {invalid}")
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise WorkflowError(f"Input staging destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    seen: set[str] = set()
    for case in manifest["cases"]:
        if case["phase"] not in selected_phases or case["kind"] != "segmentation":
            continue
        source = verify_source_file(dataset_root, case["input"])
        if case["input"]["sha256"] in seen:
            continue
        seen.add(case["input"]["sha256"])
        relative_path = Path(case["input"]["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ResegmentationIntegrityError(
                f"Unsafe staged input path: {relative_path}"
            )
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != case["input"]["sha256"]:
            raise ResegmentationIntegrityError(f"Staged CT checksum mismatch: {target}")
        copied.append(
            {
                "subject_id": case["subject_id"],
                "session": case["session"],
                "relative_path": str(relative_path),
                "sha256": case["input"]["sha256"],
                "size_bytes": target.stat().st_size,
            }
        )
    result = {
        "schema_version": 1,
        "plan_id": manifest["plan_id"],
        "created_at": utc_now(),
        "phases": list(selected_phases),
        "files": copied,
        "summary": {
            "files": len(copied),
            "bytes": sum(row["size_bytes"] for row in copied),
        },
    }
    atomic_write_json(destination / "input_manifest.json", result)
    return result
