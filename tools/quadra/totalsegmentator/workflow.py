"""Execution, preflight, resume, cohort, validation, and status operations."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .core import (
    WorkflowError,
    atomic_write_json,
    find_scan,
    load_manifest,
    load_registry,
    registry_identity,
    sha256_file,
    subject_number,
    task_classes,
    utc_now,
)
from .qc import derive_spinal_cord_segments, validate_scan_outputs

DISK_GUARD_EXIT_CODE = 4
SYSTEMIC_FAILURE_EXIT_CODE = 5
DEFAULT_MIN_FREE_GIB = 20.0


class IntegrityError(WorkflowError):
    """Raised when immutable manifest inputs no longer match the plan."""


def shell_join(command: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(value)) for value in command)


def scan_output_directory(output_root: Path, manifest: dict[str, Any], scan: dict[str, Any]) -> Path:
    return output_root / manifest["run_id"] / scan["subject_id"] / scan["session"]


def build_commands(
    scan: dict[str, Any],
    registry: dict[str, Any],
    work_directory: Path,
    executable: str = "TotalSegmentator",
    device: str = "gpu",
) -> list[dict[str, Any]]:
    commands = []
    for task, classes in task_classes(registry, scan["sex"]).items():
        task_directory = work_directory / "tasks" / task
        report_path = work_directory / "reports" / f"{task}.json"
        command = [
            executable,
            "-i",
            scan["input_path"],
            "-o",
            str(task_directory),
            "-ta",
            task,
            "--device",
            device,
            "--roi_subset",
            *classes,
            "--report",
            str(report_path),
        ]
        commands.append(
            {
                "task": task,
                "classes": classes,
                "output_directory": str(task_directory),
                "report_path": str(report_path),
                "command": command,
            }
        )
    return commands


def validate_manifest_structure(
    manifest: dict[str, Any],
    registry: dict[str, Any],
    registry_path: Path,
) -> None:
    current_registry = registry_identity(registry_path)
    if manifest.get("registry", {}).get("sha256") != current_registry["sha256"]:
        raise IntegrityError("Cohort manifest organ-registry hash does not match current registry")
    if manifest.get("totalsegmentator_version") != registry["totalsegmentator_version"]:
        raise IntegrityError("Cohort manifest TotalSegmentator version does not match registry")
    seen: set[tuple[str, str]] = set()
    for scan in manifest["scans"]:
        key = (str(scan.get("subject_id")), str(scan.get("session")))
        if key in seen:
            raise IntegrityError(f"Duplicate scan in cohort manifest: {key}")
        seen.add(key)
        if not 21 <= subject_number(key[0]) <= 48:
            raise IntegrityError(f"Out-of-cohort subject in manifest: {key[0]}")
        if key[1] not in {"test", "retest"}:
            raise IntegrityError(f"Invalid session in manifest: {key[1]}")
        sex = str(scan.get("sex"))
        expected = [
            str(entry["filename"])
            for entry in registry["organs"]
            if sex in set(entry.get("sexes", {"M", "F"}))
        ] + [str(entry["filename"]) for entry in registry["derived_organs"]]
        if scan.get("expected_masks") != expected:
            raise IntegrityError(f"Expected-mask list mismatch for {key[0]} {key[1]}")
        if scan.get("task_classes") != task_classes(registry, sex):
            raise IntegrityError(f"Task/class list mismatch for {key[0]} {key[1]}")


def validate_scan_input(scan: dict[str, Any]) -> None:
    input_path = Path(scan["input_path"])
    if not input_path.is_file():
        raise IntegrityError(f"Manifest CT input is missing: {input_path}")
    actual_size = input_path.stat().st_size
    if actual_size != int(scan.get("input_size_bytes", -1)):
        raise IntegrityError(f"Manifest CT size changed: {input_path}")
    if sha256_file(input_path) != scan.get("input_sha256"):
        raise IntegrityError(f"Manifest CT hash changed: {input_path}")


def _run_logged(command: Sequence[str], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shell_join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
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


def _copy_selected_outputs(
    work_directory: Path,
    registry: dict[str, Any],
    sex: str,
) -> None:
    masks = work_directory / "masks"
    intermediate = work_directory / "intermediate"
    masks.mkdir(parents=True, exist_ok=True)
    intermediate.mkdir(parents=True, exist_ok=True)

    for entry in registry["organs"]:
        if sex not in set(entry.get("sexes", {"M", "F"})):
            continue
        source = (
            work_directory
            / "tasks"
            / entry["task"]
            / f"{entry['source_class']}.nii.gz"
        )
        if not source.is_file():
            raise WorkflowError(f"TotalSegmentator did not produce expected mask: {source}")
        shutil.copy2(source, masks / f"{entry['filename']}.nii.gz")
    for entry in registry["intermediates"]:
        source = (
            work_directory
            / "tasks"
            / entry["task"]
            / f"{entry['source_class']}.nii.gz"
        )
        if not source.is_file():
            raise WorkflowError(f"TotalSegmentator did not produce intermediate mask: {source}")
        shutil.copy2(source, intermediate / f"{entry['filename']}.nii.gz")


def _derive_outputs(work_directory: Path) -> dict[str, Any]:
    masks = work_directory / "masks"
    intermediate = work_directory / "intermediate"
    return derive_spinal_cord_segments(
        intermediate / "spinal_cord.nii.gz",
        masks / "vertebrae_C1.nii.gz",
        masks / "vertebrae_C7.nii.gz",
        intermediate / "vertebrae_T1.nii.gz",
        masks / "vertebrae_T12.nii.gz",
        masks / "vertebrae_L1.nii.gz",
        masks / "spinal_cord_cervical.nii.gz",
        masks / "spinal_cord_thoracic.nii.gz",
    )


def _publish_directory(source: Path, destination: Path) -> None:
    """Copy to destination storage, then atomically expose the completed directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def _preserve_failed_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
    except OSError:
        shutil.move(str(source), str(destination))


def completed_scan_is_compatible(
    output_directory: Path,
    manifest: dict[str, Any],
    scan: dict[str, Any],
    registry_path: Path,
) -> bool:
    run_manifest_path = output_directory / "run_manifest.json"
    if not run_manifest_path.is_file():
        return False
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if not run_manifest.get("completed"):
            return False
        compatible = (
            run_manifest.get("cohort_schema_version") == manifest["schema_version"]
            and run_manifest.get("run_id") == manifest["run_id"]
            and run_manifest.get("subject_id") == scan["subject_id"]
            and run_manifest.get("session") == scan["session"]
            and run_manifest.get("sex") == scan["sex"]
            and run_manifest.get("input_sha256") == scan["input_sha256"]
            and run_manifest.get("registry_sha256")
            == registry_identity(registry_path)["sha256"]
            and run_manifest.get("totalsegmentator_version")
            == manifest["totalsegmentator_version"]
            and run_manifest.get("expected_masks") == scan["expected_masks"]
        )
        if not compatible:
            return False
        validate_scan_outputs(
            output_directory,
            Path(scan["input_path"]),
            scan["expected_masks"],
            scan["sex"],
        )
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError, WorkflowError):
        return False


def run_scan(
    manifest: dict[str, Any],
    scan: dict[str, Any],
    registry_path: Path,
    output_root: Path,
    scratch_root: Path,
    executable: str = "TotalSegmentator",
    device: str = "gpu",
    resume: bool = True,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    validate_manifest_structure(manifest, registry, registry_path)
    validate_scan_input(scan)
    output_directory = scan_output_directory(output_root, manifest, scan)
    if resume and completed_scan_is_compatible(
        output_directory, manifest, scan, registry_path
    ):
        return {"status": "skipped", "output_directory": str(output_directory)}
    if output_directory.exists():
        raise WorkflowError(
            f"Output already exists but is incomplete or incompatible: {output_directory}"
        )
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f"{scan['subject_id']}-{scan['session']}-", dir=str(scratch_root)
        )
    )
    commands = build_commands(scan, registry, temporary_directory, executable, device)
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "cohort_schema_version": manifest["schema_version"],
        "run_id": manifest["run_id"],
        "subject_id": scan["subject_id"],
        "session": scan["session"],
        "sex": scan["sex"],
        "input_path": scan["input_path"],
        "input_sha256": scan["input_sha256"],
        "git_commit": manifest.get("git_commit"),
        "totalsegmentator_version": manifest["totalsegmentator_version"],
        "registry_sha256": registry_identity(registry_path)["sha256"],
        "expected_masks": scan["expected_masks"],
        "commands": commands,
        "started_at": utc_now(),
        "completed": False,
    }
    atomic_write_json(temporary_directory / "run_manifest.json", run_manifest)
    try:
        for command in commands:
            Path(command["output_directory"]).mkdir(parents=True, exist_ok=True)
            Path(command["report_path"]).parent.mkdir(parents=True, exist_ok=True)
            _run_logged(
                command["command"],
                temporary_directory / "logs" / f"{command['task']}.log",
            )
        _copy_selected_outputs(temporary_directory, registry, scan["sex"])
        derivation = _derive_outputs(temporary_directory)
        qc = validate_scan_outputs(
            temporary_directory,
            Path(scan["input_path"]),
            scan["expected_masks"],
            scan["sex"],
        )
        run_manifest.update(
            {
                "completed": True,
                "completed_at": utc_now(),
                "derivation": derivation,
                "qc": qc,
            }
        )
        shutil.rmtree(temporary_directory / "tasks")
        atomic_write_json(temporary_directory / "run_manifest.json", run_manifest)
        _publish_directory(temporary_directory, output_directory)
        return {"status": "completed", "output_directory": str(output_directory), "qc": qc}
    except BaseException as exc:
        failure = {
            **run_manifest,
            "completed": False,
            "failed_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_write_json(temporary_directory / "run_manifest.json", failure)
        failed_directory = (
            output_root
            / manifest["run_id"]
            / "_failed"
            / f"{scan['subject_id']}-{scan['session']}-{temporary_directory.name.rsplit('-', 1)[-1]}"
        )
        _preserve_failed_directory(temporary_directory, failed_directory)
        raise


def nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise WorkflowError(f"No existing parent for disk check: {path}")
        candidate = candidate.parent
    return candidate


def free_disk_gib(path: Path) -> float:
    return shutil.disk_usage(nearest_existing_parent(path)).free / 1024**3


def run_cohort(
    manifest_path: Path,
    registry_path: Path,
    output_root: Path,
    scratch_root: Path,
    executable: str = "TotalSegmentator",
    device: str = "gpu",
    resume: bool = True,
    dry_run: bool = False,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> tuple[int, dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    registry = load_registry(registry_path)
    validate_manifest_structure(manifest, registry, registry_path)
    batch: dict[str, Any] = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "started_at": utc_now(),
        "dry_run": dry_run,
        "status": "running",
        "scans": [],
    }
    batch_path = output_root / manifest["run_id"] / "cohort_status.json"
    for scan in manifest["scans"]:
        commands = build_commands(
            scan,
            registry,
            Path("<scratch>") / scan["subject_id"] / scan["session"],
            executable,
            device,
        )
        if dry_run:
            batch["scans"].append(
                {
                    "subject_id": scan["subject_id"],
                    "session": scan["session"],
                    "sex": scan["sex"],
                    "expected_mask_count": len(scan["expected_masks"]),
                    "commands": [command["command"] for command in commands],
                    "status": "planned",
                }
            )
            continue
        if free_disk_gib(output_root) < min_free_gib:
            batch.update(
                {
                    "status": "stopped_low_disk",
                    "stopped_at": utc_now(),
                    "minimum_free_gib": min_free_gib,
                    "free_gib": free_disk_gib(output_root),
                }
            )
            atomic_write_json(batch_path, batch)
            return DISK_GUARD_EXIT_CODE, batch
        try:
            result = run_scan(
                manifest,
                scan,
                registry_path,
                output_root,
                scratch_root,
                executable,
                device,
                resume,
            )
            batch["scans"].append(
                {
                    "subject_id": scan["subject_id"],
                    "session": scan["session"],
                    **result,
                }
            )
        except IntegrityError as exc:
            batch["scans"].append(
                {
                    "subject_id": scan["subject_id"],
                    "session": scan["session"],
                    "status": "failed_integrity",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            batch.update({"status": "stopped_integrity_error", "stopped_at": utc_now()})
            atomic_write_json(batch_path, batch)
            return SYSTEMIC_FAILURE_EXIT_CODE, batch
        except Exception as exc:
            batch["scans"].append(
                {
                    "subject_id": scan["subject_id"],
                    "session": scan["session"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        atomic_write_json(batch_path, batch)
    failures = sum(row["status"] == "failed" for row in batch["scans"])
    batch.update(
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
                "planned": len(batch["scans"]),
                "completed": sum(row["status"] == "completed" for row in batch["scans"]),
                "skipped": sum(row["status"] == "skipped" for row in batch["scans"]),
                "failed": failures,
            },
        }
    )
    if not dry_run:
        atomic_write_json(batch_path, batch)
    return (1 if failures else 0), batch


def _version_from_output(text: str) -> str | None:
    match = re.search(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", text)
    return match.group(1) if match else None


def _classes_from_output(text: str) -> set[str]:
    candidates = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", text))
    return candidates


def preflight(
    registry_path: Path,
    dataset_root: Path,
    output_root: Path,
    executable: str = "TotalSegmentator",
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    skip_runtime: bool = False,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    result: dict[str, Any] = {
        "checked_at": utc_now(),
        "registry": registry_identity(registry_path),
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "output_root": str(output_root.expanduser().resolve()),
        "free_gib": free_disk_gib(output_root),
        "minimum_free_gib": min_free_gib,
        "runtime_skipped": skip_runtime,
        "checks": {},
    }
    if result["free_gib"] < min_free_gib:
        raise WorkflowError(
            f"Insufficient free disk: {result['free_gib']:.1f} GiB < {min_free_gib:.1f} GiB"
        )
    result["checks"]["storage"] = "ok"
    if not dataset_root.expanduser().resolve().is_dir():
        raise WorkflowError(f"Dataset root does not exist: {dataset_root}")
    result["checks"]["dataset"] = "ok"
    if skip_runtime:
        result["status"] = "local_checks_passed"
        return result

    version_process = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    version_text = f"{version_process.stdout}\n{version_process.stderr}"
    installed_version = _version_from_output(version_text)
    expected_version = str(registry["totalsegmentator_version"])
    if version_process.returncode or installed_version != expected_version:
        raise WorkflowError(
            f"TotalSegmentator version mismatch: expected {expected_version}, "
            f"found {installed_version or version_text.strip()!r}"
        )
    result["checks"]["totalsegmentator_version"] = installed_version

    required_classes = task_classes(registry, "M")
    for task, expected_classes in required_classes.items():
        process = subprocess.run(
            [executable, "--list-classes", task],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode:
            raise WorkflowError(
                f"Could not list classes for {task}: {(process.stderr or process.stdout).strip()}"
            )
        available = _classes_from_output(f"{process.stdout}\n{process.stderr}")
        missing = sorted(set(expected_classes) - available)
        if missing:
            raise WorkflowError(
                f"TotalSegmentator task {task} is missing required classes: {', '.join(missing)}"
            )
        result["checks"][f"task:{task}"] = {"classes": len(expected_classes), "status": "ok"}

    cuda_process = subprocess.run(
        [
            __import__("sys").executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'available': torch.cuda.is_available(), "
                "'count': torch.cuda.device_count(), "
                "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if cuda_process.returncode:
        raise WorkflowError(f"PyTorch CUDA check failed: {cuda_process.stderr.strip()}")
    cuda = json.loads(cuda_process.stdout)
    if not cuda["available"] or int(cuda["count"]) < 1:
        raise WorkflowError("CUDA is not available to PyTorch")
    result["checks"]["cuda"] = cuda
    result["status"] = "passed"
    return result


def validate_cohort(
    manifest_path: Path,
    output_root: Path,
    subject: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    selected = manifest["scans"]
    if subject:
        selected = [scan for scan in selected if scan["subject_id"] == find_scan(
            manifest, subject, session or "test"
        )["subject_id"]]
    if session:
        selected = [scan for scan in selected if scan["session"] == session]
    rows = []
    for scan in selected:
        directory = scan_output_directory(output_root, manifest, scan)
        try:
            qc = validate_scan_outputs(
                directory,
                Path(scan["input_path"]),
                scan["expected_masks"],
                scan["sex"],
            )
            rows.append({**scan, "status": "valid", "qc": qc})
        except Exception as exc:
            rows.append({**scan, "status": "invalid", "error": f"{type(exc).__name__}: {exc}"})
    return {
        "checked_at": utc_now(),
        "run_id": manifest["run_id"],
        "status": "valid" if all(row["status"] == "valid" for row in rows) else "invalid",
        "summary": {
            "scans": len(rows),
            "valid": sum(row["status"] == "valid" for row in rows),
            "invalid": sum(row["status"] == "invalid" for row in rows),
        },
        "scans": rows,
        "anatomical_accuracy_assessed": False,
    }


def cohort_status(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    batch_path = output_root / manifest["run_id"] / "cohort_status.json"
    batch_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if batch_path.is_file():
        try:
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            for row in batch.get("scans", []):
                batch_rows[(row.get("subject_id"), row.get("session"))] = row
        except (OSError, TypeError, json.JSONDecodeError):
            batch_rows = {}
    rows = []
    for scan in manifest["scans"]:
        directory = scan_output_directory(output_root, manifest, scan)
        if not directory.exists():
            previous = batch_rows.get((scan["subject_id"], scan["session"]), {})
            if str(previous.get("status", "")).startswith("failed"):
                status = "failed"
                detail = str(previous.get("error", ""))
            else:
                status = "pending"
                detail = ""
        else:
            try:
                run_manifest = json.loads(
                    (directory / "run_manifest.json").read_text(encoding="utf-8")
                )
                compatible = (
                    run_manifest.get("completed") is True
                    and run_manifest.get("run_id") == manifest["run_id"]
                    and run_manifest.get("subject_id") == scan["subject_id"]
                    and run_manifest.get("session") == scan["session"]
                    and run_manifest.get("sex") == scan["sex"]
                    and run_manifest.get("input_sha256") == scan["input_sha256"]
                    and run_manifest.get("totalsegmentator_version")
                    == manifest["totalsegmentator_version"]
                    and run_manifest.get("expected_masks") == scan["expected_masks"]
                )
                if not compatible:
                    raise WorkflowError("Run manifest is incomplete or incompatible")
                validate_scan_outputs(
                    directory,
                    Path(scan["input_path"]),
                    scan["expected_masks"],
                    scan["sex"],
                )
                status = "completed"
                detail = ""
            except Exception as exc:
                status = "invalid"
                detail = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "subject_id": scan["subject_id"],
                "session": scan["session"],
                "sex": scan["sex"],
                "expected_mask_count": len(scan["expected_masks"]),
                "status": status,
                "detail": detail,
                "output_directory": str(directory),
            }
        )
    return {
        "generated_at": utc_now(),
        "run_id": manifest["run_id"],
        "summary": {
            key: sum(row["status"] == key for row in rows)
            for key in ("completed", "failed", "pending", "invalid")
        },
        "scans": rows,
    }


def write_status_csv(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            fields = [
                "subject_id",
                "session",
                "sex",
                "expected_mask_count",
                "status",
                "detail",
                "output_directory",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(status["scans"])
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
