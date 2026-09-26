"""Dataset indexing and durable decision state for local mask review.

The CT and mask trees are immutable inputs.  This module only writes beneath a
caller-selected review directory.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import nibabel as nib
import numpy as np

from tools.quadra.totalsegmentator.core import (
    DEFAULT_REGISTRY,
    expected_mask_names,
    load_registry,
    registry_identity,
    sha256_file,
)


SCHEMA_VERSION = 1
SUBJECTS = tuple(f"quadra_hc_{number:03d}" for number in range(1, 49))
SESSIONS = ("test", "retest")
DECISIONS = (
    "pending",
    "acceptable",
    "requires_correction",
    "requires_resegmentation",
)
FLAGGED_DECISIONS = {"requires_correction", "requires_resegmentation"}
EXPECTED_EARLY_MASKS = 1582
EXPECTED_LATE_MASKS = 2208
DECISION_FIELDS = (
    "item_id",
    "subject_id",
    "session",
    "sex",
    "organ",
    "display_name",
    "decision",
    "note",
    "axial_slice",
    "coronal_slice",
    "sagittal_slice",
    "window_preset",
    "overlay_mode",
    "opacity",
    "view_framing",
    "crop_margin",
    "updated_at",
)


class ReviewError(RuntimeError):
    """Raised when review inputs or durable state are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def stepped_slice(current: int, delta: int, maximum: int) -> int:
    """Move one or more slices while remaining inside the volume."""
    if maximum < 0:
        raise ReviewError("Slice maximum must be non-negative")
    return min(max(int(current) + int(delta), 0), int(maximum))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _identity(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = sha256_file(resolved)
    return result


def _organ_catalog(registry: dict[str, Any]) -> list[dict[str, str]]:
    result = [
        {
            "filename": str(entry["filename"]),
            "display_name": str(entry["display_name"]),
        }
        for entry in registry["organs"]
    ]
    result.extend(
        {
            "filename": str(entry["filename"]),
            "display_name": str(entry["display_name"]),
        }
        for entry in registry["derived_organs"]
    )
    return result


def item_id(subject_id: str, session: str, organ: str) -> str:
    return f"{subject_id}|{session}|{organ}"


def _load_scan_metadata(scan_root: Path) -> dict[str, Any]:
    candidates = (scan_root / "scan_manifest.json", scan_root / "run_manifest.json")
    for candidate in candidates:
        if candidate.is_file():
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ReviewError(f"Unreadable scan manifest: {candidate}") from exc
            if not value.get("completed", True):
                raise ReviewError(f"Scan manifest is not complete: {candidate}")
            return value
    raise ReviewError(f"No scan manifest found beneath {scan_root}")


def _geometry(path: Path) -> tuple[tuple[int, ...], np.ndarray, str]:
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise ReviewError(f"Unreadable NIfTI file: {path}") from exc
    shape = tuple(int(value) for value in image.shape)
    if len(shape) != 3:
        raise ReviewError(f"Expected a 3D NIfTI image, found shape {shape}: {path}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ReviewError(f"Invalid affine: {path}")
    return shape, affine, str(image.get_data_dtype())


def _scan_source_roots(dataset_root: Path, subject_number: int) -> tuple[Path, str]:
    if subject_number <= 20:
        return (
            dataset_root
            / "Masks_QUADRA_HC_WB_001_020_totalsegmentator_2.16.0_organs_v1",
            "subjects_001_020_finalized",
        )
    return dataset_root / "Masks_QUADRA_HC_WB", "subjects_021_048_current_corrected"


def discover_review_manifest(
    dataset_root: Path | str,
    registry_path: Path | str = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Validate and hash the complete review cohort without writing outputs."""

    dataset = Path(dataset_root).expanduser().resolve()
    if not dataset.is_dir():
        raise ReviewError(f"Dataset root does not exist: {dataset}")
    image_root = dataset / "Image_QUADRA_HC_WB"
    if not image_root.is_dir():
        raise ReviewError(f"CT root does not exist: {image_root}")

    registry_path = Path(registry_path).expanduser().resolve()
    registry = load_registry(registry_path)
    catalog = _organ_catalog(registry)
    display_names = {entry["filename"]: entry["display_name"] for entry in catalog}
    scans: list[dict[str, Any]] = []
    early_count = 0
    late_count = 0

    for number, subject_id in enumerate(SUBJECTS, start=1):
        mask_root, source_selection = _scan_source_roots(dataset, number)
        subject_mask_root = mask_root / subject_id
        subject_image_root = image_root / f"QUADRA_HC_{number:03d}"
        if not subject_mask_root.is_dir():
            raise ReviewError(f"Missing subject mask directory: {subject_mask_root}")
        if not subject_image_root.is_dir():
            raise ReviewError(f"Missing subject CT directory: {subject_image_root}")
        subject_sex: str | None = None

        for session in SESSIONS:
            scan_root = subject_mask_root / session
            metadata = _load_scan_metadata(scan_root)
            metadata_subject = str(metadata.get("subject_id", "")).lower()
            if metadata_subject != subject_id:
                raise ReviewError(
                    f"Manifest subject mismatch for {scan_root}: {metadata_subject!r}"
                )
            if str(metadata.get("session", "")) != session:
                raise ReviewError(f"Manifest session mismatch for {scan_root}")
            sex = str(metadata.get("sex", "")).strip().upper()
            if sex not in {"M", "F"}:
                raise ReviewError(f"Invalid or absent sex in {scan_root}")
            if subject_sex is not None and subject_sex != sex:
                raise ReviewError(f"Sex differs between sessions for {subject_id}")
            subject_sex = sex

            ct_path = subject_image_root / f"{session}_CT-AC.nii.gz"
            if not ct_path.is_file():
                raise ReviewError(f"Missing CT file: {ct_path}")
            expected = expected_mask_names(registry, sex)
            actual_paths = sorted((scan_root / "masks").glob("*.nii.gz"))
            actual = [path.name.removesuffix(".nii.gz") for path in actual_paths]
            if sorted(expected) != sorted(actual):
                missing = sorted(set(expected) - set(actual))
                unexpected = sorted(set(actual) - set(expected))
                raise ReviewError(
                    f"Mask inventory mismatch for {scan_root}; missing={missing}, "
                    f"unexpected={unexpected}"
                )

            ct_shape, ct_affine, ct_dtype = _geometry(ct_path)
            ct_identity = _identity(ct_path)
            ct_identity.update(
                {
                    "shape": list(ct_shape),
                    "affine": ct_affine.tolist(),
                    "dtype": ct_dtype,
                }
            )
            masks: list[dict[str, Any]] = []
            for name in expected:
                mask_path = scan_root / "masks" / f"{name}.nii.gz"
                mask_shape, mask_affine, mask_dtype = _geometry(mask_path)
                if mask_shape != ct_shape or not np.allclose(mask_affine, ct_affine, atol=1e-5):
                    raise ReviewError(f"Mask geometry does not match CT: {mask_path}")
                identity = _identity(mask_path)
                identity.update(
                    {
                        "organ": name,
                        "display_name": display_names[name],
                        "dtype": mask_dtype,
                    }
                )
                masks.append(identity)
            if number <= 20:
                early_count += len(masks)
            else:
                late_count += len(masks)
            scans.append(
                {
                    "scan_id": f"{subject_id}|{session}",
                    "subject_id": subject_id,
                    "subject_number": number,
                    "session": session,
                    "sex": sex,
                    "source_selection": source_selection,
                    "ct": ct_identity,
                    "masks": masks,
                }
            )

    if len(scans) != 96:
        raise ReviewError(f"Expected 96 scans, found {len(scans)}")
    if early_count != EXPECTED_EARLY_MASKS:
        raise ReviewError(
            f"Expected {EXPECTED_EARLY_MASKS} masks for subjects 001-020, found {early_count}"
        )
    if late_count != EXPECTED_LATE_MASKS:
        raise ReviewError(
            f"Expected {EXPECTED_LATE_MASKS} masks for subjects 021-048, found {late_count}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset_root": str(dataset),
        "registry": registry_identity(registry_path),
        "organ_catalog": catalog,
        "subject_order": list(SUBJECTS),
        "session_order": list(SESSIONS),
        "counts": {
            "subjects": 48,
            "scans": 96,
            "subjects_001_020_masks": early_count,
            "subjects_021_048_masks": late_count,
            "review_items": early_count + late_count,
        },
        "source_policy": {
            "subjects_001_020": "finalized TotalSegmentator 2.16.0 organs-v1 derivative",
            "subjects_021_048": "current local organs-v1 mask set",
            "subjects_030_047": "current corrected masks only; no original/corrected toggle",
        },
        "scans": scans,
    }


def flatten_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    catalog_order = {
        str(entry["filename"]): index
        for index, entry in enumerate(manifest["organ_catalog"])
    }
    items: list[dict[str, Any]] = []
    for scan in manifest["scans"]:
        masks = sorted(scan["masks"], key=lambda value: catalog_order[value["organ"]])
        for mask in masks:
            items.append(
                {
                    "item_id": item_id(scan["subject_id"], scan["session"], mask["organ"]),
                    "subject_id": scan["subject_id"],
                    "subject_number": scan["subject_number"],
                    "session": scan["session"],
                    "sex": scan["sex"],
                    "organ": mask["organ"],
                    "display_name": mask["display_name"],
                    "source_selection": scan["source_selection"],
                    "ct": scan["ct"],
                    "mask": mask,
                }
            )
    return items


def _pending_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "subject_id": item["subject_id"],
        "session": item["session"],
        "sex": item["sex"],
        "organ": item["organ"],
        "display_name": item["display_name"],
        "decision": "pending",
        "note": "",
        "axial_slice": "",
        "coronal_slice": "",
        "sagittal_slice": "",
        "window_preset": "soft_tissue",
        "overlay_mode": "fill_and_contour",
        "opacity": "0.35",
        "view_framing": "cropped",
        "crop_margin": "20",
        "updated_at": "",
    }


def load_manifest(review_root: Path | str) -> dict[str, Any]:
    path = Path(review_root).expanduser().resolve() / "review_manifest.json"
    if not path.is_file():
        raise ReviewError(f"Review manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Unreadable review manifest: {path}") from exc
    if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ReviewError("Unsupported review manifest schema")
    return value


def _read_csv_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["item_id"]: dict(row) for row in csv.DictReader(handle)}


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"Invalid events.jsonl line {line_number}") from exc
            events.append(value)
    return events


def load_decisions(review_root: Path | str) -> dict[str, dict[str, Any]]:
    root = Path(review_root).expanduser().resolve()
    manifest = load_manifest(root)
    items = flatten_items(manifest)
    expected_ids = {item["item_id"] for item in items}
    current = {item["item_id"]: _pending_row(item) for item in items}
    csv_state = _read_csv_state(root / "decisions.csv")
    if set(csv_state) - expected_ids:
        raise ReviewError("decisions.csv contains unknown review item IDs")
    current.update(csv_state)
    for event in _read_events(root / "events.jsonl"):
        event_id = str(event.get("item_id", ""))
        if event_id not in expected_ids:
            raise ReviewError(f"Event contains unknown item ID: {event_id}")
        for field in DECISION_FIELDS:
            if field in event:
                current[event_id][field] = event[field]
    if set(current) != expected_ids:
        raise ReviewError("Decision state does not match the frozen review manifest")
    return current


def first_pending_item_id(
    manifest: dict[str, Any], state: dict[str, dict[str, Any]]
) -> str | None:
    """Return the first pending item in the frozen review order."""
    for item in flatten_items(manifest):
        item_id = item["item_id"]
        if state[item_id]["decision"] == "pending":
            return item_id
    return None


def latest_checkpoint(review_root: Path | str) -> dict[str, Any] | None:
    """Load the newest checksum checkpoint, including legacy checkpoints."""
    checkpoint_root = Path(review_root).expanduser().resolve() / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    paths = sorted(checkpoint_root.glob("review-state-*.json"), reverse=True)
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewError(f"Cannot read checkpoint: {path}") from exc
        if not isinstance(value, dict) or not value.get("checkpoint_id"):
            raise ReviewError(f"Invalid checkpoint: {path}")
        value["checkpoint_path"] = str(path)
        return value
    return None


def verify_checkpoint(
    review_root: Path | str, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    """Check whether authoritative review files still match a checkpoint.

    Materialized CSV and summary outputs are reproducible from the event log, so
    freshness is based on the frozen manifest and append-only events file. This
    avoids marking a checkpoint stale merely because a derived summary timestamp
    was refreshed by the application.
    """
    root = Path(review_root).expanduser().resolve()
    recorded = {
        str(value.get("path")): value
        for value in checkpoint.get("files", [])
        if isinstance(value, dict)
    }
    authoritative = checkpoint.get(
        "authoritative_files", ["review_manifest.json", "events.jsonl"]
    )
    mismatches: list[str] = []
    for relative in authoritative:
        expected = recorded.get(str(relative))
        path = root / str(relative)
        if expected is None:
            mismatches.append(f"{relative}: not recorded")
        elif not path.is_file():
            mismatches.append(f"{relative}: missing")
        elif path.stat().st_size != int(expected["size"]):
            mismatches.append(f"{relative}: size changed")
        elif sha256_file(path) != expected["sha256"]:
            mismatches.append(f"{relative}: checksum changed")
    return {"current": not mismatches, "mismatches": mismatches}


def _summary(manifest: dict[str, Any], state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = {decision: 0 for decision in DECISIONS}
    by_subject: dict[str, dict[str, Any]] = {}
    by_organ: dict[str, dict[str, int]] = {}
    for row in state.values():
        decision = row["decision"]
        if decision not in counts:
            raise ReviewError(f"Invalid stored decision: {decision}")
        counts[decision] += 1
        subject = row["subject_id"]
        subject_state = by_subject.setdefault(
            subject,
            {"total": 0, "decided": 0, "pending": 0, "flagged": 0, "complete": False},
        )
        subject_state["total"] += 1
        if decision == "pending":
            subject_state["pending"] += 1
        else:
            subject_state["decided"] += 1
        if decision in FLAGGED_DECISIONS:
            subject_state["flagged"] += 1
        organ_counts = by_organ.setdefault(row["organ"], {value: 0 for value in DECISIONS})
        organ_counts[decision] += 1
    for value in by_subject.values():
        value["complete"] = value["pending"] == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "review_items": len(state),
        "counts": counts,
        "subjects_total": len(manifest["subject_order"]),
        "subjects_complete": sum(1 for value in by_subject.values() if value["complete"]),
        "subjects": by_subject,
        "organs": by_organ,
        "anatomical_accuracy_claimed": False,
        "interpretation": (
            "These are human visual-review decisions for project use, not manual "
            "delineations or independent anatomical ground truth."
        ),
        "manifest_created_at": manifest["created_at"],
    }


def export_review_state(
    review_root: Path | str,
    *,
    create_checkpoint: bool = False,
) -> dict[str, Any]:
    root = Path(review_root).expanduser().resolve()
    manifest = load_manifest(root)
    state = load_decisions(root)
    ordered = [state[item["item_id"]] for item in flatten_items(manifest)]
    _atomic_write_csv(root / "decisions.csv", ordered, DECISION_FIELDS)
    flagged = [row for row in ordered if row["decision"] in FLAGGED_DECISIONS]
    _atomic_write_csv(root / "flagged_cases.csv", flagged, DECISION_FIELDS)
    summary = _summary(manifest, state)
    _atomic_write_json(root / "review_summary.json", summary)
    if create_checkpoint:
        checkpoint_root = root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_id = f"review-state-{compact_utc_now()}"
        tracked = [
            root / "review_manifest.json",
            root / "events.jsonl",
            root / "decisions.csv",
            root / "flagged_cases.csv",
            root / "review_summary.json",
        ]
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "created_at": utc_now(),
            "authoritative_files": ["review_manifest.json", "events.jsonl"],
            "counts": summary["counts"],
            "subjects_complete": summary["subjects_complete"],
            "resume_item_id": first_pending_item_id(manifest, state),
            "files": [
                {
                    "path": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in tracked
            ],
        }
        _atomic_write_json(checkpoint_root / f"{checkpoint_id}.json", checkpoint)
        summary["checkpoint"] = checkpoint
    return summary


def build_review_index(
    dataset_root: Path | str,
    review_root: Path | str,
    registry_path: Path | str = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    root = Path(review_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ReviewError(f"Refusing to overwrite non-empty review directory: {root}")
    manifest = discover_review_manifest(dataset_root, registry_path)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(root / "review_manifest.json", manifest)
    _atomic_write_text(root / "events.jsonl", "")
    rows = [_pending_row(item) for item in flatten_items(manifest)]
    _atomic_write_csv(root / "decisions.csv", rows, DECISION_FIELDS)
    return export_review_state(root, create_checkpoint=True)


def discover_resegmentation_review_manifest(
    execution_manifest_path: Path | str,
    dataset_root: Path | str,
    output_root: Path | str,
    phases: Iterable[str] | None = None,
    registry_path: Path | str = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Build a sparse review manifest from completed re-segmentation cases.

    Only each case's explicitly requested masks are reviewed.  Dedicated
    vertebrae runs may retain additional generated levels for provenance without
    forcing the reviewer to inspect every level in this limited pilot.
    """

    from tools.quadra.totalsegmentator.resegmentation import (
        case_output_directory,
        load_resegmentation_manifest,
        verify_source_file,
    )

    execution_manifest_path = Path(execution_manifest_path).expanduser().resolve()
    execution = load_resegmentation_manifest(execution_manifest_path)
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    selected_phases = set(phases or execution["phase_order"])
    unknown = sorted(selected_phases - set(execution["phase_order"]))
    if unknown:
        raise ReviewError(f"Unknown re-segmentation review phases: {unknown}")

    registry_path = Path(registry_path).expanduser().resolve()
    registry = load_registry(registry_path)
    display_names = {
        entry["filename"]: entry["display_name"] for entry in _organ_catalog(registry)
    }
    scan_map: dict[tuple[str, str], dict[str, Any]] = {}
    mask_keys: set[tuple[str, str, str]] = set()
    for case in execution["cases"]:
        if case["phase"] not in selected_phases:
            continue
        ct_path = verify_source_file(dataset_root, case["input"])
        directory = case_output_directory(output_root, execution, case)
        run_manifest_path = directory / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise ReviewError(f"Re-segmentation case is not complete: {case['case_id']}")
        try:
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewError(f"Unreadable run manifest: {run_manifest_path}") from exc
        if not run_manifest.get("completed") or run_manifest.get("case_id") != case["case_id"]:
            raise ReviewError(f"Incompatible run manifest: {run_manifest_path}")

        key = (case["subject_id"], case["session"])
        scan = scan_map.get(key)
        if scan is None:
            ct_shape, ct_affine, ct_dtype = _geometry(ct_path)
            ct_identity = _identity(ct_path)
            ct_identity.update(
                {"shape": list(ct_shape), "affine": ct_affine.tolist(), "dtype": ct_dtype}
            )
            scan = {
                "scan_id": f"{case['subject_id']}|{case['session']}",
                "subject_id": case["subject_id"],
                "subject_number": int(case["subject_id"].rsplit("_", 1)[1]),
                "session": case["session"],
                "sex": case["sex"],
                "source_selection": [],
                "ct": ct_identity,
                "masks": [],
            }
            scan_map[key] = scan
        scan["source_selection"].append(
            f"{case['phase']}:{case.get('task') or 'isolated_voxel_correction'}"
        )
        for organ in case["requested_masks"]:
            mask_key = (*key, organ)
            if mask_key in mask_keys:
                raise ReviewError(f"Duplicate re-segmentation review mask: {mask_key}")
            mask_keys.add(mask_key)
            mask_path = directory / "masks" / f"{organ}.nii.gz"
            mask_shape, mask_affine, mask_dtype = _geometry(mask_path)
            if mask_shape != tuple(scan["ct"]["shape"]) or not np.allclose(
                mask_affine, np.asarray(scan["ct"]["affine"]), atol=1e-5
            ):
                raise ReviewError(f"Mask geometry does not match CT: {mask_path}")
            identity = _identity(mask_path)
            identity.update(
                {
                    "organ": organ,
                    "display_name": display_names.get(organ, organ.replace("_", " ").title()),
                    "dtype": mask_dtype,
                }
            )
            scan["masks"].append(identity)

    if not scan_map:
        raise ReviewError("No completed re-segmentation cases were selected")
    scans = sorted(
        scan_map.values(), key=lambda value: (value["subject_number"], SESSIONS.index(value["session"]))
    )
    for scan in scans:
        scan["source_selection"] = ",".join(sorted(set(scan["source_selection"])))
    catalog_names = []
    seen_organs = set()
    for scan in scans:
        for mask in scan["masks"]:
            if mask["organ"] not in seen_organs:
                seen_organs.add(mask["organ"])
                catalog_names.append(
                    {"filename": mask["organ"], "display_name": mask["display_name"]}
                )
    subjects = list(dict.fromkeys(scan["subject_id"] for scan in scans))
    sessions = [session for session in SESSIONS if any(scan["session"] == session for scan in scans)]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset_root": str(dataset_root),
        "registry": registry_identity(registry_path),
        "organ_catalog": catalog_names,
        "subject_order": subjects,
        "session_order": sessions,
        "counts": {
            "subjects": len(subjects),
            "scans": len(scans),
            "review_items": sum(len(scan["masks"]) for scan in scans),
        },
        "source_policy": {
            "kind": "versioned flagged-mask derivative",
            "execution_manifest": str(execution_manifest_path),
            "execution_manifest_sha256": sha256_file(execution_manifest_path),
            "phases": sorted(selected_phases),
            "original_review_unchanged": True,
        },
        "scans": scans,
    }


def build_resegmentation_review_index(
    execution_manifest_path: Path | str,
    dataset_root: Path | str,
    output_root: Path | str,
    review_root: Path | str,
    phases: Iterable[str] | None = None,
    registry_path: Path | str = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    root = Path(review_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ReviewError(f"Refusing to overwrite non-empty review directory: {root}")
    manifest = discover_resegmentation_review_manifest(
        execution_manifest_path,
        dataset_root,
        output_root,
        phases,
        registry_path,
    )
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(root / "review_manifest.json", manifest)
    _atomic_write_text(root / "events.jsonl", "")
    rows = [_pending_row(item) for item in flatten_items(manifest)]
    _atomic_write_csv(root / "decisions.csv", rows, DECISION_FIELDS)
    return export_review_state(root, create_checkpoint=True)


def validate_decision(decision: str, note: str) -> None:
    if decision not in DECISIONS:
        raise ReviewError(f"Unknown review decision: {decision}")
    if decision in FLAGGED_DECISIONS and not note.strip():
        raise ReviewError(f"A note is required for {decision}")


def save_decision(
    review_root: Path | str,
    item: str,
    decision: str,
    note: str,
    view_state: dict[str, Any],
) -> dict[str, Any]:
    root = Path(review_root).expanduser().resolve()
    validate_decision(decision, note)
    manifest = load_manifest(root)
    item_map = {value["item_id"]: value for value in flatten_items(manifest)}
    if item not in item_map:
        raise ReviewError(f"Unknown review item: {item}")
    current = load_decisions(root)
    source = item_map[item]
    event = _pending_row(source)
    event.update(
        {
            "event_id": hashlib.sha256(
                f"{item}|{utc_now()}|{os.getpid()}".encode("utf-8")
            ).hexdigest()[:24],
            "event_type": "decision_saved",
            "item_id": item,
            "previous_decision": current[item]["decision"],
            "decision": decision,
            "note": note.strip(),
            "axial_slice": int(view_state["axial_slice"]),
            "coronal_slice": int(view_state["coronal_slice"]),
            "sagittal_slice": int(view_state["sagittal_slice"]),
            "window_preset": str(view_state["window_preset"]),
            "overlay_mode": str(view_state["overlay_mode"]),
            "opacity": f"{float(view_state['opacity']):.2f}",
            "view_framing": str(view_state.get("view_framing") or "cropped"),
            "crop_margin": int(view_state.get("crop_margin") or 20),
            "updated_at": utc_now(),
        }
    )
    _append_jsonl(root / "events.jsonl", event)
    export_review_state(root)
    return event


def verify_identity(identity: dict[str, Any], *, always_hash: bool = True) -> None:
    path = Path(identity["path"])
    if not path.is_file():
        raise ReviewError(f"Frozen source file is missing: {path}")
    stat = path.stat()
    if stat.st_size != int(identity["size"]):
        raise ReviewError(f"Frozen source size changed: {path}")
    if always_hash and sha256_file(path) != identity["sha256"]:
        raise ReviewError(f"Frozen source checksum changed: {path}")


def mask_bbox(mask: np.ndarray, margin: int = 5) -> dict[str, Any]:
    if mask.ndim != 3:
        raise ReviewError(f"Expected 3D mask, found {mask.ndim} dimensions")
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        raise ReviewError("Mask is empty")
    start = coordinates.min(axis=0)
    end = coordinates.max(axis=0) + 1
    shape = np.asarray(mask.shape, dtype=np.int64)
    expanded_start = np.maximum(start - int(margin), 0)
    expanded_end = np.minimum(end + int(margin), shape)
    boundary_axes = [
        axis for axis in range(3) if start[axis] == 0 or end[axis] == shape[axis]
    ]
    clipped_axes = [
        axis
        for axis in range(3)
        if expanded_start[axis] != start[axis] - int(margin)
        or expanded_end[axis] != end[axis] + int(margin)
    ]
    return {
        "start": start.tolist(),
        "end": end.tolist(),
        "expanded_start": expanded_start.tolist(),
        "expanded_end": expanded_end.tolist(),
        "center": ((start + end - 1) // 2).tolist(),
        "boundary_axes": boundary_axes,
        "margin_clipped_axes": clipped_axes,
        "touches_volume_boundary": bool(boundary_axes),
        "margin_clipped": bool(clipped_axes),
    }


def plane_slice(data: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        raw = data[:, :, index]
    elif plane == "coronal":
        raw = data[:, index, :]
    elif plane == "sagittal":
        raw = data[index, :, :]
    else:
        raise ReviewError(f"Unknown plane: {plane}")
    return np.rot90(raw)


def plane_projection(mask: np.ndarray, plane: str) -> np.ndarray:
    if plane == "axial":
        raw = mask.any(axis=2)
    elif plane == "coronal":
        raw = mask.any(axis=1)
    elif plane == "sagittal":
        raw = mask.any(axis=0)
    else:
        raise ReviewError(f"Unknown plane: {plane}")
    return np.rot90(raw)


def voxel_to_display(
    voxel: tuple[int, int, int] | list[int],
    plane: str,
    shape: tuple[int, int, int] | list[int],
) -> tuple[int, int]:
    """Map canonical NIfTI voxel coordinates to the rotated display image."""
    x, y, z = (int(value) for value in voxel)
    size_x, size_y, size_z = (int(value) for value in shape)
    if not (0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z):
        raise ReviewError(f"Voxel coordinate is outside the volume: {(x, y, z)}")
    if plane == "axial":
        return x, size_y - 1 - y
    if plane == "coronal":
        return x, size_z - 1 - z
    if plane == "sagittal":
        return y, size_z - 1 - z
    raise ReviewError(f"Unknown plane: {plane}")


def display_to_voxel(
    display_x: float,
    display_y: float,
    plane: str,
    index: int,
    shape: tuple[int, int, int] | list[int],
) -> tuple[int, int, int]:
    """Map a click in a rotated plane image to canonical voxel coordinates."""
    size_x, size_y, size_z = (int(value) for value in shape)
    if plane == "axial":
        x = min(max(int(round(display_x)), 0), size_x - 1)
        y_display = min(max(int(round(display_y)), 0), size_y - 1)
        z = min(max(int(index), 0), size_z - 1)
        return x, size_y - 1 - y_display, z
    if plane == "coronal":
        x = min(max(int(round(display_x)), 0), size_x - 1)
        z_display = min(max(int(round(display_y)), 0), size_z - 1)
        y = min(max(int(index), 0), size_y - 1)
        return x, y, size_z - 1 - z_display
    if plane == "sagittal":
        y = min(max(int(round(display_x)), 0), size_y - 1)
        z_display = min(max(int(round(display_y)), 0), size_z - 1)
        x = min(max(int(index), 0), size_x - 1)
        return x, y, size_z - 1 - z_display
    raise ReviewError(f"Unknown plane: {plane}")


def projection_bbox(projection: np.ndarray, margin: int = 5) -> tuple[int, int, int, int, bool]:
    coordinates = np.argwhere(projection)
    if not coordinates.size:
        raise ReviewError("Mask projection is empty")
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    height, width = projection.shape
    expanded = (
        max(int(x0) - margin, 0),
        max(int(y0) - margin, 0),
        min(int(x1) + margin, width),
        min(int(y1) + margin, height),
    )
    clipped = (
        expanded[0] != int(x0) - margin
        or expanded[1] != int(y0) - margin
        or expanded[2] != int(x1) + margin
        or expanded[3] != int(y1) + margin
    )
    return (*expanded, clipped)


def subject_complete(
    state: dict[str, dict[str, Any]], subject_id: str
) -> bool:
    rows = [row for row in state.values() if row["subject_id"] == subject_id]
    return bool(rows) and all(row["decision"] != "pending" for row in rows)
