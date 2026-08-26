"""Checksum-verified backup of generated Quadra evidence.

The local archive is deliberately a partial mirror of ``QUADRA_STORAGE_ROOT``.
It contains generated runs and provenance, never datasets, model weights,
package environments, or caches.  The module supports Python 3.7 so that the
same inventory and packaging code can run in both Quadra container profiles.
"""

from __future__ import print_function

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


BACKUP_SCHEMA_VERSION = 1
DEFAULT_REMOTE_ROOT = Path("/workspace/quadra")
DEFAULT_REMOTE_REPOSITORY = Path("/workspace/repos/uae-quadra-validation")
FIVE_GIB = 5 * 1024 ** 3

ALLOWLIST = (
    "runs/cohort",
    "runs/memory_optimization",
    "runs/preprocessing",
    "runs/uae",
    "runs/archive",
    "metadata/manifests",
)

EXCLUDED_ROOTS = (
    "datasets",
    "models",
    "cache",
    "runtime",
    "vendor",
    "staging",
    "checkpoints",
)

LOCAL_DIRECTORIES = (
    "runs/cohort",
    "runs/memory_optimization",
    "runs/preprocessing",
    "runs/uae",
    "runs/archive",
    "runs/analysis",
    "reviews/masks",
    "reviews/query_points",
    "metadata/manifests",
    "metadata/known_assets",
    "metadata/transfer_receipts",
    "exports/documents",
    "exports/presentations",
    "transfer/incoming",
    "transfer/inventories",
    "transfer/conflicts",
    "transfer/packages",
    "scratch",
)

IGNORED_LOCAL_NAMES = {".DS_Store", "__pycache__"}
IGNORED_LOCAL_SUFFIXES = {".pyc", ".tmp"}

ACTIVE_PROCESS_PATTERNS = (
    "aligned_organ_group_cohort",
    "plot_aligned_cohort_cycle_error",
    "streaming_cycle_error",
    "TotalSegmentator",
    "totalsegmentator",
    "artifact_backup backup-remote-package",
)


class BackupError(RuntimeError):
    """Raised when a backup cannot proceed without risking evidence."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def utc_transfer_id():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, str(path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _is_within(path, root):
    path_text = str(Path(path).resolve())
    root_text = str(Path(root).resolve())
    return path_text == root_text or path_text.startswith(root_text + os.sep)


def validate_archive_root(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise BackupError("Archive root must be absolute: {}".format(path))
    if str(path) in {"/", "/workspace", str(Path.home())}:
        raise BackupError("Refusing unsafe broad archive root: {}".format(path))
    return path


def local_layout(root):
    root = validate_archive_root(root)
    layout = {"root": root}
    for relative in LOCAL_DIRECTORIES:
        layout[relative.replace("/", "_")] = root / relative
    return layout


def prepare_local_layout(root):
    root = validate_archive_root(root)
    for relative in LOCAL_DIRECTORIES:
        path = root / relative
        if path.is_symlink():
            target = path.resolve()
            if not path.exists() or not _is_within(target, root):
                raise BackupError(
                    "Unsafe or broken archive link: {} -> {}".format(path, target)
                )
        else:
            path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "created_at": utc_now(),
        "archive_root": str(root),
        "mirrored_roots": list(ALLOWLIST),
        "excluded_roots": list(EXCLUDED_ROOTS),
        "policy": {
            "direction": "runpod_to_local",
            "delete_source": False,
            "overwrite_conflicts": False,
            "datasets_included": False,
            "models_included": False,
            "environments_included": False,
        },
    }
    # Keep local-only policy outside the mirrored manifest namespace so it can
    # never collide with a remote scientific manifest of the same name.
    manifest_path = root / "metadata/known_assets/local_archive_layout.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)
    return manifest_path


def _git_value(repository, args):
    repository = Path(repository)
    if not (repository / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository)] + list(args),
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def repository_state(repository):
    repository = Path(repository)
    branch = _git_value(repository, ["branch", "--show-current"])
    if not branch:
        branch = _git_value(repository, ["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "path": str(repository),
        "branch": branch,
        "commit": _git_value(repository, ["rev-parse", "HEAD"]),
        "status_porcelain": _git_value(
            repository, ["status", "--porcelain=v1", "--untracked-files=all"]
        ),
    }


def _category(relative):
    parts = Path(relative).parts
    if len(parts) >= 2 and parts[0] == "runs":
        return "run_{}".format(parts[1])
    if parts[:2] == ("metadata", "manifests"):
        return "metadata_manifest"
    return "unknown"


def _safe_link_target(path, root):
    raw_target = os.readlink(str(path))
    resolved = (path.parent / raw_target).resolve()
    if not _is_within(resolved, root):
        raise BackupError(
            "Symlink escapes archive root: {} -> {}".format(path, raw_target)
        )
    if not resolved.exists():
        raise BackupError("Broken symlink: {} -> {}".format(path, raw_target))
    return raw_target


def _file_entry(path, root):
    relative = path.relative_to(root).as_posix()
    item_stat = path.lstat()
    entry = {
        "path": relative,
        "category": _category(relative),
        "mode": stat.S_IMODE(item_stat.st_mode),
        "mtime_ns": getattr(
            item_stat, "st_mtime_ns", int(item_stat.st_mtime * 1000000000)
        ),
    }
    if path.is_symlink():
        target = _safe_link_target(path, root)
        entry.update(
            {
                "type": "symlink",
                "link_target": target,
                "size": len(target.encode("utf-8")),
                "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            }
        )
    elif path.is_file():
        entry.update(
            {
                "type": "file",
                "size": item_stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    else:
        raise BackupError("Unsupported inventory object: {}".format(path))
    return entry


def _run_level_status_values(value):
    """Return only run-level state, never nested scan or point statuses."""
    if not isinstance(value, dict):
        return []
    found = []
    for key in ("status", "state", "run_status", "overall_status"):
        if key in value and not isinstance(value[key], (dict, list)):
            found.append(str(value[key]).upper())
    for key in ("gate_passed", "completed", "complete"):
        if key in value and isinstance(value[key], bool):
            found.append("COMPLETE" if value[key] else "NOT_COMPLETE")
    # Some controller manifests keep their run-level state in one explicit
    # wrapper. Do not recurse through arbitrary result collections.
    for wrapper in ("run", "controller", "progress"):
        child = value.get(wrapper)
        if isinstance(child, dict):
            for key in ("status", "state", "run_status", "overall_status"):
                if key in child and not isinstance(child[key], (dict, list)):
                    found.append(str(child[key]).upper())
    return found


def _run_state(run_directory):
    candidates = []
    for path in sorted(Path(run_directory).glob("*.json")):
        lowered = path.name.lower()
        if any(token in lowered for token in ("manifest", "summary", "status")):
            candidates.append(path)
    values = []
    for path in candidates[:20]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                values.extend(_run_level_status_values(json.load(handle)))
        except (OSError, ValueError):
            continue
    in_progress_tokens = ("IN_PROGRESS", "RUNNING", "STARTED", "PENDING")
    failed_tokens = ("FAILED", "ERROR", "BLOCKED", "NOT_COMPLETE")
    complete_tokens = (
        "COMPLETE",
        "COMPLETED",
        "SUCCESS",
        "SUCCEEDED",
        "TECHNICAL_PASS",
        "PASS",
        "OK",
    )
    if any(any(token in item for token in in_progress_tokens) for item in values):
        return "IN_PROGRESS"
    if any(any(token in item for token in failed_tokens) for item in values):
        return "FAILED"
    if any(any(token in item for token in complete_tokens) for item in values):
        return "COMPLETE"
    return "UNKNOWN"


def build_inventory(root, repository=None, source_id=None):
    root = validate_archive_root(root)
    if not root.exists():
        raise BackupError("Inventory root does not exist: {}".format(root))
    entries = []
    missing_roots = []
    run_states = {}
    for allowed in ALLOWLIST:
        allowed_path = root / allowed
        if not allowed_path.exists():
            missing_roots.append(allowed)
            continue
        if allowed_path.is_symlink():
            _safe_link_target(allowed_path, root)
        if allowed.startswith("runs/"):
            for child in sorted(allowed_path.iterdir()):
                if child.is_dir() and not child.is_symlink():
                    run_states[child.relative_to(root).as_posix()] = _run_state(child)
        for directory, directory_names, file_names in os.walk(
            str(allowed_path), followlinks=False
        ):
            directory_path = Path(directory)
            directory_names[:] = sorted(directory_names)
            file_names = sorted(file_names)
            for name in list(directory_names):
                child = directory_path / name
                if child.is_symlink():
                    entries.append(_file_entry(child, root))
                    directory_names.remove(name)
            for name in file_names:
                path = directory_path / name
                entries.append(_file_entry(path, root))
    entries.sort(key=lambda item: item["path"])
    inventory = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_id": source_id or str(root),
        "root": str(root),
        "allowlist": list(ALLOWLIST),
        "excluded_roots": list(EXCLUDED_ROOTS),
        "missing_roots": missing_roots,
        "entries": entries,
        "run_states": run_states,
        "summary": {
            "files": len(entries),
            "bytes": sum(item["size"] for item in entries),
            "in_progress_runs": sorted(
                key for key, value in run_states.items() if value == "IN_PROGRESS"
            ),
        },
        "repository": repository_state(repository) if repository else None,
    }
    return inventory


def _entry_map(inventory):
    return {item["path"]: item for item in inventory.get("entries", [])}


def compare_inventories(remote, local, previous=None):
    remote_entries = _entry_map(remote)
    local_entries = _entry_map(local)
    previous_entries = _entry_map(previous or {})
    rows = []
    for path in sorted(set(remote_entries) | set(local_entries)):
        remote_item = remote_entries.get(path)
        local_item = local_entries.get(path)
        previous_item = previous_entries.get(path)
        if remote_item is None:
            status_value = "LOCAL_ONLY"
        elif local_item is None:
            status_value = "REMOTE_ONLY"
        elif (
            remote_item.get("sha256") == local_item.get("sha256")
            and remote_item.get("size") == local_item.get("size")
            and remote_item.get("type") == local_item.get("type")
        ):
            status_value = "IDENTICAL"
        elif previous_item:
            remote_same = remote_item.get("sha256") == previous_item.get("sha256")
            local_same = local_item.get("sha256") == previous_item.get("sha256")
            if not remote_same and local_same:
                status_value = "REMOTE_CHANGED"
            elif remote_same and not local_same:
                status_value = "LOCAL_CHANGED"
            else:
                status_value = "CONFLICT"
        else:
            status_value = "CONFLICT"
        rows.append(
            {
                "path": path,
                "status": status_value,
                "remote_sha256": remote_item.get("sha256") if remote_item else None,
                "local_sha256": local_item.get("sha256") if local_item else None,
                "size": (
                    remote_item.get("size")
                    if remote_item
                    else local_item.get("size")
                ),
            }
        )
    for root in remote.get("excluded_roots", []):
        rows.append({"path": root, "status": "EXCLUDED"})
    for run_path in remote.get("summary", {}).get("in_progress_runs", []):
        rows.append({"path": run_path, "status": "IN_PROGRESS"})
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"rows": rows, "counts": counts}


def _ignored_local(path):
    return path.name in IGNORED_LOCAL_NAMES or path.suffix in IGNORED_LOCAL_SUFFIXES


def _copy_file_verified(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == source_hash:
            return "existing", source_hash
        raise BackupError("Local intake conflict: {}".format(destination))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        shutil.copy2(str(source), temporary_name)
        if sha256_file(temporary_name) != source_hash:
            raise BackupError("Checksum mismatch while copying {}".format(source))
        os.replace(temporary_name, str(destination))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return "copied", source_hash


def _copy_tree_verified(source, destination, records, exclude=None):
    source = Path(source)
    destination = Path(destination)
    exclude = exclude or (lambda path: False)
    if not source.exists():
        return
    for directory, directory_names, file_names in os.walk(str(source)):
        directory_path = Path(directory)
        directory_names[:] = [
            name
            for name in sorted(directory_names)
            if not _ignored_local(directory_path / name)
            and not exclude(directory_path / name)
        ]
        for name in sorted(file_names):
            path = directory_path / name
            if _ignored_local(path) or exclude(path):
                continue
            relative = path.relative_to(source)
            target = destination / relative
            status_value, digest = _copy_file_verified(path, target)
            records.append(
                {
                    "source": str(path),
                    "destination": str(target),
                    "status": status_value,
                    "size": path.stat().st_size,
                    "sha256": digest,
                }
            )


def _record_known_assets(repository, local_root):
    repository = Path(repository)
    candidates = [
        repository
        / "outputs/quadra_totalsegmentator_stage5/backup/quadra-totalsegmentator-stage5-20260727.tar",
    ]
    records = []
    for path in candidates:
        if not path.is_file():
            continue
        sidecar = Path(str(path) + ".sha256")
        declared = None
        if sidecar.is_file():
            declared = sidecar.read_text(encoding="utf-8").strip().split()[0]
        records.append(
            {
                "asset": "totalsegmentator_stage5_masks",
                "path": str(path),
                "size": path.stat().st_size,
                "declared_sha256": declared,
                "verification": "existing_local_backup",
                "copied_into_archive": False,
            }
        )
    result = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "assets": records,
    }
    atomic_write_json(
        Path(local_root) / "metadata/known_assets/local_existing_assets.json", result
    )
    return result


def ingest_repository_artifacts(repository, local_root):
    repository = Path(repository).resolve()
    local_root = validate_archive_root(local_root)
    records = []
    mappings = [
        (
            repository / "outputs/quadra_cohort_analysis",
            local_root / "runs/analysis/quadra_cohort_analysis",
            None,
        ),
        (
            repository / "outputs/registration_cycle_error_matchSam",
            local_root / "runs/analysis/registration_cycle_error_matchSam",
            None,
        ),
        (
            repository / "outputs/documents",
            local_root / "exports/documents",
            None,
        ),
        (
            repository / "outputs/presentations",
            local_root / "exports/presentations",
            None,
        ),
        (
            repository / "data/quadra_output",
            local_root / "runs/archive/local_legacy_quadra_output",
            None,
        ),
        (
            repository / "reports/quadra",
            local_root / "exports/documents/reports_quadra",
            lambda path: path.suffix.lower() == ".pptx",
        ),
        (
            repository / "reports/quadra",
            local_root / "exports/presentations/reports_quadra",
            lambda path: path.suffix.lower() != ".pptx",
        ),
    ]
    stage3_root = repository / "outputs"
    if stage3_root.exists():
        for source in sorted(stage3_root.glob("stage*-*")):
            if source.is_dir():
                mappings.append(
                    (source, local_root / "runs/memory_optimization" / source.name, None)
                )
    stage5 = repository / "outputs/quadra_totalsegmentator_stage5"
    if stage5.exists():
        excluded_stage5 = lambda path: "backup" in path.parts or path.name == "package_backup.sh"
        mappings.append(
            (
                stage5,
                local_root / "reviews/masks/totalsegmentator_2.16.0_stage5",
                excluded_stage5,
            )
        )
    for source, destination, exclude in mappings:
        _copy_tree_verified(source, destination, records, exclude=exclude)
    known_assets = _record_known_assets(repository, local_root)
    result = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "repository": repository_state(repository),
        "records": records,
        "known_assets": known_assets,
        "summary": {
            "files": len(records),
            "bytes": sum(item["size"] for item in records),
            "copied": sum(item["status"] == "copied" for item in records),
            "existing": sum(item["status"] == "existing" for item in records),
        },
    }
    path = (
        local_root
        / "metadata/manifests"
        / "local_intake_{}.json".format(utc_transfer_id())
    )
    atomic_write_json(path, result)
    return result


def _tar_filter(info):
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def create_package(root, package_path, inventory):
    root = validate_archive_root(root)
    package_path = Path(package_path)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = package_path.parent / ("." + package_path.name + ".tmp")
    manifest_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with temporary_path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                manifest_info = tarfile.TarInfo(".backup/remote_inventory.json")
                manifest_info.size = len(manifest_bytes)
                manifest_info.mode = 0o644
                manifest_info.mtime = 0
                archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
                for allowed in ALLOWLIST:
                    path = root / allowed
                    if path.exists():
                        archive.add(
                            str(path),
                            arcname="quadra/" + allowed,
                            recursive=True,
                            filter=_tar_filter,
                        )
    os.replace(str(temporary_path), str(package_path))
    return {
        "path": str(package_path),
        "size": package_path.stat().st_size,
        "sha256": sha256_file(package_path),
    }


def _safe_member_target(root, member_name):
    member_path = Path(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise BackupError("Unsafe archive member: {}".format(member_name))
    target = root / member_path
    if not _is_within(target.parent, root):
        raise BackupError("Archive member escapes extraction root: {}".format(member_name))
    return target


def safe_extract(package_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(package_path), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = _safe_member_target(destination, member.name)
            if member.issym() or member.islnk():
                link_target = (target.parent / member.linkname).resolve()
                if not _is_within(link_target, destination):
                    raise BackupError(
                        "Archive link escapes extraction root: {} -> {}".format(
                            member.name, member.linkname
                        )
                    )
        archive.extractall(str(destination), members=members)
    manifest_path = destination / ".backup/remote_inventory.json"
    if not manifest_path.is_file():
        raise BackupError("Backup package is missing its remote inventory")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _tree_signature(root):
    root = Path(root)
    entries = []
    if root.is_file():
        return [(root.name, root.stat().st_size, sha256_file(root))]
    for directory, directory_names, file_names in os.walk(str(root)):
        directory_path = Path(directory)
        directory_names[:] = sorted(directory_names)
        for name in sorted(file_names):
            path = directory_path / name
            entries.append(
                (
                    path.relative_to(root).as_posix(),
                    path.stat().st_size,
                    sha256_file(path),
                )
            )
    return entries


def promote_extracted(extraction_root, local_root, transfer_id):
    extraction_root = Path(extraction_root)
    source_root = extraction_root / "quadra"
    local_root = validate_archive_root(local_root)
    conflict_root = local_root / "transfer/conflicts" / transfer_id
    operations = []
    conflicts = []
    for allowed in ALLOWLIST:
        source_parent = source_root / allowed
        if not source_parent.exists():
            continue
        destination_parent = local_root / allowed
        destination_parent.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_parent.iterdir()):
            destination = destination_parent / source.name
            if destination.exists():
                if _tree_signature(source) == _tree_signature(destination):
                    operations.append(
                        {"path": str(destination), "status": "existing"}
                    )
                else:
                    conflict_destination = conflict_root / allowed / source.name
                    conflict_destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(str(source), str(conflict_destination))
                    conflicts.append(
                        {
                            "path": str(destination),
                            "incoming": str(conflict_destination),
                            "status": "CONFLICT",
                        }
                    )
            else:
                os.replace(str(source), str(destination))
                operations.append({"path": str(destination), "status": "promoted"})
    if conflicts:
        raise BackupError(
            "Incoming evidence conflicts with existing local content; preserved at {}".format(
                conflict_root
            )
        )
    return operations


def _ssh_arguments(args):
    values = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-p",
        str(args.ssh_port),
    ]
    if args.ssh_key:
        values.extend(["-i", str(Path(args.ssh_key).expanduser())])
    values.append(args.ssh_host)
    return values


def _remote_module_command(args, command, extra=None):
    values = [
        args.remote_python,
        "-m",
        "tools.quadra.artifact_backup",
        command,
        "--remote-root",
        str(args.remote_root),
        "--repository-root",
        str(args.remote_repository),
    ]
    if extra:
        values.extend(extra)
    return "cd {} && {}".format(
        shlex.quote(str(args.remote_repository)),
        " ".join(shlex.quote(str(value)) for value in values),
    )


def _run_remote_json(args, command, extra=None):
    process = subprocess.run(
        _ssh_arguments(args) + [_remote_module_command(args, command, extra)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if process.returncode:
        raise BackupError(
            "Remote command failed ({}): {}".format(
                process.returncode, process.stderr.strip() or process.stdout.strip()
            )
        )
    try:
        return json.loads(process.stdout)
    except ValueError:
        raise BackupError("Remote command returned invalid JSON: {}".format(process.stdout))


def _inventory_paths(local_root, transfer_id):
    root = Path(local_root) / "transfer/inventories" / transfer_id
    return {
        "root": root,
        "remote": root / "remote_inventory.json",
        "local": root / "local_inventory.json",
        "comparison": root / "comparison.json",
        "plan": root / "backup_plan.json",
    }


def _latest_remote_inventory(local_root):
    candidates = sorted(
        (Path(local_root) / "metadata/transfer_receipts").glob(
            "*/remote_inventory.json"
        )
    )
    if not candidates:
        return None
    with candidates[-1].open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_backup_plan(args, transfer_id=None):
    local_root = validate_archive_root(args.local_root)
    prepare_local_layout(local_root)
    transfer_id = transfer_id or utc_transfer_id()
    remote = _run_remote_json(args, "backup-remote-inventory")
    local = build_inventory(local_root, source_id="local_archive")
    previous = _latest_remote_inventory(local_root)
    comparison = compare_inventories(remote, local, previous=previous)
    disk = shutil.disk_usage(str(local_root))
    required_free = remote["summary"]["bytes"] * 2 + FIVE_GIB
    plan = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "transfer_id": transfer_id,
        "created_at": utc_now(),
        "local_root": str(local_root),
        "remote_root": str(args.remote_root),
        "remote_source_id": remote.get("source_id"),
        "disk": {
            "free_bytes": disk.free,
            "required_free_bytes": required_free,
            "sufficient": disk.free >= required_free,
        },
        "comparison_counts": comparison["counts"],
        "remote_summary": remote["summary"],
        "ready": disk.free >= required_free
        and not remote["summary"].get("in_progress_runs"),
    }
    paths = _inventory_paths(local_root, transfer_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["remote"], remote)
    atomic_write_json(paths["local"], local)
    atomic_write_json(paths["comparison"], comparison)
    atomic_write_json(paths["plan"], plan)
    return plan, remote, local, comparison


def _scp_download(args, remote_path, destination):
    values = ["scp", "-P", str(args.ssh_port)]
    if args.ssh_key:
        values.extend(["-i", str(Path(args.ssh_key).expanduser())])
    values.extend(["{}:{}".format(args.ssh_host, remote_path), str(destination)])
    process = subprocess.run(values)
    if process.returncode:
        raise BackupError("SCP download failed with exit code {}".format(process.returncode))


def _receipt_directory(local_root, transfer_id):
    return Path(local_root) / "metadata/transfer_receipts" / transfer_id


def _write_receipt(local_root, transfer_id, remote, local, comparison, package):
    receipt_root = _receipt_directory(local_root, transfer_id)
    receipt_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(receipt_root / "remote_inventory.json", remote)
    atomic_write_json(receipt_root / "local_inventory.json", local)
    atomic_write_json(receipt_root / "comparison.json", comparison)
    blocking = {
        key: value
        for key, value in comparison["counts"].items()
        if key in {"REMOTE_ONLY", "REMOTE_CHANGED", "CONFLICT", "IN_PROGRESS"}
        and value
    }
    receipt = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "transfer_id": transfer_id,
        "verified_at": utc_now(),
        "status": "VERIFIED" if not blocking else "FAILED",
        "blocking_counts": blocking,
        "package": package,
        "source_deleted": False,
    }
    atomic_write_json(receipt_root / "receipt.json", receipt)
    return receipt


def command_backup_init(args):
    manifest_path = prepare_local_layout(args.local_root)
    intake = ingest_repository_artifacts(args.repository_root, args.local_root)
    result = {
        "status": "ok",
        "archive_root": str(validate_archive_root(args.local_root)),
        "layout_manifest": str(manifest_path),
        "intake_summary": intake["summary"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_backup_plan(args):
    plan, _remote, _local, _comparison = build_backup_plan(args, args.transfer_id)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0 if plan["ready"] else 2


def command_remote_inventory(args):
    inventory = build_inventory(
        args.remote_root,
        repository=args.repository_root,
        source_id=os.environ.get("RUNPOD_POD_ID") or os.uname()[1],
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


def command_remote_package(args):
    inventory = build_inventory(
        args.remote_root,
        repository=args.repository_root,
        source_id=os.environ.get("RUNPOD_POD_ID") or os.uname()[1],
    )
    package_root = (
        Path(args.remote_root) / "staging/artifact-backup" / args.transfer_id
    )
    package_path = package_root / "quadra-artifacts-{}.tar.gz".format(
        args.transfer_id
    )
    package = create_package(args.remote_root, package_path, inventory)
    result = {"inventory": inventory, "package": package}
    atomic_write_json(package_root / "package.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_backup_pull(args):
    local_root = validate_archive_root(args.local_root)
    prepare_local_layout(local_root)
    transfer_id = args.transfer_id or utc_transfer_id()
    if args.transport == "local-package":
        if not args.package_file:
            raise BackupError("--transport local-package requires --package-file")
        package_file = Path(args.package_file).expanduser().resolve()
        if not package_file.is_file():
            raise BackupError("Received package does not exist: {}".format(package_file))
        observed_hash = sha256_file(package_file)
        if args.package_sha256 and observed_hash != args.package_sha256:
            raise BackupError("Received package checksum does not match --package-sha256")
        incoming_root = local_root / "transfer/incoming" / transfer_id
        incoming_root.mkdir(parents=True, exist_ok=False)
        local_package = incoming_root / package_file.name
        shutil.copy2(str(package_file), str(local_package))
        if sha256_file(local_package) != observed_hash:
            raise BackupError("Copied package checksum changed during local intake")
        extraction_root = incoming_root / "extracted"
        remote = safe_extract(local_package, extraction_root)
        disk = shutil.disk_usage(str(local_root))
        if disk.free < remote["summary"]["bytes"] * 2 + FIVE_GIB:
            raise BackupError("Insufficient local disk space for verified transfer")
        operations = promote_extracted(extraction_root, local_root, transfer_id)
        local = build_inventory(local_root, source_id="local_archive")
        comparison = compare_inventories(remote, local, previous=remote)
        receipt = _write_receipt(
            local_root,
            transfer_id,
            remote,
            local,
            comparison,
            {
                "transport": "runpodctl_or_manual_package",
                "received_from": str(package_file),
                "local_path": str(local_package),
                "size": local_package.stat().st_size,
                "sha256": observed_hash,
                "promotion_operations": operations,
            },
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "VERIFIED" else 2
    if not args.ssh_host:
        raise BackupError("SSH transport requires --ssh-host")
    plan, planned_remote, _local, _comparison = build_backup_plan(
        args, transfer_id
    )
    if not plan["disk"]["sufficient"]:
        raise BackupError("Insufficient local disk space for verified transfer")
    if planned_remote["summary"].get("in_progress_runs"):
        raise BackupError("Remote inventory contains in-progress runs")
    package_result = _run_remote_json(
        args,
        "backup-remote-package",
        ["--transfer-id", transfer_id],
    )
    remote = package_result["inventory"]
    package = package_result["package"]
    incoming_root = local_root / "transfer/incoming" / transfer_id
    incoming_root.mkdir(parents=True, exist_ok=False)
    local_package = incoming_root / Path(package["path"]).name
    if args.transport == "runpodctl":
        result = {
            "status": "PACKAGE_READY",
            "transfer_id": transfer_id,
            "package": package,
            "remote_command": "runpodctl send {}".format(
                shlex.quote(package["path"])
            ),
            "local_import": (
                "bash setup.sh backup-pull --transport local-package "
                "--local-root {} --transfer-id {} --package-file <received-file> "
                "--package-sha256 {}"
            ).format(
                shlex.quote(str(local_root)), transfer_id, package["sha256"]
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 3
    _scp_download(args, package["path"], local_package)
    if local_package.stat().st_size != package["size"]:
        raise BackupError("Downloaded package size does not match remote package")
    if sha256_file(local_package) != package["sha256"]:
        raise BackupError("Downloaded package checksum does not match remote package")
    extraction_root = incoming_root / "extracted"
    embedded_inventory = safe_extract(local_package, extraction_root)
    if _entry_map(embedded_inventory) != _entry_map(remote):
        raise BackupError("Embedded and live remote inventories differ")
    operations = promote_extracted(extraction_root, local_root, transfer_id)
    local = build_inventory(local_root, source_id="local_archive")
    comparison = compare_inventories(remote, local, previous=planned_remote)
    receipt = _write_receipt(
        local_root,
        transfer_id,
        remote,
        local,
        comparison,
        {
            "transport": "ssh_scp",
            "remote": package,
            "local_path": str(local_package),
            "promotion_operations": operations,
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "VERIFIED" else 2


def _load_receipt(local_root, transfer_id):
    receipt_root = _receipt_directory(local_root, transfer_id)
    try:
        with (receipt_root / "receipt.json").open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
        with (receipt_root / "remote_inventory.json").open(
            "r", encoding="utf-8"
        ) as handle:
            remote = json.load(handle)
    except OSError:
        raise BackupError("Transfer receipt does not exist: {}".format(transfer_id))
    return receipt, remote


def command_backup_verify(args):
    local_root = validate_archive_root(args.local_root)
    receipt, remote = _load_receipt(local_root, args.transfer_id)
    local = build_inventory(local_root, source_id="local_archive")
    comparison = compare_inventories(remote, local, previous=remote)
    verified = not any(
        comparison["counts"].get(name, 0)
        for name in ("REMOTE_ONLY", "REMOTE_CHANGED", "CONFLICT", "IN_PROGRESS")
    )
    result = {
        "status": "VERIFIED" if verified else "FAILED",
        "transfer_id": args.transfer_id,
        "comparison_counts": comparison["counts"],
        "original_receipt_status": receipt.get("status"),
        "verified_at": utc_now(),
    }
    atomic_write_json(
        _receipt_directory(local_root, args.transfer_id) / "reverification.json",
        result,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if verified else 2


def command_backup_status(args):
    local_root = validate_archive_root(args.local_root)
    local = build_inventory(local_root, source_id="local_archive")
    result = {
        "status": "local_only",
        "local_summary": local["summary"],
        "latest_verified_transfer": None,
    }
    receipt_files = sorted(
        (local_root / "metadata/transfer_receipts").glob("*/receipt.json")
    )
    if receipt_files:
        with receipt_files[-1].open("r", encoding="utf-8") as handle:
            result["latest_verified_transfer"] = json.load(handle)
    if args.ssh_host:
        remote = _run_remote_json(args, "backup-remote-inventory")
        comparison = compare_inventories(
            remote, local, previous=_latest_remote_inventory(local_root)
        )
        result.update(
            {
                "status": "compared",
                "remote_summary": remote["summary"],
                "comparison_counts": comparison["counts"],
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def process_status(repository):
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,stat=,comm=,args="],
        universal_newlines=True,
    )
    active = []
    for line in output.splitlines():
        if "artifact_backup backup-remote-status" in line:
            continue
        if any(pattern in line for pattern in ACTIVE_PROCESS_PATTERNS):
            active.append(line.strip())
    return {
        "checked_at": utc_now(),
        "active_processes": active,
        "repository": repository_state(repository),
    }


def command_remote_status(args):
    print(json.dumps(process_status(args.repository_root), indent=2, sort_keys=True))
    return 0


def command_safe_stop(args):
    if not args.ssh_host:
        raise BackupError("safe-stop-check requires --ssh-host")
    local_root = validate_archive_root(args.local_root)
    remote = _run_remote_json(args, "backup-remote-inventory")
    runtime = _run_remote_json(args, "backup-remote-status")
    local = build_inventory(local_root, source_id="local_archive")
    comparison = compare_inventories(
        remote, local, previous=_latest_remote_inventory(local_root)
    )
    blocking_counts = {
        name: comparison["counts"].get(name, 0)
        for name in ("REMOTE_ONLY", "REMOTE_CHANGED", "CONFLICT", "IN_PROGRESS")
        if comparison["counts"].get(name, 0)
    }
    repository_dirty = bool(
        runtime.get("repository", {}).get("status_porcelain", "").strip()
    )
    safe = not blocking_counts and not runtime["active_processes"] and not repository_dirty
    result = {
        "verdict": "SAFE_TO_STOP" if safe else "NOT_SAFE_TO_STOP",
        "checked_at": utc_now(),
        "blocking_counts": blocking_counts,
        "active_processes": runtime["active_processes"],
        "repository_dirty": repository_dirty,
        "repository": runtime.get("repository"),
        "remote_summary": remote["summary"],
        "local_summary": local["summary"],
        "note": "This command never stops or terminates the pod.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if safe else 2


def _add_connection_arguments(parser, required=True):
    parser.add_argument(
        "--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT
    )
    parser.add_argument(
        "--remote-repository", type=Path, default=DEFAULT_REMOTE_REPOSITORY
    )
    parser.add_argument(
        "--ssh-host",
        default=os.environ.get("QUADRA_POD_SSH_HOST"),
        required=required and not os.environ.get("QUADRA_POD_SSH_HOST"),
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=int(os.environ.get("QUADRA_POD_SSH_PORT", "22")),
    )
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=Path(os.environ.get("QUADRA_POD_SSH_KEY", "~/.ssh/id_ed25519")),
    )
    parser.add_argument("--remote-python", default="python3")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Back up generated Quadra evidence without copying datasets or models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("backup-init")
    init_parser.add_argument("--local-root", type=Path, required=True)
    init_parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    init_parser.set_defaults(handler=command_backup_init)

    plan_parser = subparsers.add_parser("backup-plan")
    plan_parser.add_argument("--local-root", type=Path, required=True)
    plan_parser.add_argument("--transfer-id")
    _add_connection_arguments(plan_parser)
    plan_parser.set_defaults(handler=command_backup_plan)

    pull_parser = subparsers.add_parser("backup-pull")
    pull_parser.add_argument("--local-root", type=Path, required=True)
    pull_parser.add_argument("--transfer-id")
    pull_parser.add_argument(
        "--transport", choices=("auto", "ssh", "runpodctl", "local-package"), default="auto"
    )
    pull_parser.add_argument("--package-file", type=Path)
    pull_parser.add_argument("--package-sha256")
    _add_connection_arguments(pull_parser, required=False)
    pull_parser.set_defaults(handler=command_backup_pull)

    verify_parser = subparsers.add_parser("backup-verify")
    verify_parser.add_argument("--local-root", type=Path, required=True)
    verify_parser.add_argument("--transfer-id", required=True)
    verify_parser.set_defaults(handler=command_backup_verify)

    status_parser = subparsers.add_parser("backup-status")
    status_parser.add_argument("--local-root", type=Path, required=True)
    _add_connection_arguments(status_parser, required=False)
    status_parser.set_defaults(handler=command_backup_status)

    stop_parser = subparsers.add_parser("safe-stop-check")
    stop_parser.add_argument("--local-root", type=Path, required=True)
    _add_connection_arguments(stop_parser, required=False)
    stop_parser.set_defaults(handler=command_safe_stop)

    remote_inventory = subparsers.add_parser("backup-remote-inventory")
    remote_inventory.add_argument("--remote-root", type=Path, required=True)
    remote_inventory.add_argument("--repository-root", type=Path, required=True)
    remote_inventory.set_defaults(handler=command_remote_inventory)

    remote_package = subparsers.add_parser("backup-remote-package")
    remote_package.add_argument("--remote-root", type=Path, required=True)
    remote_package.add_argument("--repository-root", type=Path, required=True)
    remote_package.add_argument("--transfer-id", required=True)
    remote_package.set_defaults(handler=command_remote_package)

    remote_status = subparsers.add_parser("backup-remote-status")
    remote_status.add_argument("--remote-root", type=Path, required=True)
    remote_status.add_argument("--repository-root", type=Path, required=True)
    remote_status.set_defaults(handler=command_remote_status)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except BackupError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
