"""Strict, profile-aware bootstrap for disposable Quadra RunPod containers.

The command deliberately performs no scientific cohort work.  It restores
verified inputs into container-backed ``/workspace``, records provenance, and
prepares a disposable activation script.  Python 3.7 compatibility is required
for the released UAE-S image.
"""
from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import struct
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quadra import environment as persistent_env


SCHEMA_VERSION = 1
DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "configs/quadra/disposable-assets-v1.json"
DEFAULT_STORAGE_ROOT = Path("/workspace/quadra")
DEFAULT_REPOSITORY = Path("/workspace/repos/uae-quadra-validation")
DEFAULT_REPOSITORY_URL = "https://github.com/rifqi2708/self-supervised-anatomical-embedding-v2.git"
EXPECTED_IMAGES = {
    "uae": {
        "ref": "sunyu0410/uae:py37torch19",
        "digest": "sha256:2c0edd4a205c3c5d9d027b6c9f96f83626eb2cc3810da7876e32d4bf36653d61",
        "python": (3, 7),
    },
    "registration": {
        "ref": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "digest": "sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5",
        "python": (3, 11),
    },
}


class DisposableError(RuntimeError):
    """Raised when bootstrap cannot continue without weakening provenance."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_inventory(root):
    root = Path(root)
    records = []
    if root.is_symlink():
        raise DisposableError("Refusing to inventory a conflicting symlink: {}".format(root))
    if root.is_file():
        return {"files": 1, "bytes": root.stat().st_size,
                "entries": [{"path": root.name, "bytes": root.stat().st_size,
                             "sha256": sha256_file(root)}]}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink() or not is_within(path, root):
            raise DisposableError("Unsafe conflicting asset member: {}".format(path))
        records.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"files": len(records), "bytes": sum(item["bytes"] for item in records), "entries": records}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def is_within(path, root):
    resolved = str(Path(path).resolve())
    base = str(Path(root).resolve())
    return resolved == base or resolved.startswith(base + os.sep)


def load_catalog(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise DisposableError("Cannot read asset catalogue {}: {}".format(path, exc))
    if value.get("schema_version") != 1 or not isinstance(value.get("assets"), dict):
        raise DisposableError("Unsupported asset catalogue schema")
    required = ("whole_body_ct", "stage5_masks", "uae_models", "experiment_contract")
    for name in required:
        item = value["assets"].get(name)
        if not isinstance(item, dict):
            raise DisposableError("Catalogue is missing asset {}".format(name))
        for key in ("profiles", "filename", "archive_type", "payload_subpath", "promote_to"):
            if not item.get(key):
                raise DisposableError("Asset {} is missing {}".format(name, key))
        if item.get("bytes") is not None and int(item["bytes"]) <= 0:
            raise DisposableError("Asset {} has an invalid byte size".format(name))
        if item.get("sha256") is not None and not re.match(r"^[0-9a-f]{64}$", str(item["sha256"])):
            raise DisposableError("Asset {} has an invalid SHA-256".format(name))
        if item.get("archive_type") not in ("zip", "tar", "tar.gz"):
            raise DisposableError("Asset {} has an unsupported archive type".format(name))
        target = Path(item["promote_to"])
        if target.is_absolute() or ".." in target.parts:
            raise DisposableError("Unsafe promotion path for {}".format(name))
    return value


def required_assets(catalog, profile, require_ready=True):
    result = []
    for name, item in catalog["assets"].items():
        if profile not in item["profiles"]:
            continue
        if require_ready:
            missing = [key for key in ("drive_id", "bytes", "sha256") if not item.get(key)]
            if missing:
                raise DisposableError(
                    "Asset {} is not publish-ready; missing {}".format(name, ", ".join(missing))
                )
        result.append((name, item))
    return result


def paths_share_device(workspace, root):
    return workspace.stat().st_dev == root.stat().st_dev


def workspace_is_container_backed(workspace=Path("/workspace"), root=Path("/")):
    """Return True when /workspace is on the container/root filesystem."""
    workspace = Path(workspace)
    root = Path(root)
    if not workspace.is_dir() or not os.access(str(workspace), os.W_OK):
        raise DisposableError("Workspace is absent or not writable: {}".format(workspace))
    return paths_share_device(workspace, root)


def validate_archive_members(path, archive_type):
    names = []
    if archive_type == "zip":
        with zipfile.ZipFile(str(path), "r") as archive:
            infos = archive.infolist()
            if archive.testzip() is not None:
                raise DisposableError("ZIP integrity test failed: {}".format(path))
            for info in infos:
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise DisposableError("ZIP links are not accepted: {}".format(info.filename))
                names.append(info.filename)
    else:
        mode = "r:gz" if archive_type == "tar.gz" else "r:"
        with tarfile.open(str(path), mode) as archive:
            infos = archive.getmembers()
            for info in infos:
                if info.isdev() or info.isfifo():
                    raise DisposableError("Unsupported archive entry: {}".format(info.name))
                names.append(info.name)
                if info.issym() or info.islnk():
                    raise DisposableError("Archive links are not accepted: {}".format(info.name))
    for name in names:
        member = Path(name.replace("\\", "/"))
        if member.is_absolute() or ".." in member.parts:
            raise DisposableError("Archive path traversal rejected: {}".format(name))
    if not names:
        raise DisposableError("Archive is empty: {}".format(path))
    return names


def extract_archive(path, archive_type, destination):
    validate_archive_members(path, archive_type)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if archive_type == "zip":
        with zipfile.ZipFile(str(path), "r") as archive:
            archive.extractall(str(destination))
    else:
        mode = "r:gz" if archive_type == "tar.gz" else "r:"
        with tarfile.open(str(path), mode) as archive:
            archive.extractall(str(destination))


def verify_download(path, item):
    path = Path(path)
    if not path.is_file():
        raise DisposableError("Downloaded asset is missing: {}".format(path))
    observed_size = path.stat().st_size
    observed_hash = sha256_file(path)
    if observed_size != int(item["bytes"]):
        raise DisposableError("Size mismatch for {}: {} != {}".format(path.name, observed_size, item["bytes"]))
    if observed_hash != item["sha256"]:
        raise DisposableError("SHA-256 mismatch for {}".format(path.name))
    validate_archive_members(path, item["archive_type"])
    return {"bytes": observed_size, "sha256": observed_hash}


def validate_nifti_header(path):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(str(path), "rb") as handle:
            header = handle.read(348)
    except (OSError, EOFError) as exc:
        raise DisposableError("Unreadable NIfTI {}: {}".format(path, exc))
    if len(header) != 348:
        raise DisposableError("Truncated NIfTI header: {}".format(path))
    endian = "<" if struct.unpack("<I", header[:4])[0] == 348 else ">"
    if struct.unpack(endian + "I", header[:4])[0] != 348:
        raise DisposableError("Invalid NIfTI header: {}".format(path))
    dimensions = struct.unpack(endian + "8h", header[40:56])
    if dimensions[0] != 3 or any(value <= 0 for value in dimensions[1:4]):
        raise DisposableError("Expected a three-dimensional NIfTI: {}".format(path))
    return tuple(dimensions[1:4])


def validate_nifti_payload(path, binary=False):
    """Stream and validate the complete NIfTI payload, including gzip CRC."""
    import gzip
    import numpy as np

    dtypes = {
        2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8",
        256: "i1", 512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
    }
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), "rb") as handle:
        header = handle.read(348)
        if len(header) != 348:
            raise DisposableError("Truncated NIfTI header: {}".format(path))
        endian = "<" if struct.unpack("<I", header[:4])[0] == 348 else ">"
        dimensions = struct.unpack(endian + "8h", header[40:56])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        bitpix = struct.unpack(endian + "h", header[72:74])[0]
        vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
        slope = struct.unpack(endian + "f", header[112:116])[0]
        intercept = struct.unpack(endian + "f", header[116:120])[0]
        if datatype not in dtypes:
            raise DisposableError("Unsupported NIfTI datatype {}: {}".format(datatype, path))
        dtype = np.dtype(endian + dtypes[datatype])
        if dtype.itemsize * 8 != bitpix:
            raise DisposableError("NIfTI datatype/bitpix mismatch: {}".format(path))
        count = int(dimensions[1]) * int(dimensions[2]) * int(dimensions[3])
        handle.seek(max(vox_offset, 348))
        remaining = count * dtype.itemsize
        nonzero = 0
        while remaining:
            payload = handle.read(min(8 * 1024 * 1024, remaining))
            if not payload:
                raise DisposableError("Truncated NIfTI voxel payload: {}".format(path))
            if len(payload) % dtype.itemsize:
                raise DisposableError("Misaligned NIfTI voxel payload: {}".format(path))
            values = np.frombuffer(payload, dtype=dtype)
            if not np.isfinite(values).all():
                raise DisposableError("Non-finite NIfTI voxel values: {}".format(path))
            if binary:
                if np.any((values != 0) & (values != 1)):
                    raise DisposableError("Mask is not binary: {}".format(path))
                nonzero += int(np.count_nonzero(values))
            remaining -= len(payload)
        # Force gzip to read its trailer and validate its CRC.
        handle.read(1)
        if not binary and slope != 0.0:
            if not np.isfinite(slope) or not np.isfinite(intercept):
                raise DisposableError("Non-finite NIfTI scaling: {}".format(path))
        if binary and nonzero == 0:
            raise DisposableError("Mask is empty: {}".format(path))
    return {"voxels": count, "binary": bool(binary), "nonzero": nonzero if binary else None}


def nifti_geometry_signature(path):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), "rb") as handle:
        header = handle.read(348)
    if len(header) != 348:
        raise DisposableError("Truncated NIfTI header: {}".format(path))
    endian = "<" if struct.unpack("<I", header[:4])[0] == 348 else ">"
    dimensions = struct.unpack(endian + "8h", header[40:56])
    pixdim = struct.unpack(endian + "8f", header[76:108])
    qform_sform = struct.unpack(endian + "2h18f", header[252:328])
    return {
        "shape": tuple(dimensions[1:4]),
        "pixdim": tuple(round(value, 6) for value in pixdim[1:4]),
        "qform_sform": tuple(round(value, 6) if isinstance(value, float) else value for value in qform_sform),
        "xyzt_units": header[123],
    }


def validate_cross_asset_contract(ct_root, mask_root, contract_root):
    registry_path = Path(contract_root) / "metadata/mask_registry_and_cohort.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    scans = registry.get("scans", [])
    if len(scans) != 56:
        raise DisposableError("Experiment contract must describe 56 cohort scans")
    checked_masks = 0
    for scan in scans:
        subject = scan["subject_id"]
        number = subject.rsplit("_", 1)[-1]
        session = scan["session"]
        ct_path = Path(ct_root) / "QUADRA_HC_{}".format(number) / "{}_CT-AC.nii.gz".format(session)
        if not ct_path.is_file() or ct_path.stat().st_size != scan["input_size_bytes"]:
            raise DisposableError("Canonical CT identity mismatch: {}".format(ct_path))
        if sha256_file(ct_path) != scan["input_sha256"]:
            raise DisposableError("Canonical CT checksum mismatch: {}".format(ct_path))
        ct_geometry = nifti_geometry_signature(ct_path)
        scan_masks = Path(mask_root) / subject / session / "masks"
        for mask_name in scan["expected_masks"]:
            mask_path = scan_masks / (mask_name + ".nii.gz")
            if not mask_path.is_file():
                raise DisposableError("Expected cohort mask is missing: {}".format(mask_path))
            if nifti_geometry_signature(mask_path) != ct_geometry:
                raise DisposableError("CT-mask geometry mismatch: {}".format(mask_path))
            checked_masks += 1
    return {"ct_hashes_verified": len(scans), "ct_mask_geometries_verified": checked_masks,
            "cohort_subjects": len({scan["subject_id"] for scan in scans})}


def validate_extracted_asset(name, payload, item, extraction_root=None, full_payload=True, promoted=False):
    payload = Path(payload)
    expected = item.get("expected", {})
    evidence = {"payload": str(payload)}
    if name == "whole_body_ct":
        files = sorted(payload.rglob("*.nii.gz"))
        subjects = {path.parent.name.lower() for path in files}
        count_key = "promoted_ct_files" if promoted else "archive_ct_files"
        subject_key = "promoted_subject_directories" if promoted else "archive_subject_directories"
        if len(files) != expected.get(count_key) or len(subjects) != expected.get(subject_key):
            raise DisposableError("CT archive counts do not match the catalogue")
        for path in files:
            validate_nifti_header(path)
            if full_payload:
                validate_nifti_payload(path, binary=False)
        evidence.update(ct_files=len(files), subject_directories=len(subjects), nifti_headers_valid=len(files),
                        nifti_payloads_valid=len(files) if full_payload else None)
    elif name == "stage5_masks":
        final_masks = sorted(payload.glob("quadra_hc_*/*/masks/*.nii.gz"))
        intermediate = sorted(payload.glob("quadra_hc_*/*/intermediate/*.nii.gz"))
        subjects = {path.parts[-4] for path in final_masks}
        scans = {"{}/{}".format(path.parts[-4], path.parts[-3]) for path in final_masks}
        observed = (len(final_masks), len(intermediate), len(subjects), len(scans))
        wanted = (expected.get("final_masks"), expected.get("intermediate_masks"),
                  expected.get("subjects"), expected.get("scans"))
        if observed != wanted:
            raise DisposableError("Mask archive counts do not match the catalogue: {} != {}".format(observed, wanted))
        for path in final_masks + intermediate:
            validate_nifti_header(path)
            if full_payload:
                validate_nifti_payload(path, binary=True)
        checksum_path = Path(extraction_root or payload) / item.get("checksum_manifest", "")
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines() if checksum_path.is_file() else []
        if len(checksum_lines) != item.get("checksum_entries"):
            raise DisposableError("Stage 5 embedded checksum count mismatch")
        for line in checksum_lines:
            digest, relative = line.split("  ", 1)
            checked_path = Path(extraction_root) / relative
            if not checked_path.is_file() or sha256_file(checked_path) != digest:
                raise DisposableError("Stage 5 embedded checksum failed: {}".format(relative))
        evidence.update(final_masks=len(final_masks), intermediate_masks=len(intermediate),
                        subjects=len(subjects), scans=len(scans),
                        nifti_headers_valid=len(final_masks) + len(intermediate),
                        nifti_payloads_valid=(len(final_masks) + len(intermediate)) if full_payload else None,
                        embedded_checksums_verified=len(checksum_lines))
    elif name == "uae_models":
        observed = {}
        for filename, digest in item.get("expected_files", {}).items():
            path = payload / filename
            if not path.is_file() or sha256_file(path) != digest:
                raise DisposableError("Model checkpoint validation failed: {}".format(filename))
            observed[filename] = digest
        evidence["checkpoint_hashes"] = observed
    elif name == "experiment_contract":
        manifest = payload / "PACKAGE_MANIFEST.json"
        sums = payload / "SHA256SUMS"
        if not manifest.is_file() or not sums.is_file():
            raise DisposableError("Experiment contract metadata is missing")
        verified = 0
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            path = payload / relative
            if not path.is_file() or sha256_file(path) != digest:
                raise DisposableError("Experiment contract checksum failed: {}".format(relative))
            verified += 1
        package_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if package_manifest.get("frozen_queries") != expected.get("frozen_queries"):
            raise DisposableError("Experiment contract frozen-query count mismatch")
        evidence.update(inner_files_verified=verified,
                        frozen_queries=package_manifest.get("frozen_queries"),
                        organ_groups=len(package_manifest.get("organ_groups", [])))
    return evidence


def select_profile_payload(name, payload, item, staging):
    """Select the current 021-048 cohort without weakening archive validation."""
    if name != "whole_body_ct":
        return Path(payload)
    expected = item["expected"]
    selected = Path(staging) / "whole_body_ct-selected"
    if selected.exists():
        shutil.rmtree(str(selected))
    selected.mkdir(parents=True)
    first = int(expected["selected_subject_first"])
    last = int(expected["selected_subject_last"])
    for number in range(first, last + 1):
        source = Path(payload) / "QUADRA_HC_{:03d}".format(number)
        if not source.is_dir():
            raise DisposableError("Selected CT subject is missing: {}".format(source))
        shutil.copytree(str(source), str(selected / source.name), copy_function=os.link)
    validate_extracted_asset(name, selected, item, full_payload=False, promoted=True)
    return selected


def validate_profile(profile, image_ref=None, image_digest=None, gpu_memory_mib=None):
    expected = EXPECTED_IMAGES[profile]
    if sys.version_info[:2] != expected["python"]:
        raise DisposableError(
            "{} profile requires Python {}.{}".format(profile, *expected["python"])
        )
    if image_ref and image_ref != expected["ref"]:
        raise DisposableError("Container image tag does not match {} profile".format(profile))
    if image_digest and image_digest != expected["digest"]:
        raise DisposableError("Container image digest mismatch")
    if profile == "uae":
        if gpu_memory_mib is None:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                universal_newlines=True,
            ).strip().splitlines()
            gpu_memory_mib = int(output[0])
        if int(gpu_memory_mib) < 48 * 1024 - 512:
            raise DisposableError("UAE profile requires a GPU with at least 48 GB VRAM")
    return expected


def _run(command, cwd=None, env=None):
    subprocess.check_call([str(value) for value in command], cwd=str(cwd) if cwd else None, env=env)


def _git_output(repository, arguments):
    return subprocess.check_output(
        ["git", "-C", str(repository)] + list(arguments), universal_newlines=True
    ).strip()


def _git_output_optional(repository, arguments):
    try:
        return _git_output(repository, arguments)
    except (OSError, subprocess.CalledProcessError):
        return None


def clone_repository(repository, ref, url):
    repository = Path(repository)
    if repository.exists():
        if not (repository / ".git").exists():
            raise DisposableError("Repository destination exists but is not Git: {}".format(repository))
        if _git_output(repository, ["status", "--porcelain"]):
            raise DisposableError("Existing repository is dirty")
        try:
            resolved_ref = _git_output(repository, ["rev-parse", ref])
        except subprocess.CalledProcessError:
            raise DisposableError("Requested repository ref is absent: {}".format(ref))
        if _git_output(repository, ["rev-parse", "HEAD"]) != resolved_ref:
            raise DisposableError("Existing repository is not at requested ref {}".format(ref))
        if ref == "quadra-disposable-v1" and _git_output(repository, ["describe", "--tags", "--exact-match"]) != ref:
            raise DisposableError("Production bootstrap requires immutable tag {}".format(ref))
        return _git_output(repository, ["rev-parse", "HEAD"])
    repository.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--branch", ref, "--single-branch", url, repository])
    return _git_output(repository, ["rev-parse", "HEAD"])


def _install_gdown(staging):
    venv = Path(staging) / "gdown-venv"
    if not (venv / "bin/gdown").is_file():
        if not venv.exists():
            _run([sys.executable, "-m", "venv", str(venv)])
        _run([str(venv / "bin/python"), "-m", "pip", "install", "gdown==5.2.0"])
    return venv / "bin/gdown"


def _download(gdown, item, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            return verify_download(destination, item)
        except DisposableError:
            if destination.stat().st_size >= int(item["bytes"]):
                raise DisposableError(
                    "Existing staged download is complete-sized but invalid; inspect it before retrying: {}".format(
                        destination
                    )
                )
    _run([gdown, "--continue", "--id", item["drive_id"], "--output", destination])
    return verify_download(destination, item)


def _single_payload_root(extracted):
    extracted = Path(extracted)
    children = [item for item in extracted.iterdir() if item.name != "__MACOSX"]
    return children[0] if len(children) == 1 and children[0].is_dir() else extracted


def _promote(source, destination, storage_root):
    destination = Path(destination)
    if not is_within(destination.parent, storage_root):
        raise DisposableError("Promotion escaped storage root")
    if destination.exists() or destination.is_symlink():
        raise DisposableError("Refusing to overwrite existing asset: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(source), str(destination))


def _write_activation(root, repository, profile):
    activation = Path(root) / "runtime/activate.sh"
    content = """#!/usr/bin/env bash
profile=\"${1:-}\"
if [[ \"${profile}\" != \"{profile}\" ]]; then
  echo \"This disposable pod was bootstrapped for {profile}, not ${profile:-<unset>}.\" >&2
  return 2 2>/dev/null || exit 2
fi
export QUADRA_DISPOSABLE_PROFILE=\"{profile}\"
export QUADRA_STORAGE_ROOT=\"{root}\"
export QUADRA_REPO_ROOT=\"{repo}\"
export QUADRA_DATASET_ROOT=\"{root}/datasets/source/whole_body_ct_v1\"
export QUADRA_TOTALSEG_MASK_ROOT=\"{root}/datasets/derivatives/totalsegmentator_2.16.0_organs_v1\"
export QUADRA_MODEL_ROOT=\"{root}/models\"
export QUADRA_EXPERIMENT_CONTRACT=\"{root}/metadata/experiment-contract-v1\"
export QUADRA_OUTPUT_ROOT=\"{root}/runs\"
export PYTHONPATH=\"{repo}${{PYTHONPATH:+:${{PYTHONPATH}}}}\"
if [[ \"{profile}\" == \"registration\" ]]; then
  source \"{root}/runtime/registration-venv/bin/activate\" || return
fi
cd \"{repo}\" || return
python -m tools.quadra.disposable_pod status --profile \"{profile}\" --storage-root \"{root}\"
""".format(profile=profile, root=root, repo=repository)
    activation.parent.mkdir(parents=True, exist_ok=True)
    activation.write_text(content, encoding="utf-8")
    activation.chmod(0o755)


def _expose_repository_assets(root, repository, profile):
    """Expose immutable assets at legacy paths without duplicating large files."""
    links = []
    if profile == "uae":
        checkpoint_dir = Path(repository) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("SAM.pth", "SAMv2_iter_20000.pth"):
            source = Path(root) / "models/uae/base" / filename
            destination = checkpoint_dir / filename
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise DisposableError("Conflicting checkpoint link: {}".format(destination))
            elif destination.exists():
                if not destination.is_file() or sha256_file(destination) != sha256_file(source):
                    raise DisposableError("Conflicting checkpoint file: {}".format(destination))
            else:
                destination.symlink_to(source)
            links.append({"path": str(destination), "target": str(source)})
    return links


def _fingerprint(profile, root, repository, image_ref, image_digest, assets):
    memory_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    disk = shutil.disk_usage("/workspace")
    result = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": utc_now(),
        "disposable_profile": profile,
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "image_ref": image_ref,
        "image_digest": image_digest,
        "image_identity_verification": "operator-confirmed exact digest at bootstrap",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "workspace_disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "repository_path": str(repository),
        "repository_commit": _git_output(repository, ["rev-parse", "HEAD"]),
        "repository_tag": _git_output_optional(repository, ["describe", "--tags", "--exact-match"]),
        "container_backed_workspace": workspace_is_container_backed(),
        "assets": assets,
    }
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            universal_newlines=True,
        ).strip()
    except Exception as exc:
        result["nvidia_smi"] = "unavailable: {}".format(exc)
    return result


def _write_registration_scientific_profile(root, repository):
    """Prepare the marker consumed by existing registration cohort commands."""
    from tools.quadra import registration_runtime
    venv_python = Path(root) / "runtime/registration-venv/bin/python"
    code = (
        "import json; from tools.quadra.registration_runtime import fingerprint; "
        "print(json.dumps(fingerprint({!r})))"
    ).format(str(root))
    fingerprint = json.loads(
        subprocess.check_output([str(venv_python), "-c", code], cwd=str(repository), universal_newlines=True)
    )
    requirements = Path(repository) / "tools/quadra/environment/requirements-registration.txt"
    marker = {
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "requirements_sha256": sha256_file(requirements),
        "storage_policy": "disposable_container_disk",
    }
    atomic_json(Path(root) / "runtime/profiles/registration.json", marker)
    return marker


def _run_uae_smoke(output, config, checkpoint):
    """Run one bounded, dense, non-tiled UAE-S extraction and global match."""
    import numpy as np
    import torch
    import torch.nn.functional as torch_f
    from tools.quadra.memory_configuration_screen import _load_model

    if not torch.cuda.is_available():
        raise DisposableError("CUDA is unavailable for the UAE smoke test")
    torch.manual_seed(20260913)
    np.random.seed(20260913)
    started = time.time()
    model, training_hook_present = _load_model(Path(config), Path(checkpoint), "fp32")
    values = torch.linspace(-50.0, 50.0, 32 * 64 * 64, dtype=torch.float32)
    tensor = values.reshape(1, 1, 32, 64, 64).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model.extract_feat(tensor)
        if len(outputs) != 3:
            raise DisposableError("UAE-S smoke expected fine, coarse and semantic outputs")
        fine, coarse, semantic = [value.float() for value in outputs]
        expected = ([1, 128, 16, 32, 32], [1, 128, 8, 4, 4], [1, 128, 16, 32, 32])
        shapes = tuple([int(value) for value in feature.shape] for feature in (fine, coarse, semantic))
        if shapes != expected:
            raise DisposableError("Unexpected UAE-S smoke feature shapes: {}".format(shapes))
        coarse_up = torch_f.interpolate(
            coarse, size=fine.shape[2:], mode="trilinear", align_corners=True
        )
        centre = tuple(int(value // 2) for value in fine.shape[2:])
        coarse_centre = tuple(int(value // 2) for value in coarse.shape[2:])
        query_fine = fine[(0, slice(None)) + centre]
        query_semantic = semantic[(0, slice(None)) + centre]
        query_coarse = coarse[(0, slice(None)) + coarse_centre]
        scores = (
            torch.sum(fine[0] * query_fine[:, None, None, None], dim=0)
            + torch.sum(coarse_up[0] * query_coarse[:, None, None, None], dim=0)
            + torch.sum(semantic[0] * query_semantic[:, None, None, None], dim=0)
        ) / 3.0
        if not torch.isfinite(scores).all():
            raise DisposableError("UAE-S smoke similarity contains non-finite values")
        flat_index = int(torch.argmax(scores.reshape(-1)).item())
        z_size, y_size, x_size = [int(value) for value in scores.shape]
        point_xyz = [flat_index % x_size, (flat_index // x_size) % y_size,
                     flat_index // (x_size * y_size)]
        score_bytes = scores.detach().cpu().numpy().astype(np.float32).tobytes()
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "profile": "uae",
            "scientific_work_launched": False,
            "method": "bounded_dense_non_tiled_three_feature_global_match",
            "input_shape_ncdhw": [1, 1, 32, 64, 64],
            "feature_shapes": [list(value) for value in shapes],
            "global_argmax_xyz": point_xyz,
            "global_max_score": float(torch.max(scores).item()),
            "score_volume_sha256": hashlib.sha256(score_bytes).hexdigest(),
            "training_fp16_hook_present": bool(training_hook_present),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "seconds": float(time.time() - started),
        }
    del model, tensor, outputs, fine, coarse, semantic, coarse_up, scores
    torch.cuda.empty_cache()
    atomic_json(output, result)
    return result


def _run_registration_smoke(output):
    """Run tiny forward/reverse Elastix and continuous Transformix checks."""
    import numpy as np
    import itk
    from tools.quadra import registration_point_transform as points

    started = time.time()
    z, y, x = np.indices((24, 24, 24), dtype=np.float32)
    data = np.exp(-((x - 11) ** 2 + (y - 12) ** 2 + (z - 10) ** 2) / 20.0).astype(np.float32)
    fixed = itk.image_from_array(data)
    moving = itk.image_from_array(data.copy())
    for image in (fixed, moving):
        image.SetSpacing([1.2, 1.3, 1.4])
        image.SetOrigin([10.0, 20.0, 30.0])
    maps = points.parameter_maps()
    for item in maps:
        item.update(NumberOfResolutions=["1"], MaximumNumberOfIterations=["0"],
                    NumberOfSpatialSamples=["128"], FinalGridSpacingInPhysicalUnits=["8"])
        if "GridSpacingSchedule" in item:
            item["GridSpacingSchedule"] = ["1"]
    transforms = []
    with tempfile.TemporaryDirectory(prefix="quadra-registration-smoke-") as temporary:
        for direction, pair in (("forward", (fixed, moving)), ("reverse", (moving, fixed))):
            directory = Path(temporary) / direction
            directory.mkdir()
            method = itk.ElastixRegistrationMethod.New(pair[0], pair[1])
            method.SetParameterObject(points.parameter_object(maps))
            method.SetOutputDirectory(str(directory))
            method.SetLogToConsole(False)
            method.SetNumberOfThreads(1)
            method.UpdateLargestPossibleRegion()
            transforms.append(points.normalized_transform_maps(method.GetTransformParameterObject()))
        query = np.asarray([[16.25, 27.1, 39.37], [19.1, 32.0, 42.0]], dtype=np.float64)
        forward = points.transformix_points(query, transforms[0])
        returned = points.transformix_points(forward, transforms[1])
    cycle_error = np.linalg.norm(returned - query, axis=1)
    maximum = float(np.max(cycle_error))
    if not np.isfinite(cycle_error).all() or maximum > 1e-3:
        raise DisposableError("Registration smoke cycle error exceeded tolerance: {}".format(maximum))
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "profile": "registration",
        "scientific_work_launched": False,
        "method": "bounded_identical_volume_forward_reverse_elastix_transformix",
        "image_shape_zyx": [24, 24, 24],
        "query_points": len(query),
        "maximum_cycle_error_mm": maximum,
        "transform_families": [item["Transform"][0] for item in transforms[0]],
        "seconds": float(time.time() - started),
    }
    atomic_json(output, result)
    return result


def command_internal_smoke(args):
    result = (
        _run_uae_smoke(args.output, args.config, args.checkpoint)
        if args.profile == "uae"
        else _run_registration_smoke(args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_plan(args):
    catalog = load_catalog(args.asset_catalog)
    assets = required_assets(catalog, args.profile, require_ready=False)
    result = {
        "profile": args.profile,
        "template": EXPECTED_IMAGES[args.profile],
        "assets": [{"name": name, "ready": all(item.get(k) for k in ("drive_id", "bytes", "sha256")),
                    "filename": item["filename"], "bytes": item.get("bytes")} for name, item in assets],
        "scientific_work_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(item["ready"] for item in result["assets"]) else 2


def command_bootstrap(args):
    root = persistent_env.validate_storage_root(args.storage_root)
    if not workspace_is_container_backed():
        raise DisposableError("/workspace is a separate pod/network volume; disposable bootstrap requires container storage")
    usage = shutil.disk_usage(str(Path("/workspace")))
    if usage.free < args.minimum_free_gib * 1024 ** 3:
        raise DisposableError("Insufficient free container storage")
    if not args.image_ref or not args.confirm_image_digest:
        raise DisposableError(
            "Container identity is required through --image-ref/--confirm-image-digest "
            "or QUADRA_IMAGE_REF/QUADRA_IMAGE_DIGEST"
        )
    expected = validate_profile(args.profile, args.image_ref, args.confirm_image_digest)
    catalog = load_catalog(args.asset_catalog)
    required = required_assets(catalog, args.profile, require_ready=True)
    repository = Path(args.repository_root)
    commit = clone_repository(repository, args.repository_ref, args.repository_url)
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("datasets", "models", "runs", "metadata/manifests", "staging", "runtime"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    existing_manifest = root / "metadata/manifests/disposable-{}-environment.json".format(args.profile)
    if existing_manifest.is_file():
        status_args = argparse.Namespace(profile=args.profile, storage_root=root, asset_catalog=args.asset_catalog)
        result = command_status(status_args)
        print("Existing verified disposable profile reused; no download or installation was performed.")
        return result
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = root / "staging" / ("disposable-bootstrap-" + args.profile)
    staging.mkdir(parents=True, exist_ok=True)
    stage_identity = staging / "bootstrap-request.json"
    request_identity = {
        "profile": args.profile,
        "catalog_sha256": sha256_file(args.asset_catalog),
        "repository_commit": commit,
        "image_ref": expected["ref"],
        "image_digest": expected["digest"],
    }
    if stage_identity.is_file():
        with stage_identity.open("r", encoding="utf-8") as handle:
            if json.load(handle) != request_identity:
                raise DisposableError(
                    "Existing resumable staging belongs to a different bootstrap request: {}".format(staging)
                )
    else:
        atomic_json(stage_identity, request_identity)
    gdown = _install_gdown(staging)
    restored = {}
    staged_payloads = []
    for name, item in required:
        archive = staging / item["filename"]
        observed = _download(gdown, item, archive)
        extracted = staging / (name + "-extracted")
        extracted_marker = extracted / ".disposable-extraction.json"
        expected_extraction = {"archive_sha256": item["sha256"], "asset": name}
        if extracted_marker.is_file():
            with extracted_marker.open("r", encoding="utf-8") as handle:
                if json.load(handle) != expected_extraction:
                    raise DisposableError("Staged extraction identity mismatch: {}".format(extracted))
        else:
            if extracted.exists():
                shutil.rmtree(str(extracted))
            extract_archive(archive, item["archive_type"], extracted)
            atomic_json(extracted_marker, expected_extraction)
        payload = extracted / item["payload_subpath"]
        if not payload.is_dir() or not is_within(payload, extracted):
            raise DisposableError("Expected payload is missing for {}: {}".format(name, item["payload_subpath"]))
        observed["content_validation"] = validate_extracted_asset(name, payload, item, extraction_root=extracted)
        payload = select_profile_payload(name, payload, item, staging)
        if name == "whole_body_ct":
            observed["selection"] = {
                "subjects": "{:03d}-{:03d}".format(
                    item["expected"]["selected_subject_first"], item["expected"]["selected_subject_last"]
                ),
                "ct_files": item["expected"]["promoted_ct_files"],
            }
        destination = root / item["promote_to"]
        staged_payloads.append((name, item, archive, payload, destination, observed))
    staged_by_name = {name: payload for name, item, archive, payload, destination, observed in staged_payloads}
    cross_validation = validate_cross_asset_contract(
        staged_by_name["whole_body_ct"], staged_by_name["stage5_masks"], staged_by_name["experiment_contract"]
    )
    quarantine_root = root / "runs/archive" / ("pre-bootstrap-conflict-" + stamp)
    quarantined = []
    for name, item, archive, payload, destination, observed in staged_payloads:
        if destination.exists() or destination.is_symlink():
            before = tree_inventory(destination)
            quarantine = quarantine_root / name
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            if quarantine.exists():
                raise DisposableError("Bootstrap quarantine destination already exists: {}".format(quarantine))
            os.replace(str(destination), str(quarantine))
            quarantined.append({"asset": name, "source": str(destination), "quarantine": str(quarantine),
                                "pre_quarantine_inventory": before})
    for name, item, archive, payload, destination, observed in staged_payloads:
        _promote(payload, destination, root)
        archive.unlink()
        restored[name] = dict(observed, destination=str(destination), drive_id=item["drive_id"])
    if args.profile == "registration":
        venv = root / "runtime/registration-venv"
        _run([sys.executable, "-m", "venv", str(venv)])
        req = repository / "tools/quadra/environment/requirements-registration.txt"
        _run([venv / "bin/python", "-m", "pip", "install", "--only-binary=:all:", "-r", req])
        _run([venv / "bin/python", "-m", "pip", "check"])
        _write_registration_scientific_profile(root, repository)
    exposed_assets = _expose_repository_assets(root, repository, args.profile)
    _write_activation(root, repository, args.profile)
    smoke_directory = root / ("runs/uae" if args.profile == "uae" else "runs/preprocessing") / (
        "disposable-{}-bootstrap-smoke-{}".format(args.profile, stamp)
    )
    smoke_directory.mkdir(parents=True, exist_ok=False)
    smoke_path = smoke_directory / "smoke.json"
    smoke_python = (
        sys.executable if args.profile == "uae"
        else str(root / "runtime/registration-venv/bin/python")
    )
    smoke_command = [
        smoke_python, "-m", "tools.quadra.disposable_pod", "_smoke",
        "--profile", args.profile, "--output", str(smoke_path),
    ]
    if args.profile == "uae":
        smoke_command.extend([
            "--config", str(repository / "configs/samv2/samv2_NIHLN.py"),
            "--checkpoint", str(repository / "checkpoints/SAMv2_iter_20000.pth"),
        ])
    _run(smoke_command, cwd=repository)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS":
        raise DisposableError("Disposable bounded smoke test failed")
    fingerprint = _fingerprint(args.profile, root, repository, expected["ref"], expected["digest"], restored)
    fingerprint["repository_commit"] = commit
    fingerprint["quarantined_conflicts"] = quarantined
    fingerprint["cross_asset_validation"] = cross_validation
    fingerprint["repository_asset_links"] = exposed_assets
    fingerprint["bounded_smoke"] = {
        "path": str(smoke_path), "bytes": smoke_path.stat().st_size,
        "sha256": sha256_file(smoke_path), "status": smoke["status"],
    }
    final_free = shutil.disk_usage("/workspace").free
    if final_free < args.minimum_final_free_gib * 1024 ** 3:
        raise DisposableError(
            "Disposable bootstrap left less than {} GiB free".format(args.minimum_final_free_gib)
        )
    atomic_json(root / "metadata/manifests/disposable-{}-environment.json".format(args.profile), fingerprint)
    print("Disposable {} profile ready. No scientific cohort was launched.".format(args.profile))
    print("source {}/runtime/activate.sh {}".format(root, args.profile))
    return 0


def command_status(args):
    root = persistent_env.validate_storage_root(args.storage_root)
    manifest = root / "metadata/manifests/disposable-{}-environment.json".format(args.profile)
    if not manifest.is_file():
        raise DisposableError("Disposable profile manifest is missing")
    with manifest.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    validate_profile(args.profile, saved.get("image_ref"), saved.get("image_digest"))
    if os.environ.get("QUADRA_DISPOSABLE_PROFILE") not in (None, args.profile):
        raise DisposableError("Active disposable profile does not match requested profile")
    missing = [name for name, item in saved.get("assets", {}).items() if not Path(item["destination"]).exists()]
    if missing:
        raise DisposableError("Restored assets are missing: {}".format(", ".join(missing)))
    for item in saved.get("repository_asset_links", []):
        link = Path(item["path"])
        target = Path(item["target"])
        if not link.exists() or link.resolve() != target.resolve():
            raise DisposableError("Repository asset link changed or is missing: {}".format(link))
    repository = Path(saved.get("repository_path", DEFAULT_REPOSITORY))
    if not (repository / ".git").exists():
        raise DisposableError("Disposable repository is missing: {}".format(repository))
    if _git_output(repository, ["rev-parse", "HEAD"]) != saved.get("repository_commit"):
        raise DisposableError("Disposable repository commit changed")
    if _git_output(repository, ["status", "--porcelain"]):
        raise DisposableError("Disposable repository is dirty")
    smoke = saved.get("bounded_smoke", {})
    smoke_path = Path(smoke.get("path", ""))
    if (
        smoke.get("status") != "PASS"
        or not smoke_path.is_file()
        or smoke_path.stat().st_size != smoke.get("bytes")
        or sha256_file(smoke_path) != smoke.get("sha256")
    ):
        raise DisposableError("Bounded disposable smoke evidence is missing or changed")
    config = repository / "configs/samv2/samv2_NIHLN.py"
    if args.profile == "uae" and not config.is_file():
        raise DisposableError("UAE-S configuration is missing: {}".format(config))
    catalog = load_catalog(args.asset_catalog)
    catalogue_assets = dict(required_assets(catalog, args.profile, require_ready=True))
    validations = {}
    for name, saved_item in saved.get("assets", {}).items():
        if name not in catalogue_assets:
            raise DisposableError("Saved asset is absent from the current catalogue: {}".format(name))
        # Embedded Stage 5 archive checksums were verified at bootstrap and are
        # recorded in the immutable environment manifest. Fast status rechecks
        # counts and NIfTI headers without requiring the deleted archive logs.
        if name == "stage5_masks":
            item = dict(catalogue_assets[name])
            item.pop("checksum_manifest", None)
            item["checksum_entries"] = 0
        else:
            item = catalogue_assets[name]
        validations[name] = validate_extracted_asset(
            name, Path(saved_item["destination"]), item, full_payload=False,
            promoted=(name == "whole_body_ct")
        )
    result = {"status": "PASS", "profile": args.profile, "manifest": str(manifest),
              "assets": sorted(saved.get("assets", {})), "asset_validation": validations,
              "scientific_work_launched": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_package_results(args):
    from tools.quadra import artifact_backup
    root = persistent_env.validate_storage_root(args.storage_root).resolve()
    run = Path(args.run_directory).resolve()
    if not is_within(run, root / "runs") or not run.is_dir():
        raise DisposableError("Run directory must be an existing child of {}/runs".format(root))
    runtime = artifact_backup.process_status(args.repository_root)
    if runtime["active_processes"]:
        raise DisposableError("Cannot snapshot results while scientific or backup processes are active")
    packages = root / "staging/result-packages"
    package = packages / (args.transfer_id + ".tar.gz")
    package_receipt = packages / (args.transfer_id + ".json")
    snapshot = packages / ("." + args.transfer_id + "-snapshot")
    if package.exists() or package_receipt.exists():
        raise DisposableError("Result package transfer ID already exists: {}".format(args.transfer_id))
    if snapshot.exists():
        raise DisposableError("Interrupted snapshot exists and requires inspection: {}".format(snapshot))
    relative = run.relative_to(root)
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(run), str(target), copy_function=os.link)
    try:
        inventory = artifact_backup.build_inventory(snapshot, repository=args.repository_root, source_id=args.transfer_id)
        receipt = artifact_backup.create_package(snapshot, package, inventory)
    finally:
        shutil.rmtree(str(snapshot))
    receipt["run_directory"] = str(run)
    receipt["transfer_id"] = args.transfer_id
    atomic_json(package_receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", choices=("uae", "registration"), required=True)
    common.add_argument("--asset-catalog", type=Path, default=DEFAULT_CATALOG)
    plan = sub.add_parser("plan", parents=[common])
    plan.set_defaults(handler=command_plan)
    boot = sub.add_parser("bootstrap", parents=[common])
    boot.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    boot.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY)
    boot.add_argument("--repository-url", default=DEFAULT_REPOSITORY_URL)
    boot.add_argument("--repository-ref", default="quadra-disposable-v1")
    boot.add_argument("--image-ref", default=os.environ.get("QUADRA_IMAGE_REF"))
    boot.add_argument("--confirm-image-digest", default=os.environ.get("QUADRA_IMAGE_DIGEST"))
    boot.add_argument("--minimum-free-gib", type=int, default=65)
    boot.add_argument("--minimum-final-free-gib", type=int, default=50)
    boot.set_defaults(handler=command_bootstrap)
    status = sub.add_parser("status", parents=[common])
    status.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    status.set_defaults(handler=command_status)
    package = sub.add_parser("package-results")
    package.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    package.add_argument("--run-directory", type=Path, required=True)
    package.add_argument("--transfer-id", required=True)
    package.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY)
    package.set_defaults(handler=command_package_results)
    smoke = sub.add_parser("_smoke")
    smoke.add_argument("--profile", choices=("uae", "registration"), required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--config", type=Path)
    smoke.add_argument("--checkpoint", type=Path)
    smoke.set_defaults(handler=command_internal_smoke)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (DisposableError, persistent_env.EnvironmentError, subprocess.CalledProcessError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
