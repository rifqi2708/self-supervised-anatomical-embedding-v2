"""Persistent RunPod environment management for Quadra research.

This module intentionally supports Python 3.7 because the released UAE image
uses that interpreter.  It never upgrades the UAE environment.
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


SCHEMA_VERSION = 1
DEFAULT_STORAGE_ROOT = Path("/workspace/quadra")
DEFAULT_LEGACY_TOTALSEG_ROOT = Path("/workspace/quadra-totalsegmentator")
DEFAULT_BACKUP_ROOT = Path("/workspace/quadra_backup_20260722")
DEFAULT_REPOSITORY_NAME = "uae-quadra-validation"
SUPERPOINT_REPOSITORY = "https://github.com/rpautrat/SuperPoint.git"
SUPERPOINT_COMMIT = "1411bbd68c50163555d39c1b26e9e046ebd48f27"
SUPERPOINT_CHECKPOINT_SHA256 = (
    "cd5d19a5061848e248c17728878ea166b66512076d43c77dbcf27f4a88a56084"
)
PROFILE_IMAGES = {
    "preprocess": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "uae": "sunyu0410/uae:py37torch19",
}
PROFILE_EXPECTED_DIGESTS = {
    "preprocess": "sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5",
    "uae": "sha256:2c0edd4a205c3c5d9d027b6c9f96f83626eb2cc3810da7876e32d4bf36653d61",
}


class EnvironmentError(RuntimeError):
    """Raised when a persistent environment cannot be prepared safely."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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


def validate_storage_root(root):
    root = Path(root).expanduser()
    if not root.is_absolute():
        raise EnvironmentError("Storage root must be absolute: {}".format(root))
    if str(root) in {"/", "/workspace"}:
        raise EnvironmentError("Refusing unsafe broad storage root: {}".format(root))
    return root


def canonical_layout(root):
    root = validate_storage_root(root)
    return {
        "storage_root": root,
        "datasets": root / "datasets",
        "whole_body_ct": root / "datasets/source/whole_body_ct_v1",
        "totalsegmentator_outputs": (
            root / "datasets/derivatives/totalsegmentator_2.16.0_organs_v1"
        ),
        "cropped_dataset": root / "datasets/derivatives/cropped_v1",
        "fine_tune_dataset": root / "datasets/fine_tune_v1",
        "uae_models": root / "models/uae/base",
        "uae_fine_tuned_models": root / "models/uae/fine_tuned",
        "superpoint_models": root / "models/superpoint",
        "runs_preprocessing": root / "runs/preprocessing",
        "runs_uae": root / "runs/uae",
        "runs_archive": root / "runs/archive",
        "manifests": root / "metadata/manifests",
        "totalsegmentator_cache": root / "cache/totalsegmentator",
        "uae_cache": root / "cache/uae",
        "runtime": root / "runtime",
        "preprocess_venv": root / "runtime/preprocess-venv",
        "vscode_data": root / "runtime/vscode-cli-data",
        "runtime_bin": root / "runtime/bin",
        "profiles": root / "runtime/profiles",
        "superpoint_repository": root / "vendor/superpoint",
        "staging": root / "staging",
    }


def resolve_quadra_path(explicit, environment_name, repository_fallback):
    """Resolve an active path: explicit CLI, persistent root, local fallback."""
    if explicit:
        return Path(explicit).expanduser()
    storage_root = os.environ.get("QUADRA_STORAGE_ROOT")
    if storage_root:
        return canonical_layout(Path(storage_root))[environment_name]
    return Path(repository_fallback)


def _ensure_directory(path, root):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if not path.exists():
            raise EnvironmentError("Broken persistent link: {}".format(path))
        return
    if path.exists() and not path.is_dir():
        raise EnvironmentError("Expected directory but found another type: {}".format(path))
    path.mkdir(parents=True, exist_ok=True)
    if not _is_within(path, root):
        raise EnvironmentError("Created path escaped storage root: {}".format(path))


def ensure_link(link_path, target_path, root):
    """Create an idempotent link inside root to a verified existing target."""
    link_path = Path(link_path)
    target_path = Path(target_path)
    root = Path(root)
    if not target_path.exists():
        raise EnvironmentError("Link target does not exist: {}".format(target_path))
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if not _is_within(link_path.parent, root):
        raise EnvironmentError("Link parent escapes storage root: {}".format(link_path))
    if link_path.is_symlink():
        if not link_path.exists():
            raise EnvironmentError("Refusing to replace broken link: {}".format(link_path))
        if link_path.resolve() != target_path.resolve():
            raise EnvironmentError(
                "Link points to conflicting target: {} -> {}".format(
                    link_path, link_path.resolve()
                )
            )
        return "existing"
    if link_path.exists():
        raise EnvironmentError("Refusing to replace existing path: {}".format(link_path))
    link_path.symlink_to(target_path.resolve(), target_is_directory=target_path.is_dir())
    return "created"


def _command_output(command):
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, universal_newlines=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return "unavailable: {}".format(exc)


def _module_version(name):
    try:
        module = __import__(name)
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
        try:
            from importlib import metadata

            return str(metadata.version(name))
        except (ImportError, AttributeError):
            import pkg_resources

            return str(pkg_resources.get_distribution(name).version)
    except Exception as exc:
        return "unavailable: {}".format(exc)


def runtime_fingerprint(profile, image_ref, include_packages=True):
    torch_version = _module_version("torch")
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": utc_now(),
        "profile": profile,
        "image_ref": image_ref,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "mmcv": _module_version("mmcv"),
        "mmdet": _module_version("mmdet"),
        "totalsegmentator": _module_version("totalsegmentator"),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    if include_packages:
        fingerprint["pip_freeze"] = _command_output(
            [sys.executable, "-m", "pip", "freeze", "--all"]
        ).splitlines()
    try:
        import torch

        fingerprint["cuda_available"] = bool(torch.cuda.is_available())
        fingerprint["torch_cuda"] = str(torch.version.cuda)
        fingerprint["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:
        fingerprint["cuda_available"] = False
        fingerprint["torch_error"] = str(exc)
    return fingerprint


def resolve_docker_digest(image_ref):
    """Resolve a public Docker Hub tag to its registry manifest digest."""
    if "@" in image_ref:
        return image_ref.split("@", 1)[1]
    last_component = image_ref.rsplit("/", 1)[-1]
    if ":" in last_component:
        repository, tag = image_ref.rsplit(":", 1)
    else:
        repository, tag = image_ref, "latest"
    if "/" not in repository:
        repository = "library/" + repository
    query = urllib.parse.urlencode(
        {
            "service": "registry.docker.io",
            "scope": "repository:{}:pull".format(repository),
        }
    )
    token_url = "https://auth.docker.io/token?" + query
    try:
        with urllib.request.urlopen(token_url, timeout=30) as response:
            token = json.loads(response.read().decode("utf-8"))["token"]
    except Exception:
        if shutil.which("curl") is None:
            raise
        token_payload = subprocess.check_output(
            ["curl", "--fail", "--silent", "--show-error", token_url],
            universal_newlines=True,
        )
        token = json.loads(token_payload)["token"]
    manifest_url = "https://registry-1.docker.io/v2/{}/manifests/{}".format(
        repository, tag
    )
    accept = ", ".join(
        [
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        ]
    )
    request = urllib.request.Request(
        manifest_url,
        method="HEAD",
        headers={
            "Authorization": "Bearer " + token,
            "Accept": accept,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            digest = response.headers.get("Docker-Content-Digest")
    except Exception:
        if shutil.which("curl") is None:
            raise
        headers = subprocess.check_output(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--head",
                "--header",
                "Authorization: Bearer " + token,
                "--header",
                "Accept: " + accept,
                manifest_url,
            ],
            universal_newlines=True,
        )
        digest = None
        for line in headers.splitlines():
            if line.lower().startswith("docker-content-digest:"):
                digest = line.split(":", 1)[1].strip()
                break
    if not digest:
        raise EnvironmentError("Registry did not return a digest for " + image_ref)
    return digest


def _git_output(repository, arguments):
    return subprocess.check_output(
        ["git", "-C", str(repository)] + list(arguments),
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    ).strip()


def _git_current_branch(repository):
    return _git_output(repository, ["symbolic-ref", "--quiet", "--short", "HEAD"])


def ensure_persistent_repository(source_repository, storage_root):
    source_repository = Path(source_repository).resolve()
    target = Path(storage_root).parent / "repos" / DEFAULT_REPOSITORY_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    source_head = _git_output(source_repository, ["rev-parse", "HEAD"])
    source_branch = _git_current_branch(source_repository)
    if not target.exists():
        subprocess.check_call(
            [
                "git",
                "clone",
                "--no-hardlinks",
                "--branch",
                source_branch,
                str(source_repository),
                str(target),
            ]
        )
    else:
        if not (target / ".git").is_dir():
            raise EnvironmentError(
                "Persistent repository target is not a Git checkout: {}".format(target)
            )
        if _git_output(target, ["status", "--porcelain"]):
            raise EnvironmentError(
                "Persistent repository has uncommitted changes: {}".format(target)
            )
        target_branch = _git_current_branch(target)
        if target_branch != source_branch:
            raise EnvironmentError(
                "Persistent repository is on {}, expected {}".format(
                    target_branch, source_branch
                )
            )
        subprocess.check_call(
            ["git", "-C", str(target), "fetch", str(source_repository), source_branch]
        )
        subprocess.check_call(
            ["git", "-C", str(target), "merge", "--ff-only", "FETCH_HEAD"]
        )
    target_head = _git_output(target, ["rev-parse", "HEAD"])
    if target_head != source_head:
        raise EnvironmentError("Persistent repository did not reach the source commit")
    return target, source_branch, source_head


def _copy_verified_asset(source, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        return None
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise EnvironmentError("Conflicting persistent asset: {}".format(destination))
        return {"path": str(destination), "sha256": source_hash, "status": "existing"}
    staging = destination.parent / ("." + destination.name + ".staging")
    if staging.exists():
        staging.unlink()
    shutil.copy2(str(source), str(staging))
    if sha256_file(staging) != source_hash:
        staging.unlink()
        raise EnvironmentError("Copied asset checksum mismatch: {}".format(destination))
    os.replace(str(staging), str(destination))
    return {"path": str(destination), "sha256": source_hash, "status": "copied"}


def _tree_inventory(root):
    root = Path(root)
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def _copy_tree_once(source, destination):
    """Copy a writable cache without ever mutating its legacy source."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        return {"status": "empty-created", "path": str(destination)}
    source_inventory = _tree_inventory(source)
    if destination.exists():
        if not destination.is_dir():
            raise EnvironmentError(
                "Writable cache destination is not a directory: {}".format(destination)
            )
        destination_inventory = _tree_inventory(destination)
        if destination_inventory != source_inventory:
            raise EnvironmentError(
                "Existing cache differs from legacy source: {}".format(destination)
            )
        return {
            "status": "existing",
            "path": str(destination),
            "inventory": destination_inventory,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ("." + destination.name + ".staging")
    if staging.exists():
        raise EnvironmentError("Stale cache staging directory: {}".format(staging))
    shutil.copytree(str(source), str(staging), symlinks=True)
    if _tree_inventory(staging) != source_inventory:
        shutil.rmtree(str(staging))
        raise EnvironmentError("Copied cache inventory mismatch")
    os.replace(str(staging), str(destination))
    return {
        "status": "copied",
        "path": str(destination),
        "inventory": source_inventory,
    }


def _ensure_repository_asset_link(repository_path, relative_path, target, storage_root):
    repository_path = Path(repository_path)
    link = repository_path / relative_path
    if link.exists() and not link.is_symlink():
        if link.is_file() and sha256_file(link) == sha256_file(target):
            return "existing-local"
        raise EnvironmentError("Repository asset conflicts with persistent model: {}".format(link))
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if not link.exists() or link.resolve() != Path(target).resolve():
            raise EnvironmentError("Repository model link is invalid: {}".format(link))
        return "existing-link"
    link.symlink_to(Path(target).resolve())
    return "created-link"


def prepare_uae_assets(source_repository, persistent_repository, layout):
    assets = []
    for filename in ("SAM.pth", "SAMv2_iter_20000.pth"):
        destination = layout["uae_models"] / filename
        source_candidates = [
            Path(source_repository) / "checkpoints" / filename,
            Path(persistent_repository) / "checkpoints" / filename,
        ]
        result = None
        for candidate in source_candidates:
            result = _copy_verified_asset(candidate, destination)
            if result:
                break
        if result:
            result["repository_link"] = _ensure_repository_asset_link(
                persistent_repository,
                Path("checkpoints") / filename,
                destination,
                layout["storage_root"],
            )
            assets.append(result)
        else:
            assets.append(
                {"path": str(destination), "status": "missing", "sha256": None}
            )
    return assets


def prepare_superpoint(layout):
    repository = layout["superpoint_repository"]
    if not repository.exists():
        repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", SUPERPOINT_REPOSITORY, str(repository)]
        )
        subprocess.check_call(
            ["git", "-C", str(repository), "checkout", "--detach", SUPERPOINT_COMMIT]
        )
    else:
        if not (repository / ".git").is_dir():
            raise EnvironmentError(
                "SuperPoint destination is not a Git checkout: {}".format(repository)
            )
        if _git_output(repository, ["status", "--porcelain"]):
            raise EnvironmentError("SuperPoint checkout has uncommitted changes")
        if _git_output(repository, ["rev-parse", "HEAD"]) != SUPERPOINT_COMMIT:
            raise EnvironmentError("SuperPoint checkout is not at the pinned commit")
    checkpoint = repository / "weights/superpoint_v6_from_tf.pth"
    if sha256_file(checkpoint) != SUPERPOINT_CHECKPOINT_SHA256:
        raise EnvironmentError("Pinned SuperPoint checkpoint checksum mismatch")
    destination = layout["superpoint_models"] / checkpoint.name
    record = _copy_verified_asset(checkpoint, destination)
    record["source_commit"] = SUPERPOINT_COMMIT
    return record


def _write_activation_script(layout, repository_path):
    destination = layout["runtime"] / "activate.sh"
    content = """#!/usr/bin/env bash
# Generated by Quadra persistent environment bootstrap.
quadra_activate() {{
  local profile="${{1:-}}"
  if [[ "${{profile}}" != "preprocess" && "${{profile}}" != "uae" ]]; then
    echo "Usage: source {activation} preprocess|uae" >&2
    return 2
  fi
  export QUADRA_STORAGE_ROOT="{storage_root}"
  export QUADRA_REPO_ROOT="{repository}"
  export QUADRA_DATASET_ROOT="{whole_body_ct}"
  export QUADRA_TOTALSEG_ROOT="{totalsegmentator_outputs}"
  export QUADRA_TOTALSEG_MASK_ROOT="{totalsegmentator_outputs}"
  export QUADRA_TOTALSEG_OUTPUT_ROOT="{runs_preprocessing}"
  export QUADRA_MODEL_ROOT="{model_root}"
  export QUADRA_OUTPUT_ROOT="{runs_root}"
  export TOTALSEG_WEIGHTS_PATH="{totalsegmentator_cache}"
  export TORCH_HOME="{storage_root}/cache/torch"
  export XDG_CACHE_HOME="{storage_root}/cache/xdg"
  export VSCODE_CLI_DATA_DIR="{vscode_data}"
  export PYTHONPATH="{repository}${{PYTHONPATH:+:${{PYTHONPATH}}}}"
  if [[ "${{profile}}" == "preprocess" ]]; then
    if [[ ! -f "{preprocess_venv}/bin/activate" ]]; then
      echo "Persistent preprocessing environment is missing." >&2
      return 2
    fi
    # shellcheck disable=SC1091
    source "{preprocess_venv}/bin/activate"
  fi
  cd "{repository}" || return
  python -m tools.quadra.environment preflight \
    --profile "${{profile}}" --storage-root "{storage_root}" || return
}}
quadra_activate "$@"
quadra_status=$?
unset -f quadra_activate
return "${{quadra_status}}" 2>/dev/null || exit "${{quadra_status}}"
""".format(
        activation=destination,
        storage_root=layout["storage_root"],
        repository=repository_path,
        whole_body_ct=layout["whole_body_ct"],
        totalsegmentator_outputs=layout["totalsegmentator_outputs"],
        runs_preprocessing=layout["runs_preprocessing"],
        model_root=layout["storage_root"] / "models",
        runs_root=layout["storage_root"] / "runs",
        totalsegmentator_cache=layout["totalsegmentator_cache"],
        vscode_data=layout["vscode_data"],
        preprocess_venv=layout["preprocess_venv"],
    )
    destination.write_text(content, encoding="utf-8")
    destination.chmod(0o755)
    return destination


def _create_preprocess_venv(layout, requirements, skip_install):
    venv = layout["preprocess_venv"]
    if not (venv / "bin/python").exists():
        subprocess.check_call(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)]
        )
    if not skip_install:
        subprocess.check_call(
            [
                str(venv / "bin/python"),
                "-m",
                "pip",
                "install",
                "--requirement",
                str(requirements),
            ]
        )
    return venv


def _link_legacy_assets(layout, legacy_root, backup_root):
    candidates = [
        (layout["whole_body_ct"], legacy_root / "data/QUADRA_HC_WB"),
        (
            layout["totalsegmentator_outputs"],
            legacy_root / "outputs",
        ),
        (
            layout["manifests"] / "totalsegmentator_cohort.json",
            legacy_root / "metadata/cohort_manifest.json",
        ),
        (
            layout["manifests"] / "Demographics (All).xlsx",
            legacy_root / "metadata/Demographics (All).xlsx",
        ),
        (layout["runs_archive"] / "subject021_20260722", backup_root),
    ]
    links = []
    for link, target in candidates:
        if target.exists():
            links.append(
                {
                    "path": str(link),
                    "target": str(target.resolve()),
                    "status": ensure_link(link, target, layout["storage_root"]),
                }
            )
        else:
            links.append(
                {"path": str(link), "target": str(target), "status": "target-missing"}
            )
    return links


def _prepare_layout(layout):
    special = {
        layout["whole_body_ct"],
        layout["totalsegmentator_outputs"],
        layout["totalsegmentator_cache"],
        layout["preprocess_venv"],
        layout["superpoint_repository"],
    }
    for key, path in layout.items():
        if key == "storage_root" or path in special:
            continue
        _ensure_directory(path, layout["storage_root"])
    layout["storage_root"].mkdir(parents=True, exist_ok=True)


def _manifest_path(layout):
    return layout["manifests"] / "environment.json"


def load_environment_manifest(layout):
    path = _manifest_path(layout)
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "profiles": {},
            "assets": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise EnvironmentError("Unsupported environment manifest schema")
    return value


def profile_runtime_errors(profile, fingerprint):
    errors = []
    python_parts = tuple(int(value) for value in fingerprint["python"].split(".")[:2])
    torch_version = fingerprint["torch"]
    if profile == "preprocess":
        if python_parts < (3, 10):
            errors.append("Preprocessing requires Python 3.10 or newer")
        if torch_version.startswith("unavailable"):
            errors.append("PyTorch is unavailable")
        elif int(torch_version.split(".", 1)[0]) < 2:
            errors.append("Preprocessing requires PyTorch 2 or newer")
        if fingerprint["totalsegmentator"] != "2.16.0":
            errors.append("Preprocessing requires TotalSegmentator 2.16.0")
    elif profile == "uae":
        if python_parts != (3, 7):
            errors.append("UAE profile expects Python 3.7")
        if not torch_version.startswith("1.9"):
            errors.append("UAE profile expects PyTorch 1.9")
        if fingerprint["mmcv"].startswith("unavailable"):
            errors.append("MMCV is unavailable")
        if fingerprint["mmdet"].startswith("unavailable"):
            errors.append("MMDetection is unavailable")
    return errors


def verify_assets(layout, profile=None):
    checks = {
        "storage_root": layout["storage_root"].is_dir(),
        "repository": (
            layout["storage_root"].parent / "repos" / DEFAULT_REPOSITORY_NAME / ".git"
        ).is_dir(),
        "whole_body_ct": layout["whole_body_ct"].is_dir(),
        "totalsegmentator_outputs": layout["totalsegmentator_outputs"].is_dir(),
        "totalsegmentator_cache": layout["totalsegmentator_cache"].is_dir(),
        "preprocess_venv": (layout["preprocess_venv"] / "bin/python").is_file(),
        "superpoint_repository": (
            layout["superpoint_repository"] / ".git"
        ).is_dir(),
        "superpoint_checkpoint": (
            layout["superpoint_models"] / "superpoint_v6_from_tf.pth"
        ).is_file(),
        "uae_s_config": (
            layout["storage_root"].parent
            / "repos"
            / DEFAULT_REPOSITORY_NAME
            / "configs/samv2/samv2_NIHLN.py"
        ).is_file(),
        "uae_s_checkpoint": (
            layout["uae_models"] / "SAMv2_iter_20000.pth"
        ).is_file(),
    }
    required = ["storage_root", "repository", "whole_body_ct"]
    if profile == "preprocess":
        required.extend(
            [
                "totalsegmentator_outputs",
                "totalsegmentator_cache",
                "preprocess_venv",
                "superpoint_repository",
                "superpoint_checkpoint",
            ]
        )
    elif profile == "uae":
        required.extend(["uae_s_config", "uae_s_checkpoint"])
    return {
        "checked_at": utc_now(),
        "profile": profile,
        "checks": checks,
        "required": required,
        "ok": all(checks.get(name, False) for name in required),
    }


def bootstrap(args):
    storage_root = validate_storage_root(args.storage_root)
    layout = canonical_layout(storage_root)
    _prepare_layout(layout)
    persistent_repository, branch, commit = ensure_persistent_repository(
        args.repository_root, storage_root
    )
    links = _link_legacy_assets(
        layout, Path(args.legacy_totalseg_root), Path(args.backup_root)
    )
    requirements = (
        Path(args.repository_root)
        / "tools/quadra/environment/requirements-preprocess.txt"
    )
    profile_assets = []
    if args.profile == "preprocess":
        profile_assets.append(
            _copy_tree_once(
                Path(args.legacy_totalseg_root) / "model-cache",
                layout["totalsegmentator_cache"],
            )
        )
        _create_preprocess_venv(layout, requirements, args.skip_install)
        if not args.skip_install:
            profile_assets.append(prepare_superpoint(layout))
    else:
        profile_assets.extend(
            prepare_uae_assets(args.repository_root, persistent_repository, layout)
        )
    activation = _write_activation_script(layout, persistent_repository)
    image_ref = args.image_ref or PROFILE_IMAGES[args.profile]
    digest = None
    digest_error = None
    if not args.skip_network:
        try:
            digest = resolve_docker_digest(image_ref)
        except Exception as exc:
            digest_error = str(exc)
    expected_digest = (
        PROFILE_EXPECTED_DIGESTS.get(args.profile)
        if image_ref == PROFILE_IMAGES[args.profile]
        else None
    )
    if digest and expected_digest and digest != expected_digest:
        raise EnvironmentError(
            "Container tag digest changed: {} != {}".format(
                digest, expected_digest
            )
        )
    fingerprint = runtime_fingerprint(args.profile, image_ref)
    if args.profile == "preprocess":
        venv_python = layout["preprocess_venv"] / "bin/python"
        fingerprint["python"] = _command_output(
            [str(venv_python), "-c", "import platform; print(platform.python_version())"]
        )
        for module_name in ("torch", "totalsegmentator"):
            fingerprint[module_name] = _command_output(
                [
                    str(venv_python),
                    "-c",
                    (
                        "import importlib.metadata as m; "
                        "print(m.version({!r}))"
                    ).format(module_name),
                ]
            )
    runtime_errors = profile_runtime_errors(args.profile, fingerprint)
    asset_check = verify_assets(layout, args.profile)
    manifest = load_environment_manifest(layout)
    manifest["updated_at"] = utc_now()
    manifest["storage_root"] = str(storage_root)
    manifest["repository"] = {
        "path": str(persistent_repository),
        "branch": branch,
        "commit": commit,
    }
    manifest["legacy_links"] = links
    manifest["profiles"][args.profile] = {
        "bootstrapped_at": utc_now(),
        "image_ref": image_ref,
        "image_digest": digest,
        "expected_image_digest": expected_digest,
        "image_digest_error": digest_error,
        "fingerprint_path": str(
            layout["profiles"] / (args.profile + "-fingerprint.json")
        ),
        "runtime_errors": runtime_errors,
        "asset_check": asset_check,
        "assets": profile_assets,
    }
    atomic_write_json(
        layout["profiles"] / (args.profile + "-fingerprint.json"), fingerprint
    )
    atomic_write_json(_manifest_path(layout), manifest)
    print("Persistent Quadra profile prepared: {}".format(args.profile))
    print("Repository: {} @ {}".format(persistent_repository, commit))
    print("Activation: source {} {}".format(activation, args.profile))
    if digest_error:
        print("WARNING: image digest was not resolved: {}".format(digest_error))
    if runtime_errors:
        raise EnvironmentError("; ".join(runtime_errors))
    if not asset_check["ok"]:
        missing = [
            name
            for name in asset_check["required"]
            if not asset_check["checks"].get(name, False)
        ]
        raise EnvironmentError(
            "Profile assets are incomplete: {}".format(", ".join(missing))
        )
    return 0


def preflight(args):
    layout = canonical_layout(args.storage_root)
    manifest = load_environment_manifest(layout)
    if args.profile not in manifest.get("profiles", {}):
        raise EnvironmentError(
            "Profile has not been bootstrapped: {}".format(args.profile)
        )
    fingerprint = runtime_fingerprint(
        args.profile,
        manifest["profiles"][args.profile]["image_ref"],
        include_packages=False,
    )
    runtime_errors = profile_runtime_errors(args.profile, fingerprint)
    assets = verify_assets(layout, args.profile)
    result = {
        "status": "ok" if not runtime_errors and assets["ok"] else "failed",
        "profile": args.profile,
        "runtime_errors": runtime_errors,
        "assets": assets,
        "fingerprint": fingerprint,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


def status(args):
    layout = canonical_layout(args.storage_root)
    manifest = load_environment_manifest(layout)
    result = {
        "storage_root": str(layout["storage_root"]),
        "manifest": str(_manifest_path(layout)),
        "profiles": manifest.get("profiles", {}),
        "assets": verify_assets(layout),
        "disk": shutil.disk_usage(str(layout["storage_root"]))
        if layout["storage_root"].exists()
        else None,
    }
    if result["disk"]:
        result["disk"] = {
            "total_bytes": result["disk"].total,
            "used_bytes": result["disk"].used,
            "free_bytes": result["disk"].free,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def verify_assets_command(args):
    layout = canonical_layout(args.storage_root)
    result = verify_assets(layout, args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


def resume(args):
    activation = canonical_layout(args.storage_root)["runtime"] / "activate.sh"
    if not activation.is_file():
        raise EnvironmentError("Activation script is missing: {}".format(activation))
    print("Run in the current shell:")
    print("  source {} {}".format(activation, args.profile))
    return preflight(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare and verify persistent Quadra RunPod environments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--storage-root",
        type=Path,
        default=Path(os.environ.get("QUADRA_STORAGE_ROOT", DEFAULT_STORAGE_ROOT)),
    )

    bootstrap_parser = subparsers.add_parser(
        "bootstrap", parents=[common], help="Prepare one persistent profile."
    )
    bootstrap_parser.set_defaults(handler=bootstrap)
    bootstrap_parser.add_argument("--profile", choices=("preprocess", "uae"), required=True)
    bootstrap_parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    bootstrap_parser.add_argument(
        "--legacy-totalseg-root",
        type=Path,
        default=DEFAULT_LEGACY_TOTALSEG_ROOT,
    )
    bootstrap_parser.add_argument(
        "--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT
    )
    bootstrap_parser.add_argument("--image-ref")
    bootstrap_parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Create structure without installing packages or SuperPoint.",
    )
    bootstrap_parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip registry digest resolution.",
    )

    resume_parser = subparsers.add_parser(
        "resume", parents=[common], help="Verify and print the activation command."
    )
    resume_parser.set_defaults(handler=resume)
    resume_parser.add_argument("--profile", choices=("preprocess", "uae"), required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", parents=[common], help="Verify the active runtime and assets."
    )
    preflight_parser.set_defaults(handler=preflight)
    preflight_parser.add_argument(
        "--profile", choices=("preprocess", "uae"), required=True
    )

    status_parser = subparsers.add_parser(
        "status", parents=[common], help="Report persistent environment status."
    )
    status_parser.set_defaults(handler=status)
    status_parser.add_argument("--profile", choices=("preprocess", "uae"))

    verify_parser = subparsers.add_parser(
        "verify-assets", parents=[common], help="Verify persistent assets."
    )
    verify_parser.set_defaults(handler=verify_assets_command)
    verify_parser.add_argument("--profile", choices=("preprocess", "uae"))

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except EnvironmentError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
