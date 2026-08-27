"""Isolated CPU registration profile, dispatched by environment.py."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from tools.quadra import environment as env

PINS = {"itk": "5.4.5", "itk-elastix": "0.25.2", "numpy": "1.26.4",
        "nibabel": "5.3.2", "scipy": "1.15.3", "matplotlib": "3.9.4",
        "psutil": "7.0.0", "PyYAML": "6.0.2"}


def fingerprint(root):
    from tools.quadra.registration_point_transform import require
    require(sys.version_info[:2] == (3, 11), "Registration requires Python 3.11")
    require(platform.system() == "Linux", "Real registration is RunPod/Linux only")
    require(os.environ.get("RUNPOD_POD_ID") == "1ngcj5dw1mifiw", "Unexpected or absent RunPod identity")
    require(Path(sys.prefix).resolve() == (Path(root)/"runtime/registration-venv").resolve(),
            "Activate the isolated registration venv")
    versions = {name: importlib.metadata.version(name) for name in PINS}
    require(versions == PINS, "Registration dependency versions changed")
    packages = sorted(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines())
    return {"python": platform.python_version(), "platform": platform.platform(),
            "pod_id": os.environ["RUNPOD_POD_ID"], "packages": versions, "pip_freeze": packages,
            "image_digest": env.PROFILE_EXPECTED_DIGESTS["preprocess"],
            "image_ref": env.PROFILE_IMAGES["preprocess"], "profile": "registration",
            "image_identity_verification":"operator-confirmed digest at bootstrap; not inferred from packages"}


def preflight(root):
    from tools.quadra.registration_point_transform import load_json, require, identity
    path = Path(root)/"runtime/profiles/registration.json"
    saved = load_json(path)
    current = fingerprint(root)
    require(current == saved["fingerprint"], "Registration environment fingerprint changed")
    req = Path(__file__).parent/"environment/requirements-registration.txt"
    require(identity(req)["sha256"] == saved["requirements_sha256"], "Registration requirements changed")
    return current


def bootstrap(args):
    from tools.quadra.registration_point_transform import atomic_json, identity, load_json, require, utc_now
    root = env.validate_storage_root(args.storage_root)
    require(sys.version_info[:2] == (3, 11) and platform.system() == "Linux",
            "Use the pinned preprocessing container with Python 3.11; do not edit the pod automatically")
    require(os.environ.get("RUNPOD_POD_ID") == "1ngcj5dw1mifiw", "Unexpected RunPod")
    require(args.confirm_image_digest == env.PROFILE_EXPECTED_DIGESTS["preprocess"],
            "Live image digest must be verified before registration bootstrap")
    repo = Path(__file__).resolve().parents[2]
    require(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip(),
            "Repository must be clean")
    venv = root/"runtime/registration-venv"
    req = repo/"tools/quadra/environment/requirements-registration.txt"
    marker = root/"runtime/profiles/registration.json"
    if marker.exists():
        subprocess.check_call([str(venv/"bin/python"), "-m", "tools.quadra.environment", "preflight",
                               "--profile", "registration", "--storage-root", str(root)], cwd=repo)
        require("registration-venv" in (root/"runtime/activate.sh").read_text(),
                "Registration marker exists but activation is incomplete; inspect setup evidence")
        return
    if not venv.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    require((venv/"pyvenv.cfg").is_file(), "Conflicting runtime directory")
    require("include-system-site-packages = false" in (venv/"pyvenv.cfg").read_text(), "Venv is not isolated")
    subprocess.check_call([str(venv/"bin/python"), "-m", "pip", "install", "--only-binary=:all:", "-r", str(req)],
                          env=dict(os.environ,PIP_CACHE_DIR=str(root/"cache/pip-registration")))
    subprocess.check_call([str(venv/"bin/python"), "-m", "pip", "check"])
    code = "import json; from tools.quadra.registration_runtime import fingerprint; print(json.dumps(fingerprint({!r})))".format(str(root))
    fp = json.loads(subprocess.check_output([str(venv/"bin/python"), "-c", code], cwd=repo, text=True))
    # Preserve the previous activation script as provenance; existing envs are untouched.
    activation = root/"runtime/activate.sh"
    if activation.exists():
        backup = root/"runtime/activate.before-registration.sh"
        from tools.quadra.registration_point_transform import atomic_text
        if not backup.exists():
            atomic_text(backup, activation.read_text(), refuse=True)
        else:
            require(activation.read_bytes() == backup.read_bytes() or
                    "registration-venv" in activation.read_text(), "Unexpected activation change")
    env._write_activation_script(env.canonical_layout(root), repo)
    atomic_json(marker, {"created_at": utc_now(), "fingerprint": fp,
                        "requirements_sha256": identity(req)["sha256"]}, refuse=True)
    print("Registration profile ready. source {}/runtime/activate.sh registration".format(root))


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "preflight", "resume", "status", "verify-assets"))
    parser.add_argument("--profile", choices=("registration",), required=True)
    parser.add_argument("--storage-root", type=Path, default=Path("/workspace/quadra"))
    parser.add_argument("--confirm-image-digest")
    args = parser.parse_args(argv)
    try:
        if args.command == "bootstrap":
            bootstrap(args)
        else:
            print(json.dumps({"status": "PASS", "fingerprint": preflight(args.storage_root)}, indent=2))
        return 0
    except Exception as exc:
        print("Registration environment BLOCKED: {}".format(exc), file=sys.stderr)
        return 2
