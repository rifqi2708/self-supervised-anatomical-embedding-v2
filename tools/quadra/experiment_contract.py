"""Build and verify the small frozen-input contract for disposable Quadra pods."""
from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = 1
ALIGNED_RUN = "runs/cohort/uaes-aligned-100mm-20260820T044539Z"
REGISTRATION_RUN = "runs/cohort/registration-organ-group-cohort-20260828T071049Z"
CORRECTED_PACKAGE = "transfer/packages/sacrum030-corrected-derivative-20260828T230138Z/quadra/datasets/derivatives/totalsegmentator_2.16.0_organs_v1-sacrum030-singlevoxel-20260828T230138Z"


class ContractError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source, target):
    source = Path(source)
    if not source.is_file():
        raise ContractError("Required contract input is missing: {}".format(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target))


def _count_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def stage_contract(local_archive, staging):
    local_archive = Path(local_archive)
    staging = Path(staging)
    root = staging / "quadra-experiment-contract-v1"
    aligned = local_archive / ALIGNED_RUN
    registration = local_archive / REGISTRATION_RUN
    mappings = [
        (aligned / "frozen_queries_raw_itk.csv", root / "queries/frozen_queries_raw_itk.csv"),
        (aligned / "cohort_manifest.json", root / "accepted/uae_cohort_manifest.json"),
        (aligned / "checkpoint_summary.json", root / "accepted/uae_checkpoint_summary.json"),
        (registration / "organ_group_cohort_manifest.json", root / "accepted/registration_cohort_manifest.json"),
        (registration / "pilot_approval.json", root / "accepted/registration_pilot_approval.json"),
        (local_archive / "metadata/manifests/totalsegmentator_cohort.json", root / "metadata/mask_registry_and_cohort.json"),
        (local_archive / "metadata/manifests/registration-setup-20260827/resolved_registration_parameters.json", root / "accepted/registration_parameters.json"),
        (local_archive / "metadata/known_assets/sacrum030-corrected-derivative-20260828T230138Z.json", root / "corrected_subject030/provenance.json"),
        (local_archive / CORRECTED_PACKAGE / "derivative_manifest.json", root / "corrected_subject030/derivative_manifest.json"),
        (local_archive / CORRECTED_PACKAGE / "quadra_hc_030/test/masks/sacrum.nii.gz", root / "corrected_subject030/quadra_hc_030/test/masks/sacrum.nii.gz"),
    ]
    for source, destination in mappings:
        _copy(source, destination)
    plans = sorted((aligned / "plans").glob("*.json"))
    if len(plans) != 224:
        raise ContractError("Expected 224 aligned Test/Retest group plans, found {}".format(len(plans)))
    for path in plans:
        _copy(path, root / "organ_group_plans" / path.name)
    query_rows = _count_csv_rows(root / "queries/frozen_queries_raw_itk.csv")
    if query_rows != 108431:
        raise ContractError("Expected 108431 frozen queries, found {}".format(query_rows))
    manifest = json.loads((root / "accepted/uae_cohort_manifest.json").read_text(encoding="utf-8"))
    denominators = manifest.get("denominators", {})
    if denominators.get("subjects") != 28 or denominators.get("queries") != 108431:
        raise ContractError("Accepted UAE denominators do not match the frozen contract")
    (root / "metadata/denominators.json").write_text(
        json.dumps(denominators, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})
    package_manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": "quadra-experiment-contract-v1",
        "frozen_queries": query_rows,
        "subjects": 28,
        "organ_groups": ["abdomen", "head_neck", "pelvis", "thorax"],
        "organ_group_plan_files": len(plans),
        "contains_generated_results": False,
        "files": files,
    }
    (root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        checksum_lines.append("{}  {}".format(sha256_file(path), path.relative_to(root).as_posix()))
    (root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return root, package_manifest


def build_package(local_archive, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quadra-contract-") as temporary:
        root, manifest = stage_contract(local_archive, temporary)
        temporary_output = output.parent / ("." + output.name + ".tmp")
        with temporary_output.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    for path in sorted(item for item in root.rglob("*") if item.is_file()):
                        relative = (Path(root.name) / path.relative_to(root)).as_posix()
                        payload = path.read_bytes()
                        info = tarfile.TarInfo(relative)
                        info.size = len(payload)
                        info.mode = 0o644
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(payload))
        os.replace(str(temporary_output), str(output))
    result = {"path": str(output), "bytes": output.stat().st_size,
              "sha256": sha256_file(output), "manifest": manifest}
    return result


def verify_package(path):
    from tools.quadra.disposable_pod import extract_archive
    with tempfile.TemporaryDirectory(prefix="quadra-contract-verify-") as temporary:
        extract_archive(path, "tar.gz", temporary)
        root = Path(temporary) / "quadra-experiment-contract-v1"
        manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
        checks = {}
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            checks[relative] = sha256_file(root / relative) == digest
        if not checks or not all(checks.values()):
            raise ContractError("Experiment contract inner checksum verification failed")
        if manifest.get("frozen_queries") != 108431 or manifest.get("organ_group_plan_files") != 224:
            raise ContractError("Experiment contract counts are invalid")
    return {"status": "VERIFIED", "path": str(path), "bytes": Path(path).stat().st_size,
            "sha256": sha256_file(path), "inner_files_verified": len(checks)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--local-archive", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_package(args.local_archive, args.output) if args.command == "build" else verify_package(args.package)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ContractError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
