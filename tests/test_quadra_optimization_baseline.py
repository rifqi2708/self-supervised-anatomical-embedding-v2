import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.quadra import optimization_baseline as baseline
from tools.quadra.environment import canonical_layout


def initialize_repository(root: Path, branch: str = baseline.EXPECTED_BRANCH) -> str:
    root.mkdir(parents=True)
    subprocess.check_call(["git", "-C", str(root), "init", "-b", branch])
    subprocess.check_call(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"]
    )
    subprocess.check_call(
        ["git", "-C", str(root), "config", "user.name", "Stage Zero Test"]
    )
    (root / "README.md").write_text("baseline\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(root), "add", "README.md"])
    subprocess.check_call(["git", "-C", str(root), "commit", "-m", "baseline"])
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


class RepositoryContractTests(unittest.TestCase):
    def test_repository_requires_expected_branch_cleanliness_and_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            commit = initialize_repository(repository)
            result = baseline.inspect_repository(
                repository,
                expected_base_commit=commit,
                expected_branch=baseline.EXPECTED_BRANCH,
            )
            self.assertTrue(result["clean"])
            self.assertEqual(result["execution_commit"], commit)

            (repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(
                baseline.OptimizationBaselineError, "clean repository"
            ):
                baseline.inspect_repository(
                    repository,
                    expected_base_commit=commit,
                    expected_branch=baseline.EXPECTED_BRANCH,
                )

    def test_repository_rejects_wrong_branch_and_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            commit = initialize_repository(repository, branch="wrong-branch")
            with self.assertRaisesRegex(
                baseline.OptimizationBaselineError, "requires branch"
            ):
                baseline.inspect_repository(
                    repository,
                    expected_base_commit=commit,
                    expected_branch=baseline.EXPECTED_BRANCH,
                )

            subprocess.check_call(
                ["git", "-C", str(repository), "branch", "-m", baseline.EXPECTED_BRANCH]
            )
            with self.assertRaisesRegex(
                baseline.OptimizationBaselineError, "does not descend"
            ):
                baseline.inspect_repository(
                    repository,
                    expected_base_commit="0" * 40,
                    expected_branch=baseline.EXPECTED_BRANCH,
                )


class AssetAndRuntimeTests(unittest.TestCase):
    def test_asset_counts_and_missing_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory) / "quadra"
            layout = canonical_layout(storage_root)
            layout["whole_body_ct"].mkdir(parents=True)
            layout["totalsegmentator_outputs"].mkdir(parents=True)
            subject = layout["whole_body_ct"] / "QUADRA_HC_021"
            subject.mkdir()
            (subject / "test_CT-AC.nii.gz").touch()
            scan = (
                layout["totalsegmentator_outputs"]
                / "quadra_hc_021"
                / "test"
                / "masks"
            )
            scan.mkdir(parents=True)
            (scan / "colon.nii.gz").touch()
            expected = {
                "ct_subjects": 1,
                "ct_files": 1,
                "mask_subjects": 1,
                "stage5_scans": 1,
                "final_masks": 1,
            }
            with mock.patch.object(baseline, "EXPECTED_COUNTS", expected):
                result = baseline.inspect_assets(storage_root)
            self.assertEqual(result["counts"], expected)

            (subject / "retest_CT-AC.nii.gz").touch()
            with mock.patch.object(baseline, "EXPECTED_COUNTS", expected):
                with self.assertRaisesRegex(
                    baseline.OptimizationBaselineError, "counts differ"
                ):
                    baseline.inspect_assets(storage_root)

            missing_root = Path(directory) / "missing"
            with self.assertRaisesRegex(
                baseline.OptimizationBaselineError, "CT root is missing"
            ):
                baseline.inspect_assets(missing_root)

    def test_gpu_parser_requires_one_a6000(self):
        parsed = baseline.parse_nvidia_smi(
            "0, NVIDIA RTX A6000, 580.159.03, 49140, GPU-test\n"
        )
        self.assertEqual(parsed["memory_total_mib"], 49140)
        self.assertEqual(parsed["accepted_peak_reserved_mib"], 39312)
        with self.assertRaisesRegex(
            baseline.OptimizationBaselineError, "requires NVIDIA RTX A6000"
        ):
            baseline.parse_nvidia_smi(
                "0, NVIDIA A100-SXM4-40GB, 580.159.03, 40960, GPU-test"
            )
        with self.assertRaisesRegex(
            baseline.OptimizationBaselineError, "exactly one visible GPU"
        ):
            baseline.parse_nvidia_smi("")

    def test_runtime_requires_activation_and_preprocess_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory) / "quadra"
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    baseline.OptimizationBaselineError, "QUADRA_STORAGE_ROOT is unset"
                ):
                    baseline.inspect_active_preprocess_runtime(storage_root)

            versions = {"torch": "2.4.1", "TotalSegmentator": "2.16.0"}
            with mock.patch.dict(
                "os.environ", {"QUADRA_STORAGE_ROOT": str(storage_root)}
            ), mock.patch.object(
                baseline.importlib_metadata,
                "version",
                side_effect=lambda name: versions[name],
            ):
                result = baseline.inspect_active_preprocess_runtime(storage_root)
            self.assertEqual(result["profile"], "preprocess")

    def test_environment_profiles_require_both_pinned_image_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory) / "quadra"
            layout = canonical_layout(storage_root)
            layout["manifests"].mkdir(parents=True)
            layout["profiles"].mkdir(parents=True)
            profiles = {}
            for name in ("preprocess", "uae"):
                fingerprint = layout["profiles"] / f"{name}-fingerprint.json"
                fingerprint.write_text(
                    json.dumps({"profile": name}) + "\n", encoding="utf-8"
                )
                profiles[name] = {
                    "image_ref": baseline.PROFILE_IMAGES[name],
                    "image_digest": baseline.PROFILE_EXPECTED_DIGESTS[name],
                    "expected_image_digest": baseline.PROFILE_EXPECTED_DIGESTS[name],
                    "fingerprint_path": str(fingerprint),
                    "runtime_errors": [],
                }
            manifest_path = layout["manifests"] / "environment.json"
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "profiles": profiles}) + "\n",
                encoding="utf-8",
            )
            result = baseline.inspect_environment_profiles(storage_root)
            self.assertEqual(set(result["profiles"]), {"preprocess", "uae"})

            profiles["uae"]["image_digest"] = "sha256:" + "0" * 64
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "profiles": profiles}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                baseline.OptimizationBaselineError, "digest is incompatible"
            ):
                baseline.inspect_environment_profiles(storage_root)


class BaselineCaptureTests(unittest.TestCase):
    def _capture(self, directory: str, run_id: str = "stage0-test") -> Path:
        root = Path(directory)
        repository = root / "repository"
        config = repository / baseline.CONFIG_RELATIVE_PATH
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("model = dict(type='SAMv2')\n", encoding="utf-8")
        storage_root = root / "quadra"
        layout = canonical_layout(storage_root)
        layout["uae_models"].mkdir(parents=True, exist_ok=True)
        checkpoint = layout["uae_models"] / baseline.CHECKPOINT_FILENAME
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_hash = baseline.sha256_file(checkpoint)
        output_root = storage_root / "runs/memory_optimization"
        repository_record = {
            "path": str(repository),
            "branch": baseline.EXPECTED_BRANCH,
            "base_commit": baseline.EXPECTED_BASE_COMMIT,
            "execution_commit": "1" * 40,
            "clean": True,
        }
        assets = {
            "whole_body_ct_root": str(layout["whole_body_ct"]),
            "totalsegmentator_mask_root": str(layout["totalsegmentator_outputs"]),
            "counts": dict(baseline.EXPECTED_COUNTS),
            "validation_level": "test",
        }
        environments = {
            "manifest": {"path": "environment.json", "bytes": 1, "sha256": "2" * 64},
            "profiles": {
                name: {
                    "image_ref": baseline.PROFILE_IMAGES[name],
                    "image_digest": baseline.PROFILE_EXPECTED_DIGESTS[name],
                    "expected_image_ref": baseline.PROFILE_IMAGES[name],
                    "runtime_errors": [],
                    "fingerprint": {
                        "path": f"{name}.json",
                        "bytes": 1,
                        "sha256": "3" * 64,
                    },
                }
                for name in ("preprocess", "uae")
            },
        }
        runtime = {
            "profile": "preprocess",
            "python": "3.11.10",
            "torch": "2.4.1",
            "totalsegmentator": "2.16.0",
        }
        gpu = baseline.parse_nvidia_smi(
            "0, NVIDIA RTX A6000, 580.159.03, 49140, GPU-test"
        )
        with mock.patch.object(
            baseline, "EXPECTED_CHECKPOINT_SHA256", checkpoint_hash
        ), mock.patch.object(
            baseline, "inspect_repository", return_value=repository_record
        ), mock.patch.object(
            baseline, "inspect_assets", return_value=assets
        ), mock.patch.object(
            baseline, "inspect_environment_profiles", return_value=environments
        ), mock.patch.object(
            baseline, "inspect_active_preprocess_runtime", return_value=runtime
        ), mock.patch.object(
            baseline, "inspect_gpu", return_value=gpu
        ):
            return baseline.capture_baseline(
                repository_root=repository,
                storage_root=storage_root,
                output_root=output_root,
                run_id=run_id,
            )

    def test_capture_writes_complete_contract_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = self._capture(directory)
            manifest = baseline.load_baseline_manifest(
                run_directory / "baseline_manifest.json"
            )
            summary = json.loads(
                (run_directory / "checkpoint_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["scientific_contract"]["seed"], baseline.OPTIMIZATION_SEED
            )
            self.assertEqual(
                manifest["scientific_contract"]["matching_modes"],
                ["global_nn", "fixed_point"],
            )
            self.assertFalse(manifest["scope"]["full_cohort_cycle_error"])
            self.assertEqual(summary["status"], "PASS")
            self.assertFalse(
                summary["gates"]["ct_or_model_computation_launched"]
            )
            with self.assertRaisesRegex(
                baseline.OptimizationBaselineError, "reuse existing"
            ):
                self._capture(directory)

    def test_capture_rejects_checkpoint_mismatch_and_output_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            config = repository / baseline.CONFIG_RELATIVE_PATH
            config.parent.mkdir(parents=True)
            config.write_text("config\n", encoding="utf-8")
            storage_root = root / "quadra"
            layout = canonical_layout(storage_root)
            layout["uae_models"].mkdir(parents=True)
            (layout["uae_models"] / baseline.CHECKPOINT_FILENAME).write_bytes(
                b"wrong"
            )
            with mock.patch.object(
                baseline,
                "inspect_repository",
                return_value={"execution_commit": "1" * 40},
            ):
                with self.assertRaisesRegex(
                    baseline.OptimizationBaselineError, "checksum mismatch"
                ):
                    baseline.capture_baseline(
                        repository_root=repository,
                        storage_root=storage_root,
                        output_root=storage_root / "runs",
                        run_id="stage0-test",
                    )
                with self.assertRaisesRegex(
                    baseline.OptimizationBaselineError, "inside the Quadra"
                ):
                    baseline.capture_baseline(
                        repository_root=repository,
                        storage_root=storage_root,
                        output_root=root / "outside",
                        run_id="stage0-test",
                    )

    def test_locked_contract_rejects_changed_seed_before_stage_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            manifest = baseline.build_baseline_manifest(
                repository={},
                model={},
                assets={},
                environment={"profiles": {}},
                active_runtime={},
                gpu={},
            )
            baseline.atomic_write_json(path, manifest)
            with self.assertRaisesRegex(
                baseline.OptimizationBaselineError, "seed"
            ):
                baseline.validate_locked_contract(
                    path,
                    repository_root=Path(directory),
                    storage_root=Path(directory) / "quadra",
                    seed=1,
                )

    def test_atomic_writer_refuses_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            baseline.atomic_write_json(path, {"value": 1})
            with self.assertRaisesRegex(
                baseline.OptimizationBaselineError, "overwrite"
            ):
                baseline.atomic_write_json(path, {"value": 2})


if __name__ == "__main__":
    unittest.main()
