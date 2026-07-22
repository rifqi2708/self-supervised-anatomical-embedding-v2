import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.quadra.summarize_streaming_validation import (
    aggregate,
    load_run,
    main,
    render_report,
    seam_metrics,
    validate_compatible_runs,
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correspondence_row(phase, comparison, organ, displacement, cycle, seam):
    return {
        "phase": phase,
        "comparison": comparison,
        "organ": organ,
        "forward_displacement_mm": displacement,
        "backward_displacement_mm": displacement,
        "cycle_error_abs_delta_mm": cycle,
        "query_seam_distance_mm": seam,
        "candidate_target_seam_distance_mm": seam,
    }


def make_run(root, subject, spacing=(2.0, 2.0, 2.0), drop_matcher=False):
    run_dir = root / subject
    run_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "subject_id": subject,
        "checkpoint": {"path": "checkpoints/SAM.pth", "sha256": "checkpoint-hash"},
        "config": {"path": "configs/samv2.py", "sha256": "config-hash"},
        "checkpoint_role": "base_sam_engineering",
        "norm_spacing_xyz": list(spacing),
        "dense_crop_size_xyz": [128, 128, 64],
        "baseline_tile_size_xyz": [128, 128, 64],
        "baseline_halo_xyz": [32, 32, 16],
        "expanded_tile_size_xyz": [160, 160, 80],
        "expanded_halo_xyz": [48, 48, 24],
        "match_chunk_xyz": [64, 64, 32],
        "query_batch_size": 16,
        "num_points_per_organ_full": 2,
        "num_points_per_organ_crop": 2,
        "seed": 20260721,
        "organs": ["colon"],
        "environment": {"gpu": "Synthetic GPU"},
        "phase_status": {
            "crop": {
                "status": "complete",
                "profiles": [{"seconds": 1.0, "peak_gpu_memory_bytes": 1024}],
            },
            "full": {
                "status": "complete",
                "baseline_manifests": {
                    "test": {"generation_seconds": 2.0, "peak_gpu_memory_bytes": 2048},
                    "retest": {"generation_seconds": 2.0, "peak_gpu_memory_bytes": 2048},
                },
                "expanded_manifests": {
                    "test": {"generation_seconds": 3.0, "peak_gpu_memory_bytes": 4096},
                    "retest": {"generation_seconds": 3.0, "peak_gpu_memory_bytes": 4096},
                },
                "matching_profiles": {"forward": {"seconds": 1.0, "peak_gpu_memory_bytes": 512}},
            },
        },
    }
    write_json(run_dir / "validation_manifest.json", manifest)
    write_json(run_dir / "validation_summary.json", {"overall_status": "pass"})

    matcher = [
        {
            "phase": "crop",
            "coordinate_match": "True",
            "score_abs_diff": 1e-6 * index,
        }
        for index in range(4)
    ]
    if drop_matcher:
        matcher.pop()
    write_csv(run_dir / "matcher_equivalence.csv", matcher)

    descriptors = []
    for comparison in (
        "dense_vs_tiled_fp16",
        "dense_vs_expanded_fp16",
        "baseline_vs_expanded_fp16",
    ):
        for region, median in (("all", 0.997), ("seam", 0.996), ("interior", 0.998)):
            descriptors.append(
                {
                    "phase": "crop",
                    "comparison": comparison,
                    "timepoint": "test",
                    "organ": "colon",
                    "level": "fine",
                    "region": region,
                    "cosine_median": median,
                    "cosine_p01": 0.990,
                }
            )
    for region, median in (("all", 0.98), ("seam", 0.97), ("interior", 0.99)):
        descriptors.append(
            {
                "phase": "full",
                "comparison": "baseline_vs_expanded_fp16",
                "timepoint": "test",
                "organ": "ALL",
                "level": "fine",
                "region": region,
                "cosine_median": median,
                "cosine_p01": 0.90,
            }
        )
    write_csv(run_dir / "descriptor_summary.csv", descriptors)

    correspondence = [
        correspondence_row("crop", "dense_vs_baseline_tiled", "colon", 3.0, 2.0, 0.0),
        correspondence_row("crop", "dense_vs_baseline_tiled", "colon", 3.0, 2.0, 8.0),
        correspondence_row("crop", "dense_vs_expanded_tiled", "colon", 0.0, 0.0, 0.0),
        correspondence_row("crop", "dense_vs_expanded_tiled", "colon", 2.0, 1.0, 8.0),
        correspondence_row("full", "baseline_vs_expanded_tiled", "colon", 0.0, 0.0, 0.0),
        correspondence_row("full", "baseline_vs_expanded_tiled", "colon", 2.0, 1.0, 8.0),
    ]
    write_csv(run_dir / "correspondence_comparison.csv", correspondence)
    frozen = [
        {"phase": "crop", "organ": "colon"},
        {"phase": "crop", "organ": "colon"},
        {"phase": "full", "organ": "colon"},
        {"phase": "full", "organ": "colon"},
    ]
    write_csv(run_dir / "frozen_query_points.csv", frozen)
    return run_dir


class QuadraStreamingValidationSummaryTests(unittest.TestCase):
    def test_load_and_aggregate_preserves_raw_denominators(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = load_run(make_run(root, "quadra_hc_021"))
            second = load_run(make_run(root, "quadra_hc_022"))
            data = aggregate([first, second])

        self.assertEqual(data["matcher"]["comparisons"], 8)
        self.assertEqual(data["matcher"]["exact"], 8)
        self.assertEqual(data["crop"]["queries"], 4)
        self.assertEqual(data["crop"]["directions"], 8)
        self.assertAlmostEqual(data["crop"]["within_2mm"], 1.0)

    def test_incompatible_spacing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = load_run(make_run(root, "quadra_hc_021"))
            second = load_run(make_run(root, "quadra_hc_022", spacing=(2.5, 2.5, 2.5)))
            with self.assertRaisesRegex(ValueError, "norm_spacing_xyz"):
                validate_compatible_runs([first, second])

    def test_incomplete_matcher_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "expected 4 matcher rows"):
                load_run(make_run(root, "quadra_hc_021", drop_matcher=True))

    def test_seam_metrics_keep_near_and_far_outliers_separate(self):
        rows = [
            correspondence_row("full", "baseline_vs_expanded_tiled", "colon", 6.0, 0.0, 2.0),
            correspondence_row("full", "baseline_vs_expanded_tiled", "colon", 0.0, 0.0, 2.0),
            correspondence_row("full", "baseline_vs_expanded_tiled", "colon", 6.0, 0.0, 8.0),
            correspondence_row("full", "baseline_vs_expanded_tiled", "colon", 0.0, 0.0, 8.0),
        ]

        metrics = seam_metrics(rows)

        self.assertEqual(metrics["near_outliers"], 1)
        self.assertEqual(metrics["near_total"], 2)
        self.assertEqual(metrics["far_outliers"], 1)
        self.assertEqual(metrics["far_total"], 2)
        self.assertAlmostEqual(metrics["risk_ratio"], 1.0)

    def test_report_contains_required_definitions_and_limitations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run = load_run(make_run(Path(temp_dir), "quadra_hc_021"))
            report = render_report([run], aggregate([run]), {})

        self.assertIn("**Dense inference:**", report)
        self.assertIn("**Halo:**", report)
        self.assertIn("160×160×80", report)
        self.assertIn("Streamed global matching", report)
        self.assertIn("cannot prove full-volume dense equivalence", report)
        self.assertIn("Fine-tuning may change descriptor context sensitivity", report)

    def test_cli_writes_markdown_without_heatmap_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = make_run(root, "quadra_hc_021")
            output = root / "report.md"

            result = main(
                [
                    "--run-dir",
                    str(run_dir),
                    "--output",
                    str(output),
                    "--no-copy-heatmaps",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue(output.is_file())
            self.assertIn("## Technical summary", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
