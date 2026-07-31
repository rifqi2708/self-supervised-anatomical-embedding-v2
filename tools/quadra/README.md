# Quadra tools

This package contains the project-specific Quadra preprocessing, embedding,
cycle-error, coordinate-conversion, and analysis workflows. Run commands from
the repository root. Both direct script paths and Python module invocation are
supported.

## Pipeline

| Script | Purpose | Main defaults |
| --- | --- | --- |
| `preprocess_quadra_zcrop.py` | Crop image and mask volumes along Z using the union of masks. | `data/quadra_dataset` to `data/quadra_dataset_cropped` |
| `precompute_quadra_embeddings.py` | Generate indexed SAM embeddings for cropped Quadra images. | `checkpoints/SAM.pth`; outputs under `data/quadra_dataset_cropped_embeddings` |
| `inc_cycle_error.py` | Compute SAM cycle error while generating embeddings in-process. | Fine-tuned SAM checkpoint and `data/quadra_dataset_cropped` |
| `inc_cycle_error_samv2.py` | Compute cycle error with the SAMv2/UAE-S semantic branch. | `checkpoints/SAMv2_iter_20000.pth` |
| `exc_cycle_error.py` | Compute cycle error from precomputed embeddings. | Quadra cropped dataset and embedding index |
| `rd_cycle_error.py` | Run the paired-image precomputed-embedding cycle workflow. | Quadra male cropped dataset paths |
| `streaming_cycle_error.py` | Run 2 mm tiled UAE embeddings with exhaustive, memory-bounded global matching. | `quadra_hc_021`; `checkpoints/SAM.pth`; 100 points per mask |
| `streaming_cycle_error_uaes.py` | Compare UAE-S semantic global-NN and fixed-point cycle matching with resumable per-organ progress. | `quadra_hc_021`; `checkpoints/SAMv2_iter_20000.pth`; both matching modes |
| `environment.py` | Bootstrap and verify persistent preprocessing and UAE-S RunPod profiles. | `/workspace/quadra`; see `tools/quadra/environment/README.md` |
| `optimization_baseline.py` | Capture the immutable Stage 0 contract for full-body UAE-S memory optimization. | 2 mm; UAE-S checkpoint; both matching modes; no CT/model computation |
| `body_envelope_audit.py` | Audit conservative air removal and freeze one reviewed body-envelope crop configuration. | 56 scans; XY/XYZ; 0–60 mm margins; no cropped CT or UAE-S inference |
| `coordinate_preserving_crop.py` | Realize and validate the frozen Stage 1 body crop, 2 mm grid, stride padding, normalization, and inverse coordinates. | `xy_m010`; largest Test/Retest pair only; CPU; no saved 3D volumes |
| `memory_configuration_screen.py` | Screen dense UAE-S embedding-extraction memory on the largest whole-body, body-envelope, and organ-group plans. | FP32/AMP; conditional full FP16; fresh workers; no saved embeddings |
| `streaming_cycle_error_cohort.py` | Run a resumable sequential 2 mm streaming cohort in isolated subject subprocesses. | Inclusive `quadra_hc_021`–`quadra_hc_048`; 20 GB disk guard |
| `validate_streaming_equivalence.py` | Compare dense and tiled crop inference, verify streamed global matching, and test full-subject halo sensitivity. | `quadra_hc_021`; baseline `128×128×64`, expanded `160×160×80` tiles |
| `summarize_streaming_validation.py` | Validate compatible per-subject runs and produce a cross-subject technical Markdown report. | Explicit repeated `--run-dir` inputs |
| `totalsegmentator/` | Prepare, run, resume, and technically validate whole-body organ segmentation on RunPod. | Subjects 021–048; TotalSegmentator 2.16.0 |

Examples:

```bash
python tools/quadra/preprocess_quadra_zcrop.py --dry-run
python -m tools.quadra.precompute_quadra_embeddings --help
python tools/quadra/inc_cycle_error.py
python -m tools.quadra.inc_cycle_error_samv2
python tools/quadra/exc_cycle_error.py
python -m tools.quadra.streaming_cycle_error --help
python -m tools.quadra.streaming_cycle_error_uaes --help
python -m tools.quadra.optimization_baseline --help
python -m tools.quadra.body_envelope_audit --help
python -m tools.quadra.body_envelope_audit audit --help
python -m tools.quadra.body_envelope_audit select --help
python -m tools.quadra.coordinate_preserving_crop validate --help
python -m tools.quadra.memory_configuration_screen --help
python -m tools.quadra.streaming_cycle_error_cohort --help
python -m tools.quadra.validate_streaming_equivalence --help
python -m tools.quadra.summarize_streaming_validation --help
python -m tools.quadra.totalsegmentator --help
```

The isolated TotalSegmentator setup, cohort manifest, sex-specific prostate
routing, smoke-test command, and RunPod instructions are documented in
[`totalsegmentator/README.md`](totalsegmentator/README.md).

The cycle scripts use configuration constants near the top of each file. Check
the dataset, checkpoint, output, point-sampling, and visualization settings
before starting a long run.

## UAE-S memory optimization contract

Before running the staged full-body memory investigation, activate the
preprocessing profile and capture Stage 0:

```bash
source /workspace/quadra/runtime/activate.sh preprocess

python -m tools.quadra.optimization_baseline \
  --storage-root /workspace/quadra \
  --output-root /workspace/quadra/runs/memory_optimization
```

The command performs only Git, path, count, checksum, environment, and GPU
inspection. It does not decompress CT images, load UAE-S, generate embeddings,
or run cycle-error analysis. The resulting `baseline_manifest.json` freezes the
checkpoint, configuration, 2 mm spacing, seed, coordinate convention, matching
scope, precision candidates, and memory-measurement policy. Every later
optimization stage must call `validate_locked_contract` before reading CT data
or loading the model.

### Stage 1 body-envelope audit

Stage 1 reads the accepted 56 CT/mask sets sequentially and compares XY-only
and XYZ body-envelope crops at 0, 10, 20, 30, 40, and 60 mm margins. It uses a
conservative `HU > -800` envelope, discards only connected components smaller
than 10 mL, calculates the 2 mm stride-padded geometry, and checks all 2,208
expected masks for clipping and artificial-boundary clearance. It does not
write cropped images, resample CT data, load UAE-S, or use CUDA.

Run the audit with the accepted Stage 0 manifest:

```bash
python -m tools.quadra.body_envelope_audit audit \
  --baseline-manifest /workspace/quadra/runs/memory_optimization/stage0-20260731T085944Z/baseline_manifest.json \
  --storage-root /workspace/quadra
```

The audit is resumable with `--resume-run-directory`. Its recommendation is
not automatically frozen. After reviewing `candidate_summary.csv`,
`mask_clearance.csv`, the Markdown report, and the QC montage, explicitly
select one eligible candidate:

```bash
python -m tools.quadra.body_envelope_audit select \
  --audit-run-directory "$STAGE1_RUN_DIR" \
  --candidate-id "$REVIEWED_CANDIDATE_ID" \
  --review-rationale "Reviewed candidate tables and QC overlays." \
  --storage-root /workspace/quadra
```

Selection writes `selected_body_envelope.json` and the Stage 1
`checkpoint_summary.json`. Later stages must consume the selected manifest and
must not recalculate crop bounds independently.

### Stage 2 coordinate-preserving crop

Stage 2 consumes the reviewed `xy_m010` scan plans without redetecting the
body. It validates only the frozen largest pair, `quadra_hc_044`, processing
Test and Retest sequentially on CPU. Each scan is cropped with half-open raw
ITK bounds, resampled to its exact planned 2 mm grid, symmetrically padded to
the `(16,16,4)` XYZ model stride, and normalized with the existing UAE CT
rule. The reusable `prepare_scan_from_plan` interface returns a float32 `ZYX`
array and continuous raw-XYZ/model-XYZ transforms.

```bash
python -m tools.quadra.coordinate_preserving_crop validate \
  --baseline-manifest /workspace/quadra/runs/memory_optimization/stage0-20260731T085944Z/baseline_manifest.json \
  --stage1-checkpoint /workspace/quadra/runs/memory_optimization/stage1-audit-20260731T110726Z/checkpoint_summary.json \
  --storage-root /workspace/quadra
```

The command retains only compact manifests, CSV tables, a Markdown report,
and QC PNGs under `runs/memory_optimization/stage2-crop-<UTC>/`. It does not
load UAE-S or CUDA, and it does not retain prepared NIfTI or NumPy volumes.
Use `--resume-run-directory` only for an interrupted run; a completed Stage 2
run is immutable. Stage 3 must reuse the frozen preparation API and geometry
rather than recalculating crop bounds or preprocessing settings.

### Stage 3 largest-case UAE-S memory screen

Stage 3 first derives the largest whole-body, frozen `xy_m010`, and proposed
four-region organ-group plans from accepted Stage 1 evidence. The preprocessing
phase realizes those three plans sequentially with the Stage 2 API and retains
only tables and QC images:

```bash
python -m tools.quadra.memory_configuration_screen prepare \
  --baseline-manifest /workspace/quadra/runs/memory_optimization/stage0-20260731T085944Z/baseline_manifest.json \
  --stage1-checkpoint /workspace/quadra/runs/memory_optimization/stage1-audit-20260731T110726Z/checkpoint_summary.json \
  --stage2-checkpoint /workspace/quadra/runs/memory_optimization/stage2-crop-20260731T144720Z/checkpoint_summary.json \
  --storage-root /workspace/quadra
```

After switching to and activating the pinned UAE profile, run the bounded
precision smoke and sequential fresh-process benchmarks, then apply the frozen
ranking:

```bash
python -m tools.quadra.memory_configuration_screen benchmark \
  --run-directory "$STAGE3_RUN_DIR"

python -m tools.quadra.memory_configuration_screen select \
  --run-directory "$STAGE3_RUN_DIR"
```

The loader removes the configuration's training-time FP16 hook in memory; it
does not modify the config file. FP32 uses FP32 weights/input without autocast,
AMP uses FP32 weights/input with autocast, and full FP16 is attempted only when
AMP fails specifically from CUDA OOM. UAE-S still explicitly returns FP16
fine, coarse, and semantic embeddings in every mode. The screen measures dense
embedding extraction only: its preferred/fallback selection is provisional and
does not establish matching feasibility or numerical equivalence.

## Analysis and coordinate utilities

| Script | Purpose |
| --- | --- |
| `analyze_cycle_error_cases.py` | Generate post-run summaries and case visualizations from a cycle-error CSV. |
| `plot_cycle_error_jitter.py` | Plot organ-wise cycle-error distributions. |
| `quadra_one_cycle_visual.py` | Inspect one complete matching cycle with embeddings and similarity maps. |
| `reexport_sam_csv_raw_coords.py` | Convert SAM-display voxel coordinates in a CSV to raw ITK voxel coordinates. |
| `debug_sam_mask_coords.py` | Check stored points against SAM-oriented and raw masks. |

`coord_space_utils.py`, `embedding_cache.py`, and
`rd_cycle_error_helper.py` are supporting modules rather than primary
entrypoints.

Useful command discovery:

```bash
python -m tools.quadra.reexport_sam_csv_raw_coords --help
python -m tools.quadra.debug_sam_mask_coords --help
python -m tools.quadra.plot_cycle_error_jitter --help
```

Cycle outputs belong under `data/quadra_output/`; reusable demo and reference
files are under `tools/assets/`.

## 2 mm streaming trial on RunPod

`streaming_cycle_error.py` separates encoder tiling from the matching search
space. The encoder keeps the central region of overlapping tiles, while the
matcher streams every target location and retains the global maximum. Matching
chunks therefore limit memory without imposing a maximum anatomical
displacement.

The repository does not track checkpoints or Quadra data. After cloning the
branch on RunPod, separately place:

- the original SAM checkpoint at `checkpoints/SAM.pth`;
- the Test, Retest, and five mask volumes for `quadra_hc_021` under
  `data/quadra_dataset_cropped/` using the existing layout; and
- the CUDA environment from `requirements.txt`.

Run the selected full-subject engineering trial from the repository root:

```bash
python -m tools.quadra.streaming_cycle_error \
  --subject quadra_hc_021 \
  --num-points 100
```

Defaults use exact `2.0 mm` isotropic preprocessing, the validated expanded
`160×160×80` voxel encoder tile, a `48×48×24` halo on each side, the unchanged
`64×64×32` retained core, `64×64×32` native-grid matching chunks, and query
batches of 16. Embedding caches are namespaced by tile, halo, and retained-core
geometry, so the expanded run cannot accidentally reuse a baseline cache.

Successful embedding caches are disposable by default. The command first
validates the raw-ITK and SAM result rows, registration queries, summary and
manifest; closes both memory maps; and then removes only the canonical cache
directory for that subject. Embedded Test/Retest cache manifests remain in the
run manifest for provenance. Failed runs retain their caches. Use
`--keep-cache` when the arrays are intentionally needed after a successful run.
A cleanup failure preserves the result files, is recorded in the manifest and
returns a distinct nonzero status.

The five-subject engineering validation supporting this default is documented
in [`reports/quadra/streaming_tile_validation_quadrahc021_025.md`](../../reports/quadra/streaming_tile_validation_quadrahc021_025.md).
If the expanded encoder tile does not fit a different GPU, the previously
tested baseline can still be requested explicitly:

```bash
python -m tools.quadra.streaming_cycle_error \
  --subject quadra_hc_021 \
  --num-points 100 \
  --tile-size 128 128 64 \
  --halo 32 32 16
```

The original `SAM.pth` run is labelled `engineering_trial` in
`run_manifest.json`; it must not be reported as a result from the Quadra
fine-tuned model. Later, provide the fine-tuned checkpoint explicitly with
`--checkpoint-file`. Cache paths include the checkpoint hash, so embeddings
from different checkpoints cannot be mixed. Embeddings are stored as FP16, but
similarities and interpolation are evaluated in FP32 for stable streamed
argmax selection.

Outputs are written to a timestamped directory under
`data/quadra_output/streaming_cycle_error/`:

- `cycle_points.csv` contains the complete cycle results in
  `coord_space=raw_itk_voxel`. Cycle error in millimetres is calculated from
  the original Test image's ITK physical coordinates.
- `query_points_raw_itk.csv` contains only the original Test query point and
  identifying columns required by `tools/quadra/registration_cycle_error.py`,
  so UAE and registration use the same query points and native output grid.
- `cycle_points_sam.csv` contains the complete cycle results in
  `coord_space=sam_display_voxel` for coordinate-transform and matching
  debugging. Its `mm_error` remains the physical error calculated through the
  original Test image geometry.

`run_manifest.json` records `cache_policy` and `cache_cleanup`, including the
subject cache target, cleanup status, bytes measured/freed, deleted paths,
completion time and any error.

## Quadra 021–048 cohort

Run the production range sequentially so only one subject cache needs disk
space at a time:

```bash
python -m tools.quadra.streaming_cycle_error_cohort \
  --subject-start 21 \
  --subject-end 48
```

The runner checks free disk before every subject, saves one terminal log per
subject, and writes an incrementally updated batch manifest under
`data/quadra_output/streaming_cycle_error_batches/`. It skips only an existing
completed run whose checkpoint/config hashes, spacing, tile geometry, matching,
sampling, organs, dataset and MRI setting match. This remains resumable after
successful embeddings have been deleted. Ordinary subject failures are logged
and the cohort continues; a cache-cleanup failure or disk space below
`--min-free-gb` stops the batch immediately.

Inspect all 28 planned commands without launching model inference:

```bash
python -m tools.quadra.streaming_cycle_error_cohort --dry-run
```

Use `--rerun-completed` to force new results and `--keep-cache` to forward the
explicit cache-retention policy to every subject.

### UAE-S paired matching

The UAE-S runner caches fine, coarse, and semantic embeddings once and gives
the same frozen queries to two unrestricted matching methods. `global_nn`
averages the three FP32 similarities and selects the native-grid global
maximum. `fixed_point` follows the released UAE-S structural inference: it
iteratively matches a fine-grid neighbourhood in both directions, retains
stable anchors, and fits a robust local affine model. Its neighbourhood chooses
context anchors; it does not restrict the target search region.
The UAE-S profile batches 256 query descriptors by default on the target A6000;
`--query-batch-size` changes memory scheduling only, not the search space or
similarity formula.

Run the staged subject trial with:

```bash
python -m tools.quadra.streaming_cycle_error_uaes \
  --subject quadra_hc_021 \
  --matching-modes global_nn fixed_point \
  --num-points 5 \
  --keep-cache
```

Progress is checkpointed by organ under the timestamped run directory. A
compatible interrupted run resumes automatically. Fixed-point queries with too
few stable anchors or degenerate affine geometry remain in the output with an
explicit failure status and blank match/error values; they are never replaced
silently by global NN. Successful caches are deleted only after all requested
method outputs validate.

Prepare the 28-subject production command without launching it:

```bash
python -m tools.quadra.streaming_cycle_error_cohort \
  --model-profile uae_s \
  --matching-modes global_nn fixed_point \
  --subject-start 21 \
  --subject-end 48 \
  --num-points 100 \
  --dry-run
```

Fixed-point matching uses forward-backward consistency internally. Its cycle
error therefore measures self-consistency and must not be interpreted alone as
independent anatomical matching accuracy.

Validate the UAE-S implementation itself on bounded subject-021 crops with:

```bash
python -m tools.quadra.validate_uaes_streaming \
  --subject quadra_hc_021
```

This compares dense and expanded-tile fine, coarse and semantic descriptors;
checks dense versus streamed unrestricted global argmax coordinates; and runs
the fixed-point iterations through both dense and streamed matchers. Results,
including internal-match hashes and semantic discrepancy heatmaps, are written
under `data/quadra_output/uaes_streaming_validation/`.

Before treating the fine-tuned run as a scientific result, validate tiled
inference against dense 2 mm inference on a smaller crop that fits in memory.
Compare correspondences and cycle error separately for tile interiors and tile
boundaries; also repeat the trial with a larger halo to test sensitivity to
lost encoder context. Global streamed matching removes a displacement-window
bias, but it does not by itself prove that tile-edge descriptors are equivalent
to dense inference.

## Streaming-equivalence validation

Run the complete engineering validation on RunPod after the 2 mm streaming
trial environment is configured:

```bash
python -m tools.quadra.validate_streaming_equivalence \
  --subject quadra_hc_021 \
  --checkpoint-file checkpoints/SAM.pth \
  --num-points 100
```

The `crop` phase uses deterministic `128×128×64` organ-centred crops to compare
dense embeddings and dense matching with the deployed tiled/streamed pipeline.
The `full` phase still covers the complete subject and compares the normal
`128×128×64` tile with `32×32×16` halo against a `160×160×80` tile with
`48×48×24` halo. Both retain the same `64×64×32` core, so the comparison changes
encoder context without changing output coverage or the global search space.

Results are written under `data/quadra_output/streaming_validation/`. Use
`--phases crop` for the bounded dense reference, `--phases full` for the
full-subject halo test, and `--run-dir <existing-output>` to resume into the
same result directory. Caches for the two tile plans are kept in separate
namespaces. If the expanded full-subject tile exceeds GPU memory, the command
records the OOM and retains the expanded-context organ-crop comparison; it does
not silently reduce the halo.

The generated `validation_report.md` and `validation_summary.json` apply
pre-specified engineering thresholds. Review the continuous descriptor and
correspondence CSVs and discrepancy heatmaps before accepting the status. A run
with the original `SAM.pth` remains an engineering check and must be repeated
with the Quadra fine-tuned checkpoint before scientific reporting.

After completing multiple subjects, create a cross-subject report from explicit
run directories so similarly named or incomplete runs cannot be selected
silently:

```bash
python -m tools.quadra.summarize_streaming_validation \
  --run-dir data/quadra_output/streaming_validation/quadra_hc_021_<timestamp> \
  --run-dir data/quadra_output/streaming_validation/quadra_hc_022_<timestamp> \
  --run-dir data/quadra_output/streaming_validation/quadra_hc_023_<timestamp> \
  --run-dir data/quadra_output/streaming_validation/quadra_hc_024_<timestamp> \
  --run-dir data/quadra_output/streaming_validation/quadra_hc_025_<timestamp> \
  --validation-commit <validator-source-commit> \
  --output reports/quadra/streaming_tile_validation_quadrahc021_025.md
```

The summarizer rejects mixed checkpoint hashes, configurations, spacing,
sampling settings, tile plans, incomplete row counts, and mixed correspondence
schemas. Because version-1 validation manifests do not store a Git commit, pass
the commit that last changed the validator explicitly. The command pools raw
correspondence rows while retaining subject- and organ-level tables, and copies
only the worst crop and full-subject discrepancy heatmaps next to the report.
