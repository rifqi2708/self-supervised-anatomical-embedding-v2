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

Examples:

```bash
python tools/quadra/preprocess_quadra_zcrop.py --dry-run
python -m tools.quadra.precompute_quadra_embeddings --help
python tools/quadra/inc_cycle_error.py
python -m tools.quadra.inc_cycle_error_samv2
python tools/quadra/exc_cycle_error.py
python -m tools.quadra.streaming_cycle_error --help
```

The cycle scripts use configuration constants near the top of each file. Check
the dataset, checkpoint, output, point-sampling, and visualization settings
before starting a long run.

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

Defaults use exact `2.0 mm` isotropic preprocessing, `128×128×64` voxel input
tiles, a `32×32×16` halo, `64×64×32` native-grid matching chunks, and query
batches of 16. If the encoder tile does not fit the rented GPU, use the aligned
training-sized fallback:

```bash
python -m tools.quadra.streaming_cycle_error \
  --subject quadra_hc_021 \
  --num-points 100 \
  --tile-size 96 96 32 \
  --halo 32 32 8
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
  identifying columns required by `investigaton/full_registration_cycle_error.py`,
  so UAE and registration use the same query points and native output grid.
- `cycle_points_sam.csv` contains the complete cycle results in
  `coord_space=sam_display_voxel` for coordinate-transform and matching
  debugging. Its `mm_error` remains the physical error calculated through the
  original Test image geometry.

Before treating the fine-tuned run as a scientific result, validate tiled
inference against dense 2 mm inference on a smaller crop that fits in memory.
Compare correspondences and cycle error separately for tile interiors and tile
boundaries; also repeat the trial with a larger halo to test sensitivity to
lost encoder context. Global streamed matching removes a displacement-window
bias, but it does not by itself prove that tile-edge descriptors are equivalent
to dense inference.
