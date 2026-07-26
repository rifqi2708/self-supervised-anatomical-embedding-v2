# Quadra TotalSegmentator workflow

This package prepares and runs a reproducible TotalSegmentator 2.16.0 workflow
for the Quadra whole-body CT evaluation cohort. It is isolated from the SAM/UAE
matching code.

## Scientific boundary

The default cohort is subjects `quadra_hc_021` through `quadra_hc_048`, with
Test and Retest CT scans. Subjects 001–020 are reserved for fine-tuning and are
rejected by the cohort planner. Sex comes only from the demographics workbook;
the prostate is requested for male subjects and is forbidden for female
outputs.

The cervical and thoracic spinal-cord masks are provisional derived labels.
They clip the same TotalSegmentator `spinal_cord` mask using physical
superior–inferior vertebral landmarks. This is an auditable engineering
definition, not validated anatomical ground truth.

## Local Stage 1 validation

From the repository root, create a lightweight environment without installing
TotalSegmentator or a GPU runtime:

```bash
python3 -m venv --system-site-packages .venv-totalseg-dev
source .venv-totalseg-dev/bin/activate
python -m pip install -r tools/quadra/totalsegmentator/requirements-dev.txt
python -m unittest discover -s tests -p 'test_quadra_totalsegmentator*.py'
```

Build the real cohort manifest locally. This reads and hashes all 56 CT files
but does not run segmentation:

```bash
python -m tools.quadra.totalsegmentator prepare \
  --dataset-root 'data/QUADRA_HC_WB' \
  --demographics 'data/Demographics (All).xlsx' \
  --output /tmp/quadra-totalsegmentator-manifest.json

python -m tools.quadra.totalsegmentator run-cohort \
  --manifest /tmp/quadra-totalsegmentator-manifest.json \
  --output-root /tmp/quadra-totalsegmentator-dry-run \
  --dry-run
```

Expected preparation summary:

- 28 subjects and 56 scans;
- 12 male and 16 female subjects;
- 40 masks per male scan and 39 per female scan;
- 2,208 masks in total.

## RunPod Stage 2 setup

Use the validated general PyTorch image:

```text
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

Clone the existing repository under `/workspace`, check out the dedicated
branch, and run:

```bash
git clone https://github.com/rifqi2708/self-supervised-anatomical-embedding-v2.git
cd self-supervised-anatomical-embedding-v2
git switch codex/quadra-totalsegmentator

cp tools/quadra/totalsegmentator/.env.example \
  tools/quadra/totalsegmentator/.env
# Edit .env and supply QUADRA_DATASET_URL and QUADRA_DEMOGRAPHICS_URL.

bash tools/quadra/totalsegmentator/setup.sh
source /workspace/quadra-totalsegmentator/venv/bin/activate
export TOTALSEG_WEIGHTS_PATH=/workspace/quadra-totalsegmentator/model-cache

python -m tools.quadra.totalsegmentator preflight \
  --dataset-root /workspace/quadra-totalsegmentator/data/QUADRA_HC_WB \
  --output-root /workspace/quadra-totalsegmentator/outputs
```

`preflight` must confirm TotalSegmentator `2.16.0`, CUDA, free storage, and
every required class in the `total` and `head_glands_cavities` tasks. Do not
start inference if it fails.

If the RunPod web terminal is unavailable, install the VS Code CLI during
Stage 2 and start the authenticated tunnel:

```bash
bash tools/quadra/totalsegmentator/start_vscode_tunnel.sh
```

## Stage 3 smoke test

After preflight passes, run exactly one male scan first so the smoke test
exercises the prostate route:

```bash
python -m tools.quadra.totalsegmentator run-scan \
  --manifest /workspace/quadra-totalsegmentator/metadata/cohort_manifest.json \
  --subject quadra_hc_022 \
  --session test

python -m tools.quadra.totalsegmentator validate \
  --manifest /workspace/quadra-totalsegmentator/metadata/cohort_manifest.json \
  --output-root /workspace/quadra-totalsegmentator/outputs \
  --subject quadra_hc_022 \
  --session test
```

The smoke test is the first actual GPU inference. Local Stage 1 tests validate
only orchestration and synthetic NIfTI handling.

With TotalSegmentator 2.16.0, `--roi_subset` is used only for the `total` task.
The `head_glands_cavities` task does not support that option, so it runs its
full task and the workflow promotes only the six registry-selected head masks.

## Stage 4 full cohort

After the smoke test has been accepted:

```bash
python -m tools.quadra.totalsegmentator run-cohort \
  --manifest /workspace/quadra-totalsegmentator/metadata/cohort_manifest.json \
  --output-root /workspace/quadra-totalsegmentator/outputs \
  --scratch-root /tmp/quadra-totalsegmentator \
  --min-free-gib 20
```

The runner continues after an ordinary scan failure, stops when persistent
storage falls below the guard, and skips only outputs that pass compatibility
and technical QC.

Generate status artifacts with:

```bash
python -m tools.quadra.totalsegmentator status \
  --manifest /workspace/quadra-totalsegmentator/metadata/cohort_manifest.json \
  --output-root /workspace/quadra-totalsegmentator/outputs \
  --json-output /workspace/quadra-totalsegmentator/logs/cohort-status.json \
  --csv-output /workspace/quadra-totalsegmentator/logs/cohort-status.csv
```

Technical QC checks readability, binary values, non-empty masks, geometry and
sex routing. It does not establish anatomical accuracy; Stage 5 must include
visual/anatomical review.
