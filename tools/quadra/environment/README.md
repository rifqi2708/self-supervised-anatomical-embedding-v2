# Persistent Quadra RunPod environment

The Quadra research workflow uses one persistent `/workspace` volume with two
container images:

- preprocessing: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`;
- UAE-S: `sunyu0410/uae:py37torch19`.

The environments are deliberately separate. TotalSegmentator requires a modern
Python/PyTorch stack, while UAE-S depends on the released Python 3.7,
PyTorch 1.9 and MMCV/MMDetection stack.

## First bootstrap

From a checkout of this repository:

```bash
bash setup.sh bootstrap \
  --profile preprocess \
  --storage-root /workspace/quadra
```

After stopping the pod and switching to the UAE image:

```bash
bash setup.sh bootstrap \
  --profile uae \
  --storage-root /workspace/quadra
```

The bootstrap creates a persistent repository checkout at
`/workspace/repos/uae-quadra-validation`, records the exact commit and runtime
fingerprint, links the existing validated dataset and segmentation outputs when
they are available, copies the writable model cache, and refuses to overwrite
conflicting assets. The legacy virtual environment is not modified.

## Normal restart

After selecting the required container image:

```bash
source /workspace/quadra/runtime/activate.sh preprocess
# or
source /workspace/quadra/runtime/activate.sh uae
```

Activation sets the canonical dataset, model, output and cache paths and runs a
fast profile preflight. It never installs packages or downloads data.

Inspect the persistent state without activating a profile:

```bash
bash setup.sh status --storage-root /workspace/quadra
bash setup.sh verify-assets \
  --profile preprocess \
  --storage-root /workspace/quadra
```

## Scientific boundary

This setup prepares the software and persistent assets only. It does not select
the final organs, implement SuperPoint CT sampling, choose whole-volume versus
organ-group inference, or launch the 28-subject cohort. Full-body CT geometry
must be audited before the final 2 mm memory feasibility benchmark.
