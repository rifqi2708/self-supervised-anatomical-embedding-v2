# Start a disposable Quadra pod

Use this checklist for a new UAE-S or registration pod. Disposable pods have no
persistent volume: stopping or terminating the pod destroys `/workspace`.

## 1. Before deployment

1. Confirm that these recovery packages are available through their recorded
   Google Drive IDs in `configs/quadra/disposable-assets-v1.json`:
   whole-body CTs, Stage 5 masks, the frozen experiment contract, and (for UAE)
   the UAE model package.
2. Temporarily enable link-readable **viewer** access only for packages that are
   still private. Revoke it after every required pod has downloaded the files.
3. Use the saved RunPod template matching the task:

   - `quadra-uae-disposable-v1`: Secure Cloud, 48 GB NVIDIA GPU, 100 GB
     container disk, no pod/network volume.
   - `quadra-registration-disposable-v1`: Secure Cloud CPU, at least 6 vCPUs
     and 32 GB RAM, 100 GB container disk, no pod/network volume.

Do not attach a volume and do not reuse or migrate a stale pod.

## 2. Clone the immutable release

Open the pod terminal and run:

```bash
mkdir -p /workspace/repos

git clone \
  --branch quadra-disposable-v1 \
  --single-branch \
  https://github.com/rifqi2708/self-supervised-anatomical-embedding-v2.git \
  /workspace/repos/uae-quadra-validation

cd /workspace/repos/uae-quadra-validation
git status --short --branch
git rev-parse HEAD
```

For pre-release acceptance only, replace `quadra-disposable-v1` with the exact
reviewed branch or commit named in the acceptance record.

## 3. Bootstrap the selected profile

### UAE-S GPU pod

```bash
cd /workspace/repos/uae-quadra-validation

bash setup.sh disposable-bootstrap \
  --profile uae \
  --storage-root /workspace/quadra \
  --repository-ref quadra-disposable-v1 \
  --image-ref sunyu0410/uae:py37torch19 \
  --confirm-image-digest sha256:2c0edd4a205c3c5d9d027b6c9f96f83626eb2cc3810da7876e32d4bf36653d61

source /workspace/quadra/runtime/activate.sh uae
bash setup.sh disposable-status --profile uae --storage-root /workspace/quadra
```

### Registration CPU pod

```bash
cd /workspace/repos/uae-quadra-validation

bash setup.sh disposable-bootstrap \
  --profile registration \
  --storage-root /workspace/quadra \
  --repository-ref quadra-disposable-v1 \
  --image-ref runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --confirm-image-digest sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5

source /workspace/quadra/runtime/activate.sh registration
bash setup.sh disposable-status --profile registration --storage-root /workspace/quadra
```

Bootstrap verifies downloads, geometry, runtime identity and a bounded technical
smoke test. It does not launch segmentation, SuperPoint, or a scientific cohort.

## 4. Before stopping or terminating

1. Finish or intentionally interrupt the work at a documented stable point.
2. Package every completed or incomplete run:

   ```bash
   bash setup.sh disposable-package-results \
     --storage-root /workspace/quadra \
     --run-directory /workspace/quadra/runs/WORKFLOW/RUN_ID \
     --transfer-id UNIQUE_TRANSFER_ID
   ```

3. Pull the evidence to the Mac archive and verify checksums using
   `tools/quadra/environment/ARTIFACT_BACKUP.md`.
4. Revoke temporary Google Drive link access and record the revocation
   attestation.
5. Run `bash setup.sh safe-terminate-check ...` from the Mac using the pod's
   current SSH host and port.
6. Stop or terminate only after the command returns `SAFE_TO_TERMINATE` and the
   user separately authorizes that action.

The canonical Mac evidence archive is:

```text
/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/quadra
```

The Mac is currently the sole generated-evidence recovery location. Keep the
Google Drive input packages as independent recovery sources.
