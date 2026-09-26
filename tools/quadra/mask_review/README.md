# Quadra local anatomical mask review

This application reads the local NIfTI CT and mask files directly. It never
edits source images or masks and binds the browser server only to the local
loopback address. Review decisions are visual QA records, not manual
delineations or independent anatomical ground truth.

## Environment

Create the isolated environment outside the repository and generated-evidence
archive. For the current Mac layout:

```bash
bash tools/quadra/mask_review/setup_review_env.sh \
  "/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/.venvs/mask-review"
```

## Build the frozen review index

Use a new, empty review directory. Indexing validates all 96 scans and 3,790
mask files and records SHA-256 checksums, so it can take several minutes.

```bash
REVIEW_PYTHON="/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/.venvs/mask-review/bin/python"
DATASET_ROOT="/Users/rifqiab2708/Documents/quadra_dataset/QUADRA_HC_WB"
REVIEW_ROOT="/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/quadra/reviews/masks/quadra-hc-wb-totalsegmentator-2.16.0-organs-v1/REVIEW_ID"

"${REVIEW_PYTHON}" -m tools.quadra.mask_review build-index \
  --dataset-root "${DATASET_ROOT}" \
  --review-root "${REVIEW_ROOT}"
```

## Start or resume review

Run from the repository root so the `tools` package is importable:

```bash
"${REVIEW_PYTHON}" -m tools.quadra.mask_review serve \
  --review-root "${REVIEW_ROOT}" \
  --host 127.0.0.1 \
  --port 8501
```

Open `http://127.0.0.1:8501` if the browser does not open automatically. The
app autosaves each decision and reconstructs materialized CSV/JSON state from
the append-only event log after interruption. Each anatomical plane has First,
Middle, Last, Previous slice and Next slice controls in addition to its slider.
The default **Cropped** framing zooms all three planes to the organ's projected
bounding box plus 20 voxels of context. Select **Full body** to restore the
complete CT field of view. The dashed five-voxel mask boundary remains visible
inside the cropped frame.

Click inside any axial, coronal or sagittal image to select a canonical NIfTI
voxel. The other two slice controls update to the same 3D coordinate, and
plane-colored crosshair lines mark that coordinate in all three planes. The clickable
viewer is a bundled local component and does not send images or coordinates to
an external service.

Crosshair colors follow the conventional plane scheme: axial red, coronal
green and sagittal yellow. A compact legend is shown beneath the images.

Press **Right Arrow** while the page or a CT image has focus to mark the current
mask **Acceptable**, autosave it and advance to the next organ. The shortcut is
ignored while a slider, dropdown, radio option, button or reviewer-note field
has focus, so those controls retain their normal keyboard behavior. Holding the
key down does not create repeated decisions.

The sidebar's **Continue from saved progress** button opens the first pending
mask from the current autosaved state. A checksum checkpoint is an audit anchor,
not a rollback snapshot: if review decisions were saved after the checkpoint,
the application keeps those newer decisions rather than discarding them.

## Export a stable checkpoint

```bash
"${REVIEW_PYTHON}" -m tools.quadra.mask_review export \
  --review-root "${REVIEW_ROOT}"
```

The review directory contains the frozen source manifest, append-only event
history, current decisions, flagged queue, summary and timestamped checksum
checkpoints. It lives in the local archive but is not automatically an
independent second backup.
