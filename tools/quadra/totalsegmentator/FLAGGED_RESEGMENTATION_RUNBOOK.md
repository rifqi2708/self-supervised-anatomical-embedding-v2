# Flagged-mask re-segmentation runbook

This runbook is the handoff for the reviewed Quadra mask follow-up. It prepares
and runs only the frozen cases selected from the completed local mask review. It
does not authorize a full-cohort rerun, changing accepted masks, promoting pilot
outputs, stopping a pod, or terminating a pod.

## Frozen scope and gates

The execution manifest contains 29 cases:

- 6 sacrum pilot scans: subjects 021, 029, and 043, test and retest;
- 6 refined-vertebra pilot scans: subjects 006, 012, and 026, test and retest;
- 6 rib scans covering the 7 flagged rib-six masks;
- 7 scans covering 10 other flagged organs;
- 4 isolated-voxel corrections, performed without GPU inference.

The sacrum and vertebra batch expansions are deliberately absent from the
executable manifest. Each requires a separate manual anatomical PASS and fresh
user authorization. A technically valid NIfTI output is not an anatomical PASS.

The source CTs, original masks, completed review, and pilot outputs are
immutable inputs. Outputs are new derivatives under a versioned run directory.

## Instruction for the agent guiding the RunPod session

Use the following as the opening instruction in the future setup task:

> Guide me through the Quadra flagged-mask re-segmentation runbook one stage at
> a time. Reuse the repository's `totalseg` profile and frozen execution
> manifest. At every stage, show the concrete evidence that the stage passed and
> wait for my confirmation before the next stage. Do not start GPU inference,
> run another phase, promote or replace masks, expand either pilot, stop a pod,
> or terminate a pod without my explicit confirmation. Never print, save in Git,
> or place the TotalSegmentator license token in a command history, log, or chat.
> Treat technical QC and manual anatomical review as separate gates. Preserve
> all original CTs, masks, review files, failed work directories, logs, and run
> manifests. Before any stop recommendation, follow the repository backup and
> safe-stop procedure and report uncovered artifacts honestly.

The guide should pause at all checkpoints marked **CONFIRM** below.

## 1. Local preparation — no GPU

Run these commands from the repository root on the Mac. The intentional space
before `quadra-local-storage` means every path must stay quoted.

```bash
cd "/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /self-supervised-anatomical-embedding-v2"

QUADRA_REVIEW_ROOT="/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/quadra/reviews/masks/quadra-hc-wb-totalsegmentator-2.16.0-organs-v1/mask-review-20260918T004401Z"
QUADRA_PLAN_ROOT="/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/quadra/reviews/masks/quadra-hc-wb-totalsegmentator-2.16.0-organs-v1/mask-resegmentation-plan-20260926"
QUADRA_EXECUTION_MANIFEST="${QUADRA_PLAN_ROOT}/execution_manifest.json"
QUADRA_LOCAL_DATASET="/Users/rifqiab2708/Documents/quadra_dataset/QUADRA_HC_WB"
QUADRA_UPLOAD_TREE="${QUADRA_PLAN_ROOT}/upload/whole_body_ct_v1"

python3 -m tools.quadra.totalsegmentator reseg-prepare \
  --review-root "${QUADRA_REVIEW_ROOT}" \
  --output "${QUADRA_EXECUTION_MANIFEST}"

python3 -m tools.quadra.totalsegmentator reseg-preflight \
  --manifest "${QUADRA_EXECUTION_MANIFEST}" \
  --dataset-root "${QUADRA_LOCAL_DATASET}" \
  --output-root "${QUADRA_PLAN_ROOT}" \
  --skip-runtime
```

Expected manifest summary: 29 cases, 25 segmentation cases, 4 corrections,
and 39 requested review masks. Preparation hashes the reviewed decisions, CTs,
and correction-source masks and refuses changed sources.

Only when an upload package is needed, stage the 25 unique CTs (approximately
2.91 GiB in the current frozen manifest):

```bash
python3 -m tools.quadra.totalsegmentator reseg-stage-inputs \
  --manifest "${QUADRA_EXECUTION_MANIFEST}" \
  --dataset-root "${QUADRA_LOCAL_DATASET}" \
  --destination "${QUADRA_UPLOAD_TREE}"
```

The staging command copies only selected CTs, preserves the manifest-relative
paths, verifies hashes, and refuses conflicting files. It does not copy the
whole cohort.

**CONFIRM:** Verify the manifest summary and available local disk space before
creating the 2.91-GiB staging copy.

## 2. Select and deploy the RunPod GPU

Use this pinned container image:

```text
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

The lowest-uncertainty choice is the same class used in the prior Quadra GPU
work: one NVIDIA RTX A6000 with about 48 GiB VRAM. A smaller GPU may be adequate
for TotalSegmentator, but this workflow has not yet measured peak memory for the
refined-vertebra task, so do not trade down merely on assumption. The guiding
agent must verify current RunPod availability, pricing, volume attachment, and
CUDA visibility in the live console; this file does not freeze those changing
facts.

Attach a persistent volume with at least 50 GiB free. The volume must survive
pod stops. Do not use a disposable container filesystem for the environment,
weights, license configuration, inputs, or outputs.

**CONFIRM:** Before deployment, show the selected GPU, VRAM, image, persistent
volume mount, free-space estimate, and pricing visible in the current console.

## 3. Make the current repository code available on the pod

The runner is new code. The pod must receive the exact reviewed Git commit or a
checksum-verified code package containing these changes. Do not assume an old
pod checkout contains it. Uncommitted local changes are not recoverable from
Git until the user explicitly authorizes and completes a commit/push or another
verified transfer.

The repository should be present at:

```text
/workspace/repos/uae-quadra-validation
```

Record `git status`, the current branch, and `git rev-parse HEAD`. If the pod
checkout is dirty or differs from the chosen source, stop and reconcile it; do
not overwrite research files.

**CONFIRM:** Show repository status and the exact source revision/package
receipt before installing anything.

## 4. Bootstrap the minimal persistent TotalSegmentator profile

From the current repository checkout on the pod:

```bash
cd /workspace/repos/uae-quadra-validation

bash setup.sh bootstrap \
  --profile totalseg \
  --storage-root /workspace/quadra

source /workspace/quadra/runtime/activate.sh totalseg
```

This creates or reuses:

- virtual environment: `/workspace/quadra/runtime/preprocess-venv`;
- TotalSegmentator home/config: `/workspace/quadra/runtime/totalsegmentator-home`;
- weights cache: `/workspace/quadra/cache/totalsegmentator`;
- selected CT root: `/workspace/quadra/datasets/source/whole_body_ct_v1`;
- run root: `/workspace/quadra/runs/preprocessing`.

The requirements pin `TotalSegmentator==2.16.0`. Activation exports both
`TOTALSEG_HOME_DIR` and `TOTALSEG_WEIGHTS_PATH`, then runs a profile preflight.
It does not install the unrelated UAE-S or SuperPoint stack.

Check the runtime explicitly:

```bash
python --version
python -m pip show TotalSegmentator
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
bash setup.sh verify-assets --profile totalseg --storage-root /workspace/quadra
```

**CONFIRM:** Python, TotalSegmentator 2.16.0, CUDA, persistent paths, and asset
verification must pass before license or weight setup.

## 5. Set the academic license without exposing it

`vertebrae_pp_refined` is a licensed task. Obtain the user's own academic
license from the official TotalSegmentator process. Do not put it in a tracked
file, notebook, manifest, chat message, or command copied into the runbook.

The user should enter the token privately in the pod terminal using a method
that does not echo or preserve it in shell history, then run:

```bash
read -r -s -p "TotalSegmentator license: " QUADRA_TOTALSEG_LICENSE
printf '\n'
totalseg_set_license -l "${QUADRA_TOTALSEG_LICENSE}"
unset QUADRA_TOTALSEG_LICENSE
```

The license configuration persists under `TOTALSEG_HOME_DIR`. The agent should
verify only success/failure and file presence; never display the token or the
contents of `config.json`.

Official references:

- <https://github.com/wasserth/TotalSegmentator>
- <https://backend.totalsegmentator.com/license-academic/>

**CONFIRM:** Report that the license was accepted without revealing it.

## 6. Download and verify only the required model weights

With the persistent profile active:

```bash
totalseg_download_weights -t total
totalseg_download_weights -t vertebrae_pp_refined
```

The refined task can download dependent vertebra models. Do not infer success
only from a zero-byte directory or a partial download. Keep the network enabled
until the commands and a subsequent preflight pass. Do not enable fast or
low-resolution modes; the frozen runner never adds those flags.

**CONFIRM:** Show command exit status and non-secret cache inventory, then
proceed only if both task capabilities are available.

## 7. Upload the frozen manifest and selected CTs

Transfer the locally staged tree to:

```text
/workspace/quadra/datasets/source/whole_body_ct_v1
```

Transfer the execution manifest to:

```text
/workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json
```

For the four correction cases, the manifest also refers to source masks from
the reviewed local dataset. Corrections should normally be executed locally,
where those reviewed source masks already exist. Do not upload all masks merely
to make GPU inference possible.

After transfer:

```bash
source /workspace/quadra/runtime/activate.sh totalseg

python -m tools.quadra.totalsegmentator reseg-preflight \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --min-free-gib 20
```

Preflight checks exact input hashes, free storage, TotalSegmentator 2.16.0,
required task classes, and CUDA. A failed preflight blocks inference.

**CONFIRM:** Show the complete preflight summary. Do not start inference yet.

## 8. Dry-run, then one sacrum smoke case

First print the commands without running them:

```bash
python -m tools.quadra.totalsegmentator reseg-run-phase \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --phase sacrum_pilot \
  --dry-run
```

Confirm that every sacrum command uses `-ta total`, `--roi_subset sacrum`, and
`--robust_crop`, and contains no fast flag. Then, only after authorization, run
one case:

```bash
python -m tools.quadra.totalsegmentator reseg-run-case \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --case-id sacrum_pilot--quadra_hc_029--test

python -m tools.quadra.totalsegmentator reseg-validate \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --phase sacrum_pilot
```

The case is published atomically only after technical QC. Failure directories
and logs remain available for diagnosis. Resume skips only compatible completed
cases.

**CONFIRM:** The user must authorize the smoke inference, inspect its technical
result, and then separately authorize the rest of the sacrum pilot.

## 9. Run only the authorized pilot phase

Sacrum pilot:

```bash
python -m tools.quadra.totalsegmentator reseg-run-phase \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --phase sacrum_pilot
```

Refined-vertebra pilot, after a separate dry-run and confirmation:

```bash
python -m tools.quadra.totalsegmentator reseg-run-phase \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --phase vertebrae_pilot
```

The vertebra command runs the full `vertebrae_pp_refined` task because that
task has no safe per-level ROI route in this plan. The runner retains all 24
C1–L5 outputs for provenance, while the sparse review index contains only the
levels originally flagged for each scan. The dedicated task changes the method,
so review must confirm label identity and anatomical scope, including posterior
elements; do not assume it is interchangeable with the original `total` labels.

**CONFIRM:** Download, technically validate, and manually review each pilot
before discussing any batch expansion. The current executable manifest cannot
run the expansions accidentally.

## 10. Review a downloaded pilot with the existing local viewer

After a checksum-verified transfer of the run directory to the Mac, build a new
sparse review index. Use a new review directory; do not point it at the completed
original review.

```bash
cd "/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /self-supervised-anatomical-embedding-v2"

python3 -m tools.quadra.mask_review build-reseg-index \
  --execution-manifest "<local execution_manifest.json>" \
  --dataset-root "/Users/rifqiab2708/Documents/quadra_dataset/QUADRA_HC_WB" \
  --output-root "<local downloaded run root>" \
  --review-root "<new pilot review directory>" \
  --phase sacrum_pilot

python3 -m tools.quadra.mask_review serve \
  --review-root "<new pilot review directory>" \
  --host 127.0.0.1 \
  --port 8502
```

Review each new mask normally in the existing viewer; side-by-side display is
not required. A sacrum pilot PASS requires both sessions to improve without new
leakage or voxel errors in at least two of the three pilot subjects. If the
pilot fails, retain the original masks and document the limitation; do not
silently relabel truncation as acceptable.

## 11. Remaining small phases

Ribs and other organs require their own dry-run, user confirmation, execution,
technical validation, transfer, and manual review:

```bash
python -m tools.quadra.totalsegmentator reseg-run-phase \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --phase ribs \
  --dry-run

python -m tools.quadra.totalsegmentator reseg-run-phase \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --phase other_organs \
  --dry-run
```

Remove `--dry-run` only after explicit confirmation. The four corrections are
safer locally because they need the reviewed source masks and no GPU. The
correction routine refuses to publish unless exactly one strictly 26-isolated
foreground voxel exists in each requested mask.

## 12. Status, backup, and stop safety

```bash
python -m tools.quadra.totalsegmentator reseg-status \
  --manifest /workspace/quadra/metadata/manifests/flagged-mask-resegmentation-20260926.json \
  --dataset-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --output-root /workspace/quadra/runs/preprocessing \
  --json-output /workspace/quadra/runs/preprocessing/flagged-mask-resegmentation-20260926/status.json
```

New masks are new dataset derivatives. They are not covered merely because old
masks were backed up, and `runs/preprocessing` evidence is not a substitute for
an independent mask-derivative backup. Before recommending a stop, follow:

1. `bash setup.sh backup-plan`
2. `bash setup.sh backup-pull`
3. `bash setup.sh backup-verify`
4. `bash setup.sh backup-status`
5. `bash setup.sh safe-stop-check`

Use current pod connection details and the repository's
`tools/quadra/environment/ARTIFACT_BACKUP.md`. An old `SAFE_TO_STOP` result is
not reusable. Stopping or terminating the pod always requires a separate,
explicit user request.
