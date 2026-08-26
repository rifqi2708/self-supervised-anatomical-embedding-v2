# Quadra generated-evidence backup

This workflow copies reproducibility-critical generated evidence from RunPod to
a checksum-verified local archive. It does **not** copy CT data, masks that are
already backed up, checkpoints, package environments, model caches, or the Git
repository.

The canonical local root on Rifqi's Mac is:

```text
/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/quadra
```

The archive mirrors only the generated parts of `/workspace/quadra`:

```text
runs/{cohort,memory_optimization,preprocessing,uae,archive,analysis}
metadata/{manifests,known_assets,transfer_receipts}
reviews/{masks,query_points}
exports/{documents,presentations}
transfer/{incoming,inventories,conflicts,packages}
scratch
```

## Initialize and catalogue existing local artifacts

From the repository root:

```bash
bash setup.sh backup-init \
  --local-root "/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/quadra"
```

This copies existing reports, plots, result CSVs, presentations and mask-review
evidence into the archive. Sources remain unchanged. The existing Stage 5 mask
archive is registered as a known asset and is not duplicated.

## Plan and pull from a pod

Use the direct SSH endpoint shown by RunPod under **Connect → SSH over exposed
TCP**. The IP and port are intentionally supplied at runtime because they may
change.

```bash
export QUADRA_LOCAL_ARCHIVE="/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/quadra"
export QUADRA_POD_SSH_HOST="root@POD_IP"
export QUADRA_POD_SSH_PORT="POD_PORT"

bash setup.sh backup-plan \
  --local-root "$QUADRA_LOCAL_ARCHIVE"

bash setup.sh backup-pull \
  --local-root "$QUADRA_LOCAL_ARCHIVE" \
  --transport auto
```

The pull creates a deterministic archive on the pod, downloads it by SCP into
`transfer/incoming/<transfer-id>`, checks its outer and per-file hashes, and
promotes new run directories only after validation. A same-named run with
different contents is preserved under `transfer/conflicts/` and fails the
operation. Neither the remote source nor an existing local run is deleted.

If direct SSH is unavailable, create the package from the pod's web terminal:

```bash
cd /workspace/repos/uae-quadra-validation
python3 -m tools.quadra.artifact_backup backup-remote-package \
  --remote-root /workspace/quadra \
  --repository-root /workspace/repos/uae-quadra-validation \
  --transfer-id TRANSFER_ID
```

Send the generated package with the pod's installed `runpodctl`, receive it on
the Mac, and import it with the SHA-256 printed by the remote package command:

```bash
bash setup.sh backup-pull \
  --local-root "$QUADRA_LOCAL_ARCHIVE" \
  --transport local-package \
  --transfer-id TRANSFER_ID \
  --package-file /path/to/received-package.tar.gz \
  --package-sha256 REMOTE_SHA256
```

## Verify and assess stop safety

```bash
bash setup.sh backup-verify \
  --local-root "$QUADRA_LOCAL_ARCHIVE" \
  --transfer-id TRANSFER_ID

bash setup.sh backup-status \
  --local-root "$QUADRA_LOCAL_ARCHIVE" \
  --ssh-host "$QUADRA_POD_SSH_HOST" \
  --ssh-port "$QUADRA_POD_SSH_PORT"

bash setup.sh safe-stop-check \
  --local-root "$QUADRA_LOCAL_ARCHIVE" \
  --ssh-host "$QUADRA_POD_SSH_HOST" \
  --ssh-port "$QUADRA_POD_SSH_PORT"
```

`safe-stop-check` requires complete remote-to-local parity, no in-progress
runs, no active scientific processes and a clean pod repository. It prints
`SAFE_TO_STOP` or `NOT_SAFE_TO_STOP`; it never stops or terminates the pod.

When the pod image has no SSH server, generate both JSON evidence files in the
web terminal with `backup-remote-inventory` and `backup-remote-status`, transfer
them with the same checksum-verified `runpodctl` procedure, and run:

```bash
bash setup.sh safe-stop-check \
  --local-root "$QUADRA_LOCAL_ARCHIVE" \
  --remote-inventory-file /path/to/remote-inventory.json \
  --remote-status-file /path/to/remote-status.json
```

The verdict records whether its evidence came from live SSH or from explicitly
transferred operator files.

## Evidence rules

- RunPod-to-local transfer is one-way. There is no bidirectional sync and no
  `--delete` operation.
- New or manually corrected masks must be registered as a new derivative before
  stopping a pod. Review PNGs and review queues belong under `reviews/masks/`.
- Complete analysis evidence stays outside Git. Only deliberately curated small
  reports or figures are promoted to `reports/quadra/`.
- Google Drive is a later secondary-copy layer, not part of this first workflow.
