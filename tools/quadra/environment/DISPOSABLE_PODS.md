# Disposable Quadra pods

This workflow reconstructs UAE-S and registration pods from GitHub plus
checksum-pinned Google Drive packages. It intentionally uses container storage
only. Stopping or terminating a pod destroys `/workspace`, so generated evidence
must be verified on the Mac before either action.

For the concise operator checklist, use
[`START_DISPOSABLE_POD.md`](./START_DISPOSABLE_POD.md).

The saved-template specification is tracked in
`configs/quadra/runpod-templates-disposable-v1.json`. Create templates manually
in RunPod; do not store API keys, Drive cookies or tokens in a template.

## Readiness plan

```bash
bash setup.sh disposable-plan \
  --profile uae \
  --asset-catalog configs/quadra/disposable-assets-v1.json
```

The command returns ready only when every required package, including
`quadra-experiment-contract-v1.tar.gz`, has an exact Drive ID, byte size and
SHA-256 in the catalogue. The contract is stored private and owner-only; it is
made link-readable only for the bounded bootstrap window described below.

## Build the frozen experiment contract

On the Mac, from the disposable implementation checkout:

```bash
python3 -m tools.quadra.experiment_contract build \
  --local-archive "/Users/rifqiab2708/Documents/self-supervised-anatomical-embedding-v2 /quadra-local-storage/quadra" \
  --output "/path/to/quadra-experiment-contract-v1.tar.gz"

python3 -m tools.quadra.experiment_contract verify \
  --package "/path/to/quadra-experiment-contract-v1.tar.gz"
```

Upload that package to restricted Drive only after reviewing its manifest. Give
it temporary link access immediately before bootstrap, record the returned file
ID/size/hash in the asset catalogue, then revoke link access after both required
downloads. The package contains frozen research inputs and the corrected
subject-030 sacrum derivative; treat it as restricted research data.

## Bootstrap a fresh pod

Clone or otherwise obtain the repository bootstrap code, then run:

```bash
bash setup.sh disposable-bootstrap \
  --profile uae \
  --storage-root /workspace/quadra \
  --repository-ref quadra-disposable-v1 \
  --image-ref sunyu0410/uae:py37torch19 \
  --confirm-image-digest sha256:2c0edd4a205c3c5d9d027b6c9f96f83626eb2cc3810da7876e32d4bf36653d61
```

For registration, use `--profile registration`, the modern RunPod image and its
catalogued digest. Bootstrap refuses separate `/workspace` mounts, wrong Python
versions, wrong image identity, insufficient UAE GPU memory, unready assets,
unsafe archives and conflicting destinations. It downloads using a temporary
temporary downloader environment and never launches a cohort. It pins
`gdown==5.2.0` on Python 3.8+ and the final Python-3.7-compatible
`gdown==4.7.3` in the legacy UAE image. Interrupted downloads
resume from `staging/disposable-bootstrap-<profile>` only when the profile,
catalogue hash, image identity and repository commit are unchanged; mismatched
staging is refused for inspection rather than silently reused.

The complete 96-CT archive is verified, including every voxel payload, before
only subjects 021–048 are promoted into the disposable pod. All final and
intermediate masks are fully decompressed and checked as non-empty binary
volumes. Cross-asset validation then checks the 56 selected CT hashes and all
2,208 CT–mask geometries against the frozen contract. The UAE checkpoints are
kept under canonical storage and exposed through ignored repository
`checkpoints/` links, so legacy commands do not duplicate the weights.
Before writing the environment manifest, bootstrap also runs a bounded technical
smoke. UAE uses a dense `32 × 64 × 64` tensor and all three UAE-S features for
one unrestricted bounded argmax; registration uses identical `24³` synthetic
volumes for independent forward/reverse Elastix and continuous Transformix.
These checks exercise runtime plumbing only and are not scientific results.

Activation is profile-specific:

```bash
source /workspace/quadra/runtime/activate.sh uae
source /workspace/quadra/runtime/activate.sh registration
```

Only the profile used to build that disposable pod will activate.

For pre-tag live acceptance only, use the exact reviewed feature branch or commit
as `--repository-ref`. After acceptance, recreate both profiles from the
immutable `quadra-disposable-v1` tag; that tagged recreation is the production
reproducibility gate.

The experiment contract is exposed as `QUADRA_EXPERIMENT_CONTRACT`. Accepted
manifests inside it are provenance snapshots and retain historical absolute
paths. New disposable runs must use the contract's frozen query CSV and 224 plan
files rather than dereferencing those historical paths.

## Package and back up results

At stable subject milestones:

```bash
bash setup.sh disposable-package-results \
  --storage-root /workspace/quadra \
  --run-directory /workspace/quadra/runs/cohort/RUN_ID \
  --transfer-id UNIQUE_ID
```

Pull and verify the package using `ARTIFACT_BACKUP.md`. Coverage includes all
canonical run, analysis, review, export and manifest roots. It excludes source
datasets, weights, environments, caches and staging.

Before termination, create a small JSON attestation after Drive link access has
actually been revoked:

```json
{
  "attested_at": "2026-09-13T00:00:00Z",
  "operator": "Rifqi",
  "all_temporary_drive_links_revoked": true
}
```

Then run the live gate from the Mac:

```bash
bash setup.sh safe-terminate-check \
  --profile uae \
  --local-root "$QUADRA_LOCAL_ARCHIVE" \
  --remote-root /workspace/quadra \
  --ssh-host root@CURRENT_IP \
  --ssh-port CURRENT_PORT \
  --drive-revocation-attestation /path/to/revocation-attestation.json
```

If the image exposes only RunPod's PTY gateway and `runpodctl`, generate
`backup-remote-inventory` and `backup-remote-status` immediately after the final
package, transfer those JSON files with the same checksum-verified procedure,
and pass them as `--remote-inventory-file` and `--remote-status-file` instead of
`--ssh-host`. This fallback preserves every termination gate; it changes only
how the current remote evidence reaches the Mac. Both snapshots must be no more
than five minutes old by default; otherwise the gate fails and they must be
regenerated and transferred again.

```bash
bash setup.sh safe-terminate-check \
  --profile uae \
  --local-root "$QUADRA_LOCAL_ARCHIVE" \
  --remote-root /workspace/quadra \
  --remote-inventory-file /path/to/remote_inventory.json \
  --remote-status-file /path/to/remote_status.json \
  --drive-revocation-attestation /path/to/revocation-attestation.json
```

`SAFE_TO_TERMINATE` requires current process and repository evidence, complete
checksum parity, no unclassified repository outputs, published code at the exact
commit, ready recovery packages and the revocation attestation. The command
never stops or terminates the pod. The Mac remains the sole generated-evidence
recovery location under the selected policy.

## Scientific boundary

Bootstrap does not run TotalSegmentator or SuperPoint, change organs or queries,
choose a memory strategy, or launch a 28-subject cohort. A bounded technical
smoke test is required on each alternative UAE GPU before research execution.
