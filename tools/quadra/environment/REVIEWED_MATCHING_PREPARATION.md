# Reviewed-mask matching preparation

This implements shared preparation tickets #4–#6 under spec #3. Prepare locally
before renting the separate UAE and registration pods. These commands do not run
NN, fixed-point inference, registration, a smoke test or a scientific pilot.
Live method implementations and pilot approval remain separate tickets.

The historical 28-subject workflow and `quadra-disposable-v1` release remain
available through their existing commands. Reviewed inputs use schema version 2.
Use the exact task commit in the generated setup plans, rather than a moving
branch or the historical release tag.

## Freeze the shared contract

Use the new final-reviewed dataset with 48 subjects, 96 scans and 3,790 masks.
The freezer checks its acceptance manifest, all selected mask and CT hashes,
binary nonempty masks, millimetre units and CT/mask geometry. It rejects missing,
duplicate and sex-ineligible inventory entries. It does not change source masks.

```bash
python -m tools.quadra.aligned_organ_group_cohort freeze-reviewed \
  --dataset-root "$REVIEWED_DATASET" \
  --ct-root "$WHOLE_BODY_CT_ROOT" \
  --output-directory "$PREPARATION_ROOT/contract"
```

Write `PREPARATION_ROOT` under the local archive's `metadata/manifests/` using a
new descriptive UTC ID. Do not set the pod storage root to this partial archive.
Existing outputs and failed `.partial` directories are preserved and block an
overwrite. Investigate a failure before selecting another output ID.

The bundle contains one frozen random Test query ledger (up to 100 unique native
voxels per eligible organ, seed 20260721 plus registry index), sampling shortfalls,
384 aligned UAE crop plans, 384 corresponding native registration plans, physical
LPS coordinates, and portable mask/CT identities. Fine-grid collisions retain
both native queries. Fine coordinates use rounding of model coordinates followed
by floor division by two; this records quantization and does not establish UAE
retrieval equivalence. FPS Harris–Laplacian sampling remains a separate experiment.

The discovery/confirmation allocation is 32/16, stratified by sex and historical
cohort membership (subjects 021–048). Pilot subjects stay in discovery and cover
the largest new padded crop, both sexes, and a revised-mask subject in at most
three subjects. No matching outcomes are used to select them.

The registration parameter maps in
`configs/quadra/reviewed-registration-parameters.json` reproduce the recovered
`registration-setup-20260827/resolved_registration_parameters.json` record.
Full rigid Euler and B-spline maps are preserved, including four resolutions,
256 iterations, 8,192 samples, random seed 121212 and 32 mm final B-spline grid
spacing. Frozen geometry comes from corresponding anatomical group boxes.
Transform evaluation is reserved for the registration implementation ticket.

## Prepare independent method evidence roots

```bash
python -m tools.quadra.aligned_organ_group_cohort prepare-method \
  --contract "$PREPARATION_ROOT/contract" --method uae_nn \
  --run-directory "$LOCAL_ARCHIVE/runs/cohort/$RUN_ID-uae-nn"
```

Repeat with `uae_fixed_point` and `registration`, using distinct output roots.
All three share the input signature. Each exports `method_manifest.json` and
`query_outcomes.csv`, with every query pending and zero attempted queries.
`--resume` requires matching method, input signature, commit, implementation and
outcome-ledger hash. The synthetic-only `fixture-method` command verifies terminal
success/failure export and compatible replay; it rejects real-data contracts.
Success requires both directional statuses and a finite nonnegative cycle error;
failure retains its reason. This is engineering evidence, not anatomical accuracy.

## Package local masks for later SSH upload

```bash
python -m tools.quadra.disposable_pod reviewed-package \
  --contract "$PREPARATION_ROOT/contract" --dataset-root "$REVIEWED_DATASET" \
  --output-directory "$LOCAL_ARCHIVE/transfer/packages/$PACKAGE_ID"
python -m tools.quadra.disposable_pod reviewed-verify-package \
  --package-directory "$LOCAL_ARCHIVE/transfer/packages/$PACKAGE_ID"
```

The package contains deterministic mask and contract archives, a version 2 asset
catalog, separate setup plans and a checksum receipt. Creation verifies every
archived payload; the verification command rechecks outer and inner identities.
The whole-body CT and UAE model packages retain the existing checksum-pinned
Drive recovery sources. Only reviewed masks and the new contract need local SSH
staging. Public source control must not contain these private packages or images.

After deployment approval, upload the plan's three listed files to its declared
`/workspace/quadra/staging/reviewed-inputs/` directory on each supplied pod using
SSH/SCP or rsync. Obtain the current endpoint from a live pod read. Upload through
a temporary filename, verify the local receipt's SHA-256 remotely, then rename;
preserve any conflicting upload rather than overwrite it. RunPod MCP manages
infrastructure; SSH/file transfer uses the terminal tools.

Execute only the generated setup command with `--setup-only`. The bootstrap
checks the pinned image/profile, asset hashes, NIfTI payloads, cross-asset geometry,
runtime dependencies, exact repository commit and available disk. Version 2
refuses conflicting promoted destinations. Setup pauses before bounded smoke or
scientific work and records `SETUP_COMPLETE_PENDING_SMOKE`. It does not create the
verified environment manifest required by `disposable-status`; activation remains
blocked until the later approved smoke/pilot gate. Code/model, extraction-context,
retrieval-equivalence, resource and anatomical gates are still pending.

Do not create pods merely to wait for method implementation. Finish the method
and evidence-intake prerequisites before deploying just ahead of setup and pilot.
The baseline proposals are 100 GB container disk with no persistent/network
volume, at least 48 GB VRAM for UAE, and 8 CPU vCPUs with 64 GB RAM for registration.
Host RAM and dense retrieval feasibility remain subject to measured pilot checks.
Read live stock, pricing and selected host allocations before approving deployment.

## Recovery boundary

Source code is recovered through Git, not generated-artifact backup. Local
contracts under `metadata/manifests/` and pending method roots under `runs/cohort/`
use covered evidence categories, but local creation alone is not an independent
backup. Dataset derivatives and transfer packages are excluded from generated-
evidence backup and need their own verified recovery. Follow `ARTIFACT_BACKUP.md`
before any eventual pod lifecycle recommendation.
