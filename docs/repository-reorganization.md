# Approved repository migration

This change implements the reviewed 264-file map against publication baseline `3d89b35659cf56d26cb0cb580e77e8ab111f5d58`. The approved six formerly uncertain destinations use compatibility paths. No dirty work from other worktrees was integrated.

## Areas

```text
sam/                         original model package, unchanged
configs/                     unchanged common model settings
master_project/              earlier training, configs and notebooks
publication/
  masks/segmentation/        mask generation and resegmentation
  masks/postprocessing/      QC, manual review and defined corrections
  query_points/superpoint/   reusable detector and review support
  query_points/harris_laplacian/
  uae/                       embeddings and matching
  registration/              native comparison and point transforms
  analysis/                  comparison, reports and anatomical inspection
  operations/                existing setup, input contracts and backup
  shared/                    cross-stage contracts and coordinates
  investigations/            memory, crop, tiled and detector experiments
miscellaneous/PMCC/          coordinated numbered runners and helpers
miscellaneous/               other contributor workflows/historical support
shared/                      cross-project interfaces and compatibility
tests/publication/           automated software checks
junk/                        retained review queue; no deletion
```

The target is at most four directory levels below root. Original SAM nesting and packaged Streamlit component HTML are necessary exceptions. Model names and all existing project model changes remain intact. No upstream restoration or functional scientific changes are included.

## Scope and evidence boundary

206 tracked files are relocated to canonical code/config/doc/test homes. Fifty retain their locations. Eight protected originals (four reports/figures, two backup source variants and two reference-history texts) remain in place; their reviewed junk destinations are relative links. Original report bytes, backup variants, datasets, checkpoints, ignored generated files and other-worktree uncommitted work are not moved or overwritten. Retained links do not claim that evidence has been transferred or independently backed up.

The original map and reviewed comments are preserved in the local archive. Machine-readable mapping.json/CSV were reconciled from that reviewed Markdown, not the initial stale generator. Per-row effective locations and any evidence holds are recorded there.

## Compatibility

Old Python modules/scripts forward to one canonical implementation. Package forwarding keeps old child discovery on its shims to preserve module and exception identity. Shell aliases execute canonical launchers. Relative links retain config/requirements/reference-asset access without copying implementations. Tests have no duplicate legacy collection tree.

Run test discovery from the repository root with `python -m unittest discover -s tests -t . -p "test_*.py"`; the explicit top-level root prevents tests/publication from shadowing the runtime package. Old scientific launch commands remain available through compatibility entry points. Archived notebooks retain their JSON structure and revised code-reference paths.

Shared model utilities are not copied into PMCC. Reusable SuperPoint mask grouping, window cases and CSV export were extracted without changing their bodies; generic hashing is shared by detector methods. Stage 5R and streamed global matching remain accessible to current runners.

## Contracts and recovery

Pinned independent UAE reference files and common model configs are byte-preserved. Historical schemas/contracts/manifests and external scientific evidence are untouched. Source signatures now hash canonical implementation members: old run resumes may correctly reject a changed implementation signature. Use the old recorded commit for reproduction rather than rewriting a frozen manifest or bypassing signature checks. Immutable production tags and pod checkout directory names remain unchanged.

`publication` is the integration baseline; the implementation is reviewed on `codex/repository-reorganization` via a draft PR. The old remote `uae-quadra-validation` remains a recovery reference. GitHub default remains main. No merge or branch deletion is performed. Recovery before migration is `codex/reorganization-pre-move-20261013` at `3d89b35`.

## Verification

The full lightweight suite uses a disposable local test environment for Streamlit/openpyxl without changing the scientific environment. The original baseline passes 454 tests and this tree passes 458 tests, each with the same 16 existing Torch-dependent skips. New checks cover import/class identity, pinned-source lookup and old/new direct CLI help outside repository cwd. Detailed baseline/current test logs, SHA-256 inventory, config/launcher/asset checks and preservation evidence are saved directly under canonical local archive `runs/analysis/repo-migration-20261013T153159Z/`.

Torch/CUDA inference and real CT/registration processing remain unverified locally. No GPU/pod job, dataset download, environment change on a scientific host, evidence upload or infrastructure action was performed.

## Reference text sensitivity hold

The tracked RTSTRUCT dump contains nonempty patient-name/ID fields, so its raw text is held at its original path. The console-history reference is also held pending privacy review. Their junk entries are links, not new raw-text copies. No identifier values are published by this migration report, and this check does not establish deidentification or erase existing repository history. Any later privacy/history remediation requires a separate decision.

External setup checks can still target frozen checkouts with the original layout. Explicit external-root source lookup falls back to the legacy requirements member; current-checkout signature lookup does not silently fall back to forwarding shims. Bootstrap subprocesses use retained historical entry points so immutable production tags remain usable.
