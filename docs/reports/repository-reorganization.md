# Approved tidy repository layout

This implements the reviewed 264-file map from publication baseline `3d89b35`. The first migration moved implementations while retaining old paths. Rifqi subsequently authorized removing obsolete path compatibility so the current tree contains canonical source rather than forwarding scaffolding. Reviewed FILE_MAP comments and earlier audit snapshots remain unchanged; final execution state is recorded in a new local archive run.

## Canonical areas

```text
sam/                         original model package, unchanged
configs/{_base_,sam,samv2}/  common model settings, unchanged
master_project/              earlier scripts, fine-tuning config and notebooks
publication/
  masks/{segmentation,postprocessing}/
  query_points/{superpoint,harris_laplacian}/
  uae/                       embeddings, active streamed matching and Stage 5R
  registration/              native comparison and point transforms
  analysis/                  comparison and anatomical inspection
  operations/                existing setup, input contracts and backup
  shared/                    contracts, coordinates and common hashing
  investigations/            memory, crop, tiled and detector experiments
miscellaneous/               PMCC, other contributor work and unused support
shared/                      cross-project interfaces and launchers
tests/publication/          automated checks grouped by workflow
junk/                        retained real review-queue files
```

Aim for at most four directory levels below root. Original model nesting and packaged HTML components are necessary exceptions. No original model names, organization, modifications or scientific parameters were reverted or changed.

## Removed compatibility

The tidiness pass removes 181 obsolete tracked members: 123 Python forwarding modules, seven shell launchers, 48 symbolic links and three namespace/alias-infrastructure files. Empty legacy folders and their regenerable Python caches are removed. No real archived script or scientific asset is deleted.

Use canonical imports and `python -m publication.<stage>.<module>` or direct canonical script paths. Old root train_loop scripts, tools.quadra modules, tools interfaces and shell aliases no longer run from this tree. Reproduction of historical commands uses their recorded old commit. The root setup.sh remains a real operations entry point.

Reusable SuperPoint grouping/export helpers remain beside the detector; earlier detector-characterization runners remain in investigations. PMCC runners and directly required helpers stay together. Model utilities are shared rather than duplicated. Current worker dispatch and startup code use canonical paths. Only bootstrap into an explicitly selected immutable old checkout can choose that checkout's legacy module names and requirements paths.

## Remaining root exceptions

- `sam/` and common `configs/`: original model code/settings, preserved.
- `resources/`: inherited model documentation and example figures, preserved.
- `reports/`: original curated validation/pilot reports and figures; no evidence transfer or deletion.
- `backup/`: two original source backup variants, preserved as historical references. Their original imports are not active current-tree entry points; reproduce them from the old commit.
- `tools/assets/reference/`: only the two original reference-history texts held for privacy review. No Python source or tools/quadra workflow tree remains there.
- `docs/`, `tests/` and root setup/requirements/license/git files: maintained documentation, software checks and required repository support.

Unused dev-container and Docker/DEEDS configuration lives under miscellaneous. Rifqi confirms that the dev-container is unused and the RunPod image is entered manually, so standard root dev-container discovery is intentionally absent. The DEEDS Makefile remains an unused historical member, without a claim that its missing sources are recoverable.

## Contracts and evidence

Pinned independent UAE reference sources are byte-preserved. They contain original upstream import text as provenance; the reviewed adapter extracts reference functions without importing obsolete tools packages. Historical schemas, contracts, run manifests and external evidence are untouched.

Implementation signatures now hash all declared canonical members; a missing member fails instead of being omitted. Source changes may correctly prevent resuming an older run. Do not rewrite frozen signatures to make a resume pass. Use the original recorded commit for old evidence. Immutable production tags, pod checkout directory names and data/checkpoint locations remain unchanged.

The RTSTRUCT reference contains nonempty identification fields. Its raw bytes and the console-history reference remain at their original paths pending privacy review. No values are included here and no deidentification or history-remediation claim is made.

All primary/other-checkout modified, untracked and ignored research work remains outside this source migration. No scientific/GPU/pod job or infrastructure action is performed.

## Verification and review

Run from the repository root:

```bash
python -m unittest discover -s tests -t . -p 'test_*.py'
```

The explicit top-level root prevents the test package from shadowing runtime publication imports. The feasible suite passes 461 tests with the same 16 existing Torch-dependent skips. Tests now check canonical direct and module entrypoints, absence of scaffolding, cross-stage error identity, source/signature lookup, missing-member rejection and frozen external checkout selection. Separate checks cover shell argument boundaries, config inheritance, HTML component assets, worker strings, syntax and maintained Markdown links.

Real Torch/CUDA inference and real CT/registration processing remain unverified locally. CPU tests use synthetic fixtures in the disposable /tmp environment, not a modified scientific environment.

PR 25 targets publication; the implementation branch is codex/repository-reorganization. Publication remains at the pre-migration baseline until reviewed integration. GitHub default main and the old validation recovery reference remain unchanged. Recovery before this tidiness pass is commit cb0da25; before the full migration it is 3d89b35. No merge or force push is included.
