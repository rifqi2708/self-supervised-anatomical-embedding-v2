# Reviewed matching implementation checkpoint

This implements work items 4–12 (GitHub #7–#15) from spec #3. The new reviewed
mask contract is the primary input. Neither historical masks nor historical
query sets can substitute for its immutable identities.

## Available commands

The existing `tools.quadra.aligned_organ_group_cohort` command exposes:

- `reviewed-uae-extract`: a read-only proposal with `--dry-run`, or explicitly
  approved full-group dense extraction with verified cache provenance. Extraction
  does not automatically change to a tiled context when memory is insufficient.
- `reviewed-uae-run`: fine-grid NN or fixed-point matching, compact diagnostics
  and selected similarity views. CPU execution is restricted to synthetic
  fixtures; real execution is restricted to explicitly approved frozen pilots.
  `--select-upper-tail 5` selects each subject–organ's upper 5% only after cycle
  outcomes exist, including cutoff ties, and exports views while the original
  embedding handles remain open. It does not substitute for the broader review
  queue, which also includes failures, disagreements and controls.
- `reviewed-uae-export-views`: select cases after inspecting their outcomes and
  export additional views from retained, verified embeddings into a new evidence
  directory. Primary outcomes are preserved; no model re-extraction is implied.
- `reviewed-intake`: validate a locally staged method bundle and copy it into a
  fresh directory with an intake receipt. It does not perform SSH transport or
  replace the artifact-backup transfer receipt.
- `reviewed-compare`: consume all three compatible method bundles, explicit
  statistical rules, and a new output directory. It exports one median/p95
  figure plus pairwise denominators, absolute errors, organ/prostate detail,
  all-attempted failures and resources. No inferential analysis or ECDF is added.
- `reviewed-review-*` and `reviewed-inspect`: anatomical triage, adaptive batches,
  arbitrary-query CT inspection, versioned judgments and category freezing. See
  [the review operating guide](REVIEWED_ANATOMICAL_REVIEW.md).

The existing `tools.quadra.registration_organ_group_cohort` command adds
`reviewed-run`. It consumes pinned rigid/B-spline maps, estimates directions
independently, and evaluates continuous LPS points without clipping. Synthetic
affine fixtures require no ITK. The real runtime requires explicit pilot approval
and is restricted to the frozen pilot subjects.

Use the commands' `--help` for their complete options. Do not reuse the prepared
run stubs after changing implementation identities; create fresh execution IDs.

## Evidence and resume contract

Each method bundle contains `method_manifest.json`, `query_outcomes.csv`, an
`output_inventory.json` that hashes every member, and optional diagnostics.
The complete frozen query denominator remains present when execution is partial;
unattempted queries remain pending. Resume validates the input, implementation,
backend and settings, and cannot replace terminal outcomes.

Producers stage new diagnostic files outside the sealed bundle. Checkpoints are
built in sibling directories, using hard links for unchanged artifacts, and
promoted only after validation. Previous snapshots and failed candidates remain
preserved. A writer lock prevents two producers from modifying the same bundle.
After an abrupt process death during promotion, inspect the preserved sibling
checkpoint/previous directories; do not infer readiness from an absent or stale
current manifest. Checkpoint, hashing and retained-snapshot costs need measurement
in the pilot. No automatic cleanup is performed.

Intake verifies all member hashes and rejects differing dataset, query, crop,
coordinate and model contracts. A valid local intake is not an attestation of
remote transport or complete asset recovery. Datasets/corrected masks, model
weights, embedding caches and Git source have separate recovery contracts.

## Statistical and diagnostic decisions still requiring review

Comparison requires an explicit rules JSON with `status` (`provisional` or
`frozen`), `quantile: linear`, `organ_weighting:
query_pool_within_subject_group`, positive `minimum_valid_median` and
`minimum_valid_p95`, and `inference: none`. These are configurable rules, not
validated clinical tolerances. Each contrast recomputes both summaries on its
own shared-valid query IDs; it does not compare medians of differing populations.

Peak separation, competing-score thresholds, review controls and sparse-cell
rules remain provisional. Separated high-scoring locations are candidates, not
proof of distinct anatomical modes or causal explanations. All attempted failure
rates accompany conditional cycle errors. Neither reviewed masks nor low cycle
error establishes independent anatomical accuracy.

Dense retrieval is preferred by the automatic backend when its declared estimate
fits the configured ceiling. Current default ceilings are 512 MiB on CPU and
32 GiB on CUDA; they are provisional limits, not measured available memory.
Out-of-memory stops preserve completed outcomes and leave unattempted queries
pending. Released retrieval is adapted to FP32 scoring and interpolation; the
released source uses its input tensor dtype. This precision deviation and
real CUDA dense/streamed agreement must be evaluated during the UAE pilot.

## Next boundary

Local fixtures and pinned-code comparisons are engineering evidence. Released
GPU retrieval parity, full-group extraction context, actual Elastix execution,
GPU/CPU/RAM/disk/time requirements, diagnostic overhead and anatomical usefulness
remain pending the bounded UAE and registration pilots (#16/#17). The joint
pilot review/freezing ticket (#18) gates the cohorts. This checkpoint authorizes
none of those runs and does not change any pod lifecycle state.
