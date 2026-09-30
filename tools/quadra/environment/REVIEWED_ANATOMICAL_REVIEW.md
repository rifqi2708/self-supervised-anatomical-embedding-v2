# Reviewed matching: anatomical inspection and adaptive review

These commands use the existing `tools.quadra.aligned_organ_group_cohort` CLI.
They read the frozen reviewed-mask contract and checksum-sealed method bundles.
They never execute matching, registration, a smoke test or a scientific pilot.
Software checks establish engineering behavior; they do not establish anatomical
correspondence or identify a causal anatomical explanation automatically.

## Queue and provisional policy

`reviewed-review-queue` accepts `--contract`, repeated `--run-directory`,
`--policy` and a new `--output-directory`. Its policy JSON requires exactly:

```json
{
  "version": "pilot-provisional-v1",
  "frozen_for_cohort": false,
  "seed": 20260721,
  "disagreement_mm": 10,
  "low_cycle_mm": 1,
  "random_per_organ": 10,
  "low_cycle_per_organ": 10
}
```

The numerical values above are illustrative. Calibrate the disagreement and
low-cycle thresholds and control counts during the approved pilot, then version
and freeze the settings before cohort execution. Setting a boolean is a recorded
operator declaration, not evidence that the scientific freeze gate passed.

Queue reasons are preserved as a union: each method's upper 5% within
subject–organ using linear 95th-percentile interpolation and all cutoff ties;
all technical failures; cycle, status or physical forward-correspondence
disagreements; competing-peak and boundary flags; seeded random and low-cycle
controls. The same declared disagreement threshold applies to cycle differences
and correspondence distance, which are recorded as different reasons. Missing
peak/boundary diagnostics are counted as **unknown**, not as normal.

Controls are sampled without replacement within organ across subjects before
unioning triggers; overlap remains visible. `control_sampling.json` records
eligible counts and chosen IDs, including genuine control shortfalls.
`tail_cutoffs.json` preserves cell denominators and ties. `review_queue.csv`
preserves reasons and discovery/confirmation membership; `all_queries.csv`
keeps queries outside the initial queue available. Small cells can select more
than 5%; the queue is triage, not an anatomical failure classification.

## Inspect arbitrary queries and competing anatomy

```bash
python -m tools.quadra.aligned_organ_group_cohort reviewed-inspect \
  --contract /workspace/quadra/metadata/matching-contract-v2 \
  --run-directory /workspace/quadra/runs/cohort/METHOD_RUN \
  --ct-root /workspace/quadra/datasets/source/whole_body_ct_v1 \
  --query-id 'FROZEN_QUERY_ID' \
  --output-directory /workspace/quadra/reviews/query_points/DATASET/NEW_INSPECTION
```

Any frozen query is inspectable even if it was never selected into a queue.
The CT bytes must match the frozen portable inventory. Source coordinates are
computed from the native CT affine and checked against frozen physical LPS
coordinates when present. Matches and returns retain continuous LPS coordinates;
outside-CT locations are flagged and are not clipped.

Each available source/matched/returned point gets physical axial, coronal and
sagittal planes at local and wider radii (defaults 20 and 80 mm). Planes use
linear CT interpolation, -1024 HU outside the scan, and the same [-160,240] HU
window. `--window-min-hu` and `--window-max-hu` allow explicitly recorded display
changes shared across all points in that inspection. Numeric CT arrays are unchanged.
PNGs and numeric NPZ arrays share geometry; offsets are retained. These
are physical LPS planes even when the stored CT affine is oblique.

Retained `forward_peak_candidates` and `reverse_peak_candidates` additionally
expose CT around competing locations, using their `physical_lps_xyz` positions.
`--peak-limit` defaults to three retained candidates per direction (0–32).
This limits a requested inspection export, not retained all-query diagnostics.
Scores, ranks and separation metadata accompany each view. FP's corrected final
match remains distinct from these descriptor peaks. `inspection_manifest.json`
records point identities, geometry and file checksums. Views reveal evidence for
human interpretation; similarity alone cannot explain the underlying anatomy.

Inspection loads each required scan once as float32 and computes six planes per
point. Its real CT memory/time and visual usefulness remain pilot checks. It
does not recreate discarded similarity maps; retained embeddings or explicitly
recorded regeneration are needed for those.

## Judgments, category freezing and adaptive batches

`reviewed-review-record --review-directory REVIEW --record RECORD.json` appends
an immutable judgment. Required JSON fields are `query_id`, `reviewer_id`,
`category_version`, `category_id`, `judgment`, `anatomical_reason`, `uncertainty`,
`selection_reason`, `category_definition`, `categories_frozen`, and
`review_stage`. Allowed judgments are `supported_contributor`,
`suspected_contributor`, `unresolved` and `indeterminate`. The review stage must
match the frozen subject partition. `supersedes_record_id` can revise the same
reviewer's judgment on the same query while preserving the previous record;
`stopping_reason` optionally records a coverage/stopping decision.

A linked sequence of record hashes detects changed, missing intermediate,
reordered or branched history. An exclusive writer marker prevents concurrent
writers. Preserve a stale marker for inspection after a crash. This local hash
history is not a signed independent archive; a separately verified backup is
still necessary.

After discovery, explicitly freeze all definitions for a version:

```bash
python -m tools.quadra.aligned_organ_group_cohort reviewed-review-freeze-categories \
  --review-directory REVIEW --category-version VERSION --reviewer-id REVIEWER
```

Confirmation judgments and confirmation expansion require that frozen snapshot;
category definitions cannot then change or be added under the same version.
A new definition requires a new version. Reviewer identity is recorded, not
credentialed by the software. The tool does not contact or assign reviewers.

`reviewed-review-expand` accepts the review directory, organ (`--mask-name`),
category ID/version, `--batch-size`, `--seed`, a new output directory, and
`--partition discovery|confirmation` (default discovery). It selects unreviewed
queries across subjects in seeded round-robin order, prioritizing previously
unreviewed subjects. It preserves original reasons and adds the category-expansion
reason. Human discoveries drive expansion; the tool does not infer new failure
categories or silently choose a stopping rule.

`reviewed-review-status --review-directory REVIEW [--output-file NEW_JSON]`
reports organ coverage, subject counts, reviewers, category versions, partitions,
stopping reasons and unresolved/indeterminate judgments. It excludes superseded
judgments from current counts while preserving their history. These counts do
not estimate population prevalence from selected cases or guarantee discovery
of rare modes.

Initial human anatomical review stays in discovery subjects. Confirmation CT
inspection is blocked until `--confirmation-review-directory REVIEW` supplies a
matching experiment's verified judgment history and an explicit frozen category
snapshot. Snapshot definitions must reconcile the preceding discovery records;
inspection records the snapshot hashes. This gate preserves the planned ordering
inside this tool; it cannot prove that a researcher has never viewed the CT using
another application. Log any external early inspection as a protocol deviation.

## Routing and remaining gates

Use `reviews/query_points/<dataset-version>/<review-id>/` or
`runs/cohort/<run-id>/analysis/<analysis-id>/` under the approved archive/storage
root. Never overwrite a prior review. Current backup code includes
`reviews/query_points` and `runs/cohort`; still require a fresh transfer receipt
for remote evidence. CT, models and embeddings have separate recovery contracts.

Before cohort execution: human pilot review, diagnostic usefulness, calibration,
resource feasibility, independent anatomical confirmation and asset/evidence
recovery remain distinct gates. No command here claims anatomical accuracy.
