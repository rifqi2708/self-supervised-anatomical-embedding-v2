# UAE-S streamed matching pilot: `quadra_hc_021`

## Technical summary

- [High confidence] The implementation-equivalence checks passed: all 25
  bounded global argmax coordinates, all fixed-point internal match hashes and
  all final fixed-point coordinates agreed between dense and streamed matching.
- [High confidence] The expanded tile did not show material semantic seam
  effects on the five organ crops; worst p01 cosine was 0.999989 and the median
  seam-to-interior drop was effectively zero.
- The full 500-query pilot completed both modes without point failures in 566
  seconds using cached embeddings. The recommended production configuration is
  the `160×160×80` tile, `48×48×24` halo, `64×64×32` retained core and query
  batch 256. Subject 021's 8.25 GB cache was safely deleted after validation.

This supports proceeding to a reviewed cohort run as an engineering workflow.
It does not show that fixed-point matching is anatomically more accurate than
global NN or registration.

## Scope of the engineering claim

This is an engineering validation of the UAE-S implementation at the spacing
and tile geometry intended for the Quadra cohort. It asks whether tiling and
streaming preserve the model computations and whether both requested matching
methods run reproducibly. It does **not** establish anatomical accuracy or
clinical superiority.

## Definitions

- **Dense inference** processes one bounded `128×128×64` voxel crop in a single
  encoder call. It is used only as a reference where memory permits.
- A **tile** is the complete encoder input. Production uses
  `160×160×80` voxels (`320×320×160 mm` at 2 mm spacing).
- The **halo** is context supplied around each retained region and discarded
  after inference. It is `48×48×24` voxels on each side
  (`96×96×48 mm`).
- The **retained core** is the tile centre written to the virtual whole-volume
  embedding: `64×64×32` voxels (`128×128×64 mm`). Adjacent cores cover the
  subject without limiting possible displacement.
- A **seam** is a boundary between adjacent retained cores. Descriptor metrics
  are therefore reported separately near seams and farther inside cores.
- **Streamed global matching** scans every target embedding chunk and retains
  the global argmax. Chunking bounds memory; it does not impose a search radius.
- **Fixed-point matching** follows the released UAE-S structural workflow: a
  `5×5×5` anchor neighbourhood is matched over the full target fine grid for
  four alternating directions, stable anchors are filtered, and a robust local
  affine correction produces the final point.

## Reproducible configuration

| Item | Value |
|---|---|
| Subject | `quadra_hc_021` |
| Resampling | exact `2.0×2.0×2.0 mm` |
| Config | `configs/samv2/samv2_NIHLN.py` |
| Config SHA-256 | `cb45a8790c9524fb93cb1725b9604741cfc01de7a352bf9b2773718101126ba2` |
| Checkpoint | `checkpoints/SAMv2_iter_20000.pth` |
| Checkpoint SHA-256 | `a094d5eef867504defdc4c8e1d950835c4eb8aaa19de2027bb1a194781e423e3` |
| Cached features | fine, coarse and semantic, FP16 |
| Similarity arithmetic | FP32; `(fine + coarse + semantic) / 3` |
| Query batch | 256 descriptors for production; memory scheduling only |
| Fixed-point filter | score `>0.8`, return distance `<100 mm` |
| Organs | bladder, colon, kidney, liver and lungs |

The final affine correction reproduces the operation order in Alibaba DAMO
Academy's
[`demo_semantic_stable_points.py`](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2/blob/main/tools/demo_semantic_stable_points.py),
with explicit failures for insufficient or degenerate anchors rather than a
silent fallback to nearest-neighbour matching.

## Validation design

For each organ and both timepoints, the validator selected a deterministic
organ-centred `128×128×64` crop from the same normalized 2 mm tensor. It then:

1. compared dense and expanded-tile fine, coarse and semantic descriptors;
2. gave identical dense embeddings to dense and streamed global matchers;
3. compared every internal fixed-point argmax from the dense and streamed
   matchers; and
4. compared the final fixed-point coordinates.

The semantic gates were median cosine `≥0.99`, p01 cosine `≥0.95`, and
seam-to-interior median cosine drop `≤0.01`. Global argmax coordinates had to
be exactly equal. Fixed-point internal matches had to be exactly equal and the
final coordinates had to be within one native voxel.

## All implementation-equivalence gates passed

[High confidence] All implementation gates passed on the bounded five-organ
test:

- global dense-versus-streamed argmax: `25/25` coordinates exactly equal;
- maximum global score difference: `9.78×10⁻⁶` (diagnostic only);
- fixed-point internal match hashes: `5/5` exactly equal;
- fixed-point final coordinates: `5/5` exactly equal (`0` voxel difference);
- worst semantic median cosine: `1.000000`;
- worst semantic p01 cosine: `0.999989` (colon Test);
- maximum median seam-to-interior drop: effectively `0`.

These results support that the expanded central-crop tile and streamed matcher
do not materially change UAE-S computations in these bounded subject-021 tests.

## The full subject completed without point failures

| Stage | Queries | Global successful cycles | Fixed-point successful cycles | Runtime with reused cache | Cache outcome |
|---|---:|---:|---:|---:|---|
| 5 points/organ | 25 | 25/25 | 25/25 | 305 s after resume | retained (8.25 GB) |
| 20 points/organ | 100 | 100/100 | 100/100 | 453 s | retained |
| 100 points/organ | 500 | 500/500 | 500/500 | 566 s | deleted; 8.25 GB freed |

The initial stage first generated Test and Retest embeddings, but the reported
305 seconds covers only the corrected resume. It therefore must not be treated
as total first-subject runtime.

The final run produced 500 rows in every point/query/comparison CSV (plus the
header), reported no fixed-point failures, and retained 43–125 stable forward
anchors per query. The subject cache deletion left unrelated SAM cache
namespaces intact and returned the disk from 58% to 52% use.

### Final 100-point-per-organ cycle-error summary (mm)

| Organ | Global median | Global p95 | Fixed median | Fixed p95 |
|---|---:|---:|---:|---:|
| Bladder | 5.67 | 32.48 | 5.88 | 10.93 |
| Colon | 8.00 | 39.86 | 6.79 | 12.95 |
| Kidney | 6.26 | 27.05 | 5.88 | 9.73 |
| Liver | 6.28 | 21.58 | 6.07 | 8.85 |
| Lungs | 4.14 | 8.75 | 5.88 | 7.39 |
| **All** | **5.25** | **27.60** | **5.88** | **10.73** |

[Medium confidence] Fixed-point matching reduced the long upper tail of cycle
error for colon, kidney and liver in this sample, but often produced a roughly
6 mm error floor. Colon remained the most affected organ for both methods.
This is a consistency result, not evidence that fixed-point matches are
anatomically more accurate: the method explicitly selects anchors
using forward-backward stability.

The production query batch of 256 used approximately 2.38 GB of GPU memory
during fixed-point matching on the A6000. The 500 global forward/backward
coordinates had the same SHA-256 digest when independently computed with query
batches 64 and 256, confirming that this scheduling optimization did not alter
the selected points.

## The 28-subject run is projected at 6–7 hours

The final cached matching/export/cleanup stage took 566 seconds. Test and
Retest embedding generation recorded another 62 and 63 seconds, respectively,
giving a measured lower bound of 11.5 minutes per subject before model loading,
image preprocessing and ordinary I/O variation. A simple linear projection is
about 5.4 hours for 28 subjects; [Medium confidence] allocating 6–7 hours on a
similar A6000 is more defensible, with extra contingency for subjects whose
volume dimensions generate more tiles.

## All 28 cohort commands are ready but were not launched

The RunPod cohort dry-run generated exactly 28 compatible commands from
`quadra_hc_021` through `quadra_hc_048`, using both matching modes, 100 points
per organ, query batch 256 and delete-on-success cleanup. The final manifest
contained 28 subject entries with first/last IDs 021/048. Subjects 022–048 were
not launched.

## Recommended next steps

1. Review this pilot and the fixed-point interpretation before authorizing the
   28-subject run.
2. If approved, launch the existing cohort command without `--dry-run`; retain
   the 20 GB free-disk guard and default delete-on-success policy.
3. Analyse anatomical accuracy against the registration method or independent
   landmarks separately. Do not use lower fixed-point cycle error as the sole
   basis for method ranking.
4. Report fixed-point failures with their original denominator if they appear
   in later subjects; do not replace them with global NN.

## Questions for the cohort analysis

- Does the fixed-point reduction in the upper cycle-error tail recur across
  subjects, especially for colon, kidney and liver?
- Do method displacements correspond to better anatomical locations when
  assessed against registration or another independent reference?
- Does a later fine-tuned checkpoint preserve the same tile-context stability?

## Limitations and uncertainty

- This is one subject and one released checkpoint. It can detect implementation
  errors and major context sensitivity, but not establish population robustness.
- Dense equivalence is tested on bounded crops, not a dense 2 mm whole body.
- Cycle error measures self-consistency and can favour a method that explicitly
  optimizes forward-backward stability.
- No anatomical landmarks or independent registration ground truth are used;
  method displacement and cycle error alone cannot rank anatomical accuracy.
- The assumption that a later fine-tuned checkpoint has unchanged context
  sensitivity remains untested.
