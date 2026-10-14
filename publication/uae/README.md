# UAE embedding and matching

`aligned_organ_group_cohort.py` and the reviewed UAE adapters implement the existing matching contracts. The original SAM package and model configurations remain at `sam/` and `configs/`.

Streamed global matching in `streaming_cycle_error.py`, geometry in `streaming_embedding.py`, and global-lattice alignment remain accessible here. Streamed matching is distinct from alternative tiled encoder experiments. Earlier crop/memory stage helpers remain imported from investigations where still required; they were not discarded.

Pinned reference sources and their license in `references/` are byte-preserved. Historical contracts/evidence are not rewritten. New source signatures reflect the relocated implementation; a changed signature can require a new run contract rather than silently resuming an older run.

Use [workflow commands](../README.md) and [reviewed matching guide](../shared/REVIEWED_MATCHING_IMPLEMENTATION.md). Use the canonical publication module paths shown here.; datasets, model/checkpoint locations and output schemas retain their previous meanings. Torch/CUDA model execution requires the existing scientific environment and is not part of a repository migration check.
