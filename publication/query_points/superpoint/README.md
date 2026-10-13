# SuperPoint query points

Reusable CT preparation, native/model coordinate conversion and pinned model provenance live in `superpoint_adapter.py`. `detector_review.py` owns shared mask grouping and CSV export support. The external SuperPoint checkout and weights remain external inputs.

Earlier smoke, representative-slice, multislice, full-volume and threshold surveys are under `publication/investigations/superpoint/`. These characterize candidate supply; they do not establish anatomical matching accuracy or implement a final cohort selection policy.

Run commands from the repository root. Existing `tools.quadra.superpoint_*` entry points forward to the canonical modules. See [workflow commands](../../README.md). Method software checks are under `tests/publication/query_points_superpoint/`.
