# Harris Laplacian query points

`interior_keypoint_gate.py` implements native-grid detection, physical suppression, deterministic selection and inspection export. It dispatches `interior_keypoint_batch.py` for mask-directory orchestration. Selection behavior and candidate-supply safeguards are unchanged.

Discover options without inference:

```bash
python -m publication.query_points.harris_laplacian.interior_keypoint_gate --help
```

Use explicit input CT/mask paths and an external review output directory. These commands stop at query-point inspection; they do not run UAE or establish anatomical correspondence. Shared checksums come from `publication/shared/hashing.py`, not from a SuperPoint detector. Existing `tools.quadra.interior_keypoint_gate` invocation is supported. See [workflow commands](../../README.md) and tests under `tests/publication/query_points_harris_laplacian/`.
