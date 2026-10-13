# SuperPoint characterization investigations

The smoke, representative-slice, multislice, full-volume and threshold gates retain their original inputs and limits. They export detector-characterization evidence rather than run matching. Shared mask grouping and CSV routines live in `publication/query_points/superpoint/detector_review.py`; detector/model utilities remain beside the method.

```bash
python -m publication.investigations.superpoint.superpoint_threshold_gate --help
```

Use the pinned external SuperPoint repository/weights and explicit review output paths. Old `tools.quadra.superpoint_*` invocations remain supported. See [workflow commands](../../README.md).
