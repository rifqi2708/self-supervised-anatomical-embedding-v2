# Pinned independent UAE reference

The three `uae_*_035bbfc.py` files are unmodified bytes fetched from
[Alibaba DAMO Academy's released repository at commit 035bbfc1816e40c122303a5e21accbdc35beac43](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2/tree/035bbfc1816e40c122303a5e21accbdc35beac43):

- `tools/demo_semantic_stable_points.py`
- `tools/interfaces.py`
- `tools/utils.py`

Copyright (c) 2023 Alibaba Damo Academy. Distributed under the repository's MIT
license, reproduced in `LICENSE` in this directory. Original source headers and
line endings are retained. SHA-256 identities are checked by
`reviewed_uae_matching.reference_identity()` before executing the reference.

The fixture adapter AST-loads only the original structural-inference and anchor
constructor functions. It bypasses heavy model imports and injects retrieval as
a declared dependency. Its final return is instrumented to expose original local
arrays for trace comparison. It does not call the production fixed-point wrapper.

This establishes bounded independent structural-code comparison. It does not
establish original CUDA retrieval parity, model/extraction equivalence, GPU
feasibility or anatomical accuracy. Those remain real-pilot gates.
