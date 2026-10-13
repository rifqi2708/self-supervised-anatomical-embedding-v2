# PMCC workflow

The ten numbered PMCC scripts, `matcher.py` and `pmcc_rmse.py` form the reviewed contributor workflow. Step 7 calls step 5; steps 1–3 use landmark helpers and step 2 uses the matcher. Cross-project model interfaces and image utilities remain in root `shared/`.

Imports now use the canonical package; old `tools.PMCC_*`, `tools.matcher` and `tools.pmcc_rmse` paths forward to it. Native Windows/network data settings are unchanged. Review the script's input/output globals and original study context before execution; no dataset transfer or scientific run is part of source reorganization.
