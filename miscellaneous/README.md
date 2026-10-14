# Other contributor workflows and historical support

PMCC, CT–MRI/lymph-node/SLIC demonstrations, independent reconstruction notes, unused Docker/dev-container settings and the unused DEEDS Makefile are retained here. Classification follows reviewed purpose and scope, not author alone.

PMCC runners and directly required matching/landmark helpers are colocated in `PMCC/`. Demo model interfaces still come from root `shared/`. Historical absolute data paths are preserved; these scripts are not safe generic import targets and must be configured manually before scientific execution. Missing DEEDS source files remain a known historical completeness limitation.

The CT–MR demo figure output uses QUADRA_DEMO_OUTPUT_ROOT when supplied (set it to the external archive for retained results); its fallback is the existing ignored data/quadra_output area. It no longer writes beneath obsolete tools/assets paths. Slicer examples retain their manually configured Windows repository root, with demonstration point files now under junk/.
