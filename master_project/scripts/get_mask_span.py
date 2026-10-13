"""Earlier DICOM/PMCC landmark and registration-processing workflow, classified from data paths and helper callers.

Purpose: Earlier DICOM/PMCC landmark and registration-processing workflow, classified from data paths and helper callers.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: SimpleITK, numpy.
Role: main runner.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'master_project.scripts'
import SimpleITK as sitk
import numpy as np

def get_mask_z_span(filepath):
    mask = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(mask)
    z, y, x = np.where(arr!=0)
    spans = [(i.min(), i.max()) for i in (x, y, z)]
    return dict(zip('zyx', spans))

if __name__ == "__main__":
    # filepath = r'p:\yusun\self-supervised-anatomical-embedding-v2\data\lung\PMCC_ReIrrad_14\1\mask-lung-dilated.nrrd'
    import sys
    filepath = sys.argv[1]
    print(get_mask_z_span(filepath))