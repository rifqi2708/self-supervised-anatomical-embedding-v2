# Convert mask to .nrrd

"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: SimpleITK, pathlib, tqdm.
Role: shared helper.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'miscellaneous.PMCC'
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

data_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')

for f in tqdm(data_dir.iterdir()):
    f_dir, m_dir = f/'1', f/'2'

    f_mask = sitk.ReadImage((f_dir/'mask-lung-dilated.nii.gz').as_posix(), outputPixelType=sitk.sitkUInt8)
    m_mask = sitk.ReadImage((m_dir/'mask-lung-dilated.nii.gz').as_posix(), outputPixelType=sitk.sitkUInt8)

    sitk.WriteImage(f_mask, (f_dir/'mask-lung-dilated.nrrd').as_posix())
    sitk.WriteImage(m_mask, (m_dir/'mask-lung-dilated.nrrd').as_posix())
