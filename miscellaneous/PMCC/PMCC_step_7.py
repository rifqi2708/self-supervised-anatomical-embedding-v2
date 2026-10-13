# Case 11 and 14 don't work well.
# Crop the data using the lung mask and re-do

"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: SimpleITK, miscellaneous, numpy, pathlib.
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
import numpy as np
from pathlib import Path
from miscellaneous.PMCC.PMCC_step_5 import find_ct_file

def get_mask_z_span(filepath):
    mask = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(mask)
    z, y, x = np.where(arr!=0)

    # Use .tolist() so that the returned value is int not np.int
    # np.int will cause sitk break
    spans = [(min(i.tolist()), max(i.tolist())) for i in (z, y, x)]
    return dict(zip('zyx', spans))

def crop_z(img, z1, z2, margin=None):
    # z1, z2, range of z. z2>z1
    # z2 not inclusive like Python slicing
    # sitk.Extract(img, output_size, start_index), start_index inclusive
    # margin: the extra slices to include, clipped by the image range

    x, y, z = img.GetSize()

    # Add margin and clipe to (0, z-1)
    if margin: z1, z2 = max(0, z1 - margin), min(z, z2 + margin)

    return sitk.Extract(img, [x, y, z2-z1], [0, 0, z1])

data_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')
target_folders = ['PMCC_ReIrrad_11', 'PMCC_ReIrrad_14']

for f in data_dir.iterdir():
    if f.stem not in target_folders: continue

    f_dir, m_dir = f/'1', f/'2'

    f_img = sitk.ReadImage(find_ct_file(f_dir).as_posix())
    m_img = sitk.ReadImage(find_ct_file(m_dir).as_posix())

    f_img_cropped = crop_z(
        f_img,
        *get_mask_z_span((f_dir/'mask-lung-dilated.nrrd').as_posix())['z'],
        margin=5
    )

    m_img_cropped = crop_z(
        m_img,
        *get_mask_z_span((m_dir/'mask-lung-dilated.nrrd').as_posix())['z'],
        margin=5
    )

    sitk.WriteImage(f_img_cropped, (f/'fixed_cropped.nii.gz').as_posix())
    sitk.WriteImage(m_img_cropped, (f/'moving_cropped.nii.gz').as_posix())

    print(f)
