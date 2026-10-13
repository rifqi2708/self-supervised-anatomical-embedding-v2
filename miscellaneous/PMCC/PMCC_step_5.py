# Convert Lung_L + Lung_R as the NIfTI file
# Dilate so that the borders are included

# elastix -f fixed.nii.gz -m moving.nii.gz -out out_dir -p parameters_BSpline.txt
#  -fMask    mask for fixed image
#   -mMask    mask for moving image

"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: SimpleITK, nibabel, numpy, pathlib, pydicom, rt_utils, tqdm.
Role: main runner.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'miscellaneous.PMCC'
from pathlib import Path
import pydicom
from tqdm import tqdm
from rt_utils import RTStructBuilder
import nibabel
import SimpleITK as sitk
import numpy as np

def get_series_uid(folder):
    path = Path(folder)
    first_file = list(path.iterdir())[0]
    return pydicom.read_file(first_file).SeriesInstanceUID

def find_dcm_file(folder, modality='CT'):
    path = Path(folder)
    found = [f for f in path.iterdir() if f.suffix=='.dcm' and pydicom.read_file(f).Modality==modality]
    assert len(found)==1
    return found[0]

def find_ct_file(folder):
    path = Path(folder)
    found = [f for f in path.iterdir() if f.stem.startswith('ct') and f.suffix=='.gz']
    assert len(found)==1
    return found[0]


def make_roi_nii(arr, nft):
    '''Make an object using meta information from nft, using imaging data from arr'''
    # nft.header.set_data_dtype(np.uint8)
    return nibabel.Nifti1Image(arr, nft.affine, nft.header)

def process(folder, dilate_filter):
    path = Path(folder)
    rs_file = find_dcm_file(path, 'RTSTRUCT')
    ct_file = find_ct_file(path)

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=save_dir/path/'dcm',
        rt_struct_path=rs_file
    )

    lung_roi = np.logical_or(
        rtstruct.get_roi_mask_by_name('Lung_L'), # bool
        rtstruct.get_roi_mask_by_name('Lung_R')
    )

    mask_nii = make_roi_nii(np.moveaxis(lung_roi, 0, 1), nibabel.load(ct_file))
    nibabel.save(mask_nii, path/'mask-lung.nii.gz')

    # NIfTI will have the intrinsic scaling calculation, hence its dtype is usually float
    # To read in as uint8, use outputPixelType argument
    # Should be saved as .nrrd or .mha

    mask_sitk = sitk.ReadImage(
        fileName=(path/'mask-lung.nii.gz').as_posix(),
        outputPixelType=sitk.sitkUInt8
    )
    mask_sitk_dilated = dilate_filter.Execute(mask_sitk)
    sitk.WriteImage(mask_sitk_dilated, (path/'mask-lung-dilated.nii.gz').as_posix())

if __name__ == "__main__":

    # IO paths
    data_dir = Path(r'\\petermac.org.au\shared\ImageStore\Reirradiation\Lung')
    save_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')

    # Create the dilator
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(3)
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetForegroundValue(1)

    for f in tqdm(save_dir.iterdir()):

        process(f/'1', dilate_filter)
        process(f/'2', dilate_filter)
