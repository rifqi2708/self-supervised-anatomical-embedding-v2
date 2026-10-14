# Copy the DICOM CT files over
# So that masks in RTSTRUCT can be converted to nii
# The masks will be used in the registration
#

"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: pathlib, pydicom, shutil, tqdm.
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
import shutil
from tqdm import tqdm
import pydicom

def get_series_uid(folder):
    path = Path(folder)
    first_file = list(path.iterdir())[0]
    return pydicom.read_file(first_file).SeriesInstanceUID

if __name__ == "__main__":
    # IO paths
    data_dir = Path(r'\\petermac.org.au\shared\ImageStore\Reirradiation\Lung')
    save_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')

    for f in tqdm(save_dir.iterdir()):
        config = open(save_dir/f/'src.txt').readlines()
        dcm_fl_1 = [Path(i) for i in eval(config[2])]
        each_save_dir = save_dir/f/'1'/'dcm'
        each_save_dir.mkdir(exist_ok=True)
        for dcm_file in dcm_fl_1:
            shutil.copy(dcm_file, each_save_dir)
        assert (save_dir/f/'1'/f'ct-{get_series_uid(each_save_dir)}.nii.gz').exists()

        dcm_fl_2 = [Path(i) for i in eval(config[8])]
        each_save_dir = save_dir/f/'2'/'dcm'
        each_save_dir.mkdir(exist_ok=True)
        for dcm_file in dcm_fl_2:
            shutil.copy(dcm_file, each_save_dir)
        assert (save_dir/f/'2'/f'ct-{get_series_uid(each_save_dir)}.nii.gz').exists()