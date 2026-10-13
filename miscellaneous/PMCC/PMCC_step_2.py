# Step 2 - Match the points

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
from pathlib import Path
from miscellaneous.PMCC.matcher import Match, init
from miscellaneous.PMCC.pmcc_rmse import get_points, Mapper
import SimpleITK as sitk
import numpy as np

data_dir = Path('data/lung')

config_file = "configs/sam/sam_NIHLN.py"
checkpoint_file = "checkpoints/SAM.pth"
model = init(config_file, checkpoint_file)

def check_uniqueness(*filelists):
    assert all([len(fl)==1 for fl in filelists])

def get_files(folder):
    '''Given the patient folder, returns the RTStruct, ct1, ct2 file paths'''
    dir1, dir2 = folder/'1', folder/'2'
    rs1_files = [i for i in dir1.iterdir() if i.name.startswith('RS')]
    rs2_files = [i for i in dir2.iterdir() if i.name.startswith('RS')]
    im1_files = [i for i in dir1.iterdir() if i.name.startswith('ct')]
    im2_files = [i for i in dir2.iterdir() if i.name.startswith('ct')]

    check_uniqueness(rs1_files, rs2_files, im1_files, im2_files)

    return rs1_files[0], rs2_files[0], im1_files[0], im2_files[0]

def cal_rmsd(pts1, pts2):
    rmsds = [np.sqrt(((np.array(pt1) - np.array(pt2))**2).sum()) for pt1, pt2 in zip(pts1, pts2)]
    return np.mean(rmsds), rmsds

for folder in data_dir.iterdir():
    each_save_dir = folder / 'points'
    each_save_dir.mkdir(exist_ok=True)
    print(each_save_dir)

    # Get pt1 points
    rs1_file, rs2_file, im1_file, im2_file = get_files(folder)
    pt1 = get_points(rs1_file)
    m1 = Mapper(sitk.ReadImage(im1_file))
    pt1_ijk = [m1.lps2ijk(*i[-1]).tolist() for i in pt1]

    print(pt1, file=open(each_save_dir/'pt1.txt', 'w'))
    print(pt1_ijk, file=open(each_save_dir/'pt1_ijk.txt', 'w'))

    # Apply SAM to get the pt2_ijk
    matcher = Match(im1_file.as_posix(), im2_file.as_posix(), model)
    pt2_ijk = [matcher.match(i).tolist() for i in pt1_ijk]

    # Convert pt2_ijk to LPS
    m2 = Mapper(sitk.ReadImage(im2_file))
    pt2 = [m2.ijk2lps(*i).tolist() for i in pt2_ijk]

    # Get GT pt2
    pt2_gt = get_points(rs2_file)
    print(pt2, file=open(each_save_dir/'pt2.txt', 'w'))
    print(pt2_ijk, file=open(each_save_dir/'pt2_ijk.txt', 'w'))
    print(pt2_gt, file=open(each_save_dir/'pt2_gt.txt', 'w'))

    # Calculate RMSD
    pt2_gt_lps = [i[-1] for i in pt2_gt]
    rmsd = cal_rmsd(pt2, pt2_gt_lps)
    print(rmsd, file=open(each_save_dir/'rmsd.txt', 'w'))
    print(rmsd[0])

# data/lung/PMCC_ReIrrad_10/points
# 4.0234424598921095
# data/lung/PMCC_ReIrrad_09/points
# 3.095134101130921
# data/lung/PMCC_ReIrrad_14/points
# 2.9818021217124544
# data/lung/PMCC_ReIrrad_07/points
# 6.305258296738616
# data/lung/PMCC_ReIrrad_01/points
# 4.210557908589212
# data/lung/PMCC_ReIrrad_08/points
# 3.8968589689413697
# data/lung/PMCC_ReIrrad_02/points
# 6.662778374667537
# data/lung/PMCC_ReIrrad_03/points
# 8.384338614074593
# data/lung/PMCC_ReIrrad_11/points
# 7.317384755689669
# data/lung/PMCC_ReIrrad_06/points
# 5.991231764902554
