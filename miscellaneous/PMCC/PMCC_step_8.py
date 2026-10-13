# Prepare points.txt for transformix

"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: pathlib, shutil.
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
import shutil

data_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')
save_dir = Path(r'P:\yusun\target_registration_error\points')

def make_point_file(pts):
    lines = ['index']
    lines.append(str(len(pts)))
    for pt in pts:
        lines.append('{} {} {}'.format(*pt))

    lines = [line+'\n' for line in lines]
    return lines

for f in data_dir.iterdir():
    img_pts = eval(open(f / 'points' / 'pt1_ijk.txt').read())
    pt_file_content = make_point_file(img_pts)
    each_save_dir = save_dir / f.stem
    if not each_save_dir.exists(): each_save_dir.mkdir(exist_ok=True)

    with open(each_save_dir / 'pt1_elastix_input.txt', 'w') as outfile:
        outfile.writelines(pt_file_content)

    shutil.copy(f/'points'/'pt1_ijk.txt', each_save_dir/'pt1_ijk.txt')
    shutil.copy(f/'points'/'pt2.txt', each_save_dir/'pt2_sam.txt')
    shutil.copy(f/'points'/'pt2_gt.txt', each_save_dir/'pt2_manual.txt')
