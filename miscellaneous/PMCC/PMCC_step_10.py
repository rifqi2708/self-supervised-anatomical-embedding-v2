# Extract the points from transformix output
# Calculate the RMSE

"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: numpy, pandas, pathlib.
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
import pandas as pd
import numpy as np
point_dir = Path(r'P:\yusun\target_registration_error\points')

def parse_tfmx_output(filepath):
    lines = open(filepath).readlines()
    ijks = [] # ijk
    crds = [] # coordinate
    for line in lines:
        p2_ijk = [int(i) for i in line.split(';')[3].strip().split()[-4:-1]]
        p2_crd = [float(i) for i in line.split(';')[4].strip().split()[-4:-1]]

        ijks.append(p2_ijk)
        crds.append(p2_crd)

    return np.array(ijks), np.array(crds)

eval_content = lambda filepath: eval(open(filepath).read())

rows = []
for f in point_dir.iterdir():
    if f.is_file(): continue

    p2_elstx_ijk, p2_elstx = parse_tfmx_output(f/"warped_point/outputpoints.txt")
    p2_sam = eval_content(f/'pt2_sam.txt')

    # p1 = [i[-1] for i in eval_content(f/'pt1.txt')]
    p2_gt = [i[-1] for i in eval_content(f/'pt2_manual.txt')]

    p2_sam = np.array(p2_sam)
    # p1 = np.array(p1)
    p2_gt = np.array(p2_gt)

    # Total registration error: RMSE between warped and gt p2
    rmse_ela = np.sqrt(((p2_elstx - p2_gt)**2).sum(1).mean())

    # With SAM
    rmse_sam = np.sqrt(((p2_sam - p2_gt)**2).sum(1).mean())

    row = (f.stem, rmse_ela, rmse_sam)
    rows.append(row)

df = pd.DataFrame(rows, columns=['Label', 'RMSE TRE', 'RMSE SAM']).set_index('Label')
print(df)

#                   RMSE TRE   RMSE SAM
# Label
# PMCC_ReIrrad_01   5.091619   5.108097
# PMCC_ReIrrad_02   3.697893   7.109954
# PMCC_ReIrrad_03   9.542844  10.883492
# PMCC_ReIrrad_06   3.571964   6.211316
# PMCC_ReIrrad_07   5.230860   8.697621
# PMCC_ReIrrad_08   5.028281   4.148137
# PMCC_ReIrrad_09   7.255336   3.315698
# PMCC_ReIrrad_10   3.864688   4.433843
# PMCC_ReIrrad_11  14.944858   9.826613
# PMCC_ReIrrad_14   5.616109   3.465876

# In [134]: df.mean(0)
# Out[134]:
# RMSE TRE    6.384445
# RMSE SAM    6.320065
