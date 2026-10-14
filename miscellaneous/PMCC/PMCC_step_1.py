# Sort the DCM files into two scans, each with a CT and RTStruct

"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: miscellaneous, pathlib.
Role: shared helper.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'miscellaneous.PMCC'
from miscellaneous.PMCC.pmcc_rmse import sort_files, transfer_files
from pathlib import Path

data_dir = Path(r'\\petermac.org.au\shared\ImageStore\Reirradiation\Lung')
save_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')
for folder in data_dir.iterdir():
    # Skip 300xxx folders
    if folder.name.startswith('300'): continue

    each_save_dir = save_dir / folder.name
    if not each_save_dir.exists(): each_save_dir.mkdir(exist_ok=True)

    file_info = sort_files(folder)
    transfer_files(file_info, each_save_dir)
