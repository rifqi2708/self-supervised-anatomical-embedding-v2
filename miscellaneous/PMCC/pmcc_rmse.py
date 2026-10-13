"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: SimpleITK, numpy, pathlib, pydicom, shutil.
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
import SimpleITK as sitk
import numpy as np
import shutil

# RTStruct
def get_points(rs_file):
    rs = pydicom.read_file(rs_file)
    names = dict([(i.ROINumber, i.ROIName) for i in rs.StructureSetROISequence])
    pts = [
        (i.ReferencedROINumber, i.ContourSequence[0].ContourData)
        for i in rs.ROIContourSequence
        if i.ContourSequence[0].ContourGeometricType == "POINT"
    ]

    return sorted(
        [(num, names[num], coord) for num, coord in pts], key=lambda x: int(x[0])
    )


class Mapper:
    """
    Note that GetOrigin() will return opposite signs for the first two numbers between Slicer and SITK
    https://discourse.slicer.org/t/discrepancy-in-image-origin-values-between-3d-slicer-and-simpleitk/35946/2
    https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html
    Slicer: RAS
    SimpleITK: LPS
    """

    def __init__(self, node):
        self.node = node
        self.matrix = self.getMatrix(self.node)
        self.matrix_inv = np.linalg.inv(self.matrix)

    @staticmethod
    def getMatrix(node):
        ori = node.GetOrigin()
        spa = node.GetSpacing()

        matrix = np.zeros((4, 4), np.float32)
        matrix[0, 0] = spa[0]
        matrix[1, 1] = spa[1]
        matrix[2, 2] = spa[2]
        matrix[3, 3] = 1

        matrix[:3, 3] = ori

        return matrix

    @staticmethod
    def applyMatrix(matrix, coord):
        if len(coord) != 4:
            coord = np.array([*coord, 1])
        return (matrix @ coord)[:3]

    def ijk2lps(self, *coord):
        return self.applyMatrix(self.matrix, coord)

    def lps2ijk(self, *coord):
        return self.applyMatrix(self.matrix_inv, coord).astype(int)


def get_dcm_info(infile:str):
    dcm = pydicom.read_file(infile)
    return dcm.Modality, dcm.FrameOfReferenceUID


def dcm2nii(fl:list, outfile:str):
    img_reader = sitk.ImageSeriesReader()
    img_reader.SetFileNames(fl)
    img = img_reader.Execute()
    sitk.WriteImage(img, outfile)


def get_remain_files(data_dir:Path, filelists):
    files = set([i.name for i in data_dir.iterdir()])
    for fl in filelists:
        fnames = set([Path(i).name for i in fl])
        files = files - fnames
    return [data_dir / f for f in files if f.endswith('.dcm')]


def get_rtstruct_file(fnames:list, frameUID:str):
    dcms = {f: pydicom.read_file(f) for f in fnames}
    output = {
        i: dcm
        for i, dcm in dcms.items()
        if dcm.Modality == "RTSTRUCT" and dcm.FrameOfReferenceUID == frameUID
    }
    # assert len(output) == 1
    return list(output.keys())


def sort_files(data_dir):
    '''Scan a folder and returns
    {seriesUID: (ct_files, rs_file)}
    '''
    data_dir = Path(data_dir)
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(data_dir.as_posix())

    filelists = {i: reader.GetGDCMSeriesFileNames(data_dir.as_posix(), i) for i in ids}
    info = {
        i: get_dcm_info(fl[0]) for i, fl in filelists.items()
    }  # seriesUID: (Mod, FrameUID)

    print(info)

    remaining_files = get_remain_files(data_dir, filelists.values())

    rs_files = {
        seriesUID: get_rtstruct_file(remaining_files, frameUID)
        for seriesUID, (mod, frameUID) in info.items() if mod=='CT'
    }

    return {
        seriesUID: (filelists[seriesUID], rs_files[seriesUID])
        for seriesUID, (mod, frameUID) in info.items() if mod=='CT'
    }

def transfer_files(file_info, out_dir):
    for i, (seriesUID, (ct_files, rs_files)) in enumerate(file_info.items(), start=1):
        each_out_dir = out_dir / str(i)
        each_out_dir.mkdir(exist_ok=True)
        date = pydicom.read_file(ct_files[0]).SeriesDate
        dcm2nii(ct_files, (each_out_dir/f'ct-{seriesUID}.nii.gz').as_posix())
        for rs_file in rs_files: shutil.copy(rs_file, each_out_dir)

        print(
            f'{i}\n{seriesUID}\n{ct_files}\n{rs_files}\n\n',
            file=open(out_dir/'src.txt', 'a')
        )


if __name__ == "__main__":
    data_dir = Path(r'\\petermac.org.au\shared\ImageStore\Reirradiation\Lung')
    save_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')
    for folder in data_dir.iterdir():
        # Skip 300xxx folders
        if folder.name.startswith('300'): continue

        each_save_dir = save_dir / folder.name
        if not each_save_dir.exists(): each_save_dir.mkdir(exist_ok=True)

        file_info = sort_files(folder)
        transfer_files(file_info, each_save_dir)



    # # Example
    # data_dir = Path(
    #     r"\\petermac.org.au\shared\ImageStore\Reirradiation\Lung\PMCC_ReIrrad_14"
    # )

    # reader = sitk.ImageSeriesReader()

    # # ids = reader.GetGDCMSeriesIDs(data_dir.as_posix())
    # # print(ids)
    # # # ('1.2.246.352.221.4988577021572189020.6098343126741388731',
    # # #  '1.2.246.352.221.5330343737108144152.7175712638069911460',
    # # #  '1.2.246.352.221.5655056896598563174.9427470545387792538')

    # # mod = pydicom.read_file(reader.GetGDCMSeriesFileNames(data_dir.as_posix(), ids[0])[0]).Modality
    # # print(mod)

    # fl = reader.GetGDCMSeriesFileNames(
    #     data_dir.as_posix(), "1.2.246.352.221.5655056896598563174.9427470545387792538"
    # )
    # reader.SetFileNames(fl)
    # image = reader.Execute()
    # mapper = Mapper(image)

    # pts = get_points(data_dir / "RS.PMCC_ReIrrad_14.CT_LM_062017.dcm")
    # pts_ijk = [mapper.lps2ijk(*pt) for (num, label, pt) in pts]
    # print(pts)

    # # Can now get the ijk for the reference image points
