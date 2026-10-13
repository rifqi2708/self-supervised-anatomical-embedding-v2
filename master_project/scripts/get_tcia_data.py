"""Earlier DICOM/PMCC landmark and registration-processing workflow, classified from data paths and helper callers.

Purpose: Earlier DICOM/PMCC landmark and registration-processing workflow, classified from data paths and helper callers.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: SimpleITK, tcia_utils.
Role: shared helper.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'master_project.scripts'
from tcia_utils import nbia
import SimpleITK as sitk

collection = 'CT Lymph Nodes'
df = nbia.getBodyPartCounts(collection, format = "df")

series = nbia.getSeries(collection, format = "df")
random_row = series.loc[series['Modality'] == 'SEG'].sample(n=1)
segSeries = random_row['SeriesInstanceUID'].iloc[0]
refSeries = nbia.getSegRefSeries(segSeries)

nbia.downloadSeries([refSeries, segSeries], input_type="list", format='df')

def dcm2nii(indir, outfile):
  reader = sitk.ImageSeriesReader()
  fl = reader.GetGDCMSeriesFileNames(indir)
  reader.SetFileNames(fl)
  img = reader.Execute()
  sitk.WriteImage(img, outfile)

dcm2nii('tciaDownload/61.7.137032640290615606726307597263209689724/', 'test.nii.gz')
