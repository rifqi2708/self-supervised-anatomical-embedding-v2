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
