# Step 3 - visualise the results

"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: SimpleITK, matplotlib, miscellaneous, pathlib, tqdm.
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
from miscellaneous.PMCC.pmcc_rmse import Mapper
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm

# For Slicer
# pts = [[12.890625, 90.95938110351562, 30.0], [-43.359375, 94.47500610351562, 34.0], ...]
def add_fiducial_pts(pts):
    node_id = slicer.modules.markups.logic().AddNewFiducialNode()
    node = getNode(node_id)
    for x, y, z in pts:
        node.AddControlPoint(-x, -y, z)
    return node

def get_img_files(folder):
    dir1 = folder / '1'
    dir2 = folder / '2'
    im1_file = [i for i in dir1.iterdir() if i.name.startswith('ct')][0]
    im2_file = [i for i in dir2.iterdir() if i.name.startswith('ct')][0]
    return im1_file, im2_file

def plot_slice_pt(slices, pts, outfile, title='Marker',
                  colours=('orange', 'yellow', 'green'),
                  subtitles=('CT 1', 'CT 2 Found', 'CT 2 GT')):
    fig, ax = plt.subplots(1, 3, figsize=(15,6))
    axs = ax.ravel()

    for (a, s, p, c, t) in zip(axs, slices, pts, colours, subtitles):
        a.imshow(s, cmap='gray')
        a.plot(p[0], p[1], 'o', c=c)
        a.set_title(t+f'\nz={p[-1]}')
        a.axis('off')

    fig.suptitle(title)
    fig.savefig(outfile)
    fig.clf()
    plt.close()

def plot_img_pts(im1, im2, pts1, pts2, pts2_gt, outdir):
    for i, (pt1, pt2, pt2_gt) in tqdm(enumerate(zip(pts1, pts2, pts2_gt), start=1)):
        slice1 = im1[pt1[-1],...]
        slice2 = im2[pt2[-1],...]
        slice3 = im2[pt2_gt[-1], ...]

        title = f'Fiducial-Point-{i}'
        outfile = Path(outdir) / f'{title}.png'
        plot_slice_pt([slice1, slice2, slice3], [pt1, pt2, pt2_gt], outfile, title)

if __name__ == "__main__":
    data_dir = Path('data/lung')

    for folder in data_dir.iterdir():
        pts_dir = folder / 'points'
        plot_dir = folder / 'plot'
        plot_dir.mkdir(exist_ok=True)

        pt2 = eval(open(pts_dir/'pt2.txt').read())
        pt2_ijk = eval(open(pts_dir/'pt2_ijk.txt').read())
        pt1_ijk = eval(open(pts_dir/'pt1_ijk.txt').read())
        pt2_gt = eval(open(pts_dir/'pt2_gt.txt').read())
        pt2_gt = [i[-1] for i in pt2_gt]

        im1_file, im2_file = get_img_files(folder)
        m2 = Mapper(sitk.ReadImage(im2_file))
        pt2_gt_ijk = [m2.lps2ijk(*pt).tolist() for pt in pt2_gt]

        ct1 = sitk.GetArrayFromImage(sitk.ReadImage(im1_file))
        ct2 = sitk.GetArrayFromImage(sitk.ReadImage(im2_file))

        plot_img_pts(ct1, ct2, pt1_ijk, pt2_ijk, pt2_gt_ijk, plot_dir)

        print(folder, 'completed')

        # break


# # Multipage tiff (only grey level allowed, can't do RGB)
# import skimage.io
# import numpy as np
# from pathlib import Path
# import tifffile

# fl = [i.as_posix() for i in list(Path('.').iterdir())]
# fl.sort(key=lambda x: int(x.split('.')[0].split('-')[-1]))

# imgs = [skimage.io.imread(i) for i in fl]
# imgs_mean = [i.mean(-1).astype(np.uint8) for i in imgs]
# im = np.stack(imgs_mean, -1)
# im = im.swapaxes(0, -1)
# im = im.swapaxes(1, 2) # shape: (18, 600, 1500)
# tifffile.imwrite('test.tiff', im)
