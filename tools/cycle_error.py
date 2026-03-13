# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import os
import sys
import time
import torch
import numpy as np

sys.path.append("..")
sys.path.append(".")

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print ('Using GPU')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  
    print ('Using CPU')

from interfaces import init, get_embedding, get_sim_embed_loc
from utils import read_image


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project
CONFIG_FILE = "configs/sam/sam_NIHLN.py"
CHECKPOINT_FILE = "checkpoints/SAM.pth"
DEFAULT_IM1_FILE = "data/raw_data/NIH_lymph_node/ABD_LYMPH_001.nii.gz"
DEFAULT_IM2_FILE = "data/raw_data/NIH_lymph_node/ABD_LYMPH_002.nii.gz"

#loading data and embeddings
def load_context(im_file, model, is_mri=False):
    img, normed_im, norm_info = read_image(im_file, is_MRI=is_mri)
    embedding = get_embedding(normed_im, model)
    image_shape = img["shape"]
    if len(image_shape) != 4:
        raise ValueError(f"Unexpected image shape from read_image: {image_shape}")
    target_imshape = (image_shape[3], image_shape[1], image_shape[2])
    return {
        "im_file": im_file,
        "img": img,
        "norm_info": np.array(norm_info, dtype=float),
        "embedding": embedding,
        "target_imshape": target_imshape,
    }

#find corresponding point a pixel in query_ctx at the key ctx, and return the matched point and similarity score
def match_point(pt_query, query_ctx, key_ctx, use_sim_coarse=True):
    pt_query = np.asarray(pt_query, dtype=float)
    pt_query_normed = pt_query * query_ctx["norm_info"]
    pt_match, score = get_sim_embed_loc(
        query_ctx["embedding"],
        key_ctx["embedding"],
        pt_query_normed,
        key_ctx["target_imshape"],
        norm_info=key_ctx["norm_info"],
        write_sim=False,
        use_sim_coarse=use_sim_coarse,
    )
    return np.asarray(pt_match, dtype=int), float(score)

#find the corresponding point in ctx_ba for a point in ctx_ab, then find the corresponding point back in ctx_ab, and compute the cycle error in both voxel and mm space
def compute_cycle_for_point(pt1, ctx_ab, ctx_ba, use_sim_coarse=True):
    pt1 = np.asarray(pt1, dtype=int)
    pt2, score_12 = match_point(pt1, ctx_ab, ctx_ba, use_sim_coarse=use_sim_coarse)
    pt1_back, score_21 = match_point(pt2, ctx_ba, ctx_ab, use_sim_coarse=use_sim_coarse)

    delta = pt1_back.astype(float) - pt1.astype(float)
    voxel_error = float(np.linalg.norm(delta))
    spacing = np.asarray(ctx_ab["img"]["spacing"], dtype=float)
    mm_error = float(np.linalg.norm(delta * spacing))

    return {
        "pt1": pt1,
        "pt2": pt2,
        "pt1_back": pt1_back,
        "score_12": score_12,
        "score_21": score_21,
        "voxel_error": voxel_error,
        "mm_error": mm_error,
    }

#Select random points within the image with certain threshold
def sample_random_foreground_points(img, num_points, seed):
    img = img["img"]
    bg_th = img.min() + (img.max() - img.min()) / 10.0
    foreground_points = np.argwhere(img > bg_th)
    if foreground_points.size == 0:
        raise RuntimeError("No foreground points found with current thresholding strategy.")

    rng = np.random.default_rng(seed)
    replace = len(foreground_points) < num_points
    if replace:
        print(
            f"WARNING: requested {num_points} points but found {len(foreground_points)} foreground points; "
            "sampling with replacement."
        )
    selected_ids = rng.choice(len(foreground_points), size=num_points, replace=replace)
    return foreground_points[selected_ids].astype(int)

# Validate that a fixed point is within the image bounds and return it as a numpy array
def validate_fixed_point(fixed_point, img):
    pt = np.asarray(fixed_point, dtype=int)
    if pt.shape != (3,):
        raise ValueError(f"Fixed point must have 3 coordinates, got {fixed_point}")

    shape = np.asarray(img["img"].shape, dtype=int)
    if np.any(pt < 0) or np.any(pt >= shape):
        raise ValueError(
            f"Fixed point {pt.tolist()} out of bounds for image shape {shape.tolist()} "
            "(expected x,y,z indices in range [0, shape_i-1])."
        )
    return pt

# Compute summary statistics for a list of values
def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }

# Print a formatted table of results for each point
def print_result_table(results):
    print("idx | pt1(x,y,z) | pt2(x,y,z) | pt1_back(x,y,z) | err_voxel | err_mm | score_12 | score_21")
    for idx, record in enumerate(results):
        print(
            f"{idx:03d} | "
            f"{record['pt1'].tolist()} | {record['pt2'].tolist()} | {record['pt1_back'].tolist()} | "
            f"{record['voxel_error']:.4f} | {record['mm_error']:.4f} | "
            f"{record['score_12']:.6f} | {record['score_21']:.6f}"
        )

# Compute and print summary statistics for voxel and mm errors across all points
def print_summary(results):
    voxel_errors = [record["voxel_error"] for record in results]
    mm_errors = [record["mm_error"] for record in results]
    voxel_stats = summarize(voxel_errors)
    mm_stats = summarize(mm_errors)

    print("\nCycle error summary")
    print(
        "voxel: "
        f"count={voxel_stats['count']} mean={voxel_stats['mean']:.4f} median={voxel_stats['median']:.4f} "
        f"std={voxel_stats['std']:.4f} min={voxel_stats['min']:.4f} max={voxel_stats['max']:.4f} "
    )
    print(
        "mm:    "
        f"count={mm_stats['count']} mean={mm_stats['mean']:.4f} median={mm_stats['median']:.4f} "
        f"std={mm_stats['std']:.4f} min={mm_stats['min']:.4f} max={mm_stats['max']:.4f} "
    )

# Main function to run the cycle consistency evaluation
def run_cycle(
    im1_file=DEFAULT_IM1_FILE,
    im2_file=DEFAULT_IM2_FILE,
    point_mode="random",
    fixed_point=None,
    num_points=20,
    seed=0,
    is_mri=False,
    use_sim_coarse=True,
):
    if point_mode not in ("random", "fixed"):
        raise ValueError("point_mode must be either 'random' or 'fixed'")

    if point_mode == "random" and num_points < 1:
        raise ValueError("num_points must be >= 1 when point_mode='random'")

    for im_path in (im1_file, im2_file):
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Image file not found: {im_path}")

    time1 = time.time()
    model = init(CONFIG_FILE, CHECKPOINT_FILE)
    time2 = time.time()
    print(f"model loading time: {time2 - time1:.3f}s")

    ctx1 = load_context(im1_file, model, is_mri=is_mri)
    ctx2 = load_context(im2_file, model, is_mri=is_mri)
    time3 = time.time()
    print(f"image+embedding loading time: {time3 - time2:.3f}s")

    if point_mode == "fixed":
        if fixed_point is None:
            raise ValueError("fixed_point must be provided when point_mode='fixed'")
        points = np.asarray([validate_fixed_point(fixed_point, ctx1["img"])])
    else:
        points = sample_random_foreground_points(ctx1["img"], num_points, seed)

    results = []
    time4 = time.time()
    for point in points:
        result = compute_cycle_for_point(point, ctx1, ctx2, use_sim_coarse=use_sim_coarse)
        results.append(result)
    time5 = time.time()
    print(f"cycle matching time: {time5 - time4:.3f}s")

    print_result_table(results)
    print_summary(results)


if __name__ == "__main__":
    try:
        # Edit this single call to change behavior:
        # 1) Random mode: point_mode="random", num_points=20, seed=0
        # 2) Fixed mode: point_mode="fixed", fixed_point=(93, 139, 44)
        run_cycle(
            im1_file=DEFAULT_IM1_FILE,
            im2_file=DEFAULT_IM2_FILE,
            point_mode="random",
            fixed_point=None,
            num_points=20,
            seed=0,
            is_mri=False,
            use_sim_coarse=True,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)




# Executing task: /usr/local/bin/python /root/self-supervised-anatomical-embedding-v2/tools/cycle_error.py 

# Using CPU
# 2026-03-12 11:23:44,658 - mmdet - INFO - load model from: torchvision://resnet18
# load checkpoint from torchvision path: torchvision://resnet18
# 2026-03-12 11:23:44,728 - mmdet - WARNING - Module not exist in the state_dict_r2d: layer1.0.downsample.0
# 2026-03-12 11:23:44,728 - mmdet - WARNING - Module not exist in the state_dict_r2d: layer1.0.downsample.1
# 2026-03-12 11:23:44,761 - mmdet - INFO - These parameters in the 2d checkpoint are not loaded: {'fc.weight', 'fc.bias'}
# load checkpoint from local path: checkpoints/SAM.pth
# model loading time: 0.603s
# image+embedding loading time: 46.528s
# cycle matching time: 72.890s
# idx | pt1(x,y,z) | pt2(x,y,z) | pt1_back(x,y,z) | err_voxel | err_mm | score_12 | score_21
# 000 | [161, 62, 18] | [138, 33, 19] | [158, 62, 16] | 3.6056 | 11.6619 | 0.671904 | 0.681867
# 001 | [88, 132, 54] | [84, 106, 57] | [89, 132, 53] | 1.4142 | 5.3852 | 0.806794 | 0.776706
# 002 | [84, 39, 87] | [70, 14, 85] | [86, 40, 86] | 2.4495 | 6.7082 | 0.637367 | 0.642692
# 003 | [164, 139, 47] | [157, 112, 48] | [158, 142, 47] | 6.7082 | 13.4164 | 0.709771 | 0.737226
# 004 | [112, 38, 15] | [98, 14, 15] | [116, 22, 11] | 16.9706 | 38.5746 | 0.633192 | 0.627932
# 005 | [123, 24, 27] | [104, 12, 26] | [120, 23, 26] | 3.3166 | 8.0623 | 0.714992 | 0.722201
# 006 | [117, 139, 60] | [112, 110, 62] | [116, 138, 60] | 1.4142 | 2.8284 | 0.690407 | 0.709733
# 007 | [172, 79, 32] | [162, 44, 36] | [175, 64, 33] | 15.3297 | 31.0000 | 0.625375 | 0.795101
# 008 | [126, 111, 55] | [128, 92, 53] | [126, 110, 55] | 1.0000 | 2.0000 | 0.657469 | 0.668966
# 009 | [128, 10, 31] | [112, 0, 32] | [128, 0, 1] | 31.6228 | 151.3275 | 0.565456 | 0.811534
# 010 | [71, 170, 49] | [62, 155, 49] | [74, 186, 49] | 16.2788 | 32.5576 | 0.633842 | 0.680743
# 011 | [147, 143, 54] | [142, 111, 52] | [140, 138, 50] | 9.4868 | 26.3818 | 0.679263 | 0.737479
# 012 | [126, 15, 56] | [116, 9, 50] | [118, 8, 66] | 14.5945 | 54.3323 | 0.691015 | 0.771967
# 013 | [38, 114, 47] | [25, 104, 48] | [42, 118, 47] | 5.6569 | 11.3137 | 0.753473 | 0.784070
# 014 | [115, 164, 21] | [108, 133, 23] | [114, 163, 20] | 1.7321 | 5.7446 | 0.722953 | 0.717143
# 015 | [55, 147, 70] | [44, 120, 70] | [44, 142, 71] | 12.1244 | 24.6779 | 0.647500 | 0.749302
# 016 | [152, 107, 0] | [145, 70, 0] | [150, 106, 0] | 2.2361 | 4.4721 | 0.834161 | 0.864696
# 017 | [111, 62, 87] | [96, 42, 87] | [112, 60, 86] | 2.4495 | 6.7082 | 0.699139 | 0.741286
# 018 | [137, 92, 25] | [130, 56, 24] | [140, 88, 22] | 5.8310 | 18.0278 | 0.640450 | 0.672870
# 019 | [47, 74, 83] | [36, 50, 82] | [46, 74, 82] | 1.4142 | 5.3852 | 0.741250 | 0.739837

# Cycle error summary
# voxel: count=20 mean=7.7818 median=4.6312 std=7.7100 min=1.0000 max=31.6228 p95=17.7032
# mm:    count=20 mean=23.0283 median=11.4878 std=32.4861 min=2.0000 max=151.3275 p95=59.1821