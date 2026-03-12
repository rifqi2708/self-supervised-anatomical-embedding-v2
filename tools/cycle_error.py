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


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def print_result_table(results):
    print("idx | pt1(x,y,z) | pt2(x,y,z) | pt1_back(x,y,z) | err_voxel | err_mm | score_12 | score_21")
    for idx, record in enumerate(results):
        print(
            f"{idx:03d} | "
            f"{record['pt1'].tolist()} | {record['pt2'].tolist()} | {record['pt1_back'].tolist()} | "
            f"{record['voxel_error']:.4f} | {record['mm_error']:.4f} | "
            f"{record['score_12']:.6f} | {record['score_21']:.6f}"
        )


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
        f"p95={voxel_stats['p95']:.4f}"
    )
    print(
        "mm:    "
        f"count={mm_stats['count']} mean={mm_stats['mean']:.4f} median={mm_stats['median']:.4f} "
        f"std={mm_stats['std']:.4f} min={mm_stats['min']:.4f} max={mm_stats['max']:.4f} "
        f"p95={mm_stats['p95']:.4f}"
    )


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
