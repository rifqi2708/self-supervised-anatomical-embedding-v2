# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch

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
from cycle_error_helper import (
    sample_random_mask_points,
    validate_fixed_point,
    validate_mask_file,
    validate_origin_mask,
    validate_sampled_points_inside_mask,
    write_points_csv,
    write_summary_csv,
    print_summary,
    visualize_cycle_result,
)


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project
CONFIG_FILE = "configs/sam/sam_NIHLN.py"
CHECKPOINT_FILE = "checkpoints/SAM.pth"
DEFAULT_IM1_FILE = "data/raw_data/NIH_lymph_node/ABD_LYMPH_001.nii.gz"
DEFAULT_IM2_FILE = "data/raw_data/NIH_lymph_node/ABD_LYMPH_002.nii.gz"
DEFAULT_MASK1_FILE = "data/raw_data/NIH_lymph_node/masks/mask_ABD_LYMPH_001.nii.gz"
DEFAULT_MASK2_FILE = "data/raw_data/NIH_lymph_node/masks/mask_ABD_LYMPH_002.nii.gz"

#loading data and embeddings for one image, to be used as context for cycle error computation
def load_context(im_file, model, is_mri=False, mask_file=None):
    img, normed_im, norm_ratio = read_image(im_file, mask_path=mask_file, is_MRI=is_mri)
    embedding = get_embedding(normed_im, model)
    image_shape = img["shape"]
    if len(image_shape) != 4:
        raise ValueError(f"Unexpected image shape from read_image: {image_shape}")
    target_imshape = (image_shape[3], image_shape[1], image_shape[2])
    return {
        "im_file": im_file,
        "img": img,
        "norm_ratio": np.array(norm_ratio, dtype=float),
        "embedding": embedding,
        "target_imshape": target_imshape,
    }

#find corresponding point a pixel in query_ctx at the key ctx, and return the matched point and similarity score
def match_point(pt_query, query_ctx, key_ctx, use_sim_coarse=True):
    pt_query = np.asarray(pt_query, dtype=float)
    pt_query_normed = pt_query * query_ctx["norm_ratio"]
    pt_match, score = get_sim_embed_loc(
        query_ctx["embedding"],
        key_ctx["embedding"],
        pt_query_normed,
        key_ctx["target_imshape"],
        norm_info=key_ctx["norm_ratio"],
        write_sim=False,
        use_sim_coarse=use_sim_coarse,
    )
    return np.asarray(pt_match, dtype=int), float(score)

#find the corresponding point in ctx_ba for a point in ctx_ab, then find the corresponding point back in ctx_ab, and compute the cycle error in both voxel and mm space
def compute_cycle_for_point(pt1, ctx_ab, ctx_ba, use_sim_coarse=True):
    pt1 = np.asarray(pt1, dtype=int)
    pt2, score_12 = match_point(pt1, ctx_ab, ctx_ba, use_sim_coarse=use_sim_coarse)
    pt1_back, score_21 = match_point(pt2, ctx_ba, ctx_ab, use_sim_coarse=use_sim_coarse)

    #calculating error in voxel and mm
    delta = pt1_back.astype(float) - pt1.astype(float)
    voxel_error = float(np.linalg.norm(delta))
    spacing_yxz = np.asarray(ctx_ab["img"]["spacing"], dtype=float)
    spacing_xyz = np.array([spacing_yxz[1], spacing_yxz[0], spacing_yxz[2]], dtype=float)
    mm_error = float(np.linalg.norm(delta * spacing_xyz))

    return {
        "pt1": pt1,
        "pt2": pt2,
        "pt1_back": pt1_back,
        "score_12": score_12,
        "score_21": score_21,
        "voxel_error": voxel_error,
        "mm_error": mm_error,
    }

# Main function to run the cycle error computation and visualization for a set of points between two images
def run_cycle(
    im1_file=DEFAULT_IM1_FILE,
    im2_file=DEFAULT_IM2_FILE,
    mask1_file=None,
    mask2_file=None,
    point_mode="random",
    fixed_point=None,
    num_points=20,
    seed=0,
    is_mri=False,
    use_sim_coarse=True,
    visualize=True,
    viz_show=False,
    viz_save=True,
    viz_dir="data/quadra_output/cycle_vis",
    export_csv=True,
    viz_layout=(2, 2),
):
    if point_mode not in ("random", "fixed"):
        raise ValueError("point_mode must be either 'random' or 'fixed'")

    if point_mode == "random" and num_points < 1:
        raise ValueError("num_points must be >= 1 when point_mode='random'")

    for im_path in (im1_file, im2_file):
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Image file not found: {im_path}")

    validate_mask_file(mask1_file, "mask1_file")
    validate_mask_file(mask2_file, "mask2_file")
    if point_mode == "random" and mask1_file is None:
        raise ValueError("mask1_file must be provided when point_mode='random'")

    time1 = time.time()
    model = init(CONFIG_FILE, CHECKPOINT_FILE)
    time2 = time.time()
    print(f"model loading time: {time2 - time1:.3f}s")

    ctx1 = load_context(im1_file, model, is_mri=is_mri, mask_file=mask1_file)
    ctx2 = load_context(im2_file, model, is_mri=is_mri, mask_file=mask2_file)
    time3 = time.time()
    print(f"image+embedding loading time: {time3 - time2:.3f}s")

    if mask2_file is not None:
        validate_origin_mask(
            origin_mask=ctx2["img"].get("origin_mask"),
            image_array=ctx2["img"]["img"],
            mask_name="mask2_file",
        )

    if point_mode == "fixed":
        if fixed_point is None:
            raise ValueError("fixed_point must be provided when point_mode='fixed'")
        points = np.asarray([validate_fixed_point(fixed_point, ctx1["img"])])
    else:
        mask1_array = validate_origin_mask(
            origin_mask=ctx1["img"].get("origin_mask"),
            image_array=ctx1["img"]["img"],
            mask_name="mask1_file",
        )
        points = sample_random_mask_points(mask1_array, num_points, seed)
        validate_sampled_points_inside_mask(points, mask1_array, "mask1_file")

    if tuple(viz_layout) != (2, 2):
        raise ValueError(f"Only 2x2 layout is supported, got {viz_layout}")

    resolved_viz_dir = viz_dir if os.path.isabs(viz_dir) else os.path.abspath(viz_dir)

    csv_output_dir = None
    if export_csv:
        csv_output_dir = resolved_viz_dir
        os.makedirs(csv_output_dir, exist_ok=True)

    viz_output_dir = None
    if visualize and viz_save:
        viz_output_dir = resolved_viz_dir
        os.makedirs(viz_output_dir, exist_ok=True)

    results = []
    time4 = time.time()
    for idx, point in enumerate(points):
        result = compute_cycle_for_point(point, ctx1, ctx2, use_sim_coarse=use_sim_coarse)
        results.append(result)
        if visualize:
            save_path = None
            if viz_save:
                pt1 = result["pt1"]
                pt2 = result["pt2"]
                pt1_back = result["pt1_back"]
                save_name = (
                    f"cycle_{idx:03d}_"
                    f"q_{pt1[0]}_{pt1[1]}_{pt1[2]}_"
                    f"m_{pt2[0]}_{pt2[1]}_{pt2[2]}_"
                    f"c_{pt1_back[0]}_{pt1_back[1]}_{pt1_back[2]}.png"
                )
                save_path = os.path.join(viz_output_dir, save_name)

            visualize_cycle_result(
                query_img=ctx1["img"]["img"],
                target_img=ctx2["img"]["img"],
                result=result,
                out_path=save_path,
                show=viz_show,
                is_mri=is_mri,
                viz_layout=viz_layout,
            )
    time5 = time.time()
    print(f"cycle matching time: {time5 - time4:.3f}s")

    # print_result_table(results)
    voxel_stats, mm_stats = print_summary(results)

    if export_csv:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv_path = os.path.join(csv_output_dir, f"cycle_summary_{run_stamp}.csv")
        points_csv_path = os.path.join(csv_output_dir, f"cycle_points_{run_stamp}.csv")
        write_summary_csv(voxel_stats, mm_stats, summary_csv_path)
        write_points_csv(results, points_csv_path)
        print(f"summary csv saved: {summary_csv_path}")
        print(f"points csv saved: {points_csv_path}")


if __name__ == "__main__":
    try:
        # Edit this single call to change behavior:
        # 1) Random mode: point_mode="random", num_points=20, seed=0
        # 2) Fixed mode: point_mode="fixed", fixed_point=(93, 139, 44)
        run_cycle(
            im1_file=DEFAULT_IM1_FILE,
            im2_file=DEFAULT_IM2_FILE,
            mask1_file=DEFAULT_MASK1_FILE,
            mask2_file=DEFAULT_MASK2_FILE,
            point_mode="random",
            fixed_point=None,
            num_points=150,
            seed=0,
            is_mri=False,
            use_sim_coarse=True,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
