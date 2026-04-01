# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import torch

sys.path.append("..")
sys.path.append(".")

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Using GPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Using CPU")

from interfaces import init, get_embedding, get_sim_embed_loc
from utils import read_image

try:
    from rd_cycle_error_helper import (
        print_summary,
        sample_random_mask_points,
        validate_fixed_point,
        validate_mask_file,
        validate_origin_mask,
        validate_sampled_points_inside_mask,
        visualize_cycle_result,
        write_points_csv_with_mask,
        write_summary_with_mask_labels_csv,
    )
except ImportError:
    from rd_cycle_error_helper import (
        print_summary,
        sample_random_mask_points,
        validate_fixed_point,
        validate_mask_file,
        validate_origin_mask,
        validate_sampled_points_inside_mask,
        visualize_cycle_result,
        write_points_csv_with_mask,
        write_summary_with_mask_labels_csv,
    )


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project
CONFIG_FILE = "configs/sam/sam_NIHLN.py"
CHECKPOINT_FILE = "checkpoints/SAM.pth"
DEFAULT_IM1_FILE = "data/quadra_dataset/images/quadra_hc_001/test_QUADRA_HC_001_Test_CT-AC.nii.gz"
DEFAULT_IM2_FILE = "data/quadra_dataset/images/quadra_hc_001/retest_QUADRA_HC_001_Retest_CT-AC.nii.gz"
DEFAULT_MASK1_FILE = "data/quadra_dataset/masks/quadra_hc_001/test_QUADRA_HC_001_Test_CT-AC"


# load image and embedding once per image, reused across all masks
def load_image_context(im_file, model, is_mri=False):
    img, normed_im, norm_ratio = read_image(im_file, mask_path=None, is_MRI=is_mri)
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


# load only mask array by reusing read_image; no embedding generation here
def load_mask_array(im_file, mask_file, is_mri=False, ref_image=None, mask_name="mask"):
    validate_mask_file(mask_file, mask_name)
    img_with_mask, _, _ = read_image(im_file, mask_path=mask_file, is_MRI=is_mri)
    mask_array = validate_origin_mask(
        origin_mask=img_with_mask.get("origin_mask"),
        image_array=img_with_mask["img"],
        mask_name=mask_name,
    )
    if ref_image is not None and tuple(mask_array.shape) != tuple(np.asarray(ref_image).shape):
        raise ValueError(
            f"{mask_name} shape {mask_array.shape} does not match reference image shape {np.asarray(ref_image).shape}."
        )
    return mask_array


def list_mask_files(mask_dir):
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    files = {}
    for name in sorted(os.listdir(mask_dir)):
        path = os.path.join(mask_dir, name)
        if not os.path.isfile(path):
            continue
        if not (name.endswith(".nii.gz") or name.endswith(".nii")):
            continue
        files[name] = path
    if not files:
        raise RuntimeError(f"No .nii/.nii.gz mask files found in directory: {mask_dir}")
    return files


def normalized_path_variants(path):
    if not isinstance(path, str) or not path:
        return set()
    variants = {
        os.path.normpath(path),
        os.path.normpath(os.path.abspath(path)),
    }
    return variants


def load_boundary_manifest(boundary_manifest_path):
    if not os.path.exists(boundary_manifest_path):
        raise FileNotFoundError(f"boundary_manifest not found: {boundary_manifest_path}")
    with open(boundary_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("boundary_manifest must be a JSON object.")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("boundary_manifest must contain a non-empty 'cases' list.")
    return manifest


def build_z_offset_lookup(manifest):
    lookup = {}
    input_root = manifest.get("input_root")
    output_root = manifest.get("output_root")

    def register_path(path, z_min, root=None):
        for variant in normalized_path_variants(path):
            lookup[variant] = z_min
        if isinstance(path, str) and path and root and not os.path.isabs(path):
            joined = os.path.join(root, path)
            for variant in normalized_path_variants(joined):
                lookup[variant] = z_min

    for case in manifest.get("cases", []):
        z_info = case.get("z", {})
        if "z_min" not in z_info:
            raise ValueError(f"Case entry missing z.z_min: {case.get('case_id', '<unknown-case>')}")
        z_min = int(z_info["z_min"])
        register_path(case.get("source_image"), z_min, root=input_root)
        register_path(case.get("source_image_abs"), z_min)
        register_path(case.get("cropped_image"), z_min, root=output_root)
        register_path(case.get("cropped_image_abs"), z_min)
    if not lookup:
        raise ValueError("No valid image paths found in boundary_manifest.")
    return lookup


def resolve_z_offset(im_file, z_offset_lookup, boundary_manifest_path):
    for key in normalized_path_variants(im_file):
        if key in z_offset_lookup:
            return int(z_offset_lookup[key])
    raise KeyError(
        f"Could not find image '{im_file}' in boundary_manifest '{boundary_manifest_path}'. "
        "Ensure im1_file/im2_file match source/cropped image paths recorded in the manifest."
    )


# find corresponding point a pixel in query_ctx at the key ctx, and return the matched point and similarity score
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


# find the corresponding point in ctx_ba for a point in ctx_ab, then map back and compute cycle error
def compute_cycle_for_point(pt1, ctx_ab, ctx_ba, use_sim_coarse=True):
    pt1 = np.asarray(pt1, dtype=int)
    pt2, score_12 = match_point(pt1, ctx_ab, ctx_ba, use_sim_coarse=use_sim_coarse)
    pt1_back, score_21 = match_point(pt2, ctx_ba, ctx_ab, use_sim_coarse=use_sim_coarse)

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


# Main function to run cycle error computation and visualization
def run_cycle(
    im1_file=DEFAULT_IM1_FILE,
    im2_file=DEFAULT_IM2_FILE,
    mask1_file=None,
    point_mode="random",
    fixed_point=None,
    num_points_per_mask=100,
    num_points=None,
    seed=0,
    is_mri=False,
    use_sim_coarse=True,
    visualize=False,
    viz_show=True,
    viz_save=True,
    viz_dir="data/quadra_output",
    export_csv=True,
    viz_layout=(2, 2),
    boundary_manifest=None,
    report_original_coords=True,
):
    if point_mode not in ("random", "fixed"):
        raise ValueError("point_mode must be either 'random' or 'fixed'")

    if num_points is not None:
        print("WARNING: num_points is deprecated. Using it as num_points_per_mask.")
        num_points_per_mask = int(num_points)
    if point_mode == "random" and num_points_per_mask < 1:
        raise ValueError("num_points_per_mask must be >= 1 when point_mode='random'")

    for im_path in (im1_file, im2_file):
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Image file not found: {im_path}")

    if tuple(viz_layout) != (2, 2):
        raise ValueError(f"Only 2x2 layout is supported, got {viz_layout}")

    if point_mode == "random":
        if mask1_file is None:
            raise ValueError("mask1_file must be provided for random mode.")
        if not os.path.isdir(mask1_file):
            raise ValueError("This script now expects mask1_file to be a directory in random mode.")
        mask_map_1 = list_mask_files(mask1_file)
        mask_items = sorted(mask_map_1.items())
    else:
        if fixed_point is None:
            raise ValueError("fixed_point must be provided when point_mode='fixed'")
        mask_items = [("fixed_point", None)]

    resolved_viz_dir = viz_dir if os.path.isabs(viz_dir) else os.path.abspath(viz_dir)

    im1_z_offset = 0
    im2_z_offset = 0
    if boundary_manifest is not None:
        manifest = load_boundary_manifest(boundary_manifest)
        z_offset_lookup = build_z_offset_lookup(manifest)
        im1_z_offset = resolve_z_offset(im1_file, z_offset_lookup, boundary_manifest)
        im2_z_offset = resolve_z_offset(im2_file, z_offset_lookup, boundary_manifest)
        print(
            f"Loaded boundary manifest '{boundary_manifest}' with z offsets: "
            f"im1={im1_z_offset}, im2={im2_z_offset}"
        )
    elif report_original_coords:
        print("WARNING: report_original_coords=True but boundary_manifest is None; original coordinates not exported.")

    csv_output_dir = None
    if export_csv:
        csv_output_dir = resolved_viz_dir
        os.makedirs(csv_output_dir, exist_ok=True)

    viz_output_dir = None
    if visualize and viz_save:
        viz_output_dir = resolved_viz_dir
        os.makedirs(viz_output_dir, exist_ok=True)

    time1 = time.time()
    model = init(CONFIG_FILE, CHECKPOINT_FILE)
    time2 = time.time()
    print(f"model loading time: {time2 - time1:.3f}s")

    ctx1 = load_image_context(im1_file, model, is_mri=is_mri)
    ctx2 = load_image_context(im2_file, model, is_mri=is_mri)
    time3 = time.time()
    print(f"image+embedding loading time (once): {time3 - time2:.3f}s")

    all_results = []
    per_mask_results = {}
    time4 = time.time()
    for mask_idx, (mask_name, mask1_path) in enumerate(mask_items):
        print(f"\nProcessing mask: {mask_name}")
        if point_mode == "fixed":
            points = np.asarray([validate_fixed_point(fixed_point, ctx1["img"])])
        else:
            try:
                mask1_array = load_mask_array(
                    im1_file,
                    mask1_path,
                    is_mri=is_mri,
                    ref_image=ctx1["img"]["img"],
                    mask_name=f"mask1:{mask_name}",
                )
                mask_seed = int(seed) + int(mask_idx)
                points = sample_random_mask_points(mask1_array, num_points_per_mask, mask_seed)
                validate_sampled_points_inside_mask(points, mask1_array, f"mask1:{mask_name}")
            except RuntimeError as exc:
                print(f"WARNING: skipping mask '{mask_name}' due to sampling/validation error: {exc}")
                continue

        mask_results = []
        for point_idx, point in enumerate(points):
            result = compute_cycle_for_point(point, ctx1, ctx2, use_sim_coarse=use_sim_coarse)
            result["mask_name"] = mask_name
            if boundary_manifest is not None and report_original_coords:
                pt1_orig = np.asarray(result["pt1"], dtype=int).copy()
                pt2_orig = np.asarray(result["pt2"], dtype=int).copy()
                pt1_back_orig = np.asarray(result["pt1_back"], dtype=int).copy()
                pt1_orig[2] += int(im1_z_offset)
                pt2_orig[2] += int(im2_z_offset)
                pt1_back_orig[2] += int(im1_z_offset)
                result["pt1_orig"] = pt1_orig
                result["pt2_orig"] = pt2_orig
                result["pt1_back_orig"] = pt1_back_orig
                result["im1_z_offset"] = int(im1_z_offset)
                result["im2_z_offset"] = int(im2_z_offset)
            mask_results.append(result)
            all_results.append(result)
            if visualize:
                save_path = None
                if viz_save:
                    pt1 = result["pt1"]
                    pt2 = result["pt2"]
                    pt1_back = result["pt1_back"]
                    save_name = (
                        f"{mask_name}_cycle_{point_idx:03d}_"
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

        if mask_results:
            per_mask_results[mask_name] = mask_results

    time5 = time.time()
    print(f"\ncycle matching time: {time5 - time4:.3f}s")

    if not all_results:
        raise RuntimeError("No cycle results were produced. Check masks and sampling setup.")

    per_mask_rows = []
    for mask_name in sorted(per_mask_results.keys()):
        print(f"\nSummary for mask: {mask_name}")
        voxel_stats, mm_stats = print_summary(per_mask_results[mask_name])
        per_mask_rows.append({"mask_name": mask_name, "voxel_stats": voxel_stats, "mm_stats": mm_stats})

    print("\nGlobal summary across all masks")
    global_voxel_stats, global_mm_stats = print_summary(all_results)

    if export_csv:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv_path = os.path.join(csv_output_dir, f"cycle_summary_{run_stamp}.csv")
        points_csv_path = os.path.join(csv_output_dir, f"cycle_points_{run_stamp}.csv")
        write_summary_with_mask_labels_csv(
            per_mask_rows,
            summary_csv_path,
            global_voxel_stats=global_voxel_stats,
            global_mm_stats=global_mm_stats,
            all_masks_label="ALL_MASKS",
        )
        write_points_csv_with_mask(all_results, points_csv_path)
        print(f"summary csv saved: {summary_csv_path}")
        print(f"points csv saved: {points_csv_path}")


if __name__ == "__main__":
    try:
        run_cycle(
            im1_file=DEFAULT_IM1_FILE,
            im2_file=DEFAULT_IM2_FILE,
            mask1_file=DEFAULT_MASK1_FILE,
            point_mode="random",
            fixed_point=None,
            num_points_per_mask=100,
            seed=0,
            is_mri=False,
            use_sim_coarse=True,
            boundary_manifest=None,
            report_original_coords=True,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
