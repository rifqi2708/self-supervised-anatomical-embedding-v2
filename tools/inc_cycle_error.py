# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# This script includes embedding calculation inside the cycle pipeline.
# Companion script: tools/exc_cycle_error.py excludes embedding calculation and uses cached embeddings.
import gc
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
    print("Using GPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Using CPU")

from interfaces import get_embedding, get_sim_embed_loc, init
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
    from tools.rd_cycle_error_helper import (
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


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to project root

DATASET_ROOT = "data/quadra_dataset_cropped"
IMAGES_ROOT = os.path.join(DATASET_ROOT, "images")
MASKS_ROOT = os.path.join(DATASET_ROOT, "masks")
OUTPUT_DIR = "data/quadra_output/inc_cycle_error"
CONFIG_FILE = "configs/sam/sam_NIHLN.py"
CHECKPOINT_FILE = "checkpoints/SAM.pth"

POINT_MODE = "random"
FIXED_POINT = None
NUM_POINTS_PER_MASK = 100
SEED = 0
IS_MRI = False
USE_SIM_COARSE = True
VISUALIZE = False
VIZ_SHOW = False
VIZ_SAVE = True
EXPORT_CSV = True
VIZ_LAYOUT = (2, 2)


def is_nifti_file(name):
    return isinstance(name, str) and (name.endswith(".nii.gz") or name.endswith(".nii"))


def strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def list_subject_ids(images_root):
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Images root not found: {images_root}")
    subjects = []
    for name in sorted(os.listdir(images_root)):
        path = os.path.join(images_root, name)
        if os.path.isdir(path):
            subjects.append(name)
    if not subjects:
        raise RuntimeError(f"No subject directories found under: {images_root}")
    return subjects


def resolve_subject_pair(subject_id, images_root, masks_root):
    subject_image_dir = os.path.join(images_root, subject_id)
    image_files = []
    for name in sorted(os.listdir(subject_image_dir)):
        path = os.path.join(subject_image_dir, name)
        if os.path.isfile(path) and is_nifti_file(name):
            image_files.append(name)

    test_files = [name for name in image_files if "_Test_" in name]
    retest_files = [name for name in image_files if "_Retest_" in name]
    if len(test_files) != 1 or len(retest_files) != 1:
        raise RuntimeError(
            f"Expected one Test and one Retest image in '{subject_image_dir}', "
            f"got Test={len(test_files)} Retest={len(retest_files)}."
        )

    test_name = test_files[0]
    retest_name = retest_files[0]
    test_stem = strip_nii_suffix(test_name)
    mask1_dir = os.path.join(masks_root, subject_id, test_stem)
    if not os.path.isdir(mask1_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask1_dir}")

    return {
        "subject_id": subject_id,
        "im1_file": os.path.join(subject_image_dir, test_name),
        "im2_file": os.path.join(subject_image_dir, retest_name),
        "mask1_dir": mask1_dir,
    }


def list_mask_files(mask_dir):
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    files = {}
    for name in sorted(os.listdir(mask_dir)):
        path = os.path.join(mask_dir, name)
        if not os.path.isfile(path):
            continue
        if not is_nifti_file(name):
            continue
        files[name] = path
    if not files:
        raise RuntimeError(f"No .nii/.nii.gz mask files found in directory: {mask_dir}")
    return files


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


def load_image_context(im_file, model, is_mri=False):
    img, normed_im, norm_ratio = read_image(im_file, norm_spacing=(2.5, 2.5, 2.5), mask_path=None, is_MRI=is_mri)
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


def run_cycle_pair(
    subject_id,
    im1_file,
    im2_file,
    mask1_dir,
    model,
    point_mode=POINT_MODE,
    fixed_point=FIXED_POINT,
    num_points_per_mask=NUM_POINTS_PER_MASK,
    seed=SEED,
    is_mri=IS_MRI,
    use_sim_coarse=USE_SIM_COARSE,
    visualize=VISUALIZE,
    viz_show=VIZ_SHOW,
    viz_save=VIZ_SAVE,
    viz_dir=OUTPUT_DIR,
    viz_layout=VIZ_LAYOUT,
):
    if point_mode not in ("random", "fixed"):
        raise ValueError("point_mode must be either 'random' or 'fixed'")
    if point_mode == "random" and num_points_per_mask < 1:
        raise ValueError("num_points_per_mask must be >= 1 when point_mode='random'")
    if tuple(viz_layout) != (2, 2):
        raise ValueError(f"Only 2x2 layout is supported, got {viz_layout}")

    for im_path in (im1_file, im2_file):
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Image file not found: {im_path}")

    if point_mode == "random":
        mask_map_1 = list_mask_files(mask1_dir)
        mask_items = sorted(mask_map_1.items())
    else:
        if fixed_point is None:
            raise ValueError("fixed_point must be provided when point_mode='fixed'")
        mask_items = [("fixed_point", None)]

    if visualize and viz_save:
        os.makedirs(viz_dir, exist_ok=True)

    time1 = time.time()
    ctx1 = load_image_context(im1_file, model, is_mri=is_mri)
    ctx2 = load_image_context(im2_file, model, is_mri=is_mri)
    time2 = time.time()
    print(f"[{subject_id}] image+embedding loading time: {time2 - time1:.3f}s")

    all_results = []
    per_mask_results = {}
    time3 = time.time()

    for mask_idx, (mask_file_name, mask1_path) in enumerate(mask_items):
        mask_label = f"{subject_id}/{mask_file_name}"
        print(f"[{subject_id}] Processing mask: {mask_file_name}")

        if point_mode == "fixed":
            points = np.asarray([validate_fixed_point(fixed_point, ctx1["img"])])
        else:
            mask1_array = load_mask_array(
                im1_file,
                mask1_path,
                is_mri=is_mri,
                ref_image=ctx1["img"]["img"],
                mask_name=f"{subject_id}:{mask_file_name}",
            )
            mask_seed = int(seed) + int(mask_idx)
            points = sample_random_mask_points(mask1_array, num_points_per_mask, mask_seed)
            validate_sampled_points_inside_mask(points, mask1_array, f"{subject_id}:{mask_file_name}")

        mask_results = []
        for point_idx, point in enumerate(points):
            result = compute_cycle_for_point(point, ctx1, ctx2, use_sim_coarse=use_sim_coarse)
            result["mask_name"] = mask_label
            result["subject_id"] = subject_id
            mask_results.append(result)
            all_results.append(result)

            if visualize:
                save_path = None
                if viz_save:
                    pt1 = result["pt1"]
                    pt2 = result["pt2"]
                    pt1_back = result["pt1_back"]
                    safe_mask = strip_nii_suffix(mask_file_name)
                    save_name = (
                        f"{subject_id}_{safe_mask}_cycle_{point_idx:03d}_"
                        f"q_{pt1[0]}_{pt1[1]}_{pt1[2]}_"
                        f"m_{pt2[0]}_{pt2[1]}_{pt2[2]}_"
                        f"c_{pt1_back[0]}_{pt1_back[1]}_{pt1_back[2]}.png"
                    )
                    save_path = os.path.join(viz_dir, save_name)

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
            per_mask_results[mask_label] = mask_results

    time4 = time.time()
    print(f"[{subject_id}] cycle matching time: {time4 - time3:.3f}s")

    del ctx1
    del ctx2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_results:
        raise RuntimeError(f"No cycle results were produced for subject '{subject_id}'.")

    return all_results, per_mask_results


def run_dataset_cycle(
    dataset_root=DATASET_ROOT,
    output_dir=OUTPUT_DIR,
    config_file=CONFIG_FILE,
    checkpoint_file=CHECKPOINT_FILE,
    point_mode=POINT_MODE,
    fixed_point=FIXED_POINT,
    num_points_per_mask=NUM_POINTS_PER_MASK,
    seed=SEED,
    is_mri=IS_MRI,
    use_sim_coarse=USE_SIM_COARSE,
    visualize=VISUALIZE,
    viz_show=VIZ_SHOW,
    viz_save=VIZ_SAVE,
    export_csv=EXPORT_CSV,
    viz_layout=VIZ_LAYOUT,
):
    images_root = os.path.join(dataset_root, "images")
    masks_root = os.path.join(dataset_root, "masks")

    if export_csv or (visualize and viz_save):
        os.makedirs(output_dir, exist_ok=True)

    subject_ids = list_subject_ids(images_root)
    print(f"Found {len(subject_ids)} subjects under '{images_root}'.")

    time1 = time.time()
    model = init(config_file, checkpoint_file)
    time2 = time.time()
    print(f"model loading time: {time2 - time1:.3f}s")

    all_results = []
    per_mask_aggregate = {}
    failed_subjects = []
    processed_subjects = 0

    for subject_idx, subject_id in enumerate(subject_ids, start=1):
        print(f"\n[{subject_idx:03d}/{len(subject_ids):03d}] Subject: {subject_id}")
        try:
            pair = resolve_subject_pair(subject_id, images_root, masks_root)
            pair_results, pair_per_mask = run_cycle_pair(
                subject_id=pair["subject_id"],
                im1_file=pair["im1_file"],
                im2_file=pair["im2_file"],
                mask1_dir=pair["mask1_dir"],
                model=model,
                point_mode=point_mode,
                fixed_point=fixed_point,
                num_points_per_mask=num_points_per_mask,
                seed=seed,
                is_mri=is_mri,
                use_sim_coarse=use_sim_coarse,
                visualize=visualize,
                viz_show=viz_show,
                viz_save=viz_save,
                viz_dir=output_dir,
                viz_layout=viz_layout,
            )

            all_results.extend(pair_results)
            for mask_name, mask_results in pair_per_mask.items():
                per_mask_aggregate.setdefault(mask_name, []).extend(mask_results)
            processed_subjects += 1
        except Exception as exc:
            failed_subjects.append((subject_id, str(exc)))
            print(f"WARNING: skipping subject '{subject_id}' due to error: {exc}")
            continue

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_results:
        raise RuntimeError("No cycle results were produced for the dataset.")

    print("\nDataset summary across all masks and subjects")
    global_voxel_stats, global_mm_stats = print_summary(all_results)

    per_mask_rows = []
    for mask_name in sorted(per_mask_aggregate.keys()):
        voxel_stats, mm_stats = print_summary(per_mask_aggregate[mask_name])
        per_mask_rows.append({"mask_name": mask_name, "voxel_stats": voxel_stats, "mm_stats": mm_stats})

    if export_csv:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv_path = os.path.join(output_dir, f"cycle_summary_{run_stamp}.csv")
        points_csv_path = os.path.join(output_dir, f"cycle_points_{run_stamp}.csv")
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

    print("\nRun complete")
    print(f"Processed subjects: {processed_subjects}")
    print(f"Failed subjects: {len(failed_subjects)}")
    if failed_subjects:
        for subject_id, reason in failed_subjects:
            print(f"  - {subject_id}: {reason}")


if __name__ == "__main__":
    try:
        run_dataset_cycle(
            dataset_root=DATASET_ROOT,
            output_dir=OUTPUT_DIR,
            config_file=CONFIG_FILE,
            checkpoint_file=CHECKPOINT_FILE,
            point_mode=POINT_MODE,
            fixed_point=FIXED_POINT,
            num_points_per_mask=NUM_POINTS_PER_MASK,
            seed=SEED,
            is_mri=IS_MRI,
            use_sim_coarse=USE_SIM_COARSE,
            visualize=VISUALIZE,
            viz_show=VIZ_SHOW,
            viz_save=VIZ_SAVE,
            export_csv=EXPORT_CSV,
            viz_layout=VIZ_LAYOUT,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
