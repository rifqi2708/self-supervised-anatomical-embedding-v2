# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("..")
sys.path.append(".")

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from cycle_error_helper import (
    prepare_axial_slice,
    sample_random_mask_points,
    validate_origin_mask,
    validate_sampled_points_inside_mask,
    visualize_cycle_result,
)
from interfaces import get_embedding, init
from utils import read_image


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to project root

CONFIG_FILE = "configs/sam/sam_NIHLN.py"
CHECKPOINT_FILE = "checkpoints/SAM.pth"

IM1_FILE = "data/quadra_dataset_cropped/images/quadra_hc_001/QUADRA_HC_001_Test_CT-AC.nii.gz"
IM2_FILE = "data/quadra_dataset_cropped/images/quadra_hc_001/QUADRA_HC_001_Retest_CT-AC.nii.gz"
MASK1_FILE = "data/quadra_dataset_cropped/masks/quadra_hc_001/QUADRA_HC_001_Test_CT-AC/kidney.nii.gz"

SEED = 0
IS_MRI = False
USE_SIM_COARSE = True
OUTPUT_DIR = "data/quadra_output/one_cycle_visual_quadra_hc_001"


def load_context(im_file, model, mask_file=None, is_mri=False):
    img, normed_im, norm_ratio = read_image(
        im_file,
        mask_path=mask_file,
        norm_spacing=(2.5, 2.5, 2.5),
        is_MRI=is_mri,
    )
    embedding = get_embedding(normed_im, model)
    image_shape = img["shape"]
    target_imshape = (image_shape[3], image_shape[1], image_shape[2])  # z, y, x
    return {
        "im_file": im_file,
        "img": img,
        "norm_ratio": np.array(norm_ratio, dtype=float),
        "embedding": embedding,
        "target_imshape": target_imshape,
    }


def _embedding_norm_map(embedding_tensor, target_imshape):
    emb_norm = torch.linalg.vector_norm(embedding_tensor, dim=1, keepdim=True)
    emb_norm = F.interpolate(emb_norm, target_imshape, mode="trilinear", align_corners=False)
    return emb_norm[0, 0].detach().cpu().numpy().astype(np.float32)  # z, y, x


def _compute_sim_fine_coarse(query_ctx, key_ctx, pt_query_xyz):
    query_point_normed = np.asarray(pt_query_xyz, dtype=float) * query_ctx["norm_ratio"]

    fine_query = query_ctx["embedding"][0]
    coarse_query = query_ctx["embedding"][1]
    fine_key = key_ctx["embedding"][0]
    coarse_key = key_ctx["embedding"][1]

    coarse_query = F.interpolate(coarse_query, fine_query.shape[2:], mode="trilinear", align_corners=False)
    coarse_query = F.normalize(coarse_query, dim=1)
    coarse_key = F.interpolate(coarse_key, fine_key.shape[2:], mode="trilinear", align_corners=False)
    coarse_key = F.normalize(coarse_key, dim=1)

    query_point_fine_re = np.floor(query_point_normed / 2.0).astype(int)
    max_xyz = np.array(
        [fine_query.shape[4] - 1, fine_query.shape[3] - 1, fine_query.shape[2] - 1],
        dtype=int,
    )
    query_point_fine_re = np.clip(query_point_fine_re, 0, max_xyz)

    query_fine = fine_query[
        0,
        :,
        int(query_point_fine_re[2]),
        int(query_point_fine_re[1]),
        int(query_point_fine_re[0]),
    ].view(-1, 128)
    key_fine = fine_key[0, :, :, :, :].reshape(128, -1)

    query_coarse = coarse_query[
        0,
        :,
        int(query_point_fine_re[2]),
        int(query_point_fine_re[1]),
        int(query_point_fine_re[0]),
    ].view(-1, 128)
    key_coarse = coarse_key[0, :, :, :, :].reshape(128, -1)

    sim_fine = torch.einsum("nc,ck->nk", query_fine, key_fine).reshape(fine_key.shape[2:])
    sim_coarse = torch.einsum("nc,ck->nk", query_coarse, key_coarse).reshape(coarse_key.shape[2:])

    sim_fine_up = F.interpolate(
        sim_fine.view(1, 1, *sim_fine.shape),
        key_ctx["target_imshape"],
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    sim_coarse_up = F.interpolate(
        sim_coarse.view(1, 1, *sim_coarse.shape),
        key_ctx["target_imshape"],
        mode="trilinear",
        align_corners=False,
    )[0, 0]

    if USE_SIM_COARSE:
        sim_for_match = (sim_fine_up + sim_coarse_up) / 2.0
    else:
        sim_for_match = sim_fine_up

    ind = torch.where(sim_for_match == sim_for_match.max())
    z = int(ind[0][0].detach().cpu().numpy())
    y = int(ind[1][0].detach().cpu().numpy())
    x = int(ind[2][0].detach().cpu().numpy())
    pt_match_xyz = np.array([x, y, z], dtype=int)
    score = float(sim_for_match.max().detach().cpu().numpy())

    return (
        pt_match_xyz,
        score,
        sim_fine_up.detach().cpu().numpy().astype(np.float32),
        sim_coarse_up.detach().cpu().numpy().astype(np.float32),
    )


def match_point(pt_query_xyz, query_ctx, key_ctx):
    pt_match_xyz, score, _, _ = _compute_sim_fine_coarse(query_ctx, key_ctx, pt_query_xyz)
    return pt_match_xyz, score


def compute_cycle_for_point(pt1_xyz, ctx_ab, ctx_ba):
    pt2_xyz, score_12 = match_point(pt1_xyz, ctx_ab, ctx_ba)
    pt1_back_xyz, score_21 = match_point(pt2_xyz, ctx_ba, ctx_ab)

    delta = pt1_back_xyz.astype(float) - np.asarray(pt1_xyz, dtype=float)
    voxel_error = float(np.linalg.norm(delta))
    spacing_yxz = np.asarray(ctx_ab["img"]["spacing"], dtype=float)
    spacing_xyz = np.array([spacing_yxz[1], spacing_yxz[0], spacing_yxz[2]], dtype=float)
    mm_error = float(np.linalg.norm(delta * spacing_xyz))

    return {
        "pt1": np.asarray(pt1_xyz, dtype=int),
        "pt2": pt2_xyz.astype(int),
        "pt1_back": pt1_back_xyz.astype(int),
        "score_12": score_12,
        "score_21": score_21,
        "voxel_error": voxel_error,
        "mm_error": mm_error,
    }


def save_embedding_maps_figure(ctx1, ctx2, result, out_path, is_mri=False):
    q_local = _embedding_norm_map(ctx1["embedding"][0], ctx1["target_imshape"])
    q_global = _embedding_norm_map(ctx1["embedding"][1], ctx1["target_imshape"])
    k_local = _embedding_norm_map(ctx2["embedding"][0], ctx2["target_imshape"])
    k_global = _embedding_norm_map(ctx2["embedding"][1], ctx2["target_imshape"])

    qz = int(result["pt1"][2])
    kz = int(result["pt2"][2])

    q_local_slice = q_local[qz, :, :]
    q_global_slice = q_global[qz, :, :]
    k_local_slice = k_local[kz, :, :]
    k_global_slice = k_global[kz, :, :]

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].set_title(f"Query Local/Fine (z={qz})")
    ax[0, 0].imshow(q_local_slice, cmap="viridis")
    ax[0, 0].plot(result["pt1"][0], result["pt1"][1], "+", color="white", markersize=10, markeredgewidth=2)

    ax[0, 1].set_title(f"Query Global/Coarse (z={qz})")
    ax[0, 1].imshow(q_global_slice, cmap="viridis")
    ax[0, 1].plot(result["pt1"][0], result["pt1"][1], "+", color="white", markersize=10, markeredgewidth=2)

    ax[1, 0].set_title(f"Target Local/Fine (z={kz})")
    ax[1, 0].imshow(k_local_slice, cmap="viridis")
    ax[1, 0].plot(result["pt2"][0], result["pt2"][1], "+", color="white", markersize=10, markeredgewidth=2)

    ax[1, 1].set_title(f"Target Global/Coarse (z={kz})")
    ax[1, 1].imshow(k_global_slice, cmap="viridis")
    ax[1, 1].plot(result["pt2"][0], result["pt2"][1], "+", color="white", markersize=10, markeredgewidth=2)

    for axis in ax.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_similarity_maps_figure(ctx1, ctx2, result, out_path, is_mri=False):
    _, _, sim_fine, sim_coarse = _compute_sim_fine_coarse(ctx1, ctx2, result["pt1"])
    kz = int(result["pt2"][2])
    target_slice = prepare_axial_slice(ctx2["img"]["img"], kz, is_mri=is_mri)
    sim_fine_slice = sim_fine[kz, :, :]
    sim_coarse_slice = sim_coarse[kz, :, :]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].set_title(f"Fine Similarity (z={kz})")
    ax[0].imshow(target_slice, cmap="gray")
    ax[0].imshow(sim_fine_slice, cmap="jet", alpha=0.45)
    ax[0].plot(result["pt2"][0], result["pt2"][1], "+", color="white", markersize=10, markeredgewidth=2)

    ax[1].set_title(f"Coarse Similarity (z={kz})")
    ax[1].imshow(target_slice, cmap="gray")
    ax[1].imshow(sim_coarse_slice, cmap="jet", alpha=0.45)
    ax[1].plot(result["pt2"][0], result["pt2"][1], "+", color="white", markersize=10, markeredgewidth=2)

    for axis in ax.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    for path in (IM1_FILE, IM2_FILE, MASK1_FILE, CONFIG_FILE, CHECKPOINT_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    model = init(CONFIG_FILE, CHECKPOINT_FILE)

    ctx1 = load_context(IM1_FILE, model, mask_file=MASK1_FILE, is_mri=IS_MRI)
    ctx2 = load_context(IM2_FILE, model, mask_file=None, is_mri=IS_MRI)

    mask1 = validate_origin_mask(
        origin_mask=ctx1["img"].get("origin_mask"),
        image_array=ctx1["img"]["img"],
        mask_name="MASK1_FILE",
    )
    points = sample_random_mask_points(mask1, num_points=1, seed=SEED)
    validate_sampled_points_inside_mask(points, mask1, "MASK1_FILE")
    pt1 = points[0].astype(int)

    result = compute_cycle_for_point(pt1, ctx1, ctx2)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cycle_png = os.path.join(OUTPUT_DIR, "cycle_points.png")
    emb_png = os.path.join(OUTPUT_DIR, "embedding_maps.png")
    sim_png = os.path.join(OUTPUT_DIR, "similarity_maps.png")

    visualize_cycle_result(
        query_img=ctx1["img"]["img"],
        target_img=ctx2["img"]["img"],
        result=result,
        out_path=cycle_png,
        show=False,
        is_mri=IS_MRI,
        viz_layout=(2, 2),
    )
    save_embedding_maps_figure(ctx1, ctx2, result, emb_png, is_mri=IS_MRI)
    save_similarity_maps_figure(ctx1, ctx2, result, sim_png, is_mri=IS_MRI)

    print(f"pt1 (sampled): {result['pt1'].tolist()}")
    print(f"pt2 (matched): {result['pt2'].tolist()}")
    print(f"pt1_back (cycle): {result['pt1_back'].tolist()}")
    print(f"score_12: {result['score_12']:.6f}")
    print(f"score_21: {result['score_21']:.6f}")
    print(f"voxel_error: {result['voxel_error']:.4f}")
    print(f"mm_error: {result['mm_error']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
