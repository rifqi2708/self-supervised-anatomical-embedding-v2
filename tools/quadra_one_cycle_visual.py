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
    sample_random_mask_points,
    validate_origin_mask,
    validate_sampled_points_inside_mask,
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
EMBED_SINGLE_CHANNEL_INDICES = [0, 16, 32, 64]
EMBED_PCA_MAX_SAMPLES = 50000
EMBED_STYLES = ["l2norm", "variance", "pca1", "cosine", "single_channel"]


def _normalize_volume_for_display(img3d, is_mri=False):
    img3d = np.asarray(img3d, dtype=np.float32)
    if is_mri:
        low = float(np.min(img3d))
        high = float(np.max(img3d))
    else:
        low, high = -100.0, 200.0
    if high <= low:
        return np.zeros_like(img3d, dtype=np.float32)
    img3d = np.clip(img3d, low, high)
    return ((img3d - low) / (high - low)).astype(np.float32)


def _slice_plane_with_point(volume_yxz, point_xyz, plane):
    x = int(point_xyz[0])
    y = int(point_xyz[1])
    z = int(point_xyz[2])

    sy, sx, sz = volume_yxz.shape
    x = int(np.clip(x, 0, sx - 1))
    y = int(np.clip(y, 0, sy - 1))
    z = int(np.clip(z, 0, sz - 1))

    if plane == "axial":
        sl = volume_yxz[:, :, z]
        px, py = x, y
    elif plane == "coronal":
        sl = volume_yxz[y, :, :].T  # rows=z, cols=x
        sl = sl[::-1, :]  # flip vertically so head is up
        px, py = x, (sz - 1 - z)
    elif plane == "sagittal":
        sl = volume_yxz[:, x, :].T  # rows=z, cols=y
        sl = sl[::-1, :]  # flip vertically so head is up
        px, py = y, (sz - 1 - z)
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return sl, (px, py)


def _draw_marker(ax, xy, color):
    ax.plot(
        float(xy[0]),
        float(xy[1]),
        "+",
        markerfacecolor="none",
        markeredgecolor=color,
        markersize=6,
        markeredgewidth=1.3,
    )


def _normalize_map_for_display(map2d):
    map2d = np.asarray(map2d, dtype=np.float32)
    if map2d.size == 0:
        return map2d
    low = float(np.percentile(map2d, 1))
    high = float(np.percentile(map2d, 99))
    if high <= low:
        low = float(np.min(map2d))
        high = float(np.max(map2d))
    if high <= low:
        return np.zeros_like(map2d, dtype=np.float32)
    map2d = np.clip(map2d, low, high)
    return (map2d - low) / (high - low)


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


def _point_to_fine_index(point_xyz, norm_ratio, fine_shape_zyx):
    point_xyz = np.asarray(point_xyz, dtype=float)
    idx_xyz = np.floor((point_xyz * np.asarray(norm_ratio, dtype=float)) / 2.0).astype(int)
    max_xyz = np.array(
        [fine_shape_zyx[2] - 1, fine_shape_zyx[1] - 1, fine_shape_zyx[0] - 1],
        dtype=int,
    )
    idx_xyz = np.clip(idx_xyz, 0, max_xyz)
    return idx_xyz


def _pca1_feature_map(feature_tensor, max_samples=50000):
    # feature_tensor: [1, C, Z, Y, X]
    feat = feature_tensor[0].detach().to(dtype=torch.float32)
    c, z, y, x = feat.shape
    flat = feat.reshape(c, -1)
    n = int(flat.shape[1])
    if n < 2:
        return torch.zeros((z, y, x), dtype=feat.dtype, device=feat.device)

    if n > int(max_samples):
        sample_idx = torch.linspace(0, n - 1, steps=int(max_samples), device=feat.device).long()
        sample = flat[:, sample_idx].transpose(0, 1)
    else:
        sample = flat.transpose(0, 1)

    mean = sample.mean(dim=0)
    centered = sample - mean
    denom = max(int(centered.shape[0] - 1), 1)
    cov = (centered.transpose(0, 1) @ centered) / float(denom)
    try:
        _, eigvecs = torch.linalg.eigh(cov)
        pc1 = eigvecs[:, -1]
    except RuntimeError:
        # Fallback for environments where GPU eigendecomposition support is limited.
        _, eigvecs = torch.linalg.eigh(cov.detach().cpu())
        pc1 = eigvecs[:, -1].to(feat.device)

    full_centered = feat - mean.view(c, 1, 1, 1)
    proj = torch.einsum("c,czyx->zyx", pc1, full_centered)
    return proj


def _cosine_feature_map(feature_tensor, idx_xyz):
    # feature_tensor: [1, C, Z, Y, X], idx_xyz in fine-grid xyz coordinates
    x = int(idx_xyz[0])
    y = int(idx_xyz[1])
    z = int(idx_xyz[2])
    ref = feature_tensor[0, :, z, y, x].view(1, -1)
    ref = F.normalize(ref, dim=1)[0]
    vox = F.normalize(feature_tensor[0], dim=0)
    return torch.einsum("c,czyx->zyx", ref, vox)


def _feature_style_scalar_map(feature_tensor, style, idx_xyz=None, channel_idx=0):
    # returns [Z, Y, X]
    if style == "l2norm":
        return torch.linalg.vector_norm(feature_tensor[0], dim=0)
    if style == "variance":
        return torch.var(feature_tensor[0], dim=0, unbiased=False)
    if style == "pca1":
        return _pca1_feature_map(feature_tensor, max_samples=EMBED_PCA_MAX_SAMPLES)
    if style == "cosine":
        if idx_xyz is None:
            raise ValueError("idx_xyz is required for cosine style map.")
        return _cosine_feature_map(feature_tensor, idx_xyz)
    if style == "single_channel":
        c = int(feature_tensor.shape[1])
        ch = int(channel_idx)
        if ch < 0 or ch >= c:
            raise ValueError(f"channel_idx out of range: {ch}. Valid range: [0, {c-1}]")
        return feature_tensor[0, ch]
    raise ValueError(f"Unknown embedding style: {style}")


def _upsample_scalar_zyx_to_target(scalar_zyx, target_imshape):
    return F.interpolate(
        scalar_zyx.view(1, 1, *scalar_zyx.shape),
        target_imshape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]


def _embedding_style_maps_for_context(ctx, point_xyz, style, channel_idx=None):
    fine = ctx["embedding"][0]
    coarse = F.interpolate(ctx["embedding"][1], fine.shape[2:], mode="trilinear", align_corners=False)
    idx_xyz = _point_to_fine_index(point_xyz, ctx["norm_ratio"], fine.shape[2:])

    fine_map = _feature_style_scalar_map(fine, style, idx_xyz=idx_xyz, channel_idx=channel_idx)
    coarse_map = _feature_style_scalar_map(coarse, style, idx_xyz=idx_xyz, channel_idx=channel_idx)

    fine_map_up = _upsample_scalar_zyx_to_target(fine_map, ctx["target_imshape"])
    coarse_map_up = _upsample_scalar_zyx_to_target(coarse_map, ctx["target_imshape"])

    return (
        fine_map_up.detach().cpu().numpy().astype(np.float32),
        coarse_map_up.detach().cpu().numpy().astype(np.float32),
    )


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


def save_embedding_maps_figures(ctx1, ctx2, result, out_dir, is_mri=False):
    plane_list = ["axial", "sagittal", "coronal"]
    for style in EMBED_STYLES:
        channel_list = EMBED_SINGLE_CHANNEL_INDICES if style == "single_channel" else [None]
        for channel_idx in channel_list:
            q_local, q_global = _embedding_style_maps_for_context(
                ctx1, result["pt1"], style=style, channel_idx=channel_idx
            )
            k_local, k_global = _embedding_style_maps_for_context(
                ctx2, result["pt2"], style=style, channel_idx=channel_idx
            )

            q_local_yxz = q_local.transpose(1, 2, 0)
            q_global_yxz = q_global.transpose(1, 2, 0)
            k_local_yxz = k_local.transpose(1, 2, 0)
            k_global_yxz = k_global.transpose(1, 2, 0)

            style_tag = style if channel_idx is None else f"single_channel_ch{int(channel_idx):03d}"
            for plane in plane_list:
                q_local_slice, qxy = _slice_plane_with_point(q_local_yxz, result["pt1"], plane)
                q_global_slice, _ = _slice_plane_with_point(q_global_yxz, result["pt1"], plane)
                k_local_slice, kxy = _slice_plane_with_point(k_local_yxz, result["pt2"], plane)
                k_global_slice, _ = _slice_plane_with_point(k_global_yxz, result["pt2"], plane)

                q_local_slice = _normalize_map_for_display(q_local_slice)
                q_global_slice = _normalize_map_for_display(q_global_slice)
                k_local_slice = _normalize_map_for_display(k_local_slice)
                k_global_slice = _normalize_map_for_display(k_global_slice)

                fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
                ax[0].set_title(f"{plane.capitalize()} Query Local/Fine ({style_tag})")
                ax[0].imshow(q_local_slice, cmap="viridis")
                _draw_marker(ax[0], qxy, color="white")

                ax[1].set_title(f"{plane.capitalize()} Query Global/Coarse ({style_tag})")
                ax[1].imshow(q_global_slice, cmap="viridis")
                _draw_marker(ax[1], qxy, color="white")

                ax[2].set_title(f"{plane.capitalize()} Target Local/Fine ({style_tag})")
                ax[2].imshow(k_local_slice, cmap="viridis")
                _draw_marker(ax[2], kxy, color="white")

                ax[3].set_title(f"{plane.capitalize()} Target Global/Coarse ({style_tag})")
                ax[3].imshow(k_global_slice, cmap="viridis")
                _draw_marker(ax[3], kxy, color="white")

                for axis in ax.ravel():
                    axis.set_xticks([])
                    axis.set_yticks([])

                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"embedding_maps_{style_tag}_{plane}.png"), dpi=150)
                plt.close(fig)


def save_similarity_maps_figures(ctx1, ctx2, result, out_dir, is_mri=False):
    _, _, sim_fine, sim_coarse = _compute_sim_fine_coarse(ctx1, ctx2, result["pt1"])

    target_norm = _normalize_volume_for_display(ctx2["img"]["img"], is_mri=is_mri)
    sim_fine_yxz = sim_fine.transpose(1, 2, 0)
    sim_coarse_yxz = sim_coarse.transpose(1, 2, 0)

    planes = ["axial", "sagittal", "coronal"]
    for plane in planes:
        target_slice, kxy = _slice_plane_with_point(target_norm, result["pt2"], plane)
        sim_fine_slice, _ = _slice_plane_with_point(sim_fine_yxz, result["pt2"], plane)
        sim_coarse_slice, _ = _slice_plane_with_point(sim_coarse_yxz, result["pt2"], plane)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].set_title(f"{plane.capitalize()} Fine Similarity")
        ax[0].imshow(target_slice, cmap="gray")
        ax[0].imshow(sim_fine_slice, cmap="jet", alpha=0.45)
        _draw_marker(ax[0], kxy, color="white")

        ax[1].set_title(f"{plane.capitalize()} Coarse Similarity")
        ax[1].imshow(target_slice, cmap="gray")
        ax[1].imshow(sim_coarse_slice, cmap="jet", alpha=0.45)
        _draw_marker(ax[1], kxy, color="white")

        for axis in ax.ravel():
            axis.set_xticks([])
            axis.set_yticks([])

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"similarity_maps_{plane}.png"), dpi=150)
        plt.close(fig)


def save_cycle_points_figures(ctx1, ctx2, result, out_dir, is_mri=False):
    query_norm = _normalize_volume_for_display(ctx1["img"]["img"], is_mri=is_mri)
    target_norm = _normalize_volume_for_display(ctx2["img"]["img"], is_mri=is_mri)
    planes = ["axial", "sagittal", "coronal"]
    for plane in planes:
        q_slice, qxy = _slice_plane_with_point(query_norm, result["pt1"], plane)
        t_slice, txy = _slice_plane_with_point(target_norm, result["pt2"], plane)
        qb_slice, qxy_back = _slice_plane_with_point(query_norm, result["pt1_back"], plane)

        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        ax[0].set_title(f"{plane.capitalize()} Query")
        ax[0].imshow(q_slice, cmap="gray")
        _draw_marker(ax[0], qxy, color="lime")

        ax[1].set_title(f"{plane.capitalize()} Target")
        ax[1].imshow(t_slice, cmap="gray")
        _draw_marker(ax[1], txy, color="deepskyblue")

        ax[2].set_title(f"{plane.capitalize()} Query + Cycle")
        ax[2].imshow(qb_slice, cmap="gray")
        _draw_marker(ax[2], qxy, color="lime")
        _draw_marker(ax[2], qxy_back, color="orange")

        for axis in ax.ravel():
            axis.set_xticks([])
            axis.set_yticks([])

        fig.suptitle(
            f"score_12={result['score_12']:.6f}, score_21={result['score_21']:.6f}, "
            f"voxel_err={result['voxel_error']:.4f}, mm_err={result['mm_error']:.4f}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(out_dir, f"cycle_points_{plane}.png"), dpi=150)
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
    save_cycle_points_figures(ctx1, ctx2, result, OUTPUT_DIR, is_mri=IS_MRI)
    save_embedding_maps_figures(ctx1, ctx2, result, OUTPUT_DIR, is_mri=IS_MRI)
    save_similarity_maps_figures(ctx1, ctx2, result, OUTPUT_DIR, is_mri=IS_MRI)

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
