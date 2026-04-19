#!/usr/bin/env python3
"""Post-run cycle error analyzer with in-script arguments (no CLI)."""

import csv
import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent

for _p in (str(PROJECT_ROOT), str(TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# In-script arguments (edit these values as needed).
CSV_PATH = "data/quadra_output/inc_cycle_error/cycle_points_*.csv"
DATASET_ROOT = "data/quadra_dataset_cropped"
OUTPUT_DIR = ""  # Empty means "<csv_dir>/<csv_stem>_analysis".
TOP_K_PER_ORGAN = 5
PER_LEVEL_SAMPLES = 2
MAX_LEVELS_PER_ORGAN = 5  # 0 means all levels.
SEED = 1
IS_MRI = False
DRY_RUN = False
CONFIG_FILE = "configs/sam/sam_NIHLN.py"
CHECKPOINT_FILE = "checkpoints/SAM.pth"
USE_SIM_COARSE = True
ENABLE_SIMILARITY_MAP_VIS = True
ENABLE_EMBED_SINGLE_CHANNEL_VIS = True
EMBED_SINGLE_CHANNEL_INDICES = [0, 16, 32, 64]
EMBED_SIM_DPI = 150
AUTO_SELECT_DEVICE = True
CUDA_DEVICE_ID = "0"
PROGRESS_EVERY = 25


REQUIRED_COLUMNS = (
    "idx",
    "mask_name",
    "pt1_x",
    "pt1_y",
    "pt1_z",
    "pt2_x",
    "pt2_y",
    "pt2_z",
    "pt1_back_x",
    "pt1_back_y",
    "pt1_back_z",
    "voxel_error",
    "mm_error",
    "score_12",
    "score_21",
)


def _import_read_image():
    try:
        from utils import read_image as _read_fn
    except ModuleNotFoundError as exc:
        # Fall back only if the "utils" module itself is not found.
        # If a dependency inside utils is missing (e.g., torchio), re-raise that real error.
        if getattr(exc, "name", "") != "utils":
            raise
        from tools.utils import read_image as _read_fn
    return _read_fn


def _import_embedding_interfaces():
    try:
        from interfaces import get_embedding as _get_embedding_fn, init as _init_fn
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") != "interfaces":
            raise
        from tools.interfaces import get_embedding as _get_embedding_fn, init as _init_fn
    return _get_embedding_fn, _init_fn


def _import_torch_modules():
    import torch as _torch
    import torch.nn.functional as _f

    return _torch, _f


def _configure_device_visibility():
    if not AUTO_SELECT_DEVICE:
        return

    # Make CUDA device choice before importing torch-dependent project modules.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE_ID))
    try:
        import torch
    except ModuleNotFoundError:
        print("WARNING: torch is not installed; device auto-selection skipped.")
        return

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_ID)
        print("Using GPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU")


def _format_seconds(seconds):
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    return f"{minutes}m{rem:04.1f}s"


def is_nifti_file(name):
    return isinstance(name, str) and (name.endswith(".nii.gz") or name.endswith(".nii"))


def strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def resolve_project_path(path_like):
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_csv_path(csv_path_pattern):
    pattern_path = Path(csv_path_pattern).expanduser()
    if not pattern_path.is_absolute():
        pattern_path = PROJECT_ROOT / pattern_path
    pattern_str = str(pattern_path)

    matches = [Path(p).resolve() for p in glob.glob(pattern_str)]
    if matches:
        return max(matches, key=lambda p: (p.stat().st_mtime, str(p)))

    direct_path = pattern_path.resolve()
    if direct_path.is_file():
        return direct_path
    raise FileNotFoundError(f"No CSV matched path/pattern: {csv_path_pattern}")


def resolve_output_dir(csv_path):
    if OUTPUT_DIR:
        return resolve_project_path(OUTPUT_DIR)
    return csv_path.with_name(f"{csv_path.stem}_analysis")


def extract_subject_id_and_organ(mask_name):
    if not mask_name:
        raise ValueError("mask_name is empty")
    parts = str(mask_name).split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"mask_name must look like '<subject>/<mask_file>', got: {mask_name}")
    subject_id = parts[0]
    organ = strip_nii_suffix(Path(parts[1]).name)
    if not organ:
        raise ValueError(f"Unable to parse organ from mask_name: {mask_name}")
    return subject_id, organ


def parse_required_int(row, key):
    value = row.get(key, "")
    return int(str(value).strip())


def parse_required_float(row, key):
    value = row.get(key, "")
    return float(str(value).strip())


def load_cycle_rows(csv_path):
    rows = []
    skipped = []

    with csv_path.open("r", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        header = reader.fieldnames or []
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in header]
        if missing_cols:
            raise ValueError(f"Missing required CSV columns: {missing_cols}. Found: {header}")

        for row_number, raw in enumerate(reader, start=2):
            try:
                subject_id, organ = extract_subject_id_and_organ(raw.get("mask_name", ""))
                pt1_x = parse_required_int(raw, "pt1_x")
                pt1_y = parse_required_int(raw, "pt1_y")
                pt1_z = parse_required_int(raw, "pt1_z")
                pt2_x = parse_required_int(raw, "pt2_x")
                pt2_y = parse_required_int(raw, "pt2_y")
                pt2_z = parse_required_int(raw, "pt2_z")
                pt1_back_x = parse_required_int(raw, "pt1_back_x")
                pt1_back_y = parse_required_int(raw, "pt1_back_y")
                pt1_back_z = parse_required_int(raw, "pt1_back_z")

                dx = pt1_back_x - pt1_x
                dy = pt1_back_y - pt1_y
                dz = pt1_back_z - pt1_z
                norm_sq = int(dx * dx + dy * dy + dz * dz)

                row = {
                    "idx": parse_required_int(raw, "idx"),
                    "mask_name": str(raw.get("mask_name", "")),
                    "subject_id": subject_id,
                    "organ": organ,
                    "pt1_x": pt1_x,
                    "pt1_y": pt1_y,
                    "pt1_z": pt1_z,
                    "pt2_x": pt2_x,
                    "pt2_y": pt2_y,
                    "pt2_z": pt2_z,
                    "pt1_back_x": pt1_back_x,
                    "pt1_back_y": pt1_back_y,
                    "pt1_back_z": pt1_back_z,
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "norm_sq": norm_sq,
                    "voxel_error": parse_required_float(raw, "voxel_error"),
                    "mm_error": parse_required_float(raw, "mm_error"),
                    "score_12": parse_required_float(raw, "score_12"),
                    "score_21": parse_required_float(raw, "score_21"),
                    "source_row_number": row_number,
                }
                rows.append(row)
            except Exception as exc:
                skipped.append(
                    {
                        "scope": "row_parse",
                        "row_number": row_number,
                        "idx": raw.get("idx", ""),
                        "mask_name": raw.get("mask_name", ""),
                        "subject_id": "",
                        "organ": "",
                        "reason": f"parse_error: {exc}",
                    }
                )

    if not rows:
        raise RuntimeError(f"No valid rows loaded from CSV: {csv_path}")
    return rows, skipped


def _register_selection(selected_map, row, reason):
    idx = int(row["idx"])
    entry = selected_map.get(idx)
    if entry is None:
        entry = {
            "row": row,
            "reasons": [],
            "selected_by_topk": False,
            "selected_by_level": False,
            "image_path": "",
        }
        selected_map[idx] = entry

    if reason not in entry["reasons"]:
        entry["reasons"].append(reason)
    if reason.startswith("top_mm_rank_"):
        entry["selected_by_topk"] = True
    if reason.startswith("level_"):
        entry["selected_by_level"] = True


def select_cases(rows):
    rng = np.random.default_rng(int(SEED))
    organ_to_rows = defaultdict(list)
    for row in rows:
        organ_to_rows[row["organ"]].append(row)

    selected_map = {}
    summary_rows = []

    for organ in sorted(organ_to_rows.keys()):
        organ_rows = organ_to_rows[organ]
        level_map = defaultdict(list)
        for row in organ_rows:
            level_map[row["norm_sq"]].append(row)

        top_rows = sorted(organ_rows, key=lambda r: (-r["mm_error"], r["idx"]))
        top_rows = top_rows[: max(0, int(TOP_K_PER_ORGAN))]
        for rank, row in enumerate(top_rows, start=1):
            _register_selection(selected_map, row, f"top_mm_rank_{rank}")

        level_keys = sorted(level_map.keys(), reverse=True)
        if int(MAX_LEVELS_PER_ORGAN) > 0:
            level_keys = level_keys[: int(MAX_LEVELS_PER_ORGAN)]

        level_selected_count = 0
        for level in level_keys:
            candidates = sorted(level_map[level], key=lambda r: r["idx"])
            sample_n = max(0, int(PER_LEVEL_SAMPLES))
            if sample_n == 0:
                chosen = []
            elif len(candidates) <= sample_n:
                chosen = candidates
            else:
                picked = rng.choice(len(candidates), size=sample_n, replace=False)
                chosen = [candidates[i] for i in sorted(picked.tolist())]

            for sample_rank, row in enumerate(chosen, start=1):
                _register_selection(selected_map, row, f"level_{level}_sample_{sample_rank}")
            level_selected_count += len(chosen)

        selected_entries_for_organ = [
            entry for entry in selected_map.values() if entry["row"]["organ"] == organ
        ]
        topk_unique = sum(1 for entry in selected_entries_for_organ if entry["selected_by_topk"])
        level_unique = sum(1 for entry in selected_entries_for_organ if entry["selected_by_level"])

        summary_rows.append(
            {
                "organ": organ,
                "total_rows": len(organ_rows),
                "unique_levels": len(level_map),
                "levels_considered": len(level_keys),
                "selected_topk_candidates": len(top_rows),
                "selected_level_samples": level_selected_count,
                "selected_unique": len(selected_entries_for_organ),
                "selected_topk_unique": topk_unique,
                "selected_level_unique": level_unique,
                "rendered_images": 0,
            }
        )

    selected_entries = sorted(
        selected_map.values(),
        key=lambda entry: (entry["row"]["organ"], entry["row"]["subject_id"], entry["row"]["idx"]),
    )
    return selected_entries, summary_rows


def list_subject_pair(subject_id, images_root):
    subject_image_dir = images_root / subject_id
    if not subject_image_dir.is_dir():
        raise FileNotFoundError(f"Subject image directory not found: {subject_image_dir}")

    image_files = [
        p.name
        for p in sorted(subject_image_dir.iterdir())
        if p.is_file() and is_nifti_file(p.name)
    ]
    test_files = [name for name in image_files if "_Test_" in name]
    retest_files = [name for name in image_files if "_Retest_" in name]

    if len(test_files) != 1 or len(retest_files) != 1:
        raise RuntimeError(
            f"Expected one Test and one Retest image under '{subject_image_dir}', "
            f"got Test={len(test_files)} Retest={len(retest_files)}."
        )

    return subject_image_dir / test_files[0], subject_image_dir / retest_files[0]


def load_subject_images_cached(subject_id, images_root, cache, read_image_fn):
    if subject_id in cache:
        return cache[subject_id]

    im1_file, im2_file = list_subject_pair(subject_id, images_root)
    img1_info, _, _ = read_image_fn(str(im1_file), mask_path=None, is_MRI=IS_MRI)
    img2_info, _, _ = read_image_fn(str(im2_file), mask_path=None, is_MRI=IS_MRI)
    cache[subject_id] = {
        "query_img": img1_info["img"],
        "target_img": img2_info["img"],
        "im1_file": str(im1_file),
        "im2_file": str(im2_file),
    }
    return cache[subject_id]


def _load_embedding_context(im_file, read_image_fn, get_embedding_fn, model):
    img, normed_im, norm_ratio = read_image_fn(
        str(im_file),
        mask_path=None,
        norm_spacing=(2.5, 2.5, 2.5),
        is_MRI=IS_MRI,
    )
    embedding = get_embedding_fn(normed_im, model)
    image_shape = img["shape"]
    target_imshape = (image_shape[3], image_shape[1], image_shape[2])  # z, y, x
    return {
        "im_file": str(im_file),
        "img": img,
        "norm_ratio": np.array(norm_ratio, dtype=float),
        "embedding": embedding,
        "target_imshape": target_imshape,
    }


def load_subject_embedding_contexts_cached(
    subject_id,
    images_root,
    cache,
    read_image_fn,
    get_embedding_fn,
    model,
):
    if subject_id in cache:
        return cache[subject_id]

    im1_file, im2_file = list_subject_pair(subject_id, images_root)
    ctx1 = _load_embedding_context(im1_file, read_image_fn, get_embedding_fn, model)
    ctx2 = _load_embedding_context(im2_file, read_image_fn, get_embedding_fn, model)
    cache[subject_id] = {
        "query_ctx": ctx1,
        "target_ctx": ctx2,
    }
    return cache[subject_id]


def sanitize_filename_text(text):
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("-")
    joined = "".join(safe).strip("-")
    if not joined:
        joined = "na"
    return joined[:140]


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
        sl = volume_yxz[y, :, :].T
        sl = sl[::-1, :]
        px, py = x, (sz - 1 - z)
    elif plane == "sagittal":
        sl = volume_yxz[:, x, :].T
        sl = sl[::-1, :]
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


def _point_to_fine_index(point_xyz, norm_ratio, fine_shape_zyx):
    point_xyz = np.asarray(point_xyz, dtype=float)
    idx_xyz = np.floor((point_xyz * np.asarray(norm_ratio, dtype=float)) / 2.0).astype(int)
    max_xyz = np.array(
        [fine_shape_zyx[2] - 1, fine_shape_zyx[1] - 1, fine_shape_zyx[0] - 1],
        dtype=int,
    )
    idx_xyz = np.clip(idx_xyz, 0, max_xyz)
    return idx_xyz


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


def _upsample_scalar_zyx_to_target(scalar_zyx, target_imshape, f_mod):
    return f_mod.interpolate(
        scalar_zyx.view(1, 1, *scalar_zyx.shape),
        target_imshape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]


def _embedding_single_channel_maps_for_context(ctx, point_xyz, channel_idx, f_mod):
    fine = ctx["embedding"][0]
    coarse = f_mod.interpolate(ctx["embedding"][1], fine.shape[2:], mode="trilinear", align_corners=False)

    c = int(fine.shape[1])
    ch = int(channel_idx)
    if ch < 0 or ch >= c:
        raise ValueError(f"single channel index {ch} out of range for embedding channels [0, {c-1}]")

    fine_map = fine[0, ch]
    coarse_map = coarse[0, ch]
    fine_map_up = _upsample_scalar_zyx_to_target(fine_map, ctx["target_imshape"], f_mod)
    coarse_map_up = _upsample_scalar_zyx_to_target(coarse_map, ctx["target_imshape"], f_mod)

    return (
        fine_map_up.detach().cpu().numpy().astype(np.float32),
        coarse_map_up.detach().cpu().numpy().astype(np.float32),
    )


def _compute_sim_fine_coarse_maps(query_ctx, key_ctx, pt_query_xyz, f_mod):
    query_point_normed = np.asarray(pt_query_xyz, dtype=float) * query_ctx["norm_ratio"]

    fine_query = query_ctx["embedding"][0]
    coarse_query = query_ctx["embedding"][1]
    fine_key = key_ctx["embedding"][0]
    coarse_key = key_ctx["embedding"][1]

    coarse_query = f_mod.interpolate(coarse_query, fine_query.shape[2:], mode="trilinear", align_corners=False)
    coarse_query = f_mod.normalize(coarse_query, dim=1)
    coarse_key = f_mod.interpolate(coarse_key, fine_key.shape[2:], mode="trilinear", align_corners=False)
    coarse_key = f_mod.normalize(coarse_key, dim=1)

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

    sim_fine = f_mod.linear(query_fine, key_fine.T).reshape(fine_key.shape[2:])
    sim_coarse = f_mod.linear(query_coarse, key_coarse.T).reshape(coarse_key.shape[2:])

    sim_fine_up = f_mod.interpolate(
        sim_fine.view(1, 1, *sim_fine.shape),
        key_ctx["target_imshape"],
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    sim_coarse_up = f_mod.interpolate(
        sim_coarse.view(1, 1, *sim_coarse.shape),
        key_ctx["target_imshape"],
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    return (
        sim_fine_up.detach().cpu().numpy().astype(np.float32),
        sim_coarse_up.detach().cpu().numpy().astype(np.float32),
    )


def save_single_channel_embedding_maps_figures(
    query_ctx,
    target_ctx,
    result,
    out_dir,
    case_tag,
    f_mod,
    dpi=150,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    plane_list = ["axial", "sagittal", "coronal"]

    for channel_idx in EMBED_SINGLE_CHANNEL_INDICES:
        q_local, q_global = _embedding_single_channel_maps_for_context(
            query_ctx, result["pt1"], channel_idx=channel_idx, f_mod=f_mod
        )
        k_local, k_global = _embedding_single_channel_maps_for_context(
            target_ctx, result["pt2"], channel_idx=channel_idx, f_mod=f_mod
        )

        q_local_yxz = q_local.transpose(1, 2, 0)
        q_global_yxz = q_global.transpose(1, 2, 0)
        k_local_yxz = k_local.transpose(1, 2, 0)
        k_global_yxz = k_global.transpose(1, 2, 0)

        style_tag = f"single_channel_ch{int(channel_idx):03d}"
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
            out_path = (out_dir / f"{case_tag}_embedding_maps_{style_tag}_{plane}.png").resolve()
            fig.savefig(str(out_path), dpi=dpi)
            plt.close(fig)
            saved_paths.append(str(out_path))

    return saved_paths


def save_similarity_maps_figures(
    query_ctx,
    target_ctx,
    result,
    out_dir,
    case_tag,
    f_mod,
    is_mri=False,
    dpi=150,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_fine, sim_coarse = _compute_sim_fine_coarse_maps(query_ctx, target_ctx, result["pt1"], f_mod=f_mod)
    target_norm = _normalize_volume_for_display(target_ctx["img"]["img"], is_mri=is_mri)
    sim_fine_yxz = sim_fine.transpose(1, 2, 0)
    sim_coarse_yxz = sim_coarse.transpose(1, 2, 0)

    saved_paths = {}
    for plane in ["axial", "sagittal", "coronal"]:
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
        out_path = (out_dir / f"{case_tag}_similarity_maps_{plane}.png").resolve()
        fig.savefig(str(out_path), dpi=dpi)
        plt.close(fig)
        saved_paths[plane] = str(out_path)

    return saved_paths


def visualize_cycle_result_multiplane(query_img, target_img, result, out_dir, file_stem, is_mri=False, dpi=150):
    query_norm = _normalize_volume_for_display(query_img, is_mri=is_mri)
    target_norm = _normalize_volume_for_display(target_img, is_mri=is_mri)
    plane_paths = {}
    planes = ["axial", "sagittal", "coronal"]

    out_dir.mkdir(parents=True, exist_ok=True)

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

        image_path = (out_dir / f"{file_stem}_{plane}.png").resolve()
        fig.savefig(str(image_path), dpi=dpi)
        plt.close(fig)
        plane_paths[plane] = str(image_path)

    return plane_paths


def render_selected_cases(selected_entries, output_dir, images_root):
    skipped = []
    image_cache = {}
    embedding_ctx_cache = {}
    rendered_by_organ = defaultdict(int)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    need_embedding_context = bool(ENABLE_SIMILARITY_MAP_VIS or ENABLE_EMBED_SINGLE_CHANNEL_VIS)
    total_cases = int(len(selected_entries))
    render_start_time = time.time()
    rendered_success_count = 0

    print(f"Rendering visualization cases: {total_cases}")

    model = None
    torch_mod = None
    f_mod = None
    get_embedding_fn = None
    embedding_ready = False
    embedding_disable_reason = ""

    _configure_device_visibility()
    try:
        read_image_fn = _import_read_image()
    except Exception as exc:
        for entry in selected_entries:
            row = entry["row"]
            skipped.append(
                {
                    "scope": "render",
                    "row_number": row["source_row_number"],
                    "idx": row["idx"],
                    "mask_name": row["mask_name"],
                    "subject_id": row["subject_id"],
                    "organ": row["organ"],
                    "reason": f"render_dependency_error: {exc!s}",
                }
            )
        return rendered_by_organ, skipped

    if need_embedding_context:
        try:
            model_t0 = time.time()
            print("Loading model for similarity/embedding visualization...")
            torch_mod, f_mod = _import_torch_modules()
            get_embedding_fn, init_fn = _import_embedding_interfaces()
            config_path = resolve_project_path(CONFIG_FILE)
            checkpoint_path = resolve_project_path(CHECKPOINT_FILE)
            if not config_path.is_file():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            model = init_fn(str(config_path), str(checkpoint_path))
            embedding_ready = True
            print(f"Model loaded in {_format_seconds(time.time() - model_t0)}")
        except Exception as exc:
            embedding_ready = False
            embedding_disable_reason = str(exc)
            print(f"WARNING: embedding/similarity visualization disabled: {embedding_disable_reason}")

    try:
        progress_every = max(1, int(PROGRESS_EVERY))
        for case_idx, entry in enumerate(selected_entries, start=1):
            row = entry["row"]
            subject_id = row["subject_id"]
            organ = row["organ"]

            try:
                subject_data = load_subject_images_cached(
                    subject_id=subject_id,
                    images_root=images_root,
                    cache=image_cache,
                    read_image_fn=read_image_fn,
                )

                result = {
                    "pt1": np.array([row["pt1_x"], row["pt1_y"], row["pt1_z"]], dtype=int),
                    "pt2": np.array([row["pt2_x"], row["pt2_y"], row["pt2_z"]], dtype=int),
                    "pt1_back": np.array(
                        [row["pt1_back_x"], row["pt1_back_y"], row["pt1_back_z"]], dtype=int
                    ),
                    "score_12": float(row["score_12"]),
                    "score_21": float(row["score_21"]),
                    "voxel_error": float(row["voxel_error"]),
                    "mm_error": float(row["mm_error"]),
                }

                reason_tag = sanitize_filename_text("+".join(sorted(entry["reasons"])))
                case_tag = (
                    f"{sanitize_filename_text(subject_id)}__{sanitize_filename_text(organ)}__"
                    f"idx{int(row['idx']):06d}__{reason_tag}__"
                    f"mm{float(row['mm_error']):.3f}__vox{float(row['voxel_error']):.3f}"
                )
                organ_dir = images_dir / sanitize_filename_text(organ)
                organ_dir.mkdir(parents=True, exist_ok=True)
                case_dir = (organ_dir / case_tag).resolve()
                case_dir.mkdir(parents=True, exist_ok=True)

                cycle_paths = visualize_cycle_result_multiplane(
                    query_img=subject_data["query_img"],
                    target_img=subject_data["target_img"],
                    result=result,
                    out_dir=case_dir,
                    file_stem="cycle_points",
                    is_mri=IS_MRI,
                    dpi=150,
                )

                similarity_paths = {}
                embedding_paths = []
                if need_embedding_context and embedding_ready:
                    embed_subject = load_subject_embedding_contexts_cached(
                        subject_id=subject_id,
                        images_root=images_root,
                        cache=embedding_ctx_cache,
                        read_image_fn=read_image_fn,
                        get_embedding_fn=get_embedding_fn,
                        model=model,
                    )
                    query_ctx = embed_subject["query_ctx"]
                    target_ctx = embed_subject["target_ctx"]

                    if ENABLE_SIMILARITY_MAP_VIS:
                        similarity_paths = save_similarity_maps_figures(
                            query_ctx=query_ctx,
                            target_ctx=target_ctx,
                            result=result,
                            out_dir=case_dir,
                            case_tag="similarity_maps",
                            f_mod=f_mod,
                            is_mri=IS_MRI,
                            dpi=EMBED_SIM_DPI,
                        )

                    if ENABLE_EMBED_SINGLE_CHANNEL_VIS:
                        embedding_paths = save_single_channel_embedding_maps_figures(
                            query_ctx=query_ctx,
                            target_ctx=target_ctx,
                            result=result,
                            out_dir=case_dir,
                            case_tag="embedding_maps",
                            f_mod=f_mod,
                            dpi=EMBED_SIM_DPI,
                        )
                elif need_embedding_context and (not embedding_ready):
                    entry["embedding_maps_dir"] = ""
                    entry["embedding_maps_count"] = 0
                    entry["similarity_axial_path"] = ""
                    entry["similarity_sagittal_path"] = ""
                    entry["similarity_coronal_path"] = ""
                    entry["render_note"] = (
                        f"embedding/similarity disabled for run: {embedding_disable_reason}"
                    )

                entry["case_dir"] = str(case_dir)
                entry["image_path"] = cycle_paths.get("axial", "")
                entry["axial_image_path"] = cycle_paths.get("axial", "")
                entry["sagittal_image_path"] = cycle_paths.get("sagittal", "")
                entry["coronal_image_path"] = cycle_paths.get("coronal", "")
                entry["similarity_axial_path"] = similarity_paths.get("axial", "")
                entry["similarity_sagittal_path"] = similarity_paths.get("sagittal", "")
                entry["similarity_coronal_path"] = similarity_paths.get("coronal", "")
                entry["embedding_maps_dir"] = str(case_dir) if embedding_paths else ""
                entry["embedding_maps_count"] = int(len(embedding_paths))
                if "render_note" not in entry:
                    entry["render_note"] = ""
                rendered_by_organ[organ] += 1
                rendered_success_count += 1
            except Exception as exc:
                skipped.append(
                    {
                        "scope": "render",
                        "row_number": row["source_row_number"],
                        "idx": row["idx"],
                        "mask_name": row["mask_name"],
                        "subject_id": subject_id,
                        "organ": organ,
                        "reason": f"render_error: {exc!s}",
                    }
                )

            should_report = (case_idx == 1) or (case_idx % progress_every == 0) or (case_idx == total_cases)
            if should_report:
                elapsed = time.time() - render_start_time
                speed = case_idx / elapsed if elapsed > 0 else 0.0
                remain = total_cases - case_idx
                eta_sec = (remain / speed) if speed > 0 else 0.0
                pct = (100.0 * case_idx / total_cases) if total_cases > 0 else 100.0
                print(
                    f"[render] {case_idx}/{total_cases} ({pct:.1f}%) "
                    f"ok={rendered_success_count} skipped={len(skipped)} "
                    f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_sec)}"
                )
    finally:
        if model is not None:
            del model
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()

    return rendered_by_organ, skipped


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_selected_rows_for_export(selected_entries):
    rows = []
    for entry in selected_entries:
        row = entry["row"]
        rows.append(
            {
                "idx": row["idx"],
                "mask_name": row["mask_name"],
                "subject_id": row["subject_id"],
                "organ": row["organ"],
                "pt1_x": row["pt1_x"],
                "pt1_y": row["pt1_y"],
                "pt1_z": row["pt1_z"],
                "pt2_x": row["pt2_x"],
                "pt2_y": row["pt2_y"],
                "pt2_z": row["pt2_z"],
                "pt1_back_x": row["pt1_back_x"],
                "pt1_back_y": row["pt1_back_y"],
                "pt1_back_z": row["pt1_back_z"],
                "dx": row["dx"],
                "dy": row["dy"],
                "dz": row["dz"],
                "norm_sq": row["norm_sq"],
                "voxel_error": row["voxel_error"],
                "mm_error": row["mm_error"],
                "score_12": row["score_12"],
                "score_21": row["score_21"],
                "selection_reason": "|".join(sorted(entry["reasons"])),
                "selected_by_topk": int(entry["selected_by_topk"]),
                "selected_by_level": int(entry["selected_by_level"]),
                "case_dir": entry.get("case_dir", ""),
                "image_path": entry.get("image_path", ""),
                "axial_image_path": entry.get("axial_image_path", ""),
                "sagittal_image_path": entry.get("sagittal_image_path", ""),
                "coronal_image_path": entry.get("coronal_image_path", ""),
                "similarity_axial_path": entry.get("similarity_axial_path", ""),
                "similarity_sagittal_path": entry.get("similarity_sagittal_path", ""),
                "similarity_coronal_path": entry.get("similarity_coronal_path", ""),
                "embedding_maps_dir": entry.get("embedding_maps_dir", ""),
                "embedding_maps_count": int(entry.get("embedding_maps_count", 0)),
                "render_note": entry.get("render_note", ""),
                "source_row_number": row["source_row_number"],
            }
        )
    return rows


def main():
    csv_path = resolve_csv_path(CSV_PATH)
    output_dir = resolve_output_dir(csv_path)
    dataset_root = resolve_project_path(DATASET_ROOT)
    images_root = dataset_root / "images"

    print(f"CSV path: {csv_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Top-K per organ: {TOP_K_PER_ORGAN}")
    print(f"Per-level samples: {PER_LEVEL_SAMPLES}")
    print(f"Max levels per organ: {MAX_LEVELS_PER_ORGAN}")
    print(f"Seed: {SEED}")
    print(f"Dry run: {DRY_RUN}")
    print(f"Enable similarity maps: {ENABLE_SIMILARITY_MAP_VIS}")
    print(f"Enable single-channel embedding maps: {ENABLE_EMBED_SINGLE_CHANNEL_VIS}")
    print(f"Auto select device: {AUTO_SELECT_DEVICE}")
    print(f"CUDA device id: {CUDA_DEVICE_ID}")
    if ENABLE_SIMILARITY_MAP_VIS or ENABLE_EMBED_SINGLE_CHANNEL_VIS:
        print(f"Config file: {resolve_project_path(CONFIG_FILE)}")
        print(f"Checkpoint file: {resolve_project_path(CHECKPOINT_FILE)}")
        print(f"Single-channel indices: {EMBED_SINGLE_CHANNEL_INDICES}")

    all_rows, skipped_parse = load_cycle_rows(csv_path)
    selected_entries, summary_rows = select_cases(all_rows)
    skipped_rows = list(skipped_parse)

    print(f"Loaded valid rows: {len(all_rows)}")
    print(f"Selected unique rows: {len(selected_entries)}")
    print(f"Rows skipped during parse: {len(skipped_parse)}")

    if DRY_RUN:
        print("Dry run enabled: skipping image rendering.")
        rendered_by_organ = defaultdict(int)
        skipped_render = []
    else:
        rendered_by_organ, skipped_render = render_selected_cases(
            selected_entries=selected_entries,
            output_dir=output_dir,
            images_root=images_root,
        )
        skipped_rows.extend(skipped_render)
        print(f"Rendered images: {sum(rendered_by_organ.values())}")
        print(f"Rows skipped during render: {len(skipped_render)}")

    for summary in summary_rows:
        summary["rendered_images"] = int(rendered_by_organ.get(summary["organ"], 0))

    selected_rows_for_export = build_selected_rows_for_export(selected_entries)
    summary_rows_for_export = sorted(summary_rows, key=lambda row: row["organ"])
    skipped_rows_for_export = sorted(
        skipped_rows,
        key=lambda row: (row.get("scope", ""), int(row.get("row_number", 0) or 0)),
    )

    selected_csv = output_dir / "selected_cases.csv"
    summary_csv = output_dir / "selection_summary.csv"
    skipped_csv = output_dir / "skipped_cases.csv"

    write_csv(
        selected_csv,
        [
            "idx",
            "mask_name",
            "subject_id",
            "organ",
            "pt1_x",
            "pt1_y",
            "pt1_z",
            "pt2_x",
            "pt2_y",
            "pt2_z",
            "pt1_back_x",
            "pt1_back_y",
            "pt1_back_z",
            "dx",
            "dy",
            "dz",
            "norm_sq",
            "voxel_error",
            "mm_error",
            "score_12",
            "score_21",
            "selection_reason",
            "selected_by_topk",
            "selected_by_level",
            "case_dir",
            "image_path",
            "axial_image_path",
            "sagittal_image_path",
            "coronal_image_path",
            "similarity_axial_path",
            "similarity_sagittal_path",
            "similarity_coronal_path",
            "embedding_maps_dir",
            "embedding_maps_count",
            "render_note",
            "source_row_number",
        ],
        selected_rows_for_export,
    )
    write_csv(
        summary_csv,
        [
            "organ",
            "total_rows",
            "unique_levels",
            "levels_considered",
            "selected_topk_candidates",
            "selected_level_samples",
            "selected_unique",
            "selected_topk_unique",
            "selected_level_unique",
            "rendered_images",
        ],
        summary_rows_for_export,
    )
    write_csv(
        skipped_csv,
        ["scope", "row_number", "idx", "mask_name", "subject_id", "organ", "reason"],
        skipped_rows_for_export,
    )

    print(f"selected cases csv saved: {selected_csv}")
    print(f"selection summary csv saved: {summary_csv}")
    print(f"skipped cases csv saved: {skipped_csv}")
    if not DRY_RUN:
        print(f"images saved under: {output_dir / 'images'}")


if __name__ == "__main__":
    main()
