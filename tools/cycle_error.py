# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import csv
import os
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
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


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project
CONFIG_FILE = "configs/sam/sam_NIHLN.py"
CHECKPOINT_FILE = "checkpoints/SAM.pth"
DEFAULT_IM1_FILE = "data/raw_data/NIH_lymph_node/ABD_LYMPH_001.nii.gz"
DEFAULT_IM2_FILE = "data/raw_data/NIH_lymph_node/ABD_LYMPH_002.nii.gz"

#loading data and embeddings for one image, to be used as context for cycle error computation
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
    bg_th = img.min() + (img.max() - img.min()) / 15.0
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
        "p95": float(np.percentile(values, 95)),
    }

# Compute summary statistics for voxel and mm errors across all points
def compute_summary_stats(results):
    voxel_errors = [record["voxel_error"] for record in results]
    mm_errors = [record["mm_error"] for record in results]
    voxel_stats = summarize(voxel_errors)
    mm_stats = summarize(mm_errors)
    return voxel_stats, mm_stats

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
    voxel_stats, mm_stats = compute_summary_stats(results)

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
    return voxel_stats, mm_stats

# Write voxel/mm summary statistics to CSV
def write_summary_csv(voxel_stats, mm_stats, out_path):
    fieldnames = ["metric", "count", "mean", "median", "std", "min", "max", "p95"]
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"metric": "voxel", **voxel_stats})
        writer.writerow({"metric": "mm", **mm_stats})

# Write per-point cycle matching results to CSV
def write_points_csv(results, out_path):
    fieldnames = [
        "idx",
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
    ]
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(results):
            pt1 = np.asarray(record["pt1"], dtype=int)
            pt2 = np.asarray(record["pt2"], dtype=int)
            pt1_back = np.asarray(record["pt1_back"], dtype=int)
            writer.writerow(
                {
                    "idx": idx,
                    "pt1_x": int(pt1[0]),
                    "pt1_y": int(pt1[1]),
                    "pt1_z": int(pt1[2]),
                    "pt2_x": int(pt2[0]),
                    "pt2_y": int(pt2[1]),
                    "pt2_z": int(pt2[2]),
                    "pt1_back_x": int(pt1_back[0]),
                    "pt1_back_y": int(pt1_back[1]),
                    "pt1_back_z": int(pt1_back[2]),
                    "voxel_error": float(record["voxel_error"]),
                    "mm_error": float(record["mm_error"]),
                    "score_12": float(record["score_12"]),
                    "score_21": float(record["score_21"]),
                }
            )

# Prepare a 2D axial slice from a 3D image for visualization, applying windowing if needed
def prepare_axial_slice(img3d, z_idx, is_mri=False):
    img3d = np.asarray(img3d, dtype=np.float32)
    if img3d.ndim != 3:
        raise ValueError(f"Expected 3D image for visualization, got shape {img3d.shape}")

    z_idx = int(np.clip(z_idx, 0, img3d.shape[2] - 1))
    axial = img3d.transpose(2, 0, 1)
    if is_mri:
        window_low = float(axial.min())
        window_high = float(axial.max())
    else:
        window_low, window_high = -100.0, 200.0

    if window_high <= window_low:
        return np.zeros_like(axial[z_idx], dtype=np.float32)

    axial = np.clip(axial, window_low, window_high)
    axial = (axial - window_low) / (window_high - window_low)
    return axial[z_idx].astype(np.float32)

# Build a side-by-side canvas for two 2D slices, returning the combined image and the bounding boxes of each slice within the canvas
def build_side_by_side_canvas(left_slice, right_slice, gap=8):
    left_h, left_w = left_slice.shape
    right_h, right_w = right_slice.shape
    canvas_h = max(left_h, right_h)
    canvas_w = left_w + gap + right_w
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    left_y = (canvas_h - left_h) // 2
    right_y = (canvas_h - right_h) // 2
    left_x = 0
    right_x = left_w + gap

    canvas[left_y : left_y + left_h, left_x : left_x + left_w] = left_slice
    canvas[right_y : right_y + right_h, right_x : right_x + right_w] = right_slice
    return canvas, (left_x, left_y, left_w, left_h), (right_x, right_y, right_w, right_h)

# Draw a point with its coordinates annotated on a matplotlib axis
def draw_point_with_coord(ax, x, y, coord_text, color, label=None):
    ax.plot(
        float(x),
        float(y),
        "o",
        markerfacecolor="none",
        markeredgecolor=color,
        markersize=11,
        markeredgewidth=2,
        label=label,
    )
    ax.annotate(
        coord_text,
        xy=(float(x), float(y)),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        color=color,
        fontsize=8,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.35, "pad": 1.0},
    )

# Visualize the cycle error result for a single point, showing the original slices, matched points, and cycle points in a 2x2 layout
def visualize_cycle_result(query_img, target_img, result, out_path=None, show=True, is_mri=False, viz_layout=(2, 2)):
    if tuple(viz_layout) != (2, 2):
        raise ValueError(f"Only 2x2 layout is supported, got {viz_layout}")

    pt1 = np.asarray(result["pt1"], dtype=int)
    pt2 = np.asarray(result["pt2"], dtype=int)
    pt1_back = np.asarray(result["pt1_back"], dtype=int)

    query_slice = prepare_axial_slice(query_img, pt1[2], is_mri=is_mri)
    target_slice = prepare_axial_slice(target_img, pt2[2], is_mri=is_mri)
    original_panel, query_box, target_box = build_side_by_side_canvas(query_slice, target_slice)

    fig, ax = plt.subplots(2, 2, figsize=(16, 14))
    ax00, ax01, ax10, ax11 = ax.ravel()

    ax00.set_title("1) Original query + target")
    ax00.imshow(original_panel, cmap="gray")
    ax00.set_xticks([])
    ax00.set_yticks([])
    ax00.text(
        query_box[0] + query_box[2] * 0.5,
        max(query_box[1] - 4, 8),
        "Query",
        color="white",
        fontsize=10,
        ha="center",
        va="bottom",
    )
    ax00.text(
        target_box[0] + target_box[2] * 0.5,
        max(target_box[1] - 4, 8),
        "Target",
        color="white",
        fontsize=10,
        ha="center",
        va="bottom",
    )

    ax01.set_title("2) Query with query point")
    ax01.imshow(query_slice, cmap="gray")
    draw_point_with_coord(
        ax01,
        pt1[0],
        pt1[1],
        f"({pt1[0]}, {pt1[1]}, {pt1[2]})",
        color="lime",
        label="query point",
    )
    ax01.set_xticks([])
    ax01.set_yticks([])

    ax10.set_title("3) Target with matched query point")
    ax10.imshow(target_slice, cmap="gray")
    draw_point_with_coord(
        ax10,
        pt2[0],
        pt2[1],
        f"({pt2[0]}, {pt2[1]}, {pt2[2]})",
        color="deepskyblue",
        label="matched query point",
    )
    ax10.set_xticks([])
    ax10.set_yticks([])

    ax11.set_title("4) Query with query + cycle points")
    ax11.imshow(query_slice, cmap="gray")
    draw_point_with_coord(
        ax11,
        pt1[0],
        pt1[1],
        f"({pt1[0]}, {pt1[1]}, {pt1[2]})",
        color="lime",
        label="query point",
    )
    draw_point_with_coord(
        ax11,
        pt1_back[0],
        pt1_back[1],
        f"({pt1_back[0]}, {pt1_back[1]}, {pt1_back[2]})",
        color="orange",
        label="cycle point",
    )
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax11.legend(loc="upper right", fontsize=9, framealpha=0.8)

    fig.suptitle(
        f"score_12={result['score_12']:.6f}, score_21={result['score_21']:.6f}, "
        f"voxel_err={result['voxel_error']:.4f}, mm_err={result['mm_error']:.4f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path is not None:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)

# Main function to run the cycle error computation and visualization for a set of points between two images
def run_cycle(
    im1_file=DEFAULT_IM1_FILE,
    im2_file=DEFAULT_IM2_FILE,
    point_mode="random",
    fixed_point=None,
    num_points=20,
    seed=0,
    is_mri=False,
    use_sim_coarse=True,
    visualize=True,
    viz_show=True,
    viz_save=True,
    viz_dir="tools/cycle_vis",
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

    print_result_table(results)
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
