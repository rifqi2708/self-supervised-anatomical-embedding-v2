# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

VOXEL_DECIMALS = 2
MM_DECIMALS = 1


# Convert points from y,x,z array indexing order to x,y,z point-convention order
def yxz_to_xyz(points):
    points = np.asarray(points, dtype=int)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with 3 coordinates in y,x,z order, got shape {points.shape}")
    if points.ndim == 1:
        return points[[1, 0, 2]]
    return points[:, [1, 0, 2]]


# Convert points from x,y,z point-convention order to y,x,z array indexing order
def xyz_to_yxz(points):
    points = np.asarray(points, dtype=int)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with 3 coordinates in x,y,z order, got shape {points.shape}")
    if points.ndim == 1:
        return points[[1, 0, 2]]
    return points[:, [1, 0, 2]]


# Validate that a mask file path exists
def validate_mask_file(mask_file, mask_name):
    if mask_file is None:
        return
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"{mask_name} not found: {mask_file}")


# Validate a loaded mask array and ensure it matches image shape
def validate_origin_mask(origin_mask, image_array, mask_name):
    if origin_mask is None:
        raise ValueError(f"{mask_name} was not loaded. Provide a valid mask file.")
    origin_mask = np.asarray(origin_mask)
    image_array = np.asarray(image_array)
    if origin_mask.shape != image_array.shape:
        raise ValueError(
            f"{mask_name} shape {origin_mask.shape} does not match image shape {image_array.shape}."
        )
    return origin_mask


# Select random points from nonzero mask voxels
def sample_random_mask_points(mask, num_points, seed):
    mask = np.asarray(mask)
    foreground_points = np.argwhere(mask > 0)
    if foreground_points.size == 0:
        raise RuntimeError("No foreground points found in mask.")

    rng = np.random.default_rng(seed)
    replace = len(foreground_points) < num_points
    if replace:
        print(
            f"WARNING: requested {num_points} points but found {len(foreground_points)} foreground points; "
            "sampling with replacement."
        )
    selected_ids = rng.choice(len(foreground_points), size=num_points, replace=replace)
    selected_points_yxz = foreground_points[selected_ids].astype(int)
    return yxz_to_xyz(selected_points_yxz).astype(int)


# Validate sampled x,y,z points are inside the mask (mask is y,x,z indexing order)
def validate_sampled_points_inside_mask(points_xyz, mask_yxz, mask_name):
    points_xyz = np.asarray(points_xyz, dtype=int)
    if points_xyz.ndim == 1:
        points_xyz = points_xyz[None, :]
    points_yxz = xyz_to_yxz(points_xyz)
    mask_yxz = np.asarray(mask_yxz)

    if np.any(points_yxz < 0) or np.any(points_yxz[:, 0] >= mask_yxz.shape[0]) or np.any(
        points_yxz[:, 1] >= mask_yxz.shape[1]
    ) or np.any(points_yxz[:, 2] >= mask_yxz.shape[2]):
        raise RuntimeError(f"Sampled points contain out-of-bound coordinates for {mask_name}.")

    inside = mask_yxz[points_yxz[:, 0], points_yxz[:, 1], points_yxz[:, 2]] > 0
    if np.all(inside):
        return

    bad_idx = int(np.where(~inside)[0][0])
    bad_point_xyz = np.asarray(points_xyz[bad_idx], dtype=int)
    bad_point_yxz = np.asarray(points_yxz[bad_idx], dtype=int)
    bad_value = mask_yxz[bad_point_yxz[0], bad_point_yxz[1], bad_point_yxz[2]]
    raise RuntimeError(
        f"Sampled point {bad_point_xyz.tolist()} is outside {mask_name} "
        f"(mask value={int(bad_value)} at yxz={bad_point_yxz.tolist()})."
    )


# Validate that a fixed point is within the image bounds and return it as a numpy array
def validate_fixed_point(fixed_point, img):
    pt = np.asarray(fixed_point, dtype=int)
    if pt.shape != (3,):
        raise ValueError(f"Fixed point must have 3 coordinates, got {fixed_point}")

    shape_yxz = np.asarray(img["img"].shape, dtype=int)
    max_xyz = np.array([shape_yxz[1], shape_yxz[0], shape_yxz[2]], dtype=int)
    if np.any(pt < 0) or np.any(pt >= max_xyz):
        raise ValueError(
            f"Fixed point {pt.tolist()} out of bounds for image shape yxz={shape_yxz.tolist()} "
            f"(expected x,y,z in [0,{max_xyz[0]-1}]x[0,{max_xyz[1]-1}]x[0,{max_xyz[2]-1}])."
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
            f"{record['voxel_error']:.{VOXEL_DECIMALS}f} | {record['mm_error']:.{MM_DECIMALS}f} | "
            f"{record['score_12']:.6f} | {record['score_21']:.6f}"
        )


# Compute and print summary statistics for voxel and mm errors across all points
def print_summary(results):
    voxel_stats, mm_stats = compute_summary_stats(results)

    print("\nCycle error summary")
    print(
        "voxel: "
        f"count={voxel_stats['count']} mean={voxel_stats['mean']:.{VOXEL_DECIMALS}f} "
        f"median={voxel_stats['median']:.{VOXEL_DECIMALS}f} std={voxel_stats['std']:.{VOXEL_DECIMALS}f} "
        f"min={voxel_stats['min']:.{VOXEL_DECIMALS}f} max={voxel_stats['max']:.{VOXEL_DECIMALS}f} "
        f"p95={voxel_stats['p95']:.{VOXEL_DECIMALS}f}"
    )
    print(
        "mm:    "
        f"count={mm_stats['count']} mean={mm_stats['mean']:.{MM_DECIMALS}f} "
        f"median={mm_stats['median']:.{MM_DECIMALS}f} std={mm_stats['std']:.{MM_DECIMALS}f} "
        f"min={mm_stats['min']:.{MM_DECIMALS}f} max={mm_stats['max']:.{MM_DECIMALS}f} "
        f"p95={mm_stats['p95']:.{MM_DECIMALS}f}"
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


# Write per-point cycle matching results to CSV with a mask_name column
def write_points_csv_with_mask(results, out_path):
    include_original_coords = any(
        ("pt1_orig" in record) or ("pt2_orig" in record) or ("pt1_back_orig" in record) for record in results
    )
    include_offsets = any(("im1_z_offset" in record) or ("im2_z_offset" in record) for record in results)
    include_coord_space = any("coord_space" in record for record in results)

    fieldnames = [
        "idx",
        "mask_name",
        "subject_id",
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
    if include_coord_space:
        fieldnames.append("coord_space")
    if include_original_coords:
        fieldnames.extend(
            [
                "pt1_orig_x",
                "pt1_orig_y",
                "pt1_orig_z",
                "pt2_orig_x",
                "pt2_orig_y",
                "pt2_orig_z",
                "pt1_back_orig_x",
                "pt1_back_orig_y",
                "pt1_back_orig_z",
            ]
        )
    if include_offsets:
        fieldnames.extend(["im1_z_offset", "im2_z_offset"])

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(results):
            pt1 = np.asarray(record["pt1"], dtype=int)
            pt2 = np.asarray(record["pt2"], dtype=int)
            pt1_back = np.asarray(record["pt1_back"], dtype=int)
            row = {
                "idx": idx,
                "mask_name": str(record.get("mask_name", "")),
                "subject_id": str(record.get("subject_id", "")),
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
            if include_coord_space:
                row["coord_space"] = str(record.get("coord_space", ""))
            if include_original_coords:
                pt1_orig = np.asarray(record.get("pt1_orig", [-1, -1, -1]), dtype=int)
                pt2_orig = np.asarray(record.get("pt2_orig", [-1, -1, -1]), dtype=int)
                pt1_back_orig = np.asarray(record.get("pt1_back_orig", [-1, -1, -1]), dtype=int)
                row.update(
                    {
                        "pt1_orig_x": int(pt1_orig[0]),
                        "pt1_orig_y": int(pt1_orig[1]),
                        "pt1_orig_z": int(pt1_orig[2]),
                        "pt2_orig_x": int(pt2_orig[0]),
                        "pt2_orig_y": int(pt2_orig[1]),
                        "pt2_orig_z": int(pt2_orig[2]),
                        "pt1_back_orig_x": int(pt1_back_orig[0]),
                        "pt1_back_orig_y": int(pt1_back_orig[1]),
                        "pt1_back_orig_z": int(pt1_back_orig[2]),
                    }
                )
            if include_offsets:
                row.update(
                    {
                        "im1_z_offset": int(record.get("im1_z_offset", -1)),
                        "im2_z_offset": int(record.get("im2_z_offset", -1)),
                    }
                )
            writer.writerow(row)


# Write per-mask and optional global voxel/mm summary statistics to one CSV with mask labels
def write_summary_with_mask_labels_csv(
    per_mask_rows,
    out_path,
    global_voxel_stats=None,
    global_mm_stats=None,
    all_masks_label="ALL_MASKS",
):
    fieldnames = ["mask_name", "metric", "count", "mean", "median", "std", "min", "max", "p95"]
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_mask_rows:
            writer.writerow({"mask_name": row["mask_name"], "metric": "voxel", **row["voxel_stats"]})
            writer.writerow({"mask_name": row["mask_name"], "metric": "mm", **row["mm_stats"]})
        if global_voxel_stats is not None:
            writer.writerow({"mask_name": all_masks_label, "metric": "voxel", **global_voxel_stats})
        if global_mm_stats is not None:
            writer.writerow({"mask_name": all_masks_label, "metric": "mm", **global_mm_stats})


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
        "+",
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
        f"voxel_err={result['voxel_error']:.{VOXEL_DECIMALS}f}, mm_err={result['mm_error']:.{MM_DECIMALS}f}",
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
    #update
