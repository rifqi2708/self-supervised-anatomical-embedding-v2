#!/usr/bin/env python3
"""Create publication-ready cycle-error plots for the aligned Quadra cohort."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


EXPECTED_ROWS = 108431
GROUP_ORDER = ("abdomen", "head_neck", "pelvis", "thorax")
GROUP_LABELS = {
    "abdomen": "Abdomen",
    "head_neck": "Head and neck",
    "pelvis": "Pelvis",
    "thorax": "Thorax",
}
GROUP_COLORS = {
    "abdomen": "#D9822B",
    "head_neck": "#4C78A8",
    "pelvis": "#7A9E5B",
    "thorax": "#D46A8C",
}
SEED = 20260721


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_label(value):
    value = value.replace("_", " ")
    value = value.replace("oesophagus", "esophagus")
    return value.capitalize()


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C9C9C9",
            "axes.linewidth": 1.1,
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_data(path):
    frame = pd.read_csv(
        path,
        usecols=["query_id", "subject_id", "group_name", "mask_name", "cycle_error_mm"],
    )
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError("Expected {} rows, found {}".format(EXPECTED_ROWS, len(frame)))
    if frame["query_id"].nunique() != EXPECTED_ROWS:
        raise RuntimeError("Query IDs are not unique")
    if set(frame["group_name"].unique()) != set(GROUP_ORDER):
        raise RuntimeError("Unexpected organ groups: {}".format(sorted(frame["group_name"].unique())))
    values = frame["cycle_error_mm"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("Cycle errors must be finite and non-negative")
    return frame


def category_arrays(frame, category, order):
    return [
        frame.loc[frame[category] == item, "cycle_error_mm"].to_numpy(dtype=np.float64)
        for item in order
    ]


def add_grid(axis, axis_name="y"):
    axis.grid(True, axis=axis_name, color="#E6E6E6", linestyle="--", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def detail_limit(values):
    value = float(np.percentile(values, 99))
    return max(10.0, value * 1.08)


def figure_width(categories):
    return max(12.0, min(22.0, 1.15 * len(categories) + 5.0))


def draw_box_pair(frame, category, order, colors, title, output):
    arrays = category_arrays(frame, category, order)
    all_values = np.concatenate(arrays)
    labels = [human_label(item) for item in order]
    width = figure_width(order)
    fig, axes = plt.subplots(2, 1, figsize=(width, 10.5), sharex=True)
    for index, (axis, mode) in enumerate(zip(axes, ("Full range", "Detail view"))):
        result = axis.boxplot(
            arrays,
            patch_artist=True,
            widths=0.64,
            whis=(5, 95),
            showfliers=True,
            flierprops={"marker": "o", "markersize": 2.0, "alpha": 0.22, "markeredgewidth": 0},
            medianprops={"color": "#1F1F1F", "linewidth": 2.0},
            whiskerprops={"color": "#333333", "linewidth": 1.0},
            capprops={"color": "#333333", "linewidth": 1.0},
        )
        for patch, color in zip(result["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
            patch.set_edgecolor("#222222")
            patch.set_linewidth(1.1)
        for flier, color in zip(result["fliers"], colors):
            flier.set_markerfacecolor(color)
        axis.set_ylabel("Cycle error (mm)")
        axis.set_title(mode, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylim(bottom=0)
        if index == 1:
            axis.set_ylim(0, detail_limit(all_values))
        add_grid(axis)
    axes[-1].set_xticks(range(1, len(labels) + 1))
    axes[-1].set_xticklabels(labels, rotation=32, ha="right")
    axes[-1].set_xlabel("Organ mask" if category == "mask_name" else "Organ group")
    fig.suptitle(title, fontsize=20, y=0.995)
    fig.text(
        0.5,
        0.963,
        "Whiskers: 5th–95th percentile; lower panel ends at the pooled 99th percentile",
        ha="center",
        color="#555555",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=240)
    plt.close(fig)


def jitter_positions(length, center, rng):
    return center + rng.uniform(-0.30, 0.30, size=length)


def draw_jitter_pair(frame, category, order, colors, title, output):
    arrays = category_arrays(frame, category, order)
    all_values = np.concatenate(arrays)
    labels = [human_label(item) for item in order]
    rng = np.random.RandomState(SEED)
    positions = [jitter_positions(len(values), index + 1, rng) for index, values in enumerate(arrays)]
    fig, axes = plt.subplots(2, 1, figsize=(figure_width(order), 10.5), sharex=True)
    for index, (axis, mode) in enumerate(zip(axes, ("Full range", "Detail view"))):
        for center, (values, x_values, color) in enumerate(zip(arrays, positions, colors), start=1):
            axis.scatter(
                x_values,
                values,
                s=7,
                color=color,
                alpha=0.13,
                linewidths=0,
                rasterized=True,
            )
            median = float(np.median(values))
            p95 = float(np.percentile(values, 95))
            axis.scatter([center], [median], marker="D", s=34, color="#1F1F1F", zorder=4)
            axis.hlines(p95, center - 0.28, center + 0.28, color="#1F1F1F", linewidth=1.5, zorder=4)
        axis.set_ylabel("Cycle error (mm)")
        axis.set_title(mode, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylim(bottom=0)
        if index == 1:
            axis.set_ylim(0, detail_limit(all_values))
        add_grid(axis)
    axes[-1].set_xticks(range(1, len(labels) + 1))
    axes[-1].set_xticklabels(labels, rotation=32, ha="right")
    axes[-1].set_xlabel("Organ mask" if category == "mask_name" else "Organ group")
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#1F1F1F", label="Median"),
            Line2D([0], [0], color="#1F1F1F", linewidth=1.5, label="95th percentile"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.suptitle(title, fontsize=20, y=0.995)
    fig.text(
        0.5,
        0.963,
        "All unique query points are shown; deterministic horizontal jitter is visual only",
        ha="center",
        color="#555555",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=240)
    plt.close(fig)


def blend_with_white(hex_color, fraction):
    value = hex_color.lstrip("#")
    rgb = np.asarray([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)
    mixed = rgb * fraction + 255.0 * (1.0 - fraction)
    return "#{:02X}{:02X}{:02X}".format(*[int(round(item)) for item in mixed])


def ecdf(values):
    ordered = np.sort(values)
    cumulative = np.arange(1, len(ordered) + 1, dtype=np.float64) / float(len(ordered))
    return ordered, cumulative


def add_threshold_guides(axis):
    for value, linestyle in ((2, ":"), (5, "--"), (10, "-.")):
        axis.axvline(value, color="#666666", linestyle=linestyle, linewidth=1.0, alpha=0.75)
    axis.text(2, 0.035, "2 mm", rotation=90, va="bottom", ha="right", color="#666666", fontsize=8)
    axis.text(5, 0.035, "5 mm", rotation=90, va="bottom", ha="right", color="#666666", fontsize=8)
    axis.text(10, 0.035, "10 mm", rotation=90, va="bottom", ha="right", color="#666666", fontsize=8)


def draw_group_ecdf(frame, group, masks, output):
    subset = frame[frame["group_name"] == group]
    base = GROUP_COLORS[group]
    shades = [blend_with_white(base, 0.42 + 0.50 * index / max(1, len(masks) - 1)) for index in range(len(masks))]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for axis in axes:
        for mask, color in zip(masks, shades):
            values = subset.loc[subset["mask_name"] == mask, "cycle_error_mm"].to_numpy(dtype=np.float64)
            x_values, y_values = ecdf(values)
            axis.step(x_values, y_values, where="post", color=color, linewidth=1.15, alpha=0.9, label=human_label(mask))
        pooled_x, pooled_y = ecdf(subset["cycle_error_mm"].to_numpy(dtype=np.float64))
        axis.step(pooled_x, pooled_y, where="post", color="#1F1F1F", linewidth=2.8, label="Pooled group")
        axis.set_ylim(0, 1.002)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlabel("Cycle error (mm)")
        add_grid(axis, axis_name="both")
        add_threshold_guides(axis)
    axes[0].set_xlim(0, float(subset["cycle_error_mm"].max()) * 1.02)
    axes[0].set_title("Full range", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Cumulative proportion")
    axes[1].set_xlim(0, 20)
    axes[1].set_title("Detail: 0–20 mm", loc="left", fontsize=12, fontweight="bold")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="lower right", frameon=True, ncol=2 if len(masks) > 7 else 1)
    fig.suptitle("Cycle-error ECDF — {}".format(GROUP_LABELS[group]), fontsize=20, y=0.985)
    fig.text(0.5, 0.943, "Mask-level curves with the complete pooled group emphasized in black", ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=240)
    plt.close(fig)


def draw_combined_ecdf(frame, output):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for axis in axes:
        for group in GROUP_ORDER:
            values = frame.loc[frame["group_name"] == group, "cycle_error_mm"].to_numpy(dtype=np.float64)
            x_values, y_values = ecdf(values)
            axis.step(x_values, y_values, where="post", color=GROUP_COLORS[group], linewidth=2.3, label=GROUP_LABELS[group])
        axis.set_ylim(0, 1.002)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlabel("Cycle error (mm)")
        add_grid(axis, axis_name="both")
        add_threshold_guides(axis)
    axes[0].set_xlim(0, float(frame["cycle_error_mm"].max()) * 1.02)
    axes[0].set_title("Full range", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Cumulative proportion")
    axes[1].set_xlim(0, 20)
    axes[1].set_title("Detail: 0–20 mm", loc="left", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle("Cycle-error ECDF by organ group", fontsize=20, y=0.985)
    fig.text(0.5, 0.943, "Aligned UAE-S cohort; 108,431 unique Test-mask queries", ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=240)
    plt.close(fig)


def write_summary(frame, path):
    rows = []
    for group in GROUP_ORDER:
        subset = frame[frame["group_name"] == group]
        for mask in sorted(subset["mask_name"].unique()):
            values = subset.loc[subset["mask_name"] == mask, "cycle_error_mm"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "group_name": group,
                    "mask_name": mask,
                    "count": len(values),
                    "mean_cycle_error_mm": float(np.mean(values)),
                    "median_cycle_error_mm": float(np.median(values)),
                    "p95_cycle_error_mm": float(np.percentile(values, 95)),
                    "p99_cycle_error_mm": float(np.percentile(values, 99)),
                    "maximum_cycle_error_mm": float(np.max(values)),
                }
            )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Finalized cycle_error_points.csv")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    output = Path(args.output_directory).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError("Output directory is not empty; pass --overwrite to replace plot files")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("box", "jitter", "ecdf"):
        (output / name).mkdir(exist_ok=True)

    configure_style()
    frame = load_data(input_path)
    created = []
    for group in GROUP_ORDER:
        subset = frame[frame["group_name"] == group]
        masks = sorted(subset["mask_name"].unique())
        colors = [GROUP_COLORS[group]] * len(masks)
        box_path = output / "box" / "cycle_error_box_{}.png".format(group)
        jitter_path = output / "jitter" / "cycle_error_jitter_{}.png".format(group)
        ecdf_path = output / "ecdf" / "cycle_error_ecdf_{}.png".format(group)
        draw_box_pair(subset, "mask_name", masks, colors, "Cycle-error distribution — {}".format(GROUP_LABELS[group]), box_path)
        draw_jitter_pair(subset, "mask_name", masks, colors, "Cycle-error jitter — {}".format(GROUP_LABELS[group]), jitter_path)
        draw_group_ecdf(frame, group, masks, ecdf_path)
        created.extend((box_path, jitter_path, ecdf_path))

    combined_colors = [GROUP_COLORS[group] for group in GROUP_ORDER]
    combined_box = output / "box" / "cycle_error_box_combined.png"
    combined_jitter = output / "jitter" / "cycle_error_jitter_combined.png"
    combined_ecdf = output / "ecdf" / "cycle_error_ecdf_combined.png"
    draw_box_pair(frame, "group_name", GROUP_ORDER, combined_colors, "Cycle-error distribution by organ group", combined_box)
    draw_jitter_pair(frame, "group_name", GROUP_ORDER, combined_colors, "Cycle-error jitter by organ group", combined_jitter)
    draw_combined_ecdf(frame, combined_ecdf)
    created.extend((combined_box, combined_jitter, combined_ecdf))

    summary_path = output / "plot_summary.csv"
    write_summary(frame, summary_path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(input_path), "bytes": input_path.stat().st_size, "sha256": sha256(input_path)},
        "rows": len(frame),
        "unique_queries": int(frame["query_id"].nunique()),
        "subjects": int(frame["subject_id"].nunique()),
        "groups": list(GROUP_ORDER),
        "plot_contract": {
            "box_whiskers": "5th_to_95th_percentile",
            "detail_limit": "pooled_99th_percentile_per_figure",
            "jitter_points": "all_unique_queries_deterministic_horizontal_jitter",
            "ecdf_detail_range_mm": [0, 20],
            "cycle_error_unit": "mm",
            "scientific_boundary": "cycle_consistency_is_not_independent_anatomical_accuracy",
        },
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(created)
        ],
        "summary": {"path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
    }
    manifest_path = output / "plot_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(manifest_path))
    print("Created {} PNG figures".format(len(created)))
    print("Output directory: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
