#!/usr/bin/env python3
"""Create an organ-wise jitter plot from a cycle error CSV."""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
Y_COLUMN = "mm_error"  # Change to "voxel_error" when needed.
ALLOWED_Y_COLUMNS = ("mm_error", "voxel_error")
JITTER = 0.18
ALPHA = 0.65
POINT_SIZE = 8
SEED = 0
DPI = 200
ENABLE_OUTLIER_FILTER = False  # Set True to remove outliers.
OUTLIER_IQR_MULTIPLIER = 1.5


def strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def extract_organ_name(mask_name):
    return strip_nii_suffix(Path(mask_name).name)


def resolve_csv_path(arg_path):
    path = Path(arg_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return path


def load_organ_errors(csv_path, error_column):
    organ_to_errors = defaultdict(list)

    with csv_path.open("r", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = reader.fieldnames or []
        if "mask_name" not in fieldnames:
            raise ValueError(f"Column 'mask_name' not found in CSV. Available columns: {fieldnames}")
        if error_column not in fieldnames:
            raise ValueError(f"Column '{error_column}' not found in CSV. Available columns: {fieldnames}")

        for row in reader:
            mask_name = row.get("mask_name", "")
            err_str = row.get(error_column, "")
            if not mask_name or not err_str:
                continue
            try:
                err_val = float(err_str)
            except ValueError:
                continue
            organ_to_errors[extract_organ_name(mask_name)].append(err_val)

    if not organ_to_errors:
        raise RuntimeError(f"No valid rows found in CSV: {csv_path}")
    return organ_to_errors


def filter_outliers_iqr(values, multiplier=1.5):
    values = np.asarray(values, dtype=float)
    if values.size < 4:
        return values

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return values

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return values[(values >= lower) & (values <= upper)]


def apply_outlier_filter(organ_to_errors):
    filtered = {}
    removed = 0
    for organ, vals in organ_to_errors.items():
        before = len(vals)
        kept_vals = filter_outliers_iqr(vals, OUTLIER_IQR_MULTIPLIER)
        kept = len(kept_vals)
        if kept == 0:
            # Keep original values if filtering would remove all points in an organ.
            filtered[organ] = list(vals)
        else:
            filtered[organ] = kept_vals.tolist()
            removed += max(0, before - kept)
    return filtered, removed


def make_jitter_plot(organ_to_errors, output_path, y_column):
    organs = sorted(organ_to_errors.keys())
    n_organs = len(organs)

    fig_width = max(8.0, min(0.85 * n_organs + 3.0, 20.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    rng = np.random.default_rng(SEED)
    cmap = plt.get_cmap("tab10")

    for idx, organ in enumerate(organs):
        y = np.asarray(organ_to_errors[organ], dtype=float)
        x = idx + rng.uniform(-JITTER, JITTER, size=len(y))
        ax.scatter(x, y, s=POINT_SIZE, alpha=ALPHA, color=cmap(idx % 10), edgecolors="none")

    ylabel = "Cycle Error (mm)" if y_column == "mm_error" else "Cycle Error (voxel)"
    ax.set_xlim(-0.5, n_organs - 0.5)
    ax.set_xticks(np.arange(n_organs))
    ax.set_xticklabels(organs, rotation=35, ha="right")
    ax.set_xlabel("Organ (mask)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cycle Error Jitter by Organ ({y_column})")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    if Y_COLUMN not in ALLOWED_Y_COLUMNS:
        raise ValueError(f"Invalid Y_COLUMN='{Y_COLUMN}'. Allowed values: {ALLOWED_Y_COLUMNS}.")
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python tools/plot_cycle_error_jitter.py "
            "<path/to/cycle_points_*.csv>"
        )

    csv_path = resolve_csv_path(sys.argv[1])
    output_path = csv_path.with_name(f"{csv_path.stem}_jitter_{Y_COLUMN}.png")
    organ_to_errors = load_organ_errors(csv_path, Y_COLUMN)
    removed_points = 0
    if ENABLE_OUTLIER_FILTER:
        organ_to_errors, removed_points = apply_outlier_filter(organ_to_errors)
    make_jitter_plot(organ_to_errors, output_path, Y_COLUMN)
    print(f"Source CSV: {csv_path}")
    print(f"Saved jitter plot to: {output_path}")
    print(f"Organs plotted: {len(organ_to_errors)}")
    print(f"Outlier filter enabled: {ENABLE_OUTLIER_FILTER}")
    if ENABLE_OUTLIER_FILTER:
        print(f"Outlier IQR multiplier: {OUTLIER_IQR_MULTIPLIER}")
        print(f"Outlier points removed: {removed_points}")


if __name__ == "__main__":
    main()
