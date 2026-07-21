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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = "data/quadra_output/inc_cycle_error/cycle_points_20260405_155714.csv"
Y_COLUMN = "mm_error"  # Change to "voxel_error" when needed.
ALLOWED_Y_COLUMNS = ("mm_error", "voxel_error")
JITTER = 0.18
ALPHA = 0.65
POINT_SIZE = 8
SEED = 0
DPI = 200
ENABLE_PERCENTILE_FILTER = False  # Default behavior; can be overridden via CLI flag.
FILTER_PERCENTILE = 90.0  # Keep the central 90% of values (remove 5% low + 5% high tails).


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


def filter_outliers_percentile(values, keep_percentile=90.0):
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return values

    tail = (100.0 - keep_percentile) / 2.0
    lower = np.percentile(values, tail)
    upper = np.percentile(values, 100.0 - tail)
    return values[(values >= lower) & (values <= upper)]


def apply_percentile_filter(organ_to_errors, keep_percentile):
    filtered = {}
    removed = 0
    for organ, vals in organ_to_errors.items():
        before = len(vals)
        kept_vals = filter_outliers_percentile(vals, keep_percentile)
        kept = len(kept_vals)
        if kept == 0:
            # Keep original values if filtering would remove all points in an organ.
            filtered[organ] = list(vals)
        else:
            filtered[organ] = kept_vals.tolist()
            removed += max(0, before - kept)
    return filtered, removed


def resolve_filter_enabled(argv):
    usage = "Usage: python tools/quadra/plot_cycle_error_jitter.py [--filter | --no-filter]"
    if not argv:
        return ENABLE_PERCENTILE_FILTER
    if argv == ["-h"] or argv == ["--help"]:
        print(usage)
        raise SystemExit(0)
    if len(argv) != 1:
        raise SystemExit(usage)

    arg = argv[0].strip().lower()
    if arg in ("--filter", "--filter-on"):
        return True
    if arg in ("--no-filter", "--filter-off"):
        return False
    raise SystemExit(usage)


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
    if not INPUT_CSV:
        raise ValueError("INPUT_CSV is empty. Set INPUT_CSV to a CSV file path.")
    if not (0.0 < FILTER_PERCENTILE <= 100.0):
        raise ValueError("FILTER_PERCENTILE must be in the range (0, 100].")

    filter_enabled = resolve_filter_enabled(sys.argv[1:])
    csv_path = resolve_csv_path(INPUT_CSV)
    out_suffix = f"_jitter_{Y_COLUMN}"
    if filter_enabled:
        out_suffix += f"_p{int(FILTER_PERCENTILE)}"
    output_path = csv_path.with_name(f"{csv_path.stem}{out_suffix}.png")
    organ_to_errors = load_organ_errors(csv_path, Y_COLUMN)
    removed_points = 0
    if filter_enabled:
        organ_to_errors, removed_points = apply_percentile_filter(organ_to_errors, FILTER_PERCENTILE)
    make_jitter_plot(organ_to_errors, output_path, Y_COLUMN)
    print(f"Source CSV: {csv_path}")
    print(f"Saved jitter plot to: {output_path}")
    print(f"Organs plotted: {len(organ_to_errors)}")
    print(f"Percentile filter enabled: {filter_enabled}")
    if filter_enabled:
        print(f"Filter percentile: {FILTER_PERCENTILE}")
        print(f"Filtered points removed: {removed_points}")


if __name__ == "__main__":
    main()
