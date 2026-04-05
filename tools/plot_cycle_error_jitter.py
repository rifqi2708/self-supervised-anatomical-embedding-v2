#!/usr/bin/env python3
"""Create an organ-wise jitter plot from latest cycle error CSV."""

import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "data" / "quadra_output" / "inc_cycle_error"
CSV_GLOB = "cycle_points_*.csv"

Y_COLUMN = "mm_error"  # Change to "voxel_error" when needed.
ALLOWED_Y_COLUMNS = ("mm_error", "voxel_error")

ORDER_MODE = "name"  # "name" or "median"
JITTER = 0.18
ALPHA = 0.65
POINT_SIZE = 8
SEED = 0
DPI = 200


def _is_writable_dir(path):
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def ensure_mplconfigdir():
    existing = os.environ.get("MPLCONFIGDIR")
    if existing and _is_writable_dir(existing):
        return

    default_dir = Path.home() / ".matplotlib"
    if _is_writable_dir(default_dir):
        os.environ["MPLCONFIGDIR"] = str(default_dir)
        return

    fallback_dir = Path(tempfile.gettempdir()) / "mplconfig"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback_dir)


ensure_mplconfigdir()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def extract_organ_name(mask_name):
    return strip_nii_suffix(Path(mask_name).name)


def find_latest_csv():
    csv_paths = sorted(CSV_DIR.glob(CSV_GLOB), key=lambda p: p.stat().st_mtime)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV found with pattern '{CSV_GLOB}' under: {CSV_DIR}")
    return csv_paths[-1]


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


def order_organs(organ_to_errors, mode):
    organs = list(organ_to_errors.keys())
    if mode == "name":
        return sorted(organs)
    if mode == "median":
        return sorted(organs, key=lambda key: median(organ_to_errors[key]))
    raise ValueError(f"Unsupported ORDER_MODE: {mode}")


def make_jitter_plot(organ_to_errors, output_path, y_column):
    organs = order_organs(organ_to_errors, ORDER_MODE)
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
    title = f"Cycle Error Jitter by Organ ({y_column})"

    ax.set_xlim(-0.5, n_organs - 0.5)
    ax.set_xticks(np.arange(n_organs))
    ax.set_xticklabels(organs, rotation=35, ha="right")
    ax.set_xlabel("Organ (mask)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def validate_config():
    if Y_COLUMN not in ALLOWED_Y_COLUMNS:
        raise ValueError(
            f"Invalid Y_COLUMN='{Y_COLUMN}'. Allowed values: {ALLOWED_Y_COLUMNS}."
        )
    if ORDER_MODE not in ("name", "median"):
        raise ValueError("ORDER_MODE must be 'name' or 'median'.")


def main():
    validate_config()
    csv_path = find_latest_csv()
    output_path = csv_path.with_name(f"{csv_path.stem}_jitter_{Y_COLUMN}.png")
    organ_to_errors = load_organ_errors(csv_path, Y_COLUMN)
    make_jitter_plot(organ_to_errors=organ_to_errors, output_path=output_path, y_column=Y_COLUMN)
    print(f"Source CSV: {csv_path}")
    print(f"Saved jitter plot to: {output_path}")
    print(f"Organs plotted: {len(organ_to_errors)}")


if __name__ == "__main__":
    main()
