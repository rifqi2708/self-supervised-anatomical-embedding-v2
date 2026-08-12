"""Reusable SuperPoint utilities for bounded Quadra CT experiments.

The upstream SuperPoint checkout remains an external, pinned dependency. This
module owns the Quadra-specific input preparation and provenance checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import time
from pathlib import Path

import numpy as np


EXPECTED_SUPERPOINT_COMMIT = "1411bbd68c50163555d39c1b26e9e046ebd48f27"
EXPECTED_SUPERPOINT_CHECKPOINT_SHA256 = (
    "cd5d19a5061848e248c17728878ea166b66512076d43c77dbcf27f4a88a56084"
)
DEFAULT_WINDOW_CENTER_HU = 40.0
DEFAULT_WINDOW_WIDTH_HU = 400.0
SUPERPOINT_STRIDE = 8


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository, *args):
    return subprocess.check_output(
        ["git", "-C", str(repository), *args],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def validate_superpoint_assets(
    superpoint_root,
    checkpoint,
    expected_commit=EXPECTED_SUPERPOINT_COMMIT,
    expected_checkpoint_sha256=EXPECTED_SUPERPOINT_CHECKPOINT_SHA256,
):
    """Verify the external implementation and weight provenance."""

    root = Path(superpoint_root).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    implementation = root / "superpoint_pytorch.py"
    if not root.is_dir():
        raise FileNotFoundError("SuperPoint repository not found: {}".format(root))
    if not implementation.is_file():
        raise FileNotFoundError("PyTorch implementation not found: {}".format(implementation))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("SuperPoint checkpoint not found: {}".format(checkpoint_path))

    observed_commit = _git_output(root, "rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise RuntimeError(
            "SuperPoint commit mismatch: expected {}, observed {}".format(
                expected_commit, observed_commit
            )
        )
    if _git_output(root, "status", "--porcelain"):
        raise RuntimeError("SuperPoint repository has uncommitted changes")
    repository_origin = _git_output(root, "remote", "get-url", "origin")

    observed_sha256 = sha256_file(checkpoint_path)
    if observed_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(
            "SuperPoint checkpoint SHA-256 mismatch: expected {}, observed {}".format(
                expected_checkpoint_sha256, observed_sha256
            )
        )
    return {
        "repository": str(root),
        "repository_origin": repository_origin,
        "implementation": str(implementation),
        "commit": observed_commit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
    }


def window_and_normalize_ct(
    slice_hu,
    center=DEFAULT_WINDOW_CENTER_HU,
    width=DEFAULT_WINDOW_WIDTH_HU,
):
    """Clip one CT slice to a fixed HU window and map it to float32 [0, 1]."""

    values = np.asarray(slice_hu)
    if values.ndim != 2:
        raise ValueError("Expected one 2D CT slice, got shape {}".format(values.shape))
    if not np.isfinite(values).all():
        raise ValueError("CT slice contains non-finite values")
    if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
        raise ValueError("Window centre and width must be finite, with width > 0")

    lower = float(center) - float(width) / 2.0
    upper = float(center) + float(width) / 2.0
    normalized = (np.clip(values, lower, upper) - lower) / float(width)
    return np.asarray(normalized, dtype=np.float32)


def ensure_stride_compatible(image, stride=SUPERPOINT_STRIDE):
    """Refuse implicit resizing or padding in the reference smoke workflow."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("Expected a 2D input, got shape {}".format(array.shape))
    if stride <= 0:
        raise ValueError("Stride must be positive")
    if array.shape[0] % stride or array.shape[1] % stride:
        raise ValueError(
            "Input shape {} is not divisible by stride {}; explicit padding policy required".format(
                array.shape, stride
            )
        )
    return tuple(int(value) for value in array.shape)


def native_xy_to_model_yx(slice_xy):
    """Convert a native NIfTI ``[x, y]`` slice to image ``[row=y, col=x]``."""

    array = np.asarray(slice_xy)
    if array.ndim != 2:
        raise ValueError("Expected a native [x, y] slice, got shape {}".format(array.shape))
    return np.ascontiguousarray(array.T)


def model_keypoints_to_raw_voxels(keypoints_xy, slice_index):
    """Map SuperPoint ``(column=x, row=y)`` pixels to NIfTI voxel ``(x, y, z)``."""

    keypoints = np.asarray(keypoints_xy)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("Expected keypoints with shape [N, 2], got {}".format(keypoints.shape))
    raw_voxels = np.empty((keypoints.shape[0], 3), dtype=np.float32)
    raw_voxels[:, 0] = keypoints[:, 0]
    raw_voxels[:, 1] = keypoints[:, 1]
    raw_voxels[:, 2] = float(slice_index)
    return raw_voxels


def _axis_two_is_axial(axcodes):
    first_two = set(axcodes[:2])
    has_lr = bool(first_two.intersection({"L", "R"}))
    has_ap = bool(first_two.intersection({"A", "P"}))
    return axcodes[2] in {"S", "I"} and has_lr and has_ap


def load_axial_ct_slice(ct_path, slice_index):
    """Load one native-grid axial slice without loading the complete volume."""

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("nibabel is required to load NIfTI CT data") from exc

    path = Path(ct_path).resolve()
    image = nib.load(str(path))
    if image.ndim != 3:
        raise ValueError("Expected a 3D CT NIfTI, got shape {}".format(image.shape))
    index = int(slice_index)
    if index < 0 or index >= image.shape[2]:
        raise IndexError(
            "Slice index {} outside axis-2 bounds [0, {})".format(index, image.shape[2])
        )

    axcodes = tuple(str(code) for code in nib.aff2axcodes(image.affine))
    if not _axis_two_is_axial(axcodes):
        raise ValueError(
            "NIfTI axis 2 is not an anatomical axial direction: axcodes={}".format(axcodes)
        )
    slice_hu = np.asarray(image.dataobj[:, :, index], dtype=np.float32)
    if not np.isfinite(slice_hu).all():
        raise ValueError("Selected CT slice contains non-finite values")
    metadata = {
        "ct_path": str(path),
        "volume_shape_xyz": [int(value) for value in image.shape],
        "spacing_xyz_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation_codes": list(axcodes),
        "slice_axis": 2,
        "slice_index": index,
        "slice_shape_xy": [int(value) for value in slice_hu.shape],
        "slice_hu_min": float(slice_hu.min()),
        "slice_hu_max": float(slice_hu.max()),
        "nibabel_version": nib.__version__,
    }
    return slice_hu, metadata


def load_aligned_mask_slices(mask_dir, ct_path, slice_index):
    """Load non-empty mask slices after strict native-grid geometry checks."""

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("nibabel is required to load NIfTI masks") from exc

    directory = Path(mask_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError("Mask directory not found: {}".format(directory))
    ct_image = nib.load(str(Path(ct_path).resolve()))
    index = int(slice_index)
    if index < 0 or index >= ct_image.shape[2]:
        raise IndexError("Slice index {} outside CT bounds".format(index))

    mask_paths = sorted(
        path for path in directory.glob("*.nii*") if not path.name.startswith("._")
    )
    if not mask_paths:
        raise FileNotFoundError("No NIfTI masks found under {}".format(directory))

    visible = []
    for path in mask_paths:
        image = nib.load(str(path))
        if image.ndim != 3 or image.shape != ct_image.shape:
            raise ValueError(
                "Mask shape mismatch for {}: mask {}, CT {}".format(
                    path.name, image.shape, ct_image.shape
                )
            )
        if not np.allclose(image.affine, ct_image.affine, rtol=0.0, atol=1e-5):
            raise ValueError("Mask affine mismatch for {}".format(path.name))
        mask_xy = np.asarray(image.dataobj[:, :, index])
        if not np.isfinite(mask_xy).all():
            raise ValueError("Mask slice contains non-finite values: {}".format(path.name))
        foreground = mask_xy > 0
        pixel_count = int(foreground.sum())
        if not pixel_count:
            continue
        name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
        visible.append(
            {
                "name": name,
                "path": str(path),
                "foreground_pixel_count": pixel_count,
                "mask_yx": native_xy_to_model_yx(foreground),
            }
        )
    return visible


def _load_external_module(implementation_path):
    module_name = "quadra_pinned_superpoint_pytorch"
    spec = importlib.util.spec_from_file_location(module_name, str(implementation_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import SuperPoint implementation: {}".format(implementation_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_superpoint_model(superpoint_root, checkpoint, device="auto"):
    """Load the strictly verified PyTorch SuperPoint model."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("PyTorch is required for SuperPoint inference") from exc

    provenance = validate_superpoint_assets(superpoint_root, checkpoint)
    requested_device = str(device).lower()
    if requested_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device in {"cpu", "cuda"}:
        resolved_device = requested_device
    else:
        raise ValueError("Device must be one of: auto, cpu, cuda")
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    module = _load_external_module(provenance["implementation"])
    model = module.SuperPoint()
    load_kwargs = {"map_location": "cpu"}
    try:
        state_dict = torch.load(provenance["checkpoint"], weights_only=True, **load_kwargs)
    except TypeError:  # PyTorch versions before weights_only support
        state_dict = torch.load(provenance["checkpoint"], **load_kwargs)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(resolved_device)
    provenance.update(
        {
            "device": resolved_device,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "numpy_version": np.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if resolved_device == "cuda" else None,
            "model_configuration": {
                "nms_radius": int(model.conf.nms_radius),
                "max_num_keypoints": model.conf.max_num_keypoints,
                "detection_threshold": float(model.conf.detection_threshold),
                "remove_borders": int(model.conf.remove_borders),
                "descriptor_dim": int(model.conf.descriptor_dim),
                "stride": int(model.stride),
            },
        }
    )
    return model, provenance


def run_superpoint_on_slice(model, normalized_slice, device):
    """Run one already-normalized 2D slice and return CPU NumPy arrays."""

    import torch

    image = np.asarray(normalized_slice, dtype=np.float32)
    ensure_stride_compatible(image)
    tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model({"image": tensor})
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    keypoints = prediction["keypoints"][0].detach().cpu().numpy()
    scores = prediction["keypoint_scores"][0].detach().cpu().numpy()
    descriptors = prediction["descriptors"][0].detach().cpu().numpy()
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise RuntimeError("Unexpected keypoint shape: {}".format(keypoints.shape))
    if scores.shape != (keypoints.shape[0],):
        raise RuntimeError("Unexpected score shape: {}".format(scores.shape))
    if descriptors.shape[0] != keypoints.shape[0]:
        raise RuntimeError("Descriptor/keypoint count mismatch")
    if not all(np.isfinite(array).all() for array in (keypoints, scores, descriptors)):
        raise RuntimeError("SuperPoint output contains non-finite values")

    height, width = image.shape
    if keypoints.size:
        if np.any(keypoints[:, 0] < 0) or np.any(keypoints[:, 0] >= width):
            raise RuntimeError("SuperPoint x coordinate is outside the input image")
        if np.any(keypoints[:, 1] < 0) or np.any(keypoints[:, 1] >= height):
            raise RuntimeError("SuperPoint y coordinate is outside the input image")

    peak_gpu_bytes = None
    if device == "cuda":
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
    return {
        "keypoints_xy": keypoints,
        "scores": scores,
        "descriptors": descriptors,
        "runtime_seconds": float(elapsed),
        "peak_gpu_memory_bytes": peak_gpu_bytes,
    }


def write_json_atomic(path, payload):
    import json

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing smoke result: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            "Refusing to replace existing temporary result: {}".format(temporary)
        )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(destination))


def _prepare_atomic_destination(path, artifact_name):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing {}: {}".format(artifact_name, destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            "Refusing to replace existing temporary {}: {}".format(artifact_name, temporary)
        )
    return destination, temporary


def write_keypoints_csv_atomic(path, keypoints_xy, scores, slice_index):
    """Write model pixels and corresponding native NIfTI voxel coordinates."""

    import csv

    keypoints = np.asarray(keypoints_xy)
    score_values = np.asarray(scores)
    raw_voxels = model_keypoints_to_raw_voxels(keypoints, slice_index)
    if score_values.shape != (keypoints.shape[0],):
        raise ValueError("Score/keypoint count mismatch")
    destination, temporary = _prepare_atomic_destination(path, "keypoint CSV")
    fieldnames = [
        "point_id",
        "model_x_pixel",
        "model_y_pixel",
        "raw_x_voxel",
        "raw_y_voxel",
        "raw_z_voxel",
        "score",
        "coord_space",
        "source_plane",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point_id, (model_xy, raw_xyz, score) in enumerate(
            zip(keypoints, raw_voxels, score_values)
        ):
            writer.writerow(
                {
                    "point_id": point_id,
                    "model_x_pixel": float(model_xy[0]),
                    "model_y_pixel": float(model_xy[1]),
                    "raw_x_voxel": float(raw_xyz[0]),
                    "raw_y_voxel": float(raw_xyz[1]),
                    "raw_z_voxel": float(raw_xyz[2]),
                    "score": float(score),
                    "coord_space": "native_nifti_voxel_xyz",
                    "source_plane": "axial",
                }
            )
    os.replace(str(temporary), str(destination))
    return destination


def mask_boundary(mask_yx):
    """Return a one-pixel inner boundary using only four-neighbour adjacency."""

    mask = np.asarray(mask_yx, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask, got shape {}".format(mask.shape))
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _render_overlay_image(model_image_yx, keypoints_xy, mask_slices, title):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Pillow is required to create the review overlay") from exc

    image = np.asarray(model_image_yx)
    keypoints = np.asarray(keypoints_xy)
    ensure_stride_compatible(image)

    grayscale = np.asarray(np.clip(image, 0.0, 1.0) * 255.0, dtype=np.uint8)
    rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    palette = [
        (0, 198, 255),
        (255, 193, 7),
        (0, 230, 118),
        (179, 136, 255),
        (255, 145, 0),
        (0, 229, 255),
    ]
    for index, mask in enumerate(mask_slices):
        boundary = mask_boundary(mask["mask_yx"])
        rgb[boundary] = palette[index % len(palette)]

    header_lines = 1 + (len(mask_slices) + 1) // 2
    header_height = 16 + 16 * header_lines
    canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + header_height), "black")
    canvas.paste(Image.fromarray(rgb), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill="white")
    legend = [("SuperPoint", (255, 59, 48))] + [
        (mask["name"], palette[index % len(palette)])
        for index, mask in enumerate(mask_slices)
    ]
    for index, (label, colour) in enumerate(legend):
        column = index % 2
        row = index // 2
        x = 6 + column * (canvas.width // 2)
        y = 18 + row * 16
        draw.rectangle((x, y + 3, x + 8, y + 11), outline=colour, fill=colour)
        draw.text((x + 13, y), label, fill="white")
    if keypoints.size:
        for model_x, model_y in keypoints:
            x = float(model_x)
            y = float(model_y) + header_height
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(255, 59, 48), width=1)
    return canvas


def write_overlay_png_atomic(path, model_image_yx, keypoints_xy, mask_slices, title):
    """Render detected points and aligned mask contours in model-pixel space."""

    destination, temporary = _prepare_atomic_destination(path, "overlay PNG")
    canvas = _render_overlay_image(model_image_yx, keypoints_xy, mask_slices, title)
    canvas.save(str(temporary), format="PNG")
    os.replace(str(temporary), str(destination))
    return destination


def write_comparison_overlay_png_atomic(
    path,
    model_image_yx,
    all_keypoints_xy,
    accepted_keypoints_xy,
    mask_slices,
    title,
):
    """Write side-by-side all-candidate and inside-mask review panels."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Pillow is required to create the review overlay") from exc

    destination, temporary = _prepare_atomic_destination(path, "comparison overlay PNG")
    left = _render_overlay_image(
        model_image_yx,
        all_keypoints_xy,
        mask_slices,
        "{} | all candidates ({})".format(title, len(all_keypoints_xy)),
    )
    right = _render_overlay_image(
        model_image_yx,
        accepted_keypoints_xy,
        mask_slices,
        "{} | inside mask ({})".format(title, len(accepted_keypoints_xy)),
    )
    height = max(left.height, right.height)
    comparison = Image.new("RGB", (left.width + right.width, height), "black")
    comparison.paste(left, (0, 0))
    comparison.paste(right, (left.width, 0))
    comparison.save(str(temporary), format="PNG")
    os.replace(str(temporary), str(destination))
    return destination


def choose_max_area_slice(area_by_slice):
    """Choose the middle deterministic maximizer of a non-empty 1D area curve."""

    areas = np.asarray(area_by_slice)
    if areas.ndim != 1 or not areas.size:
        raise ValueError("Expected a non-empty 1D slice-area array")
    if not np.isfinite(areas).all() or np.any(areas < 0):
        raise ValueError("Slice areas must be finite and non-negative")
    maximum = float(areas.max())
    if maximum <= 0:
        raise ValueError("Organ group mask is empty")
    candidates = np.flatnonzero(areas == maximum)
    return int(candidates[len(candidates) // 2])


def candidate_mask_membership(mask_yx, keypoints_xy):
    """Return inside-mask membership for integer-valued model-pixel keypoints."""

    mask = np.asarray(mask_yx, dtype=bool)
    keypoints = np.asarray(keypoints_xy)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D group mask")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("Expected keypoints with shape [N, 2]")
    indices = np.rint(keypoints).astype(np.int64)
    if indices.size:
        if np.any(indices[:, 0] < 0) or np.any(indices[:, 0] >= mask.shape[1]):
            raise ValueError("Keypoint x is outside group-mask bounds")
        if np.any(indices[:, 1] < 0) or np.any(indices[:, 1] >= mask.shape[0]):
            raise ValueError("Keypoint y is outside group-mask bounds")
    return mask[indices[:, 1], indices[:, 0]]


def inside_point_boundary_distances_mm(mask_yx, keypoints_xy, spacing_xy_mm):
    """Calculate exact in-plane distances from accepted points to mask boundary."""

    mask = np.asarray(mask_yx, dtype=bool)
    keypoints = np.asarray(keypoints_xy, dtype=np.float32)
    spacing = np.asarray(spacing_xy_mm, dtype=np.float32)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Expected two positive finite in-plane spacings")
    membership = candidate_mask_membership(mask, keypoints)
    accepted = keypoints[membership]
    if not accepted.size:
        return np.empty((0,), dtype=np.float32)
    boundary_yx = np.argwhere(mask_boundary(mask))
    if not boundary_yx.size:
        raise ValueError("Non-empty mask has no boundary pixels")
    boundary_xy = boundary_yx[:, [1, 0]].astype(np.float32)
    distances = []
    for point_xy in accepted:
        delta_mm = (boundary_xy - point_xy[None, :]) * spacing[None, :]
        distances.append(float(np.sqrt(np.sum(delta_mm * delta_mm, axis=1)).min()))
    return np.asarray(distances, dtype=np.float32)
