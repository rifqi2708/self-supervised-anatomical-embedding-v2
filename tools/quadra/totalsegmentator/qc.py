"""Technical NIfTI QC and provisional spinal-cord derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .core import WorkflowError

AFFINE_ATOL = 1e-5


def _binary_data(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    try:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise WorkflowError(f"Unreadable NIfTI mask {path}: {exc}") from exc
    if data.ndim != 3:
        raise WorkflowError(f"Mask must be three-dimensional: {path} has shape {data.shape}")
    values = np.unique(data)
    if not np.all(np.isin(values, (0, 1))):
        raise WorkflowError(f"Mask must be binary: {path} contains {values[:10].tolist()}")
    if not np.any(data):
        raise WorkflowError(f"Mask is empty: {path}")
    return image, data.astype(bool, copy=False)


def validate_mask(mask_path: Path, reference_path: Path) -> dict[str, Any]:
    reference = nib.load(str(reference_path))
    mask, data = _binary_data(mask_path)
    if mask.shape != reference.shape:
        raise WorkflowError(
            f"Mask shape mismatch for {mask_path}: {mask.shape} != {reference.shape}"
        )
    if not np.allclose(mask.affine, reference.affine, atol=AFFINE_ATOL, rtol=0):
        raise WorkflowError(f"Mask affine mismatch: {mask_path}")
    return {
        "path": str(mask_path.resolve()),
        "shape": list(mask.shape),
        "voxel_count": int(data.sum()),
        "voxel_volume_mm3": float(abs(np.linalg.det(mask.affine[:3, :3]))),
        "volume_mm3": float(data.sum() * abs(np.linalg.det(mask.affine[:3, :3]))),
    }


def _world_superior_coordinates(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return (
        indices[:, 0] * affine[2, 0]
        + indices[:, 1] * affine[2, 1]
        + indices[:, 2] * affine[2, 2]
        + affine[2, 3]
    )


def _mask_superior_extent(path: Path, reference: nib.Nifti1Image) -> tuple[float, float, float]:
    image, data = _binary_data(path)
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=AFFINE_ATOL, rtol=0
    ):
        raise WorkflowError(f"Landmark geometry mismatch: {path}")
    coordinates = _world_superior_coordinates(np.argwhere(data), reference.affine)
    return float(coordinates.min()), float(coordinates.mean()), float(coordinates.max())


def derive_spinal_cord_segments(
    spinal_cord_path: Path,
    c1_path: Path,
    c7_path: Path,
    t1_path: Path,
    t12_path: Path,
    l1_path: Path,
    cervical_output: Path,
    thoracic_output: Path,
) -> dict[str, Any]:
    """Clip one spinal-cord mask using physical vertebral landmarks.

    World Z is used as the physical superior-inferior coordinate. The
    cervical/thoracic boundary is halfway between C7 and T1 centroids; the
    thoracic/upper-lumbar boundary is halfway between T12 and L1 centroids.
    """

    cord_image, cord = _binary_data(spinal_cord_path)
    c1_extent = _mask_superior_extent(c1_path, cord_image)
    c7_extent = _mask_superior_extent(c7_path, cord_image)
    t1_extent = _mask_superior_extent(t1_path, cord_image)
    t12_extent = _mask_superior_extent(t12_path, cord_image)
    l1_extent = _mask_superior_extent(l1_path, cord_image)

    cervical_thoracic = (c7_extent[1] + t1_extent[1]) / 2.0
    thoracic_lumbar = (t12_extent[1] + l1_extent[1]) / 2.0
    cervical_superior = c1_extent[2]
    if not cervical_superior > cervical_thoracic > thoracic_lumbar:
        raise WorkflowError(
            "Vertebral landmarks are not ordered superior-to-inferior as C1, C7, T1, T12, L1"
        )

    grid = np.indices(cord.shape, dtype=np.float64).reshape(3, -1).T
    superior = _world_superior_coordinates(grid, cord_image.affine).reshape(cord.shape)
    cervical = cord & (superior >= cervical_thoracic) & (superior <= cervical_superior)
    thoracic = cord & (superior >= thoracic_lumbar) & (superior < cervical_thoracic)
    if not cervical.any() or not thoracic.any():
        raise WorkflowError("Derived cervical or thoracic spinal-cord mask is empty")
    if np.any(cervical & thoracic):
        raise WorkflowError("Derived spinal-cord segments overlap")
    if np.any(cervical & ~cord) or np.any(thoracic & ~cord):
        raise WorkflowError("Derived spinal-cord segment is not a subset of source spinal cord")

    cervical_output.parent.mkdir(parents=True, exist_ok=True)
    thoracic_output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(cervical.astype(np.uint8), cord_image.affine, cord_image.header),
        str(cervical_output),
    )
    nib.save(
        nib.Nifti1Image(thoracic.astype(np.uint8), cord_image.affine, cord_image.header),
        str(thoracic_output),
    )
    return {
        "method": "physical_world_z_midpoints",
        "provisional": True,
        "boundaries_mm": {
            "cervical_superior": cervical_superior,
            "cervical_thoracic": cervical_thoracic,
            "thoracic_lumbar": thoracic_lumbar,
        },
        "landmark_centroids_mm": {
            "C1": c1_extent[1],
            "C7": c7_extent[1],
            "T1": t1_extent[1],
            "T12": t12_extent[1],
            "L1": l1_extent[1],
        },
        "cervical_voxels": int(cervical.sum()),
        "thoracic_voxels": int(thoracic.sum()),
    }


def validate_scan_outputs(
    scan_directory: Path,
    input_path: Path,
    expected_masks: list[str],
    sex: str,
) -> dict[str, Any]:
    mask_directory = scan_directory / "masks"
    results: dict[str, Any] = {}
    for name in expected_masks:
        results[name] = validate_mask(mask_directory / f"{name}.nii.gz", input_path)
    prostate_path = mask_directory / "prostate.nii.gz"
    if sex == "F" and prostate_path.exists():
        raise WorkflowError(f"Unexpected female prostate mask: {prostate_path}")

    cervical_path = mask_directory / "spinal_cord_cervical.nii.gz"
    thoracic_path = mask_directory / "spinal_cord_thoracic.nii.gz"
    source_path = scan_directory / "intermediate" / "spinal_cord.nii.gz"
    if cervical_path.exists() and thoracic_path.exists() and source_path.exists():
        _, cervical = _binary_data(cervical_path)
        _, thoracic = _binary_data(thoracic_path)
        _, source = _binary_data(source_path)
        if np.any(cervical & thoracic):
            raise WorkflowError("Cervical and thoracic spinal-cord masks overlap")
        if np.any(cervical & ~source) or np.any(thoracic & ~source):
            raise WorkflowError("Derived spinal-cord masks are not subsets of the source")
    return {
        "status": "valid",
        "checked_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "mask_count": len(results),
        "masks": results,
        "anatomical_accuracy_assessed": False,
    }
