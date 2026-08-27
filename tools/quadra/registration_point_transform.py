"""Independent continuous registration geometry and Transformix point helpers.

No imports from the historical registration implementation. Arrays use ZYX;
all point APIs use XYZ. Physical calculations use ITK LPS millimetres.
"""
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class RegistrationError(RuntimeError):
    """A non-retryable contract or integrity violation."""


class RuntimeFailure(RuntimeError):
    """An isolated registration/runtime failure."""


RAS_LPS = np.diag([-1., -1., 1., 1.])
OVERRIDES = {
    "rigid": {
        "NumberOfResolutions": "4", "MaximumNumberOfIterations": "256",
        "NumberOfSpatialSamples": "8192", "ImageSampler": "RandomCoordinate",
        "NewSamplesEveryIteration": "true", "RandomSeed": "121212",
        "AutomaticTransformInitialization": "true",
        "AutomaticTransformInitializationMethod": "GeometricalCenter",
        "WriteResultImage": "false", "ResultImageFormat": "nii.gz",
        "DefaultPixelValue": "-1024",
    },
    "bspline": {
        "NumberOfResolutions": "4", "MaximumNumberOfIterations": "256",
        "NumberOfSpatialSamples": "8192", "ImageSampler": "RandomCoordinate",
        "NewSamplesEveryIteration": "true", "RandomSeed": "121212",
        "FinalGridSpacingInPhysicalUnits": "32",
        "WriteResultImage": "false", "ResultImageFormat": "nii.gz",
        "DefaultPixelValue": "-1024",
    },
}


def require(condition, message):
    if not condition:
        raise RegistrationError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve()
    require(path.is_file(), "Missing file: {}".format(path))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def verify_identity(record):
    observed = identity(record["path"])
    require(all(observed[k] == record[k] for k in ("path", "bytes", "sha256")),
            "File identity changed: {}".format(record["path"]))
    return Path(record["path"])


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_bytes(path, value, refuse=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if refuse:
            # Atomic no-replace publication, including concurrent writers.
            os.link(temp, path)
            os.unlink(temp)
        else:
            os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_text(path, text, refuse=False):
    atomic_bytes(path, text.encode("utf-8"), refuse)


def atomic_json(path, value, refuse=False):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", refuse)


def atomic_csv(path, rows, fields=None, refuse=False):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue(), refuse)


def apply_affine(points, affine):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    affine = np.asarray(affine, dtype=np.float64)
    require(affine.shape == (4, 4) and np.isfinite(affine).all(), "Invalid affine")
    return points.dot(affine[:3, :3].T) + affine[:3, 3]


def lps_affine(source):
    return RAS_LPS.dot(np.asarray(source["affine"], dtype=np.float64))


def inside(points, shape):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    # Conservative voxel-centre support; no hidden clipping at the boundary.
    return np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1) & (
        points <= np.asarray(shape) - 1).all(axis=1)


def geometry_checks(source):
    shape = np.asarray(source["native_shape_xyz"], dtype=int)
    require(shape.shape == (3,) and (shape > 0).all(), "Invalid native shape")
    a = lps_affine(source)
    points = np.array([[x, y, z] for x in (0, shape[0]-1)
                       for y in (0, shape[1]-1) for z in (0, shape[2]-1)], dtype=float)
    points = np.vstack([points, (shape - 1) / 2, (shape - 1) * .371])
    physical = apply_affine(points, a)
    back = apply_affine(physical, np.linalg.inv(a))
    voxel_error = float(np.max(np.abs(back - points)))
    mm_error = float(np.max(np.linalg.norm(apply_affine(back, a) - physical, axis=1)))
    require(voxel_error <= 1e-6 and mm_error <= 1e-5, "Coordinate round-trip failed")
    return {"max_roundtrip_voxels": voxel_error, "max_roundtrip_mm": mm_error}


def check_itk_geometry(image, source):
    import itk
    shape = list(itk.size(image))
    a = np.eye(4)
    a[:3, :3] = np.asarray(image.GetDirection()) * np.asarray(image.GetSpacing())
    a[:3, 3] = image.GetOrigin()
    require(shape == source["native_shape_xyz"], "ITK native shape mismatch")
    require(np.allclose(a, lps_affine(source), atol=1e-5, rtol=0), "ITK LPS/NIfTI RAS mismatch")
    return geometry_checks(source)


def parameter_maps():
    import itk
    po = itk.ParameterObject.New()
    maps = []
    for name in ("rigid", "bspline"):
        item = {k: list(v) for k, v in po.GetDefaultParameterMap(name).items()}
        item.update({k: [v] for k, v in OVERRIDES[name].items()})
        # Keep input direction cosines and suppress the unnecessary result volume.
        item["UseDirectionCosines"] = ["true"]
        maps.append(item)
    return maps


def parameter_object(maps):
    import itk
    obj = itk.ParameterObject.New()
    for item in maps:
        obj.AddParameterMap(item)
    return obj


def normalized_transform_maps(obj):
    maps = []
    for index in range(obj.GetNumberOfParameterMaps()):
        item = {k: list(v) for k, v in obj.GetParameterMap(index).items()}
        values = np.asarray(item.get("TransformParameters", []), dtype=float)
        require(len(values) > 0 and np.isfinite(values).all(), "Invalid transform coefficients")
        item.pop("InitialTransformParametersFileName", None)
        item["InitialTransformParameterFileName"] = [
            "NoInitialTransform" if index == 0 else "TransformParameters.{}.txt".format(index-1)]
        item["WriteResultImage"] = ["false"]
        require(item.get("HowToCombineTransforms") == ["Compose"], "Transform composition changed")
        maps.append(item)
    require(len(maps) == 2, "Expected complete rigid/B-spline chain")
    require(maps[0]["Transform"] == ["EulerTransform"] and maps[1]["Transform"] == ["BSplineTransform"],
            "Unexpected transform family/order")
    return maps


def save_transform_chain(directory, maps):
    import itk
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(maps):
        path = directory / "TransformParameters.{}.txt".format(index)
        require(not path.exists(), "Transform already exists")
        itk.ParameterObject.WriteParameterFile(item, str(path))
        records.append(identity(path))
    return records


def transformix_points(points_lps, maps, threads=1):
    """Point-set evaluation only; no moving voxel buffer or deformation field.

    Rebuild the chain in a private directory so it is relocatable. Transformix
    OutputPoint (physical), never OutputIndexFixed/Moving (integer), is consumed.
    """
    import itk
    points = np.asarray(points_lps, dtype=np.float64).reshape(-1, 3)
    require(np.isfinite(points).all(), "Non-finite Transformix input")
    if not len(points):
        return points.copy()
    with tempfile.TemporaryDirectory(prefix="quadra-reg-points-") as name:
        directory = Path(name)
        portable = []
        for index, item in enumerate(maps):
            item = dict(item)
            item["InitialTransformParameterFileName"] = ["NoInitialTransform" if index == 0
                else str(directory / "TransformParameters.{}.txt".format(index-1))]
            portable.append(item)
        save_transform_chain(directory, portable)
        pointfile = directory / "points.txt"
        atomic_text(pointfile, "point\n{}\n".format(len(points)) + "".join(
            "{:.17g} {:.17g} {:.17g}\n".format(*p) for p in points), refuse=True)
        obj = itk.ParameterObject.New()
        obj.ReadParameterFile(str(directory / "TransformParameters.{}.txt".format(len(maps)-1)))
        filt = itk.TransformixFilter[itk.Image[itk.F, 3]].New()
        # ITK 5.4's pipeline requires the primary MovingImage input to exist.
        # An empty image satisfies that API requirement; Transformix detects its
        # zero-sized region and does not set an image container for resampling.
        empty_image = itk.Image[itk.F, 3].New()
        filt.SetMovingImage(empty_image)
        filt.SetTransformParameterObject(obj)
        filt.SetFixedPointSetFileName(str(pointfile))
        filt.SetOutputDirectory(str(directory))
        filt.SetComputeDeformationField(False)
        filt.SetLogToConsole(False)
        filt.SetNumberOfWorkUnits(threads)
        filt.UpdateLargestPossibleRegion()
        require(filt.GetOutput().GetBufferedRegion().GetNumberOfPixels() == 0,
                "Point-only Transformix unexpectedly generated a resampled image")
        require(filt.GetOutputDeformationField().GetBufferedRegion().GetNumberOfPixels() == 0,
                "Point-only Transformix unexpectedly generated a dense deformation field")
        return parse_output_points(directory / "outputpoints.txt", len(points))


def parse_output_points(path, expected_count):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        idx = re.search(r"Point\s+(\d+)\s*;", line)
        point = re.search(r"OutputPoint\s*=\s*\[([^]]+)\]", line)
        require(idx is not None and point is not None, "Malformed Transformix point output")
        require(int(idx.group(1)) == len(result), "Transformix reordered/missing point")
        values = [float(x) for x in point.group(1).split()]
        require(len(values) == 3, "Transformix output is not XYZ")
        result.append(values)
    require(len(result) == expected_count, "Transformix point-count mismatch")
    return np.asarray(result, dtype=np.float64).reshape(-1, 3)


def evaluate_cycle(rows, test, retest, forward, backward):
    query = np.asarray([[float(r["raw_"+a]) for a in "xyz"] for r in rows])
    require(inside(query, test["native_shape_xyz"]).all(), "Query outside Test CT")
    qlps = apply_affine(query, lps_affine(test))
    matched = np.asarray(forward(qlps), dtype=float)
    require(matched.shape == qlps.shape, "Forward point-count mismatch")
    matched_raw = apply_affine(matched, np.linalg.inv(lps_affine(retest)))
    forward_valid = inside(matched_raw, retest["native_shape_xyz"])
    returned = np.full_like(qlps, np.nan)
    if forward_valid.any():
        result = np.asarray(backward(matched[forward_valid]), dtype=float)
        require(result.shape == matched[forward_valid].shape, "Backward point-count mismatch")
        returned[forward_valid] = result
    returned_raw = apply_affine(returned, np.linalg.inv(lps_affine(test)))
    return_valid = inside(returned_raw, test["native_shape_xyz"])
    output = []
    for i, original in enumerate(rows):
        valid = bool(forward_valid[i] and return_valid[i])
        reason = ""
        if not np.isfinite(matched[i]).all():
            reason = "nonfinite_forward"
        elif not forward_valid[i]:
            reason = "forward_outside_retest_fov"
        elif not np.isfinite(returned[i]).all():
            reason = "nonfinite_backward"
        elif not return_valid[i]:
            reason = "returned_outside_test_fov"
        record = dict(original, valid_cycle=valid, failure_reason=reason,
                      cycle_error_mm=float(np.linalg.norm(returned[i]-qlps[i])) if valid else "")
        for prefix, value in (("query_raw", query[i]), ("matched_raw", matched_raw[i]),
                              ("returned_raw", returned_raw[i]),
                              ("query_physical", qlps[i]*[-1,-1,1]),
                              ("matched_physical", matched[i]*[-1,-1,1]),
                              ("returned_physical", returned[i]*[-1,-1,1])):
            for axis, component in zip("xyz", value):
                record[prefix+"_"+axis] = float(component) if np.isfinite(component) else ""
        for prefix, value in (("matched_raw_rounded", matched_raw[i]), ("returned_raw_rounded", returned_raw[i])):
            for axis, component in zip("xyz", value):
                record[prefix+"_"+axis] = int(np.rint(component)) if np.isfinite(component) else ""
        output.append(record)
    return output
