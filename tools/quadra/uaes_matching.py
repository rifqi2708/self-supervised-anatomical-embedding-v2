"""UAE-S streaming matching, including batched fixed-point structural inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tools.quadra.streaming_cycle_error import EmbeddingCache, stream_global_match_uaes


@dataclass(frozen=True)
class FixedPointSettings:
    margin_xyz: tuple[int, int, int] = (2, 2, 2)
    iterations: int = 4
    score_threshold: float = 0.8
    max_return_distance_mm: float = 100.0

    def to_dict(self) -> dict[str, object]:
        return {
            "margin_xyz": list(self.margin_xyz),
            "iterations": int(self.iterations),
            "score_threshold": float(self.score_threshold),
            "max_return_distance_mm": float(self.max_return_distance_mm),
        }


def native_to_fine(cache: EmbeddingCache, points_xyz) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    fine_shape = np.asarray(cache.feature_shape_xyz("fine"), dtype=np.int64)
    converted = np.floor((points * cache.norm_ratio_xyz) / 2.0).astype(np.int64)
    return np.clip(converted, 0, fine_shape - 1)


def fine_to_native(cache: EmbeddingCache, points_xyz) -> np.ndarray:
    """Reproduce the UAE demo's fine-grid-to-native coordinate conversion."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    converted = np.round((points * 2.0 + 0.5) / cache.norm_ratio_xyz).astype(np.int64)
    native_max = np.asarray(cache.native_shape_xyz, dtype=np.int64) - 1
    return np.clip(converted, 0, native_max)


def local_anchor_grid(center_fine_xyz, fine_shape_xyz, margin_xyz=(2, 2, 2)) -> np.ndarray:
    center = np.asarray(center_fine_xyz, dtype=np.int64)
    shape = np.asarray(fine_shape_xyz, dtype=np.int64)
    margin = np.asarray(margin_xyz, dtype=np.int64)
    if np.any(margin < 0):
        raise ValueError("Fixed-point margins cannot be negative.")
    axes = [
        np.arange(center[axis] - margin[axis], center[axis] + margin[axis] + 1, dtype=np.int64)
        for axis in range(3)
    ]
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing="ij")
    points = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)
    points = np.clip(points, 0, shape - 1)
    return np.unique(points, axis=0)


def _deduplicated_global_match(
    query_cache: EmbeddingCache,
    target_cache: EmbeddingCache,
    point_groups: list[np.ndarray],
    query_batch_size: int,
    match_chunk_xyz: Sequence[int],
    device=None,
):
    lengths = [len(group) for group in point_groups]
    if not lengths or any(length == 0 for length in lengths):
        raise ValueError("Every fixed-point query must contain at least one anchor.")
    flattened = np.concatenate(point_groups, axis=0).astype(np.int64, copy=False)
    unique, inverse = np.unique(flattened, axis=0, return_inverse=True)
    matched_unique, scores_unique, profile = stream_global_match_uaes(
        query_cache,
        target_cache,
        unique,
        query_batch_size=query_batch_size,
        match_chunk_xyz=match_chunk_xyz,
        device=device,
        output_space="fine",
    )
    matched = matched_unique[inverse]
    scores = scores_unique[inverse]
    point_results = []
    score_results = []
    offset = 0
    for length in lengths:
        point_results.append(matched[offset : offset + length])
        score_results.append(scores[offset : offset + length])
        offset += length
    profile.update(
        {
            "requested_anchor_count": int(len(flattened)),
            "unique_anchor_count": int(len(unique)),
            "deduplicated_anchor_count": int(len(flattened) - len(unique)),
        }
    )
    return point_results, score_results, profile


def robust_affine_predict(source_xyz, target_xyz, query_xyz, target_shape_xyz):
    """Fit the official robust local affine model and predict one target point."""
    import statsmodels.api as sm

    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    query = np.asarray(query_xyz, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape:
        raise ValueError("Affine anchors must be paired N-by-3 arrays.")
    design_3d = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    rank_3d = int(np.linalg.matrix_rank(design_3d))
    mode = "3d"
    if len(source) >= 4 and rank_3d == 4:
        design = design_3d
        query_design = np.concatenate((query, [1.0]))
    else:
        design_2d = np.column_stack((source[:, 0], source[:, 1], np.ones(len(source))))
        rank_2d = int(np.linalg.matrix_rank(design_2d))
        if len(source) < 3 or rank_2d != 3 or len(np.unique(source[:, 2])) != 1:
            raise ValueError(
                f"degenerate_affine_geometry:n={len(source)},rank3d={rank_3d},rank2d={rank_2d}"
            )
        mode = "planar"
        design = design_2d
        query_design = np.array([query[0], query[1], 1.0])

    predictions = []
    for axis in range(3):
        fit = sm.RLM(target[:, axis], design).fit()
        predictions.append(float(np.dot(query_design, fit.params)))
    predicted = np.round(predictions).astype(np.int64)
    predicted = np.clip(predicted, 0, np.asarray(target_shape_xyz, dtype=np.int64) - 1)
    return predicted, {"affine_mode": mode, "affine_rank": rank_3d, "anchor_count": int(len(source))}


def fixed_point_match_batch(
    query_cache: EmbeddingCache,
    target_cache: EmbeddingCache,
    query_points_native_xyz,
    settings: FixedPointSettings,
    query_batch_size: int,
    match_chunk_xyz: Sequence[int],
    device=None,
):
    """Match native-space points using batched unrestricted fixed-point inference."""
    started = time.time()
    original = np.asarray(query_points_native_xyz, dtype=np.int64).reshape(-1, 3)
    centers_fine = native_to_fine(query_cache, original)
    current_native = [
        fine_to_native(
            query_cache,
            local_anchor_grid(center, query_cache.feature_shape_xyz("fine"), settings.margin_xyz),
        )
        for center in centers_fine
    ]
    final_target_native = None
    final_return_native = None
    final_return_scores = None
    iteration_profiles = []

    for iteration in range(settings.iterations):
        forward = iteration % 2 == 0
        source_cache = query_cache if forward else target_cache
        destination_cache = target_cache if forward else query_cache
        current_fine = [native_to_fine(source_cache, group) for group in current_native]
        matched_fine, scores, profile = _deduplicated_global_match(
            source_cache,
            destination_cache,
            current_fine,
            query_batch_size=query_batch_size,
            match_chunk_xyz=match_chunk_xyz,
            device=device,
        )
        current_native = [fine_to_native(destination_cache, group) for group in matched_fine]
        profile["iteration"] = iteration + 1
        profile["direction"] = "query_to_target" if forward else "target_to_query"
        iteration_profiles.append(profile)
        if forward:
            final_target_native = [group.copy() for group in current_native]
        else:
            final_return_native = [group.copy() for group in current_native]
            final_return_scores = [group.copy() for group in scores]

    if final_target_native is None or final_return_native is None or final_return_scores is None:
        raise RuntimeError("Fixed-point iterations did not produce both forward and reverse states.")

    native_spacing = np.asarray(query_cache.manifest["native_spacing_xyz"], dtype=np.float64)
    results = []
    for index, query in enumerate(original):
        returned = final_return_native[index]
        target = final_target_native[index]
        scores = final_return_scores[index]
        return_mm = np.linalg.norm((returned.astype(np.float64) - query) * native_spacing, axis=1)
        keep = np.logical_and(
            return_mm < settings.max_return_distance_mm,
            scores > settings.score_threshold,
        )
        stable_source = returned[keep]
        stable_target = target[keep]
        stable_scores = scores[keep]
        if len(stable_target):
            _, unique_indices = np.unique(stable_target, axis=0, return_index=True)
            unique_indices = np.sort(unique_indices)
            stable_source = stable_source[unique_indices]
            stable_target = stable_target[unique_indices]
            stable_scores = stable_scores[unique_indices]
        base = {
            "query_index": index,
            "status": "failed",
            "failure_reason": None,
            "point_xyz": None,
            "score": float(np.mean(stable_scores)) if len(stable_scores) else None,
            "stable_anchor_count": int(len(stable_target)),
            "candidate_anchor_count": int(len(returned)),
        }
        try:
            if len(stable_target) < 3:
                raise ValueError(f"insufficient_stable_anchors:n={len(stable_target)}")
            point, fit_profile = robust_affine_predict(
                stable_source, stable_target, query, target_cache.native_shape_xyz
            )
            base.update({"status": "success", "point_xyz": point, **fit_profile})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            base["failure_reason"] = str(exc)
        results.append(base)

    profile = {
        "seconds": float(time.time() - started),
        "query_count": int(len(original)),
        "success_count": int(sum(result["status"] == "success" for result in results)),
        "failure_count": int(sum(result["status"] != "success" for result in results)),
        "settings": settings.to_dict(),
        "iterations": iteration_profiles,
    }
    return results, profile
