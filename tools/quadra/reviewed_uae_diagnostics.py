"""Compact UAE diagnostics and delayed map export while caches remain open."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def spatial_peak_candidates(source, target, point_fine, retriever, domain=None,
                            minimum_separation_mm=8., count=3):
    """Exact greedy separated maxima; every alternate requires an exhaustive pass.

    These are separated candidates, not independent anatomical explanations or
    guarantees of separate local modes. Spacing is pilot-calibrated, provisional.
    """
    if minimum_separation_mm <= 0 or count < 1:
        raise ValueError('Invalid peak spacing/count')
    peaks = []
    for rank in range(count):
        best = -np.inf; candidate = None
        for xyz, fused, heads, backend, estimated in retriever.score_blocks(source, target, [point_fine], domain):
            allowed = np.isfinite(fused[0])
            physical = target.lps(target.fine_to_native(xyz, quantized=False))
            for peak in peaks:
                allowed &= np.linalg.norm(physical-np.asarray(peak['physical_lps_xyz']), axis=1) >= minimum_separation_mm
            scores = np.where(allowed, fused[0], -np.inf)
            index = int(scores.argmax())
            if scores[index] > best:
                best = float(scores[index])
                candidate = dict(fine_xyz=xyz[index].tolist(), physical_lps_xyz=physical[index].tolist(),
                    fused_score=best, fine_score=float(heads[0][0,index]), coarse_score=float(heads[1][0,index]),
                    semantic_score=float(heads[2][0,index]), rank=rank+1,
                    minimum_separation_mm=minimum_separation_mm, search='exact_greedy_exhaustive')
        if candidate is None: break
        peaks.append(candidate)
    return peaks


def pack_anchor_traces(directory, traces):
    """Flat numeric histories with per-query/direction and per-iteration offsets."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'anchor_traces.npz'
    if path.exists(): raise ValueError('Refusing to replace anchor evidence')
    offsets = [0]; query_offsets = [0]; queries = []; directions = []; iterations = []
    query_fine = []; matched_fine = []; matched_native = []; scores = []; filters = []; fit_metadata = []
    for query_id, direction, trace in traces:
        queries.append(query_id); directions.append(direction)
        for step in trace['anchor_history']:
            query_fine.extend(step['query_fine_xyz']); matched_fine.extend(step['matched_fine_xyz'])
            matched_native.extend(step['matched_native_xyz']); scores.extend(step['scores'])
            iterations.append(step['iteration']); offsets.append(len(scores))
        query_offsets.append(len(iterations))
        filters.append(trace['filter_keep'])
        fit_metadata.append(dict(query_id=query_id, direction=direction, filter_keep=trace['filter_keep'],
            fit=trace.get('fit'), status=trace['status'], failure_reason=trace.get('failure_reason'),
            stable_returned_native_xyz=trace.get('stable_returned_native_xyz'),
            stable_matched_native_xyz=trace.get('stable_matched_native_xyz')))
    np.savez_compressed(path, query_ids=np.array(queries, dtype='U'), directions=np.array(directions, dtype='U'),
        query_offsets=np.asarray(query_offsets, dtype=np.int64), iteration_offsets=np.asarray(offsets, dtype=np.int64),
        iterations=np.asarray(iterations, dtype=np.int16),
        query_fine_xyz=np.asarray(query_fine, dtype=np.int32).reshape(-1,3),
        matched_fine_xyz=np.asarray(matched_fine, dtype=np.int32).reshape(-1,3),
        matched_native_xyz=np.asarray(matched_native, dtype=np.int32).reshape(-1,3), scores=np.asarray(scores, dtype=np.float32))
    (directory/'anchor_fit_metadata.json').write_text(json.dumps(fit_metadata, sort_keys=True, indent=2))
    return path


def decode_anchor_traces(path):
    path = Path(path)
    metadata = json.loads((path.parent/'anchor_fit_metadata.json').read_text())
    result = []
    with np.load(path, allow_pickle=False) as arrays:
        for index, query_id in enumerate(arrays['query_ids']):
            history = []
            for step in range(int(arrays['query_offsets'][index]), int(arrays['query_offsets'][index+1])):
                a,b = arrays['iteration_offsets'][step:step+2]
                iteration = int(arrays['iterations'][step])
                history.append(dict(iteration=iteration, direction='forward' if iteration%2 else 'reverse',
                    query_fine_xyz=arrays['query_fine_xyz'][a:b].tolist(), matched_fine_xyz=arrays['matched_fine_xyz'][a:b].tolist(),
                    matched_native_xyz=arrays['matched_native_xyz'][a:b].tolist(),
                    scores=[float(v) if np.isfinite(v) else None for v in arrays['scores'][a:b]]))
            result.append(dict(metadata[index], query_id=str(query_id), anchor_history=history))
    return result


def export_similarity_views(directory, query_id, source, target, point_fine, retriever,
                            map_role, domain=None, ct_image=None, center_fine=None, color_scale=(-1.,1.)):
    """Save numeric orthogonal slices and optional physically sampled CT overlays.

    Three planes are harvested during exhaustive scoring; no full similarity
    volume is retained. Fixed-point corrected coordinates do not define its seed.
    """
    if map_role not in ('seed', 'anchor', 'reverse_query'):
        raise ValueError('Unknown map interpretation')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    center = np.asarray(center_fine if center_fine is not None else np.asarray(target.shape_xyz)//2, dtype=np.int64)
    if np.any(center < 0) or np.any(center >= target.shape_xyz): raise ValueError('Similarity view center outside fine grid')
    x,y,z = map(int,target.shape_xyz)
    slices = dict(axial=np.full((y,x), np.nan, dtype=np.float32), coronal=np.full((z,x), np.nan, dtype=np.float32),
                  sagittal=np.full((z,y), np.nan, dtype=np.float32))
    for xyz, fused, heads, backend, estimate in retriever.score_blocks(source, target, [point_fine], domain):
        for name, axis, indices in [('axial',2,(1,0)), ('coronal',1,(2,0)), ('sagittal',0,(2,1))]:
            mask = xyz[:,axis] == center[axis]
            slices[name][xyz[mask,indices[0]],xyz[mask,indices[1]]] = fused[0,mask]
    prefix = query_id.replace(':','_')+'-'+map_role
    path = directory/(prefix+'.npz')
    if path.exists(): raise ValueError('Refusing to replace similarity evidence')
    ct_slices = {}
    if ct_image is not None:
        from scipy.ndimage import map_coordinates
        affine_ras = np.asarray(ct_image.affine)
        lps_to_ct = np.linalg.inv(np.diag([-1.,-1.,1.,1.]) @ affine_ras)
        data = np.asanyarray(ct_image.dataobj)
        for name, axis, indices in [('axial',2,(1,0)), ('coronal',1,(2,0)), ('sagittal',0,(2,1))]:
            grid = np.indices(slices[name].shape)
            xyz = np.empty((grid[0].size,3)); xyz[:,axis] = center[axis]
            xyz[:,indices[0]] = grid[0].ravel(); xyz[:,indices[1]] = grid[1].ravel()
            lps = target.lps(target.fine_to_native(xyz, quantized=False))
            native = lps @ lps_to_ct[:3,:3].T + lps_to_ct[:3,3]
            ct_slices['ct_'+name] = map_coordinates(data, native.T, order=1, mode='constant', cval=-1024).reshape(slices[name].shape).astype(np.float32)
    np.savez_compressed(path, **dict(slices, **ct_slices))
    metadata = dict(query_id=query_id, map_role=map_role, query_fine_xyz=list(map(int,point_fine)),
                    plane_center_fine_xyz=center.tolist(), color_scale=list(color_scale),
                    target_native_to_lps=target.native_to_lps.tolist(), target_norm_ratio_xyz=target.norm_ratio_xyz.tolist(),
                    similarity_formula='mean(fine,coarse,semantic)', backend=backend,
                    after_cycle_selection=True, ct_overlay_available=bool(ct_slices))
    (directory/(prefix+'.json')).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    if ct_slices:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1,3,figsize=(12,4))
        for axis, name in zip(axes,('axial','coronal','sagittal')):
            axis.imshow(ct_slices['ct_'+name], cmap='gray', vmin=-160, vmax=240, origin='lower')
            axis.imshow(slices[name], cmap='viridis', vmin=color_scale[0], vmax=color_scale[1], alpha=.55, origin='lower')
            axis.set_title(name); axis.set_axis_off()
        figure.suptitle(query_id+' | '+map_role)
        figure.tight_layout(); figure.savefig(directory/(prefix+'.png'), dpi=120); plt.close(figure)
    return path
