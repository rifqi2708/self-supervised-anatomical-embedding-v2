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
    if not np.isfinite(minimum_separation_mm) or minimum_separation_mm <= 0 or count < 1:
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


def similarity_view_reserve_bytes(target, ct_available=True):
    """Conservative per-view write reservation, not a measured GPU memory budget."""
    x,y,z=map(int,target.shape_xyz)
    numeric=4*(x*y+x*z+y*z)*3*(2 if ct_available else 1)
    # Two 12x8 inch rows at 120dpi; reserve RGBA canvas plus metadata.
    return numeric+12*120*8*120*4+65536


def export_similarity_views(directory, query_id, source, target, point_fine, retriever,
                            map_role, domain=None, ct_image=None, center_fine=None, color_scale=(-1.,1.),
                            resource_guard=None):
    """Retain corrected context AND the global finite similarity maximum.

    The first exhaustive pass finds the stable global maximum and keeps context
    planes. If centres differ, a second exhaustive pass keeps peak planes. No 3D
    similarity volume is stored. Extra scoring overhead requires pilot timing.
    """
    if map_role not in ('seed', 'anchor', 'reverse_query'): raise ValueError('Unknown map interpretation')
    point=np.asarray(point_fine,dtype=np.int64)
    if point.shape!=(3,) or np.any(point<0) or np.any(point>=source.shape_xyz):
        raise ValueError('similarity_query_outside_fine_grid')
    query_heads=source.descriptors([point])
    if not all(np.isfinite(v).all() for v in query_heads) or not any(np.linalg.norm(v)>1e-12 for v in query_heads):
        raise ValueError('similarity_invalid_or_empty_query_descriptor')
    box=retriever.domain(target,domain)
    center = np.asarray(center_fine if center_fine is not None else np.asarray(target.shape_xyz)//2, dtype=np.int64)
    if center.shape!=(3,) or np.any(center<0) or np.any(center>=target.shape_xyz):
        raise ValueError('Similarity view center outside fine grid')
    directory=Path(directory);prefix=query_id.replace(':','_')+'-'+map_role
    path=directory/(prefix+'.npz')
    if any((directory/(prefix+extension)).exists() for extension in ('.npz','.json','.png')):
        raise ValueError('Refusing to replace similarity evidence')
    reserve=similarity_view_reserve_bytes(target,ct_image is not None)
    if resource_guard: resource_guard(reserve)
    directory.mkdir(parents=True,exist_ok=True)
    x,y,z=map(int,target.shape_xyz)
    planes=(('axial',2,(1,0)),('coronal',1,(2,0)),('sagittal',0,(2,1)))
    def empty_planes():
        return dict(axial=np.full((y,x),np.nan,np.float32),coronal=np.full((z,x),np.nan,np.float32),
                    sagittal=np.full((z,y),np.nan,np.float32))
    def capture(slices,at,xyz,scores):
        for name,axis,indices in planes:
            mask=xyz[:,axis]==at[axis]
            slices[name][xyz[mask,indices[0]],xyz[mask,indices[1]]]=scores[mask]
    context=empty_planes();best=-np.inf;winner=None;best_linear=None;searched=0;nonfinite=0
    backend=None;estimated=0
    for xyz,fused,heads,backend,estimate in retriever.score_blocks(source,target,[point],domain):
        estimated=max(estimated,int(estimate));scores=fused[0]
        capture(context,center,xyz,scores);searched+=len(xyz);nonfinite+=int((~np.isfinite(scores)).sum())
        safe=np.where(np.isfinite(scores),scores,-np.inf);index=int(safe.argmax())
        value=float(safe[index]);location=xyz[index];linear=int((location[2]*y+location[1])*x+location[0])
        if np.isfinite(value) and (value>best or (value==best and (best_linear is None or linear<best_linear))):
            best=value;winner=location.copy();best_linear=linear
    if winner is None: raise ValueError('similarity_no_finite_target_score')
    passes=1;peak=context
    if not np.array_equal(winner,center):
        if resource_guard: resource_guard(reserve)
        peak=empty_planes();passes=2
        for xyz,fused,heads,other_backend,estimate in retriever.score_blocks(source,target,[point],domain):
            if other_backend!=backend: raise ValueError('Similarity backend changed between passes')
            capture(peak,winner,xyz,fused[0])
    saved=dict(context)
    for role,slices in (('context',context),('peak',peak)):
        saved.update({role+'_'+name:values for name,values in slices.items()})
    ct_slices={}
    if ct_image is not None:
        from scipy.ndimage import map_coordinates
        lps_to_ct=np.linalg.inv(np.diag([-1.,-1.,1.,1.])@np.asarray(ct_image.affine))
        data=np.asarray(ct_image.dataobj,dtype=np.float32)
        for role,slices,at in (('context',context,center),('peak',peak,winner)):
            for name,axis,indices in planes:
                grid=np.indices(slices[name].shape)
                xyz=np.empty((grid[0].size,3));xyz[:,axis]=at[axis]
                xyz[:,indices[0]]=grid[0].ravel();xyz[:,indices[1]]=grid[1].ravel()
                lps=target.lps(target.fine_to_native(xyz,quantized=False))
                native=lps@lps_to_ct[:3,:3].T+lps_to_ct[:3,3]
                ct_slices['ct_'+role+'_'+name]=map_coordinates(data,native.T,order=1,mode='constant',cval=-1024).reshape(slices[name].shape).astype(np.float32)
        ct_slices.update({'ct_'+name:ct_slices['ct_context_'+name] for name,_,_ in planes})
    saved.update(ct_slices)
    row_roles=['corrected_context','global_maximum'] if passes==2 else ['context_and_global_maximum']
    metadata=dict(schema_version=2,query_id=query_id,map_role=map_role,query_fine_xyz=point.tolist(),
        plane_center_fine_xyz=center.tolist(),context_center_fine_xyz=center.tolist(),
        context_center_interpretation='requested_corrected_or_matched_point_context' if center_fine is not None else 'default_grid_center',
        global_maximum_fine_xyz=winner.tolist(),global_maximum_lps_xyz=target.lps(target.fine_to_native(winner,quantized=False))[0].tolist(),
        global_maximum_score=best,global_maximum_tie_policy='first_global_zyx_flat_index',
        admissible_domain=box.tolist(),searched_target_locations=searched,nonfinite_target_scores=nonfinite,
        scoring_passes=passes,extra_scoring_overhead='second exhaustive pass when peak and context centres differ; pilot timing pending',
        color_scale=list(color_scale),target_native_to_lps=target.native_to_lps.tolist(),target_norm_ratio_xyz=target.norm_ratio_xyz.tolist(),
        similarity_formula='mean(fine,normalized_interpolated_coarse,semantic)',backend=backend,
        estimated_scoring_bytes=estimated,reserved_view_bytes=reserve,stored_numeric_array_bytes=sum(v.nbytes for v in saved.values()),
        numeric_key_policy='axial/coronal/sagittal retain context aliases; context_* and peak_* explicit',
        png_row_roles=row_roles,after_cycle_selection=True,ct_overlay_available=bool(ct_slices))
    if resource_guard: resource_guard(reserve)
    np.savez_compressed(path,**saved)
    (directory/(prefix+'.json')).write_text(json.dumps(metadata,indent=2,sort_keys=True))
    if ct_slices:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows=[('context',context,center,'Corrected/matched context')]
        if passes==2: rows.append(('peak',peak,winner,'Global finite maximum'))
        else: rows[0]=('context',context,center,'Context and global finite maximum')
        figure,axes=plt.subplots(len(rows),3,figsize=(12,4*len(rows)),squeeze=False)
        for axes_row,(role,slices,at,label) in zip(axes,rows):
            for axis,(name,_,_) in zip(axes_row,planes):
                axis.imshow(ct_slices['ct_'+role+'_'+name],cmap='gray',vmin=-160,vmax=240,origin='lower')
                axis.imshow(np.ma.masked_invalid(slices[name]),cmap='viridis',vmin=color_scale[0],vmax=color_scale[1],alpha=.55,origin='lower')
                axis.set_title(label+'\n'+name+' | fine xyz '+str(at.tolist()),fontsize=10);axis.set_axis_off()
        figure.suptitle(query_id+' | '+map_role+' | shared similarity scale '+str(list(color_scale)))
        figure.tight_layout(rect=(0,0,1,.95));figure.savefig(directory/(prefix+'.png'),dpi=120);plt.close(figure)
    return path
