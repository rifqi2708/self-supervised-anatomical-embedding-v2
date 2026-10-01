"""Fine-grid UAE inference for the immutable reviewed-mask contract.

CPU fixtures and CUDA scoring share the declared FP32 operation. Extraction is
deliberately outside this module; retrieval equivalence never validates context.
"""
from __future__ import annotations

import itertools
import ast
import hashlib
import json
import shutil
import tempfile
import time
import resource
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class FineGridCache:
    """Adapt existing EmbeddingCache without resampling native similarity maps."""
    def __init__(self, cache, native_to_lps=None):
        self.cache = cache
        self.shape_xyz = np.asarray(cache.feature_shape_xyz('fine'), dtype=np.int64)
        self.native_shape_xyz = np.asarray(cache.native_shape_xyz, dtype=np.int64)
        self.norm_ratio_xyz = np.asarray(cache.norm_ratio_xyz, dtype=np.float64)
        self.native_to_lps = np.asarray(native_to_lps if native_to_lps is not None else np.eye(4), dtype=np.float64)
        if self.native_to_lps.shape!=(4,4) or not np.isfinite(self.native_to_lps).all() or abs(np.linalg.det(self.native_to_lps[:3,:3]))<1e-12:
            raise ValueError('Invalid embedding physical affine')
        if self.norm_ratio_xyz.shape!=(3,) or not np.isfinite(self.norm_ratio_xyz).all() or np.any(self.norm_ratio_xyz<=0):
            raise ValueError('Invalid embedding normalization ratio')
        if np.any(self.shape_xyz<=0) or np.any(self.native_shape_xyz<=0): raise ValueError('Empty embedding geometry')

    @classmethod
    def from_arrays(cls, fine, coarse, semantic, norm_ratio=(2., 2., 2.), native_to_lps=None):
        class Arrays:
            native_shape_xyz = (fine.shape[3], fine.shape[2], fine.shape[1])
            norm_ratio_xyz = np.asarray(norm_ratio)
            def feature_shape_xyz(self, level):
                value = self.valid_array(level)
                return (value.shape[3], value.shape[2], value.shape[1])
            def valid_array(self, level):
                return dict(fine=fine, coarse=coarse, semantic=semantic)[level]
        return cls(Arrays(), native_to_lps)

    def descriptors(self, points):
        points = np.asarray(points, dtype=np.int64).reshape(-1, 3)
        if np.any(points < 0) or np.any(points >= self.shape_xyz):
            raise ValueError('query_outside_fine_grid')
        fine, semantic = [np.asarray(self.cache.valid_array(head)[:, points[:, 2], points[:, 1], points[:, 0]].T, dtype=np.float32)
                          for head in ('fine', 'semantic')]
        coarse = self.cache.valid_array('coarse')
        coarse_xyz = np.asarray(coarse.shape[:0:-1], dtype=np.int64)
        positions = (points + .5) * coarse_xyz / self.shape_xyz - .5
        lower = np.floor(positions).astype(np.int64)
        fraction = (positions - lower).astype(np.float32)
        interpolated = np.zeros((len(points), coarse.shape[0]), dtype=np.float32)
        for corner in itertools.product((0, 1), repeat=3):
            offset = np.asarray(corner)
            coords = np.clip(lower + offset, 0, coarse_xyz - 1)
            weights = np.prod(np.where(offset, fraction, 1-fraction), axis=1, dtype=np.float32)
            interpolated += np.asarray(coarse[:, coords[:, 2], coords[:, 1], coords[:, 0]].T, dtype=np.float32) * weights[:, None]
        norms = np.linalg.norm(interpolated, axis=1, keepdims=True)
        interpolated /= np.maximum(norms, np.float32(1e-12))
        return fine, interpolated, semantic

    def fine_to_native(self, points, quantized=True):
        continuous = (np.asarray(points, dtype=np.float64) * 2 + .5) / self.norm_ratio_xyz
        return np.rint(continuous).astype(np.int64) if quantized else continuous

    def native_to_fine(self, points):
        return np.floor(np.asarray(points) * self.norm_ratio_xyz * .5).astype(np.int64)

    def lps(self, native):
        points = np.asarray(native, dtype=np.float64).reshape(-1, 3)
        return points @ self.native_to_lps[:3, :3].T + self.native_to_lps[:3, 3]


class FineGridRetriever:
    """Exhaustive head-mean scoring, with dense released-order tie semantics."""
    def __init__(self, backend='auto', chunk_locations=32768, dense_budget_bytes=512*1024**2, device='cpu'):
        if backend not in ('dense', 'streamed', 'auto') or chunk_locations <= 0 or dense_budget_bytes <= 0:
            raise ValueError('Invalid retrieval backend or budget')
        self.backend, self.chunk_locations = backend, int(chunk_locations)
        self.dense_budget_bytes, self.device = int(dense_budget_bytes), device
        self._cuda_resident = None

    def domain(self, target, domain=None):
        box = np.array(domain if domain is not None else [[0, 0, 0], target.shape_xyz], dtype=np.int64)
        if box.shape != (2, 3) or np.any(box[0] < 0) or np.any(box[1] > target.shape_xyz) or np.any(box[1] <= box[0]):
            raise ValueError('invalid_admissible_domain')
        return box

    def _cuda_plan(self,source,target,points,domain):
        box=self.domain(target,domain)
        channels=sum(source.cache.valid_array(h).shape[0] for h in ('fine','coarse','semantic'))
        count=int(np.prod(box[1]-box[0]));full_target=int(np.prod(target.shape_xyz));full_source=int(np.prod(source.shape_xyz))
        estimated=max(4*(4*len(points)*count+channels*count+channels*len(points)),
                      4*(4*len(points)*full_target+4*channels*(full_target+full_source)))
        backend=self.backend
        if backend=='auto':backend='dense' if estimated<=self.dense_budget_bytes else 'streamed'
        if backend=='dense' and estimated>self.dense_budget_bytes:raise MemoryError('dense_retrieval_budget_exceeded')
        return box,estimated,backend

    def _resident_cuda(self):
        if self._cuda_resident is None:
            from tools.quadra.reviewed_uae_cuda import ResidentCudaScorer
            self._cuda_resident=ResidentCudaScorer(self.device,self.dense_budget_bytes//2,max_entries=2)
        return self._cuda_resident

    def cuda_candidates(self,source,target,point,domain,separation,count):
        box,estimated,backend=self._cuda_plan(source,target,[point],domain)
        if backend!='dense':return None  # preserve the existing streamed numerical path
        return self._resident_cuda().spatial_candidates(source,target,point,box,separation,count)

    def cuda_profile(self):
        return self._cuda_resident.profile() if self._cuda_resident is not None else None

    def close(self):
        if self._cuda_resident is not None:self._cuda_resident.close()

    def score_blocks(self, source, target, points, domain=None):
        if self.device!='cpu':
            box,estimated,backend=self._cuda_plan(source,target,points,domain)
            if backend=='dense':
                yield self._resident_cuda().score_block(source,target,points,box,estimated)
                return
        box = self.domain(target, domain)
        size = box[1] - box[0]
        count = int(np.prod(size))
        q = source.descriptors(points)
        channels = sum(v.shape[1] for v in q)
        estimated = 4 * (4*len(points)*count + channels*count + channels*len(points))
        if self.device != 'cpu':
            # The released implementation upsamples the complete coarse arrays
            # and creates full similarity maps before applying the controlled ROI.
            full_target=int(np.prod(target.shape_xyz)); full_source=int(np.prod(source.shape_xyz))
            estimated=max(estimated,4*(4*len(points)*full_target+4*channels*(full_target+full_source)))
        backend = self.backend
        if backend == 'auto': backend = 'dense' if estimated <= self.dense_budget_bytes else 'streamed'
        if backend == 'dense' and estimated > self.dense_budget_bytes:
            raise MemoryError('dense_retrieval_budget_exceeded')
        block = count if backend == 'dense' else self.chunk_locations
        if self.device != 'cpu':
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('Requested CUDA is unavailable')
            if torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError('FP32 retrieval requires torch.backends.cuda.matmul.allow_tf32=False')
            if backend == 'dense':
                fused_full= released_dense_scores(source,target,points,self.device)
                # Preserve the released scoring operation, then explicitly select
                # the identical admissible domain used by NN and fixed-point.
                xs=np.arange(box[0,0],box[1,0]);ys=np.arange(box[0,1],box[1,1]);zs=np.arange(box[0,2],box[1,2])
                zz,yy,xx=np.meshgrid(zs,ys,xs,indexing='ij')
                xyz=np.column_stack((xx.ravel(),yy.ravel(),zz.ravel()))
                selected=fused_full[:,zz.ravel(),yy.ravel(),xx.ravel()]
                # Independent head scores are computed by bounded FP32 descriptors
                # for diagnostic candidates; the matching fused score is released.
                keys=target.descriptors(xyz)
                heads=[np.matmul(a,b.T).astype(np.float32) for a,b in zip(q,keys)]
                yield xyz,selected,heads,'released_dense_fp32_adapter',estimated
                return
            q_device = [torch.as_tensor(v, dtype=torch.float32, device=self.device) for v in q]
        for start in range(0, count, block):
            linear = np.arange(start, min(start+block, count), dtype=np.int64)
            xyz = np.column_stack((linear % size[0], (linear // size[0]) % size[1], linear // (size[0]*size[1]))) + box[0]
            keys = target.descriptors(xyz)
            if self.device == 'cpu':
                heads = [np.matmul(a, b.T).astype(np.float32) for a, b in zip(q, keys)]
            else:
                heads = [(a @ torch.as_tensor(b.T.copy(), dtype=torch.float32, device=self.device)).cpu().numpy() for a,b in zip(q_device, keys)]
            fused = ((heads[0] + heads[1]) + heads[2]) / np.float32(3)
            yield xyz, fused, heads, backend, estimated

    def match(self, source, target, points, domain=None):
        points = np.asarray(points, dtype=np.int64).reshape(-1, 3)
        if self.device!='cpu':
            box,estimated,backend=self._cuda_plan(source,target,points,domain)
            if backend=='dense':return self._resident_cuda().match(source,target,points,box,estimated)
        best = np.full(len(points), -np.inf, dtype=np.float32)
        winners = np.zeros((len(points), 3), dtype=np.int64)
        finite_queries = np.ones(len(points), dtype=bool)
        searched = 0
        for xyz, scores, heads, backend, estimated in self.score_blocks(source, target, points, domain):
            finite_queries &= np.isfinite(scores).all(axis=1)
            safe = np.where(np.isfinite(scores), scores, -np.inf)
            indices = safe.argmax(axis=1)
            values = safe[np.arange(len(points)), indices]
            improved = values > best  # blocks and entries use globally increasing ZYX indices
            best[improved], winners[improved] = values[improved], xyz[indices[improved]]
            searched += len(xyz)
        query_heads = source.descriptors(points)
        nonempty = np.any(np.column_stack([np.linalg.norm(v, axis=1) > 1e-12 for v in query_heads]), axis=1)
        statuses = ['success' if finite_queries[i] and nonempty[i] and np.isfinite(best[i]) else 'failed' for i in range(len(points))]
        return dict(points_fine_xyz=winners.tolist(), scores=[float(x) if np.isfinite(x) else None for x in best],
                    statuses=statuses, failure_reasons=[None if s=='success' else 'invalid_or_empty_descriptor' for s in statuses],
                    backend=backend, estimated_dense_bytes=estimated, searched_target_locations=searched,
                    similarity_formula='mean(fine,normalized_interpolated_coarse,semantic)',
                    tie_policy='first_global_zyx_flat_index', score_dtype='float32',
                    admissible_domain=self.domain(target, domain).tolist())


REFERENCE_COMMIT = '035bbfc1816e40c122303a5e21accbdc35beac43'
REFERENCE_HASHES = {
    'uae_fixed_point_035bbfc.py': 'bfe08036f3dd724851ca97c96e7dc45feb5e9f8809fe189ad2956d1993e847e0',
    'uae_interfaces_035bbfc.py': 'a1bf3616b7d87845a972f2b3f32711f9bd1899aaeb1dcf184b898bd53720ac77',
    'uae_utils_035bbfc.py': '8074334d8b51462217037e7127935e27b55636ffdd4f61470f61ee0ac9fd2cb7',
}


def reference_identity():
    root = Path(__file__).parent / 'references'
    for name, digest in REFERENCE_HASHES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned released reference changed: '+name)
    return dict(repository='https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2',
                commit=REFERENCE_COMMIT, files=REFERENCE_HASHES.copy(),
                scope='original structural inference; fixture retrieval injected; GPU parity pending')


def released_dense_scores(source,target,points,device='cpu'):
    """Pinned original dense numerical statements, exposing their fused map.

    The adapter explicitly promotes architectural FP16 storage to FP32 input.
    The early return only skips original argmax/visualization; F.interpolate,
    F.normalize, einsum and the original head fusion execute unmodified.
    """
    import torch
    import torch.nn.functional as functional
    reference_identity()
    path=Path(__file__).parent/'references/uae_interfaces_035bbfc.py'
    tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='get_sim_semantic_embed_loc_multi_embedding_space')
    body=[]
    for statement in function.body:
        body.append(statement)
        if isinstance(statement,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='sim_all' for v in statement.targets):
            body.append(ast.Return(value=ast.Name(id='sim_all',ctx=ast.Load())))
            break
    function.body=body
    namespace=dict(np=np,torch=torch,F=functional)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),namespace)
    arrays=[]
    for cache in (source,target):
        heads=[cache.cache.valid_array(head) for head in ('fine','coarse','semantic')]
        if any(v.shape[0]!=128 for v in heads): raise ValueError('Released dense reference requires original 128-channel heads')
        arrays.append([torch.as_tensor(np.asarray(v,dtype=np.float32).copy(),device=device).unsqueeze(0) for v in heads])
    with torch.no_grad():
        values=namespace[function.name](arrays[0],arrays[1],np.asarray(points),torch.device(device))
    return values[:,0].cpu().numpy().astype(np.float32)


def _local_patch(center, shape, margin):
    points = np.array(list(itertools.product(*[range(int(c-m), int(c+m+1)) for c,m in zip(center, margin)])), dtype=np.int64)
    return np.unique(np.clip(points, 0, np.asarray(shape)-1), axis=0)


def is_cuda_out_of_memory(error):
    """Torch 1.9 reports CUDA OOM as RuntimeError; newer releases subclass it."""
    message=str(error).lower().replace('_',' ')
    return 'cuda' in message and ('out of memory' in message or 'outofmemory' in message)


def fixed_point(source, target, original_native, retriever, margin=(2, 2, 2), iterations=4,
                score_threshold=.8, max_return_distance_normalized=100., source_domain=None, target_domain=None):
    """Released structural semantics, with declared guards for undefined fits."""
    import statsmodels.api as sm
    if len(margin) != 3 or any(m < 0 for m in margin) or iterations < 2 or iterations % 2:
        raise ValueError('Fixed-point requires nonnegative margins and positive even iteration count')
    original = np.rint(original_native).astype(np.int64)
    seed = source.native_to_fine(original)
    history = []
    settings = dict(margin_xyz=list(margin), iterations=iterations, score_threshold=score_threshold,
                    return_distance_normalized_units=max_return_distance_normalized,
                    return_distance_mm_default_normalization=2*max_return_distance_normalized,
                    guards_variant='rank/anchor-count/nonfinite/out-of-domain guards; no NN fallback',
                    normalized_units_per_mm_default=.5)
    base = dict(status='failed', point_native_xyz=None, score=None, failure_reason=None,
                anchor_history=history, filter_keep=[], settings=settings, fit=None,retrieval_profiles=[])
    if np.any(seed < 0) or np.any(seed >= source.shape_xyz):
        base['failure_reason'] = 'query_outside_fine_grid'; return base
    current = _local_patch(seed, source.shape_xyz, margin)
    final_target = final_return = final_scores = None
    try:
        for iteration in range(iterations):
            a,b = (source,target) if iteration % 2 == 0 else (target,source)
            if iteration:
                current = a.native_to_fine(current)
                # Original retrieval clips queries to embedding bounds, preserve that operation.
                current = np.clip(current, 0, a.shape_xyz-1)
            result = retriever.match(a, b, current, target_domain if iteration % 2 == 0 else source_domain)
            base['retrieval_profiles'].append({k:result[k] for k in ('backend','estimated_dense_bytes','searched_target_locations','admissible_domain','similarity_formula','tie_policy','score_dtype')})
            native = b.fine_to_native(result['points_fine_xyz'])
            history.append(dict(iteration=iteration+1, direction='forward' if iteration%2==0 else 'reverse',
                                query_fine_xyz=current.tolist(), matched_fine_xyz=result['points_fine_xyz'],
                                matched_native_xyz=native.tolist(), scores=result['scores']))
            if any(s != 'success' for s in result['statuses']):
                raise ValueError('invalid_anchor_descriptor')
            current = native
            if iteration % 2 == 0: final_target = native.copy()
            else: final_return, final_scores = native.copy(), np.asarray(result['scores'])
        distance = np.linalg.norm((final_return-original) * source.norm_ratio_xyz, axis=1)
        keep = (distance < max_return_distance_normalized) & (final_scores > score_threshold)
        base['filter_keep'] = keep.tolist()
        returned, matched = final_return[keep], final_target[keep]
        base['score'] = float(final_scores[keep].mean()) if keep.any() else None  # released score precedes deduplication
        if len(matched):
            _, indices = np.unique(matched, axis=0, return_index=True)
            # Released final_k_count is a dict built in first encounter order.
            # np.unique sorts keys; restore the original insertion order here.
            indices.sort(); returned, matched = returned[indices], matched[indices]
        base['stable_returned_native_xyz'] = returned.tolist()
        base['stable_matched_native_xyz'] = matched.tolist()
        planar = len(matched) > 0 and len(np.unique(matched[:, 2])) == 1
        design = np.column_stack((returned[:, :2] if planar else returned, np.ones(len(returned))))
        required = 3 if planar else 4
        rank = int(np.linalg.matrix_rank(design)) if len(design) else 0
        if len(matched) < required: raise ValueError('insufficient_stable_anchors:n={}'.format(len(matched)))
        if rank != required: raise ValueError('degenerate_affine_geometry:rank={}'.format(rank))
        if planar and len(np.unique(returned[:, 2])) != 1:
            raise ValueError('degenerate_planar_geometry:multiple_source_z')
        matrix = np.eye(4)
        for axis in range(2 if planar else 3):
            parameters = sm.RLM(matched[:, axis], design).fit().params
            if planar: matrix[axis] = [parameters[0], parameters[1], 0., parameters[2]]
            else: matrix[axis] = parameters
        if planar:
            matrix[2, 2] = source.norm_ratio_xyz[2]/target.norm_ratio_xyz[2]
            matrix[2, 3] = matched[0, 2]-returned[0, 2]*matrix[2, 2]
        corrected_continuous = matched - (returned-original) @ matrix[:3, :3].T
        if not np.isfinite(corrected_continuous).all(): raise ValueError('nonfinite_affine_correction')
        corrected_integer = corrected_continuous.astype(np.int64)  # original truncates, then clips, then means
        clipped = np.clip(corrected_integer, 0, target.native_shape_xyz-1)
        point = clipped.mean(axis=0).astype(np.int64)
        base['fit'] = dict(matrix=matrix.tolist(), rank=rank, mode='planar' if planar else '3d',
                           correction_continuous_xyz=corrected_continuous.tolist(), corrected_integer_xyz=corrected_integer.tolist(),
                           corrected_clipped_xyz=clipped.tolist(), clipping_occurred=bool(np.any(clipped != corrected_integer)))
        predicted_fine = target.native_to_fine(point)
        box = retriever.domain(target, target_domain)
        if np.any(predicted_fine < box[0]) or np.any(predicted_fine >= box[1]):
            raise ValueError('corrected_prediction_outside_admissible_domain')
        base.update(status='success', point_native_xyz=point.tolist())
    except RuntimeError:
        # Runtime failures (especially CUDA OOM) stop the session. They are not
        # anatomical/structural query failures and must never trigger continuation.
        raise
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        base['failure_reason'] = str(exc)
    return base


def released_reference(source, target, original_native, retriever, margin=(2, 2, 2), iterations=4):
    """Execute the pinned original function, never the production FP wrapper.

    Only imports/model setup are omitted. The final return is instrumented to
    expose original local filter arrays; no numerical statement is rewritten.
    Retrieval is a declared fixture dependency, not a claim of original GPU parity.
    """
    import statsmodels.api as sm
    reference_identity()
    root = Path(__file__).parent/'references'
    namespace = dict(np=np, sm=sm, torch=SimpleNamespace(device=lambda value:value))
    history = []
    def matching(a,b,points,device):
        src,dst = (source,target) if a is emb1 else (target,source)
        result = retriever.match(src,dst,points)
        matched = np.asarray(result['points_fine_xyz'])
        history.append(dict(iteration=len(history)+1, direction='forward' if len(history)%2==0 else 'reverse',
            query_fine_xyz=np.clip(points, 0, src.shape_xyz-1).tolist(), matched_fine_xyz=matched.tolist(),
            matched_native_xyz=dst.fine_to_native(matched).tolist(), scores=result['scores']))
        return matched, np.asarray(result['scores'])
    namespace['get_sim_semantic_embed_loc_multi_embedding_space'] = matching
    for file, name in [('uae_utils_035bbfc.py', 'make_local_grid_patch'), ('uae_fixed_point_035bbfc.py', 'fixed_point_iterations')]:
        tree = ast.parse((root/file).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name==name)
        if name == 'fixed_point_iterations':
            function.body[-1].value = ast.Tuple(elts=[function.body[-1].value, ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], ctx=ast.Load())
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), file, 'exec'), namespace)
    emb1 = [SimpleNamespace(shape=(1, 1, int(source.shape_xyz[2]), int(source.shape_xyz[1]), int(source.shape_xyz[0])))]
    emb2 = [SimpleNamespace(shape=(1, 1, int(target.shape_xyz[2]), int(target.shape_xyz[1]), int(target.shape_xyz[0])))]
    shape = (1, int(target.native_shape_xyz[1]), int(target.native_shape_xyz[0]), int(target.native_shape_xyz[2]))
    try:
        (score, point), local = namespace['fixed_point_iterations'](emb1,source.norm_ratio_xyz,emb2,target.norm_ratio_xyz,shape,
                                                                 np.asarray(original_native), *margin, iterations=iterations)
        keep = (local['dis'] < 100) & (local['pt_final_score'] > .8)
        return dict(status='success', point_native_xyz=point.tolist(), score=float(score), anchor_history=history, filter_keep=keep.tolist())
    except (ValueError, IndexError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
        return dict(status='failed', point_native_xyz=None, failure_reason=type(exc).__name__+':'+str(exc), anchor_history=history)


def cycle_outcome(query, source, target, method, retriever, source_domain=None, target_domain=None,
                  margin=(2,2,2), peak_count=3, peak_separation_mm=8., peak_score_gap=.03):
    """Independent directional matching, preserving the frozen native query."""
    from tools.quadra.reviewed_uae_diagnostics import spatial_peak_candidates
    original_fine = np.array([int(query['fine_'+axis]) for axis in 'xyz'], dtype=np.int64)
    original_physical = np.array([float(query['physical_lps_'+axis]) for axis in 'xyz'])
    if all('model_continuous_'+axis in query for axis in 'xyz'):
        original_continuous=np.array([float(query['model_continuous_'+axis]) for axis in 'xyz'])
        if not np.allclose(source.lps(original_continuous)[0],original_physical,atol=1e-5,rtol=0):
            raise ValueError('frozen_model_physical_geometry_mismatch')
    else:
        # Only synthetic contracts may lack the authoritative frozen model values.
        original_continuous=(original_physical-source.native_to_lps[:3,3]) @ np.linalg.inv(source.native_to_lps[:3,:3]).T
    original_native = np.rint(original_continuous).astype(np.int64)
    row = dict(query, method=method, status='failed', forward_status='pending', reverse_status='pending',
               cycle_error_mm=None, failure_reason=None, source_fine_xyz=original_fine.tolist(),
               source_native_xyz=original_native.tolist(), boundary_anomaly=None, competing_peak=None)
    row.update(source_continuous_native_xyz=original_continuous.tolist(),
               source_model_quantization_error_mm=float(np.linalg.norm(source.lps(original_native)[0]-original_physical)),
               source_descriptor_center_lps_xyz=source.lps(source.fine_to_native(original_fine,quantized=False))[0].tolist())
    row['source_fine_quantization_error_mm']=float(np.linalg.norm(np.asarray(row['source_descriptor_center_lps_xyz'])-original_physical))
    traces = []; maps = []
    if np.any(original_fine < 0) or np.any(original_fine >= source.shape_xyz):
        row.update(forward_status='failed', reverse_status='not_attempted', failure_reason='query_outside_fine_grid')
        return row, traces, maps
    converted_fine = source.native_to_fine(original_native)
    if not np.array_equal(converted_fine, original_fine):
        row.update(forward_status='failed', reverse_status='not_attempted', failure_reason='frozen_query_quantization_mismatch')
        return row, traces, maps
    for direction, a,b, domain, query_point in [('forward',source,target,target_domain,original_native), ('reverse',target,source,source_domain,None)]:
        if direction == 'reverse': query_point = np.asarray(row['matched_native_xyz'])
        fine_point = original_fine if direction=='forward' else a.native_to_fine(query_point)
        if method == 'uae_nn':
            try:
                result = retriever.match(a,b,[fine_point],domain)
                point_fine = np.asarray(result['points_fine_xyz'][0])
                point_native = b.fine_to_native(point_fine)
                result = dict(status=result['statuses'][0], failure_reason=result['failure_reasons'][0],
                              score=result['scores'][0], point_native_xyz=point_native.tolist(),
                              point_continuous_native_xyz=b.fine_to_native(point_fine,quantized=False).tolist(),
                              point_fine_xyz=point_fine.tolist(), backend=result['backend'])
            except ValueError as exc:
                result = dict(status='failed',failure_reason=str(exc))
        elif method == 'uae_fixed_point':
            result = fixed_point(a,b,query_point,retriever,margin=margin,
                                 source_domain=source_domain if direction=='forward' else target_domain,
                                 target_domain=domain)
            traces.append((query['query_id'],direction,result))
        else: raise ValueError('Unknown UAE method')
        row[direction+'_status'] = result['status']
        row[direction+'_score'] = result.get('score')
        row[direction+'_retrieval_backend'] = result.get('backend', [p['backend'] for p in result.get('retrieval_profiles',[])])
        # Structural failure can still have meaningful seed similarity evidence.
        # Preserve it before deciding whether a reverse match can be attempted.
        valid_query=all(np.isfinite(v).all() for v in a.descriptors([fine_point]))
        nonempty_query=any(np.linalg.norm(v)>1e-12 for v in a.descriptors([fine_point]))
        peaks=spatial_peak_candidates(a,b,fine_point,retriever,domain,peak_separation_mm,peak_count) if valid_query and nonempty_query else []
        row[direction+'_peak_candidates']=peaks
        row[direction+'_peak_evidence_status']='available' if peaks else 'unavailable_invalid_or_empty_query_descriptor'
        if peaks and peak_count>1:
            row['competing_peak']=bool(row['competing_peak']) or (len(peaks)>1 and peaks[0]['fused_score']-peaks[1]['fused_score'] <= peak_score_gap)
        if result['status'] != 'success':
            row['failure_reason'] = direction+':'+str(result['failure_reason'])
            if direction=='forward': row['reverse_status']='not_attempted'
            if peaks:
                maps.append(dict(direction=direction,source=a,target=b,point_fine=fine_point.tolist(),domain=domain,
                    map_role='seed' if direction=='forward' else 'reverse_query',center_fine=peaks[0]['fine_xyz']))
            break
        point = np.asarray(result['point_native_xyz'])
        prefix = 'matched' if direction=='forward' else 'returned'
        physical = b.lps(point)[0]
        row[prefix+'_native_xyz'] = point.tolist(); row[prefix+'_lps_xyz'] = physical.tolist()
        row[prefix+'_model_xyz'] = point.tolist()
        continuous = result.get('point_continuous_native_xyz', point.tolist())
        row[prefix+'_continuous_native_xyz'] = continuous
        row[prefix+'_continuous_lps_xyz'] = b.lps(continuous)[0].tolist()
        row[prefix+'_fine_xyz'] = result.get('point_fine_xyz', b.native_to_fine(point).tolist())
        for axis,value in zip('xyz',physical): row[prefix+'_lps_'+axis] = float(value)
        box = retriever.domain(b,domain)
        boundary = bool(np.any(np.asarray(row[prefix+'_fine_xyz'])==box[0]) or np.any(np.asarray(row[prefix+'_fine_xyz'])==box[1]-1))
        row[direction+'_boundary'] = boundary
        row['boundary_anomaly']=bool(row['boundary_anomaly']) or boundary or bool((result.get('fit') or {}).get('clipping_occurred'))
        maps.append(dict(direction=direction,source=a,target=b,point_fine=fine_point.tolist(),domain=domain,
                         map_role='seed' if direction=='forward' else 'reverse_query',center_fine=row[prefix+'_fine_xyz']))
        if method=='uae_fixed_point' and result['anchor_history']:
            first=result['anchor_history'][0]
            # This is an actual first-iteration anchor, explicitly distinguished from the seed.
            maps.append(dict(direction=direction,source=a,target=b,point_fine=first['query_fine_xyz'][0],domain=domain,
                             map_role='anchor',center_fine=first['matched_fine_xyz'][0]))
    if row['forward_status']=='success' and row['reverse_status']=='success':
        row.update(status='success',failure_reason=None,cycle_error_mm=float(np.linalg.norm(np.asarray(row['returned_lps_xyz'])-original_physical)))
    return row,traces,maps


def run_reviewed_uae(args, retrieval_factory=FineGridRetriever):
    from tools.quadra import aligned_organ_group_cohort as cohort
    from tools.quadra.reviewed_matching_contract import read_contract
    from tools.quadra.reviewed_evidence import write_method_bundle, read_method_bundle
    from tools.quadra.streaming_cycle_error import EmbeddingCache
    from tools.quadra.reviewed_uae_diagnostics import pack_anchor_traces, export_similarity_views
    contract,queries,signature = read_contract(args.contract)
    root=Path(args.run_directory)
    config = json.loads(Path(args.cache_index).read_text())
    fixture = contract.get('fixture_only') is True
    if config.get('fixture_only',False) != fixture:
        raise cohort.CohortError('Fixture and real cache index identities differ')
    if not fixture:
        if config.get('complete') is not True:
            raise cohort.CohortError('Reviewed extraction index must complete both sessions before matching')
        if not args.approved_pilot: raise cohort.CohortError('Real-data UAE execution requires explicit approved pilot gate')
        allowed = set(contract['pilot_subjects'])
        requested = set(args.subject or allowed)
        if not requested <= allowed: raise cohort.CohortError('This command is bounded to the frozen pilot; cohort remains gated')
        queries = [q for q in queries if q['subject_id'] in requested]
        if any(not all('model_continuous_'+axis in q for axis in 'xyz') for q in queries):
            raise cohort.CohortError('Real UAE queries require authoritative frozen model continuous coordinates')
        if not args.device.startswith('cuda'): raise cohort.CohortError('Real UAE pilot requires explicitly selected CUDA scoring')
    elif args.subject:
        queries = [q for q in queries if q['subject_id'] in set(args.subject)]
    if args.checkpoint_queries < 1 or args.peak_count < 1 or args.diagnostic_budget_bytes < 1 or args.min_free_bytes < 0:
        raise cohort.CohortError('Resource/diagnostic/checkpoint settings must be positive')
    if not np.isfinite(args.peak_separation_mm) or args.peak_separation_mm<=0 or not np.isfinite(args.peak_score_gap) or args.peak_score_gap<0:
        raise cohort.CohortError('Peak separation must be finite and positive; score gap finite and nonnegative')
    if args.select_upper_tail is not None and not 0<args.select_upper_tail<=100:
        raise cohort.CohortError('Post-cycle tail percentage must lie in (0,100]')
    dense_budget=args.dense_budget_bytes if args.dense_budget_bytes is not None else (512*1024**2 if args.device=='cpu' else 32*1024**3)
    root.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+root.name+'-diagnostics-',dir=str(root.parent)))
    settings = dict(margin_xyz=list(args.margin), iterations=4, score_threshold=.8, return_distance_normalized_units=100.,
                    return_distance_mm_default_normalization=200., peak_count=args.peak_count,
                    peak_separation_mm=args.peak_separation_mm, peak_score_gap=args.peak_score_gap,
                    diagnostic_budget_bytes=args.diagnostic_budget_bytes, min_free_bytes=args.min_free_bytes,
                    chunk_locations=args.chunk_locations, dense_budget_bytes=dense_budget, device=args.device,
                    checkpoint_queries=args.checkpoint_queries, selected_query_ids=sorted(args.selected_query or []),
                    selected_subject_ids=sorted({q['subject_id'] for q in queries}),tf32_matmul=False)
    settings['select_upper_tail_percent']=args.select_upper_tail
    metadata = dict(settings=settings, backend=args.backend, cache_policy='retain_until_first_anatomical_review',
                    pending_gates=['real_gpu_reference_parity','extraction_context_equivalence','resource_pilot','anatomical_review'],
                    reference=reference_identity(), fixture_signature=cohort.sha256_file(args.cache_index),
                    diagnostic_threshold_status='provisional_until_pilot', retrieval_score_atol=1e-6,
                    retrieval_score_rtol=1e-5, extraction_context_validated=False, scientific_work_launched=False,
                    resource_guard_policy='between_queries_and_similarity_passes; reserve context_and_peak_views; anchor trace overhead measured',
                    map_capture_policy='context_and_global_finite_maximum; second exhaustive pass when centres differ; pilot overhead pending',
                    dense_budget_status='provisional CPU512MiB/CUDA32GiB defaults; real fit measured by pilot; OOM blocks',
                    original_retrieval_precision_deviation='FP32 coarse interpolation/normalization and scoring; released code uses input tensor dtype; real GPU comparison pending')
    metadata['cache_native_coordinate_frame']='fixture_cache_voxel_xyz' if fixture else 'padded_2mm_model_voxel_xyz'
    metadata['physical_coordinate_frame']='LPS_mm'
    rows=[]; completed=set(); old_cache_identities=[]
    if (root/'output_inventory.json').exists():
        if not args.resume: raise cohort.CohortError('Existing method evidence requires explicit resume')
        old,previous=read_method_bundle(root,args.contract)
        for key in ('settings','backend','fixture_signature'):
            if old.get(key)!=metadata[key]: raise cohort.CohortError('Cannot resume changed '+key)
        from tools.quadra.reviewed_evidence import execution_signature
        if old['execution_signature']!=execution_signature(): raise cohort.CohortError('Cannot resume changed source')
        rows=[r for r in previous if r['status']!='pending']; completed={r['query_id'] for r in rows}
        metadata['scientific_work_launched']=old.get('scientific_work_launched',False)
        old_cache_identities=list(old.get('cache_identities',[]))
    elif (root/'method_manifest.json').exists():
        prepared=cohort.load_json(root/'method_manifest.json')
        if prepared.get('status')!='awaiting_pilot': raise cohort.CohortError('Unsealed previous execution preserved')
    entries={}
    for item in config['groups']:
        key=(item['subject_id'],item['group_name'])
        if key in entries: raise cohort.CohortError('Duplicate cache index group')
        entries[key]=item
    cache_handles=[]; opened={}; cache_identities=list(old_cache_identities)
    retriever=retrieval_factory(args.backend,args.chunk_locations,dense_budget,args.device)
    previous_tf32=None
    if args.device!='cpu':
        import torch
        torch.cuda.set_device(torch.device(args.device))
        previous_tf32=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.reset_peak_memory_stats(args.device)
    started=time.monotonic(); blocked=None; selected=set(args.selected_query or []); batch_attempts=0
    diagnostic_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and p.name not in ('query_outcomes.csv','method_manifest.json','output_inventory.json'))
    staging_bytes=0
    if not selected <= {q['query_id'] for q in queries}: raise cohort.CohortError('Unknown selected map query')
    try:
        for query in queries:
            if query['query_id'] in completed: continue
            if shutil.disk_usage(root).free < args.min_free_bytes:
                blocked='disk_free_guard'; break
            if diagnostic_bytes >= args.diagnostic_budget_bytes:
                blocked='diagnostic_budget_guard'; break
            key=(query['subject_id'],query['group_name'])
            if key not in opened:
                if key not in entries: raise cohort.CohortError('Missing cache group: '+str(key))
                item=entries[key]; pair=[]
                for session in ('test','retest'):
                    spec=item[session]; cache_root=Path(spec['cache_directory'])
                    cache=EmbeddingCache(cache_root); cache_handles.append(cache)
                    if fixture:
                        affine=np.asarray(spec['native_to_lps'])
                    else:
                        plan_path='plans/{}-{}-{}.json'.format(key[0],session,key[1])
                        plan=json.loads((Path(args.contract)/plan_path).read_text())
                        affine=np.diag([-1.,-1.,1.,1.]) @ np.asarray(plan['source_ct']['affine']) @ np.asarray(plan['model_to_raw_continuous_affine'])
                        # Existing extraction cache identities must be pinned by the producer.
                        if spec.get('plan_sha256')!=cohort.sha256_file(Path(args.contract)/plan_path): raise cohort.CohortError('Embedding cache crop identity differs')
                        if spec.get('input_signature')!=signature: raise cohort.CohortError('Embedding cache reviewed input identity differs')
                        if cache.manifest.get('model_profile')!='uae_s': raise cohort.CohortError('Embedding model profile differs')
                        if cache.native_shape_xyz != tuple(plan['padded_shape_xyz']): raise cohort.CohortError('Embedding native model grid differs from frozen crop')
                        if not np.allclose(cache.norm_ratio_xyz,[1,1,1],atol=1e-6): raise cohort.CohortError('Embedding normalization differs from exact 2mm model grid')
                        provenance=cache.manifest.get('reviewed_provenance',{})
                        expected=dict(input_signature=signature,plan_sha256=cohort.sha256_file(Path(args.contract)/plan_path),
                            source_ct_sha256=plan['source_ct']['sha256'],model_config_sha256=contract['model']['config_sha256'],
                            model_checkpoint_sha256=contract['model']['expected_checkpoint_sha256'],
                            model_precision='fp32',embedding_dtype='float16',preprocessing='global_lattice_2mm_clip_-1024_3071_scale_255_offset_50')
                        if any(provenance.get(k)!=v for k,v in expected.items()): raise cohort.CohortError('Embedding cache model/preprocessing/reviewed provenance differs')
                        if not np.allclose(provenance.get('native_to_lps',np.zeros((4,4))),affine,atol=1e-5,rtol=0): raise cohort.CohortError('Embedding native physical geometry differs')
                        expected_fine=np.asarray(plan['padded_shape_xyz'])//2
                        if not np.array_equal(cache.feature_shape_xyz('fine'),expected_fine) or cache.feature_shape_xyz('semantic')!=cache.feature_shape_xyz('fine'):
                            raise cohort.CohortError('Embedding fine/semantic grid differs from frozen crop')
                        expected_box=np.column_stack((np.ceil(np.asarray(plan['valid_model_box_xyz'])[0]/2),np.ceil(np.asarray(plan['valid_model_box_xyz'])[1]/2))).T.astype(np.int64).tolist()
                        if spec.get('admissible_domain') != expected_box: raise cohort.CohortError('Embedding admissible fine domain differs from frozen crop')
                        for head in ('fine','coarse','semantic'):
                            if cache.valid_array(head).dtype!=np.float16: raise cohort.CohortError('Embedding storage dtype differs')
                        if spec.get('ct_path') and spec.get('ct_sha256')!=plan['source_ct']['sha256']:
                            raise cohort.CohortError('Diagnostic CT selection differs from frozen crop source')
                    identity=dict(subject_id=key[0],group_name=key[1],session=session,
                                  files={name:cohort.file_identity(cache_root/name) for name in ('manifest.json','fine.npy','coarse.npy','semantic.npy')})
                    if not fixture and spec.get('cache_file_sha256')!={name:record['sha256'] for name,record in identity['files'].items()}:
                        raise cohort.CohortError('Embedding cache file content differs from its extraction index')
                    old_identity=next((v for v in cache_identities if (v['subject_id'],v['group_name'],v['session'])==(key[0],key[1],session)),None)
                    if old_identity is not None and old_identity!=identity: raise cohort.CohortError('Resumed embedding identity differs')
                    if old_identity is None: cache_identities.append(identity)
                    pair.append((FineGridCache(cache,affine),spec))
                opened[key]=pair
            (source,source_spec),(target,target_spec)=opened[key]
            metadata['scientific_work_launched'] = not fixture
            try:
                row,traces,maps=cycle_outcome(query,source,target,args.method,retriever,
                    source_spec.get('admissible_domain'),target_spec.get('admissible_domain'),tuple(args.margin),
                    args.peak_count,args.peak_separation_mm,args.peak_score_gap)
            except (MemoryError, RuntimeError) as exc:
                if isinstance(exc,RuntimeError) and not is_cuda_out_of_memory(exc): raise
                blocked='cuda_out_of_memory' if is_cuda_out_of_memory(exc) else str(exc); break
            rows.append(row)
            if traces:
                trace_dir=staging/'diagnostics'/query['query_id'].replace(':','_')
                row['anchor_trace_file']=pack_anchor_traces(trace_dir,traces).relative_to(staging).as_posix()
                added=sum(p.stat().st_size for p in trace_dir.iterdir() if p.is_file())
                staging_bytes+=added;diagnostic_bytes+=added
            if query['query_id'] in selected:
                for index,view in enumerate(maps):
                    spec=target_spec if view['direction']=='forward' else source_spec
                    ct=None
                    if spec.get('ct_path'):
                        import nibabel as nib
                        ct_path=Path(spec['ct_path'])
                        if spec.get('ct_sha256')!=cohort.sha256_file(ct_path): raise cohort.CohortError('Diagnostic CT identity changed')
                        ct=nib.load(str(ct_path))
                    def guard_view(required_bytes):
                        if diagnostic_bytes+required_bytes>args.diagnostic_budget_bytes:
                            raise MemoryError('diagnostic_view_reservation_budget')
                        if shutil.disk_usage(root).free<args.min_free_bytes+required_bytes:
                            raise MemoryError('diagnostic_view_reservation_disk')
                    try:
                        path=export_similarity_views(staging/'similarity_views'/view['direction'],query['query_id'],
                            view['source'],view['target'],view['point_fine'],retriever,view['map_role'],view['domain'],ct,view['center_fine'],
                            resource_guard=guard_view)
                    except (MemoryError,RuntimeError) as exc:
                        if isinstance(exc,RuntimeError) and not is_cuda_out_of_memory(exc): raise
                        blocked='cuda_out_of_memory_diagnostic_export' if is_cuda_out_of_memory(exc) else str(exc)
                        row['diagnostic_capture_status']='incomplete_resource_stop';break
                    row.setdefault('similarity_view_files',[]).append(path.relative_to(staging).as_posix())
                    added=sum(p.stat().st_size for p in path.parent.glob(path.stem+'.*') if p.is_file())
                    staging_bytes+=added;diagnostic_bytes+=added
                if blocked: break
            # Seal bounded batches, avoiding whole-cohort CSV/hash work per query.
            # A resource guard always seals the current partial batch at exit.
            metadata.update(elapsed_seconds=time.monotonic()-started,cache_identities=cache_identities)
            batch_attempts+=1
            if batch_attempts >= args.checkpoint_queries:
                write_method_bundle(args.contract,root,args.method,rows,metadata,resume=args.resume or (root/'output_inventory.json').exists(),artifact_source=staging)
                batch_attempts=0
        if not blocked and args.select_upper_tail is not None:
            # Select from completed cycle outcomes while the original cache handles
            # are still open. No IDs need to be known before scientific execution.
            from collections import defaultdict
            from tools.quadra.reviewed_uae_export import export_reviewed_views
            cells=defaultdict(list)
            for row in rows:
                if row['status']=='success': cells[(row['subject_id'],row['mask_name'])].append(row)
            selected_after=[]
            for cell,values in sorted(cells.items()):
                errors=sorted(float(row['cycle_error_mm']) for row in values)
                count=int(np.ceil(len(errors)*args.select_upper_tail/100.))
                cutoff=errors[-count]
                selected_after.extend(dict(query_id=row['query_id'],subject_id=row['subject_id'],mask_name=row['mask_name'],
                    cycle_error_mm=float(row['cycle_error_mm']),cutoff_mm=cutoff,selection_reason='postcycle_upper_tail')
                    for row in values if float(row['cycle_error_mm'])>=cutoff)
            metadata['postcycle_selection']=dict(percent=args.select_upper_tail,
                policy='ceil_fraction_sorted_cutoff_include_all_ties',query_ids=[r['query_id'] for r in selected_after])
            if selected_after:
                cohort.atomic_csv(staging/'postcycle_selection.csv',selected_after)
                write_method_bundle(args.contract,root,args.method,rows,metadata,resume=args.resume or (root/'output_inventory.json').exists(),artifact_source=staging)
                hook_args=SimpleNamespace(contract=args.contract,run_directory=str(root),cache_index=args.cache_index,
                    output_directory=str(staging/'postcycle_views'),queue_file=None,query_id=[r['query_id'] for r in selected_after],
                    backend=None,device=None,reconstruction_reason=None,cache_root=None,ct_root=None,anchor_index=0,
                    budget_bytes=max(1,args.diagnostic_budget_bytes-diagnostic_bytes),min_free_bytes=args.min_free_bytes)
                result=export_reviewed_views(hook_args,cache_pairs=opened)
                if result: blocked='postcycle_similarity_export_resource_guard'
                added=sum(p.stat().st_size for p in (staging/'postcycle_views').rglob('*') if p.is_file())
                staging_bytes+=added;diagnostic_bytes+=added
        if blocked:
            metadata['resource_stop']=blocked
            metadata['technical_gate']='blocked_resource'
            cohort.atomic_json(staging/'resource_guard.json',dict(reason=blocked,query_id=query['query_id'],
                preserved_attempted_queries=len(rows),remaining_queries=len(queries)-len(rows),silent_thinning=False))
        metadata.update(elapsed_seconds=time.monotonic()-started,cache_identities=cache_identities,
                        diagnostic_bytes=diagnostic_bytes,diagnostic_staging_bytes=staging_bytes,
                        retained_evidence_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
        metadata['resources']=dict(wall_seconds=metadata['elapsed_seconds'],
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
            peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(args.device)) if args.device!='cpu' else 0,
            retained_diagnostic_bytes=diagnostic_bytes,staging_diagnostic_bytes=staging_bytes,
            cuda_residency=retriever.cuda_profile())
        write_method_bundle(args.contract,root,args.method,rows,metadata,resume=args.resume or (root/'output_inventory.json').exists(),artifact_source=staging)
    finally:
        retriever.close()
        for cache in cache_handles: cache.close()  # close mappings, retain files for anatomical review
        if previous_tf32 is not None: torch.backends.cuda.matmul.allow_tf32=previous_tf32
        # On success all staging files are sealed copies. On exceptions preserve the
        # staging directory as partial diagnostic evidence for explicit recovery.
        if not any(staging.rglob('*')): staging.rmdir()
    return 2 if blocked else 0


def add_commands(subparsers):
    from tools.quadra.reviewed_uae_export import add_commands as add_export_commands
    add_export_commands(subparsers)
    extract=subparsers.add_parser('reviewed-uae-extract',help='Prepare exact frozen pilot group embeddings; no automatic tiled fallback')
    extract.add_argument('--contract',required=True)
    extract.add_argument('--cache-root')
    extract.add_argument('--evidence-directory')
    extract.add_argument('--ct-root')
    extract.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/samv2/samv2_NIHLN.py'))
    extract.add_argument('--checkpoint')
    extract.add_argument('--subject',action='append')
    extract.add_argument('--min-free-bytes',type=int,default=2*1024**3)
    extract.add_argument('--approved-pilot',action='store_true')
    extract.add_argument('--dry-run',action='store_true')
    extract.add_argument('--resume',action='store_true')
    extract.set_defaults(reviewed_handler=extract_reviewed_uae)
    parser=subparsers.add_parser('reviewed-uae-run',help='Fine-grid reviewed UAE matching; real execution bounded to an explicitly approved pilot')
    parser.add_argument('--contract',required=True)
    parser.add_argument('--run-directory',required=True)
    parser.add_argument('--method',required=True,choices=('uae_nn','uae_fixed_point'))
    parser.add_argument('--cache-index',required=True)
    parser.add_argument('--backend',choices=('auto','dense','streamed'),default='auto')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--chunk-locations',type=int,default=32768)
    parser.add_argument('--checkpoint-queries',type=int,default=100)
    parser.add_argument('--dense-budget-bytes',type=int,help='Explicit proposal ceiling; provisional defaults CPU512MiB/CUDA32GiB, actual fit remains a pilot gate')
    parser.add_argument('--diagnostic-budget-bytes',type=int,default=4*1024**3)
    parser.add_argument('--min-free-bytes',type=int,default=2*1024**3)
    parser.add_argument('--peak-count',type=int,default=3)
    parser.add_argument('--peak-separation-mm',type=float,default=8.)
    parser.add_argument('--peak-score-gap',type=float,default=.03)
    parser.add_argument('--margin',nargs=3,type=int,default=[2,2,2])
    parser.add_argument('--selected-query',action='append')
    parser.add_argument('--select-upper-tail',type=float,help='Export this percent per subject-organ after cycle outcomes; all cutoff ties included')
    parser.add_argument('--subject',action='append')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--approved-pilot',action='store_true')
    parser.set_defaults(reviewed_handler=run_reviewed_uae)


def extract_reviewed_uae(args):
    """Reuse the existing aligned dense extractor, and pin its cache identities.

    This prepares only frozen pilot subjects after an explicit launch approval.
    A dense extraction failure preserves state and blocks; extraction changes
    require their own real-data equivalence gate, never an automatic fallback.
    """
    from tools.quadra import aligned_organ_group_cohort as cohort
    from tools.quadra import organ_group_lattice_alignment as lattice
    from tools.quadra.reviewed_matching_contract import read_contract
    contract,queries,signature=read_contract(args.contract)
    pilots=set(contract.get('pilot_subjects',[]))
    subjects=set(args.subject or pilots)
    if args.dry_run:
        print(json.dumps(dict(input_signature=signature,pilot_subjects=sorted(subjects),
            plan_count=sum(Path(name).name.split('-')[0] in subjects for name in contract.get('plan_files',[])),
            extraction='existing_global_lattice_dense_full_group',scientific_work_launched=False,
            pending_gates=['explicit_pilot_approval','resource_feasibility','extraction_context_equivalence']),sort_keys=True))
        return 0
    if contract.get('fixture_only') or not args.approved_pilot:
        raise cohort.CohortError('Real UAE extraction requires explicit approved pilot; fixtures use deterministic cached arrays')
    if not subjects or not subjects <= pilots: raise cohort.CohortError('Extraction is bounded to frozen pilot subjects')
    if any(not getattr(args,name) for name in ('cache_root','evidence_directory','ct_root','checkpoint')):
        raise cohort.CohortError('Extraction requires explicit cache/evidence/CT roots and checkpoint')
    if cohort.sha256_file(args.config)!=contract['model']['config_sha256'] or cohort.sha256_file(args.checkpoint)!=contract['model']['expected_checkpoint_sha256']:
        raise cohort.CohortError('Extraction model assets differ from frozen contract')
    evidence=Path(args.evidence_directory); caches=Path(args.cache_root)
    state_path=evidence/'extraction_state.json';index_path=evidence/'cache_index.json'
    if evidence.exists() and any(evidence.iterdir()) and not args.resume:
        raise cohort.CohortError('Existing extraction evidence requires explicit compatible resume')
    evidence.mkdir(parents=True,exist_ok=True);caches.mkdir(parents=True,exist_ok=True)
    groups={};records=[]
    identity=dict(input_signature=signature,config_sha256=contract['model']['config_sha256'],
                  checkpoint_sha256=contract['model']['expected_checkpoint_sha256'],
                  extraction_implementation_signature=cohort.sha256_payload({file:cohort.sha256_file(Path(__file__).parent/file)
                    for file in ('reviewed_uae_matching.py','organ_group_lattice_alignment.py','memory_configuration_screen.py')}))
    if index_path.exists():
        old=json.loads(index_path.read_text())
        if old.get('identity')!=identity: raise cohort.CohortError('Extraction resume identity changed')
        groups={(g['subject_id'],g['group_name']):g for g in old['groups']}
    if state_path.exists():
        old_state=json.loads(state_path.read_text())
        if any(old_state.get(k)!=v for k,v in identity.items()): raise cohort.CohortError('Extraction state identity changed')
        records=list(old_state.get('outputs',[]))
    state=dict(identity,status='starting',scientific_work_launched=False,extraction_backend='dense_full_group',
               extraction_context_validated=False,cache_retention='through_first_anatomical_review',outputs=records)
    cohort.atomic_json(state_path,state)
    try:
        import torch
        from tools.quadra import memory_configuration_screen as stage3
        torch.manual_seed(stage3.SEED);np.random.seed(stage3.SEED)
        torch.backends.cudnn.benchmark=stage3.CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic=stage3.CUDNN_DETERMINISTIC
        torch.backends.cuda.matmul.allow_tf32=False
        model,hook=stage3._load_model(Path(args.config),Path(args.checkpoint),'fp32')
        if not hook or next(model.parameters()).dtype!=torch.float32: raise cohort.CohortError('FP32 model/architectural output hook contract failed')
        state['scientific_work_launched']=True
        for name in contract['plan_files']:
            plan=json.loads((Path(args.contract)/name).read_text())
            if plan['subject_id'] not in subjects: continue
            key=(plan['subject_id'],plan['group_name']);session=plan['session']
            group=groups.setdefault(key,dict(subject_id=key[0],group_name=key[1]))
            if session in group:
                spec=group[session]
                for file,digest in spec['cache_file_sha256'].items():
                    if cohort.sha256_file(Path(spec['cache_directory'])/file)!=digest: raise cohort.CohortError('Retained extraction cache changed')
                continue
            cache_dir=caches/(key[0]+'-'+session+'-'+key[1])
            if cache_dir.exists(): raise cohort.CohortError('Unindexed cache preserved; recover it explicitly instead of overwriting')
            # Final cache bytes are computable from frozen architecture; reserve
            # conservatively for all three FP16 heads before loading a model crop.
            voxels=int(np.prod(plan['padded_shape_xyz']))
            estimated_cache=128*2*(2*(voxels//8)+(voxels//64))
            if shutil.disk_usage(caches).free < args.min_free_bytes+estimated_cache:
                state.update(status='resource_stopped',failure_reason='extraction_disk_guard',pending_plan=name)
                cohort.atomic_json(state_path,state);return 2
            portable=dict(plan);portable['source_ct']=dict(plan['source_ct'])
            ct_path=Path(args.ct_root)/plan['source_ct']['path']
            portable['source_ct']['path']=str(ct_path)
            cache,resources=lattice._extract_cache(model,portable)
            partial=cache_dir.with_name('.'+cache_dir.name+'.partial');partial.mkdir()
            features={}
            for head in ('fine','coarse','semantic'):
                array=cache.valid_array(head)
                if array.dtype!=np.float16 or not np.isfinite(array).all(): raise cohort.CohortError('Extracted cache violates finite FP16 output contract')
                np.save(partial/(head+'.npy'),array)
                features[head]=dict(file=head+'.npy',channels=int(array.shape[0]),valid_shape_xyz=list(array.shape[:0:-1]))
            affine=np.diag([-1.,-1.,1.,1.]) @ np.asarray(plan['padded_2mm_affine'])
            provenance=dict(input_signature=signature,plan_sha256=cohort.sha256_file(Path(args.contract)/name),source_ct_sha256=plan['source_ct']['sha256'],
                model_config_sha256=identity['config_sha256'],model_checkpoint_sha256=identity['checkpoint_sha256'],
                model_precision='fp32',embedding_dtype='float16',native_to_lps=affine.tolist(),
                preprocessing='global_lattice_2mm_clip_-1024_3071_scale_255_offset_50',extraction_backend='dense_full_group')
            manifest=dict(schema_version=2,complete=True,model_profile='uae_s',native_sam_shape_xyz=plan['padded_shape_xyz'],
                native_spacing_xyz=[2.,2.,2.],norm_ratio_xyz=[1.,1.,1.],features=features,reviewed_provenance=provenance,
                extraction_resources=resources)
            cohort.atomic_json(partial/'manifest.json',manifest)
            partial.rename(cache_dir)
            domain=np.ceil(np.asarray(plan['valid_model_box_xyz'])/2).astype(np.int64).tolist()
            group[session]=dict(cache_directory=str(cache_dir.resolve()),input_signature=signature,
                plan_sha256=provenance['plan_sha256'],admissible_domain=domain,ct_path=str(ct_path.resolve()),ct_sha256=plan['source_ct']['sha256'],
                cache_file_sha256={file:cohort.sha256_file(cache_dir/file) for file in ('manifest.json','fine.npy','coarse.npy','semantic.npy')})
            records.append(dict(plan=name,cache_directory=str(cache_dir.resolve()),resources=resources,retained_bytes=sum(p.stat().st_size for p in cache_dir.iterdir())))
            cohort.atomic_json(index_path,dict(fixture_only=False,complete=False,identity=identity,groups=[groups[k] for k in sorted(groups)]))
            state.update(status='partial',outputs=records);cohort.atomic_json(state_path,state)
            del cache;torch.cuda.empty_cache()
        state.update(status='technical_complete',outputs=records)
        cohort.atomic_json(state_path,state)
        if any(not {'test','retest'} <= set(g) for g in groups.values()): raise cohort.CohortError('Incomplete directional cache group')
        cohort.atomic_json(index_path,dict(fixture_only=False,complete=True,identity=identity,groups=[groups[k] for k in sorted(groups)]))
    except Exception as exc:
        state.update(status='blocked',failure_reason=type(exc).__name__+':'+str(exc))
        cohort.atomic_json(state_path,state)
        raise
    return 0
