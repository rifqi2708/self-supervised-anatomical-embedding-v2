"""Bounded CUDA residency for the declared released FP32 retrieval variant.

Fine and semantic heads are promoted without normalization. Only coarse heads
use the pinned trilinear/align_corners=False and channel-normalization operations.
Transient maps remain in memory; this module never writes dense similarity data.
"""
from collections import OrderedDict
import numpy as np


class ResidentCudaScorer:
    def __init__(self,device,capacity_bytes,max_entries=2):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
        self.torch=torch;self.device=device;self.capacity=int(capacity_bytes);self.max_entries=int(max_entries)
        self.entries=OrderedDict();self.bytes=0;self.loads=0;self.hits=0;self.evictions=0;self.peak_bytes=0
        if self.capacity<=0 or self.max_entries<2:raise ValueError('Invalid CUDA residency bound')

    def profile(self):
        return dict(entries=len(self.entries),max_entries=self.max_entries,resident_bytes=self.bytes,
                    capacity_bytes=self.capacity,payload_loads=self.loads,cache_hits=self.hits,
                    evictions=self.evictions,peak_resident_bytes=self.peak_bytes,
                    numerical_policy='pinned FP32 coarse interpolation/normalization; fine/semantic unchanged')

    def close(self):
        self.entries.clear();self.bytes=0

    def _estimate(self,cache):
        voxels=int(np.prod(cache.shape_xyz))
        return 4*voxels*sum(cache.cache.valid_array(h).shape[0] for h in ('fine','coarse','semantic'))

    def _heads(self,cache):
        torch=self.torch
        if torch.backends.cuda.matmul.allow_tf32:raise RuntimeError('FP32 retrieval requires torch.backends.cuda.matmul.allow_tf32=False')
        if cache in self.entries:
            self.hits+=1;heads,size=self.entries.pop(cache);self.entries[cache]=(heads,size);return heads
        if any(cache.cache.valid_array(h).shape[0]!=128 for h in ('fine','coarse','semantic')):
            raise ValueError('Released dense reference requires original 128-channel heads')
        size=self._estimate(cache)
        if size>self.capacity:raise MemoryError('cuda_resident_payload_budget_exceeded')
        while self.entries and (len(self.entries)>=self.max_entries or self.bytes+size>self.capacity):
            _,(_,old_size)=self.entries.popitem(last=False);self.bytes-=old_size;self.evictions+=1
        heads=[]
        for name in ('fine','coarse','semantic'):
            value=torch.from_numpy(np.array(cache.cache.valid_array(name),copy=True)).to(device=self.device,dtype=torch.float32)
            if name=='coarse':
                value=torch.nn.functional.interpolate(value.unsqueeze(0),tuple(cache.shape_xyz[::-1]),mode='trilinear',align_corners=False)
                value=torch.nn.functional.normalize(value,dim=1)[0]
            heads.append(value)
        self.entries[cache]=(heads,size);self.bytes+=size;self.loads+=1;self.peak_bytes=max(self.peak_bytes,self.bytes)
        return heads

    def _maps(self,source,target,points,box,with_heads=False):
        torch=self.torch
        if self._estimate(source)+(0 if source is target else self._estimate(target))>self.capacity:
            raise MemoryError('cuda_resident_pair_budget_exceeded')
        source_heads=self._heads(source);target_heads=self._heads(target)
        points=np.asarray(points,dtype=np.int64).reshape(-1,3)
        index=torch.as_tensor(points.T.copy(),device=self.device,dtype=torch.long)
        fused=None;head_maps=[]
        for a,b in zip(source_heads,target_heads):
            query=a[:,index[2],index[1],index[0]].transpose(1,0)
            score=torch.einsum('nc,ck->nk',query,b.reshape(b.shape[0],-1))
            fused=score if fused is None else fused+score
            if with_heads:head_maps.append(score)
        fused=fused/3
        shape=tuple(map(int,target.shape_xyz[::-1]))
        def restrict(value):
            return value.view(len(points),*shape)[:,int(box[0,2]):int(box[1,2]),int(box[0,1]):int(box[1,1]),int(box[0,0]):int(box[1,0])].reshape(len(points),-1)
        return restrict(fused),[restrict(h) for h in head_maps]

    def match(self,source,target,points,box,estimated):
        torch=self.torch;points=np.asarray(points,dtype=np.int64).reshape(-1,3)
        query_heads=source.descriptors(points)  # small query-only legacy validity check
        nonempty=np.any(np.column_stack([np.linalg.norm(v,axis=1)>1e-12 for v in query_heads]),axis=1)
        with torch.no_grad():
            fused,_=self._maps(source,target,points,box)
            finite=torch.isfinite(fused)
            safe=torch.where(finite,fused,torch.full_like(fused,-float('inf')))
            indices=torch.argmax(safe,dim=1).cpu().numpy()
            scores=safe.max(dim=1).values.cpu().numpy()
            valid=finite.all(dim=1).cpu().numpy() & nonempty & np.isfinite(scores)
        size=box[1]-box[0]
        xyz=np.column_stack((indices%size[0],(indices//size[0])%size[1],indices//(size[0]*size[1])))+box[0]
        xyz[~np.isfinite(scores)]=0  # legacy matcher never improves its zero-initialized winner
        statuses=['success' if v else 'failed' for v in valid]
        return dict(points_fine_xyz=xyz.tolist(),scores=[float(v) if np.isfinite(v) else None for v in scores],
            statuses=statuses,failure_reasons=[None if v else 'invalid_or_empty_descriptor' for v in valid],
            backend='resident_released_dense_fp32_adapter',estimated_dense_bytes=estimated,searched_target_locations=int(np.prod(size)),
            similarity_formula='mean(fine,normalized_interpolated_coarse,semantic)',tie_policy='first_global_zyx_flat_index',
            score_dtype='float32',admissible_domain=box.tolist(),cuda_residency=self.profile())

    def score_block(self,source,target,points,box,estimated):
        points=np.asarray(points,dtype=np.int64).reshape(-1,3)
        source.descriptors(points)  # preserve out-of-grid exception semantics
        with self.torch.no_grad():fused,heads=self._maps(source,target,points,box,with_heads=True)
        size=box[1]-box[0];linear=np.arange(int(np.prod(size)),dtype=np.int64)
        xyz=np.column_stack((linear%size[0],(linear//size[0])%size[1],linear//(size[0]*size[1])))+box[0]
        return xyz,fused.cpu().numpy(),[h.cpu().numpy() for h in heads],'resident_released_dense_fp32_adapter',estimated

    def spatial_candidates(self,source,target,point,box,separation,count):
        torch=self.torch
        source.descriptors([point])
        with torch.no_grad():
            fused,heads=self._maps(source,target,[point],box,with_heads=True)
            scores=fused[0];allowed=torch.isfinite(scores);size=box[1]-box[0]
            linear=torch.arange(int(np.prod(size)),device=self.device,dtype=torch.long)
            xyz=torch.stack((linear%int(size[0]),(linear//int(size[0]))%int(size[1]),linear//int(size[0]*size[1])),dim=1)
            xyz=xyz+torch.as_tensor(box[0],device=self.device,dtype=torch.long)
            native=(xyz.to(dtype=torch.float64)*2+.5)/torch.as_tensor(target.norm_ratio_xyz,device=self.device,dtype=torch.float64)
            affine=torch.as_tensor(target.native_to_lps,device=self.device,dtype=torch.float64)
            physical=native@affine[:3,:3].T+affine[:3,3]
            # Bound affine roundoff using absolute physical-coordinate magnitudes.
            # A large origin/shear increases the CPU-resolved uncertainty band;
            # it never changes the declared physical separation threshold.
            native_bound=(target.shape_xyz.astype(np.float64)*2+.5)/target.norm_ratio_xyz
            physical_bound=np.abs(target.native_to_lps[:3,:3])@native_bound+np.abs(target.native_to_lps[:3,3])
            roundoff_band=64*np.finfo(np.float64).eps*(1+float(np.linalg.norm(physical_bound)))
            peaks=[]
            for rank in range(count):
                safe=torch.where(allowed,scores,torch.full_like(scores,-float('inf')))
                best,index=torch.max(safe,dim=0)
                if not bool(torch.isfinite(best).item()):break
                i=int(index.item());point_xyz=xyz[i].cpu().numpy()
                # Export the same CPU affine operation used in historical records.
                point_lps=target.lps(target.fine_to_native(point_xyz,quantized=False))[0]
                peaks.append(dict(fine_xyz=point_xyz.tolist(),physical_lps_xyz=point_lps.tolist(),fused_score=float(best.item()),
                    fine_score=float(heads[0][0,i].item()),coarse_score=float(heads[1][0,i].item()),semantic_score=float(heads[2][0,i].item()),
                    rank=rank+1,minimum_separation_mm=separation,search='exact_greedy_exhaustive'))
                distance=torch.linalg.norm(physical-torch.as_tensor(point_lps,device=self.device,dtype=torch.float64),dim=1)
                keep=distance>=separation
                # GPU/CPU FMA can differ at an exact physical spacing boundary.
                # Re-evaluate the narrow uncertainty band with the original CPU
                # absolute-LPS affine/norm, including shear and large origins.
                borderline=torch.where(torch.abs(distance-separation)<=roundoff_band)[0]
                if len(borderline):
                    boundary_xyz=xyz[borderline].cpu().numpy()
                    boundary_lps=target.lps(target.fine_to_native(boundary_xyz,quantized=False))
                    original_keep=np.linalg.norm(boundary_lps-point_lps,axis=1)>=separation
                    keep[borderline]=torch.as_tensor(original_keep,device=self.device,dtype=torch.bool)
                allowed &= keep
        return peaks
