"""Public CUDA matching/compact diagnostic seams, exercised on a real GPU."""
import json
import hashlib
import subprocess
import sys
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.quadra import reviewed_uae_matching as uae
from tools.quadra import reviewed_uae_diagnostics as diagnostics
try:
    import torch
    CUDA=torch.cuda.is_available()
except ImportError:
    CUDA=False


@unittest.skipUnless(CUDA,'Requires the scientific Torch CUDA runtime')
class ReviewedCudaMatchingTests(unittest.TestCase):
    def setUp(self):
        self.previous=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32=self.previous

    def test_public_cli_initializes_cold_cuda_before_recording_peak_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);contract=root/'contract';contract.mkdir()
            queries=contract/'frozen_queries_raw_itk.csv'
            queries.write_text('query_id,subject_id,group_name,mask_name,raw_x,raw_y,raw_z,fine_x,fine_y,fine_z,physical_lps_x,physical_lps_y,physical_lps_z\nq,s,g,organ,2,2,2,2,2,2,2,2,2\n')
            (contract/'matching_contract.json').write_text(json.dumps(dict(schema_version=2,dataset_id='coldfixture',fixture_only=True,
                subjects=['s'],query_count=1,files=[dict(path=queries.name,bytes=queries.stat().st_size,sha256=hashlib.sha256(queries.read_bytes()).hexdigest())])))
            values=np.zeros((128,5,5,5),np.float16);values[:125]=np.eye(125,dtype=np.float16).reshape(125,5,5,5)
            entries={}
            for session in ('test','retest'):
                cache=root/session;cache.mkdir()
                for head in ('fine','coarse','semantic'):np.save(cache/(head+'.npy'),values)
                (cache/'manifest.json').write_text(json.dumps(dict(complete=True,model_profile='uae_s',native_sam_shape_xyz=[5,5,5],
                    norm_ratio_xyz=[2,2,2],native_spacing_xyz=[1,1,1],features={h:dict(file=h+'.npy',valid_shape_xyz=[5,5,5]) for h in ('fine','coarse','semantic')})))
                entries[session]=dict(cache_directory=str(cache),native_to_lps=np.eye(4).tolist())
            index=root/'index.json';index.write_text(json.dumps(dict(fixture_only=True,groups=[dict(subject_id='s',group_name='g',**entries)])))
            command=[sys.executable,'-m','tools.quadra.aligned_organ_group_cohort','reviewed-uae-run','--contract',str(contract),
                '--cache-index',str(index),'--run-directory',str(root/'run'),'--method','uae_nn','--device','cuda:0','--backend','dense',
                '--dense-budget-bytes',str(1024**3),'--peak-count','1','--min-free-bytes','0']
            child=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,env=os.environ.copy())
            self.assertEqual(child.returncode,0,child.stdout)
            manifest=json.loads((root/'run'/'method_manifest.json').read_text())
            self.assertEqual(manifest['successful_queries'],1)
            self.assertGreater(manifest['resources']['peak_gpu_memory_bytes'],0)

    def test_dense_matches_pinned_original_maps_and_bounds_pair_residency(self):
        rng=np.random.RandomState(19)
        def head(shape):
            value=rng.normal(size=shape).astype(np.float32)
            value/=np.maximum(np.linalg.norm(value,axis=0,keepdims=True),1e-12)
            return value.astype(np.float16)
        a=uae.FineGridCache.from_arrays(head((128,5,7,6)),head((128,3,4,3)),head((128,5,7,6)))
        b=uae.FineGridCache.from_arrays(head((128,6,5,7)),head((128,3,3,4)),head((128,6,5,7)))
        points=[[1,2,3],[4,5,2]];domain=[[1,1,1],[6,5,6]]
        original=uae.released_dense_scores(a,b,points,'cuda:0')[:,1:6,1:5,1:6].reshape(2,-1)
        matcher=uae.FineGridRetriever('dense',31,1024**3,'cuda:0')
        actual=matcher.match(a,b,points,domain)
        expected=[]
        for row in original:
            z,y,x=np.unravel_index(int(row.argmax()),(5,4,5));expected.append([x+1,y+1,z+1])
        self.assertEqual(actual['points_fine_xyz'],expected)
        np.testing.assert_allclose(actual['scores'],original.max(axis=1),atol=1e-6,rtol=1e-5)
        observed=np.concatenate([block[1] for block in matcher.score_blocks(a,b,points,domain)],axis=1)
        np.testing.assert_allclose(observed,original,atol=1e-6,rtol=1e-5)
        repeated=matcher.match(a,b,points,domain)
        profile=repeated['cuda_residency']
        self.assertEqual(profile['entries'],2)
        self.assertEqual(profile['payload_loads'],2)
        self.assertGreater(profile['cache_hits'],0)
        self.assertLessEqual(profile['resident_bytes'],profile['capacity_bytes'])
        c=uae.FineGridCache.from_arrays(head((128,5,7,6)),head((128,3,4,3)),head((128,5,7,6)))
        switched=matcher.match(b,c,[[1,2,3]])['cuda_residency']
        self.assertEqual(switched['entries'],2)
        self.assertEqual(switched['payload_loads'],3)
        matcher.close()

    def test_compact_gpu_peaks_keep_legacy_physical_boundary_and_stable_ties(self):
        theta=.1;affine=np.eye(4)
        affine[:3,:3]=2*np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        affine[:3,3]=[-123.45,-245.67,-30.12]
        query=np.zeros((128,1,1,1),np.float16);query[0]=1
        values=np.zeros((128,5,5,5),np.float16)
        for xyz,score in [([2,2,2],1),([4,2,2],.95),([0,2,2],.95),([2,4,2],.9)]:
            x,y,z=xyz;values[0,z,y,x]=score
        a=uae.FineGridCache.from_arrays(query,query,query,norm_ratio=(1,1,1),native_to_lps=affine)
        b=uae.FineGridCache.from_arrays(values,values,values,norm_ratio=(1,1,1),native_to_lps=affine)
        expected=diagnostics.spatial_peak_candidates(a,b,[0,0,0],uae.FineGridRetriever('dense'),minimum_separation_mm=8.,count=3)
        matcher=uae.FineGridRetriever('dense',31,1024**3,'cuda:0')
        actual=diagnostics.spatial_peak_candidates(a,b,[0,0,0],matcher,minimum_separation_mm=8.,count=3)
        self.assertEqual([p['fine_xyz'] for p in actual],[p['fine_xyz'] for p in expected])
        self.assertEqual([p['physical_lps_xyz'] for p in actual],[p['physical_lps_xyz'] for p in expected])
        for observed,reference in zip(actual,expected):
            for key in ['fused_score','fine_score','coarse_score','semantic_score']:
                self.assertAlmostEqual(observed[key],reference[key],places=6)
        matcher.close()
        for matrix,origin in [(np.array([[2.,.25,0],[.5,3.,.125],[0,0,1.5]]),[1e12,-1e12,1e12]),
                              (affine[:3,:3],[1e16,-1e16,1e16])]:
            stress=affine.copy();stress[:3,:3]=matrix;stress[:3,3]=origin
            a=uae.FineGridCache.from_arrays(query,query,query,norm_ratio=(1,1,1),native_to_lps=stress)
            b=uae.FineGridCache.from_arrays(values,values,values,norm_ratio=(1,1,1),native_to_lps=stress)
            expected=diagnostics.spatial_peak_candidates(a,b,[0,0,0],uae.FineGridRetriever('dense'),minimum_separation_mm=8.,count=3)
            matcher=uae.FineGridRetriever('dense',31,1024**3,'cuda:0')
            actual=diagnostics.spatial_peak_candidates(a,b,[0,0,0],matcher,minimum_separation_mm=8.,count=3)
            self.assertEqual([p['fine_xyz'] for p in actual],[p['fine_xyz'] for p in expected])
            self.assertEqual([p['physical_lps_xyz'] for p in actual],[p['physical_lps_xyz'] for p in expected])
            matcher.close()

    def test_nonfinite_and_empty_queries_keep_legacy_failures_and_roi_winners(self):
        values=np.zeros((128,1,1,4),np.float16);values[0]=1
        source=uae.FineGridCache.from_arrays(values,values,values)
        bad=values.copy();bad[0,0,0,2]=np.nan
        target=uae.FineGridCache.from_arrays(bad,values,values)
        empty=np.zeros_like(values)
        scenarios=[(source,target),(uae.FineGridCache.from_arrays(bad,values,values),source),
                   (uae.FineGridCache.from_arrays(empty,empty,empty),source)]
        matcher=uae.FineGridRetriever('dense',31,1024**3,'cuda:0')
        for a,b in scenarios:
            point=[[2,0,0]] if a is not source and a.cache.valid_array('fine')[0,0,0,2]!=0 else [[0,0,0]]
            old=uae.FineGridRetriever('dense').match(a,b,point,[[1,0,0],[4,1,1]])
            actual=matcher.match(a,b,point,[[1,0,0],[4,1,1]])
            self.assertEqual(actual['statuses'],old['statuses'])
            self.assertEqual(actual['points_fine_xyz'],old['points_fine_xyz'])
            self.assertEqual(actual['scores'],old['scores'])
        malformed=np.ones((1,1,1,1),np.float16)
        cache=uae.FineGridCache.from_arrays(malformed,malformed,malformed)
        with self.assertRaisesRegex(ValueError,'128-channel'):
            matcher.match(cache,cache,[[0,0,0]])
        matcher.close()

    def test_fixed_point_traces_and_selected_context_peak_planes_remain_visible(self):
        value=np.zeros((128,5,5,5),np.float16);value[:125]=np.eye(125,dtype=np.float16).reshape(125,5,5,5)
        reflected=np.flip(np.swapaxes(value,2,3),axis=3).copy()
        a=uae.FineGridCache.from_arrays(value,value,value)
        b=uae.FineGridCache.from_arrays(reflected,reflected,reflected)
        matcher=uae.FineGridRetriever('dense',31,1024**3,'cuda:0')
        result=uae.fixed_point(a,b,[1,2,2],matcher,margin=(1,1,1))
        reference=uae.released_reference(a,b,[1,2,2],uae.FineGridRetriever('dense'),margin=(1,1,1))
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['point_native_xyz'],reference['point_native_xyz'])
        self.assertEqual(result['anchor_history'],reference['anchor_history'])
        self.assertEqual(result['filter_keep'],reference['filter_keep'])
        with tempfile.TemporaryDirectory() as directory:
            path=diagnostics.export_similarity_views(Path(directory),'fixture',a,a,[3,3,3],matcher,'seed',center_fine=[2,2,2])
            metadata=json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(metadata['global_maximum_fine_xyz'],[3,3,3])
            with np.load(path,allow_pickle=False) as planes:
                self.assertEqual(float(np.nanmax(planes['peak_axial'])),1.)
                self.assertEqual(float(np.nanmax(planes['context_axial'])),0.)
        matcher.close()


if __name__=='__main__':unittest.main()
