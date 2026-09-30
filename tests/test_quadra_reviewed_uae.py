import json
import tempfile
import unittest
import csv
import hashlib
from pathlib import Path

import numpy as np
import nibabel as nib

from tools.quadra import reviewed_uae_matching as uae


class ReviewedUaeContractTests(unittest.TestCase):
    def test_oblique_half_integer_query_uses_frozen_model_coordinates(self):
        theta=.1
        affine=np.eye(4);affine[:3,:3]=2*np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        affine[:3,3]=[-123.45,-245.67,-30.12]
        values=np.eye(125,dtype=np.float16).reshape(125,5,5,5)
        cache=uae.FineGridCache.from_arrays(values,values,values,norm_ratio=(1.,1.,1.),native_to_lps=affine)
        point=np.array([1.5,2.,2.]);physical=cache.lps(point)[0]
        query=dict(query_id='oblique',subject_id='s',group_name='head',mask_name='brain')
        for axis,m,p,f in zip('xyz',point,physical,[1,1,1]):
            query['model_continuous_'+axis]=m;query['physical_lps_'+axis]=p;query['fine_'+axis]=f
        result,_,_=uae.cycle_outcome(query,cache,cache,'uae_nn',uae.FineGridRetriever(),peak_count=1)
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['source_native_xyz'],[2,2,2])
    def test_coarse_half_pixel_interpolation_has_known_head_mean_score(self):
        fine=np.ones((1,1,1,4),dtype=np.float16)
        query_coarse=np.array([1.,0.],dtype=np.float16).reshape(2,1,1,1)
        target_coarse=np.array([[1.,0.],[0.,1.]],dtype=np.float16).reshape(2,1,1,2)
        query=uae.FineGridCache.from_arrays(fine,query_coarse,fine)
        target=uae.FineGridCache.from_arrays(fine,target_coarse,fine)
        result=uae.FineGridRetriever().match(query,target,[[0,0,0]],[[1,0,0],[2,1,1]])
        # Coarse sample = normalized [.75,.25]; head mean = (1+.9486833+1)/3.
        self.assertAlmostEqual(result['scores'][0],.98289443,places=6)

    def test_fine_grid_matching_preserves_domains_and_global_tie_order(self):
        # First dense ZYX location wins a tie, even when chunks traverse X first.
        target = np.zeros((2, 1, 2, 3), np.float16)
        target[0, 0, 0, 2] = target[0, 0, 1, 0] = 1
        query = np.zeros_like(target); query[0, 0, 0, 0] = 1
        a = uae.FineGridCache.from_arrays(query, query, query)
        b = uae.FineGridCache.from_arrays(target, target, target)
        for backend in ('dense', 'streamed'):
            match = uae.FineGridRetriever(backend=backend, chunk_locations=2).match(a, b, [[0, 0, 0]])
            self.assertEqual(match['points_fine_xyz'], [[2, 0, 0]])
            restricted = uae.FineGridRetriever(backend=backend, chunk_locations=2).match(
                a, b, [[0, 0, 0]], [[0, 1, 0], [3, 2, 1]])
            self.assertEqual(restricted['points_fine_xyz'], [[0, 1, 0]])
            self.assertEqual(restricted['searched_target_locations'], 3)

    def test_fixed_point_matches_independent_released_success_and_trace(self):
        # Identity descriptors create a nondegenerate 3D affine fit.
        fine = np.eye(125, dtype=np.float16).reshape(125, 5, 5, 5)
        cache = uae.FineGridCache.from_arrays(fine, fine, fine)
        matcher = uae.FineGridRetriever(backend='streamed', chunk_locations=13)
        result = uae.fixed_point(cache, cache, [2, 2, 2], matcher, margin=(1, 1, 1))
        reference = uae.released_reference(cache, cache, [2, 2, 2], matcher, margin=(1, 1, 1))
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['point_native_xyz'], reference['point_native_xyz'])
        self.assertEqual(result['anchor_history'], reference['anchor_history'])
        self.assertEqual(result['filter_keep'], reference['filter_keep'])
        self.assertEqual(result['settings']['return_distance_normalized_units'], 100)
        self.assertEqual(result['settings']['return_distance_mm_default_normalization'], 200)

    def test_fixed_point_failure_is_not_nearest_neighbour_substitution(self):
        fine = np.ones((1, 1, 1, 1), np.float16)
        cache = uae.FineGridCache.from_arrays(fine, fine, fine)
        result = uae.fixed_point(cache, cache, [0, 0, 0], uae.FineGridRetriever())
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['point_native_xyz'])
        self.assertIn('insufficient', result['failure_reason'])

    def test_translated_fixed_point_and_coarse_resampling_agree_across_backends(self):
        fine=np.eye(125,dtype=np.float16).reshape(125,5,5,5)
        shifted=np.roll(fine,1,axis=3)
        # Coarse differs in resolution, exercising align_corners=False interpolation
        # and descriptor normalization rather than an equal-grid shortcut.
        coarse=np.zeros((125,2,3,2),np.float16);coarse[0]=1
        source=uae.FineGridCache.from_arrays(fine,coarse,fine)
        target=uae.FineGridCache.from_arrays(shifted,coarse,shifted)
        dense=uae.FineGridRetriever('dense')
        streamed=uae.FineGridRetriever('streamed',chunk_locations=7)
        outputs=[]
        for matcher in (dense,streamed):
            result=uae.fixed_point(source,target,[2,2,2],matcher,margin=(1,1,1))
            reference=uae.released_reference(source,target,[2,2,2],matcher,margin=(1,1,1))
            self.assertEqual(result['status'],'success')
            self.assertEqual(result['point_native_xyz'],reference['point_native_xyz'])
            self.assertEqual(result['anchor_history'],reference['anchor_history'])
            outputs.append(result)
        self.assertEqual(outputs[0]['point_native_xyz'],outputs[1]['point_native_xyz'])
        self.assertEqual(outputs[0]['filter_keep'],outputs[1]['filter_keep'])

    def test_permuted_reflected_3d_and_planar_predictions_match_released_order(self):
        for z_size,point in [(5,[1,2,2]),(1,[1,2,0])]:
            fine=np.eye(25*z_size,dtype=np.float16).reshape(25*z_size,z_size,5,5)
            reflected=np.flip(np.swapaxes(fine,2,3),axis=3)
            source=uae.FineGridCache.from_arrays(fine,fine,fine)
            target=uae.FineGridCache.from_arrays(reflected,reflected,reflected)
            matcher=uae.FineGridRetriever('streamed',chunk_locations=11)
            actual=uae.fixed_point(source,target,point,matcher,margin=(1,1,1))
            original=uae.released_reference(source,target,point,matcher,margin=(1,1,1))
            self.assertEqual(actual['status'],'success')
            self.assertEqual(actual['point_native_xyz'],original['point_native_xyz'])
            self.assertEqual(actual['anchor_history'],original['anchor_history'])
            self.assertEqual(actual['filter_keep'],original['filter_keep'])
            self.assertEqual(actual['fit']['mode'],'planar' if z_size==1 else '3d')

    def test_diagnostics_find_separated_peaks_and_decode_anchor_records(self):
        from tools.quadra import reviewed_uae_diagnostics as diagnostics
        values = np.zeros((1, 1, 1, 6), dtype=np.float16)
        values[0, 0, 0] = [1, .99, 0, .95, 0, .9]
        source = uae.FineGridCache.from_arrays(np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)))
        target = uae.FineGridCache.from_arrays(values, values, values)
        peaks = diagnostics.spatial_peak_candidates(source, target, [0, 0, 0], uae.FineGridRetriever(chunk_locations=2), minimum_separation_mm=2.5, count=2)
        self.assertEqual([p['fine_xyz'] for p in peaks], [[0, 0, 0], [3, 0, 0]])
        for invalid in (float('nan'),float('inf'),0.,-1.):
            with self.assertRaisesRegex(ValueError,'Invalid peak'):
                diagnostics.spatial_peak_candidates(source,target,[0,0,0],uae.FineGridRetriever(),minimum_separation_mm=invalid)
        fine = np.eye(125, dtype=np.float16).reshape(125, 5, 5, 5)
        cache = uae.FineGridCache.from_arrays(fine, fine, fine)
        result = uae.fixed_point(cache, cache, [2, 2, 2], uae.FineGridRetriever(), margin=(1, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = diagnostics.pack_anchor_traces(Path(directory), [('q1', 'forward', result)])
            decoded = diagnostics.decode_anchor_traces(path)
            self.assertEqual(decoded[0]['anchor_history'], result['anchor_history'])
            self.assertEqual(decoded[0]['filter_keep'], result['filter_keep'])

    def test_selected_map_rejects_invalid_query_and_preserves_stable_finite_peak(self):
        from tools.quadra.reviewed_uae_diagnostics import export_similarity_views
        source_values=np.ones((1,1,1,1),dtype=np.float16)
        source=uae.FineGridCache.from_arrays(source_values,source_values,source_values)
        values=np.zeros((1,1,2,3),dtype=np.float16);values[0,0,0,2]=values[0,0,1,0]=1
        target=uae.FineGridCache.from_arrays(values,values,values)
        matcher=uae.FineGridRetriever('streamed',chunk_locations=2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=export_similarity_views(root,'tie',source,target,[0,0,0],matcher,'seed',center_fine=[1,1,0])
            metadata=json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(metadata['global_maximum_fine_xyz'],[2,0,0])
            self.assertEqual(metadata['global_maximum_score'],1.)
            empty=np.zeros_like(source_values);empty_cache=uae.FineGridCache.from_arrays(empty,empty,empty)
            for label,cache,point,domain,reason in [
                ('empty',empty_cache,[0,0,0],None,'invalid_or_empty_query_descriptor'),
                ('outside',source,[1,0,0],None,'query_outside_fine_grid'),
                ('domain',source,[0,0,0],[[0,0,0],[4,2,1]],'invalid_admissible_domain')]:
                with self.assertRaisesRegex(ValueError,reason):
                    export_similarity_views(root,label,cache,target,point,matcher,'seed',domain=domain)
                self.assertFalse((root/(label+'-seed.npz')).exists())
            nonfinite=np.full_like(values,np.nan);bad=uae.FineGridCache.from_arrays(nonfinite,nonfinite,nonfinite)
            with self.assertRaisesRegex(ValueError,'no_finite_target_score'):
                export_similarity_views(root,'nonfinite',source,bad,[0,0,0],matcher,'seed')
            self.assertFalse((root/'nonfinite-seed.npz').exists())

    def test_cohort_adapter_exports_every_query_cycle_and_delayed_map(self):
        from tools.quadra import aligned_organ_group_cohort as cohort
        from tools.quadra.reviewed_evidence import read_method_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); contract = root/'contract'; contract.mkdir()
            queries = contract/'frozen_queries_raw_itk.csv'
            queries.write_text('query_id,subject_id,group_name,mask_name,raw_x,raw_y,raw_z,fine_x,fine_y,fine_z,physical_lps_x,physical_lps_y,physical_lps_z\nq1,s1,head,brain,2,2,2,2,2,2,2,2,2\nq2,s1,head,brain,9,2,2,9,2,2,9,2,2\n')
            (contract/'matching_contract.json').write_text(json.dumps(dict(schema_version=2,dataset_id='fixture', fixture_only=True,
                subjects=['s1'], query_count=2, files=[dict(path=queries.name,bytes=queries.stat().st_size,sha256=hashlib.sha256(queries.read_bytes()).hexdigest())])))
            fine = np.eye(125,dtype=np.float16).reshape(125,5,5,5)
            ct_path=root/'fixture-ct.nii.gz'
            image=nib.Nifti1Image(np.arange(125,dtype=np.int16).reshape(5,5,5),np.diag([-1.,-1.,1.,1.]))
            image.header.set_xyzt_units('mm');nib.save(image,str(ct_path))
            caches = {}
            for session in ('test','retest'):
                path = root/session; path.mkdir()
                for head in ('fine','coarse','semantic'): np.save(path/(head+'.npy'), fine)
                (path/'manifest.json').write_text(json.dumps(dict(complete=True,model_profile='uae_s',native_sam_shape_xyz=[5,5,5],
                    norm_ratio_xyz=[2,2,2],native_spacing_xyz=[1,1,1],features={h:dict(file=h+'.npy',valid_shape_xyz=[5,5,5]) for h in ('fine','coarse','semantic')})))
                caches[session] = dict(cache_directory=str(path),native_to_lps=np.eye(4).tolist(),ct_path=str(ct_path),ct_sha256=hashlib.sha256(ct_path.read_bytes()).hexdigest())
            config = root/'caches.json';config.write_text(json.dumps(dict(fixture_only=True, groups=[dict(subject_id='s1',group_name='head',**caches)])))
            self.assertEqual(cohort.main(['reviewed-uae-extract','--contract',str(contract),'--dry-run']),0)
            self.assertEqual(cohort.main(['reviewed-uae-extract','--contract',str(contract)]),3)
            for option,value in [('--peak-separation-mm','nan'),('--peak-separation-mm','inf'),('--peak-score-gap','nan'),('--peak-score-gap','inf'),('--peak-score-gap','-1')]:
                invalid_run=root/('invalid-'+option.strip('-')+'-'+value)
                self.assertEqual(cohort.main(['reviewed-uae-run','--contract',str(contract),'--run-directory',str(invalid_run),
                    '--method','uae_nn','--cache-index',str(config),option,value]),3)
                self.assertFalse(invalid_run.exists())
            run = root/'run'
            result = cohort.main(['reviewed-uae-run','--contract',str(contract),'--run-directory',str(run),'--method','uae_nn',
                                  '--cache-index',str(config),'--peak-count','2','--selected-query','q1'])
            self.assertEqual(result,0)
            manifest, rows = read_method_bundle(run,contract)
            self.assertEqual(manifest['attempted_queries'],2)
            self.assertEqual(rows[0]['status'],'success')
            self.assertEqual(float(rows[0]['cycle_error_mm']),0)
            self.assertEqual(rows[1]['status'],'failed')
            self.assertTrue(list((run/'similarity_views').rglob('*.npz')))
            self.assertTrue(list((run/'similarity_views').rglob('*.png')))
            self.assertEqual(manifest['cache_policy'],'retain_until_first_anatomical_review')
            guard_run=root/'guard-run'
            guard=['reviewed-uae-run','--contract',str(contract),'--run-directory',str(guard_run),'--method','uae_nn',
                   '--cache-index',str(config),'--min-free-bytes','999999999999999']
            self.assertEqual(cohort.main(guard),2)
            guard_manifest,guard_rows=read_method_bundle(guard_run,contract)
            self.assertEqual(guard_manifest['status'],'partial')
            self.assertEqual([r['status'] for r in guard_rows],['pending','pending'])
            self.assertTrue((guard_run/'resource_guard.json').is_file())
            # Two successful FP queries both add trace files; late selection must
            # survive a sealed checkpoint and compatible resume.
            second = queries.read_text().replace('9,2,2,9,2,2,9,2,2','3,2,2,3,2,2,3,2,2')
            second = second.replace('q2,s1,head,brain,3,2,2,3,2,2,3,2,2','q2,s1,head,brain,3,3,3,3,3,3,3.25,3,3')
            queries.write_text(second)
            manifest_contract=json.loads((contract/'matching_contract.json').read_text())
            manifest_contract['files'][0].update(bytes=queries.stat().st_size,sha256=hashlib.sha256(queries.read_bytes()).hexdigest())
            (contract/'matching_contract.json').write_text(json.dumps(manifest_contract))
            command=['reviewed-uae-run','--contract',str(contract),'--run-directory',str(root/'fp-run'),'--method','uae_fixed_point',
                     '--cache-index',str(config),'--peak-count','2','--margin','1','1','1','--checkpoint-queries','1']
            self.assertEqual(cohort.main(command),0)
            _,fp_rows=read_method_bundle(root/'fp-run',contract)
            self.assertEqual([r['status'] for r in fp_rows],['success','success'])
            self.assertEqual(cohort.main(command+['--resume']),0)
            self.assertEqual(len(list((root/'fp-run'/'diagnostics').rglob('anchor_traces.npz'))),2)
            # Choose a case only after outcomes exist; reconstruct its retained
            # seed, reverse query and actual anchors without primary rerunning.
            before=hashlib.sha256((root/'fp-run'/'query_outcomes.csv').read_bytes()).hexdigest()
            chosen=max(fp_rows,key=lambda row:float(row['cycle_error_mm']))['query_id']
            queue=root/'late-queue.csv';queue.write_text('query_id\n'+chosen+'\n')
            export=root/'late-export'
            self.assertEqual(cohort.main(['reviewed-uae-export-views','--contract',str(contract),'--run-directory',str(root/'fp-run'),
                '--cache-index',str(config),'--output-directory',str(export),'--queue-file',str(queue)]),0)
            self.assertEqual(hashlib.sha256((root/'fp-run'/'query_outcomes.csv').read_bytes()).hexdigest(),before)
            exported=json.loads((export/'export_manifest.json').read_text())
            self.assertFalse(exported['source_matching_rerun'])
            self.assertEqual(exported['status'],'complete')
            self.assertEqual(len(exported['records'][0]['views']),4)
            seed=export/'forward'/(chosen+'-seed.npz')
            seed_metadata=json.loads(seed.with_suffix('.json').read_text())
            self.assertEqual(seed_metadata['global_maximum_fine_xyz'],[3,3,3])
            self.assertTrue(all(a!=b for a,b in zip(seed_metadata['global_maximum_fine_xyz'],seed_metadata['plane_center_fine_xyz'])))
            self.assertEqual(seed_metadata['scoring_passes'],2)
            self.assertEqual(seed_metadata['png_row_roles'],['corrected_context','global_maximum'])
            self.assertTrue(seed.with_suffix('.png').is_file())
            with np.load(seed,allow_pickle=False) as planes:
                for name in ('axial','coronal','sagittal'):
                    self.assertEqual(float(np.nanmax(planes['peak_'+name])),1.)
                    self.assertEqual(float(np.nanmax(planes['context_'+name])),0.)
                    np.testing.assert_array_equal(planes[name],planes['context_'+name])
                    self.assertIn('ct_peak_'+name,planes.files)
                    self.assertAlmostEqual(float(planes['ct_peak_'+name][3,3]),100.75,places=5)
            self.assertEqual(cohort.main(['reviewed-uae-export-views','--contract',str(contract),'--run-directory',str(root/'fp-run'),
                '--cache-index',str(config),'--output-directory',str(root/'late-export-guard'),'--query-id',chosen,'--budget-bytes','1']),2)
            stopped=json.loads((root/'late-export-guard'/'export_manifest.json').read_text())
            self.assertEqual(stopped['status'],'resource_stopped')
            self.assertTrue((root/'late-export-guard'/'export_inventory.json').is_file())
            reserved=root/'late-export-reservation'
            self.assertEqual(cohort.main(['reviewed-uae-export-views','--contract',str(contract),'--run-directory',str(root/'fp-run'),
                '--cache-index',str(config),'--output-directory',str(reserved),'--query-id',chosen,'--budget-bytes','100000']),2)
            reserve_state=json.loads((reserved/'export_manifest.json').read_text())
            self.assertEqual(reserve_state['status'],'resource_stopped')
            self.assertIn('reservation_budget',reserve_state['failure_reason'])
            self.assertFalse(list(reserved.rglob('*.npz')))
            self.assertTrue((reserved/'export_inventory.json').is_file())
            tail=root/'tail-run'
            self.assertEqual(cohort.main(['reviewed-uae-run','--contract',str(contract),'--run-directory',str(tail),'--method','uae_nn',
                '--cache-index',str(config),'--peak-count','1','--select-upper-tail','5']),0)
            tail_manifest,_=read_method_bundle(tail,contract)
            self.assertEqual(tail_manifest['postcycle_selection']['query_ids'],['q2'])
            self.assertTrue((tail/'postcycle_views'/'export_manifest.json').is_file())
            # A fixture retrieval adapter emulates Torch 1.9 RuntimeError OOM
            # after one complete cycle, exercising sealed partial CLI evidence.
            class ResourceLimited(uae.FineGridRetriever):
                calls=0
                def match(self,*args,**kwargs):
                    self.calls+=1
                    if self.calls==3: raise RuntimeError('CUDA out of memory. Tried to allocate fixture bytes')
                    return super().match(*args,**kwargs)
            oom=root/'oom-run'
            args=cohort.build_parser().parse_args(['reviewed-uae-run','--contract',str(contract),'--run-directory',str(oom),
                '--method','uae_nn','--cache-index',str(config),'--peak-count','1'])
            self.assertEqual(uae.run_reviewed_uae(args,retrieval_factory=ResourceLimited),2)
            oom_manifest,oom_rows=read_method_bundle(oom,contract)
            self.assertEqual([r['status'] for r in oom_rows],['success','pending'])
            self.assertEqual(oom_manifest['resource_stop'],'cuda_out_of_memory')

    def test_empty_descriptor_and_degenerate_geometry_remain_explicit_failures(self):
        empty=np.zeros((2,2,2,2),dtype=np.float16)
        cache=uae.FineGridCache.from_arrays(empty,empty,empty)
        result=uae.FineGridRetriever().match(cache,cache,[[0,0,0]])
        self.assertEqual(result['statuses'],['failed'])
        self.assertEqual(result['failure_reasons'],['invalid_or_empty_descriptor'])
        line=np.eye(5,dtype=np.float16).reshape(5,1,1,5)
        cache=uae.FineGridCache.from_arrays(line,line,line)
        failed=uae.fixed_point(cache,cache,[2,0,0],uae.FineGridRetriever())
        self.assertEqual(failed['status'],'failed')
        self.assertIn('degenerate',failed['failure_reason'])
        self.assertIsNone(failed['point_native_xyz'])

    def test_cuda_runtime_oom_is_not_a_structural_match_failure(self):
        class OomRetriever(uae.FineGridRetriever):
            def match(self,*args,**kwargs): raise RuntimeError('CUDA out of memory. Torch 1.9 fixture')
        fine=np.eye(125,dtype=np.float16).reshape(125,5,5,5)
        cache=uae.FineGridCache.from_arrays(fine,fine,fine)
        with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):
            uae.fixed_point(cache,cache,[2,2,2],OomRetriever())

    def test_resource_stop_preserves_frozen_denominator(self):
        # The common evidence interface retains unattempted queries as pending.
        from tools.quadra.reviewed_evidence import write_method_bundle,read_method_bundle
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);contract=root/'inputs';contract.mkdir()
            path=contract/'frozen_queries_raw_itk.csv';path.write_text('query_id,subject_id,group_name,mask_name\nq1,s1,head,brain\nq2,s1,head,brain\n')
            (contract/'matching_contract.json').write_text(json.dumps(dict(schema_version=2,dataset_id='fixture',fixture_only=True,subjects=['s1'],query_count=2,
                files=[dict(path=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())])))
            write_method_bundle(contract,root/'run','uae_nn',[dict(query_id='q1',status='failed',forward_status='failed',reverse_status='not_attempted',failure_reason='resource_guard')],dict(resource_stop='diagnostic_budget_guard'))
            manifest,rows=read_method_bundle(root/'run',contract)
            self.assertEqual(manifest['status'],'partial')
            self.assertEqual([r['status'] for r in rows],['failed','pending'])


if __name__ == '__main__':
    unittest.main()
