"""Observable anatomical-review artifacts through the approved cohort CLI seam."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from tools.quadra.reviewed_evidence import write_method_bundle

from tools.quadra import aligned_organ_group_cohort as cohort


class ReviewEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        contract = root / 'contract'
        contract.mkdir()
        rows = [dict(query_id='q%02d' % i, subject_id='s1', mask_name='brain',
                     group_name='head', raw_x=2, raw_y=3, raw_z=4) for i in range(20)]
        cohort.atomic_csv(contract / 'frozen_queries_raw_itk.csv', rows)
        path = contract / 'frozen_queries_raw_itk.csv'
        payload = dict(schema_version=2, fixture_only=True, dataset_id='synthetic',
                       subjects=[dict(subject_id='s1', review_partition='discovery')],
                       query_count=20, files=[dict(path=path.name, bytes=path.stat().st_size,
                                                  sha256=cohort.sha256_file(path))])
        ctroot = root/'cts'
        ctroot.mkdir()
        inventory = []
        grid = np.indices((7,8,9))
        for session, delta in [('test',0), ('retest',1000)]:
            image = nib.Nifti1Image((grid[0]+10*grid[1]+100*grid[2]+delta).astype(np.float32), np.diag([2.,3.,4.,1.]))
            image.header.set_xyzt_units('mm')
            path = ctroot/(session+'.nii.gz')
            nib.save(image,str(path))
            inventory.append(dict(subject_id='s1',session=session, ct_relative_path=path.name, ct_sha256=cohort.sha256_file(path)))
        path=contract/'portable_mask_inventory.csv'
        cohort.atomic_csv(path,inventory)
        payload['files'].append(dict(path=path.name, bytes=path.stat().st_size,sha256=cohort.sha256_file(path)))
        cohort.atomic_json(contract / 'matching_contract.json', payload)
        run = root / 'nn'
        run.mkdir()
        outcomes = [dict(r, method='uae_nn', status='success', forward_status='success',
                         reverse_status='success', cycle_error_mm=i if i < 18 else 19,
                         boundary_anomaly=i == 5, competing_peak=i == 6, failure_reason='') for i, r in enumerate(rows)]
        outcomes[7].update(status='failed', forward_status='failed', reverse_status='not_attempted',
                           failure_reason='no_match', cycle_error_mm='')
        for row in outcomes:
            row.update(matched_lps_x=-4.5,matched_lps_y=-10.5,matched_lps_z=16,
                       returned_lps_x=-4,returned_lps_y=-9,returned_lps_z=16)
        write_method_bundle(contract,run,'uae_nn',outcomes)
        policy = root / 'policy.json'
        cohort.atomic_json(policy, dict(version='pilot-provisional-v1', frozen_for_cohort=False,
            seed=10, disagreement_mm=10, low_cycle_mm=1, random_per_organ=2,
            low_cycle_per_organ=1))
        return contract, run, policy

    def test_queue_keeps_cutoff_ties_and_multiple_failure_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, run, policy = self.fixture(root)
            output = root / 'review'
            self.assertEqual(cohort.main(['reviewed-review-queue', '--contract', str(contract),
                '--run-directory', str(run), '--policy', str(policy), '--output-directory', str(output)]), 0)
            selected = {r['query_id']: json.loads(r['inclusion_reasons'])
                        for r in cohort.read_csv(output / 'review_queue.csv')}
            self.assertIn('upper_5_percent:uae_nn', selected['q18'])
            self.assertIn('upper_5_percent:uae_nn', selected['q19'])
            self.assertIn('technical_failure:uae_nn', selected['q07'])
            self.assertIn('boundary:uae_nn', selected['q05'])
            self.assertIn('competing_peak:uae_nn', selected['q06'])
            manifest = cohort.load_json(output / 'review_manifest.json')
            self.assertFalse(manifest['anatomical_validation'])
            self.assertEqual(manifest['query_count'], 20)
            self.assertEqual(manifest['policy']['version'], 'pilot-provisional-v1')

    def test_adaptive_expansion_preserves_category_and_unreviewed_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, run, policy = self.fixture(root)
            review = root/'review'
            self.assertEqual(cohort.main(['reviewed-review-queue','--contract',str(contract),
                '--run-directory',str(run),'--policy',str(policy),'--output-directory',str(review)]),0)
            record = root/'record.json'
            cohort.atomic_json(record,dict(query_id='q18',reviewer_id='reviewer-one',
                category_version='v1',category_id='similar-anatomy',judgment='suspected_contributor',
                anatomical_reason='Fixture hypothesis only', uncertainty='Unconfirmed',
                selection_reason='upper tail',category_definition='Similar appearance at competing locations',
                categories_frozen=False,review_stage='discovery'))
            self.assertEqual(cohort.main(['reviewed-review-record','--review-directory',str(review),
                '--record',str(record)]),0)
            expanded=root/'expanded'
            self.assertEqual(cohort.main(['reviewed-review-expand','--review-directory',str(review),
                '--mask-name','brain','--category-id','similar-anatomy','--category-version','v1',
                '--batch-size','5','--seed','12','--output-directory',str(expanded)]),0)
            rows=cohort.read_csv(expanded/'review_queue.csv')
            self.assertEqual(len(rows),5)
            self.assertNotIn('q18',{r['query_id'] for r in rows})
            self.assertTrue(all('adaptive_category:v1:similar-anatomy' in json.loads(r['inclusion_reasons']) for r in rows))

    def test_pilot_scope_keeps_controls_and_coverage_out_of_unrun_subjects(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract,run,policy=self.fixture(root)
            queries=cohort.read_csv(contract/'frozen_queries_raw_itk.csv')
            outcomes=cohort.read_csv(run/'query_outcomes.csv')
            for query,row in zip(queries[10:],outcomes[10:]):
                query['subject_id']='s2'
                row.update(subject_id='s2',status='pending',forward_status='pending',
                           reverse_status='pending',cycle_error_mm='')
            cohort.atomic_csv(contract/'frozen_queries_raw_itk.csv',queries)
            payload=cohort.load_json(contract/'matching_contract.json')
            path=contract/'frozen_queries_raw_itk.csv'
            payload['files'][0].update(bytes=path.stat().st_size,sha256=cohort.sha256_file(path))
            payload['subjects'].append(dict(subject_id='s2',review_partition='confirmation'))
            cohort.atomic_json(contract/'matching_contract.json',payload)
            scoped_run=root/'scoped_run'
            write_method_bundle(contract,scoped_run,'uae_nn',outcomes)
            output=root/'pilot_review'
            self.assertEqual(cohort.main(['reviewed-review-queue','--contract',str(contract),
                '--run-directory',str(scoped_run),'--policy',str(policy),'--subject','s1',
                '--output-directory',str(output)]),0)
            manifest=cohort.load_json(output/'review_manifest.json')
            self.assertEqual(manifest['query_count'],10)
            self.assertEqual(manifest['selected_subject_ids'],['s1'])
            self.assertEqual(json.loads((output/'control_sampling.json').read_text())[0]['random_eligible'],10)
            self.assertTrue(all(r['subject_id']=='s1' for r in cohort.read_csv(output/'all_queries.csv')))
            self.assertTrue(all(r['subject_id']=='s1' for r in cohort.read_csv(output/'review_queue.csv')))
            self.assertEqual(cohort.main(['reviewed-review-queue','--contract',str(contract),
                '--run-directory',str(scoped_run),'--policy',str(policy),'--subject','missing',
                '--output-directory',str(root/'invalid_scope')]),3)
            self.assertFalse((root/'invalid_scope').exists())

    def test_arbitrary_query_ct_inspection_uses_physical_planes_and_fractional_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract,run,policy=self.fixture(root)
            output=root/'inspection'
            self.assertEqual(cohort.main(['reviewed-inspect','--contract',str(contract),
                '--run-directory',str(run),'--ct-root',str(root/'cts'),'--query-id','q03',
                '--local-radius-mm','1','--wide-radius-mm','2','--pixel-mm','.5',
                '--window-min-hu','0','--window-max-hu','2000','--output-directory',str(output)]),0)
            manifest=cohort.load_json(output/'inspection_manifest.json')
            self.assertEqual(manifest['ct_window_hu'],[0.,2000.])
            self.assertEqual(manifest['points'][0]['physical_lps_mm'],[-4.,-9.,16.])
            self.assertEqual(manifest['points'][1]['native_voxel_xyz'],[2.25,3.5,4.])
            arrays=np.load(output/'point_01.npz')
            for key in ('local_axial','local_coronal','local_sagittal'):
                self.assertAlmostEqual(float(arrays[key][2,2]),1437.25,places=3)
            self.assertFalse(manifest['anatomical_validation'])
            (root/'cts'/'test.nii.gz').write_bytes(b'stale CT')
            self.assertEqual(cohort.main(['reviewed-inspect','--contract',str(contract),
                '--run-directory',str(run),'--ct-root',str(root/'cts'),'--query-id','q03',
                '--output-directory',str(root/'rejected')]),3)

    def test_confirmation_requires_explicit_frozen_discovery_category_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract,run,policy=self.fixture(root)
            payload=cohort.load_json(contract/'matching_contract.json')
            queries=cohort.read_csv(contract/'frozen_queries_raw_itk.csv')
            queries[-1]['subject_id']='s2'
            cohort.atomic_csv(contract/'frozen_queries_raw_itk.csv',queries)
            path=contract/'frozen_queries_raw_itk.csv'
            payload['files'][0].update(bytes=path.stat().st_size,sha256=cohort.sha256_file(path))
            payload['subjects'].append(dict(subject_id='s2',review_partition='confirmation'))
            cohort.atomic_json(contract/'matching_contract.json',payload)
            outcomes=cohort.read_csv(run/'query_outcomes.csv')
            outcomes[-1]['subject_id']='s2'
            run=root/'updated'
            write_method_bundle(contract,run,'uae_nn',outcomes)
            review=root/'review'
            self.assertEqual(cohort.main(['reviewed-review-queue','--contract',str(contract),
                '--run-directory',str(run),'--policy',str(policy),'--output-directory',str(review)]),0)
            value=dict(query_id='q18',reviewer_id='one',category_version='v1',category_id='ambiguous',
                judgment='indeterminate',anatomical_reason='Cannot establish correspondence',uncertainty='High',
                selection_reason='tail',category_definition='Multiple similar structures',categories_frozen=False,
                review_stage='discovery')
            record=root/'record.json'
            cohort.atomic_json(record,value)
            self.assertEqual(cohort.main(['reviewed-review-record','--review-directory',str(review),'--record',str(record)]),0)
            value.update(query_id='q19',review_stage='confirmation',categories_frozen=True)
            cohort.atomic_json(record,value)
            self.assertEqual(cohort.main(['reviewed-review-record','--review-directory',str(review),'--record',str(record)]),3)
            self.assertEqual(cohort.main(['reviewed-review-freeze-categories','--review-directory',str(review),
                '--category-version','v1','--reviewer-id','one']),0)
            self.assertEqual(cohort.main(['reviewed-review-record','--review-directory',str(review),'--record',str(record)]),0)
            value['category_definition']='Changed after freezing'
            cohort.atomic_json(record,value)
            self.assertEqual(cohort.main(['reviewed-review-record','--review-directory',str(review),'--record',str(record)]),3)
            report=root/'coverage.json'
            self.assertEqual(cohort.main(['reviewed-review-status','--review-directory',str(review),'--output-file',str(report)]),0)
            coverage=cohort.load_json(report)
            self.assertEqual(coverage['coverage'][0]['reviewed_queries'],2)
            self.assertEqual(coverage['indeterminate_queries'],['q18','q19'])

    def test_inspection_exposes_retained_peak_anatomy_instead_of_only_the_final_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract,run,policy=self.fixture(root)
            rows=cohort.read_csv(run/'query_outcomes.csv')
            rows[3]['forward_peak_candidates']=[dict(physical_lps_xyz=[-6.,-9.,16.],
                fine_xyz=[3,3,4],rank=2,fused_score=.9,minimum_separation_mm=2)]
            newrun=root/'peaks'
            write_method_bundle(contract,newrun,'uae_nn',rows)
            output=root/'inspection'
            self.assertEqual(cohort.main(['reviewed-inspect','--contract',str(contract),
                '--run-directory',str(newrun),'--ct-root',str(root/'cts'),'--query-id','q03',
                '--local-radius-mm','1','--wide-radius-mm','2','--pixel-mm','.5',
                '--output-directory',str(output)]),0)
            manifest=cohort.load_json(output/'inspection_manifest.json')
            peaks=[p for p in manifest['points'] if p.get('peak_candidate')]
            self.assertEqual(len(peaks),1)
            self.assertEqual(peaks[0]['physical_lps_mm'],[-6.,-9.,16.])
            self.assertEqual(peaks[0]['peak_candidate']['rank'],2)

    def test_modified_judgment_is_rejected_in_coverage_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract,run,policy=self.fixture(root)
            review=root/'review'
            self.assertEqual(cohort.main(['reviewed-review-queue','--contract',str(contract),
                '--run-directory',str(run),'--policy',str(policy),'--output-directory',str(review)]),0)
            record=root/'record.json'
            cohort.atomic_json(record,dict(query_id='q18',reviewer_id='one',category_version='v1',
                category_id='uncertain',judgment='indeterminate',anatomical_reason='Unknown anatomy',
                uncertainty='High',selection_reason='tail',category_definition='Uncertain structure',
                categories_frozen=False,review_stage='discovery'))
            self.assertEqual(cohort.main(['reviewed-review-record','--review-directory',str(review),'--record',str(record)]),0)
            path=next((review/'judgments').glob('*.json'))
            value=cohort.load_json(path)
            value['judgment']='supported_contributor'
            cohort.atomic_json(path,value)
            self.assertEqual(cohort.main(['reviewed-review-status','--review-directory',str(review)]),3)

    def test_queue_detects_different_correspondences_with_identical_cycle_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract,run,policy=self.fixture(root)
            rows=cohort.read_csv(run/'query_outcomes.csv')
            for row in rows:
                row['method']='uae_fixed_point'
                row.pop('boundary_anomaly')
                row.pop('competing_peak')
            rows[3]['matched_lps_x']=100
            fp=root/'fp'
            write_method_bundle(contract,fp,'uae_fixed_point',rows)
            output=root/'review'
            command=['reviewed-review-queue','--contract',str(contract),'--run-directory',str(run),
                '--run-directory',str(fp),'--policy',str(policy),'--output-directory',str(output)]
            self.assertEqual(cohort.main(command),0)
            selected={r['query_id']:json.loads(r['inclusion_reasons']) for r in cohort.read_csv(output/'review_queue.csv')}
            self.assertIn('correspondence_disagreement:uae_fixed_point:uae_nn',selected['q03'])
            manifest=cohort.load_json(output/'review_manifest.json')
            self.assertEqual(manifest['diagnostic_unknown_counts']['uae_fixed_point:competing_peak'],20)
            repeat=root/'repeat'
            self.assertEqual(cohort.main(command[:-1]+[str(repeat)]),0)
            self.assertEqual((repeat/'review_queue.csv').read_bytes(),(output/'review_queue.csv').read_bytes())

    def test_confirmation_ct_is_hidden_until_discovery_categories_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract,run,policy=self.fixture(root)
            payload=cohort.load_json(contract/'matching_contract.json')
            queries=cohort.read_csv(contract/'frozen_queries_raw_itk.csv')
            queries[-1]['subject_id']='s2'
            cohort.atomic_csv(contract/'frozen_queries_raw_itk.csv',queries)
            inventory=cohort.read_csv(contract/'portable_mask_inventory.csv')
            cohort.atomic_csv(contract/'portable_mask_inventory.csv',inventory+[dict(r,subject_id='s2') for r in inventory])
            for item in payload['files']:
                path=contract/item['path']
                item.update(bytes=path.stat().st_size,sha256=cohort.sha256_file(path))
            payload['subjects'].append(dict(subject_id='s2',review_partition='confirmation'))
            cohort.atomic_json(contract/'matching_contract.json',payload)
            rows=cohort.read_csv(run/'query_outcomes.csv'); rows[-1]['subject_id']='s2'
            run=root/'updated'; write_method_bundle(contract,run,'uae_nn',rows)
            command=['reviewed-inspect','--contract',str(contract),'--run-directory',str(run),
                '--ct-root',str(root/'cts'),'--query-id','q19','--local-radius-mm','1',
                '--wide-radius-mm','2','--pixel-mm','.5','--output-directory',str(root/'blocked')]
            self.assertEqual(cohort.main(command),3)
            review=root/'review'
            self.assertEqual(cohort.main(['reviewed-review-queue','--contract',str(contract),
                '--run-directory',str(run),'--policy',str(policy),'--output-directory',str(review)]),0)
            record=root/'record.json'
            cohort.atomic_json(record,dict(query_id='q18',reviewer_id='one',category_version='v1',category_id='uncertain',
                judgment='indeterminate',anatomical_reason='Unknown anatomy',uncertainty='High',selection_reason='tail',
                category_definition='Uncertain structure',categories_frozen=False,review_stage='discovery'))
            self.assertEqual(cohort.main(['reviewed-review-record','--review-directory',str(review),'--record',str(record)]),0)
            self.assertEqual(cohort.main(command+['--confirmation-review-directory',str(review)]),3)
            self.assertEqual(cohort.main(['reviewed-review-freeze-categories','--review-directory',str(review),
                '--category-version','v1','--reviewer-id','one']),0)
            self.assertEqual(cohort.main(command[:-1]+[str(root/'allowed'),'--confirmation-review-directory',str(review)]),0)
