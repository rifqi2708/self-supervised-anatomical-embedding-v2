"""Reviewed registration behavior at the cohort CLI/exported evidence seam."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.quadra import registration_organ_group_cohort as command


class ReviewedRegistrationTests(unittest.TestCase):
    def fixture(self, root):
        contract = root/'contract'; contract.mkdir()
        source = dict(path='synthetic.nii.gz', bytes=0, sha256='0'*64,
                      native_shape_xyz=[10,10,10], affine=[[-2,0,0,20],[0,-3,0,30],[0,0,4,40],[0,0,0,1]])
        queries = [dict(query_id='q1', subject_id='s1', group_name='abdomen', mask_name='liver',
                        raw_x=2,raw_y=2,raw_z=2,physical_lps_x=-16,physical_lps_y=-24,physical_lps_z=48)]
        with (contract/'frozen_queries_raw_itk.csv').open('w',newline='') as h:
            w=csv.DictWriter(h, fieldnames=list(queries[0]));w.writeheader();w.writerows(queries)
        plans=[]
        for session in ('test','retest'):
            path='registration_plans/s1-{}-abdomen.json'.format(session)
            p=contract/path;p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(dict(subject_id='s1',session=session,group_name='abdomen',source_ct=source,
                crop_start_xyz=[1,1,1],crop_end_xyz=[9,9,9],crop_geometry=dict(source, native_shape_xyz=[8,8,8],
                    affine=[[-2,0,0,18],[0,-3,0,27],[0,0,4,44],[0,0,0,1]]))))
            plans.append(path)
        params=Path(command.ROOT)/'configs/quadra/reviewed-registration-parameters.json'
        (contract/'registration_parameters.json').write_bytes(params.read_bytes())
        files=[dict(path=p.relative_to(contract).as_posix(),bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(contract.rglob('*')) if p.is_file()]
        (contract/'matching_contract.json').write_text(json.dumps(dict(schema_version=2,fixture_only=True,
            dataset_id='synthetic-reviewed',subjects=[dict(subject_id='s1',review_partition='discovery')],
            pilot_subjects=['s1'],query_count=1,registration_plan_files=plans,files=files)))
        # Independently supplied directional transforms, deliberately not inverses.
        transforms=root/'transforms.json'
        transforms.write_text(json.dumps({'forward':[[1,0,0,0.5],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            'backward':[[1,0,0,-0.25],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}))
        return contract,transforms

    def run_fixture(self, root, transforms=None, extra=()):
        contract,supplied=self.fixture(root)
        if transforms is not None:supplied.write_text(json.dumps(transforms))
        run=root/'run'
        rc=command.main(['reviewed-run','--contract',str(contract),'--run-directory',str(run),
                         '--fixture-transforms',str(supplied)]+list(extra))
        return rc,run

    def test_fractional_lps_cycle_uses_two_independent_directions(self):
        with tempfile.TemporaryDirectory() as d:
            rc,run=self.run_fixture(Path(d))
            self.assertEqual(rc,0)
            with (run/'query_outcomes.csv').open() as h: row=next(csv.DictReader(h))
            self.assertEqual(row['status'],'success')
            self.assertAlmostEqual(float(row['matched_lps_x']),-15.5)
            self.assertAlmostEqual(float(row['matched_raw_x']),2.25)
            self.assertAlmostEqual(float(row['returned_raw_x']),2.125)
            self.assertAlmostEqual(float(row['cycle_error_mm']),0.25)
            manifest=json.loads((run/'method_manifest.json').read_text())
            self.assertFalse(manifest['scientific_work_launched'])

    def test_point_outside_group_crop_is_retained_without_clipping(self):
        transforms={'forward':[[1,0,0,14],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                    'backward':[[1,0,0,-14],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
        with tempfile.TemporaryDirectory() as d:
            rc,run=self.run_fixture(Path(d),transforms)
            self.assertEqual(rc,0)
            with (run/'query_outcomes.csv').open() as h:row=next(csv.DictReader(h))
            self.assertEqual(row['status'],'failed')
            self.assertEqual(row['forward_status'],'out_of_domain')
            self.assertEqual(row['reverse_status'],'not_attempted')
            self.assertEqual(row['failure_reason'],'forward_outside_retest_crop')
            self.assertEqual(float(row['matched_raw_x']),9.0)  # inside native FOV, outside ROI
            self.assertEqual(float(row['matched_lps_x']),-2.0)
            self.assertEqual(row['cycle_error_mm'],'')

    def test_resigned_inconsistent_frozen_physical_query_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,transforms=self.fixture(root)
            path=contract/'frozen_queries_raw_itk.csv'
            text=path.read_text().replace(',-16,',',-15,');path.write_text(text)
            manifest=json.loads((contract/'matching_contract.json').read_text())
            for item in manifest['files']:
                if item['path']==path.name:item.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (contract/'matching_contract.json').write_text(json.dumps(manifest))
            rc=command.main(['reviewed-run','--contract',str(contract),'--run-directory',str(root/'run'),
                             '--fixture-transforms',str(transforms)])
            self.assertEqual(rc,2)
            self.assertFalse((root/'run/query_outcomes.csv').exists())

    def test_return_outside_test_crop_preserves_forward_match_and_return(self):
        transforms={'forward':[[1,0,0,0.5],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                    'backward':[[1,0,0,-10],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
        with tempfile.TemporaryDirectory() as d:
            rc,run=self.run_fixture(Path(d),transforms)
            self.assertEqual(rc,0)
            with (run/'query_outcomes.csv').open() as h:row=next(csv.DictReader(h))
            self.assertEqual(row['status'],'failed')
            self.assertEqual(row['forward_status'],'success')
            self.assertEqual(row['reverse_status'],'out_of_domain')
            self.assertAlmostEqual(float(row['returned_raw_x']),-2.75)
            self.assertEqual(row['failure_reason'],'returned_outside_test_crop')
            self.assertEqual(row['cycle_error_mm'],'')

    def test_compatible_resume_keeps_terminal_queries_once_and_rejects_changed_backend(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,transforms=self.fixture(root);run=root/'run'
            args=['reviewed-run','--contract',str(contract),'--run-directory',str(run),
                  '--fixture-transforms',str(transforms)]
            self.assertEqual(command.main(args),0)
            before=(run/'query_outcomes.csv').read_bytes()
            self.assertEqual(command.main(args+['--resume']),0)
            self.assertEqual((run/'query_outcomes.csv').read_bytes(),before)
            changed=json.loads(transforms.read_text());changed['backward'][0][3]=-0.5
            transforms.write_text(json.dumps(changed))
            self.assertEqual(command.main(args+['--resume']),2)
            self.assertEqual((run/'query_outcomes.csv').read_bytes(),before)

    def test_real_runtime_requires_explicit_pilot_approval(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,_=self.fixture(root)
            rc=command.main(['reviewed-run','--contract',str(contract),'--run-directory',str(root/'run'),
                             '--ct-root',str(root/'cts')])
            self.assertEqual(rc,2)
            self.assertFalse((root/'run').exists())

    def test_resigned_crop_origin_disagreeing_with_native_ct_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,transforms=self.fixture(root)
            path=contract/'registration_plans/s1-test-abdomen.json'
            plan=json.loads(path.read_text());plan['crop_geometry']['affine'][0][3]=17
            path.write_text(json.dumps(plan))
            manifest=json.loads((contract/'matching_contract.json').read_text())
            for item in manifest['files']:
                if item['path']==path.relative_to(contract).as_posix():
                    item.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (contract/'matching_contract.json').write_text(json.dumps(manifest))
            rc=command.main(['reviewed-run','--contract',str(contract),'--run-directory',str(root/'run'),
                             '--fixture-transforms',str(transforms)])
            self.assertEqual(rc,2)

    def test_failed_reverse_estimation_retains_forward_coordinates_and_directional_evidence(self):
        from tools.quadra.reviewed_registration import AffineFixtureBackend, run_reviewed_registration
        from tools.quadra.reviewed_evidence import read_method_bundle
        class FailedReverse(AffineFixtureBackend):
            def estimate(self, fixed_plan, moving_plan, parameters, directory, direction):
                if direction=='backward':raise RuntimeError('injected independent reverse failure')
                return super().estimate(fixed_plan,moving_plan,parameters,directory,direction)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,transforms=self.fixture(root)
            run_reviewed_registration(contract,root/'run',FailedReverse(transforms))
            manifest,rows=read_method_bundle(root/'run',contract)
            row=rows[0]
            self.assertEqual(row['forward_status'],'success')
            self.assertEqual(row['reverse_status'],'registration_failed')
            self.assertAlmostEqual(float(row['matched_lps_x']),-15.5)
            self.assertEqual(row['cycle_error_mm'],'')
            estimates=json.loads((root/'run/registration/s1/abdomen/directional_estimates.json').read_text())
            self.assertEqual([x['status'] for x in estimates['estimates']],['success','failed'])
            self.assertEqual(manifest['failed_queries'],1)

    def test_contract_stop_in_later_group_keeps_earlier_sealed_outcomes_and_pending_denominator(self):
        from tools.quadra.reviewed_registration import AffineFixtureBackend, run_reviewed_registration
        from tools.quadra.reviewed_evidence import read_method_bundle
        from tools.quadra.registration_point_transform import RegistrationError
        class StopsLater(AffineFixtureBackend):
            def estimate(self, fixed_plan, moving_plan, parameters, directory, direction):
                if fixed_plan['group_name']=='thorax':raise RegistrationError('injected resource or contract stop')
                return super().estimate(fixed_plan,moving_plan,parameters,directory,direction)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,transforms=self.fixture(root)
            for session in ('test','retest'):
                source=contract/'registration_plans'/('s1-{}-abdomen.json'.format(session))
                plan=json.loads(source.read_text());plan['group_name']='thorax'
                (contract/'registration_plans'/('s1-{}-thorax.json'.format(session))).write_text(json.dumps(plan))
            path=contract/'frozen_queries_raw_itk.csv'
            with path.open() as h:rows=list(csv.DictReader(h));fields=list(rows[0])
            rows.append(dict(rows[0],query_id='q2',group_name='thorax'))
            with path.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
            manifest=json.loads((contract/'matching_contract.json').read_text())
            manifest['query_count']=2
            manifest['registration_plan_files']+=['registration_plans/s1-{}-thorax.json'.format(s) for s in ('test','retest')]
            manifest['files']=[dict(path=p.relative_to(contract).as_posix(),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(contract.rglob('*'))
                if p.is_file() and p.name!='matching_contract.json']
            (contract/'matching_contract.json').write_text(json.dumps(manifest))
            with self.assertRaises(RegistrationError):
                run_reviewed_registration(contract,root/'run',StopsLater(transforms))
            saved,outcomes=read_method_bundle(root/'run',contract)
            self.assertEqual(saved['status'],'partial')
            self.assertEqual(saved['attempted_queries'],1)
            self.assertEqual([row['status'] for row in outcomes],['success','pending'])
            self.assertTrue((root/'.run.registration-staging/registration/s1/thorax').exists())
