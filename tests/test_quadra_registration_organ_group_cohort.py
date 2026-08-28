"""CPU-only cohort orchestration tests. Real CT processing is RunPod-only."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

from tools.quadra import registration_organ_group_cohort as reg
from tools.quadra import registration_point_transform as pt
from tests.test_quadra_registration_organ_group import aligned, query


class ContractTests(unittest.TestCase):
    def test_full_native_plan_set_and_no_cross_subject_reuse(self):
        plans,sources = {},{}
        for s in reg.SUBJECTS:
            for session in ('test','retest'):
                for g in reg.GROUPS:
                    a = aligned()
                    a.update(subject_id=s,session=session,group_name=g)
                    p = reg.pilot.serializable_plan(a)
                    plans[s+'-'+session+'-'+g] = p
                    sources[s+'-'+session] = p['source_ct']
        reg.validate_plan_set(plans,sources)
        self.assertEqual(len(plans),224)
        removed = plans.pop('quadra_hc_021-test-pelvis')
        with self.assertRaises(pt.RegistrationError):reg.validate_plan_set(plans,sources)
        plans['quadra_hc_021-test-pelvis'] = removed
        removed['source_uae_plan']['margin_mm'] = 120
        with self.assertRaises(pt.RegistrationError):reg.validate_plan_set(plans,sources)

    def test_native_phase_and_no_new_spacing(self):
        a = aligned()
        p = reg.pilot.serializable_plan(a)
        self.assertEqual(p['source_ct']['affine'],a['source_ct']['affine'])
        self.assertEqual(p['nominal_margin_mm'],100)
        self.assertLess(p['geometry_checks']['max_roundtrip_voxels'],1e-6)

    def test_explicit_approval_required_before_io(self):
        args = SimpleNamespace(approve_pilot=False,review_rationale='reviewed')
        with self.assertRaises(pt.RegistrationError):reg.prepare(args)
        args.approve_pilot=True;args.review_rationale='  '
        with self.assertRaises(pt.RegistrationError):reg.prepare(args)

    def test_wrong_pilot_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pilot.json';pt.atomic_json(path,{'status':'PASS'})
            with self.assertRaises(pt.RegistrationError):reg.accepted_pilot(path)

    def test_unchanged_pilot_cli_remains_pilot_only(self):
        self.assertEqual(reg.pilot.PILOT,'quadra_hc_044')
        self.assertEqual(reg.pilot.QUERY_COUNT,3914)
        args=reg.parser().parse_args(['run','--run-directory','/tmp/new-cohort'])
        self.assertEqual(args.command,'run')

    def test_task_signature_is_subject_group_direction_specific(self):
        m={'signature':'frozen'}
        signatures={reg.task_signature(m,s,g,d) for s in reg.SUBJECTS for g in reg.GROUPS for d in reg.DIRECTIONS}
        self.assertEqual(len(signatures),336)

    def test_unknown_task_scope_rejected(self):
        with self.assertRaises(pt.RegistrationError):reg.group_dir(Path('/tmp'),'quadra_hc_001','pelvis')
        with self.assertRaises(pt.RegistrationError):reg.group_dir(Path('/tmp'),'quadra_hc_021','../../escape')


class PointTests(unittest.TestCase):
    def setUp(self):
        self.q=query([1,2,3])
        self.row=dict(self.q,valid_cycle='True',cycle_error_mm='0.5',failure_reason='',
                      query_physical_x='0',query_physical_y='0',query_physical_z='0',
                      returned_physical_x='0.3',returned_physical_y='0.4',returned_physical_z='0')

    def test_cycle_distance_reconciled_and_metadata_unchanged(self):
        reg.validate_points([self.row],[self.q])
        for key,value in [('cycle_error_mm','1'),('raw_x','7'),('valid_cycle','false'),('query_id','other')]:
            with self.assertRaises(pt.RegistrationError):reg.validate_points([dict(self.row,**{key:value})],[self.q])

    def test_invalid_points_keep_empty_error_not_zero(self):
        row=dict(self.row,valid_cycle='False',cycle_error_mm='',failure_reason='forward_outside_retest_crop')
        reg.validate_points([row],[self.q])
        with self.assertRaises(pt.RegistrationError):reg.validate_points([dict(row,cycle_error_mm='0')],[self.q])

    def test_duplicate_or_missing_ids_rejected(self):
        with self.assertRaises(pt.RegistrationError):reg.validate_points([self.row,self.row],[self.q])
        with self.assertRaises(pt.RegistrationError):reg.validate_points([],[self.q])

    def test_point_marker_identity_and_count(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);csv=path/'points.csv';pt.atomic_csv(csv,[self.row])
            m={'signature':'s','limits':{'ram_ceiling_bytes':100}}
            meta=dict(schema_version=1,status='success',signature=reg.task_signature(m,reg.pilot.PILOT,'abdomen','points'),
                      subject_id=reg.pilot.PILOT,group_name='abdomen',direction='points',files=[pt.identity(csv)],
                      point_csv=pt.identity(csv),peak_rss_bytes=50,queries=1,valid_queries=1,invalid_queries=0)
            marker=path/'marker.json';pt.atomic_json(marker,meta)
            with mock.patch.object(reg,'rows_for',return_value=[self.q]):
                self.assertIsNotNone(reg.validate_result(m,reg.pilot.PILOT,'abdomen','points',marker))
                pt.atomic_csv(csv,[dict(self.row,cycle_error_mm='9')])
                with self.assertRaises(pt.RegistrationError):reg.validate_result(m,reg.pilot.PILOT,'abdomen','points',marker)


class ResumeTests(unittest.TestCase):
    def test_completed_marker_does_not_launch(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(reg,'validate_result',return_value={'status':'success'}), \
             mock.patch.object(reg.subprocess,'Popen') as launch:
            reg.run_task(Path(d),{},'quadra_hc_021','pelvis','forward')
            launch.assert_not_called()

    def test_completed_atomic_worker_result_is_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            dest=reg.group_dir(d,'quadra_hc_021','pelvis')
            p=dest/'forward-attempt-1/worker_result.json';pt.atomic_json(p,{})
            with mock.patch.object(reg,'validate_result',side_effect=[None,{'status':'success'}]),mock.patch.object(reg.subprocess,'Popen') as launch:
                reg.run_task(Path(d),{},'quadra_hc_021','pelvis','forward')
                self.assertEqual(pt.load_json(dest/'forward.json'),{'status':'success'})
                launch.assert_not_called()

    def test_retry_limit_survives_resume(self):
        with tempfile.TemporaryDirectory() as d:
            m={'signature':'s','limits':{'direction_timeout_seconds':1}}
            process=mock.Mock(pid=123,returncode=-11)
            process.poll.return_value=-11
            observed=mock.Mock();observed.create_time.return_value=1234.0
            with mock.patch.object(reg,'validate_result',return_value=None),mock.patch.object(reg,'assert_idle'), \
                 mock.patch.object(reg.subprocess,'Popen',return_value=process) as launch, \
                 mock.patch('psutil.Process',return_value=observed),mock.patch.object(reg.base,'stop_owned_process'):
                with self.assertRaises(pt.RuntimeFailure):reg.run_task(Path(d),m,'quadra_hc_021','pelvis','forward')
                self.assertEqual(launch.call_count,2)
                with self.assertRaises(pt.RuntimeFailure):reg.run_task(Path(d),m,'quadra_hc_021','pelvis','forward')
                self.assertEqual(launch.call_count,2)

    def test_guard_failure_stops_only_owned_worker_and_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            m={'signature':'s','limits':{'direction_timeout_seconds':60}}
            process=mock.Mock(pid=123,returncode=None);process.poll.return_value=None
            observed=mock.Mock();observed.create_time.return_value=1234.0
            observed.children.return_value=[];observed.memory_info.return_value=SimpleNamespace(rss=100)
            observed.cpu_percent.return_value=100
            with mock.patch.object(reg,'validate_result',return_value=None),mock.patch.object(reg,'assert_idle'), \
                 mock.patch.object(reg.subprocess,'Popen',return_value=process) as launch, \
                 mock.patch('psutil.Process',return_value=observed),mock.patch.object(reg.base,'stop_owned_process') as stop, \
                 mock.patch.object(reg.base,'check_resources',side_effect=pt.RegistrationError('RAM guard')):
                with self.assertRaises(pt.RegistrationError):reg.run_task(Path(d),m,'quadra_hc_021','pelvis','forward')
                stop.assert_called_once_with(process);self.assertEqual(launch.call_count,1)

    def test_other_scientific_worker_blocks_launch(self):
        proc=SimpleNamespace(pid=123,info={'cmdline':['python','-m',reg.pilot.MODULE,'_worker']})
        with mock.patch('psutil.process_iter',return_value=[proc]):
            with self.assertRaises(pt.RegistrationError):reg.assert_idle(Path('/tmp/test'))

    def test_atomic_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json';pt.atomic_json(p,{},refuse=True)
            with self.assertRaises(FileExistsError):pt.atomic_json(p,{},refuse=True)


class OutcomeTests(unittest.TestCase):
    def test_complete_and_partial_denominators(self):
        count=dict(groups_completed=112,failed_groups=[],queries_valid=108431,queries_invalid=0,queries_failed=0,
                   subjects_completed=28,registrations_completed=224)
        self.assertEqual(reg.completion_status(count),'TECHNICAL_PASS')
        count.update(queries_valid=108430,queries_invalid=1)
        self.assertEqual(reg.completion_status(count),'PARTIAL')
        count.update(queries_valid=108429)
        with self.assertRaises(pt.RegistrationError):reg.completion_status(count)

    def test_failed_group_accounts_for_every_query(self):
        count=dict(groups_completed=111,failed_groups=['quadra_hc_021:pelvis'],queries_valid=108031,queries_invalid=0,
                   queries_failed=400,subjects_completed=27,registrations_completed=222)
        self.assertEqual(reg.completion_status(count),'PARTIAL')

    def test_status_has_heartbeat_and_no_scheduler(self):
        with tempfile.TemporaryDirectory() as d:
            reg.status_update(Path(d),status='RUNNING',current_subject='quadra_hc_021',worker_pid=12)
            r=pt.load_json(Path(d)/'cohort_status.json')
            self.assertEqual(r['worker_pid'],12);self.assertIn('heartbeat_at',r)
            reg.status_update(Path(d),groups_completed=4)
            self.assertEqual(pt.load_json(Path(d)/'cohort_status.json')['worker_pid'],12)

    def test_backup_covers_module_and_run_root(self):
        from tools.quadra.artifact_backup import ACTIVE_PROCESS_PATTERNS,ALLOWLIST
        self.assertTrue(any(x in reg.MODULE for x in ACTIVE_PROCESS_PATTERNS))
        self.assertIn('runs/cohort',ALLOWLIST)


@unittest.skipUnless(os.environ.get('QUADRA_REGISTRATION_INTEGRATION') == '1' and
                     os.environ.get('RUNPOD_POD_ID') in ('2ohlzqc00kd7sn','1ngcj5dw1mifiw'),
                     'Synthetic shared-kernel registration runs on RunPod only')
class SharedKernelIntegrationTests(unittest.TestCase):
    def test_new_cohort_uses_the_pilot_registration_and_point_kernel(self):
        import itk
        z,y,x=np.indices((64,64,64),dtype=np.float32)
        data=(-1024+1500*np.exp(-((x-29)**2+(y-33)**2+(z-28)**2)/180)
              +900*np.exp(-((x-43)**2+(y-20)**2+(z-39)**2)/55)).astype(np.float32)
        image=itk.image_from_array(data);image.SetSpacing([2.,2.,2.]);image.SetOrigin([10.,20.,30.])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'synthetic.nii.gz';itk.imwrite(image,str(path))
            a=np.diag([-2.,-2.,2.,1.]);a[:3,3]=[-10.,-20.,30.]
            source=dict(pt.identity(path),affine=a.tolist(),native_shape_xyz=[64,64,64])
            aligned_plan=dict(subject_id='quadra_hc_021',session='test',group_name='abdomen',source_ct=source,
                              padded_2mm_affine=a.tolist(),padded_shape_xyz=[64]*3,valid_model_box_xyz=[[0]*3,[64]*3],
                              included_masks=[],margin_mm=100,strategy='organ_group_global_lattice')
            plans=[reg.pilot.serializable_plan(dict(aligned_plan,session=s)) for s in ('test','retest')]
            q=dict(query([30,30,30]),subject_id='quadra_hc_021')
            m=dict(masks=[],parameters=pt.parameter_maps(),limits={'ram_ceiling_bytes':10**12})
            work=root/'registration';work.mkdir()
            reg.pilot.perform_task(m,work,'quadra_hc_021','abdomen','forward',plans,[q],'synthetic')
            meta=pt.load_json(work/'worker_result.json')
            self.assertEqual(meta['subject_id'],'quadra_hc_021')
            self.assertEqual(meta['fixed_geometry'],plans[0]['crop_geometry'])
            self.assertEqual(len(pt.load_json(meta['maps_json']['path'])),2)
            point_work=root/'points';point_work.mkdir()
            reg.pilot.perform_task(m,point_work,'quadra_hc_021','abdomen','points',plans,[q],'synthetic-points',[meta,meta])
            rows=pt.read_csv(point_work/'points.csv')
            reg.validate_points(rows,[q])
            self.assertEqual(rows[0]['valid_cycle'],'True')
            self.assertEqual(reg.base.forbidden_outputs(work),[])


if __name__ == '__main__':
    unittest.main()
