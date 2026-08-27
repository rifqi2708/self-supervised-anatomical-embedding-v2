"""Native group geometry and safety; real CTs are never used in local tests."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools.quadra import registration_organ_group as reg
from tools.quadra import registration_organ_group_report as report
from tools.quadra import registration_point_transform as pt


def aligned(oblique=True):
    a = np.eye(4)
    angle = .31 if oblique else 0
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    a[:3, :3] = rotation @ np.diag([1.2, 2.3, 3.4])
    a[:3, 3] = [12.5, -20.3, 100.1]
    b = a.copy()
    b[:3, :3] = rotation @ np.diag([2., 2., 2.])
    return dict(subject_id=reg.PILOT, session="test", group_name="abdomen", margin_mm=100,
                strategy="organ_group_global_lattice", source_ct={"affine":a.tolist(),"native_shape_xyz":[20,30,40]},
                padded_2mm_affine=b.tolist(), padded_shape_xyz=[20,30,40], valid_model_box_xyz=[[2,3,4],[10,20,30]])


def query(point, ident="q1"):
    return dict(query_id=ident, subject_id=reg.PILOT, group_name="abdomen", mask_name="liver",
                **{"raw_"+a:str(v) for a,v in zip("xyz",point)})


class GeometryTests(unittest.TestCase):
    def test_outward_cell_bounds_and_anisotropic_oblique_geometry(self):
        p = reg.native_plan(aligned())
        low, high = np.asarray(p["requested_native_cell_bounds"])
        np.testing.assert_array_equal(p["crop_start_xyz"], np.floor(low).clip(0))
        np.testing.assert_array_equal(p["crop_end_xyz"], np.minimum(np.ceil(high), [20,30,40]))
        self.assertLessEqual(p["geometry_checks"]["max_roundtrip_voxels"], 1e-6)
        full = np.asarray(p["source_ct"]["affine"])
        crop = np.asarray(p["crop_geometry"]["affine"])
        np.testing.assert_allclose(crop[:3,3],pt.apply_affine(p["crop_start_xyz"],full)[0])
        np.testing.assert_allclose(full[:3,:3],crop[:3,:3])

    def test_full_extent_clamps_to_original_fov_without_padding(self):
        a = aligned(False)
        a["valid_model_box_xyz"] = [[0,0,0],[20,30,40]]
        p = reg.native_plan(a)
        self.assertEqual(p["crop_start_xyz"],[0,0,0])
        self.assertEqual(p["crop_end_xyz"][0],20)
        self.assertTrue(p["original_fov_clamped"][1][0])

    def test_half_open_stop_retains_last_voxel(self):
        a = aligned(False)
        a["padded_2mm_affine"] = a["source_ct"]["affine"]
        p = reg.native_plan(a)
        self.assertEqual(p["crop_start_xyz"],[2,3,4])
        self.assertEqual(p["crop_end_xyz"],[10,20,30])
        self.assertTrue(reg.inside_crop([[9,19,29]],p)[0])
        self.assertFalse(reg.inside_crop([[10,19,29]],p)[0])

    def test_non_parallel_grid_rejected(self):
        a = aligned()
        a["padded_2mm_affine"][0][1] += .5
        with self.assertRaises(pt.RegistrationError): reg.native_plan(a)

    def test_invalid_box_rejected(self):
        for box in ([[0,0,0],[0,1,1]],[[0,0,0],[21,30,40]],[[0,0,0],[1,2,float('nan')]]):
            a = aligned(); a['valid_model_box_xyz'] = box
            with self.assertRaises(pt.RegistrationError): reg.native_plan(a)

    def test_different_sessions_remain_different(self):
        a,b = aligned(),aligned()
        b['session']='retest';b['valid_model_box_xyz'][0][1]+=3
        self.assertNotEqual(reg.native_plan(a)['crop_start_xyz'],reg.native_plan(b)['crop_start_xyz'])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.p=reg.native_plan(aligned())
        self.q=(np.array(self.p['crop_start_xyz'])+np.array(self.p['crop_end_xyz'])-1)/2

    def test_identity_preserves_original_raw_and_physical(self):
        original=query(self.q)
        r=reg.evaluate_group([original],self.p,self.p,lambda x:x,lambda x:x)[0]
        self.assertTrue(r['valid_cycle'])
        self.assertLess(r['cycle_error_mm'],1e-10)
        for i,a in enumerate('xyz'):
            self.assertEqual(r['raw_'+a],original['raw_'+a])
            self.assertAlmostEqual(r['matched_raw_'+a],self.q[i])
        expected=pt.apply_affine(self.q,self.p['source_ct']['affine'])[0]
        np.testing.assert_allclose([r['query_physical_'+a] for a in 'xyz'],expected)

    def test_fractional_forward_is_passed_unrounded(self):
        delta=np.array([.311,-.213,.117]); received=[]
        def reverse(p): received.append(p.copy());return p-delta
        r=reg.evaluate_group([query(self.q)],self.p,self.p,lambda x:x+delta,reverse)[0]
        np.testing.assert_allclose(received[0],pt.apply_affine(self.q,pt.lps_affine(self.p['source_ct']))+delta)
        self.assertLess(r['cycle_error_mm'],1e-10)

    def test_outside_crop_inside_full_ct_is_invalid_without_reverse(self):
        outside=np.array(self.p['crop_start_xyz'],float)-[1,0,0]
        self.assertTrue(pt.inside([outside],self.p['source_ct']['native_shape_xyz'])[0])
        physical=pt.apply_affine([outside],pt.lps_affine(self.p['source_ct']))
        reverse=mock.Mock()
        r=reg.evaluate_group([query(self.q)],self.p,self.p,lambda p:physical,reverse)[0]
        reverse.assert_not_called()
        self.assertFalse(r['valid_cycle']);self.assertEqual(r['cycle_error_mm'],'')
        self.assertEqual(r['failure_reason'],'forward_outside_retest_crop')
        self.assertAlmostEqual(r['matched_raw_x'],outside[0])

    def test_return_outside_and_nonfinite_are_retained(self):
        for backward, reason in ((lambda p:p+10000,'returned_outside_test_crop'),(lambda p:p*np.nan,'nonfinite_backward')):
            r=reg.evaluate_group([query(self.q)],self.p,self.p,lambda p:p,backward)[0]
            self.assertEqual(r['failure_reason'],reason);self.assertEqual(r['cycle_error_mm'],'')

    def test_query_outside_crop_rejected(self):
        with self.assertRaises(pt.RegistrationError):
            reg.evaluate_group([query([0,0,0])],self.p,self.p,lambda p:p,lambda p:p)

    def test_global_rint_not_local_rint(self):
        a=aligned(False);a['padded_2mm_affine']=a['source_ct']['affine'];a['valid_model_box_xyz']=[[3,3,4],[12,20,30]]
        p=reg.native_plan(a);q=[4.5,7,8]
        r=reg.evaluate_group([query(q)],p,p,lambda x:x,lambda x:x)[0]
        self.assertEqual(r['matched_raw_rounded_x'],int(np.rint(r['matched_raw_x'])))


class SafetyTests(unittest.TestCase):
    def test_no_cohort_or_approval_entrypoint(self):
        for command in ('run','approve-pilot'):
            with self.assertRaises(SystemExit):reg.parser().parse_args([command])
        self.assertEqual(reg.PILOT,'quadra_hc_044')

    def test_wrong_whole_body_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.json';pt.atomic_json(p,{'status':'TECHNICAL_PASS'})
            with self.assertRaises(pt.RegistrationError):reg.validate_source_checkpoint(p)

    def test_output_identity_and_signature_reject_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=root/'maps.json'
            pt.atomic_json(f,[{'Transform':['EulerTransform'],'HowToCombineTransforms':['Compose']},
                              {'Transform':['BSplineTransform'],'HowToCombineTransforms':['Compose']}])
            m={'signature':'sig','limits':{'ram_ceiling_bytes':1000}}
            meta={'status':'success','schema_version':1,'signature':reg.task_signature(m,'abdomen','forward'),
                  'files':[pt.identity(f)],'maps_json':pt.identity(f),'peak_rss_bytes':500}
            marker=root/'forward.json';pt.atomic_json(marker,meta)
            self.assertIsNotNone(reg.validate_result(m,'abdomen','forward',marker))
            with self.assertRaises(pt.RegistrationError):reg.validate_result(m,'pelvis','forward',marker)
            pt.atomic_json(f,[])
            with self.assertRaises(pt.RegistrationError):reg.validate_result(m,'abdomen','forward',marker)

    def test_atomic_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';pt.atomic_json(p,{},refuse=True)
            with self.assertRaises(FileExistsError):pt.atomic_json(p,{},refuse=True)

    def test_failure_classes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(reg.classify_failure(d,-11,False)['classification'],'process_signal')
            self.assertEqual(reg.classify_failure(d,1,False)['classification'],'process_exit')
            self.assertEqual(reg.classify_failure(d,0,True)['classification'],'timeout')
            pt.atomic_json(Path(d)/'worker_error.json',{'classification':'contract_error','message':'geometry'})
            self.assertEqual(reg.classify_failure(d,2,False)['classification'],'contract_error')

    def test_forbidden_output_and_backup_detection(self):
        from tools.quadra.artifact_backup import ACTIVE_PROCESS_PATTERNS
        self.assertIn('registration_organ_group',ACTIVE_PROCESS_PATTERNS)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'warped.nii.gz';pt.atomic_bytes(p,b'bad')
            self.assertEqual(reg.base.forbidden_outputs(d),[str(p)])

    def test_plot_grain_is_all_valid_points_not_medians(self):
        rows=[dict(query([1,2,3],str(i)),valid_cycle='True',cycle_error_mm=str(i+.13)) for i in range(12)]
        series=report.point_series(rows,'mask_name')
        np.testing.assert_array_equal(series[0][1],np.arange(12)+.13)
        with self.assertRaises(pt.RegistrationError):report.point_series(rows+rows,'mask_name')
        with self.assertRaises(pt.RegistrationError):
            report.point_series([dict(rows[0],cycle_error_mm='nan')],'mask_name')

    def test_resume_valid_marker_never_launches_a_worker(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(reg,'validate_result',return_value={'status':'success'}) as validate, \
             mock.patch.object(reg.subprocess,'Popen') as launch:
            result=reg.run_task(Path(d),{},'abdomen','forward')
            self.assertEqual(result,{'status':'success'})
            launch.assert_not_called();validate.assert_called_once()

    def test_resume_recovers_atomic_worker_result_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);attempt=root/'groups/abdomen/forward-attempt-123'
            attempt.mkdir(parents=True);pt.atomic_json(attempt/'worker_result.json',{})
            with mock.patch.object(reg,'validate_result',side_effect=[None,{'status':'success'}]), \
                 mock.patch.object(reg.subprocess,'Popen') as launch:
                reg.run_task(root,{},'abdomen','forward')
                launch.assert_not_called()
                self.assertEqual(pt.load_json(root/'groups/abdomen/forward.json'),{'status':'success'})

    def test_existing_scoped_worker_blocks_duplicate(self):
        fake=mock.Mock(pid=123,info={'cmdline':['python','-m',reg.MODULE,'_worker','--run-directory','/tmp/pilot']})
        with mock.patch('psutil.process_iter',return_value=[fake]):
            with self.assertRaises(pt.RegistrationError):reg.assert_no_worker(Path('/tmp/pilot'))


@unittest.skipUnless(os.environ.get('QUADRA_REGISTRATION_INTEGRATION') == '1' and importlib.util.find_spec('itk'),
                     'ITK synthetic image tests run on RunPod only')
class ITKTests(unittest.TestCase):
    def test_real_roi_preserves_voxels_and_oblique_origin(self):
        import itk
        plan=reg.native_plan(aligned())
        data=np.arange(40*30*20,dtype=np.float32).reshape(40,30,20)
        image=itk.image_from_array(data)
        a=pt.lps_affine(plan['source_ct']);spacing=np.linalg.norm(a[:3,:3],axis=0)
        image.SetSpacing(spacing);image.SetOrigin(a[:3,3]);image.SetDirection(itk.matrix_from_array(a[:3,:3]/spacing))
        result=reg.crop_image(image,plan)
        start,stop=plan['crop_start_xyz'],plan['crop_end_xyz']
        expected=data[start[2]:stop[2],start[1]:stop[1],start[0]:stop[0]].copy()
        del image,data
        np.testing.assert_array_equal(itk.array_from_image(result),expected)
        pt.check_itk_geometry(result,plan['crop_geometry'])


if __name__ == '__main__':
    unittest.main()
