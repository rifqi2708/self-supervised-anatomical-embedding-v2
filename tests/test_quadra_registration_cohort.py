"""Lightweight synthetic checks; ITK execution is opt-in on RunPod only."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools.quadra import registration_point_transform as points
from tools.quadra import registration_cycle_error_cohort as cohort
from tools.quadra import registration_report as report
from tools.quadra import registration_runtime as runtime


def source():
    angle = .31
    rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0,0,1.]])
    affine = np.eye(4)
    affine[:3,:3] = rot.dot(np.diag([1.2,2.3,3.4]))
    affine[:3,3] = [12.5,-20.3,100.1]
    return {"native_shape_xyz":[20,30,40],"affine":affine.tolist()}


def query(x=5,y=6,z=7):
    return {"query_id":"q1","raw_x":str(x),"raw_y":str(y),"raw_z":str(z),
            "subject_id":"quadra_hc_044","group_name":"abdomen","mask_name":"liver"}


class PointTests(unittest.TestCase):
    def test_anisotropic_oblique_roundtrip(self):
        result = points.geometry_checks(source())
        self.assertLess(result["max_roundtrip_voxels"],1e-6)
        self.assertLess(result["max_roundtrip_mm"],1e-5)

    def test_ras_lps_and_xyz(self):
        s = source()
        raw = np.array([[3,4,5.]])
        ras = points.apply_affine(raw,s["affine"])
        lps = points.apply_affine(raw,points.lps_affine(s))
        np.testing.assert_allclose(ras*[-1,-1,1],lps)
        self.assertFalse(np.allclose(lps,points.apply_affine(raw[:,::-1],points.lps_affine(s))))

    def test_identity_cycle(self):
        result = points.evaluate_cycle([query()],source(),source(),lambda p:p,lambda p:p)[0]
        self.assertTrue(result["valid_cycle"])
        self.assertLess(result["cycle_error_mm"],1e-10)

    def test_fractional_forward_is_not_rounded(self):
        s = source()
        shift = np.array([.371,-.237,.113])
        received = []
        def reverse(p):
            received.append(p.copy())
            return p-shift
        result = points.evaluate_cycle([query()],s,s,lambda p:p+shift,reverse)[0]
        expected = points.apply_affine([[5,6,7]],points.lps_affine(s))+shift
        np.testing.assert_array_equal(received[0],expected)
        self.assertLess(result["cycle_error_mm"],1e-10)
        self.assertNotEqual(result["matched_raw_x"],result["matched_raw_rounded_x"])

    def test_outside_forward_never_calls_reverse(self):
        backward = mock.Mock()
        row = points.evaluate_cycle([query()],source(),source(),lambda p:p+10000,backward)[0]
        backward.assert_not_called()
        self.assertFalse(row["valid_cycle"])
        self.assertEqual(row["cycle_error_mm"],"")
        self.assertEqual(row["failure_reason"],"forward_outside_retest_fov")
        self.assertGreater(abs(row["matched_raw_x"]),20)

    def test_nonfinite_and_return_outside_rows_are_retained(self):
        for fn,reason in ((lambda p:p*np.nan,"nonfinite_backward"),
                          (lambda p:p+10000,"returned_outside_test_fov")):
            row = points.evaluate_cycle([query()],source(),source(),lambda p:p,fn)[0]
            self.assertEqual(row["failure_reason"],reason)
            self.assertEqual(row["cycle_error_mm"],"")
        row = points.evaluate_cycle([query()],source(),source(),lambda p:p*np.nan,lambda p:p)[0]
        self.assertEqual(row["failure_reason"],"nonfinite_forward")

    def test_no_silent_fov_clipping(self):
        values = [[0,0,0],[19,29,39],[-.001,0,0],[19.001,0,0]]
        self.assertEqual(points.inside(values,[20,30,40]).tolist(),[True,True,False,False])

    def test_point_parser_uses_physical_not_integer_indices(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"outputpoints.txt"
            p.write_text("Point 0 ; OutputIndexFixed = [ 1 2 3 ] ; OutputPoint = [ 1.25 -2.5 3.75 ] ;\n")
            np.testing.assert_array_equal(points.parse_output_points(p,1),[[1.25,-2.5,3.75]])
            with self.assertRaises(points.RegistrationError): points.parse_output_points(p,2)
            p.write_text("Point 1 ; OutputPoint = [ 1 2 3 ] ;\n")
            with self.assertRaises(points.RegistrationError): points.parse_output_points(p,1)


class CacheAccountingTests(unittest.TestCase):
    def sample(self, root, version=1, usage=900, stats=None):
        if stats is None:
            stats = {"total_inactive_file":750, "total_cache":800,
                     "total_dirty":20, "total_writeback":10}
        (root/"memory.stat").write_text("\n".join("{} {}".format(k,v) for k,v in stats.items()))
        usage_path = root/("memory.usage_in_bytes" if version == 1 else "memory.current")
        usage_path.write_text(str(usage))
        return cohort.cgroup_memory_sample(root,version,1000,usage_path)

    def limits(self, root, host_available=1500):
        (root/"memory.limit_in_bytes").write_text("1000")
        with mock.patch.object(cohort,"cgroup_directories",return_value=[root]), \
             mock.patch.object(cohort.os,"sched_getaffinity",return_value={0,1},create=True), \
             mock.patch("psutil.virtual_memory",return_value=mock.Mock(total=2000,available=host_available)):
            return cohort.resource_limits()

    def guard(self, limits, rss=100):
        with mock.patch.object(cohort,"resource_limits",return_value=limits), \
             mock.patch("shutil.disk_usage",return_value=mock.Mock(free=100*1024**3)):
            return cohort.check_resources(limits,"/tmp",rss)

    def test_cache_heavy_low_working_memory_passes_and_reports_both(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.sample(root)
            limits = self.limits(root)
            self.assertEqual(limits["raw_available_ram_bytes"],100)
            self.assertEqual(limits["available_ram_bytes"],820)
            self.assertEqual(limits["ram_ceiling_bytes"],800)
            result = self.guard(limits)
            self.assertEqual(result["memory_accounting"][0]["reclaimable_file_estimate_bytes"],720)
            self.assertEqual(result["memory_accounting"][0]["working_set_estimate_bytes"],180)
            self.assertEqual(result["memory_accounting_policy"],cohort.MEMORY_ACCOUNTING_POLICY)

    def test_real_working_pressure_still_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.sample(root,stats={"total_inactive_file":0,"total_cache":0,
                                    "total_dirty":0,"total_writeback":0})
            with self.assertRaisesRegex(points.RegistrationError,"headroom"):
                self.guard(self.limits(root))

    def test_process_ceiling_not_relaxed_by_cache_discount(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.sample(root)
            with self.assertRaisesRegex(points.RegistrationError,"RAM guard"):
                self.guard(self.limits(root),rss=801)

    def test_host_pressure_still_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.sample(root)
            limits = self.limits(root,host_available=100)
            self.assertEqual(limits["available_ram_bytes"],100)
            with self.assertRaisesRegex(points.RegistrationError,"headroom"):
                self.guard(limits)

    def test_v2_uses_its_own_fields_and_excludes_dirty_writeback(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.sample(Path(t),version=2,stats={"inactive_file":750,"file":800,
                           "file_dirty":20,"file_writeback":10,"total_inactive_file":900})
            self.assertEqual(s["reclaimable_file_estimate_bytes"],720)

    def test_v1_hierarchical_counters_not_local_values(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.sample(Path(t),stats={"total_inactive_file":100,"total_cache":500,
                           "total_dirty":20,"total_writeback":10,"inactive_file":900,"cache":900})
            self.assertEqual(s["reclaimable_file_estimate_bytes"],70)

    def test_cache_and_usage_caps_and_nonnegative_discount(self):
        with tempfile.TemporaryDirectory() as t:
            for inactive,cache,dirty,writeback,expected in [
                (950,300,0,0,300),(950,1000,0,0,900),(500,800,400,200,0),(0,800,0,0,0)]:
                s = self.sample(Path(t),stats={"total_inactive_file":inactive,"total_cache":cache,
                                "total_dirty":dirty,"total_writeback":writeback})
                self.assertEqual(s["reclaimable_file_estimate_bytes"],expected)

    def test_missing_or_malformed_stats_cannot_bypass_headroom(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.sample(root)
            for text in (None,"","total_inactive_file 900\n", "total_inactive_file -1\n",
                         "total_inactive_file bad\n", "total_inactive_file 900\ntotal_inactive_file 900\n"):
                with self.subTest(text=text):
                    if text is None:
                        (root/"memory.stat").unlink()
                    else:
                        (root/"memory.stat").write_text(text)
                    limits = self.limits(root)
                    self.assertEqual(limits["available_ram_bytes"],100)
                    self.assertEqual(limits["memory_accounting"][0]["reclaimable_file_estimate_bytes"],0)
                    with self.assertRaisesRegex(points.RegistrationError,"headroom"):
                        self.guard(limits)

    def test_missing_or_invalid_usage_blocks(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            with self.assertRaisesRegex(points.RegistrationError,"Missing cgroup"):
                cohort.cgroup_memory_sample(root,1,1000,root/"absent")
            for text in ("bad","-1"):
                with self.assertRaisesRegex(points.RegistrationError,"Invalid cgroup"):
                    self.sample(root,usage=text)

    def test_busy_ancestor_cannot_be_hidden_by_child_cache(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            child = root/"child"
            child.mkdir()
            self.sample(root,stats={"total_inactive_file":0,"total_cache":0,
                                   "total_dirty":0,"total_writeback":0})
            self.sample(child)
            for p in (root,child): (p/"memory.limit_in_bytes").write_text("1000")
            with mock.patch.object(cohort,"cgroup_directories",return_value=[root,child]), \
                 mock.patch.object(cohort.os,"sched_getaffinity",return_value={0,1},create=True), \
                 mock.patch("psutil.virtual_memory",return_value=mock.Mock(total=2000,available=1500)):
                limits = cohort.resource_limits()
            self.assertEqual(limits["available_ram_bytes"],100)
            with self.assertRaisesRegex(points.RegistrationError,"headroom"):
                self.guard(limits)

    def test_frozen_accounting_policy_cannot_change(self):
        with mock.patch.object(cohort,"resource_limits",return_value={"memory_accounting_policy":"different"}):
            with self.assertRaisesRegex(points.RegistrationError,"accounting policy changed"):
                cohort.check_resources({"memory_accounting_policy":cohort.MEMORY_ACCOUNTING_POLICY},"/tmp")


class SafetyTests(unittest.TestCase):
    def test_container_rooted_v1_limits_not_host_resources(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            membership = root/"membership"
            membership.write_text("8:memory:/docker/example\n11:cpu,cpuacct:/docker/example\n")
            for folder, files in {
                "memory": {"memory.limit_in_bytes":"512000000", "memory.usage_in_bytes":"32000000"},
                "cpu,cpuacct": {"cpu.cfs_quota_us":"50000", "cpu.cfs_period_us":"100000"},
            }.items():
                (root/folder).mkdir()
                for name, value in files.items(): (root/folder/name).write_text(value)
            paths = cohort.cgroup_directories(root, membership)
            self.assertIn(root/"cpu,cpuacct", paths)
            with mock.patch.object(cohort, "cgroup_directories", return_value=paths), \
                 mock.patch.object(cohort.os, "sched_getaffinity", return_value=set(range(96)), create=True), \
                 mock.patch("psutil.virtual_memory", return_value=mock.Mock(total=512*1024**3, available=400*1024**3)):
                limits = cohort.resource_limits()
            self.assertEqual(limits["effective_ram_bytes"], 512000000)
            self.assertEqual(limits["ram_ceiling_bytes"], 409600000)
            self.assertEqual(limits["available_ram_bytes"], 480000000)
            self.assertEqual(limits["effective_cpu_count"], .5)
            self.assertEqual(limits["threads"], 1)

    def test_nested_v2_limits_and_ancestor_headroom(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            membership = root/"membership"
            membership.write_text("0::/parent/child\n")
            (root/"parent/child").mkdir(parents=True)
            for folder, maximum, used, cpu in [
                (root, "max", "100", "max 100000"),
                (root/"parent", "1000", "950", "200000 100000"),
                (root/"parent/child", "800", "100", "150000 100000"),
            ]:
                (folder/"memory.max").write_text(maximum)
                (folder/"memory.current").write_text(used)
                (folder/"cpu.max").write_text(cpu)
            paths = cohort.cgroup_directories(root, membership)
            with mock.patch.object(cohort, "cgroup_directories", return_value=paths), \
                 mock.patch.object(cohort.os, "sched_getaffinity", return_value=set(range(8)), create=True), \
                 mock.patch("psutil.virtual_memory", return_value=mock.Mock(total=2000, available=1500)):
                limits = cohort.resource_limits()
            self.assertEqual(limits["effective_ram_bytes"], 800)
            self.assertEqual(limits["effective_cpu_count"], 1.5)
            self.assertEqual(limits["available_ram_bytes"], 50)
            with mock.patch.object(cohort, "resource_limits", return_value=limits), \
                 mock.patch("shutil.disk_usage", return_value=mock.Mock(free=100*1024**3)):
                with self.assertRaisesRegex(points.RegistrationError, "headroom"):
                    cohort.check_resources(limits, root)

    def test_native_pair_lower_bound_rejects_tiny_allocation(self):
        sources = {"quadra_hc_044-"+s: {"native_shape_xyz":[512,512,531]} for s in ("test","retest")}
        with self.assertRaisesRegex(points.RegistrationError, "inputs alone"):
            cohort.require_native_pair_capacity(sources, {"ram_ceiling_bytes":409600000})
        cohort.require_native_pair_capacity(sources, {"ram_ceiling_bytes":2*1024**3})

    def test_fractional_cpu_allocation_decrease_rejected(self):
        limits = {"effective_ram_bytes":1000, "threads":1, "effective_cpu_count":1.5}
        with mock.patch.object(cohort, "resource_limits", return_value=dict(limits, effective_cpu_count=1.0)):
            with self.assertRaisesRegex(points.RegistrationError, "CPU quota"):
                cohort.check_resources(limits, "/tmp")

    def test_atomic_bytes_preserves_crlf_and_refuses_replace(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"q.csv"
            value = b"a,b\r\n1,2\r\n"
            points.atomic_bytes(p,value,True)
            self.assertEqual(p.read_bytes(),value)
            with self.assertRaises(FileExistsError): points.atomic_bytes(p,b"bad",True)
            self.assertEqual(p.read_bytes(),value)

    def test_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"x"
            p.write_text("good")
            identity = points.identity(p)
            p.write_text("changed")
            with self.assertRaises(points.RegistrationError): points.verify_identity(identity)

    def test_duplicate_controller(self):
        with tempfile.TemporaryDirectory() as t:
            with cohort.controller_lock(t):
                with self.assertRaises(points.RegistrationError):
                    with cohort.controller_lock(t): pass

    def test_orphan_worker_blocks_resume(self):
        proc = mock.Mock(pid=42,info={"cmdline":["python","-m","tools.quadra.registration_cycle_error_cohort",
                                               "_worker","--run-directory","/run"]})
        with mock.patch("psutil.process_iter",return_value=[proc]):
            with self.assertRaises(points.RegistrationError): cohort.assert_no_orphan_worker("/run")

    def test_parameter_overrides(self):
        for stage in ("rigid","bspline"):
            p = points.OVERRIDES[stage]
            for key,value in {"NumberOfResolutions":"4","MaximumNumberOfIterations":"256",
                              "NumberOfSpatialSamples":"8192","RandomSeed":"121212",
                              "WriteResultImage":"false","ImageSampler":"RandomCoordinate",
                              "NewSamplesEveryIteration":"true","DefaultPixelValue":"-1024"}.items():
                self.assertEqual(p[key],value)
        self.assertEqual(points.OVERRIDES["bspline"]["FinalGridSpacingInPhysicalUnits"],"32")
        self.assertEqual(points.OVERRIDES["rigid"]["AutomaticTransformInitializationMethod"],"GeometricalCenter")

    def test_resource_disk_and_ram_guards(self):
        limits = {"effective_ram_bytes":1000,"ram_ceiling_bytes":800,"threads":8,"min_disk_free_bytes":100}
        with mock.patch.object(cohort,"resource_limits",return_value=limits), \
             mock.patch("shutil.disk_usage",return_value=mock.Mock(free=50)):
            with self.assertRaisesRegex(points.RegistrationError,"Disk"): cohort.check_resources(limits,"/tmp")
        with mock.patch.object(cohort,"resource_limits",return_value=limits), \
             mock.patch("shutil.disk_usage",return_value=mock.Mock(free=200)):
            with self.assertRaisesRegex(points.RegistrationError,"RAM guard"): cohort.check_resources(limits,"/tmp",801)

    def test_volume_quota_overrides_shared_filesystem_free(self):
        limits = {"effective_ram_bytes":1000,"ram_ceiling_bytes":800,"threads":1,
                  "min_disk_free_bytes":100,"workspace_capacity_bytes":500,"workspace_path":"/workspace"}
        with mock.patch.object(cohort,"resource_limits",return_value=limits), \
             mock.patch("shutil.disk_usage",return_value=mock.Mock(free=10**15)), \
             mock.patch.object(cohort,"workspace_usage_bytes",return_value=450):
            with self.assertRaisesRegex(points.RegistrationError,"Disk"):
                cohort.check_resources(limits,"/workspace/run")

    def test_forbidden_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/"dvf.mha").touch()
            (Path(t)/"parameters.txt").touch()
            self.assertEqual(len(cohort.forbidden_outputs(t)),1)

    def test_signature_and_resume_rejection(self):
        m = {"signature":"contract","limits":{"ram_ceiling_bytes":100}}
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"forward.json"
            points.atomic_json(p,{"signature":"wrong","status":"success"})
            with self.assertRaises(points.RegistrationError): cohort.validate_result(m,"s","forward",p)
            self.assertIsNone(cohort.validate_result(m,"s","forward",Path(t)/"missing.json"))

    def test_registration_patterns_in_backup(self):
        from tools.quadra.artifact_backup import ACTIVE_PROCESS_PATTERNS
        self.assertIn("registration_cycle_error_cohort",ACTIVE_PROCESS_PATTERNS)
        self.assertIn("registration_runtime",ACTIVE_PROCESS_PATTERNS)

    def test_all_commands_parse(self):
        for cmd in ("pilot","run","finalize","status"):
            args = cohort.build_parser().parse_args([cmd,"--run-directory","/run"])
            self.assertEqual(args.command,cmd)
        args = cohort.build_parser().parse_args(["approve-pilot","--run-directory","/run","--review-rationale","not approved"])
        self.assertFalse(args.confirm_review)

    def test_runtime_pins(self):
        self.assertEqual(runtime.PINS["itk-elastix"],"0.25.2")
        self.assertEqual(runtime.PINS["itk"],"5.4.5")
        self.assertEqual(runtime.PINS["numpy"],"1.26.4")

    def test_registration_pod_identity_is_explicit_and_bounded(self):
        with mock.patch.dict(os.environ, RUNPOD_POD_ID="2ohlzqc00kd7sn"):
            self.assertEqual(runtime.verify_pod("2ohlzqc00kd7sn"), "2ohlzqc00kd7sn")
            with self.assertRaises(points.RegistrationError): runtime.verify_pod("1ngcj5dw1mifiw")
        with mock.patch.dict(os.environ, RUNPOD_POD_ID="unapproved-pod"):
            with self.assertRaises(points.RegistrationError): runtime.verify_pod()

    def test_dirty_repository_and_wrong_ancestry_block(self):
        with mock.patch.object(cohort.subprocess,"call",return_value=1):
            with self.assertRaisesRegex(points.RegistrationError,"ancestry"): cohort.repository()
        with mock.patch.object(cohort.subprocess,"call",return_value=0), \
             mock.patch.object(cohort.subprocess,"check_output",return_value=" M changed.py"):
            with self.assertRaisesRegex(points.RegistrationError,"dirty"): cohort.repository()

    def test_small_mask_and_duplicate_query_rules(self):
        rows = [dict(query(x=i),query_id="s:liver:{:03d}".format(i),point_index=str(i),
                     mask_registry_index="0",sampling_seed="20260721",available_unique_voxels="2",
                     sampled_points_for_mask="2",mask_query_shortfall="98",sampling_policy="all_available")
                for i in range(2)]
        registry = [{"filename":"liver"}]
        cohort.validate_mask_queries("s","liver",rows,registry)
        rows[1]["raw_x"] = rows[0]["raw_x"]
        with self.assertRaisesRegex(points.RegistrationError,"Duplicate"): cohort.validate_mask_queries("s","liver",rows,registry)

    def test_pilot_approval_requires_explicit_review(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            points.atomic_json(p/"pilot_checkpoint.json",{"status":"REVIEW_REQUIRED","technical_gates_passed":True,
                "manifest_signature":"sig","evidence":[],"report":points.identity(Path(__file__))})
            args = argparse.Namespace(run_directory=p,confirm_review=False,review_rationale="A sufficiently long rationale")
            with mock.patch.object(cohort,"load_run",return_value={"signature":"sig"}):
                with self.assertRaises(points.RegistrationError): cohort.approve(args)
                args.confirm_review = True
                cohort.approve(args)
                self.assertTrue((p/"pilot_approval.json").is_file())
                with self.assertRaises(FileExistsError): cohort.approve(args)

    def test_timeout_retries_once_and_preserves_both_failures(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            m = {"signature":"s","limits":{"threads":1,"direction_timeout_seconds":1}}
            proc = mock.Mock(pid=123,returncode=-15)
            proc.poll.return_value = None
            pp = mock.Mock()
            pp.create_time.return_value = 1.
            with mock.patch.object(cohort,"assert_no_orphan_worker"), \
                 mock.patch("psutil.Process",return_value=pp), \
                 mock.patch.object(cohort.subprocess,"Popen",return_value=proc) as launch, \
                 mock.patch.object(cohort,"stop_owned_process") as stop, \
                 mock.patch.object(cohort.time,"monotonic",side_effect=[0,2,3,5]):
                with self.assertRaises(points.RuntimeFailure): cohort.run_task(root,m,"s","forward")
                self.assertEqual(launch.call_count,2)
                self.assertEqual(stop.call_count,2)
            files = list(root.glob("subjects/s/forward-attempt-*/controller_failure.json"))
            self.assertEqual(len(files),2)
            self.assertTrue(all(points.load_json(p)["classification"] == "timeout" for p in files))

    def test_resource_guard_never_retries(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            m = {"signature":"s","limits":{"threads":1,"direction_timeout_seconds":100}}
            proc = mock.Mock(pid=123)
            proc.poll.return_value = None
            pp = mock.Mock()
            pp.create_time.return_value = 1.
            pp.children.return_value = []
            pp.memory_info.return_value.rss = 10
            with mock.patch.object(cohort,"assert_no_orphan_worker"), \
                 mock.patch.object(cohort.subprocess,"Popen",return_value=proc) as launch, \
                 mock.patch.object(cohort,"stop_owned_process") as stop, \
                 mock.patch("psutil.Process",return_value=pp), \
                 mock.patch.object(cohort,"check_resources",side_effect=points.RegistrationError("RAM guard failed")):
                with self.assertRaises(points.RegistrationError): cohort.run_task(root,m,"s","forward")
                self.assertEqual(launch.call_count,1)
                stop.assert_called_once_with(proc)

    def test_report_statistics_exclude_invalid_without_zero(self):
        rows = [dict(query(),valid_cycle="True",cycle_error_mm=v) for v in (1,3,5)]
        rows.append(dict(query(),valid_cycle="False",cycle_error_mm=""))
        result = report.summaries(rows,[])[0]
        self.assertEqual(result["expected_queries"],4)
        self.assertEqual(result["valid_queries"],3)
        self.assertEqual(result["invalid_queries"],1)
        self.assertEqual(result["median_mm"],3.)

    def test_report_denominators_and_repeat_publication(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            rows = [dict(query(),valid_cycle="True",cycle_error_mm="2",failure_reason=""),
                    dict(query(),query_id="q2",valid_cycle="False",cycle_error_mm="",failure_reason="forward_outside_retest_fov")]
            points.atomic_csv(root/"points.csv",rows)
            points.atomic_json(root/"cohort_status.json",{"status":"RUNNING"})
            meta = {"point_csv":points.identity(root/"points.csv"),"wall_time_seconds":1,
                    "peak_rss_bytes":100,"files":[]}
            m = {"denominators":{"subject_query_counts":{cohort.PILOT:2}},"repository":{"commit":"test"},
                 "signature":"test","queries":{"sha256":"test"}}
            with mock.patch.object(cohort,"validate_result",side_effect=lambda m,s,d,p:meta if d == "points" else None), \
                 mock.patch.object(report,"plots",return_value=[]):
                first = report.build_report(root,m,True)
                second = report.build_report(root,m,True)
            self.assertNotEqual(first,second)
            self.assertIn("REVIEW_REQUIRED",first.read_text())
            table = points.read_csv(first.parent/"pooled_summary.csv")[0]
            self.assertEqual((table["expected_queries"],table["valid_queries"],table["invalid_queries"]),("2","1","1"))
            self.assertEqual(table["median_mm"],"2.0")
            evidence = points.load_json(first.parent/"analysis_manifest.json")
            for item in evidence["outputs"]: points.verify_identity(item)


@unittest.skipUnless(os.environ.get("QUADRA_REGISTRATION_INTEGRATION") == "1" and
                     os.environ.get("RUNPOD_POD_ID") in runtime.APPROVED_PODS,
                     "Library-dependent synthetic registration runs on RunPod only")
class RunPodIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import itk
        cls.temp = tempfile.TemporaryDirectory()
        z,y,x = np.indices((24,24,24))
        data = np.exp(-((x-11)**2+(y-12)**2+(z-10)**2)/20).astype(np.float32)
        cls.image = itk.image_from_array(data)
        cls.image.SetSpacing([1.2,1.3,1.4])
        cls.image.SetOrigin([10.,20.,30.])
        maps = points.parameter_maps()
        for m in maps:
            m.update(NumberOfResolutions=["1"],MaximumNumberOfIterations=["0"],
                     NumberOfSpatialSamples=["128"],FinalGridSpacingInPhysicalUnits=["8"])
            if "GridSpacingSchedule" in m:
                m["GridSpacingSchedule"] = ["1"]
        filt = itk.ElastixRegistrationMethod.New(cls.image,cls.image)
        filt.SetParameterObject(points.parameter_object(maps))
        filt.SetOutputDirectory(cls.temp.name)
        filt.SetLogToConsole(False)
        filt.SetNumberOfThreads(1)
        filt.UpdateLargestPossibleRegion()
        cls.maps = points.normalized_transform_maps(filt.GetTransformParameterObject())
        cls.test_points = np.array([[16.25,27.1,39.37],[19.1,32.,42.]])

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_identity_and_transform_reload(self):
        output = points.transformix_points(self.test_points,self.maps)
        np.testing.assert_allclose(output,self.test_points,atol=1e-4,rtol=0)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"maps.json"
            points.atomic_json(p,self.maps)
            np.testing.assert_array_equal(points.transformix_points(self.test_points,points.load_json(p)),output)

    def test_composed_translation_and_fractional_reverse(self):
        forward,backward = copy.deepcopy(self.maps),copy.deepcopy(self.maps)
        forward[0]["TransformParameters"] = ["0","0","0",".375","-.625",".25"]
        backward[0]["TransformParameters"] = ["0","0","0","-.375",".625","-.25"]
        shifted = points.transformix_points(self.test_points,forward)
        np.testing.assert_allclose(shifted,self.test_points+[.375,-.625,.25],atol=1e-4,rtol=0)
        np.testing.assert_allclose(points.transformix_points(shifted,backward),self.test_points,atol=1e-4,rtol=0)

    def test_independent_geometric_initialization_direction(self):
        import itk
        moving = itk.image_from_array(itk.array_from_image(self.image))
        moving.SetSpacing(self.image.GetSpacing())
        shift = np.array([.375,-.625,.25])
        moving.SetOrigin(np.asarray(self.image.GetOrigin())+shift)
        transforms = []
        for fixed,other in ((self.image,moving),(moving,self.image)):
            with tempfile.TemporaryDirectory() as t:
                maps = points.parameter_maps()
                for m in maps:
                    m.update(NumberOfResolutions=["1"],MaximumNumberOfIterations=["0"],
                             NumberOfSpatialSamples=["128"],FinalGridSpacingInPhysicalUnits=["8"])
                    if "GridSpacingSchedule" in m:
                        m["GridSpacingSchedule"] = ["1"]
                filt = itk.ElastixRegistrationMethod.New(fixed,other)
                filt.SetParameterObject(points.parameter_object(maps))
                filt.SetOutputDirectory(t)
                filt.SetLogToConsole(False)
                filt.SetNumberOfThreads(1)
                filt.UpdateLargestPossibleRegion()
                transforms.append(points.normalized_transform_maps(filt.GetTransformParameterObject()))
        forward = points.transformix_points(self.test_points,transforms[0])
        np.testing.assert_allclose(forward,self.test_points+shift,atol=1e-4,rtol=0)
        np.testing.assert_allclose(points.transformix_points(forward,transforms[1]),self.test_points,atol=1e-4,rtol=0)

    def test_tiny_dvf_reference_at_voxel_centres(self):
        import itk
        maps = copy.deepcopy(self.maps)
        maps[0]["TransformParameters"] = ["0","0","0",".375","-.625",".25"]
        with tempfile.TemporaryDirectory() as t:
            directory = Path(t)
            maps[1]["InitialTransformParameterFileName"] = [str(directory/"TransformParameters.0.txt")]
            points.save_transform_chain(directory,maps)
            obj = itk.ParameterObject.New()
            obj.ReadParameterFile(str(directory/"TransformParameters.1.txt"))
            filt = itk.TransformixFilter.New(self.image)
            filt.SetTransformParameterObject(obj)
            filt.SetOutputDirectory(t)
            filt.SetComputeDeformationField(True)
            filt.SetLogToConsole(False)
            filt.UpdateLargestPossibleRegion()
            dvf = itk.array_from_image(filt.GetOutputDeformationField())
            indices = np.array([[5,6,7],[8,9,10]])
            physical = indices*np.asarray(self.image.GetSpacing())+np.asarray(self.image.GetOrigin())
            transformed = points.transformix_points(physical,maps)
            np.testing.assert_allclose(transformed-physical,dvf[tuple(indices.T[::-1])],atol=1e-4,rtol=0)


if __name__ == "__main__":
    unittest.main()
