#!/usr/bin/env python
"""Approved native organ-group cohort; immutable pilot, unchanged scientific kernel."""
import argparse
from collections import Counter
import functools
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.quadra import registration_organ_group as pilot
from tools.quadra import registration_cycle_error_cohort as base
from tools.quadra import registration_point_transform as pt

MODULE = "tools.quadra.registration_organ_group_cohort"
WORKFLOW = "quadra-native-organ-group-registration-cohort-v1"
PILOT_COMMIT = "2f1af308e1423688c483a5eaf0238fd3dcd3dda4"
PILOT_SIGNATURE = "7560d6aeca985ce4965b946ebf6e8897ef679f9b173000bf2c61201589101de8"
PILOT_CHECKPOINT_SHA = "156ba70c4e12e6e8fd4464fe87cf95aea6fee5d30e79f5bdceec81f2e376bf3e"
PILOT_RUN = "/workspace/quadra/runs/cohort/registration-organ-group-native-20260827T234840Z"
GROUPS = pilot.GROUPS
SUBJECTS = base.SUBJECTS
DIRECTIONS = ("forward", "backward", "points")
require = pt.require


def repository():
    value = pilot.repository()
    require(subprocess.call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", PILOT_COMMIT, "HEAD"]) == 0,
            "Missing accepted organ-group pilot ancestry")
    return value


def accepted_pilot(checkpoint):
    require(pt.identity(checkpoint)["sha256"] == PILOT_CHECKPOINT_SHA, "Wrong pilot checkpoint")
    cp = pt.load_json(checkpoint)
    require(cp["status"] == "REVIEW_REQUIRED" and cp["technical_gates_passed"] is True and
            cp["cohort_authorized"] is False and cp["manifest_signature"] == PILOT_SIGNATURE,
            "Pilot was changed or did not pass technical gates")
    pm = pilot.load_run(Path(checkpoint).parent, active=False)
    require(pm["signature"] == PILOT_SIGNATURE, "Wrong pilot manifest")
    for ref in cp["evidence"]:
        pt.verify_identity(ref)
    require(pilot.counters(Path(checkpoint).parent, pm) ==
            dict(groups_completed=4, registrations_completed=8, queries_valid=3914, queries_invalid=0),
            "Pilot completion changed")
    return pm


def validate_plan_set(plans, sources):
    expected = {s+"-"+session+"-"+g for s in SUBJECTS for session in ("test", "retest") for g in GROUPS}
    require(set(plans) == expected, "Wrong 224-plan cohort domain")
    for key, plan in plans.items():
        a = plan["source_uae_plan"]
        require(key == a["subject_id"]+"-"+a["session"]+"-"+a["group_name"], "Plan key mismatch")
        require(a["margin_mm"] == 100 and a["strategy"] == "organ_group_global_lattice", "Wrong spatial policy")
        require(plan == pilot.serializable_plan(a), "Native crop policy changed")
        require(plan["source_ct"] == sources[a["subject_id"]+"-"+a["session"]], "Source identity mismatch")


def validate_query_set(rows, source, uae, plans):
    import yaml
    registry = yaml.safe_load(pt.verify_identity(uae["registry"]).read_text())
    registry = registry["organs"] + registry["derived_organs"]
    require(len(registry) == 40, "Mask registry changed")
    buckets = base.validate_queries(rows, source["denominators"], registry)
    require(set(r["subject_id"] for r in rows) == set(SUBJECTS), "Subject set changed")
    sexes = {s:{r["sex"] for r in rows if r["subject_id"] == s} for s in SUBJECTS}
    require(all(len(v) == 1 for v in sexes.values()) and
            Counter(next(iter(v)) for v in sexes.values()) == {"M":12, "F":16}, "Sex denominator changed")
    for s in SUBJECTS:
        for g in GROUPS:
            items = [r for r in rows if r["subject_id"] == s and r["group_name"] == g]
            plan = plans[s+"-test-"+g]
            require(items and {r["mask_name"] for r in items} == set(plan["source_uae_plan"]["included_masks"]),
                    "Group mask membership changed")
            points = [[float(r["raw_"+a]) for a in "xyz"] for r in items]
            require(pilot.inside_crop(points, plan).all(), "Query outside native group crop")
    return buckets


def prepare(args):
    import nibabel as nib
    from tools.quadra.registration_runtime import preflight
    require(args.approve_pilot and args.review_rationale.strip(), "Explicit pilot approval and rationale required")
    root = args.storage_root.resolve()
    repo = repository()
    pm = accepted_pilot(args.pilot_checkpoint)
    source = pilot.validate_source_checkpoint(Path(pm["whole_body_checkpoint"]["path"]))
    require(str(root) == pm["storage_root"], "Storage root changed")
    require(preflight(root) == pm["environment"] and pt.parameter_maps() == pm["parameters"], "Runtime changed")
    base.check_resources(pm["limits"], root)
    uae = pt.load_json(pt.verify_identity(source["uae_manifest"]))
    plans = {}
    for ref in uae["outputs"]["plans"]:
        a = pt.load_json(pt.verify_identity(ref))
        key = a["subject_id"]+"-"+a["session"]+"-"+a["group_name"]
        require(key not in plans, "Duplicate plan")
        plans[key] = pilot.serializable_plan(a)
    validate_plan_set(plans, source["sources"])
    for key, ref in pm["plans"].items():
        require(plans[pilot.PILOT+"-"+key] == pt.load_json(pt.verify_identity(ref)), "Pilot plan changed")
    rows = pt.read_csv(pt.verify_identity(source["queries"]))
    buckets = validate_query_set(rows, source, uae, plans)
    expected_masks = {(key, mask) for key, plan in plans.items()
                      for mask in plan["source_uae_plan"]["included_masks"]}
    expected_masks = {(key.rsplit("-", 1)[0], mask) for key, mask in expected_masks}
    require(len(source["masks"]) == len(expected_masks) == 2208 and
            {(x["scan_key"], x["mask_name"]) for x in source["masks"]} == expected_masks, "Mask inventory changed")
    for num, (key, original) in enumerate(sorted(source["sources"].items()), 1):
        print("[{}/56] Validate native source and masks: {}".format(num, key), flush=True)
        base.check_resources(pm["limits"], root)
        image = nib.load(str(pt.verify_identity(original)))
        require(list(image.shape) == original["native_shape_xyz"] and
                np.allclose(image.affine, original["affine"], atol=1e-5, rtol=0), "CT geometry changed")
        for mask in (x for x in source["masks"] if x["scan_key"] == key):
            image = nib.load(str(pt.verify_identity(mask)))
            require(list(image.shape) == original["native_shape_xyz"] and
                    np.allclose(image.affine, original["affine"], atol=1e-5, rtol=0), "Mask geometry changed")
            if key.endswith("-test"):
                items = buckets[(key[:-5], mask["mask_name"])]
                data = np.asanyarray(image.dataobj)
                points = np.asarray([[int(r["raw_"+a]) for a in "xyz"] for r in items])
                require(np.all(data[tuple(points.T)] != 0) and
                        np.count_nonzero(data) == int(items[0]["available_unique_voxels"]), "Mask membership/quota changed")
                del data
    run = root/"runs/cohort"/("registration-organ-group-cohort-"+time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run.mkdir(parents=True, exist_ok=False)
    pt.atomic_bytes(run/"frozen_queries_raw_itk.csv", pt.verify_identity(source["queries"]).read_bytes(), refuse=True)
    refs = {}
    for key, plan in plans.items():
        path = run/"plans"/(key+".json")
        pt.atomic_json(path, plan, refuse=True)
        refs[key] = pt.identity(path)
    approval = {"schema_version":1, "cohort_authorized":True, "created_at":pt.utc_now(),
                "pilot_checkpoint":pt.identity(args.pilot_checkpoint), "pilot_manifest_signature":PILOT_SIGNATURE,
                "review_rationale":args.review_rationale.strip(), "subjects":SUBJECTS,
                "scientific_parameters_changed":False, "scope":"28-subject native organ-group registration only"}
    pt.atomic_json(run/"pilot_approval.json", approval, refuse=True)
    m = dict(schema_version=1, workflow=WORKFLOW, created_at=pt.utc_now(), run_directory=str(run),
             storage_root=str(root), repository=repo, environment=pm["environment"],
             pilot_checkpoint=pt.identity(args.pilot_checkpoint), pilot_manifest=pt.identity(Path(args.pilot_checkpoint).parent/"organ_group_manifest.json"),
             approval=pt.identity(run/"pilot_approval.json"), whole_body_manifest=pm["whole_body_manifest"],
             uae_manifest=source["uae_manifest"], sources=source["sources"], masks=source["masks"],
             queries=pt.identity(run/"frozen_queries_raw_itk.csv"), plans=refs, parameters=pm["parameters"],
             limits=pm["limits"], runtime_contract=pm["runtime_contract"], subjects=SUBJECTS,
             group_order=list(GROUPS), expected_queries=108431, expected_registrations=224, expected_groups=112,
             denominators=source["denominators"], settings=dict(pm["settings"], cohort_authorized=True,
                 retry_limit=1, isolated_failure_policy="record_failed_group_continue", pilot_reuse=pilot.PILOT))
    m["signature"] = pt.digest(m)
    pt.atomic_json(run/"organ_group_cohort_manifest.json", m, refuse=True)
    reuse_pilot(run, m, pm)
    status_update(run, status="PREPARED", controller_pid=None, worker_pid=None, error=None, **counters(run, m))
    print("Preparation PASS\nRun directory: {}\n108431 unchanged queries; pilot reused; 216 new registrations".format(run), flush=True)


def load_run(run, active=True):
    m = pt.load_json(Path(run)/"organ_group_cohort_manifest.json")
    require(m["workflow"] == WORKFLOW and m["schema_version"] == 1 and
            pt.digest({k:v for k,v in m.items() if k != "signature"}) == m["signature"], "Cohort manifest changed")
    require(m["run_directory"] == str(Path(run).resolve()) and m["subjects"] == SUBJECTS and
            m["group_order"] == list(GROUPS) and (m["expected_queries"],m["expected_groups"],m["expected_registrations"]) ==
            (108431,112,224), "Cohort domain changed")
    require(m["pilot_checkpoint"]["sha256"] == PILOT_CHECKPOINT_SHA and m["queries"]["sha256"] == base.QUERY_SHA,
            "Frozen input identity changed")
    for ref in [m[k] for k in ("pilot_checkpoint", "pilot_manifest", "approval", "whole_body_manifest", "uae_manifest", "queries")]:
        pt.verify_identity(ref)
    approval = pt.load_json(m["approval"]["path"])
    require(approval["cohort_authorized"] is True and approval["subjects"] == SUBJECTS and
            approval["pilot_checkpoint"] == m["pilot_checkpoint"] and approval["pilot_manifest_signature"] == PILOT_SIGNATURE and
            approval["review_rationale"].strip(), "Missing or incompatible cohort approval")
    pm = pt.load_json(m["pilot_manifest"]["path"])
    require(pm["signature"] == PILOT_SIGNATURE and all(m[k] == pm[k] for k in ("parameters","limits","environment","runtime_contract")),
            "Validated registration contract changed")
    require(m["settings"] == dict(pm["settings"], cohort_authorized=True, retry_limit=1,
            isolated_failure_policy="record_failed_group_continue", pilot_reuse=pilot.PILOT), "Scientific policy changed")
    source = pt.load_json(m["whole_body_manifest"]["path"])
    require(m["sources"] == source["sources"] and m["masks"] == source["masks"] and
            m["denominators"] == source["denominators"], "Frozen dataset changed")
    plans = {k:pt.load_json(pt.verify_identity(v)) for k,v in m["plans"].items()}
    validate_plan_set(plans, m["sources"])
    base.validate_runtime_contract(m)
    if active:
        from tools.quadra.registration_runtime import preflight
        require(repository() == m["repository"], "Execution repository changed")
        require(preflight(m["storage_root"]) == m["environment"] and pt.parameter_maps() == m["parameters"], "Environment/parameters changed")
        base.check_resources(m["limits"], run)
    return m


@functools.lru_cache(maxsize=2)
def query_index(path, sha):
    require(pt.identity(path)["sha256"] == sha, "Query file changed")
    rows = pt.read_csv(path)
    require(len(rows) == len({r["query_id"] for r in rows}) == 108431, "Query denominator changed")
    return {(s,g):[r for r in rows if r["subject_id"] == s and r["group_name"] == g] for s in SUBJECTS for g in GROUPS}


def rows_for(m, subject, group):
    return query_index(m["queries"]["path"], m["queries"]["sha256"])[(subject,group)]


def task_signature(m, subject, group, direction):
    return pt.digest(dict(manifest=m["signature"], subject=subject, group=group, direction=direction))


def group_dir(run, subject, group):
    require(subject in SUBJECTS and group in GROUPS, "Unknown task scope")
    return Path(run)/"subjects"/subject/"groups"/group


def validate_points(rows, expected):
    require(len(rows) == len(expected) and [r["query_id"] for r in rows] == [r["query_id"] for r in expected], "Point IDs/order changed")
    for r, q in zip(rows, expected):
        require(all(r.get(k) == v for k,v in q.items()), "Frozen point metadata changed")
        require(r["valid_cycle"] in ("True","False"), "Invalid validity flag")
        if r["valid_cycle"] == "True":
            value = float(r["cycle_error_mm"])
            delta = [float(r["returned_physical_"+a])-float(r["query_physical_"+a]) for a in "xyz"]
            require(np.isfinite(value) and value >= 0 and not r["failure_reason"] and
                    abs(float(np.linalg.norm(delta))-value) <= 1e-7, "Invalid cycle evaluation")
        else:
            require(r["cycle_error_mm"] == "" and bool(r["failure_reason"]), "Invalid observation hidden")


def validate_result(m, subject, group, direction, path):
    if not Path(path).exists():
        return None
    meta = pt.load_json(path)
    require(meta.get("status") == "success" and meta.get("schema_version") == 1 and
            meta["signature"] == task_signature(m,subject,group,direction) and
            (meta["subject_id"],meta["group_name"],meta["direction"]) == (subject,group,direction), "Incompatible task result")
    for f in meta["files"]:
        pt.verify_identity(f)
    if "reused_from" in meta:
        require(subject == pilot.PILOT, "Only approved pilot may be reused")
        old = pt.load_json(pt.verify_identity(meta["reused_from"]))
        require({k:v for k,v in meta.items() if k not in ("signature","reused_from")} ==
                {k:v for k,v in old.items() if k != "signature"}, "Reused pilot result was altered")
        require(old["signature"] == pilot.task_signature({"signature":PILOT_SIGNATURE},group,direction), "Wrong pilot result")
    require(meta["peak_rss_bytes"] <= m["limits"]["ram_ceiling_bytes"], "Saved RAM violation")
    if direction == "points":
        rows = pt.read_csv(pt.verify_identity(meta["point_csv"]))
        validate_points(rows, rows_for(m,subject,group))
        require(len(rows) == meta["queries"] == meta["valid_queries"]+meta["invalid_queries"] and
                sum(r["valid_cycle"] == "True" for r in rows) == meta["valid_queries"], "Point counts changed")
    else:
        maps = pt.load_json(pt.verify_identity(meta["maps_json"]))
        require([x["Transform"][0] for x in maps] == ["EulerTransform","BSplineTransform"] and
                all(x["HowToCombineTransforms"] == ["Compose"] for x in maps), "Incomplete transform composition")
        sessions = ("test","retest") if direction == "forward" else ("retest","test")
        for key, session in zip(("fixed_geometry","moving_geometry"), sessions):
            plan = pt.load_json(m["plans"][subject+"-"+session+"-"+group]["path"])
            require(meta[key] == plan["crop_geometry"], "Saved registration geometry changed")
    return meta


def reuse_pilot(run, m, pm):
    for g in GROUPS:
        for d in DIRECTIONS:
            path = Path(m["pilot_checkpoint"]["path"]).parent/"groups"/g/(d+".json")
            old = pilot.validate_result(pm,g,d,path)
            require(old is not None, "Pilot result absent")
            meta = dict(old, signature=task_signature(m,pilot.PILOT,g,d), reused_from=pt.identity(path))
            marker = group_dir(run,pilot.PILOT,g)/(d+".json")
            pt.atomic_json(marker,meta,refuse=True)
            validate_result(m,pilot.PILOT,g,d,marker)


def status_update(run, **fields):
    path = Path(run)/"cohort_status.json"
    value = pt.load_json(path) if path.exists() else {"workflow":WORKFLOW}
    value.update(fields, heartbeat_at=pt.utc_now())
    pt.atomic_json(path,value)


def assert_idle(run):
    import psutil
    for proc in psutil.process_iter(["cmdline"]):
        cmd = proc.info["cmdline"] or []
        if proc.pid != os.getpid() and any(x in cmd for x in (MODULE,pilot.MODULE,"tools.quadra.registration_cycle_error_cohort")):
            require(not ("_worker" in cmd or "run" in cmd or "pilot" in cmd), "Another registration process is active: {}".format(proc.pid))


def worker(args):
    m = load_run(args.run_directory)
    destination = group_dir(args.run_directory,args.subject,args.group)
    work = args.attempt_directory.resolve()
    require(work.is_relative_to(destination.resolve()) and work != destination.resolve(), "Attempt escapes task directory")
    plans = [pt.load_json(m["plans"][args.subject+"-"+s+"-"+args.group]["path"]) for s in ("test","retest")]
    directional = None
    if args.direction == "points":
        directional = [validate_result(m,args.subject,args.group,d,destination/(d+".json")) for d in ("forward","backward")]
        require(all(directional), "Missing directional registration")
    pilot.perform_task(m, work, args.subject, args.group, args.direction, plans,
                       rows_for(m,args.subject,args.group), task_signature(m,args.subject,args.group,args.direction), directional)


def run_task(run, m, subject, group, direction):
    import psutil
    destination = group_dir(run,subject,group)
    destination.mkdir(parents=True,exist_ok=True)
    marker = destination/(direction+".json")
    previous = validate_result(m,subject,group,direction,marker)
    if previous:
        return previous
    for result in sorted(destination.glob(direction+"-attempt-*/worker_result.json")):
        recovered = validate_result(m,subject,group,direction,result)
        pt.atomic_json(marker,recovered,refuse=True)
        return recovered
    failures = sorted(destination.glob(direction+"-attempt-*/controller_failure.json"))
    for f in failures:
        failure = pt.load_json(f)
        require(failure["signature"] == task_signature(m,subject,group,direction) and
                failure["classification"] != "contract_error", "Previous fatal task failure")
    for retry in range(len(failures),2):
        assert_idle(run)
        attempt = destination/(direction+"-attempt-"+str(time.time_ns()))
        attempt.mkdir()
        env = dict(os.environ,ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1",OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable,"-u","-m",MODULE,"_worker","--run-directory",str(run),"--subject",subject,
                   "--group",group,"--direction",direction,"--attempt-directory",str(attempt)]
        start, peak, timed_out = time.monotonic(),0,False
        with (attempt/"stdout.log").open("x") as log:
            proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            observed = psutil.Process(proc.pid)
            observed.cpu_percent(None)
            status_update(run,worker_pid=proc.pid,worker_create_time=observed.create_time(),current_subject=subject,
                          current_group=group,current_direction=direction,attempt_directory=str(attempt),retry=retry)
            try:
                while proc.poll() is None:
                    if time.monotonic()-start > m["limits"]["direction_timeout_seconds"]:
                        timed_out = True
                        break
                    try:
                        rss = sum(p.memory_info().rss for p in [observed]+observed.children(recursive=True) if p.is_running())
                        cpu = observed.cpu_percent(None)
                    except psutil.NoSuchProcess:
                        rss,cpu = 0,0
                    peak = max(peak,rss)
                    resources = base.check_resources(m["limits"],run,rss)
                    resources["worker_cpu_percent"] = cpu
                    status_update(run,resources=resources,worker_peak_rss_bytes=peak,progress=base.log_progress(attempt))
                    time.sleep(5)
            finally:
                base.stop_owned_process(proc)
                status_update(run,worker_pid=None)
        if proc.returncode == 0:
            meta = validate_result(m,subject,group,direction,attempt/"worker_result.json")
            require(meta is not None, "Worker exited without result")
            meta.update(peak_rss_bytes=max(peak,meta["peak_rss_bytes"]),retry_index=retry,
                        controller_wall_time_seconds=time.monotonic()-start)
            require(meta["peak_rss_bytes"] <= m["limits"]["ram_ceiling_bytes"], "Sampled RAM exceeded ceiling")
            pt.atomic_json(marker,meta,refuse=True)
            return meta
        failure = pilot.classify_failure(attempt,proc.returncode,timed_out)
        failure.update(signature=task_signature(m,subject,group,direction),retry_index=retry)
        pt.atomic_json(attempt/"controller_failure.json",failure,refuse=True)
        status_update(run,last_failure=failure)
        if failure["classification"] == "contract_error":
            raise pt.RegistrationError(failure.get("message","Contract failure"))
    raise pt.RuntimeFailure("{} {} {} failed after one retry".format(subject,group,direction))


def validate_failure(m,subject,group,path):
    if not path.exists():
        return None
    value = pt.load_json(path)
    require(value["signature"] == task_signature(m,subject,group,"failure") and value["classification"] == "isolated_runtime_failure",
            "Incompatible failed-group record")
    for f in value["evidence"]:
        pt.verify_identity(f)
    return value


def counters(run,m):
    count = dict(subjects_completed=0,groups_completed=0,registrations_completed=0,queries_valid=0,queries_invalid=0,
                 queries_failed=0,failed_groups=[],failed_subjects=[])
    for s in SUBJECTS:
        groups_done = 0
        for g in GROUPS:
            dest = group_dir(run,s,g)
            failed = validate_failure(m,s,g,dest/"failure.json")
            points = None
            for d in DIRECTIONS:
                meta = validate_result(m,s,g,d,dest/(d+".json"))
                if meta:
                    if d == "points":
                        points = meta
                        count["groups_completed"] += 1
                        groups_done += 1
                        count["queries_valid"] += meta["valid_queries"]
                        count["queries_invalid"] += meta["invalid_queries"]
                    else:
                        count["registrations_completed"] += 1
            require(not (failed and points), "Conflicting success/failure markers")
            if failed:
                count["failed_groups"].append(s+":"+g)
                count["queries_failed"] += len(rows_for(m,s,g))
        if groups_done == 4:
            count["subjects_completed"] += 1
    count["failed_subjects"] = sorted({key.split(":")[0] for key in count["failed_groups"]})
    return count


def completion_status(count):
    require(count["groups_completed"]+len(count["failed_groups"]) == 112 and
            count["queries_valid"]+count["queries_invalid"]+count["queries_failed"] == 108431, "Incomplete cohort denominator")
    if count["failed_groups"] or count["queries_invalid"]:
        return "PARTIAL"
    require(count["subjects_completed"] == 28 and count["registrations_completed"] == 224, "Missing registrations")
    return "TECHNICAL_PASS"


def finalize(run,m):
    checkpoint = run/"checkpoint_summary.json"
    if checkpoint.exists():
        cp = pt.load_json(checkpoint)
        require(cp["manifest_signature"] == m["signature"] and cp["counts"] == counters(run,m), "Completion checkpoint changed")
        for ref in cp["evidence"]:
            pt.verify_identity(ref)
        return cp
    count = counters(run,m)
    outcome = completion_status(count)
    require(repository() == m["repository"] and not base.forbidden_outputs(run), "Final integrity check failed")
    status_update(run,status="FINALIZING",**count)
    from tools.quadra.registration_organ_group_cohort_report import build_report
    report = build_report(run,m,count,outcome)
    evidence = [pt.identity(p) for p in sorted(run.rglob("*")) if p.is_file() and
                p.name not in ("controller.lock","controller.log","cohort_status.json")]
    cp = dict(schema_version=1,workflow=WORKFLOW,status=outcome,created_at=pt.utc_now(),
              manifest_signature=m["signature"],counts=count,report=pt.identity(report),evidence=evidence,
              interpretation="Technical execution and cycle consistency, not anatomical accuracy")
    pt.atomic_json(checkpoint,cp,refuse=True)
    return cp


def execute(args):
    import psutil
    run = args.run_directory.resolve()
    with base.controller_lock(run):
        m = load_run(run)
        assert_idle(run)
        if (run/"checkpoint_summary.json").exists():
            print(finalize(run,m)["status"],flush=True)
            return
        status_update(run,status="RUNNING",controller_pid=os.getpid(),controller_create_time=psutil.Process().create_time(),
                      worker_pid=None,error=None,**counters(run,m))
        outcome = "INCOMPLETE"
        try:
            for s in SUBJECTS:
                for g in GROUPS:
                    dest = group_dir(run,s,g)
                    if validate_failure(m,s,g,dest/"failure.json"):
                        continue
                    try:
                        for d in DIRECTIONS:
                            print("{} {} {}".format(s,g,d),flush=True)
                            run_task(run,m,s,g,d)
                            status_update(run,**counters(run,m))
                    except pt.RuntimeFailure as exc:
                        evidence = [pt.identity(p) for p in sorted(dest.glob("*-attempt-*/controller_failure.json"))]
                        require(len(evidence) >= 2, "Isolated failure lacks retry evidence")
                        pt.atomic_json(dest/"failure.json",dict(signature=task_signature(m,s,g,"failure"),
                            classification="isolated_runtime_failure",message=str(exc),evidence=evidence,created_at=pt.utc_now()),refuse=True)
                        status_update(run,**counters(run,m))
            cp = finalize(run,m)
            outcome = cp["status"]
        except pt.RegistrationError as exc:
            outcome = "BLOCKED"
            status_update(run,error=str(exc))
            raise
        except BaseException as exc:
            status_update(run,error=str(exc))
            raise
        finally:
            status_update(run,status=outcome,controller_pid=None,worker_pid=None)
        print("{}\nRun directory: {}\nReport: {}".format(outcome,run,cp["report"]["path"]),flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--storage-root",type=Path,default=Path("/workspace/quadra"))
    prep.add_argument("--pilot-checkpoint",type=Path,default=Path(PILOT_RUN)/"pilot_checkpoint.json")
    prep.add_argument("--approve-pilot",action="store_true")
    prep.add_argument("--review-rationale",required=True)
    for name in ("run","status","finalize","_worker"):
        item = sub.add_parser(name)
        item.add_argument("--run-directory",type=Path,required=True)
        if name == "status":
            item.add_argument("--json",action="store_true")
        if name == "_worker":
            item.add_argument("--subject",choices=SUBJECTS,required=True)
            item.add_argument("--group",choices=GROUPS,required=True)
            item.add_argument("--direction",choices=DIRECTIONS,required=True)
            item.add_argument("--attempt-directory",type=Path,required=True)
    return p


def main(argv=None):
    import json
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "run":
            execute(args)
        elif args.command == "_worker":
            worker(args)
        elif args.command == "finalize":
            with base.controller_lock(args.run_directory):
                assert_idle(args.run_directory)
                m = load_run(args.run_directory)
                cp = finalize(args.run_directory,m)
                status_update(args.run_directory,status=cp["status"],controller_pid=None,worker_pid=None)
                print(cp["status"])
        else:
            print(json.dumps(pt.load_json(args.run_directory/"cohort_status.json"),indent=2))
        return 0
    except Exception as exc:
        traceback.print_exc()
        if args.command == "_worker":
            pt.atomic_json(args.attempt_directory/"worker_error.json",dict(classification="contract_error" if isinstance(exc,pt.RegistrationError)
                else "runtime_error",message=str(exc)),refuse=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
