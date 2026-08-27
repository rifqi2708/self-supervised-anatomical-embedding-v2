#!/usr/bin/env python
"""Native organ-group registration pilot; deliberately no cohort/approval CLI.

Consumes immutable UAE-S regions and the completed whole-body registration
contract. Masks localize each scan independently; they are not metric masks.
"""
import argparse
from collections import Counter
import itertools
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

from tools.quadra import registration_cycle_error_cohort as base
from tools.quadra import registration_point_transform as pt

MODULE = "tools.quadra.registration_organ_group"
WORKFLOW = "quadra-native-organ-group-registration-pilot-v1"
BASE_COMMIT = "391926019cc0138c4f9bb1975acbd3861bbb7ceb"
WHOLE_CHECKPOINT_SHA = "ebb940a3bf7e78b65f070d9452baad0b8bbf5e0ac1f001b59887dda51e6cf3fa"
WHOLE_MANIFEST_SHA = "48b9e6ebaa24b266578133ae33ab1a6f7f1486ff87d9ce6fdbddca7d842c8d83"
WHOLE_RUN = "/workspace/quadra/runs/cohort/registration-rigid-bspline-continuous-20260827T083336Z"
PILOT = "quadra_hc_044"
GROUPS = ("pelvis", "abdomen", "thorax", "head_neck")
QUERY_COUNT = 3914
require = pt.require


def native_plan(aligned):
    """Map valid UAE voxel-cell bounds to native cells, outward and FOV-clamped."""
    source = aligned["source_ct"]
    raw = np.asarray(source["affine"], float)
    model = np.asarray(aligned["padded_2mm_affine"], float)
    shape = np.asarray(source["native_shape_xyz"], int)
    box = np.asarray(aligned["valid_model_box_xyz"], float)
    require(box.shape == (2, 3) and np.isfinite(box).all() and
            np.all(box[1] > box[0]), "Invalid valid UAE box")
    model_shape = np.asarray(aligned["padded_shape_xyz"], int)
    require(np.all(box[0] >= 0) and np.all(box[1] <= model_shape), "UAE box exceeds input")
    transform = np.linalg.inv(raw).dot(model)
    # The per-scan lattice must preserve the source's directions, even if oblique.
    require(np.allclose(transform[:3, :3], np.diag(np.diag(transform[:3, :3])),
                        atol=1e-7, rtol=0) and np.all(np.diag(transform[:3, :3]) > 0),
            "UAE/native axes are not parallel")
    corners = np.asarray(list(itertools.product(*zip(box[0]-.5, box[1]-.5))))
    native_cells = pt.apply_affine(corners, transform)+.5
    low, high = native_cells.min(axis=0), native_cells.max(axis=0)
    # Stabilize only floating-point roundoff around exact integer boundaries.
    low = np.where(abs(low-np.rint(low)) < 1e-8, np.rint(low), low)
    high = np.where(abs(high-np.rint(high)) < 1e-8, np.rint(high), high)
    start = np.maximum(0, np.floor(low).astype(int))
    stop = np.minimum(shape, np.ceil(high).astype(int))
    require(np.all(stop > start), "Empty native ROI")
    require(np.all(start <= np.maximum(low, 0)+1e-7) and
            np.all(stop >= np.minimum(high, shape)-1e-7), "ROI fails outward containment")
    affine = raw.copy()
    affine[:3, 3] = pt.apply_affine(start, raw)[0]
    roi = dict(source, affine=affine.tolist(), native_shape_xyz=(stop-start).tolist())
    spacing = np.linalg.norm(raw[:3, :3], axis=0)
    return {"schema_version": 1, "subject_id": aligned["subject_id"],
            "session": aligned["session"], "group_name": aligned["group_name"],
            "source_ct": source, "source_uae_plan": aligned,
            "crop_start_xyz": start.tolist(), "crop_end_xyz": stop.tolist(),
            "crop_geometry": roi, "nominal_margin_mm": 100,
            "policy": "uae_valid_cell_extent_to_native_outward_clamp_v1",
            "requested_native_cell_bounds": [low.tolist(), high.tolist()],
            "outward_rounding_mm": [(np.maximum(low, 0)-start).clip(0)*spacing,
                                     (stop-np.minimum(high, shape)).clip(0)*spacing],
            "original_fov_clamped": [list(map(bool, low < 0)), list(map(bool, high > shape))],
            "geometry_checks": pt.geometry_checks(roi)}


def serializable_plan(aligned):
    plan = native_plan(aligned)
    plan["outward_rounding_mm"] = [x.tolist() for x in plan["outward_rounding_mm"]]
    return plan


def inside_crop(raw, plan):
    return pt.inside(np.asarray(raw, float)-plan["crop_start_xyz"],
                     plan["crop_geometry"]["native_shape_xyz"])


def crop_image(image, plan):
    """ITK ROI in native geometry, retaining no dependency on the full image."""
    import itk
    pt.check_itk_geometry(image, plan["source_ct"])
    region = itk.ImageRegion[3]()
    region.SetIndex([int(v) for v in plan["crop_start_xyz"]])
    region.SetSize([int(v) for v in plan["crop_geometry"]["native_shape_xyz"]])
    filt = itk.RegionOfInterestImageFilter.New(image, RegionOfInterest=region)
    filt.UpdateLargestPossibleRegion()
    result = filt.GetOutput()
    result.DisconnectPipeline()
    pt.check_itk_geometry(result, plan["crop_geometry"])
    return result


def load_crop(plan):
    import itk
    path = pt.verify_identity(plan["source_ct"])
    raw = itk.imread(str(path), itk.F)
    result = crop_image(raw, plan)
    del raw
    data = itk.array_view_from_image(result)
    require(all(np.isfinite(s).all() for s in data), "Non-finite native crop")
    return result


def evaluate_group(rows, test, retest, forward, backward):
    """Reuse continuous evaluator on ROI geometry, then restore full raw indices."""
    query = np.asarray([[float(r["raw_"+a]) for a in "xyz"] for r in rows])
    require(inside_crop(query, test).all(), "Query outside Test crop")
    local = [dict(r, **{"raw_"+a: float(p[i]-test["crop_start_xyz"][i])
                       for i, a in enumerate("xyz")}) for r, p in zip(rows, query)]
    evaluated = pt.evaluate_cycle(local, test["crop_geometry"], retest["crop_geometry"], forward, backward)
    output = []
    for original, result in zip(rows, evaluated):
        result.update(original)
        result["failure_reason"] = result["failure_reason"].replace("retest_fov", "retest_crop").replace("test_fov", "test_crop")
        for prefix, plan in (("query_raw", test), ("matched_raw", retest), ("returned_raw", test)):
            for i, a in enumerate("xyz"):
                key = prefix+"_"+a
                if result[key] != "":
                    result[key] = float(result[key])+plan["crop_start_xyz"][i]
                if prefix != "query_raw":
                    result[prefix+"_rounded_"+a] = int(np.rint(result[key])) if result[key] != "" else ""
        output.append(result)
    return output


def validate_source_checkpoint(path):
    require(pt.identity(path)["sha256"] == WHOLE_CHECKPOINT_SHA, "Wrong completed whole-body checkpoint")
    cp = pt.load_json(path)
    require(cp["status"] == "TECHNICAL_PASS", "Whole-body source incomplete")
    manifest = Path(path).parent/"registration_manifest.json"
    require(pt.identity(manifest)["sha256"] == WHOLE_MANIFEST_SHA, "Wrong whole-body manifest")
    source = base.load_run(Path(path).parent, active=False)
    require(source["limits"]["threads"] == 1, "Expected approved single-thread runtime")
    require(source["queries"]["sha256"] == base.QUERY_SHA, "Wrong frozen queries")
    return source


def repository():
    value = base.repository()
    require(subprocess.call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                             BASE_COMMIT, "HEAD"]) == 0, "Missing whole-body execution ancestry")
    return value


def pilot_rows(manifest, group=None):
    rows = [r for r in pt.read_csv(manifest["queries"]["path"]) if r["subject_id"] == PILOT]
    require(len(rows) == QUERY_COUNT and len({r["query_id"] for r in rows}) == QUERY_COUNT,
            "Pilot query denominator or identity changed")
    return rows if group is None else [r for r in rows if r["group_name"] == group]


def prepare(args):
    import nibabel as nib
    from tools.quadra.registration_runtime import preflight
    repo = repository()
    source = validate_source_checkpoint(args.whole_body_checkpoint)
    environment = preflight(args.storage_root)
    require(environment == source["environment"], "Environment differs from completed registration")
    require(pt.parameter_maps() == source["parameters"], "Registration parameters changed")
    uae = pt.load_json(pt.verify_identity(source["uae_manifest"]))
    records = uae["outputs"]["plans"]
    require(len(records) == 224, "Wrong UAE plan denominator")
    plans, references = {}, []
    for record in records:
        aligned = pt.load_json(pt.verify_identity(record))
        key = aligned["subject_id"]+"-"+aligned["session"]+"-"+aligned["group_name"]
        require(key not in plans, "Duplicate UAE plan")
        plans[key] = aligned
        if aligned["subject_id"] == PILOT:
            references.append(record)
    selected = {}
    for session in ("test", "retest"):
        for group in GROUPS:
            aligned = plans[PILOT+"-"+session+"-"+group]
            require(aligned["margin_mm"] == 100 and aligned["strategy"] == "organ_group_global_lattice",
                    "Wrong UAE crop policy")
            require(aligned["source_ct"] == source["sources"][PILOT+"-"+session], "CT identity mismatch")
            selected[session+"-"+group] = serializable_plan(aligned)
    rows = pilot_rows(source)
    expected_groups = Counter(r["group_name"] for r in rows)
    require(set(expected_groups) == set(GROUPS), "Missing query group")
    for group in GROUPS:
        group_rows = [r for r in rows if r["group_name"] == group]
        require({r["mask_name"] for r in group_rows} == set(selected["test-"+group]["source_uae_plan"]["included_masks"]),
                "Group mask membership changed")
        points = np.asarray([[float(r["raw_"+a]) for a in "xyz"] for r in group_rows])
        require(inside_crop(points, selected["test-"+group]).all(), "Queries outside native crop")
    masks = [x for x in source["masks"] if x["scan_key"].startswith(PILOT+"-")]
    for session in ("test", "retest"):
        original = source["sources"][PILOT+"-"+session]
        image = nib.load(str(pt.verify_identity(original)))
        require(list(image.shape) == original["native_shape_xyz"] and
                np.allclose(image.affine, original["affine"], atol=1e-5, rtol=0), "CT native geometry changed")
    for mask in masks:
        path = pt.verify_identity(mask)
        image = nib.load(str(path))
        original = source["sources"][mask["scan_key"]]
        require(list(image.shape) == original["native_shape_xyz"] and
                np.allclose(image.affine, original["affine"], atol=1e-5, rtol=0), "Mask geometry changed")
        if mask["scan_key"].endswith("-test"):
            items = [r for r in rows if r["mask_name"] == mask["mask_name"]]
            data = np.asanyarray(image.dataobj)
            points = np.asarray([[int(r["raw_"+a]) for a in "xyz"] for r in items])
            require(len(items) > 0 and np.all(data[tuple(points.T)] != 0), "Query outside original mask")
            require(np.count_nonzero(data) == int(items[0]["available_unique_voxels"]), "Mask quota changed")
            del data
    run = args.storage_root/"runs/cohort"/("registration-organ-group-native-"+time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    base.check_resources(source["limits"], args.storage_root)
    run.mkdir(parents=True, exist_ok=False)
    pt.atomic_bytes(run/"frozen_queries_raw_itk.csv", pt.verify_identity(source["queries"]).read_bytes(), refuse=True)
    output_plans = {}
    for key, plan in selected.items():
        path = run/"plans"/(key+".json")
        pt.atomic_json(path, plan, refuse=True)
        output_plans[key] = pt.identity(path)
    m = {"schema_version": 1, "workflow": WORKFLOW, "created_at": pt.utc_now(),
         "run_directory": str(run), "storage_root": str(args.storage_root), "repository": repo,
         "environment": environment, "whole_body_checkpoint": pt.identity(args.whole_body_checkpoint),
         "whole_body_manifest": pt.identity(args.whole_body_checkpoint.parent/"registration_manifest.json"),
         "uae_manifest": source["uae_manifest"], "uae_plan_references": references,
         "queries": pt.identity(run/"frozen_queries_raw_itk.csv"), "plans": output_plans, "masks": masks,
         "parameters": source["parameters"], "limits": source["limits"], "runtime_contract": source["runtime_contract"],
         "pilot_subject": PILOT, "group_order": list(GROUPS), "group_query_counts": dict(expected_groups),
         "expected_queries": QUERY_COUNT, "expected_registrations": 8,
         "settings": {"cohort_authorized": False, "metric_masks": False, "resampling": False,
                      "normalization": False, "padding": False, "physical_export": "RAS_mm",
                      "point_policy": "continuous_no_intermediate_rounding",
                      "domain_policy": "native_crop_voxel_centres_no_reverse_extrapolation",
                      "no_uae_comparison": True}}
    m["signature"] = pt.digest(m)
    pt.atomic_json(run/"organ_group_manifest.json", m, refuse=True)
    status_update(run, status="PREPARED", controller_pid=None, worker_pid=None, groups_completed=0,
                  registrations_completed=0, queries_valid=0, queries_invalid=0)
    print("Preparation PASS\nRun directory: {}\nPilot: {}; {} unchanged queries; eight registrations".format(run, PILOT, len(rows)), flush=True)


def load_run(run, active=True):
    from tools.quadra.registration_runtime import preflight
    m = pt.load_json(Path(run)/"organ_group_manifest.json")
    require(m["schema_version"] == 1 and m["workflow"] == WORKFLOW, "Wrong organ-group workflow")
    require(pt.digest({k:v for k,v in m.items() if k != "signature"}) == m["signature"], "Manifest signature changed")
    require(str(Path(run).resolve()) == m["run_directory"] and m["pilot_subject"] == PILOT and
            m["settings"]["cohort_authorized"] is False, "Invalid pilot scope")
    require(m["whole_body_checkpoint"]["sha256"] == WHOLE_CHECKPOINT_SHA and
            m["whole_body_manifest"]["sha256"] == WHOLE_MANIFEST_SHA and
            m["queries"]["sha256"] == base.QUERY_SHA, "Wrong immutable source lineage")
    require(m["limits"]["threads"] == 1 and m["expected_queries"] == QUERY_COUNT and
            m["expected_registrations"] == 8 and m["group_order"] == list(GROUPS),
            "Pilot runtime or denominator changed")
    base.validate_runtime_contract(m)
    require(set(m["plans"]) == {s+"-"+g for s in ("test", "retest") for g in GROUPS},
            "Missing or extra native plans")
    for ref in [m["whole_body_checkpoint"], m["whole_body_manifest"], m["uae_manifest"], m["queries"]]+m["uae_plan_references"]+list(m["plans"].values()):
        pt.verify_identity(ref)
    source = pt.load_json(m["whole_body_manifest"]["path"])
    require(m["parameters"] == source["parameters"] and m["limits"] == source["limits"] and
            m["environment"] == source["environment"], "Inherited registration contract changed")
    for ref in m["plans"].values():
        plan = pt.load_json(ref["path"])
        require(plan == serializable_plan(plan["source_uae_plan"]), "Native crop derivation changed")
    if active:
        require(repository() == m["repository"], "Execution repository changed")
        require(preflight(m["storage_root"]) == m["environment"], "Execution environment changed")
        require(pt.parameter_maps() == m["parameters"], "Parameter maps changed")
    return m


def status_update(run, **fields):
    path = Path(run)/"cohort_status.json"
    value = pt.load_json(path) if path.exists() else {"workflow": WORKFLOW, "current_subject": PILOT}
    value.update(fields, heartbeat_at=pt.utc_now())
    pt.atomic_json(path, value)


def task_signature(m, group, direction):
    return pt.digest({"manifest": m["signature"], "group": group, "direction": direction})


def validate_result(m, group, direction, path):
    if not Path(path).exists():
        return None
    meta = pt.load_json(path)
    require(meta.get("status") == "success" and meta.get("schema_version") == 1 and
            meta["signature"] == task_signature(m, group, direction), "Incompatible task result")
    for f in meta["files"]:
        pt.verify_identity(f)
    require(meta["peak_rss_bytes"] <= m["limits"]["ram_ceiling_bytes"], "Saved RAM violation")
    if direction == "points":
        rows = pt.read_csv(pt.verify_identity(meta["point_csv"]))
        require([r["query_id"] for r in rows] == [r["query_id"] for r in pilot_rows(m, group)], "Point IDs/order changed")
        require(len(rows) == meta["queries"] == meta["valid_queries"]+meta["invalid_queries"], "Point counts changed")
        require(sum(r["valid_cycle"] == "True" for r in rows) == meta["valid_queries"], "Validity count mismatch")
        for r in rows:
            require(r["valid_cycle"] in ("True", "False"), "Invalid validity flag")
            require((r["cycle_error_mm"] == "" and bool(r["failure_reason"])) if r["valid_cycle"] == "False"
                    else np.isfinite(float(r["cycle_error_mm"])) and float(r["cycle_error_mm"]) >= 0 and not r["failure_reason"],
                    "Invalid cycle output")
    else:
        maps = pt.load_json(pt.verify_identity(meta["maps_json"]))
        require([x["Transform"][0] for x in maps] == ["EulerTransform", "BSplineTransform"], "Incomplete transform chain")
        require(all(x["HowToCombineTransforms"] == ["Compose"] for x in maps), "Wrong composition")
    return meta


def worker(args):
    import itk
    import resource
    m = load_run(args.run_directory)
    work = args.attempt_directory.resolve()
    require(work.is_relative_to(args.run_directory.resolve()/"groups"/args.group), "Worker output escapes group")
    group, direction = args.group, args.direction
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)
    itk.MultiThreaderBase.SetGlobalMaximumNumberOfThreads(1)
    plans = [pt.load_json(pt.verify_identity(m["plans"][s+"-"+group])) for s in ("test", "retest")]
    for mask in m["masks"]:
        if mask["mask_name"] in plans[0]["source_uae_plan"]["included_masks"]:
            pt.verify_identity(mask)
    start = time.monotonic()
    meta = {"schema_version": 1, "status": "success", "signature": task_signature(m, group, direction),
            "group_name": group, "subject_id": PILOT, "direction": direction, "created_at": pt.utc_now()}
    files = []
    if direction != "points":
        fixed_plan, moving_plan = plans if direction == "forward" else plans[::-1]
        t = time.monotonic()
        fixed = load_crop(fixed_plan)
        moving = load_crop(moving_plan)
        meta["load_seconds"] = time.monotonic()-t
        meta["fixed_geometry"] = fixed_plan["crop_geometry"]
        meta["moving_geometry"] = moving_plan["crop_geometry"]
        filt = itk.ElastixRegistrationMethod.New(fixed, moving)
        filt.SetParameterObject(pt.parameter_object(m["parameters"]))
        filt.SetNumberOfThreads(1)
        filt.SetOutputDirectory(str(work))
        filt.SetLogToFile(True)
        filt.SetLogToConsole(False)
        t = time.monotonic()
        filt.UpdateLargestPossibleRegion()
        meta["registration_seconds"] = time.monotonic()-t
        maps = pt.normalized_transform_maps(filt.GetTransformParameterObject())
        files += pt.save_transform_chain(work/"transforms", maps)
        pt.atomic_json(work/"transform_maps.json", maps, refuse=True)
        meta["maps_json"] = pt.identity(work/"transform_maps.json")
        files.append(meta["maps_json"])
        shape = np.asarray(fixed_plan["crop_geometry"]["native_shape_xyz"])
        sample = pt.apply_affine(np.array([(shape-1)*f for f in (.25, .5, .75)]), pt.lps_affine(fixed_plan["crop_geometry"]))
        a = pt.transformix_points(sample, maps, 1)
        b = pt.transformix_points(sample, pt.load_json(work/"transform_maps.json"), 1)
        require(np.isfinite(a).all() and np.max(abs(a-b)) <= 1e-4, "Transform reload check failed")
        meta["transform_reload_max_mm"] = float(np.max(abs(a-b)))
        from tools.quadra.registration_report import registration_qc
        registration_qc(fixed, moving, fixed_plan["crop_geometry"], moving_plan["crop_geometry"],
                        maps, 1, work/"registration_qc.png")
        files.append(pt.identity(work/"registration_qc.png"))
        del fixed, moving, filt
    else:
        fm = validate_result(m, group, "forward", args.run_directory/"groups"/group/"forward.json")
        bm = validate_result(m, group, "backward", args.run_directory/"groups"/group/"backward.json")
        require(fm is not None and bm is not None, "Missing directional registration")
        maps_f, maps_b = pt.load_json(fm["maps_json"]["path"]), pt.load_json(bm["maps_json"]["path"])
        rows = evaluate_group(pilot_rows(m, group), *plans,
                              lambda p: pt.transformix_points(p, maps_f, 1),
                              lambda p: pt.transformix_points(p, maps_b, 1))
        pt.atomic_csv(work/"points.csv", rows, refuse=True)
        meta.update(point_csv=pt.identity(work/"points.csv"), queries=len(rows),
                    valid_queries=sum(r["valid_cycle"] for r in rows),
                    invalid_queries=sum(not r["valid_cycle"] for r in rows))
        files.append(meta["point_csv"])
    if (work/"elastix.log").exists():
        files.append(pt.identity(work/"elastix.log"))
    meta.update(files=files, wall_time_seconds=time.monotonic()-start,
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024), completed_at=pt.utc_now())
    require(meta["peak_rss_bytes"] <= m["limits"]["ram_ceiling_bytes"], "Worker exceeded RAM ceiling")
    require(not base.forbidden_outputs(work), "Forbidden full-volume artifact")
    pt.atomic_json(work/"worker_result.json", meta, refuse=True)


def assert_no_worker(run):
    import psutil
    for proc in psutil.process_iter(["cmdline"]):
        cmd = proc.info["cmdline"] or []
        if MODULE in cmd and "_worker" in cmd and str(run) in cmd:
            raise pt.RegistrationError("Existing worker {} owns this run".format(proc.pid))


def classify_failure(attempt, returncode, timeout):
    if timeout:
        return {"classification": "timeout", "message": "Six-hour worker limit exceeded"}
    error = Path(attempt)/"worker_error.json"
    if error.exists():
        return pt.load_json(error)
    return {"classification": "process_signal" if returncode < 0 else "process_exit", "exit_code": returncode}


def run_task(run, m, group, direction):
    import psutil
    destination = run/"groups"/group
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination/(direction+".json")
    previous = validate_result(m, group, direction, marker)
    if previous:
        return previous
    for result in sorted(destination.glob(direction+"-attempt-*/worker_result.json")):
        recovered = validate_result(m, group, direction, result)
        pt.atomic_json(marker, recovered, refuse=True)
        return recovered
    for retry in range(2):
        assert_no_worker(run)
        attempt = destination/(direction+"-attempt-"+str(time.time_ns()))
        attempt.mkdir()
        env = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-u", "-m", MODULE, "_worker", "--run-directory", str(run),
                   "--group", group, "--direction", direction, "--attempt-directory", str(attempt)]
        start, peak, timed_out = time.monotonic(), 0, False
        with (attempt/"stdout.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            observed = psutil.Process(proc.pid)
            observed.cpu_percent(None)
            status_update(run, worker_pid=proc.pid, worker_create_time=observed.create_time(),
                          current_group=group, current_direction=direction, attempt_directory=str(attempt), retry=retry)
            try:
                while proc.poll() is None:
                    if time.monotonic()-start > m["limits"]["direction_timeout_seconds"]:
                        timed_out = True
                        break
                    try:
                        rss = sum(p.memory_info().rss for p in [observed]+observed.children(recursive=True) if p.is_running())
                        cpu = observed.cpu_percent(None)
                    except psutil.NoSuchProcess:
                        rss, cpu = 0, 0
                    peak = max(peak, rss)
                    resources = base.check_resources(m["limits"], run, rss)
                    resources["worker_cpu_percent"] = cpu
                    status_update(run, resources=resources, worker_peak_rss_bytes=peak, progress=base.log_progress(attempt))
                    time.sleep(5)
            finally:
                base.stop_owned_process(proc)
                status_update(run, worker_pid=None)
        if proc.returncode == 0:
            meta = validate_result(m, group, direction, attempt/"worker_result.json")
            require(meta is not None, "Worker exited without a committed result")
            meta.update(peak_rss_bytes=max(peak, meta["peak_rss_bytes"]), retry_index=retry,
                        controller_wall_time_seconds=time.monotonic()-start)
            require(meta["peak_rss_bytes"] <= m["limits"]["ram_ceiling_bytes"], "Sampled RAM exceeded ceiling")
            pt.atomic_json(marker, meta, refuse=True)
            return meta
        failure = classify_failure(attempt, proc.returncode, timed_out)
        pt.atomic_json(attempt/"controller_failure.json", failure, refuse=True)
        if failure["classification"] == "contract_error":
            raise pt.RegistrationError(failure["message"])
        if retry == 1:
            raise pt.RuntimeFailure("{} {} failed twice: {}".format(group, direction, attempt))


def counters(run, m):
    count = dict(groups_completed=0, registrations_completed=0, queries_valid=0, queries_invalid=0)
    for group in GROUPS:
        for direction in ("forward", "backward", "points"):
            meta = validate_result(m, group, direction, run/"groups"/group/(direction+".json"))
            if meta:
                if direction == "points":
                    count["groups_completed"] += 1
                    count["queries_valid"] += meta["valid_queries"]
                    count["queries_invalid"] += meta["invalid_queries"]
                else:
                    count["registrations_completed"] += 1
    return count


def execute(args):
    import psutil
    run = args.run_directory.resolve()
    with base.controller_lock(run):
        m = load_run(run)
        assert_no_worker(run)
        cp = run/"pilot_checkpoint.json"
        if cp.exists():
            saved = pt.load_json(cp)
            require(saved["manifest_signature"] == m["signature"], "Pilot checkpoint changed")
            for ref in saved["evidence"]:
                pt.verify_identity(ref)
            print("REVIEW_REQUIRED: existing pilot verified; no cohort launched", flush=True)
            return
        status_update(run, status="RUNNING", controller_pid=os.getpid(),
                      controller_create_time=psutil.Process().create_time(), error=None, **counters(run, m))
        outcome = "INCOMPLETE"
        try:
            for group in GROUPS:
                for direction in ("forward", "backward", "points"):
                    print("{} {} {}".format(PILOT, group, direction), flush=True)
                    run_task(run, m, group, direction)
                    status_update(run, **counters(run, m))
            require(not base.forbidden_outputs(run), "Forbidden full-volume output")
            require(repository() == m["repository"], "Repository changed during pilot")
            count = counters(run, m)
            require(count["registrations_completed"] == 8 and count["groups_completed"] == 4 and
                    count["queries_valid"]+count["queries_invalid"] == QUERY_COUNT, "Incomplete pilot denominator")
            from tools.quadra.registration_organ_group_report import build_report
            report = build_report(run, m)
            evidence = [pt.identity(p) for p in sorted(run.rglob("*")) if p.is_file() and
                        p.name not in ("controller.lock", "cohort_status.json", "controller.log")]
            outcome = "REVIEW_REQUIRED"
            pt.atomic_json(cp, {"status": outcome, "technical_gates_passed": count["queries_invalid"] == 0,
                               "cohort_authorized": False, "subject_id": PILOT, "counts": count,
                               "manifest_signature": m["signature"], "report": pt.identity(report),
                               "evidence": evidence, "created_at": pt.utc_now()}, refuse=True)
        except pt.RegistrationError as exc:
            outcome = "BLOCKED"
            status_update(run, error=str(exc))
            raise
        except BaseException as exc:
            status_update(run, error=str(exc))
            raise
        finally:
            try:
                count = counters(run, m)
            except pt.RegistrationError as exc:
                outcome, count = "BLOCKED", {}
                status_update(run, error=str(exc))
            status_update(run, status=outcome, controller_pid=None, worker_pid=None, **count)
        print("{}\nRun directory: {}\nReport: {}\nNo cohort has been launched.".format(outcome, run, report), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--storage-root", type=Path, default=Path("/workspace/quadra"))
    prep.add_argument("--whole-body-checkpoint", type=Path, default=Path(WHOLE_RUN)/"checkpoint_summary.json")
    for command in ("pilot", "status", "_worker"):
        item = sub.add_parser(command)
        item.add_argument("--run-directory", required=True, type=Path)
        if command == "status":
            item.add_argument("--json", action="store_true")
        if command == "_worker":
            item.add_argument("--group", choices=GROUPS, required=True)
            item.add_argument("--direction", choices=("forward", "backward", "points"), required=True)
            item.add_argument("--attempt-directory", type=Path, required=True)
    return p


def main(argv=None):
    import json
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "pilot":
            execute(args)
        elif args.command == "_worker":
            worker(args)
        else:
            print(json.dumps(pt.load_json(args.run_directory/"cohort_status.json"), indent=2))
        return 0
    except Exception as exc:
        traceback.print_exc()
        if args.command == "_worker":
            pt.atomic_json(args.attempt_directory/"worker_error.json",
                           {"classification": "contract_error" if isinstance(exc, pt.RegistrationError) else "runtime_error",
                            "message": str(exc)}, refuse=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
