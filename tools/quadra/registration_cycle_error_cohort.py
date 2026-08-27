#!/usr/bin/env python
"""Native whole-body rigid/B-spline cohort with continuous point evaluation.

Real-image commands require the isolated RunPod registration profile. The pilot
always stops for explicit human review. No legacy registration imports or DVFs.
"""
import argparse
from collections import Counter, defaultdict
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quadra.registration_point_transform import (RegistrationError, RuntimeFailure,
    require, utc_now, digest, identity, verify_identity, load_json, read_csv,
    atomic_json, atomic_csv, atomic_text, atomic_bytes, apply_affine, lps_affine, geometry_checks,
    check_itk_geometry, parameter_maps, parameter_object, normalized_transform_maps,
    save_transform_chain, transformix_points, evaluate_cycle)

COHORT_COMMIT = "3079f6d1dcfe90001a45b48b685b803fcee6b327"
BACKUP_COMMIT = "e55e2b7cfc9ca29c9d91d34244dbc83a41046cf6"
QUERY_SHA = "92ad6f6ed4763d006b7bdd0a2e157b6ac5eb92a09e7fd903d88bc2387f3428d6"
CHECKPOINT_SHA = "1132d16e38846a4d27bc52b9ff9317abbb1665ce398696faf284412028cbb9a0"
UAE_MANIFEST_SHA = "7f2489891be785bd7f761b5fb1ae444a152d174dd8126f51fa4cbc5787d3a078"
SOURCE_RUN = "/workspace/quadra/runs/cohort/uaes-aligned-100mm-20260820T044539Z"
SUBJECTS = ["quadra_hc_{:03d}".format(i) for i in range(21, 49)]
PILOT = "quadra_hc_044"
WORKFLOW = "quadra-native-continuous-registration-v1"
TERMINAL = ("TECHNICAL_PASS", "PARTIAL", "BLOCKED", "INCOMPLETE")
_WORKSPACE_USAGE = {}
MEMORY_ACCOUNTING_POLICY = "clean-inactive-file-headroom-v1"


def workspace_usage_bytes(workspace, max_age=60):
    """Count quota usage, since network-volume df reports the shared filesystem."""
    key, now = str(Path(workspace).resolve()), time.monotonic()
    previous = _WORKSPACE_USAGE.get(key)
    if previous and now-previous[0] < max_age:
        return previous[1]
    value = int(subprocess.check_output(["du", "-sb", key], text=True, timeout=60).split()[0])
    _WORKSPACE_USAGE[key] = (now, value)
    return value


def repository():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT)] + list(args), text=True).strip()
    for commit in (COHORT_COMMIT, BACKUP_COMMIT):
        require(subprocess.call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"]) == 0,
                "Missing accepted repository ancestry")
    require(not git("status", "--porcelain"), "Repository is dirty")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "clean": True}


def cgroup_directories(base=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Find host-relative and container-rooted cgroup v1/v2 limit locations."""
    base = Path(base)
    paths = {base}
    for line in Path(membership).read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        # Docker may mount the current group directly at the controller root,
        # even while /proc/self/cgroup still reports /docker/<container-id>.
        roots = [base] + [base/name for name in controllers.split(",") if name]
        if controllers:
            roots.append(base/controllers)  # e.g. the combined cpu,cpuacct mount
        for root in roots:
            paths.add(root)
            current = root/relative.lstrip("/")
            while current != root and root in current.parents:
                paths.add(current)
                current = current.parent
    return sorted(paths)


def cgroup_memory_sample(root, version, limit, usage_path):
    """Conservative cache-adjusted headroom; never change the kernel limit.

    Docker discounts total_inactive_file (v1) / inactive_file (v2):
    https://docs.docker.com/reference/cli/docker/container/stats/
    Here we additionally cap by file cache and exclude ALL dirty/writeback
    bytes (possibly double-counting exclusions, deliberately conservative).
    This is an estimate, not a promise that reclaim will succeed. Missing or
    malformed stats get zero discount; missing usage cannot pass a guard.
    """
    root, usage_path = Path(root), Path(usage_path)
    require(usage_path.is_file(), "Missing cgroup memory usage: " + str(usage_path))
    raw = usage_path.read_text().strip()
    require(raw.isdigit(), "Invalid cgroup memory usage: " + str(usage_path))
    usage = int(raw)
    names = (["total_inactive_file", "total_cache", "total_dirty", "total_writeback"]
             if version == 1 else ["inactive_file", "file", "file_dirty", "file_writeback"])
    counters, discount, reason = {}, 0, "missing_or_invalid_stats_no_discount"
    try:
        stats = {}
        for line in (root/"memory.stat").read_text().splitlines():
            key, value = line.split()
            if key in names:
                if key in stats or not value.isdigit():
                    raise ValueError("Invalid or duplicate memory counter")
                stats[key] = int(value)
        inactive, cache, dirty, writeback = [stats[key] for key in names]
        discount = max(0, min(inactive, cache, usage)-dirty-writeback)
        counters = dict(inactive_file_bytes=inactive, file_cache_bytes=cache,
                        dirty_bytes=dirty, writeback_bytes=writeback)
        reason = "clean_inactive_file_estimate"
    except (OSError, ValueError, KeyError):
        pass  # Fail conservatively: account all charged memory as working memory.
    working = usage-discount
    return dict(path=str(root), cgroup_version=version, limit_bytes=limit,
                usage_bytes=usage, reclaimable_file_estimate_bytes=discount,
                working_set_estimate_bytes=working, raw_headroom_bytes=max(0, limit-usage),
                estimated_headroom_bytes=max(0, limit-working),
                cache_accounting=reason, **counters)


def resource_limits():
    import psutil
    host = psutil.virtual_memory()
    total, available, raw_available = host.total, host.available, host.available
    memory_samples = []
    cpus = float(len(os.sched_getaffinity(0)))
    for root in cgroup_directories():
        for version, name, usage_name in ((2, "memory.max", "memory.current"),
                                         (1, "memory.limit_in_bytes", "memory.usage_in_bytes")):
            p = root/name
            if p.is_file() and p.read_text().strip().isdigit():
                limit = int(p.read_text().strip())
                total = min(total, limit)
                sample = cgroup_memory_sample(root, version, limit, root/usage_name)
                memory_samples.append(sample)
                available = min(available, sample["estimated_headroom_bytes"])
                raw_available = min(raw_available, sample["raw_headroom_bytes"])
        p = root/"cpu.max"
        if p.is_file():
            quota, period = p.read_text().split()
            if quota != "max":
                cpus = min(cpus, int(quota)/int(period))
        q, p = root/"cpu.cfs_quota_us", root/"cpu.cfs_period_us"
        if q.is_file() and p.is_file() and int(q.read_text()) > 0:
            cpus = min(cpus, int(q.read_text())/int(p.read_text()))
    return {"effective_ram_bytes": total, "ram_ceiling_bytes": int(.8*total),
            "available_ram_bytes": available, "effective_cpu_count": cpus,
            "memory_accounting_policy": MEMORY_ACCOUNTING_POLICY,
            "host_available_ram_bytes": host.available,
            "raw_available_ram_bytes": raw_available, "memory_accounting": memory_samples,
            "threads": min(8, max(1, math.floor(cpus))), "min_disk_free_bytes": 10*1024**3,
            "direction_timeout_seconds": 6*3600}


def check_resources(limits, run_dir, rss=0):
    import psutil
    import shutil
    current = resource_limits()
    if "memory_accounting_policy" in limits:
        require(current.get("memory_accounting_policy") == limits["memory_accounting_policy"],
                "Memory accounting policy changed")
    require(current["effective_ram_bytes"] >= limits["effective_ram_bytes"], "Allocated RAM decreased")
    require(current["threads"] >= limits["threads"], "Allocated CPUs decreased")
    require(current.get("effective_cpu_count", current["threads"]) >=
            limits.get("effective_cpu_count", limits["threads"]), "Allocated CPU quota decreased")
    disk = shutil.disk_usage(run_dir).free
    if "workspace_capacity_bytes" in limits:
        used = workspace_usage_bytes(limits["workspace_path"])
        disk = min(disk, max(0, limits["workspace_capacity_bytes"]-used))
    require(disk >= limits["min_disk_free_bytes"], "Disk guard failed")
    require(rss <= limits["ram_ceiling_bytes"], "RAM guard failed")
    available = current.get("available_ram_bytes", psutil.virtual_memory().available)
    require(available >= .2*limits["effective_ram_bytes"], "Available RAM headroom failed")
    return {"rss_bytes": rss, "disk_free_bytes": disk, "available_ram_bytes": available,
            "memory_accounting_policy": current.get("memory_accounting_policy"),
            "host_available_ram_bytes": current.get("host_available_ram_bytes"),
            "raw_available_ram_bytes": current.get("raw_available_ram_bytes"),
            "memory_accounting": current.get("memory_accounting", [])}


def runtime_contract(limits, requested_threads=None, rationale=""):
    """Freeze an explicit single-thread revision without changing scientific maps."""
    selected = dict(limits)
    rationale = rationale.strip()
    if requested_threads is None:
        require(not rationale, "Thread rationale requires --threads 1")
        policy = "allocated_cpu_threads_v1"
    else:
        require(type(requested_threads) is int and requested_threads == 1,
                "Only the explicitly approved single-thread override is supported")
        require(len(rationale) >= 20, "Explicit thread revision rationale required")
        require(limits["threads"] >= 1, "No registration thread capacity")
        selected["threads"] = 1
        policy = "explicit_single_thread_v1"
    contract = {"schema_version":1, "thread_policy":policy,
                "requested_threads":requested_threads, "selected_threads":selected["threads"],
                "allocated_thread_capacity":limits["threads"], "rationale":rationale}
    return selected, contract


def validate_runtime_contract(manifest):
    # Historical manifests predate the explicit thread-policy field; keep them readable.
    contract = manifest.get("runtime_contract")
    if contract is None:
        return
    require(contract.get("schema_version") == 1, "Wrong runtime contract schema")
    detected = dict(manifest["limits"], threads=contract["allocated_thread_capacity"])
    selected, expected = runtime_contract(detected, contract["requested_threads"], contract["rationale"])
    require(contract == expected and selected["threads"] == manifest["limits"]["threads"],
            "Runtime thread contract changed")


def require_native_pair_capacity(sources, limits):
    """Reject obviously impossible allocations using metadata only, not CT IO.

    Two float32 inputs are only a lower bound: ITK and Elastix also need image
    pyramids and working memory. Passing this check does not prove feasibility.
    """
    pairs = defaultdict(int)
    for key, source in sources.items():
        pairs[key.rsplit("-", 1)[0]] += math.prod(source["native_shape_xyz"])*4
    minimum = max(pairs.values())
    require(minimum <= limits["ram_ceiling_bytes"],
            "Allocated RAM cannot hold even the native float32 CT pair within the 80% ceiling: "
            "at least {} bytes for inputs alone, ceiling {} bytes".format(minimum, limits["ram_ceiling_bytes"]))


@contextlib.contextmanager
def controller_lock(run_dir):
    with (Path(run_dir)/"controller.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RegistrationError("A controller already owns this run")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_queries(rows, denominators, registry):
    require(len(rows) == 108431 and len({r["query_id"] for r in rows}) == len(rows), "Query denominator/IDs changed")
    require(dict(Counter(r["subject_id"] for r in rows)) == denominators["subject_query_counts"], "Subject query counts changed")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["subject_id"], r["mask_name"])].append(r)
    for (subject, mask), items in groups.items():
        validate_mask_queries(subject,mask,items,registry)
    return groups


def validate_mask_queries(subject,mask,items,registry):
    i = int(items[0]["mask_registry_index"])
    require(0 <= i < len(registry) and registry[i]["filename"] == mask, "Mask registry index changed")
    require(len({tuple(int(r["raw_"+a]) for a in "xyz") for r in items}) == len(items), "Duplicate mask query")
    available = int(items[0]["available_unique_voxels"])
    require(len(items) == min(100, available) and available > 0, "Mask quota changed")
    for index, row in enumerate(items):
        require(int(row["point_index"]) == index and row["query_id"] == "{}:{}:{:03d}".format(subject,mask,index), "Query ordering changed")
        require(int(row["mask_registry_index"]) == i and int(row["sampling_seed"]) == 20260721+i, "Sampling seed changed")
        require(int(row["sampled_points_for_mask"]) == len(items) and int(row["mask_query_shortfall"]) == 100-len(items), "Sampling metadata changed")
        require(int(row["available_unique_voxels"]) == available, "Inconsistent mask denominator")
        require(row["sampling_policy"] == ("all_available" if available < 100 else "random_without_replacement"), "Sampling policy changed")


def prepare(args):
    import nibabel as nib
    import numpy as np
    import yaml
    from tools.quadra.registration_runtime import preflight
    root, source_dir = args.storage_root.resolve(), args.uae_run_directory.resolve()
    repo, fp = repository(), preflight(root)
    cp = source_dir/"checkpoint_summary.json"
    require(identity(cp)["sha256"] == CHECKPOINT_SHA, "Wrong UAE cohort checkpoint")
    checkpoint = load_json(cp)
    require(checkpoint["status"] == "TECHNICAL_PASS", "UAE cohort is not complete")
    um_path = source_dir/"cohort_manifest.json"
    require(identity(um_path)["sha256"] == UAE_MANIFEST_SHA, "Wrong UAE cohort manifest")
    um = load_json(um_path)
    qp = verify_identity(checkpoint["outputs"]["queries"])
    require(identity(qp)["sha256"] == QUERY_SHA, "Frozen query file changed")
    rows = read_csv(qp)
    registry = yaml.safe_load(verify_identity(um["registry"]).read_text())
    registry = registry["organs"] + registry["derived_organs"]
    require(len(registry) == 40, "Wrong mask registry")
    grouped = validate_queries(rows, um["denominators"], registry)
    cohort = load_json(verify_identity(um["cohort_manifest"]))
    scans = cohort["scans"]
    require(len(scans) == 56 and sorted({s["subject_id"] for s in scans}) == SUBJECTS, "Wrong scan/subject set")
    require(Counter(s["sex"] for s in scans) == {"M":24,"F":32}, "Wrong cohort sex counts")
    sources = {}
    for record in um["outputs"]["plans"]:
        plan = load_json(verify_identity(record))
        key = plan["subject_id"]+"-"+plan["session"]
        source = plan["source_ct"]
        require(key not in sources or sources[key] == source, "Conflicting source geometry across groups")
        sources[key] = source
    require(len(sources) == 56, "Missing original CT identities")
    limits, execution_runtime = runtime_contract(resource_limits(), args.threads, args.thread_rationale)
    if "workspace_capacity_bytes" in fp:
        limits.update(workspace_capacity_bytes=fp["workspace_capacity_bytes"], workspace_path="/workspace")
    require_native_pair_capacity(sources, limits)
    run_dir = root/"runs/cohort"/("registration-rigid-bspline-continuous-"+time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir.mkdir(parents=True, exist_ok=False)
    masks, checks = [], []
    # Preparation is sequential and may decompress one mask at a time, never CTs.
    for num, scan in enumerate(scans, 1):
        key = scan["subject_id"]+"-"+scan["session"]
        source = sources[key]
        print("[{}/56] Validate {}".format(num,key), flush=True)
        check_resources(limits, run_dir)
        path = verify_identity(source)
        require(path.is_relative_to(root/"datasets/source/whole_body_ct_v1"), "CT outside canonical root")
        ct = nib.load(str(path))
        require(list(ct.shape) == source["native_shape_xyz"] and np.allclose(ct.affine, source["affine"], atol=1e-5,rtol=0), "CT geometry changed")
        require(source["sha256"] == scan["input_sha256"], "Cohort CT identity changed")
        checks.append(dict(scan_key=key, **geometry_checks(source)))
        expected = [r["filename"] for r in registry if not (r["filename"] == "prostate" and scan["sex"] == "F")]
        require(set(expected) == set(scan["expected_masks"]), "Sex-specific mask set changed")
        for mask in expected:
            mp = root/"datasets/derivatives/totalsegmentator_2.16.0_organs_v1"/scan["subject_id"]/scan["session"]/"masks"/(mask+".nii.gz")
            mi = identity(mp)
            image = nib.load(str(mp))
            require(image.shape == ct.shape and np.allclose(image.affine,ct.affine,atol=1e-5,rtol=0), "Mask geometry changed: "+str(mp))
            if scan["session"] == "test":
                query_rows = grouped[(scan["subject_id"],mask)]
                data = np.asanyarray(image.dataobj)
                points = np.asarray([[int(r["raw_"+a]) for a in "xyz"] for r in query_rows])
                require((points >= 0).all() and (points < np.asarray(data.shape)).all(), "Query outside mask array")
                require(np.all(data[tuple(points.T)] != 0), "Frozen query outside its mask")
                require(np.count_nonzero(data) == int(query_rows[0]["available_unique_voxels"]), "Mask voxel count changed")
                require(all(r["sex"] == scan["sex"] for r in query_rows), "Query sex changed")
                del data
            masks.append(dict(mi, scan_key=key, mask_name=mask))
    require(len(masks) == 2208, "Wrong mask denominator")
    maps = parameter_maps()
    atomic_bytes(run_dir/"frozen_queries_raw_itk.csv", qp.read_bytes(), refuse=True)
    # Text copy must preserve exact bytes, not merely parsed values.
    require(identity(run_dir/"frozen_queries_raw_itk.csv")["sha256"] == QUERY_SHA, "Query copy changed bytes")
    manifest = {"schema_version":1,"workflow":WORKFLOW,"created_at":utc_now(),
        "run_directory":str(run_dir),"storage_root":str(root),"repository":repo,"environment":fp,
        "uae_checkpoint":identity(cp),"uae_manifest":identity(um_path),
        "queries":identity(run_dir/"frozen_queries_raw_itk.csv"),"sources":sources,"masks":masks,
        "geometry_checks":checks,"parameters":maps,"limits":limits,"runtime_contract":execution_runtime,
        "denominators":um["denominators"],
        "settings":{"registration_extent":"native_whole_body","registration_seed":121212,
                    "physical_export":"RAS_mm","transform_physical":"LPS_mm","coordinate_space":"raw_itk_voxel",
                    "point_policy":"continuous_no_intermediate_rounding","in_fov":"voxel_centres_0_to_size_minus_1",
                    "pilot":PILOT,"retry_limit":1,"no_uae_comparison":True}}
    manifest["signature"] = digest(manifest)
    atomic_json(run_dir/"registration_manifest.json",manifest,refuse=True)
    status_update(run_dir,status="PREPARED",subjects_completed=0,queries_valid=0,queries_invalid=0,
                  failed_subjects=[],registrations_completed=0,controller_pid=None,worker_pid=None)
    print("Preparation PASS\nRun directory: {}\nFrozen queries: {}".format(run_dir,len(rows)),flush=True)


def load_run(run_dir, active=True):
    run_dir = Path(run_dir).resolve()
    m = load_json(run_dir/"registration_manifest.json")
    require(m["workflow"] == WORKFLOW and m["schema_version"] == 1, "Wrong run contract")
    require(digest({k:v for k,v in m.items() if k != "signature"}) == m["signature"], "Manifest signature mismatch")
    require(str(run_dir) == m["run_directory"], "Run directory moved")
    validate_runtime_contract(m)
    verify_identity(m["queries"])
    verify_identity(m["uae_checkpoint"])
    verify_identity(m["uae_manifest"])
    if active:
        from tools.quadra.registration_runtime import preflight
        require(repository() == m["repository"], "Execution repository changed")
        require(preflight(m["storage_root"]) == m["environment"], "Execution environment changed")
        require(parameter_maps() == m["parameters"], "Resolved registration parameters changed")
    return m


def status_update(run_dir, **fields):
    path = Path(run_dir)/"cohort_status.json"
    value = load_json(path) if path.exists() else {"workflow":WORKFLOW}
    value.update(fields,heartbeat_at=utc_now())
    atomic_json(path,value)


def rows_for(m,subject):
    return [r for r in read_csv(m["queries"]["path"]) if r["subject_id"] == subject]


def task_signature(m,subject,direction):
    return digest({"contract":m["signature"],"subject":subject,"direction":direction})


def validate_result(m,subject,direction,marker):
    marker = Path(marker)
    if not marker.exists():
        return None
    meta = load_json(marker)
    require(meta.get("schema_version") == 1, "Unknown result schema")
    require(meta["signature"] == task_signature(m,subject,direction) and meta["status"] == "success", "Incompatible resume result")
    for item in meta["files"]:
        verify_identity(item)
    require(meta["peak_rss_bytes"] <= m["limits"]["ram_ceiling_bytes"], "Saved RAM limit violation")
    if direction == "points":
        rows = read_csv(meta["point_csv"]["path"])
        expected = rows_for(m,subject)
        require([r["query_id"] for r in rows] == [r["query_id"] for r in expected], "Point output IDs/order changed")
        require(sum(r["valid_cycle"] == "True" for r in rows) == meta["valid_queries"], "Point valid count changed")
        require(meta["queries"] == len(rows) and meta["invalid_queries"]+meta["valid_queries"] == len(rows), "Point counts changed")
        for row in rows:
            require(row["valid_cycle"] in ("True", "False"), "Invalid point status")
            if row["valid_cycle"] == "True":
                require(math.isfinite(float(row["cycle_error_mm"])) and float(row["cycle_error_mm"]) >= 0
                        and not row["failure_reason"], "Invalid valid-cycle row")
            else:
                require(row["cycle_error_mm"] == "" and bool(row["failure_reason"]), "Invalid-cycle value is not empty")
    else:
        maps = load_json(meta["maps_json"]["path"])
        require(len(maps) == 2 and [x["Transform"][0] for x in maps] == ["EulerTransform","BSplineTransform"], "Broken transform chain")
    return meta


def forbidden_outputs(run_dir):
    endings = (".nii", ".nii.gz", ".npy", ".npz", ".mha", ".mhd", ".raw", ".pth", ".pt")
    return [str(p) for p in Path(run_dir).rglob("*") if p.is_file() and p.name.lower().endswith(endings)]


def worker(args):
    import itk
    import numpy as np
    import psutil
    m = load_run(args.run_directory)
    work = args.attempt_directory.resolve()
    require(work.is_relative_to(args.run_directory.resolve()/"subjects"), "Worker output escapes run")
    subject,direction = args.subject,args.direction
    require(subject in SUBJECTS and direction in ("forward","backward","points"), "Invalid worker task")
    threads = m["limits"]["threads"]
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(threads)
    itk.MultiThreaderBase.SetGlobalMaximumNumberOfThreads(threads)
    sources = [m["sources"][subject+"-"+s] for s in ("test","retest")]
    for source in sources:
        verify_identity(source)
    for mask in m["masks"]:
        if mask["scan_key"].startswith(subject+"-"):
            verify_identity(mask)
    started = time.monotonic()
    files = []
    meta = {"schema_version":1,"status":"success","signature":task_signature(m,subject,direction),
            "subject_id":subject,"direction":direction,"created_at":utc_now()}
    if direction in ("forward","backward"):
        fixed_source,moving_source = sources if direction == "forward" else sources[::-1]
        t = time.monotonic()
        fixed = itk.imread(fixed_source["path"],itk.F)
        moving = itk.imread(moving_source["path"],itk.F)
        meta["geometry_checks"] = [check_itk_geometry(fixed,fixed_source),check_itk_geometry(moving,moving_source)]
        for image in (fixed,moving):
            # Slice-wise finiteness avoids an additional full-size Boolean volume.
            data = itk.array_view_from_image(image)
            require(all(np.isfinite(s).all() for s in data), "Non-finite CT input")
        meta["load_seconds"] = time.monotonic()-t
        t = time.monotonic()
        filt = itk.ElastixRegistrationMethod.New(fixed,moving)
        filt.SetParameterObject(parameter_object(m["parameters"]))
        filt.SetNumberOfThreads(threads)
        filt.SetOutputDirectory(str(work))
        filt.SetLogToFile(True)
        filt.SetLogToConsole(False)
        filt.UpdateLargestPossibleRegion()
        maps = normalized_transform_maps(filt.GetTransformParameterObject())
        meta["registration_seconds"] = time.monotonic()-t
        files.extend(save_transform_chain(work/"transforms",maps))
        atomic_json(work/"transform_maps.json",maps,refuse=True)
        meta["maps_json"] = identity(work/"transform_maps.json")
        files.append(meta["maps_json"])
        # Verify a reloaded complete chain on deterministic physical sample points.
        shape = np.asarray(fixed_source["native_shape_xyz"])
        pts = apply_affine(np.array([(shape-1)*v for v in (.25,.5,.75)]),lps_affine(fixed_source))
        a = transformix_points(pts,maps,threads)
        b = transformix_points(pts,load_json(work/"transform_maps.json"),threads)
        require(np.isfinite(a).all() and np.max(np.abs(a-b)) <= 1e-4, "Transform chain reload failed")
        meta["transform_reload_max_mm"] = float(np.max(np.abs(a-b)))
        if subject == PILOT:
            from tools.quadra.registration_report import registration_qc
            registration_qc(fixed,moving,fixed_source,moving_source,maps,threads,work/"registration_qc.png")
            files.append(identity(work/"registration_qc.png"))
        del fixed,moving,filt
    else:
        forward = validate_result(m,subject,"forward",args.run_directory/"subjects"/subject/"forward.json")
        backward = validate_result(m,subject,"backward",args.run_directory/"subjects"/subject/"backward.json")
        require(forward and backward,"Missing forward/backward transforms")
        fm,bm = load_json(forward["maps_json"]["path"]),load_json(backward["maps_json"]["path"])
        output = evaluate_cycle(rows_for(m,subject),*sources,
            lambda p:transformix_points(p,fm,threads),lambda p:transformix_points(p,bm,threads))
        atomic_csv(work/"points.csv",output,refuse=True)
        meta.update(point_csv=identity(work/"points.csv"),queries=len(output),
                    valid_queries=sum(r["valid_cycle"] for r in output),
                    invalid_queries=sum(not r["valid_cycle"] for r in output))
        files.append(meta["point_csv"])
    import resource
    meta["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    require(meta["peak_rss_bytes"] <= m["limits"]["ram_ceiling_bytes"],"Worker exceeded RAM limit")
    meta.update(files=files,wall_time_seconds=time.monotonic()-started,completed_at=utc_now())
    if (work/"elastix.log").exists():
        meta["files"].append(identity(work/"elastix.log"))
    require(not forbidden_outputs(work),"Forbidden full-volume output retained")
    atomic_json(work/"worker_result.json",meta,refuse=True)


def log_progress(attempt):
    logfile = attempt/"elastix.log"
    if not logfile.exists():
        return {"log_bytes":0,"last_line":""}
    with logfile.open("rb") as stream:
        stream.seek(max(0,logfile.stat().st_size-2048))
        lines = stream.read().decode(errors="replace").splitlines()
    return {"log_bytes":logfile.stat().st_size,"last_line":lines[-1][:300] if lines else ""}


def stop_owned_process(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()


def assert_no_orphan_worker(run_dir):
    import psutil
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = proc.info["cmdline"] or []
        if "tools.quadra.registration_cycle_error_cohort" in cmd and "_worker" in cmd and str(run_dir) in cmd:
            raise RegistrationError("Existing worker {} still owns this run; do not start a duplicate".format(proc.pid))


def run_task(run_dir,m,subject,direction):
    import psutil
    destination = run_dir/"subjects"/subject
    destination.mkdir(parents=True,exist_ok=True)
    marker = destination/(direction+".json")
    previous = validate_result(m,subject,direction,marker)
    if previous:
        return previous
    # Completed but unpublished worker outputs survive controller interruption.
    for result in sorted(destination.glob(direction+"-attempt-*/worker_result.json")):
        recovered = validate_result(m,subject,direction,result)
        atomic_json(marker,recovered,refuse=True)
        return recovered
    for retry in range(2):
        assert_no_orphan_worker(run_dir)
        attempt = destination/(direction+"-attempt-"+str(time.time_ns()))
        attempt.mkdir()
        env = dict(os.environ,ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(m["limits"]["threads"]),
                   OMP_NUM_THREADS=str(m["limits"]["threads"]),OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        cmd = [sys.executable,"-u","-m","tools.quadra.registration_cycle_error_cohort","_worker",
               "--run-directory",str(run_dir),"--subject",subject,"--direction",direction,
               "--attempt-directory",str(attempt)]
        start = time.monotonic()
        peak = 0
        timed_out = False
        with (attempt/"stdout.log").open("w") as output:
            proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            observed_process = psutil.Process(proc.pid)
            observed_process.cpu_percent(None)
            status_update(run_dir,worker_pid=proc.pid,current_subject=subject,current_direction=direction,
                          worker_started_at=utc_now(),worker_create_time=observed_process.create_time(),
                          attempt_directory=str(attempt),retry=retry)
            try:
                while proc.poll() is None:
                    if time.monotonic()-start > m["limits"]["direction_timeout_seconds"]:
                        timed_out = True
                        break
                    try:
                        pp = observed_process
                        rss = sum(p.memory_info().rss for p in [pp]+pp.children(recursive=True) if p.is_running())
                        cpu = pp.cpu_percent(None)
                    except psutil.NoSuchProcess:
                        rss,cpu = 0,0
                    peak = max(peak,rss)
                    resources = check_resources(m["limits"],run_dir,rss)
                    resources["worker_cpu_percent"] = cpu
                    status_update(run_dir,resources=resources,progress=log_progress(attempt),worker_peak_rss_bytes=peak)
                    time.sleep(5)
            finally:
                stop_owned_process(proc)
                status_update(run_dir,worker_pid=None)
        if proc.returncode == 0:
            meta = validate_result(m,subject,direction,attempt/"worker_result.json")
            require(meta is not None,"Worker exited without result")
            meta["peak_rss_bytes"] = max(peak,meta["peak_rss_bytes"])
            meta["controller_wall_time_seconds"] = time.monotonic()-start
            meta["retry_index"] = retry
            atomic_json(marker,meta,refuse=True)
            return meta
        error_path = attempt/"worker_error.json"
        failure = ({"classification":"timeout", "message":"Six-hour worker limit exceeded"} if timed_out
                   else load_json(error_path) if error_path.exists() else
                   {"classification":"process_exit","exit_code":proc.returncode})
        atomic_json(attempt/"controller_failure.json",failure,refuse=True)
        if failure["classification"] == "contract_error":
            raise RegistrationError(failure["message"])
        if retry == 1:
            raise RuntimeFailure("{} {} failed twice; see {}".format(subject,direction,attempt))


def counters(run_dir,m):
    subjects,valid,invalid,registrations = 0,0,0,0
    for subject in SUBJECTS:
        for d in ("forward","backward"):
            if validate_result(m,subject,d,run_dir/"subjects"/subject/(d+".json")):
                registrations += 1
        result = validate_result(m,subject,"points",run_dir/"subjects"/subject/"points.json")
        if result:
            subjects += 1
            valid += result["valid_queries"]
            invalid += result["invalid_queries"]
    return dict(subjects_completed=subjects,queries_valid=valid,queries_invalid=invalid,registrations_completed=registrations)


def execute(args,pilot):
    run_dir = args.run_directory.resolve()
    with controller_lock(run_dir):
        m = load_run(run_dir)
        assert_no_orphan_worker(run_dir)
        if pilot and (run_dir/"pilot_checkpoint.json").exists():
            saved = load_json(run_dir/"pilot_checkpoint.json")
            require(saved["manifest_signature"] == m["signature"], "Pilot signature changed")
            for item in saved["evidence"]+[saved["report"]]:
                verify_identity(item)
            print("REVIEW_REQUIRED: existing pilot evidence verified; no cohort launched", flush=True)
            return
        if not pilot:
            approval = load_json(run_dir/"pilot_approval.json")
            require(approval["manifest_signature"] == m["signature"],"Wrong pilot approval")
            verify_identity(approval["pilot_checkpoint"])
            for item in load_json(approval["pilot_checkpoint"]["path"])["evidence"]:
                verify_identity(item)
            require(approval["human_approved"] is True,"Pilot not approved")
        require(not (run_dir/"checkpoint_summary.json").exists(),"Run already finalized")
        failures = []
        import psutil
        status_update(run_dir,status="RUNNING",mode="pilot" if pilot else "cohort",controller_pid=os.getpid(),
                      controller_started_at=utc_now(),controller_create_time=psutil.Process().create_time(),
                      failed_subjects=failures,error=None,**counters(run_dir,m))
        outcome = None
        try:
            for subject in ([PILOT] if pilot else [s for s in SUBJECTS if s != PILOT]):
                try:
                    for direction in ("forward","backward","points"):
                        print("{} {}".format(subject,direction),flush=True)
                        run_task(run_dir,m,subject,direction)
                        status_update(run_dir,**counters(run_dir,m))
                except RuntimeFailure as exc:
                    failures.append({"subject_id":subject,"reason":str(exc)})
                    status_update(run_dir,failed_subjects=failures)
                    if pilot:
                        raise
            require(not forbidden_outputs(run_dir),"Forbidden full-volume artifact")
            require(repository() == m["repository"],"Repository changed during execution")
            count = counters(run_dir,m)
            if pilot:
                from tools.quadra.registration_report import build_report
                report = build_report(run_dir,m,pilot=True)
                result = validate_result(m,PILOT,"points",run_dir/"subjects"/PILOT/"points.json")
                evidence = [identity(run_dir/"subjects"/PILOT/(d+".json")) for d in ("forward","backward","points")]
                evidence += [f for d in ("forward","backward","points") for f in load_json(run_dir/"subjects"/PILOT/(d+".json"))["files"]]
                evidence += [identity(p) for p in sorted(report.parent.rglob("*")) if p.is_file()]
                outcome = "REVIEW_REQUIRED"
                payload = {"status":outcome,"technical_gates_passed":result["invalid_queries"] == 0,
                           "manifest_signature":m["signature"],"subject_id":PILOT,"evidence":evidence,
                           "report":identity(report),"counts":count,"created_at":utc_now()}
                atomic_json(run_dir/"pilot_checkpoint.json",payload,refuse=True)
            else:
                outcome = "TECHNICAL_PASS" if count["subjects_completed"] == 28 and count["queries_valid"] == 108431 and not failures else "PARTIAL"
        except RegistrationError as exc:
            outcome = "BLOCKED"
            status_update(run_dir,error=str(exc))
            raise
        except BaseException as exc:
            outcome = "INCOMPLETE"
            status_update(run_dir,error=str(exc))
            raise
        finally:
            try:
                count = counters(run_dir,m)
            except Exception as exc:
                count = {}
                outcome = "BLOCKED"
                status_update(run_dir,error="Result integrity check failed: "+str(exc))
            status_update(run_dir,status=outcome or "INCOMPLETE",controller_pid=None,worker_pid=None,
                          failed_subjects=failures,**count)
        print("{}: {}".format(outcome,run_dir),flush=True)
    if not pilot:
        finalize(args)


def approve(args):
    m = load_run(args.run_directory)
    checkpoint = args.run_directory/"pilot_checkpoint.json"
    cp = load_json(checkpoint)
    require(args.confirm_review and len(args.review_rationale.strip()) >= 20,"Explicit human QC review rationale required")
    require(cp["status"] == "REVIEW_REQUIRED" and cp["technical_gates_passed"] is True,"Pilot has unresolved technical/invalid-point issues")
    require(cp["manifest_signature"] == m["signature"],"Pilot contract mismatch")
    for record in cp["evidence"]+[cp["report"]]:
        verify_identity(record)
    atomic_json(args.run_directory/"pilot_approval.json",{
        "human_approved":True,"review_rationale":args.review_rationale,"created_at":utc_now(),
        "manifest_signature":m["signature"],"pilot_checkpoint":identity(checkpoint)},refuse=True)
    print("Pilot approval recorded; cohort has not been launched")


def finalize(args):
    run_dir = args.run_directory.resolve()
    with controller_lock(run_dir):
        m = load_run(run_dir,active=False)
        status = load_json(run_dir/"cohort_status.json")
        require(status["status"] in TERMINAL,"Cannot finalize an active/review-pending run")
        require(not forbidden_outputs(run_dir),"Forbidden retained volume")
        marker = run_dir/"checkpoint_summary.json"
        if marker.exists():
            for f in load_json(marker)["outputs"]:
                verify_identity(f)
            print("Already finalized; evidence verified")
            return
        from tools.quadra.registration_report import build_report
        report = build_report(run_dir,m,pilot=False)
        outputs = [identity(p) for p in sorted(report.parent.rglob("*")) if p.is_file()]
        atomic_json(marker,{"status":status["status"],"workflow":WORKFLOW,"created_at":utc_now(),
            "manifest_signature":m["signature"],"counts":counters(run_dir,m),"outputs":outputs,
            "anatomical_accuracy_validated":False,"uae_comparison_performed":False},refuse=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--storage-root",type=Path,default=Path("/workspace/quadra"))
    p.add_argument("--uae-run-directory",type=Path,default=Path(SOURCE_RUN))
    p.add_argument("--threads",type=int,choices=(1,),default=None,
                   help="Explicit single-thread runtime revision; omission retains allocated-CPU policy")
    p.add_argument("--thread-rationale",default="",
                   help="Required explanation for --threads 1, frozen in the new manifest")
    for command in ("pilot","approve-pilot","run","status","finalize","_worker"):
        p = sub.add_parser(command)
        p.add_argument("--run-directory",required=True,type=Path)
        if command == "approve-pilot":
            p.add_argument("--confirm-review",action="store_true")
            p.add_argument("--review-rationale",required=True)
        if command == "status":
            p.add_argument("--json",action="store_true")
        if command == "_worker":
            p.add_argument("--subject",required=True)
            p.add_argument("--direction",required=True,choices=("forward","backward","points"))
            p.add_argument("--attempt-directory",required=True,type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare": prepare(args)
        elif args.command == "pilot": execute(args,True)
        elif args.command == "run": execute(args,False)
        elif args.command == "approve-pilot": approve(args)
        elif args.command == "finalize": finalize(args)
        elif args.command == "_worker": worker(args)
        else: print(json.dumps(load_json(args.run_directory/"cohort_status.json"),indent=2))
        return 0
    except Exception as exc:
        if args.command == "_worker":
            atomic_json(args.attempt_directory/"worker_error.json",{
                "classification":"contract_error" if isinstance(exc,RegistrationError) else "runtime_error",
                "message":str(exc),"traceback":traceback.format_exc()},refuse=True)
        print("ERROR: {}".format(exc),file=sys.stderr)
        traceback.print_exc()
        if args.command in ("pilot", "run"):
            try:
                # Preserve a failure report without sealing an interrupted run;
                # a later explicit invocation may safely resume validated work.
                with controller_lock(args.run_directory):
                    status = load_json(args.run_directory/"cohort_status.json")
                    if status["status"] in TERMINAL and status.get("controller_pid") is None:
                        from tools.quadra.registration_report import build_report
                        build_report(args.run_directory,load_run(args.run_directory,active=False),args.command == "pilot")
            except Exception as report_error:
                print("Failure report unavailable: {}".format(report_error),file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
