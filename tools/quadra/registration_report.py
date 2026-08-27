"""Registration-only Markdown reports and bounded slice QC."""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np

from tools.quadra.registration_point_transform import (apply_affine, lps_affine,
    transformix_points, atomic_text, atomic_csv, atomic_json, identity, read_csv, load_json,
    require, utc_now)


def registration_qc(fixed,moving,fixed_source,moving_source,maps,threads,path):
    import itk
    from scipy.ndimage import map_coordinates
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shape = np.asarray(fixed_source["native_shape_xyz"])
    spacing = np.linalg.norm(np.asarray(fixed_source["affine"])[:3,:3],axis=0)
    fixed_data = itk.array_view_from_image(fixed)
    moving_data = itk.array_view_from_image(moving)
    fig,axes = plt.subplots(3,3,figsize=(10,10))
    def sample(data,raw):
        return map_coordinates(data,raw.T[[2,1,0]],order=1,mode="constant",cval=-1024,prefilter=False).reshape(192,192)
    def window(data):
        return np.clip((data+160)/400,0,1)
    for row,(name,u,v,slice_axis) in enumerate((("axial",0,1,2),("coronal",0,2,1),("sagittal",1,2,0))):
        uu,vv = np.meshgrid(np.linspace(0,shape[u]-1,192),np.linspace(0,shape[v]-1,192))
        raw = np.tile((shape-1)/2,(192*192,1))
        raw[:,u],raw[:,v] = uu.ravel(),vv.ravel()
        physical = apply_affine(raw,lps_affine(fixed_source))
        before_raw = apply_affine(physical,np.linalg.inv(lps_affine(moving_source)))
        after = transformix_points(physical,maps,threads)
        after_raw = apply_affine(after,np.linalg.inv(lps_affine(moving_source)))
        fixed_plane = window(sample(fixed_data,raw))
        before_plane = window(sample(moving_data,before_raw))
        after_plane = window(sample(moving_data,after_raw))
        extent = [0,(shape[u]-1)*spacing[u],0,(shape[v]-1)*spacing[v]]
        axes[row,0].imshow(fixed_plane,cmap="gray",origin="lower",extent=extent,aspect="equal")
        for col,plane in ((1,before_plane),(2,after_plane)):
            # Red/cyan image overlay: disagreement is colored, agreement grey.
            axes[row,col].imshow(np.stack([fixed_plane,plane,plane],axis=-1),origin="lower",extent=extent,aspect="equal")
        for col,title in enumerate(("Fixed CT","Before overlay","After overlay")):
            axes[row,col].set_title(name+" — "+title,fontsize=10)
            axes[row,col].axis("off")
    fig.suptitle("Native registration QC — central physical planes\nRed: fixed; cyan: moving. Not anatomical ground truth.",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(path,dpi=160)
    plt.close(fig)


def summaries(rows,fields):
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[f] for f in fields)].append(r)
    output = []
    for key,items in sorted(buckets.items()):
        valid = np.asarray([float(r["cycle_error_mm"]) for r in items if str(r["valid_cycle"]) == "True"])
        require(np.isfinite(valid).all(),"Non-finite valid cycle errors in report")
        record = dict(zip(fields,key))
        record.update(expected_queries=len(items),valid_queries=len(valid),invalid_queries=len(items)-len(valid))
        for label,fn in (("mean",np.mean),("median",np.median),("q25",lambda x:np.percentile(x,25)),
                         ("q75",lambda x:np.percentile(x,75)),("p95",lambda x:np.percentile(x,95))):
            record[label+"_mm"] = float(fn(valid)) if len(valid) else ""
        output.append(record)
    return output


def plots(rows,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [r for r in rows if str(r["valid_cycle"]) == "True"]
    if not valid:
        return []
    paths = []
    groups = ("pelvis","abdomen","thorax","head_neck")
    for group in groups+("combined",):
        items = [r for r in valid if group == "combined" or r["group_name"] == group]
        field = "group_name" if group == "combined" else "mask_name"
        cats = sorted({r[field] for r in items})
        if not cats:
            continue
        series = [np.asarray([float(r["cycle_error_mm"]) for r in items if r[field] == c]) for c in cats]
        for kind in ("boxplot","ecdf","subject_jitter"):
            fig,ax = plt.subplots(figsize=(max(9,len(cats)*.65),6))
            if kind == "boxplot":
                ax.boxplot(series,tick_labels=cats,showfliers=True,
                           flierprops={"markersize":1,"alpha":.25},medianprops={"color":"#2767A5"})
                ax.tick_params(axis="x",rotation=70)
                ax.set_ylabel("Cycle error (mm)")
            elif kind == "ecdf":
                # One curve per group (combined) or pooled points (within group).
                plot_series = series if group == "combined" else [np.concatenate(series)]
                labels = cats if group == "combined" else [group]
                colors = ["#2767A5","#A36F19","#D26B32","#697B38"]
                for i,(values,label) in enumerate(zip(plot_series,labels)):
                    x = np.sort(values)
                    ax.step(x,np.arange(1,len(x)+1)/len(x),where="post",label=label,
                            color=colors[i%4],linestyle=("-","--",":","-.")[i%4])
                ax.set_xlabel("Cycle error (mm)")
                ax.set_ylabel("Fraction of valid queries")
                ax.legend()
            else:
                rng = np.random.default_rng(20260721)
                for i,c in enumerate(cats):
                    buckets = defaultdict(list)
                    for r in items:
                        if r[field] == c:
                            buckets[r["subject_id"]].append(float(r["cycle_error_mm"]))
                    medians = [np.median(v) for _,v in sorted(buckets.items())]
                    ax.scatter(i+rng.uniform(-.16,.16,len(medians)),medians,s=22,color="#2767A5",alpha=.65)
                ax.set_xticks(range(len(cats)),cats,rotation=70)
                ax.set_ylabel("Subject median cycle error (mm)")
            ax.set_title("Registration {} — {}\n{} valid queries; invalid points excluded and reported separately".format(kind.replace("_"," "),group,len(items)))
            ax.grid(axis="y",alpha=.18)
            ax.set_axisbelow(True)
            fig.tight_layout()
            path = output/(group+"_"+kind+".png")
            fig.savefig(path,dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def build_report(run_dir,m,pilot):
    from tools.quadra.registration_cycle_error_cohort import validate_result, rows_for, PILOT, SUBJECTS
    run_dir = Path(run_dir)
    target = run_dir/"analysis"/(("pilot-report-" if pilot else "run-report-")+str(time.time_ns()))
    target.mkdir(parents=True)
    rows,profiles,failures,qc_images = [],[],[],[]
    subjects = [PILOT] if pilot else SUBJECTS
    for subject in subjects:
        point = validate_result(m,subject,"points",run_dir/"subjects"/subject/"points.json")
        if point:
            rows.extend(read_csv(point["point_csv"]["path"]))
        else:
            # Explicit denominator rows; missing output is not silently filtered.
            for r in rows_for(m,subject):
                rows.append(dict(r,valid_cycle=False,cycle_error_mm="",failure_reason="subject_incomplete"))
        for d in ("forward","backward","points"):
            meta = validate_result(m,subject,d,run_dir/"subjects"/subject/(d+".json"))
            if meta:
                attempts = sorted((run_dir/"subjects"/subject).glob(d+"-attempt-*"))
                profiles.append({"subject_id":subject,"direction":d,"wall_time_seconds":meta["wall_time_seconds"],
                                 "peak_rss_bytes":meta["peak_rss_bytes"],"load_seconds":meta.get("load_seconds",""),
                                 "registration_seconds":meta.get("registration_seconds",""),
                                 "attempts":len(attempts), "retry_or_resume_attempts":max(0,len(attempts)-1)})
                qc_images += [Path(f["path"]) for f in meta["files"] if f["path"].endswith("registration_qc.png")]
            for p in sorted((run_dir/"subjects"/subject).glob(d+"-attempt-*/controller_failure.json")):
                f = load_json(p)
                failures.append({"subject_id":subject,"direction":d,"classification":f["classification"],
                                 "message":f.get("message",str(f.get("exit_code",""))),"evidence":str(p)})
    expected = sum(m["denominators"]["subject_query_counts"][s] for s in subjects)
    require(len(rows) == expected and len({r["query_id"] for r in rows}) == expected,"Report denominator mismatch")
    fields = sorted(set().union(*(r.keys() for r in rows)))
    atomic_csv(target/"cycle_error_points.csv",rows,fields=fields,refuse=True)
    tables = {}
    for name,fields in (("pooled",[]),("subject",["subject_id"]),("group",["group_name"]),("mask",["mask_name"])):
        tables[name] = summaries(rows,fields)
        atomic_csv(target/(name+"_summary.csv"),tables[name],refuse=True)
    if profiles:
        atomic_csv(target/"runtime_memory.csv",profiles,refuse=True)
    atomic_csv(target/"worker_failures.csv",failures,
               fields=["subject_id","direction","classification","message","evidence"],refuse=True)
    invalid = defaultdict(int)
    for r in rows:
        if str(r["valid_cycle"]) != "True":
            invalid[r["failure_reason"]] += 1
    atomic_csv(target/"invalid_reasons.csv",[{"reason":k,"queries":v} for k,v in sorted(invalid.items())],
               fields=["reason","queries"],refuse=True)
    figures = plots(rows,target)
    pooled = tables["pooled"][0]
    status = load_json(run_dir/"cohort_status.json")["status"]
    if pilot and status == "RUNNING":
        status = "REVIEW_REQUIRED"
    lines = ["# Quadra registration {} report".format("pilot" if pilot else "cohort"),"",
        "## Technical summary","",
        "Status: **{}**. {} subjects in scope; {} expected queries; {} valid cycles; {} invalid/unavailable cycles.".format(status,len(subjects),expected,pooled["valid_queries"],pooled["invalid_queries"]),"",
        "Completed task profiles: {}. Recorded failed attempts: {}. Expected registrations: {}. Point evaluations attempted: {}.".format(len(profiles),len(failures),2*len(subjects),sum(r.get("failure_reason") != "subject_incomplete" for r in rows)),"",
        "This is an execution and cycle-consistency report, not evidence of anatomical accuracy.","",
        "## Data and measurement","",
        "The immutable UAE-S Test-mask query CSV supplies coordinates and identifiers only. No queries were regenerated and no UAE-S result comparison was performed. The planned method registers native whole-body CTs in both directions independently; actual completion is reported above. Cycle error is the Euclidean distance between the original and returned Test points in physical millimetres. Groups/masks stratify results; they do not constrain registration.","",
        "Invalid points are retained in the point table with failure reasons. Statistics exclude them and expose the valid denominator. Subject-jitter figures contain subject medians, not independent voxel observations.","",
        "## Registration method","",
        "Rigid then B-spline; 4 resolutions, 256 maximum iterations per resolution, 8192 RandomCoordinate samples, new samples each iteration; geometric-centre rigid initialization; final B-spline grid 32 mm; background -1024 HU. RandomSeed=121212. Native float32 HU inputs; no UAE normalization, resampling or cropping.","",
        "Frozen execution threads: {}. Runtime policy: {}. Revision rationale: {}.".format(
            m.get("limits",{}).get("threads","unavailable"),
            m.get("runtime_contract",{}).get("thread_policy","historical allocated-CPU policy"),
            m.get("runtime_contract",{}).get("rationale","") or "none"),"",
        "Transformix evaluates physical LPS points continuously. The independently estimated reverse transform receives the unrounded forward point. RAS physical columns are exported alongside continuous raw XYZ and separate np.rint compatibility coordinates. Dense DVFs and full warped CTs are not retained.","",
        "## Cycle-consistency results","",
        "| Group | Expected | Valid | Invalid | Median mm | P95 mm |",
        "|---|---:|---:|---:|---:|---:|"]
    for item in tables["group"]:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(item["group_name"],item["expected_queries"],item["valid_queries"],item["invalid_queries"],item["median_mm"],item["p95_mm"]))
    lines += ["","Detailed mean, quartiles and p95 values are in the pooled, subject, group and mask CSVs. No statistical significance or superiority claims are made.",""]
    for path in figures:
        lines += ["### "+path.stem.replace("_"," "),"",
                  "This figure describes valid registration cycles only. Inspect its tail and group spread alongside the invalid-point counts; low error alone does not establish correct anatomical correspondence.","",
                  "![{}]({})".format(path.stem,path.name),""]
    lines += ["## Runtime, failures and QC","",
        "Per-direction loading/registration time and peak process RAM are recorded in runtime_memory.csv when available. Worker attempts and errors remain under subjects/. Retry count is the number of attempt directories beyond the first for each task.","",
        "Worker failure classifications are in worker_failures.csv; invalid-point reasons and counts are in invalid_reasons.csv. Pilot overlays require visual review and are not a cohort-wide anatomical validation.","",
        "## Limitations and next steps","",
        "Continuous point evaluation differs from the historical dense-DVF workflow with intermediate rounding. Software/default maps and resource limits are frozen in registration_manifest.json; identical historical execution is not claimed. Field-of-view failures and incomplete subjects can bias valid-only summaries. No registration-versus-UAE-S comparison is included.","",
        "Review the pilot before cohort authorization." if pilot else "Review failures and distributions before planning a separate paired UAE-S comparison or anatomical validation.","",
        "## Reproducibility","",
        "Generated: {}. Execution commit: `{}`. Manifest signature: `{}`. Query SHA-256: `{}`.".format(utc_now(),m["repository"]["commit"],m["signature"],m["queries"]["sha256"]),"",
        "All data, tables and figures are private run evidence. Leave the pod running; request checksum-verified backup before any later lifecycle decision.",""]
    import os
    for path in qc_images:
        lines += ["### Pilot QC: "+path.parent.name,"","![Registration overlay]({})".format(os.path.relpath(path,target)),""]
    atomic_text(target/"registration_run_report.md","\n".join(lines),refuse=True)
    atomic_json(target/"analysis_manifest.json",{"created_at":utc_now(),"manifest_signature":m["signature"],
        "query_source":m["queries"],"expected_queries":expected,"valid_queries":pooled["valid_queries"],
        "invalid_queries":pooled["invalid_queries"],"uae_comparison":False,
        "outputs":[identity(p) for p in sorted(target.iterdir()) if p.is_file()]},refuse=True)
    return target/"registration_run_report.md"
