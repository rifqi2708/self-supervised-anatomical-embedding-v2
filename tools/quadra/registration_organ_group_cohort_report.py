"""Registration-only cohort report; exact point-level jitter and explicit failures."""
from collections import Counter
import os
from pathlib import Path
import time

from tools.quadra import registration_point_transform as pt
from tools.quadra.registration_report import summaries
from tools.quadra.registration_organ_group_report import plots


def collect_rows(run,m):
    from tools.quadra import registration_organ_group_cohort as reg
    rows,profiles,qc = [],[],[]
    for s in reg.SUBJECTS:
        for g in reg.GROUPS:
            dest = reg.group_dir(run,s,g)
            points = None
            for d in reg.DIRECTIONS:
                meta = reg.validate_result(m,s,g,d,dest/(d+".json"))
                if meta is None:
                    continue
                profiles.append(dict({k:meta.get(k,"") for k in ("subject_id","group_name","direction","load_seconds",
                    "registration_seconds","wall_time_seconds","controller_wall_time_seconds","peak_rss_bytes","retry_index")},
                    reused_pilot="reused_from" in meta))
                qc += [(s,g,d,Path(f["path"])) for f in meta["files"] if f["path"].endswith("registration_qc.png")]
                if d == "points":
                    points = pt.read_csv(meta["point_csv"]["path"])
            if points is not None:
                rows.extend(points)
            else:
                failed = reg.validate_failure(m,s,g,dest/"failure.json")
                pt.require(failed is not None,"Report is missing a group result or explicit failure")
                rows.extend(dict(r,valid_cycle="False",cycle_error_mm="",failure_reason="registration_failed: "+failed["message"])
                            for r in reg.rows_for(m,s,g))
    rows.sort(key=lambda r:r["query_id"])
    expected = pt.read_csv(pt.verify_identity(m["queries"]))
    pt.require(len(rows) == len({r["query_id"] for r in rows}) == 108431 and
               {r["query_id"] for r in rows} == {r["query_id"] for r in expected},"Report denominator changed")
    return rows,profiles,qc


def build_report(run,m,count,outcome):
    from tools.quadra.registration_organ_group import GROUPS
    rows,profiles,qc = collect_rows(run,m)
    out = run/"analysis"/("cohort-report-"+str(time.time_ns()))
    out.mkdir(parents=True,exist_ok=False)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    pt.atomic_csv(out/"cycle_error_points.csv",rows,fields=fields,refuse=True)
    tables = {}
    for name,keys in (("pooled",[]),("subject",["subject_id"]),("group",["group_name"]),
                      ("mask",["group_name","mask_name"]),("subject_group",["subject_id","group_name"])):
        tables[name] = summaries(rows,keys)
        pt.atomic_csv(out/(name+"_summary.csv"),tables[name],refuse=True)
    invalid = Counter(r["failure_reason"] for r in rows if r["valid_cycle"] == "False")
    pt.atomic_csv(out/"invalid_reasons.csv",[{"reason":k,"count":v} for k,v in sorted(invalid.items())],
                  fields=["reason","count"],refuse=True)
    pt.atomic_csv(out/"runtime_memory.csv",profiles,refuse=True)
    pt.require(sum(r["valid_cycle"] == "True" for r in rows) == count["queries_valid"] and
               sum(invalid.values()) == count["queries_invalid"]+count["queries_failed"],"Summary count mismatch")
    figures = plots(rows,out,scope_label="28-subject cohort")
    qc_lines = ["# Native organ-group registration QC", "", "Red: fixed; cyan: moving. Central planes only; not anatomical ground truth.", ""]
    for subject,group,direction,path in qc:
        qc_lines += ["## {} — {} — {}".format(subject,group,direction),"",
                     "![Before/after overlay]({})".format(os.path.relpath(path,out)),""]
    pt.atomic_text(out/"qc_index.md","\n".join(qc_lines),refuse=True)
    fmt = lambda x:"n/a" if x == "" else "{:.3f}".format(x)
    pooled = tables["pooled"][0]
    lines = ["# Native organ-group registration cohort", "", "## Technical summary", "",
        "**{}** — {} / 28 subjects and {} / 224 directional registrations completed; {} / 108,431 queries have valid cycles. "
        "There are {} invalid evaluations and {} queries unavailable because their group failed after retry.".format(
            outcome,count["subjects_completed"],count["registrations_completed"],count["queries_valid"],count["queries_invalid"],count["queries_failed"]),"",
        "Pooled cycle error: median {} mm, p95 {} mm. This is a technical execution and cycle-consistency result, "
        "not anatomical accuracy or a comparison with UAE-S or whole-body registration.".format(fmt(pooled["median_mm"]),fmt(pooled["p95_mm"])),"",
        "## Scope and measurement", "",
        "The cohort comprises subjects 021–048, with 12 male and 16 female subjects. All 108,431 frozen Test-mask query IDs "
        "are retained unchanged, including small-mask shortfalls. There is no replacement sampling. Each subject has four "
        "independently localized groups and two independent directional registrations per group. The approved subject-044 "
        "pilot contributes its original eight registrations and 3,914 results without rerunning or changing its evidence.","",
        "Cycle error is the Euclidean physical distance from the Test query to its returned Test point after forward and "
        "independently estimated backward mapping. Valid observations alone enter statistics; invalid and failed observations "
        "remain in the point table with empty errors and explicit reasons. Queries are nested within subjects, not independent subjects.","",
        "## Cycle-error distributions", "", "| Group | Expected | Valid | Invalid/unavailable | Median mm | P95 mm |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in tables["group"]:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(row["group_name"],row["expected_queries"],row["valid_queries"],
                     row["invalid_queries"],fmt(row["median_mm"]),fmt(row["p95_mm"])))
    for group in GROUPS+("combined",):
        lines += ["### "+group.replace("_"," ").title(),""]
        for kind,explanation in (
            ("boxplot","Boxes summarize quartiles and medians, with outliers retained. Inspect tails as well as central values; low cycle error does not prove anatomical correctness."),
            ("ecdf","The ECDF gives the fraction of valid queries at or below each error. It exposes tail behaviour without assigning zero error to excluded observations."),
            ("point_jitter","Every dot is one valid query at its exact cycle error in millimetres. Jitter is horizontal only; no averaging or subsampling is applied. Dense regions show overlap, not independent subject replication.")):
            p = out/(group+"_"+kind+".png")
            if p in figures:
                lines += [explanation,"","![{}]({})".format(p.stem,p.name),""]
    lines += ["## Unchanged native registration method", "",
        "Test and Retest each use their own frozen organ-group localization. The valid, non-padding physical extent of the "
        "aligned nominal-100-mm UAE plan is mapped outward to the native CT voxel cells and clamped to acquired FOV. "
        "The registration sees native float32 HU data, spacing, directions and origin: no 2 mm resampling, neural normalization, "
        "metric mask, additional padding, body redetection or parameter tuning is introduced.","",
        "Rigid plus B-spline: four resolutions, 256 maximum iterations per resolution, 8,192 RandomCoordinate spatial samples, "
        "new samples every iteration, seed 121212, geometrical-centre rigid initialization, final B-spline grid spacing 32 mm, "
        "background -1024 HU, one ITK thread. Complete resolved parameter maps and tested package versions are frozen in the manifest.","",
        "Transformix evaluates the full rigid/B-spline composition at continuous physical LPS points. Forward coordinates are "
        "not rounded before reverse evaluation. Physical export columns are RAS; raw XYZ columns refer to the original CT. "
        "Only compatibility coordinates use np.rint. Forward points outside the Retest crop are never reverse-extrapolated; "
        "returned points outside the Test crop are also invalid.","",
        "## Runtime, integrity and visual QC", "",
        "runtime_memory.csv contains per-subject/group/direction timings, peak worker RSS, retries and pilot reuse flags. "
        "A six-hour direction timeout, 80% effective-RAM ceiling and 10 GiB quota-free disk guard apply. Isolated runtime "
        "failures receive one identical-settings retry; contract, geometry or resource failures stop the controller. "
        "No warped CT, dense DVF or tensor volume is retained.","",
        "[Inspect all saved before/after overlays](qc_index.md). These central-plane overlays can reveal gross alignment "
        "problems but do not validate anatomy throughout each volume. Full numerical summaries are provided by subject, group and mask.","",
        "## Limitations and next review", "",
        "Cycle consistency is not anatomical ground truth. Both scans' masks provide an anatomical localization prior. "
        "Independent regional transforms may disagree in overlapping regions and do not form one continuous whole-body "
        "transform. Cropping changes image centres, pyramid support and metric sampling despite unchanged parameters. "
        "Continuous evaluation differs from the historical rounded DVF lookup method. No superiority or significance claim is made.","",
        "Review failed/invalid rows, distribution tails and QC before scientific interpretation. Which structures and "
        "subjects account for any large errors, and do corresponding anatomical landmarks agree independently of the cycle? "
        "These remain open questions. Arrange a checksum-verified local backup before any pod lifecycle decision.","",
        "## Reproducibility", "",
        "Execution commit: `{}`. Manifest signature: `{}`. Query SHA-256: `{}`. "
        "The approval and pilot checkpoint identities are frozen in the cohort manifest.".format(m["repository"]["commit"],m["signature"],m["queries"]["sha256"]),""]
    report = out/"registration_report.md"
    pt.atomic_text(report,"\n".join(lines),refuse=True)
    pt.atomic_json(out/"analysis_manifest.json",dict(created_at=pt.utc_now(),source_manifest=pt.identity(run/"organ_group_cohort_manifest.json"),
        scope="28-subject organ-group registration only",status=outcome,counts=count,report_generator=pt.identity(__file__),
        report_surface="private Markdown with PNG and CSV, as requested",jitter_contract=dict(observation="individual_query",vertical_jitter=False,subsampling=False),
        structure="technical summary; scope/definitions before distributions; method; QA; limitations; next review/questions; reproducibility",
        outputs=[pt.identity(p) for p in sorted(out.iterdir()) if p.is_file()]),refuse=True)
    return report
