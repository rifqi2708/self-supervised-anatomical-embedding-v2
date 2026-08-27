"""Private Markdown/PNG technical report for the subject-044 regional pilot."""
from collections import Counter
from pathlib import Path
import os

import numpy as np

from tools.quadra import registration_point_transform as pt
from tools.quadra.registration_report import summaries


def point_series(rows, field):
    valid = [r for r in rows if str(r["valid_cycle"]) == "True"]
    pt.require(len({r["query_id"] for r in valid}) == len(valid), "Duplicate plot query IDs")
    pt.require(all(np.isfinite(float(r["cycle_error_mm"])) and float(r["cycle_error_mm"]) >= 0
                   for r in valid), "Invalid plotted cycle error")
    return [(name, np.array([float(r["cycle_error_mm"]) for r in valid if r[field] == name]))
            for name in sorted({r[field] for r in valid})]


def plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.quadra.registration_organ_group import GROUPS
    valid_values = [float(r["cycle_error_mm"]) for r in rows if r["valid_cycle"] == "True"]
    if not valid_values:
        return []
    paths = []
    ymax = max(1, max(valid_values)*1.06)
    for group in GROUPS+("combined",):
        items = [r for r in rows if group == "combined" or r["group_name"] == group]
        series = point_series(items, "group_name" if group == "combined" else "mask_name")
        valid = sum(len(values) for _, values in series)
        if not valid:
            continue
        for kind in ("boxplot", "ecdf", "point_jitter"):
            fig, ax = plt.subplots(figsize=(max(10, len(series)*.7), 6.5))
            labels = [name.replace("_", " ")+"\n(n={})".format(len(v)) for name,v in series]
            if kind == "boxplot":
                ax.boxplot([v for _,v in series], tick_labels=labels, showfliers=True,
                           flierprops={"markersize":2, "alpha":.3}, medianprops={"color":"#2767A5"})
                ax.tick_params(axis="x", rotation=70)
                ax.set_ylabel("Cycle error (mm)")
            elif kind == "point_jitter":
                rng = np.random.default_rng(20260721)
                for i, (_, values) in enumerate(series):
                    ax.scatter(i+rng.uniform(-.22, .22, len(values)), values, s=8,
                               color="#2767A5", alpha=.4, linewidths=0)
                ax.set_xticks(range(len(series)), labels, rotation=70 if group != "combined" else 0)
                ax.set_ylabel("Individual query cycle error (mm)")
                ax.set_ylim(0, ymax)
                fig.text(.5, .012, "One dot per valid query; horizontal jitter only; no averaging or subsampling.",
                         ha="center", fontsize=9)
            else:
                curves = series if group == "combined" else [(group, np.concatenate([v for _,v in series]))]
                for i, (name, values) in enumerate(curves):
                    ax.step(np.sort(values), np.arange(1, len(values)+1)/len(values), where="post",
                            label=name.replace("_", " "), color=("#2767A5", "#A36F19", "#D26B32", "#697B38")[i],
                            linestyle=("-", "--", ":", "-.")[i])
                ax.legend()
                ax.set_xlabel("Cycle error (mm)")
                ax.set_ylabel("Fraction of valid queries")
            ax.set_title("Organ-group registration {} — {}\nSubject 044; {:,} valid / {:,} expected; {} invalid".format(
                kind.replace("_", " "), group.replace("_", " "), valid, len(items), len(items)-valid))
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            fig.tight_layout(rect=(0, .04, 1, 1))
            path = output/(group+"_"+kind+".png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def build_report(run, m):
    from tools.quadra.registration_organ_group import GROUPS, validate_result, pilot_rows
    out = run/"analysis"/("pilot-report-"+str(__import__("time").time_ns()))
    out.mkdir(parents=True, exist_ok=False)
    rows, profiles, qc, crop_rows = [], [], [], []
    for group in GROUPS:
        for direction in ("forward", "backward", "points"):
            meta = validate_result(m, group, direction, run/"groups"/group/(direction+".json"))
            pt.require(meta is not None, "Incomplete pilot report")
            profiles.append({key:meta.get(key, "") for key in ("group_name", "direction", "load_seconds", "registration_seconds",
                                                              "wall_time_seconds", "peak_rss_bytes", "retry_index")})
            qc += [(group, direction, Path(f["path"])) for f in meta["files"] if f["path"].endswith("registration_qc.png")]
            if direction == "points":
                rows += pt.read_csv(meta["point_csv"]["path"])
        for session in ("test", "retest"):
            plan = pt.load_json(m["plans"][session+"-"+group]["path"])
            crop_rows.append(dict(group_name=group, session=session, start_xyz=str(plan["crop_start_xyz"]),
                                  end_xyz=str(plan["crop_end_xyz"]), shape_xyz=str(plan["crop_geometry"]["native_shape_xyz"]),
                                  voxel_count=int(np.prod(plan["crop_geometry"]["native_shape_xyz"])),
                                  maximum_rounding_mm=float(np.max(plan["outward_rounding_mm"])),
                                  roundtrip_voxels=plan["geometry_checks"]["max_roundtrip_voxels"],
                                  roundtrip_mm=plan["geometry_checks"]["max_roundtrip_mm"]))
    expected = pilot_rows(m)
    pt.require(len(rows) == len(expected) and {r["query_id"] for r in rows} == {r["query_id"] for r in expected}, "Report query mismatch")
    rows.sort(key=lambda r: r["query_id"])
    pt.atomic_csv(out/"cycle_error_points.csv", rows, refuse=True)
    tables = {}
    for name, fields in (("pooled", []), ("group", ["group_name"]), ("mask", ["group_name", "mask_name"])):
        tables[name] = summaries(rows, fields)
        pt.atomic_csv(out/(name+"_summary.csv"), tables[name], refuse=True)
    pt.atomic_csv(out/"runtime_memory.csv", profiles, refuse=True)
    pt.atomic_csv(out/"native_crop_geometry.csv", crop_rows, refuse=True)
    invalid = Counter(r["failure_reason"] for r in rows if r["valid_cycle"] == "False")
    pt.atomic_csv(out/"invalid_reasons.csv", [{"reason":k,"count":v} for k,v in sorted(invalid.items())],
                  fields=["reason", "count"], refuse=True)
    figures = plots(rows, out)
    valid = sum(r["valid_cycle"] == "True" for r in rows)
    lines = ["# Subject 044 organ-group registration pilot", "", "## Technical summary", "",
             "**REVIEW_REQUIRED.** All eight directional registrations and four group evaluations completed. "
             "{} / {} queries have valid continuous cycles; {} are invalid. The cohort has not been authorized or launched.".format(valid,len(rows),len(rows)-valid), "",
             "This is a one-subject technical and cycle-consistency result, not anatomical accuracy or a UAE-S comparison.", "",
             "## Scope and measurement", "",
             "The subject and original frozen Test queries are identical to the whole-body pilot. Each group has independent Test-to-Retest "
             "and Retest-to-Test rigid-plus-B-spline transforms. Cycle error is the Euclidean distance between the original and returned Test "
             "points in physical millimetres. Only valid rows enter summaries; all invalid rows and reasons remain in the CSV.", "",
             "## Cycle consistency by group", "", "| Group | Expected | Valid | Invalid | Median mm | P95 mm |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in tables["group"]:
        fmt = lambda x: "n/a" if x == "" else "{:.3f}".format(x)
        lines.append("| {} | {} | {} | {} | {} | {} |".format(r["group_name"],r["expected_queries"],r["valid_queries"],
                     r["invalid_queries"],fmt(r["median_mm"]),fmt(r["p95_mm"])))
    lines += ["", "Use the plots to inspect spread and tails. Low cycle error does not prove a correct anatomical match. "
              "All dots represent individual queries; queries from this one subject are not independent subjects.", ""]
    for group in GROUPS+("combined",):
        lines += ["### "+group.replace("_", " ").title(), "",
                  "The box plot summarizes quartiles and outliers, the ECDF shows the fraction below each error, and the jitter plot "
                  "retains every valid point with its exact Y value. Invalid counts are displayed rather than represented as zero.", ""]
        for p in figures:
            if p.name.startswith(group+"_"):
                lines += ["![{}]({})".format(p.stem,p.name), ""]
    lines += ["## Cropping and registration method", "",
              "Each scan uses its own frozen UAE-S organ-group plan: mask bounding-box union plus nominal 100 mm margin and the "
              "already-frozen outward stride snapping. Only the valid acquired-image physical extent is mapped back to the native "
              "CT grid, rounded outward and clamped to the acquired FOV. No new body/mask detection, resampling, HU normalization, "
              "metric masks or neural-network padding is used. Native spacing, direction and physical origin are preserved.", "",
              "This is comparable physical coverage, not identical UAE-S tensors: native-grid rounding and FOV boundaries are recorded "
              "in native_crop_geometry.csv. Non-padding regions come from different Test and Retest masks, not a shared voxel box.", "",
              "The complete parameter maps are inherited unchanged from the approved whole-body run: four resolutions, 256 maximum "
              "iterations per resolution, 8,192 RandomCoordinate samples, new samples each iteration, seed 121212, geometrical-centre "
              "rigid initialization, 32 mm final B-spline grid and -1024 HU background. FP32 native CTs; one ITK thread.", "",
              "Transformix evaluates the complete composition at continuous LPS points. Only compatibility export coordinates use np.rint. "
              "Physical export columns are RAS; raw XYZ always refers to the original CT, not the local crop.", "",
              "## Geometry, runtime and failures", "",
              "All crop transforms and actual ITK output geometry are checked. Native voxel-centre crop support defines evaluability. "
              "A forward point outside the Retest crop is retained as invalid and is never passed to the reverse mapping; returned "
              "points outside the Test crop are also invalid. There is no silent clamping or extrapolation.", "",
              "Per-task runtime and peak RSS are in runtime_memory.csv. Frozen guards use one thread, a six-hour direction timeout, "
              "10 GiB minimum free quota space and the existing 80% effective RAM ceiling. Guards may stop only this pipeline's workers.", "",
              "## Visual registration QC", "",
              "Each overlay uses central physical planes of the fixed crop. Red is fixed CT and cyan is moving CT; agreement appears grey. "
              "These bounded views help identify gross alignment errors but cannot establish registration accuracy across the complete crop.", ""]
    for group,direction,path in qc:
        lines += ["### {} — {}".format(group.replace("_", " "),direction), "",
                  "![Registration overlay]({})".format(os.path.relpath(path,out)), ""]
    lines += ["## Limitations and next decision", "",
              "Both scans' segmentations provide an anatomical localization prior. Four independent group transforms do not form a "
              "single globally continuous whole-body transform; overlapping crops can produce different mappings. Cropping also changes "
              "the registration image centre, pyramid support and metric sampling even though parameter maps are unchanged.", "",
              "Review all eight overlays, geometry/containment checks, invalid rows and resource profiles before approving any cohort "
              "implementation. Any invalid-point or technical discrepancy requires explicit resolution. No automatic cohort continuation "
              "is available in this pilot CLI. Leave the pod running and arrange a checksum-verified evidence backup.", "",
              "Open questions: are the regional alignments acceptable on visual review, and does this pilot justify a separately approved "
              "28-subject regional experiment? This pilot cannot answer cohort generalizability or anatomical accuracy.", "",
              "## Reproducibility", "",
              "Manifest signature: `{}`. Execution commit: `{}`. Frozen full query SHA-256: `{}`.".format(
                  m["signature"],m["repository"]["commit"],m["queries"]["sha256"]), ""]
    report = out/"pilot_report.md"
    pt.atomic_text(report, "\n".join(lines), refuse=True)
    pt.atomic_json(out/"analysis_manifest.json", {"source_manifest":pt.identity(run/"organ_group_manifest.json"),
                   "created_at":pt.utc_now(), "expected_queries":len(rows), "valid_queries":valid,
                   "invalid_queries":len(rows)-valid, "scope":"subject_044_pilot_only", "cohort_authorized":False,
                   "jitter_contract":{"observation":"individual_query","y":"cycle_error_mm","subsampling":False,"vertical_jitter":False},
                   "report_surface":"private_markdown_and_png_as_requested", "report_generator":pt.identity(__file__),
                   "outputs":[pt.identity(p) for p in sorted(out.iterdir()) if p.is_file()]}, refuse=True)
    return report
