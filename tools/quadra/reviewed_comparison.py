"""Descriptive, subject-paired cycle comparison with explicit denominators."""
from __future__ import print_function

import json
import math
from collections import defaultdict
from pathlib import Path

from tools.quadra import aligned_organ_group_cohort as cohort
from tools.quadra import reviewed_evidence as evidence


def _quantile(values, fraction):
    values = sorted(values)
    if not values:
        return ''
    position = (len(values)-1)*fraction
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def _valid(row):
    if row['status'] != 'success':
        return False
    error = float(row['cycle_error_mm'])
    if not math.isfinite(error) or error < 0:
        raise cohort.CohortError('Nonfinite or negative successful cycle error')
    if row['forward_status'] != 'success' or row['reverse_status'] != 'success':
        raise cohort.CohortError('Successful cycle has failed directional status')
    return True


def compare(args):
    combined = evidence.reconcile_bundles(args.contract,args.run_directory)
    bundles = combined['bundles']
    if set(bundles) != {'uae_nn','uae_fixed_point','registration'}:
        raise cohort.CohortError('Comparison requires all three method bundles')
    rules = cohort.load_json(args.rules)
    if (rules.get('status') not in ('provisional','frozen')
            or rules.get('quantile') != 'linear'
            or rules.get('organ_weighting') != 'query_pool_within_subject_group'
            or rules.get('inference') != 'none'
            or any(type(rules.get(k)) is not int or rules[k] < 1
                for k in ('minimum_valid_median','minimum_valid_p95'))):
        raise cohort.CohortError('Declare supported quantile, weighting and sparse-cell rules')
    output = Path(args.output_directory)
    if output.exists():
        raise cohort.CohortError('Analysis output already exists; use a new analysis ID')
    paired, absolute, resources = [], [], []
    indexes = {method:{r['query_id']:r for r in bundle['rows']} for method,bundle in bundles.items()}
    groups = defaultdict(list)
    for query in combined['queries']:
        groups[(query['subject_id'],query.get('group_name','unknown'))].append(query['query_id'])
    for (subject,group), query_ids in sorted(groups.items()):
        for method in ('uae_fixed_point','registration'):
            shared = [q for q in query_ids if _valid(indexes['uae_nn'][q]) and _valid(indexes[method][q])]
            baseline = [float(indexes['uae_nn'][q]['cycle_error_mm']) for q in shared]
            other = [float(indexes[method][q]['cycle_error_mm']) for q in shared]
            for metric,fraction in [('median',.5),('p95',.95)]:
                eligible = len(shared) >= rules['minimum_valid_'+metric]
                left,right = _quantile(baseline,fraction),_quantile(other,fraction)
                paired.append(dict(subject_id=subject,group_name=group,method=method,
                    comparator='uae_nn',metric=metric,unit='mm',frozen_query_count=len(query_ids),
                    shared_valid_count=len(shared),shared_query_ids=';'.join(shared),
                    baseline_mm=left if eligible else '',method_mm=right if eligible else '',
                    difference_mm=right-left if eligible else '',
                    eligibility='eligible' if eligible else 'sparse_or_no_shared_valid',
                    interpretation='conditional_cycle_consistency'))
    for method,bundle in sorted(bundles.items()):
        cells = defaultdict(list)
        for row in bundle['rows']:
            cells[('group',row['subject_id'],row.get('group_name','unknown'))].append(row)
            cells[('organ',row['subject_id'],row['mask_name'])].append(row)
        for (level,subject,name), rows in sorted(cells.items()):
            values = [float(r['cycle_error_mm']) for r in rows if _valid(r)]
            attempted = sum(r['status'] != 'pending' for r in rows)
            failed = sum(r['status']=='failed' for r in rows)
            absolute.append(dict(method=method,level=level,subject_id=subject,name=name,unit='mm',
                eligible_queries=len(rows),attempted_queries=attempted,successful_queries=len(values),
                failed_queries=failed,pending_queries=len(rows)-attempted,
                failure_rate_all_attempted=failed/float(attempted) if attempted else '',
                median_mm=_quantile(values,.5) if len(values)>=rules['minimum_valid_median'] else '',
                p95_mm=_quantile(values,.95) if len(values)>=rules['minimum_valid_p95'] else '',
                sex_specific_population='male_only' if name=='prostate' else 'eligible_subjects',
                summary_population='method_specific_successes'))
        manifest = bundle['manifest']
        resources.append(dict(method=method,run_status=manifest['status'],
            query_count=manifest['query_count'],attempted_queries=manifest['attempted_queries'],
            wall_time_seconds=manifest.get('resources',{}).get('wall_time_seconds',manifest.get('elapsed_seconds','')),
            diagnostic_bytes=manifest.get('diagnostic_bytes',''),
            checkpoint_policy=manifest.get('settings',{}).get('checkpoint_queries','per_group_or_fixture'),
            resources_json=json.dumps(manifest.get('resources',{}),sort_keys=True),
            measurements='reported' if manifest.get('resources') or 'elapsed_seconds' in manifest else 'not_measured',
            evidence_bytes=sum(r['bytes'] for r in cohort.load_json(Path(bundle['directory'])/'output_inventory.json')['files']),
            extraction_context_validation=manifest.get('extraction_context_validation','pending_real_pilot'),
            anatomical_validation=manifest.get('anatomical_validation','pending')))
    output.mkdir(parents=True)
    cohort.atomic_csv(output/'paired_differences.csv',paired)
    cohort.atomic_csv(output/'absolute_errors_and_failures.csv',absolute)
    cohort.atomic_csv(output/'organ_detail.csv',[r for r in absolute if r['level']=='organ'])
    cohort.atomic_csv(output/'prostate_detail.csv',[r for r in absolute if r['name']=='prostate'],
        fieldnames=list(absolute[0]))
    cohort.atomic_csv(output/'resources.csv',resources)
    cohort.atomic_json(output/'analysis_rules.json',rules)
    _figure(output/'paired_differences.png',paired,rules,combined['contract'].get('fixture_only',False))
    manifest = dict(schema_version=1,input_signature=combined['input_signature'],
        rules=rules,fixture_only=combined['contract'].get('fixture_only',False),
        source_bundles=[dict(method=m,execution_commit=b['manifest']['execution_commit'],
            execution_signature=b['manifest']['execution_signature'],
            inventory_sha256=cohort.sha256_file(Path(b['directory'])/'output_inventory.json'),
            method_manifest_sha256=cohort.sha256_file(Path(b['directory'])/'method_manifest.json'),
            outcomes_sha256=b['manifest']['outcomes_sha256']) for m,b in sorted(bundles.items())],
        execution_commit=cohort.git_output(['rev-parse','HEAD']),
        source_tree_clean=not bool(cohort.git_output(['status','--porcelain'])),
        analysis_implementation_sha256=cohort.sha256_file(Path(__file__)),
        evidence_reader_sha256=cohort.sha256_file(Path(evidence.__file__)),
        interpretation='Cycle consistency, not anatomical accuracy; conditional errors accompany all-attempted failures.',
        cohort_rules_freeze='pending_review' if rules['status']=='provisional' else 'declared_frozen',
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=cohort.sha256_file(p))
            for p in sorted(output.iterdir()) if p.is_file()])
    cohort.atomic_json(output/'analysis_manifest.json',manifest)
    return 0


def _figure(path, rows, rules, fixture_only=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    groups = sorted({r['group_name'] for r in rows})
    methods = [('uae_fixed_point','Fixed-point − NN','#306998',-.12),
               ('registration','Registration − NN','#c56b20',.12)]
    fig,axes = plt.subplots(1,2,figsize=(max(9,len(groups)*1.8),4.8),sharey=False)
    for ax,metric in zip(axes,('median','p95')):
        for method,label,color,offset in methods:
            selected = [r for r in rows if r['method']==method and r['metric']==metric and r['eligibility']=='eligible']
            ax.scatter([groups.index(r['group_name'])+offset for r in selected],
                [r['difference_mm'] for r in selected],c=color,label=label,alpha=.65,s=25)
        ax.axhline(0,color='gray',lw=.8)
        ax.set_xlim(-.5,len(groups)-.5)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups,rotation=25,ha='right')
        ax.set_title(metric+' cycle error'); ax.set_ylabel('Paired difference (mm)')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(('ENGINEERING FIXTURE — ' if fixture_only else '')+
        'Subject-level differences on pairwise shared-valid queries',fontsize=12)
    fig.text(.5,.01,'Negative values indicate lower cycle error. Rules: '+rules['status']+'. Sparse cells are retained in the tables.',
        ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94)); fig.savefig(str(path),dpi=180); plt.close(fig)


def add_commands(subparsers):
    parser = subparsers.add_parser('reviewed-compare',help='One paired cycle figure and denominator tables')
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--run-directory',type=Path,action='append',required=True)
    parser.add_argument('--rules',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.set_defaults(reviewed_handler=compare)
