"""Auditable anatomical triage and CT inspection; no anatomical truth inference.

Commands are exposed through aligned_organ_group_cohort. Numerical triage settings
are explicitly supplied and remain provisional until the pilot review freezes them.
Python 3.7 compatible.
"""
from __future__ import print_function

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import uuid


def _cohort():
    from tools.quadra import aligned_organ_group_cohort
    return aligned_organ_group_cohort


def _data(args):
    from tools.quadra.reviewed_evidence import reconcile_bundles
    return reconcile_bundles(args.contract, args.run_directory)


def _partition(contract, subject):
    for row in contract.get('subjects', []):
        if isinstance(row, dict) and row['subject_id'] == subject:
            return row['review_partition']
    raise _cohort().CohortError('Missing frozen discovery/confirmation assignment')


def _finite(row):
    value = row.get('cycle_error_mm', '')
    if row.get('status') != 'success' or value in ('', None):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise _cohort().CohortError('Invalid successful cycle error')
    return number


def _flag(row, key):
    value = row.get(key)
    if value in ('', None):
        return None
    if value in (True, 'True', 'true', '1', 1):
        return True
    if value in (False, 'False', 'false', '0', 0):
        return False
    raise _cohort().CohortError('Invalid diagnostic flag: ' + key)


def _new_output(path):
    path = Path(path)
    if path.exists():
        raise _cohort().CohortError('Refusing to overwrite review evidence')
    path.mkdir(parents=True)
    return path


def queue(args):
    import numpy as np
    cohort = _cohort()
    data = _data(args)
    requested_subjects=set(args.subject or [])
    available_subjects={q['subject_id'] for q in data['queries']}
    if requested_subjects and not requested_subjects <= available_subjects:
        raise cohort.CohortError('Unknown or query-empty review subject scope')
    if requested_subjects:
        data['queries']=[q for q in data['queries'] if q['subject_id'] in requested_subjects]
    policy = cohort.load_json(args.policy)
    required = {'version', 'frozen_for_cohort', 'seed', 'disagreement_mm',
                'low_cycle_mm', 'random_per_organ', 'low_cycle_per_organ'}
    if set(policy) != required or not policy['version'] or not isinstance(policy['frozen_for_cohort'], bool):
        raise cohort.CohortError('Review policy must declare all versioned triage settings')
    for key in ('disagreement_mm', 'low_cycle_mm'):
        if not math.isfinite(float(policy[key])) or float(policy[key]) < 0:
            raise cohort.CohortError('Invalid review threshold: ' + key)
    for key in ('seed', 'random_per_organ', 'low_cycle_per_organ'):
        if isinstance(policy[key], bool) or not isinstance(policy[key], int) or policy[key] < 0:
            raise cohort.CohortError('Invalid review sampling setting: ' + key)
    queries = {q['query_id']: q for q in data['queries']}
    reasons = defaultdict(set)
    cells = defaultdict(list)
    unknown = Counter()
    indexed = {}
    cutoffs = []
    for method, bundle in sorted(data['bundles'].items()):
        indexed[method] = {r['query_id']: r for r in bundle['rows']}
        for row in bundle['rows']:
            qid = row['query_id']
            if qid not in queries:
                continue
            query = queries[qid]
            error = _finite(row)
            if error is not None:
                cells[(method, query['subject_id'], query['mask_name'])].append((qid, error))
            if row['status'] == 'failed':
                reasons[qid].add('technical_failure:' + method)
            for key, name in [('competing_peak', 'competing_peak'), ('boundary_anomaly', 'boundary')]:
                flagged = _flag(row, key)
                if flagged is True:
                    reasons[qid].add(name + ':' + method)
                elif flagged is None:
                    unknown[method + ':' + key] += 1
    for (method, subject, organ), values in sorted(cells.items()):
        cutoff = float(np.percentile([e for _, e in values], 95))
        selected = [qid for qid, error in values if error >= cutoff]
        for qid in selected:
            reasons[qid].add('upper_5_percent:' + method)
        cutoffs.append(dict(method=method, subject_id=subject, mask_name=organ,
                            valid_count=len(values), cutoff_mm=cutoff,
                            selected_count=len(selected), tie_policy='include_all_at_cutoff'))
    for left, right in combinations(sorted(indexed), 2):
        for qid in sorted(queries):
            a, b = _finite(indexed[left][qid]), _finite(indexed[right][qid])
            if a is not None and b is not None and abs(a-b) >= float(policy['disagreement_mm']):
                reasons[qid].add('cycle_disagreement:' + left + ':' + right)
            elif indexed[left][qid]['status'] != indexed[right][qid]['status']:
                reasons[qid].add('status_disagreement:' + left + ':' + right)
            locations=[]
            for method in (left,right):
                values=[indexed[method][qid].get('matched_lps_'+axis,'') for axis in 'xyz']
                if all(v not in ('',None) for v in values):
                    point=[float(v) for v in values]
                    if not all(math.isfinite(v) for v in point):
                        raise cohort.CohortError('Invalid match coordinate in review evidence')
                    locations.append(point)
            if len(locations)==2 and math.sqrt(sum((a-b)**2 for a,b in zip(*locations))) >= float(policy['disagreement_mm']):
                reasons[qid].add('correspondence_disagreement:' + left + ':' + right)
    # Controls are drawn before unioning other triggers. Overlap remains explicit.
    organs = defaultdict(list)
    for qid, query in sorted(queries.items()):
        organs[query['mask_name']].append(qid)
    controls=[]
    rng = np.random.RandomState(policy['seed'])
    for organ, ids in sorted(organs.items()):
        count = min(len(ids), policy['random_per_organ'])
        random_ids=[]
        if count:
            random_ids=[ids[int(index)] for index in rng.choice(len(ids),count,replace=False)]
            for qid in random_ids:
                reasons[qid].add('seeded_random_control')
        low = [qid for qid in ids if all(_finite(indexed[m][qid]) is not None and
            _finite(indexed[m][qid]) <= float(policy['low_cycle_mm']) for m in indexed)]
        count = min(len(low), policy['low_cycle_per_organ'])
        low_ids=[]
        if count:
            low_ids=[low[int(index)] for index in rng.choice(len(low),count,replace=False)]
            for qid in low_ids:
                reasons[qid].add('seeded_low_cycle_control')
        controls.append(dict(mask_name=organ,random_eligible=len(ids),low_cycle_eligible=len(low),
            random_query_ids=random_ids,low_cycle_query_ids=low_ids))
    selected = [dict(queries[qid], review_partition=_partition(data['contract'], queries[qid]['subject_id']),
                     inclusion_reasons=json.dumps(sorted(why))) for qid, why in sorted(reasons.items())]
    output = _new_output(args.output_directory)
    fields = list(dict.fromkeys(k for row in selected for k in row)) or ['query_id', 'inclusion_reasons']
    cohort.atomic_csv(output/'review_queue.csv', selected, fieldnames=fields, refuse=True)
    cohort.atomic_csv(output/'all_queries.csv', data['queries'], refuse=True)
    cohort.atomic_json(output/'tail_cutoffs.json', cutoffs, refuse=True)
    cohort.atomic_json(output/'control_sampling.json', controls, refuse=True)
    manifest = dict(schema_version=1, input_signature=data['input_signature'],
        query_count=len(queries), selected_count=len(selected), policy=policy,
        selected_subject_ids=sorted({q['subject_id'] for q in data['queries']}),
        review_scope='explicit_subject_subset' if requested_subjects else 'full_frozen_contract',
        queue_sha256=cohort.sha256_file(output/'review_queue.csv'),
        all_queries_sha256=cohort.sha256_file(output/'all_queries.csv'),
        contract_subjects=data['contract']['subjects'],
        diagnostic_unknown_counts=dict(unknown), anatomical_validation=False,
        quantile_policy='numpy_linear_95th_percentile_all_cutoff_ties',
        disagreement_policy='absolute_cycle_difference_or_forward_correspondence_lps_distance_or_status',
        control_policy='seeded_without_replacement_per_organ_across_subjects_allow_trigger_overlap',
        source_bundles={m: dict(outcomes_sha256=b['manifest']['outcomes_sha256'],
            method=m) for m, b in sorted(data['bundles'].items())})
    cohort.atomic_json(output/'review_manifest.json', manifest, refuse=True)
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _review(directory):
    cohort = _cohort()
    directory = Path(directory)
    manifest = cohort.load_json(directory/'review_manifest.json')
    for name, key in [('review_queue.csv', 'queue_sha256'), ('all_queries.csv', 'all_queries_sha256')]:
        if cohort.sha256_file(directory/name) != manifest[key]:
            raise cohort.CohortError('Review provenance changed: ' + name)
    queries = {r['query_id']: r for r in cohort.read_csv(directory/'all_queries.csv')}
    return directory, manifest, queries


def _records(directory, manifest, queries):
    cohort=_cohort()
    records=[]
    for path in sorted((directory/'judgments').glob('*.json')):
        value=cohort.load_json(path)
        digest=value.pop('payload_sha256',None)
        if (digest!=cohort.sha256_payload(value) or path.stem!=value.get('record_id')
                or value.get('input_signature')!=manifest['input_signature']
                or value.get('query_id') not in queries):
            raise cohort.CohortError('Anatomical judgment checksum or provenance changed')
        value['payload_sha256']=digest
        records.append(value)
    records.sort(key=lambda r:r['sequence_number'])
    for index,value in enumerate(records):
        previous=records[index-1] if index else None
        if (value['sequence_number']!=index+1 or value['previous_record_id']!=(previous['record_id'] if previous else None)
                or value['previous_payload_sha256']!=(previous['payload_sha256'] if previous else None)):
            raise cohort.CohortError('Anatomical judgment history is missing, reordered or branched')
    return records


def _categories(directory, version):
    cohort=_cohort()
    path=directory/'categories'/(cohort.sha256_payload(version)+'.json')
    if not path.exists():
        return None
    value=cohort.load_json(path)
    digest=value.pop('payload_sha256',None)
    if (digest!=cohort.sha256_payload(value) or value['category_version']!=version
            or value['input_signature']!=cohort.load_json(directory/'review_manifest.json')['input_signature']):
        raise cohort.CohortError('Frozen category snapshot changed')
    return value


def freeze_categories(args):
    cohort=_cohort()
    directory,manifest,queries=_review(args.review_directory)
    if not args.category_version or not args.reviewer_id or _categories(directory,args.category_version):
        raise cohort.CohortError('Category freeze requires a new version and reviewer identity')
    records=_records(directory,manifest,queries)
    definitions={r['category_id']:r['category_definition'] for r in records if
        r['review_partition']=='discovery' and r['category_version']==args.category_version}
    if not definitions:
        raise cohort.CohortError('No discovery category definitions to freeze')
    value=dict(category_version=args.category_version,definitions=definitions,
        reviewer_id=args.reviewer_id,input_signature=manifest['input_signature'],
        frozen_utc=datetime.now(timezone.utc).isoformat())
    value['payload_sha256']=cohort.sha256_payload(value)
    path=directory/'categories'
    path.mkdir(exist_ok=True)
    cohort.atomic_json(path/(cohort.sha256_payload(args.category_version)+'.json'),value,refuse=True)
    return 0


def _record(args):
    cohort = _cohort()
    directory, manifest, queries = _review(args.review_directory)
    incoming = cohort.load_json(args.record)
    required = {'query_id', 'reviewer_id', 'category_version', 'category_id', 'judgment',
                'anatomical_reason', 'uncertainty', 'selection_reason', 'category_definition',
                'categories_frozen', 'review_stage'}
    optional = {'supersedes_record_id', 'stopping_reason'}
    if set(incoming)-required-optional or required-set(incoming):
        raise cohort.CohortError('Incomplete or unexpected anatomical judgment fields')
    if incoming['query_id'] not in queries or any(not incoming[k] for k in required-{'categories_frozen'}):
        raise cohort.CohortError('Unknown query or missing judgment provenance')
    if incoming['judgment'] not in ('supported_contributor', 'suspected_contributor', 'unresolved', 'indeterminate'):
        raise cohort.CohortError('Unknown anatomical evidence judgment')
    partition = _partition({'subjects': manifest['contract_subjects']}, queries[incoming['query_id']]['subject_id'])
    if incoming['review_stage'] != partition or not isinstance(incoming['categories_frozen'], bool):
        raise cohort.CohortError('Review stage must preserve the frozen subject partition')
    records_dir = directory/'judgments'
    records_dir.mkdir(exist_ok=True)
    previous = _records(directory,manifest,queries)
    category_key = (incoming['category_version'], incoming['category_id'])
    same = [r for r in previous if (r['category_version'], r['category_id']) == category_key]
    if any(r['category_definition'] != incoming['category_definition'] for r in same):
        raise cohort.CohortError('Category definition changed without a new version')
    frozen=_categories(directory,incoming['category_version'])
    if frozen and frozen['definitions'].get(incoming['category_id'])!=incoming['category_definition']:
        raise cohort.CohortError('Category is missing or changed after version freeze')
    if partition == 'confirmation' and (not incoming['categories_frozen'] or frozen is None):
        raise cohort.CohortError('Confirmation requires an explicit frozen discovery category snapshot')
    if incoming['categories_frozen'] != bool(frozen):
        raise cohort.CohortError('Judgment freeze flag differs from the category snapshot')
    supersedes = incoming.get('supersedes_record_id')
    if supersedes and not any(r['record_id'] == supersedes and r['query_id'] == incoming['query_id']
                             and r['reviewer_id'] == incoming['reviewer_id'] for r in previous):
        raise cohort.CohortError('Unknown or unrelated superseded judgment')
    value = dict(incoming, record_id=uuid.uuid4().hex, review_partition=partition,
                 input_signature=manifest['input_signature'],
                 recorded_utc=datetime.now(timezone.utc).isoformat(), anatomical_ground_truth=False)
    value.update(sequence_number=len(previous)+1,previous_record_id=previous[-1]['record_id'] if previous else None,
        previous_payload_sha256=previous[-1]['payload_sha256'] if previous else None)
    value['payload_sha256']=cohort.sha256_payload(value)
    cohort.atomic_json(records_dir/(value['record_id']+'.json'), value, refuse=True)
    print(json.dumps(value, sort_keys=True))
    return 0


def record(args):
    # Exclusive writer marker prevents concurrent reviewers branching the history.
    directory=Path(args.review_directory)
    lock=directory/'.judgment-writing.lock'
    try:
        fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        raise _cohort().CohortError('Review writer already active; preserve any stale lock for inspection')
    try:
        os.close(fd)
        return _record(args)
    finally:
        lock.unlink()


def status(args):
    cohort = _cohort()
    directory, manifest, queries = _review(args.review_directory)
    judgments = _records(directory,manifest,queries)
    superseded = {r.get('supersedes_record_id') for r in judgments}
    current = [r for r in judgments if r['record_id'] not in superseded]
    coverage = []
    for organ in sorted({q['mask_name'] for q in queries.values()}):
        ids = {qid for qid, q in queries.items() if q['mask_name']==organ}
        entries = [r for r in current if r['query_id'] in ids]
        coverage.append(dict(mask_name=organ, total_queries=len(ids),
            reviewed_queries=len({r['query_id'] for r in entries}),
            reviewed_subjects=len({queries[r['query_id']]['subject_id'] for r in entries}),
            judgments=dict(Counter(r['judgment'] for r in entries)),
            partitions=dict(Counter(r['review_partition'] for r in entries)),
            category_versions=sorted({r['category_version'] for r in entries}),
            reviewer_ids=sorted({r['reviewer_id'] for r in entries}),
            stopping_reasons=sorted({r['stopping_reason'] for r in entries if r.get('stopping_reason')})))
    result = dict(input_signature=manifest['input_signature'], coverage=coverage,
                  anatomical_validation=False, exhaustive_failure_discovery=False,
                  indeterminate_queries=sorted({r['query_id'] for r in current if r['judgment']=='indeterminate'}))
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_file:
        cohort.atomic_json(args.output_file, result, refuse=True)
    return 0


def expand(args):
    import numpy as np
    cohort = _cohort()
    directory, manifest, queries = _review(args.review_directory)
    if not 1 <= args.batch_size <= 1000 or args.seed < 0:
        raise cohort.CohortError('Invalid adaptive review batch size or seed')
    records = _records(directory,manifest,queries)
    matches = [r for r in records if r['category_id']==args.category_id and r['category_version']==args.category_version]
    if not matches:
        raise cohort.CohortError('Adaptive expansion requires an existing versioned category')
    frozen=_categories(directory,args.category_version)
    if args.partition=='confirmation' and (frozen is None or args.category_id not in frozen['definitions']):
        raise cohort.CohortError('Confirmation expansion requires frozen discovery definitions')
    reviewed = {r['query_id'] for r in records}
    by_subject = defaultdict(list)
    subjects = {'subjects': manifest['contract_subjects']}
    for qid, q in sorted(queries.items()):
        if q['mask_name']==args.mask_name and qid not in reviewed and _partition(subjects,q['subject_id'])==args.partition:
            by_subject[q['subject_id']].append(qid)
    rng=np.random.RandomState(args.seed)
    subjects_order=sorted(by_subject)
    rng.shuffle(subjects_order)
    # New subjects first; subsequent round-robin batches cover each available subject.
    already_subjects={queries[r['query_id']]['subject_id'] for r in records if queries[r['query_id']]['mask_name']==args.mask_name}
    subjects_order.sort(key=lambda subject: subject in already_subjects)
    for ids in by_subject.values():
        rng.shuffle(ids)
    selected=[]
    while len(selected)<args.batch_size and any(by_subject.values()):
        for subject in subjects_order:
            if by_subject[subject] and len(selected)<args.batch_size:
                selected.append(by_subject[subject].pop())
    original={r['query_id']:json.loads(r['inclusion_reasons']) for r in cohort.read_csv(directory/'review_queue.csv')}
    rows=[dict(queries[qid],review_partition=args.partition,inclusion_reasons=json.dumps(sorted(set(
        original.get(qid,[])+['adaptive_category:'+args.category_version+':'+args.category_id])))) for qid in selected]
    output=_new_output(args.output_directory)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['query_id','inclusion_reasons']
    cohort.atomic_csv(output/'review_queue.csv',rows,fieldnames=fields,refuse=True)
    cohort.atomic_json(output/'expansion_manifest.json',dict(input_signature=manifest['input_signature'],
        category_id=args.category_id,category_version=args.category_version,partition=args.partition,
        mask_name=args.mask_name,seed=args.seed,requested_count=args.batch_size,selected_count=len(rows),
        selection_policy='unreviewed_queries_round_robin_subjects_prefer_unreviewed_subjects',
        parent_queue_sha256=manifest['queue_sha256'],anatomical_validation=False),refuse=True)
    return 0


def inspect(args):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import map_coordinates
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cohort = _cohort()
    data = _data(args)
    queries = {q['query_id']: q for q in data['queries']}
    if args.query_id not in queries:
        raise cohort.CohortError('Query identity is not in the frozen ledger')
    for v in (args.local_radius_mm, args.wide_radius_mm, args.pixel_mm):
        if not math.isfinite(v) or v <= 0:
            raise cohort.CohortError('Inspection radii and pixel spacing must be finite and positive')
    if (not math.isfinite(args.window_min_hu) or not math.isfinite(args.window_max_hu)
            or args.window_min_hu>=args.window_max_hu):
        raise cohort.CohortError('CT window must have finite increasing HU limits')
    if args.wide_radius_mm < args.local_radius_mm or args.wide_radius_mm/args.pixel_mm > 512:
        raise cohort.CohortError('Invalid or excessive inspection grid')
    if not 0 <= args.peak_limit <= 32:
        raise cohort.CohortError('Peak inspection limit must be between 0 and 32')
    query = queries[args.query_id]
    confirmation_evidence=None
    if _partition(data['contract'],query['subject_id'])=='confirmation':
        if not args.confirmation_review_directory:
            raise cohort.CohortError('Confirmation CT requires previously frozen discovery category snapshots')
        directory,review_manifest,review_queries=_review(args.confirmation_review_directory)
        if review_manifest['input_signature']!=data['input_signature']:
            raise cohort.CohortError('Confirmation review belongs to a different frozen experiment')
        records=_records(directory,review_manifest,review_queries)
        snapshots=[]
        for path in sorted((directory/'categories').glob('*.json')):
            snapshot=_categories(directory,cohort.load_json(path)['category_version'])
            prior={r['category_id']:r['category_definition'] for r in records if
                r['review_partition']=='discovery' and r['category_version']==snapshot['category_version']
                and r['recorded_utc']<=snapshot['frozen_utc']}
            if not prior or prior!=snapshot['definitions']:
                raise cohort.CohortError('Category snapshot does not reconcile prior discovery judgments')
            snapshots.append(dict(category_version=snapshot['category_version'],sha256=cohort.sha256_file(path)))
        if not snapshots:
            raise cohort.CohortError('No frozen discovery category snapshot for confirmation CT')
        confirmation_evidence=dict(input_signature=review_manifest['input_signature'],snapshots=snapshots)
    inventory = cohort.read_csv(Path(args.contract)/'portable_mask_inventory.csv')
    points = [('source', 'test', None, None)]
    for method, bundle in sorted(data['bundles'].items()):
        row = next(r for r in bundle['rows'] if r['query_id']==args.query_id)
        for label, session, prefix in [('matched', 'retest', 'matched_lps_'), ('returned', 'test', 'returned_lps_')]:
            values = [row.get(prefix+axis, '') for axis in 'xyz']
            if all(v not in ('', None) for v in values):
                position = np.asarray(values, dtype=float)
                if not np.isfinite(position).all():
                    raise cohort.CohortError('Non-finite inspection coordinate')
                points.append((method+':'+label, session, position, None))
        for direction, session in [('forward','retest'),('reverse','test')]:
            candidates = row.get(direction+'_peak_candidates', '')
            if not candidates:
                continue
            if isinstance(candidates,str):
                candidates=json.loads(candidates)
            if not isinstance(candidates,list):
                raise cohort.CohortError('Invalid retained peak candidate list')
            for candidate in candidates[:args.peak_limit]:
                position=np.asarray(candidate['physical_lps_xyz'],dtype=float)
                if position.shape!=(3,) or not np.isfinite(position).all():
                    raise cohort.CohortError('Invalid retained peak physical coordinate')
                points.append((method+':'+direction+':peak:'+str(candidate.get('rank','unknown')),
                               session,position,candidate))
    output = _new_output(args.output_directory)
    scans, evidence = {}, []
    for index, (label, session, position, candidate) in enumerate(points):
        if session not in scans:
            assets = [r for r in inventory if r['subject_id']==query['subject_id'] and r['session']==session]
            identities = {(r['ct_relative_path'], r['ct_sha256']) for r in assets}
            if len(identities)!=1:
                raise cohort.CohortError('Missing or inconsistent frozen CT identity')
            relative, expected = next(iter(identities))
            ctroot = Path(args.ct_root).resolve()
            path = ctroot/relative
            if ctroot not in path.resolve().parents or path.is_symlink() or cohort.sha256_file(path)!=expected:
                raise cohort.CohortError('CT path or checksum differs from the frozen inventory')
            image = nib.load(str(path))
            affine = np.diag([-1., -1., 1., 1.]).dot(image.affine)
            if (len(image.shape)!=3 or image.header.get_xyzt_units()[0]!='mm'
                    or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<1e-10):
                raise cohort.CohortError('Invalid CT geometry for physical inspection')
            scans[session] = (np.asarray(image.dataobj,dtype=np.float32), affine, expected)
        voxels, affine, expected = scans[session]
        if position is None:
            position = affine.dot([float(query['raw_'+a]) for a in 'xyz']+[1])[:3]
            fields=['physical_lps_'+axis for axis in 'xyz']
            if any(field in query for field in fields):
                if not all(field in query for field in fields) or not np.allclose(
                        position,[float(query[field]) for field in fields],atol=1e-5,rtol=0):
                    raise cohort.CohortError('Native CT coordinates disagree with frozen physical query')
        inverse = np.linalg.inv(affine)
        center_voxel = inverse.dot(list(position)+[1])[:3]
        figure, axes = plt.subplots(2, 3, figsize=(12, 8))
        arrays = {}
        for rowindex, radius in enumerate((args.local_radius_mm, args.wide_radius_mm)):
            ticks = np.linspace(-radius, radius, int(math.ceil(2*radius/args.pixel_mm))+1)
            arrays[('local_' if rowindex==0 else 'wide_')+'offsets_mm']=ticks
            u, v = np.meshgrid(ticks, ticks)
            for column, (plane, dims) in enumerate([('axial', (0,1)), ('coronal', (0,2)), ('sagittal', (1,2))]):
                physical = np.repeat(position[:,None], u.size, axis=1)
                physical[dims[0]] += u.ravel(); physical[dims[1]] += v.ravel()
                coordinates = inverse[:3,:3].dot(physical)+inverse[:3,3,None]
                values = map_coordinates(voxels, coordinates, order=1,
                    mode='constant', cval=-1024, prefilter=False).reshape(u.shape)
                name = ('local_' if rowindex==0 else 'wide_')+plane
                arrays[name]=values
                ax = axes[rowindex,column]
                ax.imshow(values, cmap='gray', vmin=args.window_min_hu, vmax=args.window_max_hu, origin='lower',
                          extent=[-radius,radius,-radius,radius])
                ax.axhline(0, color='cyan', linewidth=.5); ax.axvline(0, color='cyan', linewidth=.5)
                ax.set_title(name+'; physical LPS mm'); ax.set_xlabel('offset '+ 'xyz'[dims[0]]+' (mm)')
                ax.set_ylabel('offset '+'xyz'[dims[1]]+' (mm)')
        name = 'point_{:02d}'.format(index)
        np.savez_compressed(str(output/(name+'.npz')), **arrays)
        figure.suptitle(args.query_id+' '+label+' '+session+'; CT window [{:g},{:g}] HU'.format(args.window_min_hu,args.window_max_hu))
        figure.tight_layout(); figure.savefig(str(output/(name+'.png')), dpi=100); plt.close(figure)
        evidence.append(dict(label=label, session=session, physical_lps_mm=position.tolist(),
            native_voxel_xyz=center_voxel.tolist(),
            center_inside_ct=bool(np.all(center_voxel>=0) and np.all(center_voxel<=np.asarray(voxels.shape)-1)),
            ct_sha256=expected, numeric_file=name+'.npz', view_file=name+'.png', peak_candidate=candidate))
    cohort.atomic_json(output/'inspection_manifest.json', dict(query=query, input_signature=data['input_signature'],
        query_id=args.query_id, review_partition=_partition(data['contract'], query['subject_id']),
        coordinate_frame='LPS_mm', local_radius_mm=args.local_radius_mm,
        wide_radius_mm=args.wide_radius_mm, requested_pixel_mm=args.pixel_mm,
        ct_window_hu=[args.window_min_hu,args.window_max_hu], interpolation='linear', outside_ct_hu=-1024,
        anatomical_validation=False, confirmation_review=confirmation_evidence, points=evidence,files=[dict(path=p.name,bytes=p.stat().st_size,
            sha256=cohort.sha256_file(p)) for p in sorted(output.iterdir()) if p.is_file()]), refuse=True)
    return 0


def add_commands(subparsers):
    p = subparsers.add_parser('reviewed-review-queue', help='Build an auditable anatomical triage queue')
    for option in ('contract', 'policy', 'output-directory'):
        p.add_argument('--'+option, type=Path, required=True)
    p.add_argument('--run-directory', type=Path, action='append', required=True)
    p.add_argument('--subject', action='append', help='Limit review and controls to a frozen subject; repeat for a pilot')
    p.set_defaults(reviewed_handler=queue)
    p = subparsers.add_parser('reviewed-review-expand', help='Expand review across subjects for a versioned category')
    p.add_argument('--review-directory', type=Path, required=True)
    p.add_argument('--mask-name', required=True)
    p.add_argument('--category-id', required=True)
    p.add_argument('--category-version', required=True)
    p.add_argument('--partition', choices=('discovery','confirmation'), default='discovery')
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output-directory', type=Path, required=True)
    p.set_defaults(reviewed_handler=expand)
    p = subparsers.add_parser('reviewed-inspect', help='Inspect any frozen query in CT physical planes')
    for option in ('contract', 'ct-root', 'output-directory'):
        p.add_argument('--'+option, type=Path, required=True)
    p.add_argument('--run-directory', type=Path, action='append', required=True)
    p.add_argument('--query-id', required=True)
    p.add_argument('--local-radius-mm', type=float, default=20.)
    p.add_argument('--wide-radius-mm', type=float, default=80.)
    p.add_argument('--pixel-mm', type=float, default=1.)
    p.add_argument('--window-min-hu', type=float, default=-160.)
    p.add_argument('--window-max-hu', type=float, default=240.)
    p.add_argument('--peak-limit', type=int, default=3)
    p.add_argument('--confirmation-review-directory',type=Path)
    p.set_defaults(reviewed_handler=inspect)
    p = subparsers.add_parser('reviewed-review-freeze-categories', help='Freeze discovery definitions before confirmation')
    p.add_argument('--review-directory', type=Path, required=True)
    p.add_argument('--category-version', required=True)
    p.add_argument('--reviewer-id', required=True)
    p.set_defaults(reviewed_handler=freeze_categories)
    p = subparsers.add_parser('reviewed-review-record', help='Append a versioned anatomical judgment')
    p.add_argument('--review-directory', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p.set_defaults(reviewed_handler=record)
    p = subparsers.add_parser('reviewed-review-status', help='Report review coverage without claiming validation')
    p.add_argument('--review-directory', type=Path, required=True)
    p.add_argument('--output-file', type=Path)
    p.set_defaults(reviewed_handler=status)
