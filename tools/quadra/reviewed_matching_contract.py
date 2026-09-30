"""Portable reviewed-mask contracts shared by independently prepared methods.

Preparation never executes matching, registration, or a scientific pilot.
The legacy cohort contracts remain unchanged. Python 3.7 syntax is retained.
"""
from __future__ import print_function

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

METHODS = ('uae_nn', 'uae_fixed_point', 'registration')


def freeze_reviewed(args):
    """Freeze reviewed inputs without reading any matching outcomes."""
    import nibabel as nib
    import numpy as np
    from tools.quadra import aligned_organ_group_cohort as cohort
    from tools.quadra import organ_group_lattice_alignment as lattice
    from tools.quadra import registration_organ_group as registration

    dataset = Path(args.dataset_root).resolve()
    ct_root = Path(args.ct_root).resolve()
    target = Path(args.output_directory).resolve()
    partial = target.with_name('.' + target.name + '.partial')
    if target.exists() or partial.exists():
        raise cohort.CohortError('Refusing to overwrite a complete or partial freeze')
    source_manifest = cohort.load_json(dataset / 'final_dataset_manifest.json')
    counts = source_manifest.get('counts', {})
    if (source_manifest.get('dataset_kind') != 'quadra_final_reviewed_masks_for_matching'
            or any(counts.get(key) != value for key, value in dict(subjects=48, scans=96, masks=3790).items())
            or source_manifest.get('validation', {}).get('all_decisions_acceptable') is not True
            or source_manifest.get('validation', {}).get('copy_checksums_verified') is not True):
        raise cohort.CohortError('The final reviewed dataset has not passed its input contract')
    inventory_path = dataset / 'final_mask_inventory.csv'
    rows = cohort.read_csv(inventory_path)
    registry = cohort.registry_records(args.registry)
    config_path = cohort.PROJECT_ROOT/'configs/samv2/samv2_NIHLN.py'
    if cohort.sha256_file(config_path) != cohort.stage3.EXPECTED_CONFIG_SHA256:
        raise cohort.CohortError('UAE model configuration differs from the accepted reference')
    by_scan = defaultdict(dict)  # type: Dict[Tuple[str, str], Dict[str, Any]]
    sexes = {}  # type: Dict[str, str]
    for row in rows:
        subject, session, sex, organ = (row[k] for k in ('subject_id', 'session', 'sex', 'organ'))
        if subject not in ['quadra_hc_{:03d}'.format(n) for n in range(1, 49)] or session not in ('test', 'retest') or sex not in ('F', 'M'):
            raise cohort.CohortError('Invalid subject, session or sex in reviewed inventory')
        if subject in sexes and sexes[subject] != sex:
            raise cohort.CohortError('Inconsistent subject sex')
        sexes[subject] = sex
        if organ in by_scan[(subject, session)]:
            raise cohort.CohortError('Duplicate reviewed mask inventory row')
        by_scan[(subject, session)][organ] = row
    if (len(rows), len(by_scan), len(sexes), Counter(sexes.values())) != (3790, 96, 48, Counter(F=25, M=23)):
        raise cohort.CohortError('Reviewed cohort or sex denominators differ from the approved inventory')
    for (subject, session), scan in by_scan.items():
        expected = {r['filename'] for r in registry if r['filename'] != 'prostate' or sexes[subject] == 'M'}
        if set(scan) != expected:
            raise cohort.CohortError('Missing, extra or sex-ineligible reviewed masks')
    partial.mkdir(parents=True)
    parameters_path = Path(args.registration_parameters)
    parameters = json.loads(parameters_path.read_text(encoding='utf-8'))
    approved_parameters = json.loads((cohort.PROJECT_ROOT/'configs/quadra/reviewed-registration-parameters.json').read_text(encoding='utf-8'))
    if parameters != approved_parameters:
        raise cohort.CohortError('Registration parameter maps differ from the recovered approved settings')
    if len(parameters) != 2 or [p.get('Transform') for p in parameters] != [['EulerTransform'], ['BSplineTransform']]:
        raise cohort.CohortError('Registration requires the recovered rigid and B-spline parameter maps')
    for parameter in parameters:
        for key, value in {'NumberOfResolutions':['4'], 'MaximumNumberOfIterations':['256'],
                           'NumberOfSpatialSamples':['8192'], 'RandomSeed':['121212'],
                           'NewSamplesEveryIteration':['true'], 'DefaultPixelValue':['-1024']}.items():
            if parameter.get(key) != value:
                raise cohort.CohortError('Registration parameter differs from the approved contract: '+key)
    if parameters[1].get('FinalGridSpacingInPhysicalUnits') != ['32']:
        raise cohort.CohortError('B-spline spacing differs from the approved contract')
    cohort.atomic_json(partial/'registration_parameters.json', parameters, refuse=True)
    queries, sampling, assets, plans, native_names = [], [], [], [], []
    revised = set()
    for (subject, session), scan in sorted(by_scan.items()):
        ct_relative = 'QUADRA_HC_{}/{}_CT-AC.nii.gz'.format(subject[-3:], session)
        ct_path = ct_root / ct_relative
        ct_identity = cohort.file_identity(ct_path)
        if {r['ct_sha256'] for r in scan.values()} != {ct_identity['sha256']}:
            raise cohort.CohortError('CT checksum differs from the reviewed inventory')
        image = nib.load(str(ct_path))
        affine = np.asarray(image.affine, dtype=np.float64)
        if len(image.shape) != 3 or image.header.get_xyzt_units()[0] != 'mm' or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
            raise cohort.CohortError('Invalid CT physical geometry or units')
        source = dict(path=ct_relative, bytes=ct_identity['bytes'], sha256=ct_identity['sha256'],
                      native_shape_xyz=list(image.shape), spacing_xyz_mm=np.linalg.norm(affine[:3, :3], axis=0).tolist(),
                      affine=affine.tolist(), spatial_unit='mm')
        group_bounds = defaultdict(list)
        scan_queries = []
        for item in registry:
            organ = item['filename']
            if organ not in scan: continue
            row = scan[organ]
            relative = '{}/{}/masks/{}.nii.gz'.format(subject, session, organ)
            path = dataset / relative
            identity = cohort.file_identity(path)
            if identity['sha256'] != row['dataset_mask_sha256']:
                raise cohort.CohortError('Reviewed mask checksum changed: ' + relative)
            mask = nib.load(str(path))
            if mask.shape != image.shape or not np.allclose(mask.affine, affine, atol=1e-5, rtol=0):
                raise cohort.CohortError('Reviewed mask/CT geometry mismatch: ' + relative)
            data = np.asanyarray(mask.dataobj)
            if np.issubdtype(data.dtype, np.integer):
                binary = data.min() >= 0 and data.max() == 1
            else:
                binary = np.isfinite(data).all() and np.all((data == 0) | (data == 1)) and data.any()
            if not binary:
                raise cohort.CohortError('Reviewed mask is not finite, binary and non-empty: ' + relative)
            occupied = [np.flatnonzero(data.any(axis=tuple(a for a in range(3) if a != axis))) for axis in range(3)]
            start = [int(v[0]) for v in occupied]
            stop = [int(v[-1]) + 1 for v in occupied]
            group_bounds[item['group_name']].append((start, stop, organ))
            assets.append(dict(subject_id=subject, session=session, sex=sexes[subject], organ=organ,
                               mask_relative_path=relative, mask_sha256=identity['sha256'], mask_bytes=identity['bytes'],
                               ct_relative_path=ct_relative, ct_sha256=ct_identity['sha256'], ct_bytes=ct_identity['bytes'],
                               selected_source=row['selected_source'], acceptance_stage=row['acceptance_stage']))
            if row['selected_source'] != 'original': revised.add(subject)
            if session == 'test':
                candidates = np.flatnonzero(data)
                available = len(candidates)
                count = min(100, available)
                seed = 20260721 + item['registry_index']
                selected = np.random.default_rng(seed).choice(available, size=count, replace=False)
                points = np.stack(np.unravel_index(candidates[selected], data.shape), axis=1)
                sampling.append(dict(subject_id=subject, sex=sexes[subject], mask_name=organ,
                                     group_name=item['group_name'], mask_registry_index=item['registry_index'],
                                     available_unique_voxels=available, sampled_unique_points=count,
                                     requested_points=100, shortfall=100-count, sampling_seed=seed))
                for index, point in enumerate(points):
                    physical = lattice.apply_affine(point, affine)
                    scan_queries.append(dict(query_id='{}:{}:{:03d}'.format(subject, organ, index),
                        subject_id=subject, sex=sexes[subject], group_name=item['group_name'], mask_name=organ,
                        mask_registry_index=item['registry_index'], point_index=index,
                        raw_x=int(point[0]), raw_y=int(point[1]), raw_z=int(point[2]),
                        physical_lps_x=float(-physical[0]), physical_lps_y=float(-physical[1]), physical_lps_z=float(physical[2]),
                        sampling_seed=seed, available_unique_voxels=available, sampled_points_for_mask=count,
                        mask_query_shortfall=100-count, sampling_policy='all_available' if count < 100 else 'random_without_replacement',
                        mask_sha256=identity['sha256'], ct_sha256=ct_identity['sha256']))
            del data
        scan_plans = {}
        for group, bounds in sorted(group_bounds.items()):
            plan = lattice.aligned_plan_from_union(dict(subject_id=subject, session=session, sex=sexes[subject],
                scan_key=subject+'_'+session, group_name=group, margin_mm=100, source_ct=source,
                mask_union_start_xyz=np.min([b[0] for b in bounds], axis=0).tolist(),
                mask_union_end_xyz=np.max([b[1] for b in bounds], axis=0).tolist(), included_masks=[b[2] for b in bounds]), 100)
            name = 'plans/{}-{}-{}.json'.format(subject, session, group)
            cohort.atomic_json(partial / name, plan, refuse=True)
            plans.append((name, plan)); scan_plans[group] = plan
            native_name = 'registration_plans/{}-{}-{}.json'.format(subject, session, group)
            cohort.atomic_json(partial / native_name, registration.serializable_plan(plan), refuse=True)
            native_names.append(native_name)
        for row in scan_queries:
            model = lattice.apply_affine([row['raw_x'], row['raw_y'], row['raw_z']], scan_plans[row['group_name']]['raw_to_model_continuous_affine'])
            fine = np.floor(np.rint(model) / 2).astype(np.int64)
            for axis, value, index in zip('xyz', model, fine):
                row['model_continuous_'+axis] = float(value); row['fine_'+axis] = int(index)
            row['fine_quantization_policy'] = 'rint_model_then_floor_half'
        queries.extend(scan_queries)
        print('Verified reviewed inputs: {} {}'.format(subject, session), flush=True)
    ranked = sorted(plans, key=lambda p: (-p[1]['padded_2mm_voxels'], p[1]['subject_id'], p[0]))
    pilots = [ranked[0][1]['subject_id']]
    opposite = next(p[1]['subject_id'] for p in ranked if sexes[p[1]['subject_id']] != sexes[pilots[0]])
    if opposite not in pilots: pilots.append(opposite)
    if not revised:
        raise cohort.CohortError('Pilot coverage requires a corrected or resegmented mask subject')
    if not set(pilots) & revised:
        pilots.append(next(p[1]['subject_id'] for p in ranked if p[1]['subject_id'] in revised))
    strata = defaultdict(list)
    for subject, sex in sorted(sexes.items()):
        strata[(sex, int(subject[-3:]) >= 21)].append(subject)
    quotas = {key: len(values)*16//48 for key, values in strata.items()}
    remaining = 16-sum(quotas.values())
    for stratum in sorted(strata, key=lambda key: (-(len(strata[key])*16 % 48), key))[:remaining]:
        quotas[stratum] += 1
    confirmation = set()
    for stratum, values in sorted(strata.items()):
        eligible = sorted((s for s in values if s not in pilots), key=lambda s: cohort.sha256_payload([20260721, 'confirmation', s]))
        if len(eligible) < quotas[stratum]: raise cohort.CohortError('Pilot assignment prevents stratified confirmation allocation')
        confirmation.update(eligible[:quotas[stratum]])
    cohort.atomic_csv(partial / 'frozen_queries_raw_itk.csv', queries, refuse=True)
    cohort.atomic_csv(partial / 'mask_query_sampling_summary.csv', sampling, refuse=True)
    cohort.atomic_csv(partial / 'portable_mask_inventory.csv', assets, refuse=True)
    manifest = dict(schema_version=2, dataset_id=dataset.name, fixture_only=False,
        source_inventory_sha256=cohort.sha256_file(inventory_path),
        source_dataset_manifest_sha256=cohort.sha256_file(dataset/'final_dataset_manifest.json'),
        registry_sha256=cohort.sha256_file(args.registry), execution_commit=cohort.git_output(['rev-parse', 'HEAD']),
        generator_implementation_signature=implementation_signature(),
        source_tree_clean=not bool(cohort.git_output(['status', '--porcelain'])),
        model=dict(config_sha256=cohort.sha256_file(cohort.PROJECT_ROOT/'configs/samv2/samv2_NIHLN.py'),
                   expected_checkpoint_sha256=cohort.stage3.EXPECTED_CHECKPOINT_SHA256,
                   checkpoint_validation='pending_remote_asset_verification'),
        query_count=len(queries), counts=dict(subjects=48, scans=96, masks=3790, female_subjects=25, male_subjects=23),
        subjects=[dict(subject_id=s, sex=sexes[s], previous_cohort_member=int(s[-3:]) >= 21,
                       review_partition='confirmation' if s in confirmation else 'discovery') for s in sorted(sexes)],
        pilot_subjects=pilots, largest_plan=ranked[0][0], plan_files=[p[0] for p in plans],
        registration_plan_files=native_names,
        settings=dict(seed=20260721, points_per_mask=100, margin_mm=100, spacing_xyz_mm=[2,2,2], stride_xyz=[16,16,4],
                      model_precision='fp32', embedding_dtype='fp16', similarity_dtype='fp32',
                      registration_threads=1,
                      query_coordinate_frame='raw_itk_voxel', physical_coordinate_frame='LPS_mm',
                      fine_quantization_policy='rint_model_then_floor_half', retrieval_domain='shared_fine_grid',
                      pilot_status='not_started', matching_status='not_started'),
        sampling_shortfall=sum(s['shortfall'] for s in sampling),
        maximum_queries=189500,
        source_selection_counts=dict(Counter(a['selected_source'] for a in assets)),
        scientific_work_launched=False, files=[])
    for path in sorted(p for p in partial.rglob('*') if p.is_file()):
        identity = cohort.file_identity(path)
        manifest['files'].append(dict(path=path.relative_to(partial).as_posix(), bytes=identity['bytes'], sha256=identity['sha256']))
    cohort.atomic_json(partial / 'matching_contract.json', manifest, refuse=True)
    read_contract(partial)
    os.rename(str(partial), str(target))
    print('Frozen reviewed contract: {} queries; no pilot started'.format(len(queries)), flush=True)
    return 0


def read_contract(root):
    from tools.quadra import aligned_organ_group_cohort as cohort
    root = Path(root).resolve()
    manifest = cohort.load_json(root / 'matching_contract.json')
    if manifest.get('schema_version') != 2:
        raise cohort.CohortError('Unsupported reviewed matching contract')
    seen = set()
    for record in manifest.get('files', []):
        relative = Path(record['path'])
        path = root / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink():
            raise cohort.CohortError('Unsafe contract member')
        if root not in path.resolve().parents or record['path'] in seen:
            raise cohort.CohortError('Unsafe or duplicate contract member')
        seen.add(record['path'])
        observed = cohort.file_identity(path)
        if (observed['sha256'], observed['bytes']) != (record['sha256'], record['bytes']):
            raise cohort.CohortError('Contract member identity changed: ' + record['path'])
    if 'frozen_queries_raw_itk.csv' not in seen:
        raise cohort.CohortError('Contract has no verified query ledger')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != seen | {'matching_contract.json'}:
        raise cohort.CohortError('Unlisted or missing contract files')
    rows = cohort.read_csv(root / 'frozen_queries_raw_itk.csv')
    ids = [row['query_id'] for row in rows]
    if len(rows) != manifest['query_count'] or len(set(ids)) != len(ids) or not rows:
        raise cohort.CohortError('Query denominator or query identity mismatch')
    if manifest.get('fixture_only') is not True:
        validate_reviewed_contract(root, manifest, rows, seen)
    return manifest, rows, cohort.sha256_payload(manifest)


def validate_reviewed_contract(root, manifest, queries, members):
    """Reconcile declared identities, not just the hashes of CSV containers."""
    from tools.quadra import aligned_organ_group_cohort as cohort
    expected = dict(subjects=48, scans=96, masks=3790, female_subjects=25, male_subjects=23)
    if manifest.get('counts') != expected:
        raise cohort.CohortError('Reviewed contract cohort denominators changed')
    registry = cohort.PROJECT_ROOT/'tools/quadra/totalsegmentator/organs.yaml'
    if cohort.sha256_file(registry) != manifest['registry_sha256']:
        raise cohort.CohortError('Reviewed organ registry changed')
    subjects = {s['subject_id']: s for s in manifest['subjects']}
    if (len(manifest['subjects']) != 48 or set(subjects) != {'quadra_hc_{:03d}'.format(n) for n in range(1,49)}
            or Counter(s['sex'] for s in subjects.values()) != Counter(F=25, M=23)
            or Counter(s['review_partition'] for s in subjects.values()) != Counter(discovery=32, confirmation=16)):
        raise cohort.CohortError('Reviewed subject eligibility or partition changed')
    pilots = manifest['pilot_subjects']
    if not 1 <= len(pilots) <= 3 or len(set(pilots)) != len(pilots) or any(subjects[p]['review_partition'] != 'discovery' for p in pilots):
        raise cohort.CohortError('Invalid pilot assignment')
    for plan_key, prefix in [('plan_files','plans/'), ('registration_plan_files','registration_plans/')]:
        names = manifest[plan_key]
        if len(names) != 384 or len(set(names)) != 384 or any(n not in members or not n.startswith(prefix) for n in names):
            raise cohort.CohortError('Incomplete crop plan contract')
    for required in ('portable_mask_inventory.csv', 'mask_query_sampling_summary.csv', 'registration_parameters.json'):
        if required not in members: raise cohort.CohortError('Missing reviewed contract component: '+required)
    inventory = cohort.read_csv(root/'portable_mask_inventory.csv')
    assets = {}  # type: Dict[Tuple[str, str, str], Any]
    scans = {}  # type: Dict[Tuple[str, str], Tuple[str, str, str]]
    organs = {r['filename'] for r in cohort.registry_records(cohort.PROJECT_ROOT/'tools/quadra/totalsegmentator/organs.yaml')}
    for row in inventory:
        subject, session, organ = (row[k] for k in ('subject_id', 'session', 'organ'))
        key = (subject, session, organ)
        if (key in assets or subject not in subjects or session not in ('test','retest')
                or row['sex'] != subjects[subject]['sex'] or organ not in organs
                or (organ == 'prostate' and row['sex'] != 'M')
                or row['mask_relative_path'] != '{}/{}/masks/{}.nii.gz'.format(subject,session,organ)
                or row['ct_relative_path'] != 'QUADRA_HC_{}/{}_CT-AC.nii.gz'.format(subject[-3:],session)):
            raise cohort.CohortError('Invalid reviewed inventory identity or eligibility')
        identity = (row['ct_relative_path'], row['ct_sha256'], row['ct_bytes'])
        if (subject,session) in scans and scans[(subject,session)] != identity:
            raise cohort.CohortError('Inconsistent CT identity within a scan')
        scans[(subject,session)] = identity
        assets[key] = row
    if len(assets) != 3790 or len(scans) != 96:
        raise cohort.CohortError('Reviewed inventory denominators changed')
    if dict(Counter(r['selected_source'] for r in inventory)) != manifest['source_selection_counts']:
        raise cohort.CohortError('Reviewed source selection changed')
    grouped = Counter()  # type: Counter[Tuple[str, str]]
    for row in queries:
        key = (row['subject_id'], 'test', row['mask_name'])
        source = assets.get(key)
        if source is None or (row['mask_sha256'],row['ct_sha256']) != (source['mask_sha256'],source['ct_sha256']):
            raise cohort.CohortError('Query provenance differs from the reviewed inventory')
        grouped[(row['subject_id'],row['mask_name'])] += 1
    sampling = cohort.read_csv(root/'mask_query_sampling_summary.csv')
    supplied = {}
    for row in sampling:
        sampling_key = (row['subject_id'],row['mask_name'])
        count, available = int(row['sampled_unique_points']), int(row['available_unique_voxels'])
        if sampling_key in supplied or count != min(100,available) or int(row['shortfall']) != 100-count:
            raise cohort.CohortError('Invalid reviewed sampling denominator')
        supplied[sampling_key] = count
    if dict(grouped) != supplied or len(supplied) != 1895 or manifest['sampling_shortfall'] != 189500-len(queries):
        raise cohort.CohortError('Frozen queries do not reconcile sampling shortfalls')


def prepare_method(args):
    from tools.quadra import aligned_organ_group_cohort as cohort
    contract, queries, signature = read_contract(args.contract)
    run = Path(args.run_directory).resolve()
    manifest_path = run / 'method_manifest.json'
    identity = {
        'input_signature': signature, 'method': args.method,
        'execution_commit': cohort.git_output(['rev-parse', 'HEAD']),
        'implementation_signature': implementation_signature(),
    }
    if run.exists():
        if not args.resume or not manifest_path.is_file():
            raise cohort.CohortError('Existing run requires compatible --resume')
        previous = cohort.load_json(manifest_path)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise cohort.CohortError('Incompatible method, code or input contract')
        observed = cohort.file_identity(run / 'query_outcomes.csv')
        if observed['sha256'] != previous['outcomes_sha256']:
            raise cohort.CohortError('Prepared outcome ledger changed')
        print(json.dumps(previous, indent=2, sort_keys=True))
        return 0
    run.mkdir(parents=True)
    outcomes = [dict(row, method=args.method, status='pending',
                     forward_status='pending', reverse_status='pending') for row in queries]
    cohort.atomic_csv(run / 'query_outcomes.csv', outcomes, refuse=True)
    manifest = dict(identity, schema_version=2, dataset_id=contract['dataset_id'],
                    fixture_only=contract.get('fixture_only', False),
                    status='awaiting_pilot', scientific_work_launched=False,
                    query_count=len(queries), attempted_queries=0,
                    outcomes_sha256=cohort.sha256_file(run / 'query_outcomes.csv'),
                    pending_gates=['method_implementation', 'resource_pilot', 'anatomical_review'])
    cohort.atomic_json(manifest_path, manifest, refuse=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def implementation_signature():
    from tools.quadra import aligned_organ_group_cohort as cohort
    return cohort.sha256_payload({name: cohort.sha256_file(cohort.PROJECT_ROOT/'tools/quadra'/name)
        for name in ('aligned_organ_group_cohort.py', 'reviewed_matching_contract.py')})


def fixture_method(args):
    """Exercise the outcome adapter with explicit fixtures, never live images."""
    from tools.quadra import aligned_organ_group_cohort as cohort
    contract, queries, signature = read_contract(args.contract)
    if contract.get('fixture_only') is not True:
        raise cohort.CohortError('Fixture execution is forbidden for a real-data contract')
    run = Path(args.run_directory)
    manifest = cohort.load_json(run/'method_manifest.json')
    if manifest['input_signature'] != signature or manifest['implementation_signature'] != implementation_signature():
        raise cohort.CohortError('Incompatible fixture input or implementation')
    fixture_signature = cohort.sha256_file(args.outcomes)
    if manifest['status'] == 'fixture_complete':
        if not args.resume or manifest['fixture_signature'] != fixture_signature:
            raise cohort.CohortError('Fixture replay requires compatible resume')
        if cohort.sha256_file(run/'query_outcomes.csv') != manifest['outcomes_sha256']:
            raise cohort.CohortError('Fixture outcome ledger changed')
        return 0
    fixtures = json.loads(Path(args.outcomes).read_text(encoding='utf-8'))
    records = {r['query_id']: r for r in fixtures}
    if len(records) != len(fixtures) or set(records) != {r['query_id'] for r in queries}:
        raise cohort.CohortError('Fixture outcomes must reconcile every frozen query exactly once')
    outcomes = []
    for query in queries:
        record = records[query['query_id']]
        if record['status'] not in ('success', 'failed'):
            raise cohort.CohortError('Unknown fixture terminal outcome')
        if record['status'] == 'success':
            error = float(record['cycle_error_mm'])
            if not math.isfinite(error) or error < 0 or record['forward_status'] != 'success' or record['reverse_status'] != 'success':
                raise cohort.CohortError('Successful fixture cycle has invalid directional status or error')
        elif not record.get('failure_reason'):
            raise cohort.CohortError('Failed fixture must preserve its failure reason')
        allowed = {'query_id','status','forward_status','reverse_status','cycle_error_mm','failure_reason'}
        if set(record) - allowed:
            raise cohort.CohortError('Fixture outcomes cannot override query provenance')
        outcomes.append(dict(query, **record, method=manifest['method']))
    fields = list(dict.fromkeys(key for row in outcomes for key in row))
    cohort.atomic_csv(run/'query_outcomes.csv', outcomes, fieldnames=fields)
    manifest.update(status='fixture_complete', attempted_queries=len(outcomes),
        failed_queries=sum(r['status']=='failed' for r in outcomes),
        successful_queries=sum(r['status']=='success' for r in outcomes), fixture_signature=fixture_signature,
        outcomes_sha256=cohort.sha256_file(run/'query_outcomes.csv'), scientific_work_launched=False)
    cohort.atomic_json(run/'method_manifest.json', manifest)
    return 0


def add_commands(subparsers):
    from tools.quadra import aligned_organ_group_cohort as cohort
    from tools.quadra import reviewed_evidence
    reviewed_evidence.add_commands(subparsers)
    from tools.quadra import reviewed_anatomical_review
    reviewed_anatomical_review.add_commands(subparsers)
    from tools.quadra import reviewed_comparison
    reviewed_comparison.add_commands(subparsers)
    from tools.quadra import reviewed_uae_matching
    reviewed_uae_matching.add_commands(subparsers)
    freeze = subparsers.add_parser('freeze-reviewed', help='Freeze all reviewed inputs locally without matching')
    freeze.add_argument('--dataset-root', type=Path, required=True)
    freeze.add_argument('--ct-root', type=Path, required=True)
    freeze.add_argument('--registry', type=Path, default=cohort.PROJECT_ROOT/'tools/quadra/totalsegmentator/organs.yaml')
    freeze.add_argument('--registration-parameters', type=Path,
                        default=cohort.PROJECT_ROOT/'configs/quadra/reviewed-registration-parameters.json')
    freeze.add_argument('--output-directory', type=Path, required=True)
    freeze.set_defaults(reviewed_handler=freeze_reviewed)
    prepare = subparsers.add_parser('prepare-method', help='Prepare one method; never launch a pilot')
    prepare.add_argument('--contract', type=Path, required=True)
    prepare.add_argument('--method', choices=METHODS, required=True)
    prepare.add_argument('--run-directory', type=Path, required=True)
    prepare.add_argument('--resume', action='store_true')
    prepare.set_defaults(reviewed_handler=prepare_method)
    fixture = subparsers.add_parser('fixture-method', help='Exercise only an explicitly synthetic outcome adapter')
    fixture.add_argument('--contract', type=Path, required=True)
    fixture.add_argument('--run-directory', type=Path, required=True)
    fixture.add_argument('--outcomes', type=Path, required=True)
    fixture.add_argument('--resume', action='store_true')
    fixture.set_defaults(reviewed_handler=fixture_method)
