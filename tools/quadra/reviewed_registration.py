"""Reviewed native-HU registration adapter; continuous LPS, independent directions.

The public cohort CLI owns invocation. Fixtures do not load CT or ITK. Real
execution requires explicit pilot approval and remains pending scientific review.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np

from tools.quadra import registration_organ_group as group
from tools.quadra import registration_point_transform as pt
from tools.quadra import reviewed_matching_contract as contract_api


class AffineFixtureBackend:
    fixture_only = True

    def __init__(self, path):
        self.identity = dict(name='independent_lps_affine_fixture', **pt.identity(path))
        self.transforms = pt.load_json(path)
        pt.require(set(self.transforms) == {'forward', 'backward'},
                   'Supply both independent directional fixture transforms')
        for values in self.transforms.values():
            a = np.asarray(values, dtype=float)
            pt.require(a.shape == (4,4) and np.isfinite(a).all() and
                       np.allclose(a[3], [0,0,0,1], atol=0, rtol=0), 'Invalid physical affine fixture')

    def estimate(self, fixed_plan, moving_plan, parameters, directory, direction):
        a = np.asarray(self.transforms[direction], dtype=float)
        # Each supplied direction is used as-is; no inverse is constructed.
        return lambda points: pt.apply_affine(points, a)


class ElastixBackend:
    fixture_only = False

    def __init__(self, ct_root, storage_root):
        from tools.quadra.registration_runtime import preflight
        self.ct_root = Path(ct_root).resolve()
        self.identity = dict(name='itk_elastix_independent_rigid_bspline',
                             environment=preflight(storage_root))

    def estimate(self, fixed_plan, moving_plan, parameters, directory, direction):
        import itk
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)
        itk.MultiThreaderBase.SetGlobalMaximumNumberOfThreads(1)
        resolved = []
        for plan in (fixed_plan, moving_plan):
            plan = copy.deepcopy(plan)
            relative = Path(plan['source_ct']['path'])
            pt.require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe CT relative path')
            path = (self.ct_root/relative).resolve()
            pt.require(self.ct_root in path.parents, 'CT path escapes selected source root')
            plan['source_ct']['path'] = str(path)
            resolved.append(plan)
        fixed, moving = [group.load_crop(plan) for plan in resolved]
        filt = itk.ElastixRegistrationMethod.New(fixed, moving)
        filt.SetParameterObject(pt.parameter_object(parameters))
        filt.SetNumberOfThreads(1)
        filt.SetOutputDirectory(str(directory))
        filt.SetLogToFile(True); filt.SetLogToConsole(False)
        filt.UpdateLargestPossibleRegion()
        maps = pt.normalized_transform_maps(filt.GetTransformParameterObject())
        pt.save_transform_chain(directory/'transforms', maps)
        pt.atomic_json(directory/'transform_maps.json', maps, refuse=True)
        return lambda points: pt.transformix_points(points, maps, 1)


def _plans(contract_root, contract):
    plans = {}
    for name in contract['registration_plan_files']:
        plan = pt.load_json(Path(contract_root)/name)
        key = (plan['subject_id'],plan['group_name'],plan['session'])
        pt.require(key not in plans, 'Duplicate reviewed registration crop plan')
        source = plan['source_ct']
        start = np.asarray(plan['crop_start_xyz'], dtype=float)
        stop = np.asarray(plan['crop_end_xyz'], dtype=float)
        shape = np.asarray(source['native_shape_xyz'])
        pt.require(start.shape == (3,) and stop.shape == (3,) and
                   np.isfinite(start).all() and np.isfinite(stop).all() and
                   np.equal(start, np.rint(start)).all() and np.equal(stop, np.rint(stop)).all() and
                   (start >= 0).all() and (stop <= shape).all() and (stop > start).all(),
                   'Invalid native registration crop bounds')
        affine = np.asarray(source['affine'], dtype=float).copy()
        affine[:3, 3] = pt.apply_affine(start, affine)[0]
        geometry = plan['crop_geometry']
        pt.require(list(stop-start) == geometry['native_shape_xyz'] and
                   np.allclose(geometry['affine'], affine, atol=1e-5, rtol=0),
                   'Registration crop geometry disagrees with native CT')
        pt.geometry_checks(source); pt.geometry_checks(geometry)
        if contract.get('fixture_only') is not True:
            pt.require(plan == group.serializable_plan(plan['source_uae_plan']),
                       'Reviewed registration physical-context derivation changed')
        plans[key] = plan
    return plans


def _outcome(row):
    reason = row['failure_reason']
    valid = row['valid_cycle']
    forward_valid = reason not in ('nonfinite_forward','forward_outside_retest_crop')
    result = dict(query_id=row['query_id'], method='registration',
                  status='success' if valid else 'failed',
                  forward_status='success' if forward_valid else 'out_of_domain' if 'outside' in reason else 'nonfinite',
                  reverse_status='success' if valid else 'not_attempted' if not forward_valid else
                                 'out_of_domain' if 'outside' in reason else 'nonfinite',
                  cycle_error_mm=row['cycle_error_mm'] if valid else '', failure_reason=reason,
                  forward_in_domain=forward_valid, reverse_in_domain=bool(valid),
                  coordinate_frame='LPS_mm', point_evaluation='continuous_no_clipping')
    for prefix in ('query','matched','returned'):
        for axis,sign in zip('xyz',(-1,-1,1)):
            value = row[prefix+'_physical_'+axis]
            result[prefix+'_lps_'+axis] = float(value)*sign if value != '' else ''
            result[prefix+'_raw_'+axis] = row[prefix+'_raw_'+axis]
    return result


def run_reviewed_registration(contract_root, run_directory, backend, subject_ids=None,
                               allow_scientific=False, resume=False):
    """Run an injected registration backend and seal each completed group.

    Fixture transforms are independent 4x4 physical-LPS affine matrices.
    Unfinished attempts remain in sibling staging and require inspection; a
    compatible completed resume does not rerun any terminal query. Resource
    feasibility and anatomical validation remain separate real-pilot gates.
    """
    from tools.quadra.reviewed_evidence import execution_signature, read_method_bundle, write_method_bundle
    contract, queries, input_signature = contract_api.read_contract(contract_root)
    is_fixture = contract.get('fixture_only') is True
    pt.require(is_fixture == backend.fixture_only, 'Fixture backend and contract type differ')
    selected = set(subject_ids or sorted({r['subject_id'] for r in queries}))
    pt.require(selected and selected <= {r['subject_id'] for r in queries}, 'Unknown or empty subject selection')
    if not is_fixture:
        pt.require(allow_scientific, 'Explicit scientific pilot approval is required')
        pt.require(selected <= set(contract['pilot_subjects']), 'Cohort execution is not authorized by the pilot command')
    parameters_path = Path(contract_root)/'registration_parameters.json'
    parameters = pt.load_json(parameters_path)
    expected = pt.load_json(Path(__file__).resolve().parents[2]/'configs/quadra/reviewed-registration-parameters.json')
    pt.require(parameters == expected, 'Reviewed registration parameters differ from the pinned approved maps')
    metadata = dict(backend=backend.identity, settings=dict(
        registration_parameters_sha256=pt.identity(parameters_path)['sha256'],
        registration_threads=1, physical_coordinate_frame='LPS_mm', independent_directions=True,
        point_evaluation='continuous_no_clipping', selected_subjects=sorted(selected)),
        scientific_work_launched=not is_fixture,
        pending_gates=['real_data_geometry_and_resource_pilot','anatomical_review'],
        validation_scope='synthetic_physical_transforms' if is_fixture else 'bounded_real_pilot')
    outcomes = {}
    run = Path(run_directory)
    if (run/'method_manifest.json').exists():
        saved = pt.load_json(run/'method_manifest.json')
        pt.require(saved.get('method') == 'registration' and saved.get('input_signature') == input_signature,
                   'Existing registration run has a different method or input contract')
        if saved.get('status') == 'awaiting_pilot':
            pt.require(pt.identity(run/'query_outcomes.csv')['sha256'] == saved['outcomes_sha256'],
                       'Prepared registration ledger changed')
        else:
            pt.require(resume, 'Existing registration evidence requires explicit compatible resume')
    if resume and (run/'output_inventory.json').exists():
        old_manifest, old_rows = read_method_bundle(run, contract_root)
        pt.require(old_manifest.get('execution_signature') == execution_signature(),
                   'Registration implementation changed since the checkpoint')
        old_meta = old_manifest.get('metadata', old_manifest)
        pt.require(old_meta.get('backend') == metadata['backend'] and old_meta.get('settings') == metadata['settings'],
                   'Registration resume backend or settings changed')
        outcomes = {r['query_id']: r for r in old_rows if r['status'] in ('success','failed')}
    elif run.exists():
        # The writer verifies prepared runs; existing unsealed runtime artifacts
        # require operator inspection rather than implicit re-estimation.
        pt.require((run/'method_manifest.json').is_file(), 'Existing unsealed registration run requires inspection')
    plans = _plans(contract_root, contract)
    stage = run.with_name('.'+run.name+'.registration-staging')
    checkpoint_exists = resume and (run/'output_inventory.json').exists()
    start = time.monotonic()
    manifest = None
    for subject, organ_group in sorted({(r['subject_id'],r['group_name']) for r in queries if r['subject_id'] in selected}):
        rows = [r for r in queries if r['subject_id'] == subject and r['group_name'] == organ_group]
        if all(r['query_id'] in outcomes for r in rows): continue
        pt.require(not any(r['query_id'] in outcomes for r in rows), 'Incomplete group checkpoint cannot be silently re-estimated')
        test, retest = [plans[(subject,organ_group,session)] for session in ('test','retest')]
        query_native = np.asarray([[float(r['raw_'+a]) for a in 'xyz'] for r in rows])
        query_lps = pt.apply_affine(query_native, pt.lps_affine(test['source_ct']))
        for row, physical in zip(rows, query_lps):
            if all('physical_lps_'+a in row for a in 'xyz'):
                frozen = np.asarray([float(row['physical_lps_'+a]) for a in 'xyz'])
                pt.require(np.allclose(physical, frozen, atol=1e-5, rtol=0),
                           'Frozen native query and physical LPS coordinates disagree')
        directory = stage/'registration'/subject/organ_group
        pt.require(not directory.exists(), 'Unfinished staged registration group requires inspection before retry')
        directory.mkdir(parents=True)
        phase = 'forward_registration'
        forward = None
        estimates = []
        try:
            forward = backend.estimate(test, retest, parameters, directory/'forward', 'forward')
            estimates.append(dict(direction='forward',status='success',fixed_session='test',moving_session='retest'))
            phase = 'reverse_registration'
            reverse = backend.estimate(retest, test, parameters, directory/'backward', 'backward')
            estimates.append(dict(direction='backward',status='success',fixed_session='retest',moving_session='test'))
            phase = 'point_evaluation'
            for row in group.evaluate_group(rows,test,retest,forward,reverse):
                outcomes[row['query_id']] = _outcome(row)
        except pt.RegistrationError as exc:
            # Preserve the stopping evidence and untouched pending denominator,
            # as well as previously sealed groups. No implicit re-estimation.
            stop = dict(classification=type(exc).__name__,phase=phase,message=str(exc))
            pt.atomic_json(directory/'stop.json',stop,refuse=True)
            metadata['execution_stop'] = stop
            write_method_bundle(contract_root,run,'registration',list(outcomes.values()),
                metadata=metadata,resume=checkpoint_exists,artifact_source=stage)
            raise
        except Exception as exc:
            reason = '{}_runtime_failure: {}: {}'.format(phase,type(exc).__name__,str(exc))
            estimates.append(dict(direction='forward' if phase=='forward_registration' else 'backward',
                                  status='failed',phase=phase,failure_reason=reason))
            if phase == 'reverse_registration' and forward is not None:
                # A failed reverse estimate must not erase the known forward
                # map/coordinates. No inverse or NN substitute is introduced.
                partial = group.evaluate_group(rows,test,retest,forward,
                                                lambda p: np.full_like(p,np.nan,dtype=float))
                for row in partial:
                    result = _outcome(row)
                    if result['forward_status'] == 'success':
                        result.update(reverse_status='registration_failed',failure_reason=reason)
                    outcomes[row['query_id']] = result
            else:
                for row, physical in zip(rows, query_lps):
                    result = dict(query_id=row['query_id'], method='registration',status='failed',
                        forward_status='registration_failed' if phase=='forward_registration' else 'point_evaluation_failed',
                        reverse_status='not_attempted' if phase=='forward_registration' else 'point_evaluation_failed',
                        cycle_error_mm='',failure_reason=reason, coordinate_frame='LPS_mm')
                    result.update({'query_lps_'+a:float(v) for a,v in zip('xyz',physical)})
                    result.update({'query_raw_'+a:float(row['raw_'+a]) for a in 'xyz'})
                    outcomes[row['query_id']] = result
            pt.atomic_json(directory/'failure.json',dict(classification=type(exc).__name__,phase=phase,message=str(exc)))
        pt.atomic_json(directory/'directional_estimates.json',dict(
            subject_id=subject,group_name=organ_group,independent_directions=True,
            registration_parameters_sha256=metadata['settings']['registration_parameters_sha256'],
            estimates=estimates,backend=backend.identity),refuse=True)
        metadata['resources'] = dict(wall_time_seconds=time.monotonic()-start,
                                     measurements_scope='adapter_wall_time_only_not_pilot_peak_memory')
        manifest = write_method_bundle(contract_root,run,'registration',list(outcomes.values()),
            metadata=metadata,resume=checkpoint_exists,artifact_source=stage)
        checkpoint_exists = True
    if manifest is None:
        # Verified completed resume is a no-op; keep its original inventory.
        if checkpoint_exists:
            return old_manifest
        return write_method_bundle(contract_root,run,'registration',list(outcomes.values()),metadata=metadata,resume=resume)
    return manifest


def add_command(subparsers):
    parser = subparsers.add_parser('reviewed-run', help='Reviewed registration fixtures, or an explicitly approved bounded pilot')
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--run-directory',type=Path,required=True)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument('--fixture-transforms',type=Path)
    backend.add_argument('--ct-root',type=Path)
    parser.add_argument('--storage-root',type=Path,default=Path('/workspace/quadra'))
    parser.add_argument('--subjects',nargs='+')
    parser.add_argument('--approve-pilot',action='store_true')
    parser.add_argument('--review-rationale')
    parser.add_argument('--resume',action='store_true')


def execute(args):
    if args.fixture_transforms:
        backend = AffineFixtureBackend(args.fixture_transforms)
    else:
        pt.require(args.approve_pilot and args.review_rationale and args.review_rationale.strip(),
                   'Explicit pilot approval and review rationale required before loading the real runtime')
        backend = ElastixBackend(args.ct_root,args.storage_root)
    return run_reviewed_registration(args.contract,args.run_directory,backend,args.subjects,
                                     allow_scientific=args.approve_pilot,resume=args.resume)
