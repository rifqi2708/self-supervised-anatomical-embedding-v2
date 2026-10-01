"""Portable, sealed method evidence; no transport or scientific execution."""
from __future__ import print_function

import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from tools.quadra import aligned_organ_group_cohort as cohort
from tools.quadra import reviewed_matching_contract as contracts


def execution_signature():
    names = ['aligned_organ_group_cohort.py', 'reviewed_matching_contract.py',
             'reviewed_evidence.py', 'reviewed_uae_matching.py',
             'reviewed_uae_diagnostics.py', 'reviewed_uae_cuda.py', 'reviewed_registration.py',
             'reviewed_uae_export.py',
             'registration_runtime.py', 'registration_point_transform.py',
             'registration_organ_group.py', 'registration_organ_group_cohort.py',
             'streaming_cycle_error.py', 'streaming_embedding.py',
             'organ_group_lattice_alignment.py', 'memory_configuration_screen.py',
             'uaes_matching.py']
    return cohort.sha256_payload({name: cohort.sha256_file(cohort.PROJECT_ROOT/'tools/quadra'/name)
        for name in names if (cohort.PROJECT_ROOT/'tools/quadra'/name).is_file()})


def contract_identity(root, manifest, signature):
    files = {item['path']: item['sha256'] for item in manifest['files']}
    return dict(input_signature=signature, dataset_id=manifest['dataset_id'],
        query_sha256=files['frozen_queries_raw_itk.csv'],
        crop_signature=cohort.sha256_payload({k:v for k,v in files.items()
            if k.startswith(('plans/', 'registration_plans/'))}),
        coordinate_settings=manifest.get('settings', {}), model=manifest.get('model', {}))


def _members(root):
    result = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise cohort.CohortError('Evidence cannot contain symlinks')
        if path.is_file() and path != root/'output_inventory.json':
            result.append(dict(path=path.relative_to(root).as_posix(),
                bytes=path.stat().st_size, sha256=cohort.sha256_file(path)))
    return result


def write_method_bundle(contract_root, run_directory, method, outcomes, metadata=None, resume=False,
                        artifact_source=None):
    """Build a checkpoint beside the previous bundle, then promote it.

    Immutable retained artifacts use hard links on the same filesystem. Control
    files are replaced atomically, so writes never change the previous snapshot.
    Failed candidates and previous checkpoints remain available for inspection.
    """
    import fcntl
    target = Path(run_directory).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    token = uuid.uuid4().hex
    candidate = target.with_name('.'+target.name+'.checkpoint-'+token)
    previous = target.with_name('.'+target.name+'.previous-'+token)
    with target.with_name('.'+target.name+'.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise cohort.CohortError('Another producer is writing this method bundle')
        if target.exists():
            _members(target)  # reject symlinks before creating a candidate
            shutil.copytree(str(target),str(candidate),copy_function=os.link)
        else:
            candidate.mkdir()
        manifest = _write_method_checkpoint(contract_root,candidate,method,outcomes,
            metadata,resume,artifact_source)
        moved = False
        try:
            if target.exists():
                os.rename(str(target),str(previous)); moved=True
            os.rename(str(candidate),str(target))
        except Exception:
            if moved and not target.exists():
                os.rename(str(previous),str(target))
            raise
    return manifest


def _write_method_checkpoint(contract_root, run_directory, method, outcomes, metadata=None, resume=False,
                             artifact_source=None):
    """Merge outcomes into the frozen denominator and seal all run evidence.

    Existing terminal outcomes are immutable on resume. Pending rows may advance.
    The inventory is written last; interrupted writes remain unsealed evidence.
    """
    contract, queries, signature = contracts.read_contract(contract_root)
    if method not in contracts.METHODS:
        raise cohort.CohortError('Unknown matching method')
    root = Path(run_directory)
    metadata = dict(metadata or {})
    launched = metadata.pop('scientific_work_launched', None)
    if launched is not None and type(launched) is not bool:
        raise cohort.CohortError('Scientific launch attestation must be boolean or unknown')
    if contract.get('fixture_only') and launched is True:
        raise cohort.CohortError('A synthetic fixture cannot attest scientific execution')
    forbidden = {'schema_version', 'method', 'input_signature', 'dataset_id',
        'query_count', 'attempted_queries', 'successful_queries', 'failed_queries',
        'outcomes_sha256', 'contract_identity', 'execution_signature'}
    if forbidden & set(metadata):
        raise cohort.CohortError('Metadata cannot override evidence identities or denominators')
    previous_rows = []
    previous_manifest = None
    if (root/'method_manifest.json').exists():
        previous_manifest = cohort.load_json(root/'method_manifest.json')
        if previous_manifest.get('method') != method or previous_manifest.get('input_signature') != signature:
            raise cohort.CohortError('Existing method/input contract differs')
        if previous_manifest.get('status') == 'awaiting_pilot':
            if cohort.sha256_file(root/'query_outcomes.csv') != previous_manifest['outcomes_sha256']:
                raise cohort.CohortError('Prepared outcome ledger changed')
        else:
            if not resume:
                raise cohort.CohortError('Existing evidence requires explicit compatible resume')
            previous_manifest, previous_rows = read_method_bundle(root, contract_root)
            if previous_manifest.get('execution_signature') != execution_signature():
                raise cohort.CohortError('Cannot resume after changing implementation')
            for key in ('settings', 'backend', 'fixture_signature'):
                if previous_manifest.get(key) != metadata.get(key):
                    raise cohort.CohortError('Cannot resume with changed '+key)
    elif root.exists() and any(root.iterdir()):
        # Diagnostic producers may create files before the first checkpoint.
        if (root/'query_outcomes.csv').exists() or (root/'output_inventory.json').exists():
            raise cohort.CohortError('Unsealed existing outcome evidence is preserved, not overwritten')
    records = {}
    frozen = {q['query_id']: q for q in queries}
    for row in outcomes:
        query_id = row['query_id']
        if query_id in records or query_id not in frozen:
            raise cohort.CohortError('Duplicate or unknown query outcome')
        for key, value in frozen[query_id].items():
            if key in row and str(row[key]) != str(value):
                raise cohort.CohortError('Outcome changed frozen query provenance: '+key)
        if row.get('method', method) != method:
            raise cohort.CohortError('Outcome method differs')
        record = dict(frozen[query_id]); record.update(row); record['method'] = method
        status = record.get('status')
        if status not in ('success', 'failed', 'pending'):
            raise cohort.CohortError('Unknown query status')
        if status == 'success':
            error = float(record['cycle_error_mm'])
            if not math.isfinite(error) or error < 0 or any(record.get(k) != 'success' for k in ('forward_status','reverse_status')):
                raise cohort.CohortError('Invalid successful cycle')
        elif status == 'failed':
            if (not record.get('failure_reason') or record.get('cycle_error_mm') not in ('',None)
                    or all(record.get(k)=='success' for k in ('forward_status','reverse_status'))):
                raise cohort.CohortError('Failed cycle requires a reason, a failed direction and blank cycle error')
        records[query_id] = record
    merged = [records.get(q['query_id'], dict(q, method=method, status='pending',
        forward_status='pending', reverse_status='pending')) for q in queries]
    fields = list(dict.fromkeys(k for row in merged for k in row))
    encoded = [{k:json.dumps(v, sort_keys=True) if isinstance(v,(dict,list,tuple)) else '' if v is None else v
        for k,v in row.items()} for row in merged]
    encoded_by_id = {row['query_id']:row for row in encoded}
    for old in previous_rows:
        if old['status'] != 'pending':
            new = encoded_by_id[old['query_id']]
            if any(str(new.get(k, '')) != str(v) for k,v in old.items()):
                raise cohort.CohortError('Resume attempted to replace a terminal outcome')
    root.mkdir(parents=True, exist_ok=True)
    if artifact_source is not None:
        staging = Path(artifact_source).resolve()
        if staging == root.resolve() or root.resolve() in staging.parents:
            raise cohort.CohortError('Stage producer artifacts outside the sealed bundle')
        staged_members = _members(staging)
        # Validate the entire incoming set before copying any new member.
        for member in staged_members:
            relative = Path(member['path'])
            if relative.name in ('query_outcomes.csv','method_manifest.json','output_inventory.json'):
                raise cohort.CohortError('Producer artifacts cannot replace bundle control files')
            origin, destination = staging/relative, root/relative
            if destination.exists():
                if cohort.sha256_file(destination) != member['sha256']:
                    raise cohort.CohortError('Conflicting producer artifact is preserved: '+member['path'])
        for member in staged_members:
            relative = Path(member['path'])
            origin, destination = staging/relative, root/relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(destination.parent),delete=False) as handle:
                temporary = Path(handle.name)
            shutil.copyfile(str(origin),str(temporary))
            if cohort.sha256_file(temporary) != member['sha256']:
                raise cohort.CohortError('Producer artifact changed while copying')
            os.rename(str(temporary),str(destination))
    cohort.atomic_csv(root/'query_outcomes.csv', encoded, fieldnames=fields)
    attempted = sum(r['status'] != 'pending' for r in merged)
    manifest = dict(metadata, schema_version=2, method=method, dataset_id=contract['dataset_id'],
        input_signature=signature, contract_identity=contract_identity(contract_root, contract, signature),
        execution_signature=execution_signature(), execution_commit=cohort.git_output(['rev-parse','HEAD']),
        source_tree_clean=not bool(cohort.git_output(['status','--porcelain'])),
        fixture_only=contract.get('fixture_only', False),
        scientific_work_launched=False if contract.get('fixture_only') else launched,
        query_count=len(merged), attempted_queries=attempted,
        successful_queries=sum(r['status']=='success' for r in merged),
        failed_queries=sum(r['status']=='failed' for r in merged),
        status=('fixture_complete' if contract.get('fixture_only') else 'technical_complete')
            if attempted == len(merged) else 'partial',
        outcomes_sha256=cohort.sha256_file(root/'query_outcomes.csv'),
        anatomical_validation='pending',
        recovery_contracts=dict(generated_evidence='this_bundle',
            corrected_masks='separate_verified_dataset_derivative', model='separate_checkpoint_asset',
            embeddings='separate_retention_or_regeneration_policy', source='separate_git_commit'))
    cohort.atomic_json(root/'method_manifest.json', manifest)
    cohort.atomic_json(root/'output_inventory.json', dict(schema_version=1,
        method=method, input_signature=signature, files=_members(root)))
    return manifest


def read_method_bundle(run_directory, contract_root=None):
    root = Path(run_directory)
    inventory = cohort.load_json(root/'output_inventory.json')
    records = inventory.get('files', [])
    names = [r['path'] for r in records]
    if len(names) != len(set(names)) or not {'method_manifest.json','query_outcomes.csv'} <= set(names):
        raise cohort.CohortError('Incomplete or duplicate evidence inventory')
    for record in records:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise cohort.CohortError('Unsafe evidence member')
    if _members(root) != records:
        raise cohort.CohortError('Evidence checksum, size or coverage mismatch')
    manifest = cohort.load_json(root/'method_manifest.json')
    if inventory.get('schema_version') != 1 or manifest.get('schema_version') != 2 or manifest.get('method') not in contracts.METHODS:
        raise cohort.CohortError('Unsupported sealed method evidence')
    rows = cohort.read_csv(root/'query_outcomes.csv')
    for row in rows:
        if row['status'] not in ('success','failed','pending'):
            raise cohort.CohortError('Unknown sealed outcome status')
        if row['status'] == 'failed' and (not row.get('failure_reason') or row.get('cycle_error_mm') not in ('',None)
                or all(row.get(k)=='success' for k in ('forward_status','reverse_status'))):
            raise cohort.CohortError('Invalid sealed failure outcome')
        if row['status'] == 'success':
            error = float(row['cycle_error_mm'])
            if not math.isfinite(error) or error < 0 or any(row.get(k) != 'success' for k in ('forward_status','reverse_status')):
                raise cohort.CohortError('Invalid sealed successful cycle')
    ids = [r['query_id'] for r in rows]
    if (len(ids) != len(set(ids)) or len(rows) != manifest['query_count']
            or cohort.sha256_file(root/'query_outcomes.csv') != manifest['outcomes_sha256']
            or any(r.get('method') != manifest['method'] for r in rows)
            or inventory['method'] != manifest['method'] or inventory['input_signature'] != manifest['input_signature']):
        raise cohort.CohortError('Method/query evidence identities disagree')
    for key, value in [('attempted_queries',sum(r['status']!='pending' for r in rows)),
                       ('successful_queries',sum(r['status']=='success' for r in rows)),
                       ('failed_queries',sum(r['status']=='failed' for r in rows))]:
        if manifest.get(key) != value:
            raise cohort.CohortError('Outcome denominators disagree')
    if contract_root is not None:
        contract, queries, signature = contracts.read_contract(contract_root)
        if (manifest['contract_identity'] != contract_identity(contract_root,contract,signature)
                or manifest['input_signature'] != signature or manifest['dataset_id'] != contract['dataset_id']
                or manifest['fixture_only'] != contract.get('fixture_only',False)):
            raise cohort.CohortError('Dataset/query/crop/coordinate/model experiment identities differ')
        frozen = {r['query_id']:r for r in queries}
        if set(ids) != set(frozen) or any(str(row.get(k,'')) != str(v)
            for row in rows for k,v in frozen[row['query_id']].items()):
            raise cohort.CohortError('Frozen query provenance changed')
    return manifest, rows


def reconcile_bundles(contract_root, bundle_dirs):
    contract, queries, signature = contracts.read_contract(contract_root)
    bundles = {}
    for directory in bundle_dirs:
        manifest, rows = read_method_bundle(directory,contract_root)
        method = manifest['method']
        if method in bundles:
            raise cohort.CohortError('Duplicate method bundle')
        bundles[method] = dict(manifest=manifest, rows=rows, directory=str(directory))
    if {'uae_nn','uae_fixed_point'} <= set(bundles):
        left,right = [bundles[m]['manifest'] for m in ('uae_nn','uae_fixed_point')]
        required = not contract.get('fixture_only',False)
        if required or 'cache_identities' in left or 'cache_identities' in right:
            if (not left.get('fixture_signature') or left.get('fixture_signature') != right.get('fixture_signature')):
                raise cohort.CohortError('NN and fixed-point must use the identical extraction cache index')
            def payloads(manifest):
                result = {}
                for entry in manifest.get('cache_identities',[]):
                    key = (entry['subject_id'],entry['group_name'],entry['session'])
                    if key in result:
                        raise cohort.CohortError('Duplicate embedding cache identity')
                    result[key] = {head:entry['files'][head+'.npy']['sha256'] for head in ('fine','coarse','semantic')}
                return result
            a,b = payloads(left),payloads(right)
            for key in set(a)&set(b):
                if a[key] != b[key]:
                    raise cohort.CohortError('NN and fixed-point used different embedding payloads')
            if required:
                indexes = {m:{r['query_id']:r for r in bundles[m]['rows']} for m in ('uae_nn','uae_fixed_point')}
                common = set(a)&set(b)
                for query in queries:
                    qid = query['query_id']
                    attempted = [indexes[m][qid]['status']!='pending'
                        for m in ('uae_nn','uae_fixed_point')]
                    if all(attempted) and any((query['subject_id'],query['group_name'],session) not in common
                        for session in ('test','retest')):
                        raise cohort.CohortError('Paired attempted queries have no shared embedding provenance')
    return dict(contract=contract,queries=queries,input_signature=signature,bundles=bundles)


def intake(args):
    source, target = Path(args.source).resolve(), Path(args.output_directory).resolve()
    if target.exists():
        raise cohort.CohortError('Intake destination already exists; preserve conflicts under a new ID')
    manifest, rows = read_method_bundle(source,args.contract)
    target.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+target.name+'.staged-',dir=str(target.parent)))
    try:
        bundle = stage/'bundle'; shutil.copytree(str(source),str(bundle))
        manifest, rows = read_method_bundle(bundle,args.contract)
        inventory_sha = cohort.sha256_file(source/'output_inventory.json')
        if inventory_sha != cohort.sha256_file(bundle/'output_inventory.json'):
            raise cohort.CohortError('Source inventory changed during intake')
        from tools.quadra import artifact_backup
        cohort.atomic_json(stage/'intake_receipt.json',dict(schema_version=1,
            verification='checksum_verified', method=manifest['method'],
            input_signature=manifest['input_signature'], original_run_id=source.name,
            original_execution_commit=manifest['execution_commit'], source_inventory_sha256=inventory_sha,
            query_count=len(rows), run_status=manifest['status'],
            remote_transport_verification='not_attested_local_staged_intake_only',
            backup_allowlist=list(artifact_backup.ALLOWLIST),
            backup_coverage='requires_location_and_transfer_receipt_check; checksums cover every listed bundle member',
            recovery_contracts=manifest['recovery_contracts']))
        os.rename(str(stage),str(target))
    except Exception:
        # Preserve rejected staging for inspection instead of deleting evidence.
        raise
    return 0


def add_commands(subparsers):
    parser = subparsers.add_parser('reviewed-intake',help='Checksum-verify staged method evidence')
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.set_defaults(reviewed_handler=intake)
