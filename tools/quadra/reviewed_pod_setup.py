"""Prepare private SSH-staged inputs for the existing disposable bootstrap."""
from __future__ import print_function

import copy
import csv
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path


def write_archive(output, members):
    """Write deterministic archives, checking each payload before inclusion."""
    from tools.quadra.disposable_pod import DisposableError
    temporary = output.with_name('.'+output.name+'.partial')
    with temporary.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as archive:
                for name, source, expected in sorted(members, key=lambda item: item[0]):
                    if source.is_symlink(): raise DisposableError('Refusing a symlink package input')
                    payload = source.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != expected:
                        raise DisposableError('Package input changed: '+name)
                    info = tarfile.TarInfo(name)
                    info.size = len(payload); info.mode = 0o444; info.mtime = 0
                    archive.addfile(info, io.BytesIO(payload))
    os.rename(str(temporary), str(output))


def verify_archive(path, expected):
    """Read every packed payload and reject extra, duplicate or unsafe entries."""
    from tools.quadra.disposable_pod import DisposableError
    seen = set()
    with tarfile.open(str(path), 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise DisposableError('Unexpected or duplicate package member: '+member.name)
            size, sha = expected[member.name]
            stream = archive.extractfile(member)
            if stream is None: raise DisposableError('Unreadable package member')
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
            if member.size != size or digest.hexdigest() != sha:
                raise DisposableError('Package member identity mismatch: '+member.name)
            seen.add(member.name)
    if seen != set(expected): raise DisposableError('Missing package payloads')
    return len(seen)


def command_verify_package(args):
    from tools.quadra import disposable_pod as disposable
    output = Path(args.package_directory)
    receipt = json.loads((output/'setup_package_receipt.json').read_text())
    for record in receipt['files']:
        path = output/record['filename']
        if (Path(record['filename']).name != record['filename'] or path.is_symlink()
                or path.stat().st_size != record['bytes'] or disposable.sha256_file(path) != record['sha256']):
            raise disposable.DisposableError('Setup package file identity changed')
    with tarfile.open(str(output/'matching-contract-v2.tar.gz'), 'r:gz') as archive:
        manifest_stream = archive.extractfile('matching-contract-v2/matching_contract.json')
        inventory_stream = archive.extractfile('matching-contract-v2/portable_mask_inventory.csv')
        if manifest_stream is None or inventory_stream is None:
            raise disposable.DisposableError('Missing archived reviewed contract')
        manifest_bytes = manifest_stream.read()
        contract = json.loads(manifest_bytes.decode('utf-8'))
        inventory_bytes = inventory_stream.read()
    inventory = list(csv.DictReader(io.StringIO(inventory_bytes.decode('utf-8'))))
    contracts = {'matching-contract-v2/'+r['path']:(r['bytes'],r['sha256']) for r in contract['files']}
    contracts['matching-contract-v2/matching_contract.json'] = (len(manifest_bytes),hashlib.sha256(manifest_bytes).hexdigest())
    masks = {'reviewed-masks/'+r['mask_relative_path']:(int(r['mask_bytes']),r['mask_sha256']) for r in inventory}
    if len(masks) != 3790: raise disposable.DisposableError('Reviewed package denominator changed')
    verify_archive(output/'matching-contract-v2.tar.gz', contracts)
    verify_archive(output/'reviewed-masks.tar.gz', masks)
    print('Local package outer hashes and all archived payload identities verified; remote verification remains pending.')
    return 0


def command_package(args):
    from tools.quadra import aligned_organ_group_cohort as cohort
    from tools.quadra import disposable_pod as disposable
    from tools.quadra import reviewed_matching_contract as reviewed
    contract_root = Path(args.contract).resolve()
    dataset = Path(args.dataset_root).resolve()
    output = Path(args.output_directory).resolve()
    if output.exists(): raise disposable.DisposableError('Refusing to overwrite a setup package')
    try:
        contract, _, signature = reviewed.read_contract(contract_root)
    except cohort.CohortError as exc:
        raise disposable.DisposableError(str(exc))
    if contract.get('counts', {}).get('masks') != 3790:
        raise disposable.DisposableError('Setup requires the approved 48-subject reviewed contract')
    inventory = cohort.read_csv(contract_root/'portable_mask_inventory.csv')
    if len(inventory) != 3790 or len({r['mask_relative_path'] for r in inventory}) != 3790:
        raise disposable.DisposableError('Reviewed package mask inventory mismatch')
    members = []
    for record in inventory:
        relative = Path(record['mask_relative_path'])
        source = dataset/relative
        if relative.is_absolute() or '..' in relative.parts or dataset not in source.resolve().parents:
            raise disposable.DisposableError('Unsafe reviewed mask package member')
        members.append(('reviewed-masks/'+relative.as_posix(), source, record['mask_sha256']))
    output.mkdir(parents=True)
    mask_package = output/'reviewed-masks.tar.gz'
    write_archive(mask_package, members)
    contract_package = output/'matching-contract-v2.tar.gz'
    records = list(contract['files']) + [dict(path='matching_contract.json', sha256=disposable.sha256_file(contract_root/'matching_contract.json'))]
    write_archive(contract_package, [('matching-contract-v2/'+r['path'], contract_root/r['path'], r['sha256']) for r in records])
    mask_verified = verify_archive(mask_package, {'reviewed-masks/'+r['mask_relative_path']:
        (int(r['mask_bytes']),r['mask_sha256']) for r in inventory})
    contract_verified = verify_archive(contract_package, {'matching-contract-v2/'+r['path']:
        (int(r.get('bytes', (contract_root/r['path']).stat().st_size)),r['sha256']) for r in records})
    catalog = copy.deepcopy(disposable.load_catalog(args.base_catalog))
    catalog.update(schema_version=2, catalogue_id='quadra-reviewed-matching-v2',
                   input_signature=signature, subjects=dict(first=1, last=48, count=48))
    assets = catalog['assets']
    assets['whole_body_ct']['expected'].update(promoted_ct_files=96, promoted_subject_directories=48,
                                             selected_subject_first=1, selected_subject_last=48)
    remote = str(Path(args.remote_storage_root)/'staging/reviewed-inputs')
    assets['stage5_masks'] = dict(profiles=['uae','registration'], filename=mask_package.name,
        archive_type='tar.gz', payload_subpath='reviewed-masks',
        promote_to='datasets/derivatives/'+contract['dataset_id'],
        local_path=remote+'/'+mask_package.name, bytes=mask_package.stat().st_size,
        sha256=disposable.sha256_file(mask_package), checksum_entries=0,
        source_kind='new_final_reviewed_masks',
        expected=dict(final_masks=3790, intermediate_masks=0, subjects=48, scans=96))
    assets['experiment_contract'] = dict(profiles=['uae','registration'], filename=contract_package.name,
        archive_type='tar.gz', payload_subpath='matching-contract-v2',
        promote_to='metadata/matching-contract-v2', local_path=remote+'/'+contract_package.name,
        bytes=contract_package.stat().st_size, sha256=disposable.sha256_file(contract_package),
        expected=dict(schema_version=2, frozen_queries=contract['query_count'], organ_groups=4))
    disposable.atomic_json(output/'disposable-reviewed-assets.json', catalog)
    for profile, image in disposable.EXPECTED_IMAGES.items():
        command = ['bash', 'setup.sh', 'disposable-bootstrap', '--profile', profile,
            '--asset-catalog', remote+'/disposable-reviewed-assets.json',
            '--storage-root', str(args.remote_storage_root), '--repository-ref', cohort.git_output(['rev-parse','HEAD']),
            '--image-ref', image['ref'], '--confirm-image-digest', image['digest'], '--setup-only']
        disposable.atomic_json(output/(profile+'-setup-plan.json'), dict(
            profile=profile, input_signature=signature, bootstrap_command=command,
            source_code_commit=cohort.git_output(['rev-parse','HEAD']),
            scientific_work_launched=False, pod_created=False,
            setup_status='local_package_ready_remote_setup_not_started',
            upload_files=[mask_package.name, contract_package.name, 'disposable-reviewed-assets.json'],
            upload_destination=remote, other_asset_transport='existing_checksum_pinned_drive_packages',
            pending_gates=['deployment_approval','remote_asset_access','runtime_preflight','bounded_smoke','scientific_pilot'],
            proposed_container_disk_gb=100, proposed_gpu_memory_gb=48 if profile=='uae' else None,
            proposed_host_ram_gb=64, proposed_vcpus=8 if profile=='registration' else None))
    disposable.atomic_json(output/'setup_package_receipt.json', dict(
        schema_version=2, input_signature=signature, masks=3790, contract_queries=contract['query_count'],
        source_masks_preserved=True, scientific_work_launched=False,
        archive_payload_verification=dict(mask_members=mask_verified, contract_members=contract_verified),
        status='LOCAL_PACKAGE_PREPARED_NOT_REMOTE_VERIFIED',
        files=[dict(filename=p.name, bytes=p.stat().st_size, sha256=disposable.sha256_file(p))
               for p in sorted(output.iterdir()) if p.is_file()]))
    print('Reviewed setup package prepared; no pod, smoke or pilot was launched.')
    return 0


def add_commands(subparsers):
    from tools.quadra import disposable_pod
    package = subparsers.add_parser('reviewed-package', help='Prepare private SSH uploads locally')
    package.add_argument('--contract', type=Path, required=True)
    package.add_argument('--dataset-root', type=Path, required=True)
    package.add_argument('--output-directory', type=Path, required=True)
    package.add_argument('--base-catalog', type=Path, default=disposable_pod.DEFAULT_CATALOG)
    package.add_argument('--remote-storage-root', type=Path, default=Path('/workspace/quadra'))
    package.set_defaults(handler=command_package)
    verify = subparsers.add_parser('reviewed-verify-package', help='Verify local upload hashes and archived payloads')
    verify.add_argument('--package-directory', type=Path, required=True)
    verify.set_defaults(handler=command_verify_package)
