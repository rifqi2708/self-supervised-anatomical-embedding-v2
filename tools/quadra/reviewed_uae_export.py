"""Read-only retrospective similarity reconstruction from retained embeddings."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from tools.quadra import aligned_organ_group_cohort as cohort
from tools.quadra.reviewed_evidence import read_method_bundle, execution_signature
from tools.quadra.reviewed_matching_contract import read_contract
from tools.quadra.reviewed_uae_matching import FineGridCache, FineGridRetriever, is_cuda_out_of_memory
from tools.quadra.reviewed_uae_diagnostics import decode_anchor_traces, export_similarity_views
from tools.quadra.streaming_cycle_error import EmbeddingCache


def selected_ids(queue_file=None, query_ids=None):
    ids=list(query_ids or [])
    if queue_file:
        path=Path(queue_file)
        if path.suffix.lower()=='.json':
            payload=json.loads(path.read_text())
            records=payload.get('queries',payload.get('records',[])) if isinstance(payload,dict) else payload
            ids.extend(record['query_id'] if isinstance(record,dict) else record for record in records)
        else:
            with path.open(newline='') as handle: ids.extend(row['query_id'] for row in csv.DictReader(handle))
    return list(dict.fromkeys(ids))


def export_reviewed_views(args, cache_pairs=None):
    """Export new cases without touching primary matching or re-extracting a model.

    ``cache_pairs`` is the existing runtime adapter seam: the after-cycle hook
    supplies still-open caches. A retrospective invocation verifies/reopens the
    same persisted arrays and closes only the handles that it owns.
    """
    contract,queries,signature=read_contract(args.contract)
    manifest,rows=read_method_bundle(args.run_directory,args.contract)
    if manifest['method'] not in ('uae_nn','uae_fixed_point'): raise cohort.CohortError('Similarity export requires UAE evidence')
    index=json.loads(Path(args.cache_index).read_text())
    if cohort.sha256_file(args.cache_index)!=manifest['fixture_signature']:
        raise cohort.CohortError('Similarity cache index differs from the primary calculation')
    changed_code=manifest['execution_signature']!=execution_signature()
    if changed_code and not args.reconstruction_reason:
        raise cohort.CohortError('Changed numerical implementation requires an explicit reconstruction reason')
    ids=selected_ids(args.queue_file,args.query_id)
    by_id={row['query_id']:row for row in rows}
    if not ids or any(q not in by_id for q in ids): raise cohort.CohortError('Export requires known query identities')
    if any(by_id[q]['status']=='pending' for q in ids): raise cohort.CohortError('Select queries with recorded terminal outcomes')
    if args.anchor_index<0 or args.budget_bytes<1 or args.min_free_bytes<0:
        raise cohort.CohortError('Invalid anchor/resource settings')
    output=Path(args.output_directory)
    primary=Path(args.run_directory).resolve()
    if output.resolve()==primary or primary in output.resolve().parents:
        raise cohort.CohortError('New exports must be outside the sealed primary bundle')
    if output.exists(): raise cohort.CohortError('Similarity export requires a fresh output directory')
    output.mkdir(parents=True)
    entries={(item['subject_id'],item['group_name']):item for item in index['groups']}
    if len(entries)!=len(index['groups']): raise cohort.CohortError('Duplicate similarity cache group')
    captured={(item['subject_id'],item['group_name'],item['session']):item for item in manifest['cache_identities']}
    settings=manifest['settings']
    backend=args.backend or manifest['backend'];device=args.device or settings['device']
    if (backend!=manifest['backend'] or device!=settings['device']) and not args.reconstruction_reason:
        raise cohort.CohortError('Backend/device changes require declared reconstruction provenance')
    retriever=FineGridRetriever(backend,settings['chunk_locations'],settings['dense_budget_bytes'],device)
    state=dict(schema_version=1,status='partial',source_method=manifest['method'],input_signature=signature,
        source_outcomes_sha256=manifest['outcomes_sha256'],source_execution_signature=manifest['execution_signature'],
        reconstruction_execution_signature=execution_signature(),reconstruction_reason=args.reconstruction_reason,
        changed_numerical_implementation=changed_code,cache_index_sha256=manifest['fixture_signature'],
        numerical_settings=settings,backend=backend,device=device,selected_query_ids=ids,
        cache_provenance=manifest['cache_identities'],source_matching_rerun=False,model_reextracted=False,
        anatomy_validation='pending',resource_guard_policy='between_selected_views; preserve partial evidence',records=[])
    cohort.atomic_json(output/'export_manifest.json',state)
    opened=dict(cache_pairs or {});handles=[];previous_tf32=None;verified_groups=set()
    try:
        if device!='cpu':
            import torch
            previous_tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        for query_id in ids:
            row=by_id[query_id];key=(row['subject_id'],row['group_name'])
            if key not in entries: raise cohort.CohortError('Selected query has no retained cache group')
            entry=entries[key]
            pair=[]
            for session in ('test','retest'):
                spec=entry[session]
                cache_root=Path(spec['cache_directory'])
                if args.cache_root: cache_root=Path(args.cache_root)/cache_root.name
                expected=captured.get((key[0],key[1],session))
                if expected is None: raise cohort.CohortError('Missing primary cache identity')
                if key not in verified_groups:
                    for file,identity in expected['files'].items():
                        observed=cohort.file_identity(cache_root/file)
                        if (observed['sha256'],observed['bytes'])!=(identity['sha256'],identity['bytes']):
                            raise cohort.CohortError('Retained embedding payload changed: '+file)
                if key in opened:
                    cache=opened[key][0 if session=='test' else 1][0]
                else:
                    handle=EmbeddingCache(cache_root);handles.append(handle)
                    if contract.get('fixture_only'):
                        affine=spec['native_to_lps']
                    else:
                        plan=json.loads((Path(args.contract)/'plans'/('{}-{}-{}.json'.format(key[0],session,key[1]))).read_text())
                        affine=np.diag([-1.,-1.,1.,1.]) @ np.asarray(plan['padded_2mm_affine'])
                    cache=FineGridCache(handle,affine)
                pair.append((cache,spec))
            opened[key]=pair
            verified_groups.add(key)
            (source,source_spec),(target,target_spec)=pair
            query_fine=[int(row['fine_'+axis]) for axis in 'xyz']
            maps=[]
            if all(0<=p<int(n) for p,n in zip(query_fine,source.shape_xyz)):
                center=_list(row.get('matched_fine_xyz')) or (target.shape_xyz//2).tolist()
                maps.append(dict(direction='forward',source=source,target=target,point=query_fine,role='seed',center=center,spec=target_spec))
            matched=_list(row.get('matched_native_xyz'))
            if matched is not None:
                reverse_point=target.native_to_fine(matched).tolist()
                center=_list(row.get('returned_fine_xyz')) or (source.shape_xyz//2).tolist()
                maps.append(dict(direction='reverse',source=target,target=source,point=reverse_point,role='reverse_query',center=center,spec=source_spec))
            trace_file=row.get('anchor_trace_file')
            if trace_file:
                relative=Path(trace_file)
                if relative.is_absolute() or '..' in relative.parts: raise cohort.CohortError('Unsafe trace reference')
                for trace in decode_anchor_traces(primary/relative):
                    if trace['query_id']!=query_id: raise cohort.CohortError('Trace query identity differs')
                    if not trace['anchor_history']: continue
                    first=trace['anchor_history'][0]
                    if args.anchor_index>=len(first['query_fine_xyz']): raise cohort.CohortError('Requested anchor index is unavailable')
                    forward=trace['direction']=='forward';a,b=(source,target) if forward else (target,source)
                    maps.append(dict(direction=trace['direction'],source=a,target=b,
                        point=first['query_fine_xyz'][args.anchor_index],role='anchor',center=first['matched_fine_xyz'][args.anchor_index],
                        spec=target_spec if forward else source_spec))
            record=dict(query_id=query_id,source_cycle_error_mm=row.get('cycle_error_mm'),source_status=row['status'],views=[])
            state['records'].append(record)
            if not maps: record['status']='unavailable_outside_grid'
            for view in maps:
                retained=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                if retained>=args.budget_bytes or shutil.disk_usage(output).free<args.min_free_bytes:
                    state.update(status='resource_stopped',failure_reason='similarity_export_resource_guard',pending_query_id=query_id)
                    cohort.atomic_json(output/'export_manifest.json',state);_seal_export(output);return 2
                spec=view['spec'];ct=None
                if spec.get('ct_path'):
                    import nibabel as nib
                    ct_path=Path(spec['ct_path'])
                    if args.ct_root:
                        session='retest' if view['direction']=='forward' else 'test'
                        if contract.get('fixture_only'): raise cohort.CohortError('CT relocation requires real frozen source paths')
                        plan=json.loads((Path(args.contract)/'plans'/('{}-{}-{}.json'.format(key[0],session,key[1]))).read_text())
                        ct_path=Path(args.ct_root)/plan['source_ct']['path']
                    if cohort.sha256_file(ct_path)!=spec['ct_sha256']: raise cohort.CohortError('Diagnostic CT changed')
                    ct=nib.load(str(ct_path))
                path=export_similarity_views(output/view['direction'],query_id,view['source'],view['target'],view['point'],retriever,
                    view['role'],spec.get('admissible_domain'),ct,view['center'])
                record['views'].append(dict(path=path.relative_to(output).as_posix(),map_role=view['role'],
                    direction=view['direction'],query_fine_xyz=view['point'],anchor_index=args.anchor_index if view['role']=='anchor' else None))
                record['status']='complete'
            cohort.atomic_json(output/'export_manifest.json',state)
        state['status']='complete';cohort.atomic_json(output/'export_manifest.json',state);_seal_export(output)
        # Read-only reconstruction must preserve primary numerical evidence.
        if cohort.sha256_file(primary/'query_outcomes.csv')!=state['source_outcomes_sha256']:
            raise cohort.CohortError('Primary outcome evidence changed during reconstruction')
        return 0
    except (MemoryError,RuntimeError) as error:
        if isinstance(error,RuntimeError) and not is_cuda_out_of_memory(error):
            state.update(status='blocked',failure_reason=type(error).__name__+':'+str(error));cohort.atomic_json(output/'export_manifest.json',state);_seal_export(output);raise
        state.update(status='resource_stopped',failure_reason='cuda_out_of_memory' if is_cuda_out_of_memory(error) else str(error))
        cohort.atomic_json(output/'export_manifest.json',state);_seal_export(output);return 2
    except Exception as error:
        state.update(status='blocked',failure_reason=type(error).__name__+':'+str(error))
        cohort.atomic_json(output/'export_manifest.json',state);_seal_export(output);raise
    finally:
        for handle in handles: handle.close()
        if previous_tf32 is not None: torch.backends.cuda.matmul.allow_tf32=previous_tf32


def _list(value):
    if isinstance(value,str): return json.loads(value) if value else None
    return value


def _seal_export(root):
    files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=cohort.sha256_file(p))
           for p in sorted(root.rglob('*')) if p.is_file() and p.name!='export_inventory.json']
    cohort.atomic_json(root/'export_inventory.json',dict(schema_version=1,files=files))


def add_commands(subparsers):
    parser=subparsers.add_parser('reviewed-uae-export-views',help='Reconstruct newly selected cases from retained embeddings without rerunning primary matching')
    for name in ('contract','run-directory','cache-index','output-directory'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--query-id',action='append')
    parser.add_argument('--queue-file')
    parser.add_argument('--cache-root');parser.add_argument('--ct-root')
    parser.add_argument('--backend',choices=('auto','dense','streamed'))
    parser.add_argument('--device');parser.add_argument('--reconstruction-reason')
    parser.add_argument('--anchor-index',type=int,default=0)
    parser.add_argument('--budget-bytes',type=int,default=1024**3)
    parser.add_argument('--min-free-bytes',type=int,default=2*1024**3)
    parser.set_defaults(reviewed_handler=export_reviewed_views)
