#!/usr/bin/env python3
"""Immutable train-pool expansion: new A trajectories and seed42/43/44 B grids.

Does not train a controller or regenerate inherited validation outcomes.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from safetensors.torch import load_file
from controller.profile import GRID,file_hash,tensor_hash,write_json,write_jsonl,write_tensors,atomic_exclusive_bytes,load_official_medqa
from controller.sampler import train_pool
from controller.server_training import repeat_evidence,current_sources,bind_archived_realignment
from scripts.migrate_b_seed_repeats import contained,verify_hashes,relocate_config,runtime_fingerprint
from scripts.run_matched_controller_ppo import completed,new_attempt,commit,reset_cache,event
from scripts.profile_controller_pool import capture,check_parity
from scripts.pilot_greedy_b import reconstruct_payloads,score_text,read_rows
from scripts.audit_saved_loop_budgets import repetition
from scripts.verify_controller_profile import reference_phi

SCHEMA='medqa_train_pool_expansion_v1'
SEEDS=(42,43,44)


def selected_pool(data_root,parent_ids,total=1024):
    if total<=len(parent_ids) or (total-len(parent_ids))%16:
        raise ValueError('Expansion must add full batch16 cohorts')
    items,meta=train_pool(data_root,total,42,parent_ids)
    if meta['source_ids'][:len(parent_ids)]!=list(parent_ids):
        raise ValueError('Inherited question order changed')
    return items,meta


def tensors_from_samples(features,samples,split):
    if (features.ndim!=3 or features.shape[1]!=7 or samples.shape!=(len(features),8,3)
            or not torch.isfinite(features).all() or not ((samples==0)|(samples==1)).all()):
        raise ValueError('Invalid expanded feature/sample grid')
    if split not in ('train','validation'):
        raise ValueError('Test tensors prohibited')
    q=samples.float().mean(-1)
    out=dict(features=features,correctness=q,correctness_samples=samples)
    if split=='train':
        future=q.flip(1).cummax(1).values.flip(1)
        targets=torch.stack((q[:,:-1],future[:,1:]),dim=-1)
        out.update(actions=(targets[:,:,0]<targets[:,:,1]).long(),q_targets=targets)
    return out


def keep_last_logits(model):
    """Only unused HF vocabulary logits are shortened; hidden states/KV unchanged."""
    original=model.forward
    def forward(*args,**kwargs):
        if kwargs.get('labels') is not None:
            raise ValueError('No LLM training in profiling')
        kwargs['logits_to_keep']=1
        return original(*args,**kwargs)
    model.forward=forward


def disk_guard(root,required_bytes):
    free=shutil.disk_usage(root).free
    if free<required_bytes:
        raise RuntimeError(f'Disk guard: free={free}, required={required_bytes}; no files deleted')
    return free


def preflight_lengths(items,cfg):
    from models import ModelWrapper
    from transformers import AutoTokenizer
    from controller.matched import render_batch
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_name'],revision=cfg['model_revision'],local_files_only=True)
    tokenizer.padding_side='left'
    wrapper=ModelWrapper.__new__(ModelWrapper);wrapper.tokenizer=tokenizer
    lengths=[]
    for start in range(0,len(items),16):
        _,e=render_batch(wrapper,SimpleNamespace(**cfg),items[start:start+16])
        lengths.extend((e['A']['attention_mask'].sum(1)+e['B']['attention_mask'].sum(1)+80).tolist())
    if max(lengths)+cfg['max_new_tokens']>cfg['vllm_max_model_len']:
        raise ValueError('Context limit insufficient; no truncation permitted')
    return dict(max_prompt_embeds_tokens=max(lengths),max_output_tokens=4096,
                model_context_limit=cfg['vllm_max_model_len'],silent_truncation=False)


def prepare(config_path,dry_run=False):
    settings=yaml.safe_load(contained(config_path,ROOT).read_text())
    if settings['schema']!=SCHEMA or settings['total_train']!=1024 or settings['request_seeds']!=list(SEEDS):
        raise ValueError('Expected deliberate train1024/three-seed expansion')
    base=contained(ROOT/settings['parent_profile'],ROOT)
    repeats=contained(ROOT/settings['parent_repeats'],ROOT)
    data=contained(ROOT/settings['data_root'],ROOT)
    out=contained(ROOT/settings['out'],ROOT)
    if out.exists() or out==ROOT or base in out.parents or repeats in out.parents:
        raise ValueError('Require fresh separate campaign output')
    pm=json.loads((base/'manifest.json').read_text())
    with (base/'pipeline.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        _,evidence=repeat_evidence(ROOT,base,pm,repeats)
    items,selection=selected_pool(data,pm['selections']['train']['source_ids'],settings['total_train'])
    new=items[len(pm['selections']['train']['source_ids']):]
    cfg=relocate_config(pm['config'],ROOT,data,settings['gpu_memory_utilization'],'cuda:1')
    cfg.update(device='cuda:1',hf_memory_reservation_gib=settings['hf_memory_reservation_gib'],
        vllm_max_model_len=8192,vllm_max_num_seqs=16,vllm_max_num_batched_tokens=4096,
        train_samples=len(items),validation_samples=128,test_samples=0)
    if cfg['model_revision']!='1cfa9a7208912126459214e8b04321603b3df60c':
        raise ValueError('Pinned revision required')
    lengths=preflight_lengths(new,cfg)
    teacher_bytes=sum(contained(p,ROOT).stat().st_size for p in (base/'train/teachers').rglob('*') if p.is_file())
    estimated=int(teacher_bytes/256*len(new)*1.1)+1024**3
    reserve=int(settings['disk_reserve_gib']*1024**3)
    disk_guard(ROOT,estimated+reserve)
    sources=current_sources(ROOT,config_path)
    plan=dict(schema=SCHEMA,output=str(out),parent_profile=str(base),parent_repeats=str(repeats),
        parent_manifest_sha256=file_hash(base/'manifest.json'),parent_repeat_evidence=evidence,
        config=cfg,source_sha256=sources,runtime=runtime_fingerprint(),
        selection=selection,new_source_ids=[x['source_id'] for x in new],
        inherited_train_ids=pm['selections']['train']['source_ids'],
        validation_ids=pm['selections']['validation']['source_ids'],
        new_questions=len(new),new_cohorts=len(new)//16,request_seeds=list(SEEDS),
        grid=list(GRID),expected_new_B_completions=len(new)*8*3,
        expected_B_completions_per_seed=len(new)*8,
        expected_total_train=1024,expected_total_validation=128,
        estimated_new_bytes=estimated,disk_reserve_bytes=reserve,
        runtime_disk_min_bytes=4*1024**3,
        teacher_bytes_per_cohort=math.ceil(teacher_bytes/16*1.1),
        rest_seconds_43_44=settings['rest_seconds_43_44'],
        hf_logits_to_keep=1,length_audit=lengths,
        cache='New-shard cold each seed/K; warm fixed768-question cohort order; preceding commits replayed explicitly on resume',
        lineage='Inherited256train/128validation retain mixed GB10/H100 archives; new768 are H100; old archives referenced, never copied/rewritten',
        normalization='Raw features; future training must fit statistics on full train1024 only',
        test_used_for_inference_or_training=False,optimizer_updates=0,controller_training_started=False,
        archived_realignment_sha256=file_hash(base/'shared/realignment.safetensors'))
    if dry_run:
        print(json.dumps({k:plan[k] for k in ('new_questions','new_cohorts','expected_new_B_completions','estimated_new_bytes','length_audit')},indent=2))
        return plan
    out.mkdir()
    write_json(out/'plan.json',plan)
    for name in sources:
        atomic_exclusive_bytes(out/'source_snapshot'/name,contained(ROOT/name,ROOT).read_bytes())
    write_json(out/'selection_summary.json',dict(total_train=1024,new=768,validation=128,
        source_ids=selection['source_ids'],seed=42,heldout_overlap=False,selection_by_outcomes=False))
    print(json.dumps(dict(status='prepared',output=str(out),plan_sha256=file_hash(out/'plan.json'),
        new_questions=len(new),B_calls=plan['expected_new_B_completions'],GPU_initialized=torch.cuda.is_initialized())))
    return plan


def load_plan(root):
    root=contained(root,ROOT)
    plan=json.loads((root/'plan.json').read_text())
    if plan['schema']!=SCHEMA or plan['output']!=str(root):
        raise ValueError('Wrong campaign')
    contained(plan['config']['data_dir'],ROOT)
    if (len(plan['new_source_ids'])!=768 or len(plan['selection']['source_ids'])!=1024
            or plan['selection']['source_ids']!=plan['inherited_train_ids']+plan['new_source_ids']
            or len(set(plan['selection']['source_ids']))!=1024
            or plan['request_seeds']!=list(SEEDS)):
        raise ValueError('Expansion selection changed')
    verify_hashes(ROOT,plan['source_sha256'])
    verify_hashes(root/'source_snapshot',plan['source_sha256'])
    if runtime_fingerprint()!=plan['runtime']:
        raise ValueError('Runtime packages changed')
    base=contained(plan['parent_profile'],ROOT)
    if file_hash(base/'manifest.json')!=plan['parent_manifest_sha256']:
        raise ValueError('Parent manifest changed')
    repeats=contained(plan['parent_repeats'],ROOT)
    if file_hash(repeats/'plan.json')!=plan['parent_repeat_evidence']['plan_sha256']:
        raise ValueError('Parent repeat plan changed')
    if file_hash(repeats/'finished.json')!=plan['parent_repeat_evidence']['finished_sha256']:
        raise ValueError('Parent repeat finish changed')
    return plan


def campaign_items(plan):
    pool,meta=load_official_medqa(plan['config']['data_dir'],'train')
    if meta['sha256']!=plan['selection']['dataset_sha256']:
        raise ValueError('Official train bytes changed')
    from controller.matched import historical_style
    lookup={x['source_id']:x for x in pool}
    return [dict(lookup[sid],question=historical_style(lookup[sid])) for sid in plan['new_source_ids']]


def teacher(root,g):
    path=completed(root/'new_train/teachers'/f'cohort{g:03d}')
    return None if path is None else contained(path,ROOT)


def ensure_teacher(wrapper,args,root,plan,items,g):
    path=teacher(root,g)
    ids=plan['new_source_ids'][g*16:(g+1)*16]
    if path is None:
        count=len(list((root/'new_train/teachers').glob('cohort*/complete.json')))
        needed=(plan['new_cohorts']-count)*plan['teacher_bytes_per_cohort']+plan['disk_reserve_bytes']
        disk_guard(ROOT,needed)
        base=root/'new_train/teachers'/f'cohort{g:03d}'
        path=new_attempt(base)
        event(root,stage='A_capture_start',cohort=g,new_questions=plan['new_questions'])
        capture(wrapper,args,items[g*16:(g+1)*16],path,'train')
        commit(base,path)
        event(root,stage='A_capture_complete',cohort=g,archive_bytes=sum(p.stat().st_size for p in path.iterdir() if p.is_file()))
    meta=json.loads((path/'cohort.json').read_text())
    if meta['source_ids']!=ids or meta['split']!='train':
        raise ValueError('Teacher cohort changed')
    return path


def progress(paths):
    existing=[completed(p) is not None for p in paths]
    first=next((n for n,yes in enumerate(existing) if not yes),len(paths))
    if any(existing[first:]):
        raise ValueError('Nonconsecutive commits')
    return first


@torch.no_grad()
def decode_general(engine,payloads,items,k,seed):
    from vllm import SamplingParams
    if (seed not in SEEDS or len(items)!=16 or len(payloads)!=16
            or any(i['source_split']!='train' or ':train:' not in i['source_id'] for i in items)):
        raise ValueError('Expected approved seed and batch16')
    started=time.perf_counter()
    outputs=engine.generate(payloads,SamplingParams(temperature=.6,top_p=.95,top_k=0,seed=seed,max_tokens=4096),use_tqdm=False)
    if len(outputs)!=16: raise RuntimeError('Incomplete B cohort')
    rows=[]
    for payload,item,out in zip(payloads,items,outputs):
        generated=out.outputs[0];ids=list(generated.token_ids)
        if not 0<=len(ids)<=4096: raise ValueError('Invalid B length')
        metric=score_text(generated.text,item['gold']);strict=metric['medqa_boxed_v1']
        rows.append(dict(source_id=item['source_id'],source_split='train',k=k,gold=item['gold'],
            raw_prediction=generated.text,generated_token_ids=ids,output_tokens=len(ids),
            finish_reason=generated.finish_reason,stop_reason=generated.stop_reason,
            strict_prediction=strict['prediction'],strict_correct=strict['correct'],strict_parse_failed=strict['parse_failed'],
            legacy_prediction=metric['legacy_prediction'],legacy_correct=metric['legacy_correct'],
            repetition=repetition(ids),temperature=.6,top_p=.95,top_k=0,decoder_seed=seed,engine_seed=42,
            batch=16,prompt_sha256=tensor_hash(payload['prompt_embeds']),execution_domain='H100_new768',
            cached_tokens=getattr(out,'num_cached_tokens',None)))
    return rows,time.perf_counter()-started


def run_sequence(wrapper,args,root,plan,items,seed,k):
    bases=[root/f'seed{seed}/primary/train/K{k}/cohort{g:03d}' for g in range(plan['new_cohorts'])]
    first=progress(bases)
    if first==len(bases): return
    reset_cache(wrapper)
    for g,base in enumerate(bases):
        disk_guard(ROOT,plan['runtime_disk_min_bytes'])
        if seed==42 and k==10:
            tp=ensure_teacher(wrapper,args,root,plan,items,g)
        else:
            tp=teacher(root,g)
            if tp is None: raise ValueError('Missing committed A; never regenerate implicitly')
        t=load_file(str(tp/'teacher.safetensors'))
        reconstruction=load_file(str(tp/'reconstruction.safetensors'))
        payloads=reconstruct_payloads(t,reconstruction,k)
        for payload in payloads:
            if len(payload['prompt_embeds'])+4096>args.vllm_max_model_len:
                raise ValueError('No context truncation permitted')
            payload['prompt_embeds']=payload['prompt_embeds'].to(wrapper.HF_device)
        replay=g<first
        target=root/f'seed{seed}/cache_replay/train/K{k}/cohort{g:03d}' if replay else base
        path=new_attempt(target)
        event(root,stage='B_start',seed=seed,k=k,cohort=g,cache_replay=replay)
        rows,seconds=decode_general(wrapper.vllm_engine,payloads,items[g*16:(g+1)*16],k,seed)
        write_jsonl(path/'outcomes.jsonl',rows)
        write_json(path/'summary.json',dict(seconds=seconds,seed=seed,k=k,cohort=g,
            correct=sum(r['strict_correct'] for r in rows),main_completions=0 if replay else 16,
            cache_replay=replay,cohort_order=plan['new_source_ids'][g*16:(g+1)*16]))
        if replay: write_json(path/'replay_complete.json',dict(main_completions=0,replay_completions=16))
        else: commit(base,path)
        event(root,stage='B_complete',seed=seed,k=k,cohort=g,cache_replay=replay,
            main_B_completions=0 if replay else 16,seconds=seconds,correct=sum(r['strict_correct'] for r in rows))
        del t,reconstruction,payloads
    verify_hashes(ROOT,plan['source_sha256'])


def verify_seed(root,plan,seed):
    """Independent raw-state/feature/payload/strict-score checks, without GPU."""
    items=campaign_items(plan);correct=0;count=0
    for g in range(plan['new_cohorts']):
        tp=teacher(root,g)
        if tp is None: raise ValueError('Missing teacher')
        t=load_file(str(tp/'teacher.safetensors'))
        grid=load_file(str(tp/'feature_grid.safetensors'))
        rec=load_file(str(tp/'reconstruction.safetensors'))
        texts=json.loads((tp/'prompts.json').read_text())
        if (t['hidden'].shape!=(16,81,37,2560) or t['latent'].shape!=(16,80,2560)
                or t['hidden'].dtype!=torch.bfloat16 or t['latent'].dtype!=torch.bfloat16
                or not torch.isfinite(t['hidden']).all() or not torch.isfinite(t['latent']).all()
                or grid['features'].shape!=(16,8,7685)):
            raise ValueError('Invalid teacher shape/precision')
        cohort=items[g*16:(g+1)*16]
        meta=json.loads((tp/'cohort.json').read_text())
        if meta['source_ids']!=[i['source_id'] for i in cohort] or meta['split']!='train':
            raise ValueError('Teacher source identities differ')
        for i,item in enumerate(cohort):
            if item['question_stem'] not in texts['A'][i] or item['question_stem'] not in texts['B'][i]:
                raise ValueError('Prompt/source mismatch')
            h=t['hidden'][i,:,-1].float()
            n=int(t['A_mask'][i].sum())
            if not torch.equal(t[f'prefix_{i}'][n:],t['latent'][i]):
                raise ValueError('Actual latent-prefix mismatch')
            for j,k in enumerate(GRID):
                x=grid['features'][i,j]
                torch.testing.assert_close(x[-5:-1],reference_phi(h,k),atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(grid['phi4'][i,j],x[-5:-1],atol=1e-6,rtol=1e-5)
                if (not torch.equal(x[:2560],h[k]) or not torch.equal(x[2560:5120],h[0])
                        or not torch.equal(x[5120:7680],h[k]-h[k-10]) or x[-1]!=k/80):
                    raise ValueError('Noncausal/changed raw observation')
        for k in GRID:
            path=completed(root/f'seed{seed}/primary/train/K{k}/cohort{g:03d}')
            if path is None: raise ValueError('Missing seed commit')
            rows=read_rows(contained(path/'outcomes.jsonl',ROOT))
            payloads=reconstruct_payloads(t,rec,k)
            if len(rows)!=16 or [r['source_id'] for r in rows]!=[x['source_id'] for x in cohort]:
                raise ValueError('Incomplete/reordered B cohort')
            for row,item,payload in zip(rows,cohort,payloads):
                if (row['source_split']!='train' or row['k']!=k or row['gold']!=item['gold']
                        or row['decoder_seed']!=seed or row['engine_seed']!=42 or row['batch']!=16
                        or row['temperature']!=.6 or row['top_p']!=.95 or row['top_k']!=0
                        or row['prompt_sha256']!=tensor_hash(payload['prompt_embeds'])
                        or row['output_tokens']!=len(row['generated_token_ids']) or row['output_tokens']>4096):
                    raise ValueError('B identity/protocol/payload mismatch')
                strict=score_text(row['raw_prediction'],item['gold'])['medqa_boxed_v1']
                if any(row[a]!=strict[b] for a,b in [('strict_correct','correct'),('strict_prediction','prediction'),('strict_parse_failed','parse_failed')]):
                    raise ValueError('Independent strict score mismatch')
                count+=1;correct+=row['strict_correct']
    if count!=plan['expected_B_completions_per_seed']: raise ValueError('Wrong count')
    parity=json.loads((root/'new_train_prefix_parity.json').read_text())
    if not parity['passed']: raise ValueError('New A prefix parity failed')
    return dict(status='complete',verified=True,seed=seed,main_B_completions=count,correct=correct,
                test_used=False,optimizer_updates=0,plan_sha256=file_hash(root/'plan.json'))


def run_seed(root,seed,resume=False):
    plan=load_plan(root)
    if seed not in SEEDS: raise ValueError('Unsupported seed')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=2:
        raise ValueError('Require two explicitly selected GPUs')
    if (root/f'seed{seed}').exists() and not resume:
        raise ValueError('Existing seed directory requires deliberate --resume')
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/f'seed{seed}/finished.json').exists():
            print(json.dumps(verify_seed(root,plan,seed)));return
        from models import ModelWrapper
        from utils import set_seed
        set_seed(42)
        args=SimpleNamespace(**plan['config'])
        items=campaign_items(plan)
        wrapper=ModelWrapper(args.model_name,torch.device(args.device),use_vllm=True,args=args)
        keep_last_logits(wrapper.HF_model)
        workers=wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
        if any(w['head_dtype']!='torch.bfloat16' or w['embedding_dtype']!='torch.bfloat16'
               or w['quantization'] is not None or w['model_revision']!=args.model_revision for w in workers):
            raise ValueError('B precision/revision mismatch')
        if wrapper.HF_model.config._commit_hash!=args.model_revision or next(wrapper.HF_model.parameters()).dtype!=torch.bfloat16:
            raise ValueError('A precision/revision mismatch')
        realign=bind_archived_realignment(wrapper,plan['parent_profile'],plan['archived_realignment_sha256'])
        write_json(root/f'seed{seed}/preflight_{time.time_ns()}.json',dict(workers=workers,
            hf_dtype='torch.bfloat16',hf_logits_to_keep=1,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            runtime=runtime_fingerprint(),**realign))
        if seed==42:
            tp=ensure_teacher(wrapper,args,root,plan,items,0)
            parity=root/'new_train_prefix_parity.json'
            if not parity.exists(): check_parity(wrapper,args,tp,parity)
        elif not (root/'seed42/finished.json').exists():
            raise ValueError('Seed42 must finish/verify first')
        for k in GRID: run_sequence(wrapper,args,root,plan,items,seed,k)
        result=verify_seed(root,plan,seed)
        write_json(root/f'seed{seed}/finished.json',result)
        event(root,stage='seed_complete',**{k:v for k,v in result.items() if k!='status'})


def samples_from_rows(rows,ids):
    lookup={sid:i for i,sid in enumerate(ids)}
    samples=torch.full((len(ids),8,3),-1,dtype=torch.int8)
    seen=set()
    for row in rows:
        if row['source_id'] not in lookup: continue
        key=(row['source_id'],row['k'])
        if key in seen or row['k'] not in GRID or row['seeds']!=list(SEEDS):
            raise ValueError('Duplicate/invalid parent aggregate')
        if len(row['correctness'])!=3 or any(x not in (0,1) for x in row['correctness']):
            raise ValueError('Invalid parent seed values')
        seen.add(key)
        samples[lookup[row['source_id']],list(GRID).index(row['k'])]=torch.tensor(row['correctness'],dtype=torch.int8)
    if (samples<0).any(): raise ValueError('Missing aggregate grid')
    return samples


def finalize(root):
    plan=load_plan(root)
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for seed in SEEDS:
            saved=json.loads((root/f'seed{seed}/finished.json').read_text())
            if not saved.get('verified') or saved['plan_sha256']!=file_hash(root/'plan.json'):
                raise ValueError('Require verified seeds')
            verify_seed(root,plan,seed)
        base=Path(plan['parent_profile'])
        pm=json.loads((base/'manifest.json').read_text())
        repeat_evidence(ROOT,base,pm,plan['parent_repeats'])
        parent_rows=read_rows(Path(plan['parent_repeats'])/'per_question_K_successes.jsonl')
        parent_train=completed(base/'train/dataset')
        parent_valid=completed(base/'validation/dataset')
        old_x=load_file(str(parent_train/'dataset.safetensors'))['features']
        for split,p in (('train',parent_train),('validation',parent_valid)):
            expected_ids=pm['selections'][split]['source_ids']
            meta=json.loads((p/'manifest.json').read_text())
            if meta['source_ids']!=expected_ids or meta['split']!=split or meta['allowed_for_gradient_updates']!=(split=='train'):
                raise ValueError('Inherited dataset split/identity mismatch')
            x=load_file(str(p/'dataset.safetensors'))['features']
            for g in range(len(expected_ids)//16):
                tp=completed(base/split/'teachers'/f'cohort{g:03d}')
                expected=load_file(str(tp/'feature_grid.safetensors'))['features'][:,:7]
                if not torch.equal(x[g*16:(g+1)*16],expected):
                    raise ValueError('Inherited raw features differ from teacher archive')
        new_x=[];q=torch.zeros((plan['new_questions'],8,3),dtype=torch.int8)
        for g in range(plan['new_cohorts']):
            tp=teacher(root,g)
            new_x.append(load_file(str(tp/'feature_grid.safetensors'))['features'][:,:7])
            for j,k in enumerate(GRID):
                for s,seed in enumerate(SEEDS):
                    p=completed(root/f'seed{seed}/primary/train/K{k}/cohort{g:03d}')
                    q[g*16:(g+1)*16,j,s]=torch.tensor([r['strict_correct'] for r in read_rows(p/'outcomes.jsonl')],dtype=torch.int8)
        features=torch.cat([old_x,*new_x])
        samples=torch.cat([samples_from_rows(parent_rows,plan['inherited_train_ids']),q])
        datasets={
            'train':tensors_from_samples(features,samples,'train'),
            'validation':tensors_from_samples(load_file(str(parent_valid/'dataset.safetensors'))['features'],
                samples_from_rows(parent_rows,plan['validation_ids']),'validation')}
        exports={}
        for split,data in datasets.items():
            path=completed(root/split/'dataset')
            if path is None:
                path=new_attempt(root/split/'dataset')
                write_tensors(path/'dataset.safetensors',data)
                ids=plan['selection']['source_ids'] if split=='train' else plan['validation_ids']
                write_json(path/'manifest.json',dict(schema=SCHEMA,split=split,source_ids=ids,
                    questions=len(ids),features_shape=list(data['features'].shape),
                    seeds=list(SEEDS),correctness='mean strict correctness across three B seeds; no any-of-three reward',
                    allowed_for_gradient_updates=split=='train',features_are_raw=True,
                    plan_sha256=file_hash(root/'plan.json'),file_sha256=file_hash(path/'dataset.safetensors')))
                commit(root/split/'dataset',path)
            saved=load_file(str(path/'dataset.safetensors'))
            if set(saved)!=set(data) or any(not torch.equal(saved[k],v) for k,v in data.items()):
                raise ValueError('Existing expanded export differs from verified data')
            exports[split]=dict(path=str(path),questions=len(data['features']),sha256=file_hash(path/'dataset.safetensors'))
        finish=dict(status='complete',verified=True,train=1024,validation=128,new_questions=768,
            new_B_completions=18432,total_including_inherited_B=27648,
            exports=exports,plan_sha256=file_hash(root/'plan.json'),optimizer_updates=0,
            controller_training_started=False,test_used=False,original_archives_rewritten=False)
        if (root/'finished.json').exists():
            if json.loads((root/'finished.json').read_text())!=finish:
                raise ValueError('Existing finish marker differs')
        else: write_json(root/'finished.json',finish)
        print(json.dumps(finish,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','verify','finalize'))
    p.add_argument('--config',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.action=='prepare':
        if args.config is None: p.error('--config required')
        prepare(args.config,args.dry_run);return
    if args.out is None: p.error('--out required')
    root=contained(args.out,ROOT)
    if args.action=='run':
        if args.seed is None: p.error('--seed required')
        run_seed(root,args.seed,args.resume)
    elif args.action=='verify':
        if args.seed is None: p.error('--seed required')
        print(json.dumps(verify_seed(root,load_plan(root),args.seed),indent=2))
    else: finalize(root)


if __name__=='__main__':
    main()
