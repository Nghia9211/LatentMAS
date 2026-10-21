#!/usr/bin/env python3
"""Historical-style temp0.6/batch16 live-reward PPO, strictly official train32.

Stages prepare/profile/train/evaluate are separate. Immutable attempts/checkpoints;
--resume continues only committed iterations and unchanged source/protocol.
"""
import argparse
import fcntl
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import yaml
from safetensors.torch import load_file
from controller.profile import (file_hash,tensor_hash,write_json,write_jsonl,write_tensors,
    atomic_exclusive_bytes,verify_file_manifest,raw_observation,tie_targets)
from controller.matched import (train_items,test_items,render_batch,reference_prompt_audit,
    CohortDecisions,online_rollout,prefixes_for_b,require_train)
from controller.warmstart import WarmStartController,check_gradients
from controller.ppo import PPOController,ppo_loss
from latent_batch import rollout

SOURCES=['controller/__init__.py','controller/training.py','controller/matched.py','controller/ppo.py','controller/warmstart.py','controller/profile.py',
    'latent_batch.py','models.py','methods/latent_mas.py','prompts.py','utils.py',
    'scripts/run_matched_controller_ppo.py','configs/controller_ppo_matched.yaml',
    'scripts/audit_saved_loop_budgets.py','tests/test_controller_matched.py']


def event(root,**fields):
    row=dict(utc=datetime.now(timezone.utc).isoformat(),**fields)
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
    print(json.dumps(row),flush=True)


def new_attempt(base):
    i=0
    while (base/f'attempt_{i:03d}').exists():i+=1
    path=base/f'attempt_{i:03d}';path.mkdir(parents=True);return path


def completed(base):
    marker=base/'complete.json'
    if not marker.exists():return None
    m=json.loads(marker.read_text());path=base/m['attempt'];verify_file_manifest(path,m['checksums']);return path


def commit(base,path):
    write_json(base/'complete.json',dict(attempt=path.name,checksums={str(p.relative_to(path)):file_hash(p) for p in path.rglob('*') if p.is_file()}))


def reset_cache(wrapper):
    if not wrapper.vllm_engine.reset_prefix_cache():raise RuntimeError('Cache reset failed')




@torch.no_grad()
def decode_cohort(wrapper,args,items,payloads,budgets,root,path,purpose):
    from vllm import SamplingParams
    from utils import extract_gsm8k_answer,normalize_answer,medqa_scoring_metadata
    from scripts.audit_saved_loop_budgets import repetition
    if len(items)!=16 or len(payloads)!=16:raise ValueError('No partial/cohort regrouping allowed')
    if purpose in ('live_train_reward','train_warmstart_grid','cache_reconstruction') and any(i['source_split']!='train' for i in items):
        raise ValueError('Held-out data cannot produce training reward')
    params=SamplingParams(temperature=.6,top_p=.95,top_k=0,seed=42,max_tokens=4096)
    event(root,stage='B_cohort_start',purpose=purpose,split=items[0]['source_split'],budgets=budgets)
    started=time.perf_counter();outputs=wrapper.vllm_engine.generate(payloads,params,use_tqdm=False)
    seconds=time.perf_counter()-started
    if len(outputs)!=16:raise RuntimeError('Incomplete B cohort; do not fabricate rewards')
    rows=[]
    for item,payload,out,k in zip(items,payloads,outputs,budgets):
        output=out.outputs[0];ids=list(output.token_ids)
        pred=normalize_answer(extract_gsm8k_answer(output.text))
        metric=medqa_scoring_metadata(output.text,item['gold'],pred,pred==item['gold'])
        strict=metric['medqa_boxed_v1']
        rows.append(dict(source_id=item['source_id'],source_split=item['source_split'],k=k,gold=item['gold'],
            raw_prediction=output.text,generated_token_ids=ids,output_tokens=len(ids),
            finish_reason=output.finish_reason,stop_reason=output.stop_reason,
            strict_prediction=strict['prediction'],strict_correct=strict['correct'],strict_parse_failed=strict['parse_failed'],
            legacy_prediction=pred,legacy_correct=metric['legacy_correct'],repetition=repetition(ids),
            temperature=.6,decoder_seed=42,batch=16,prompt_sha256=tensor_hash(payload['prompt_embeds']),
            cached_tokens=getattr(out,'num_cached_tokens',None)))
    write_jsonl(path/'outcomes.jsonl',rows)
    event(root,stage='B_cohort_complete',purpose=purpose,seconds=seconds,correct=sum(r['strict_correct'] for r in rows),
        tokens=sum(r['output_tokens'] for r in rows),caps=sum(r['finish_reason']=='length' for r in rows))
    return rows,seconds


@torch.no_grad()
def capture_teacher(wrapper,args,items,path):
    texts,encoded=render_batch(wrapper,args,items);states=[];latent=[]
    def observer(**kw):
        states.append(kw['hidden_last_position'].detach().cpu().clone())
        if kw['step']:latent.append(kw['latent_input'].detach().cpu().clone())
    ids=encoded['A']['input_ids'].to(wrapper.HF_device);mask=encoded['A']['attention_mask'].to(wrapper.HF_device)
    past,prefixes=rollout(wrapper,ids,mask,latent_steps=80,observer=observer);del past
    tensors=dict(hidden=torch.stack(states,dim=1),latent=torch.stack(latent,dim=1))
    if tensors['hidden'].shape[:3]!=(16,81,37) or tensors['latent'].shape!=(16,80,2560) or any(v.dtype!=torch.bfloat16 or not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError('Rich trajectory shape/dtype/finiteness mismatch')
    for i,prefix in enumerate(prefixes):tensors[f'prefix_{i}']=prefix.cpu()
    for key in ('A','B'):
        tensors[key+'_ids']=encoded[key]['input_ids'];tensors[key+'_mask']=encoded[key]['attention_mask']
    write_tensors(path/'teacher.safetensors',tensors);write_json(path/'prompts.json',texts)
    return tensors,texts,encoded


def read_teacher(path):
    t=load_file(str(path/'teacher.safetensors'));texts=json.loads((path/'prompts.json').read_text())
    encoded={key:dict(input_ids=t[key+'_ids'],attention_mask=t[key+'_mask']) for key in ('A','B')}
    return t,texts,encoded


def profile(wrapper,args,items,root):
    require_train(items)
    teachers=[]
    for group in range(2):
        base=root/'profile'/f'cohort{group}';path=completed(base)
        if path is None:
            path=new_attempt(base);event(root,stage='A_full_grid_capture',cohort=group,split='train')
            capture_teacher(wrapper,args,items[group*16:(group+1)*16],path);commit(base,path)
        teachers.append(path)
    # Parity on train only: full80 slicing vs production fixedK10.
    if not (root/'train_prefix_parity.json').exists():
        t,_,e=read_teacher(teachers[0]);past,prefixes=rollout(wrapper,e['A']['input_ids'].to(args.device2),
            e['A']['attention_mask'].to(args.device2),latent_steps=10);del past
        for i,p in enumerate(prefixes):
            expected=t[f'prefix_{i}'][:int(e['A']['attention_mask'][i].sum())+10]
            if not torch.equal(p.cpu(),expected):raise RuntimeError('Batch16 prefix parity failed')
        write_json(root/'train_prefix_parity.json',dict(passed=True,cases=16,train_only=True,fixed_K10_exact=True))
    for k in range(10,81,10):
        reset_cache(wrapper)  # Each K is a fresh historical-style full-run sequence.
        for group in range(2):
            base=root/'profile'/f'K{k}'/f'cohort{group}';existing=completed(base)
            t,texts,e=read_teacher(teachers[group]);cohort=items[group*16:(group+1)*16]
            prefixes=[t[f'prefix_{i}'][:int(e['A']['attention_mask'][i].sum())+k].to(args.device2) for i in range(16)]
            payloads=prefixes_for_b(wrapper,texts,e,prefixes)
            # Resume replay prior committed cohorts to reconstruct warm cache,
            # but archive/flag extra generation calls instead of pretending none.
            if existing is not None:
                if group==0 and completed(root/'profile'/f'K{k}'/'cohort1') is None:
                    path=new_attempt(root/'cache_reconstruction'/f'profileK{k}');decode_cohort(wrapper,args,cohort,payloads,[k]*16,root,path,'cache_reconstruction')
                continue
            path=new_attempt(base);decode_cohort(wrapper,args,cohort,payloads,[k]*16,root,path,'train_warmstart_grid');commit(base,path)
    x=[];correctness=[]
    for group in range(2):
        t,_,_=read_teacher(teachers[group]);h=t['hidden'][:,:,-1]
        x.extend(torch.stack([raw_observation(h[i],k) for k in range(10,80,10)]) for i in range(16))
        outcomes=[]
        for k in range(10,81,10):
            path=completed(root/'profile'/f'K{k}'/f'cohort{group}')
            rows=[json.loads(line) for line in (path/'outcomes.jsonl').read_text().splitlines()]
            if [r['source_id'] for r in rows]!=[i['source_id'] for i in items[group*16:(group+1)*16]] or any(r['source_split']!='train' for r in rows):raise ValueError('Profile source mismatch')
            outcomes.append([float(r['strict_correct']) for r in rows])
        correctness.extend(torch.tensor(outcomes).T)
    x=torch.stack(x);q=torch.stack(correctness)
    from controller.warmstart import supervision
    actions,targets=zip(*(supervision(row) for row in q))
    if not (root/'warm_dataset.safetensors').exists():
        write_tensors(root/'warm_dataset.safetensors',dict(features=x,actions=torch.stack(actions),q_targets=torch.stack(targets),correctness=q))
    if not (root/'warm_dataset_manifest.json').exists():
        write_json(root/'warm_dataset_manifest.json',dict(file_sha256=file_hash(root/'warm_dataset.safetensors'),
            source_ids=[i['source_id'] for i in items],split='train',temperature=.6,batch16=True,seed=42,
            observations=['h_k','h0','delta_checkpoint','phi4','k/80'],label_only='Observed fixed-cohort counterfactual correctness, not policy inputs'))
    return x,torch.stack(actions),torch.stack(targets)


def save_checkpoint(path,model,optimizer,iteration):
    payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
        iteration=iteration,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    buf=io.BytesIO();torch.save(payload,buf);atomic_exclusive_bytes(path,buf.getvalue())


def ppo_update(model,optimizer,x,episode,teacher_x,ties,cfg):
    from torch.nn import functional as F
    pieces=[];local=dict(cfg,tie_weight=0)
    for _ in range(cfg['optimization_epochs']):
        order=torch.randperm(32,device=x.device)
        for start in range(0,32,cfg['minibatch_questions']):
            index=order[start:start+cfg['minibatch_questions']]
            data={k:v[index] for k,v in episode.items()}
            optimizer.zero_grad(set_to_none=True)
            loss,parts=ppo_loss(model,x[index],data,torch.zeros_like(ties[index]),local)
            logits,_=model(teacher_x[index]);mask=ties[index]
            tie=F.cross_entropy(logits[mask],torch.zeros(int(mask.sum()),dtype=torch.long,device=x.device)) if mask.any() else logits.sum()*0
            loss=loss+cfg['tie_weight']*tie
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite PPO loss')
            loss.backward()
            if not check_gradients(model):raise FloatingPointError('Zero PPO gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True);optimizer.step()
            pieces.append(float(loss.detach()))
    return sum(pieces)/len(pieces),len(pieces)


def train(wrapper,args,items,root,cfg,arithmetic):
    require_train(items)
    dataset_manifest=json.loads((root/'warm_dataset_manifest.json').read_text())
    if dataset_manifest['split']!='train' or dataset_manifest['source_ids']!=[i['source_id'] for i in items] or file_hash(root/'warm_dataset.safetensors')!=dataset_manifest['file_sha256']:
        raise ValueError('Warmstart train-data provenance/hash mismatch')
    if arithmetic not in ('preserve','mask'):raise ValueError('Explicit stopping arithmetic choice required')
    lock=dict(arithmetic=arithmetic,iterations=200,split='train',source_ids=[i['source_id'] for i in items],
        reward='Actual mixed-budget B cohort sampling0.6,seed42,batch16,not cached greedy correctness')
    if (root/'training_lock.json').exists():
        if json.loads((root/'training_lock.json').read_text())!=lock:raise ValueError('Training lock differs')
    else:write_json(root/'training_lock.json',lock)
    parity=root/f'mixed_prefix_parity_{arithmetic}.json'
    if not parity.exists():
        path=completed(root/'profile'/'cohort0');t,_,e=read_teacher(path)
        class Forced:
            budgets=torch.tensor(list(range(10,81,10))*2,device=args.device2)
            def begin(self,h):pass
            def step(self,k,h,active):return active&(self.budgets==k)
        past,prefixes,physical=online_rollout(wrapper,e['A']['input_ids'].to(args.device2),
            e['A']['attention_mask'].to(args.device2),Forced(),preserve_arithmetic=arithmetic=='preserve');del past
        rows=[]
        for i,prefix in enumerate(prefixes):
            k=int(Forced.budgets[i]);target=t[f'prefix_{i}'][:int(e['A']['attention_mask'][i].sum())+k]
            exact=torch.equal(prefix.cpu(),target)
            rows.append(dict(row=i,k=k,exact=exact,max_abs_diff=float((prefix.cpu().float()-target.float()).abs().max())))
        write_json(parity,dict(passed=all(r['exact'] for r in rows),arithmetic=arithmetic,rows=rows,physical_steps=physical,train_only=True))
    if not json.loads(parity.read_text())['passed']:
        raise RuntimeError('Mixed-budget A prefix differs from historical-style fixed arithmetic; HOLD before training')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    d=load_file(str(root/'warm_dataset.safetensors'))
    teacher_x=d['features'].to(args.device);actions=d['actions'].to(args.device);targets=d['q_targets'].to(args.device)
    ties=targets[:,:,0]==targets[:,:,1]
    model=PPOController(teacher_x.shape[-1],cfg['width']).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0)
    warmpath=root/'initial_checkpoint.pt'
    if not warmpath.exists():
        from controller.training import train_update
        warm=WarmStartController('raw_mlp',teacher_x.shape[-1],cfg['width']).to(args.device)
        opt=torch.optim.AdamW(warm.parameters(),lr=cfg['learning_rate'],weight_decay=0)
        for _ in range(cfg['warmstart_epochs']):train_update(warm,opt,teacher_x,actions,targets,dict(q_weight=1,grad_clip=1))
        model.load_warmstart(warm);del warm,opt
        save_checkpoint(warmpath,model,optimizer,0)
    last=0;checkpoint=warmpath
    for n in range(1,201):
        path=completed(root/'train'/f'iteration{n:03d}')
        if path is None:break
        last=n;checkpoint=path/'checkpoint.pt'
    saved=torch.load(checkpoint,map_location=args.device,weights_only=True)
    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    for n in range(last+1,201):
        path=new_attempt(root/'train'/f'iteration{n:03d}');started=time.perf_counter();reset_cache(wrapper)
        fields={k:[] for k in ('features','actions','old_logprobs','old_values','visited')};rewards=[];budgets=[];physical=[]
        # The same pi_old stays unchanged across both cohorts until all rewards arrive.
        for group in range(2):
            cohort=items[group*16:(group+1)*16];texts,e=render_batch(wrapper,args,cohort)
            decisions=CohortDecisions(model,16,args.device,True)
            past,prefixes,physical_steps=online_rollout(wrapper,e['A']['input_ids'].to(args.device2),
                e['A']['attention_mask'].to(args.device2),decisions,preserve_arithmetic=arithmetic=='preserve');del past
            payloads=prefixes_for_b(wrapper,texts,e,prefixes);sub=path/f'cohort{group}';sub.mkdir()
            write_json(sub/'prompts.json',texts)
            write_tensors(sub/'observations.safetensors',dict(features=decisions.x,visited=decisions.visited,
                **{f'observed_h_{i}':torch.stack(h) for i,h in enumerate(decisions.histories)}))
            ks=decisions.budgets.cpu().tolist()
            rows,_=decode_cohort(wrapper,args,cohort,payloads,ks,root,sub,'live_train_reward')
            rewards.extend(float(r['strict_correct']) for r in rows);budgets.extend(ks);physical.append(physical_steps)
            for key,value in [('features',decisions.x),('actions',decisions.actions),('old_logprobs',decisions.logprobs),('old_values',decisions.values),('visited',decisions.visited)]:fields[key].append(value.detach())
            del payloads,prefixes,decisions
        data={k:torch.cat(v) for k,v in fields.items()};r=torch.tensor(rewards,device=args.device)
        episode={k:data[k] for k in ('actions','old_logprobs','old_values','visited')}
        episode['returns']=r[:,None].expand(32,7).clone();episode['advantages']=(episode['returns']-episode['old_values'])*episode['visited']
        write_tensors(path/'episodes.safetensors',dict(**data,rewards=r,returns=episode['returns'],advantages=episode['advantages']))
        loss,updates=ppo_update(model,optimizer,data['features'],episode,teacher_x,ties,cfg)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        write_json(path/'result.json',dict(iteration=n,train_only=True,correct=int(r.sum()),episodes=32,
            mean_selected_K=sum(budgets)/32,cohort_physical_steps=physical,loss=loss,optimizer_updates=updates,
            seconds_including_live_A_B=seconds,actual_B_completions=32,checkpoint_selected_by_test=False))
        save_checkpoint(path/'checkpoint.pt',model,optimizer,n);commit(root/'train'/f'iteration{n:03d}',path)
        event(root,stage='PPO_iteration_complete',iteration=n,total=200,correct=int(r.sum()),seconds=seconds,mean_K=sum(budgets)/32)
    if not (root/'final_controller.safetensors').exists():write_tensors(root/'final_controller.safetensors',model.state_dict())
    if not (root/'training_finished.json').exists():write_json(root/'training_finished.json',dict(status='complete',iterations=200,
        train_episodes=6400,train_split='train',test_used=False,checkpoint_sha256=file_hash(root/'final_controller.safetensors'),arithmetic=arithmetic))
    event(root,stage='train_complete',iterations=200,train_episodes=6400)


def evaluate(wrapper,args,root,cfg):
    finish=json.loads((root/'training_finished.json').read_text())
    items=test_items(args.data_dir,args.reference)
    model=PPOController(7685,cfg['width']).to(args.device);model.load_state_dict(load_file(str(root/'final_controller.safetensors')));model.eval()
    if file_hash(root/'final_controller.safetensors')!=finish['checkpoint_sha256']:raise ValueError('Final checkpoint changed')
    texts,e=render_batch(wrapper,args,items);reports={}
    for mode in ('PPO','K10','K20','K80'):
        base=root/'evaluation'/mode;path=completed(base)
        if path is None:
            path=new_attempt(base);reset_cache(wrapper)
            if mode=='PPO':
                decisions=CohortDecisions(model,16,args.device,False)
                past,prefixes,physical=online_rollout(wrapper,e['A']['input_ids'].to(args.device2),e['A']['attention_mask'].to(args.device2),decisions,preserve_arithmetic=finish['arithmetic']=='preserve')
                budgets=decisions.budgets.cpu().tolist()
                write_tensors(path/'controller_observations.safetensors',dict(features=decisions.x,visited=decisions.visited))
            else:
                k=int(mode[1:]);past,prefixes=rollout(wrapper,e['A']['input_ids'].to(args.device2),e['A']['attention_mask'].to(args.device2),latent_steps=k);budgets=[k]*16;physical=k
            del past
            payloads=prefixes_for_b(wrapper,texts,e,prefixes)
            write_json(path/'prompts.json',texts)
            rows,seconds=decode_cohort(wrapper,args,items,payloads,budgets,root,path,'heldout_'+mode)
            write_json(path/'summary.json',dict(mode=mode,samples=16,strict_accuracy=sum(r['strict_correct'] for r in rows)/16,
                correct=sum(r['strict_correct'] for r in rows),mean_K=sum(budgets)/16,cohort_physical_A_steps=physical,
                mean_B_tokens=sum(r['output_tokens'] for r in rows)/16,caps=sum(r['finish_reason']=='length' for r in rows),
                loops=sum(r['repetition']['loop_flag'] for r in rows),B_cohort_seconds=seconds,test_used_for_training=False))
            commit(base,path)
        reports[mode]=json.loads((path/'summary.json').read_text())
    if not (root/'evaluation_finished.json').exists():write_json(root/'evaluation_finished.json',dict(status='complete',reports=reports,
        checkpoint_sha256=finish['checkpoint_sha256'],matched_first16=True,test_used_for_selection=False))
    event(root,stage='evaluation_complete',reports=reports)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/controller_ppo_matched.yaml')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','profile','train','evaluate','all'],default='prepare')
    p.add_argument('--stopping-arithmetic',choices=['preserve','mask']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text());root=a.out.resolve()
    # Serialize stages/resume for the same run. Kernel releases flock on exit;
    # no stale PID deletion or global process termination is needed.
    if root==ROOT or len(root.parts)<5:
        raise ValueError('Use a dedicated matched run directory, not a broad workspace/root')
    if a.resume:
        if not (root/'manifest.json').is_file():raise ValueError('No prepared run to resume')
        lock_handle=(root/'pipeline.lock').open('a')
        fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fixed=dict(temperature=.6,generate_bs=16,train_split='train',train_samples=32,test_samples=16,
        test_cohort='historical_first16',vllm_dtype='bfloat16',seed=42,num_agents=2,iterations=200,
        enable_prefix_caching=True,batch_invariant=False,cache_protocol='cold_at_sequence_start_warm_between_cohorts',
        task='medqa',method='latent_mas',model_name='Qwen/Qwen3-4B',latent_only=False,sequential_info_only=False,
        gamma=1.,time_penalty=0.,max_new_tokens=4096,prompt='sequential',think=False)
    for key,value in fixed.items():
        if cfg.get(key)!=value:raise ValueError('Protocol mismatch: '+key)
    for key in ('data_dir','reference'):
        cfg[key]=str((ROOT/cfg[key]).resolve()) if not Path(cfg[key]).is_absolute() else cfg[key]
    items,split=train_items(cfg['data_dir']);args=SimpleNamespace(**cfg)
    reference=json.loads(Path(cfg['reference']).read_text())['summary']['run_config']
    for key in ('temperature','top_p','generate_bs','seed','prompt','think','latent_space_realign','max_new_tokens','num_agents'):
        if cfg[key]!=reference[key]:raise ValueError('Historical run_config differs: '+key)
    hashes={name:file_hash(ROOT/name) for name in SOURCES}
    manifest=dict(config=cfg,source_sha256=hashes,train_manifest=split,reference_sha256=file_hash(cfg['reference']),
        test300_sha256=file_hash(ROOT/'data/medqa.json'),train_samples_from_test=False,
        old_greedy_labels_reused=False,expected_live_B_train_completions=6400,expected_warmstart_B_completions=256,
        provenance_caveat='Historical unrecorded A state/model versions cannot be retroactively guaranteed bitwise. Match stored prompt strings and recorded protocol.')
    if root.exists():
        if not a.resume:raise FileExistsError('Use a new output or --resume')
        if json.loads((root/'manifest.json').read_text())!=manifest:raise ValueError('Resume source/protocol changed')
    else:
        if root==ROOT or len(root.parts)<5:raise ValueError('Use dedicated output')
        root.mkdir(parents=True);write_json(root/'manifest.json',manifest)
        for name in SOURCES:atomic_exclusive_bytes(root/'source_snapshot'/name,(ROOT/name).read_bytes())
        lock_handle=(root/'pipeline.lock').open('a')
        fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.stage=='all' and (root/'evaluation_finished.json').exists():
        print('All stages already complete; no GPU initialization');return
    os.environ.pop('VLLM_BATCH_INVARIANT',None)
    from transformers import AutoTokenizer
    from models import ModelWrapper
    from utils import set_seed
    shell=ModelWrapper.__new__(ModelWrapper);shell.tokenizer=AutoTokenizer.from_pretrained(cfg['model_name'],revision=cfg['model_revision'],local_files_only=True);shell.tokenizer.padding_side='left'
    audit=reference_prompt_audit(shell,args,args.data_dir,args.reference)
    if not (root/'historical_prompt_audit.json').exists():write_json(root/'historical_prompt_audit.json',audit)
    event(root,stage='prepared',train_split='train',train_samples=32,test_for_eval_only=True,A_B_prompts_exact=True)
    if a.stage=='prepare':return
    if a.stage in ('train','all') and a.stopping_arithmetic is None:raise ValueError('Choose --stopping-arithmetic explicitly before training')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    set_seed(42);wrapper=ModelWrapper(args.model_name,torch.device(args.device),use_vllm=True,args=args)
    workers=wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
    for worker in workers:
        if worker['head_dtype']!='torch.bfloat16' or worker['embedding_dtype']!='torch.bfloat16' or worker['quantization'] is not None or worker['model_revision']!=args.model_revision:raise ValueError('B precision/revision mismatch')
    if wrapper.HF_model.config._commit_hash!=args.model_revision or next(wrapper.HF_model.parameters()).dtype!=torch.bfloat16:raise ValueError('A precision/revision mismatch')
    write_json(root/f'preflight_{time.time_ns()}.json',dict(workers=workers,hf_dtype='torch.bfloat16',revision=args.model_revision,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,batch_invariant_env=os.environ.get('VLLM_BATCH_INVARIANT'),
        prefix_caching=True,temperature=.6,inference_batch=16,test_inputs_to_training=False))
    event(root,stage='engine_ready',stage_requested=a.stage)
    try:
        if a.stage in ('profile','all'):profile(wrapper,args,items,root)
        if a.stage in ('train','all'):train(wrapper,args,items,root,cfg,a.stopping_arithmetic)
        if a.stage in ('evaluate','all'):evaluate(wrapper,args,root,cfg)
    except Exception as error:
        event(root,stage='failed',type=type(error).__name__,message=str(error));raise


if __name__=='__main__':main()
