#!/usr/bin/env python3
"""Versioned train1024 entrypoint; reuse original normalized live PPO math.

Default is CPU prepare. No old run/source archive/checkpoint is mutated.
"""
import argparse,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from safetensors.torch import load_file
from controller.expanded_dataset import load_expanded_dataset
from controller.normalized import SCHEMA,ObservationNormalizer
from controller.profile import load_official_medqa,file_hash,write_json,write_tensors,atomic_exclusive_bytes
from controller.matched import historical_style,require_train
from controller.server_training import current_sources
from scripts.migrate_b_seed_repeats import contained,verify_hashes
from scripts.run_matched_controller_ppo import completed,new_attempt,commit
from scripts.train_normalized_controller import load_config,train

EXPANSION='medqa_train_pool_expansion_v1'

def approved_items(plan,parent_manifest,split):
    if split not in ('train','validation'):raise ValueError('Train/validation only')
    selection=plan['selection'] if split=='train' else parent_manifest['selections']['validation']
    ids=selection['source_ids']
    pool,meta=load_official_medqa(plan['config']['data_dir'],split)
    if meta['sha256']!=selection['dataset_sha256']:raise ValueError('Official dataset changed')
    lookup={i['source_id']:i for i in pool}
    if len(ids)!=(1024 if split=='train' else 128) or len(set(ids))!=len(ids):raise ValueError('Wrong approved pool size')
    items=[dict(lookup[i],question=historical_style(lookup[i])) for i in ids]
    if any(i['source_split']!=split or ':'+split+':' not in i['source_id'] for i in items):raise ValueError('Split firewall failed')
    if split=='train':
        for start in range(0,len(items),32):require_train(items[start:start+32])
    return items

def training_config(plan,settings):
    cfg=dict(plan['config'],train_samples=1024,validation_samples=128,
        iterations=settings['iterations'],warmstart_epochs=settings['warmstart_epochs'],
        controller_normalization=SCHEMA,projection_norm=settings['projection_norm'],normalization_eps=float(settings['eps']))
    cfg['device']=cfg['device2']
    if (cfg['width']!=128 or cfg['optimization_epochs']!=4 or cfg['minibatch_questions']!=8
            or cfg['time_penalty']!=0 or cfg['tie_weight']!=.1 or cfg['entropy_weight']!=.01
            or cfg['learning_rate']!=.0003 or cfg['temperature']!=.6 or cfg['generate_bs']!=16):
        raise ValueError('Baseline must retain original PPO/reward protocol')
    tie_weight=settings.get('tie_weight',cfg['tie_weight'])
    if type(tie_weight) not in (int,float) or tie_weight not in (0,.1):
        raise ValueError('Expected tie_weight 0 or 0.1')
    cfg['tie_weight']=tie_weight
    return cfg

def schedule(iterations,every):return sorted({0,iterations}|set(range(every,iterations+1,every)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--stage',choices=('prepare','train'),default='prepare')
    ap.add_argument('--dry-run',action='store_true')
    cli=ap.parse_args();config=contained(cli.config,ROOT);settings=load_config(config,allow_zero_warmstart=True,allow_tie_weight=True)
    if settings.get('migration'):raise ValueError('Expanded verified dataset already has immutable lineage')
    profile=contained(ROOT/settings['profile'],ROOT);out=contained(ROOT/settings['out'],ROOT)
    if out==profile or profile in out.parents or out in profile.parents or out==ROOT:raise ValueError('Separate output required')
    with (profile/'pipeline.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        plan=json.loads((profile/'plan.json').read_text());finish=json.loads((profile/'finished.json').read_text())
        if plan['schema']!=EXPANSION or not finish.get('verified'):raise ValueError('Verified train1024 expansion required')
        d,dm=load_expanded_dataset(profile,'train');v,vm=load_expanded_dataset(profile,'validation')
        parent=contained(plan['parent_profile'],ROOT);pm=json.loads((parent/'manifest.json').read_text())
        if file_hash(parent/'manifest.json')!=plan['parent_manifest_sha256']:raise ValueError('Parent changed')
        items=approved_items(plan,pm,'train');valid_items=approved_items(plan,pm,'validation')
        ids=[i['source_id'] for i in items];vids=[i['source_id'] for i in valid_items]
        if ids!=dm['source_ids'] or vids!=vm['source_ids'] or set(ids)&set(vids):raise ValueError('Identity mismatch')
        cfg=training_config(plan,settings);checkpoints=schedule(cfg['iterations'],settings['validation_every'])
        sources=current_sources(ROOT,config)
        baselines={str(k):dict(accuracy=float(v['correctness'][:,j].mean())) for j,k in enumerate(range(10,81,10))}
        manifest=dict(schema='expanded_normalized_live_ppo_v1',config=cfg,profile=str(profile),
            profile_plan_sha256=file_hash(profile/'plan.json'),profile_finished_sha256=file_hash(profile/'finished.json'),
            train_dataset_sha256=dm['file_sha256'],validation_dataset_sha256=vm['file_sha256'],
            source_sha256=sources,train_ids=ids,validation_ids=vids,
            normalization_fit_split='train',normalization_fit_rows=len(ids)*7,
            validation_checkpoints=checkpoints,expected_training_B_completions=cfg['iterations']*32,
            expected_validation_B_completions=len(checkpoints)*128,raw_profile_reused=True,old_weights_reused=False,
            test_used_for_training=False,realignment_profile=str(parent),
            warmstart_skipped=cfg['warmstart_epochs']==0,
            protocol=('Normalized MLP128, '+('matched random initialization; no supervised updates' if cfg['warmstart_epochs']==0 else str(cfg['warmstart_epochs'])+' warmstart CE+Q epochs')+', live binary PPO reward, '+('no tie auxiliary' if cfg['tie_weight']==0 else 'original tie auxiliary')),
            migration=dict(schema=EXPANSION,archived_realignment_sha256=plan['archived_realignment_sha256'],
                baseline_source='Verified three-seed profile, mixed GB10/H100 inherited data, different engine lifetimes',
                hardware_parity_established=False),
            wandb={k:settings['wandb_'+k] for k in ('mode','project','entity','name')})
        summary=dict(output=str(out),train=len(ids),validation=len(vids),iterations=cfg['iterations'],
            validation_checkpoints=checkpoints,normalization_observations=len(ids)*7,
            train_B=manifest['expected_training_B_completions'],validation_B=manifest['expected_validation_B_completions'],
            optimizer_updates=cfg['iterations']*16,warmstart_epochs=cfg['warmstart_epochs'],
            warmstart_skipped=cfg['warmstart_epochs']==0,tie_weight=cfg['tie_weight'],
            tie_auxiliary_enabled=cfg['tie_weight']!=0,test_used=False,source_files=len(sources))
        if cli.dry_run:
            print(json.dumps(dict(status='dry_run',**summary,CUDA_initialized=torch.cuda.is_initialized())));return
        if out.exists():
            if not (out/'manifest.json').exists() or json.loads((out/'manifest.json').read_text())!=manifest:
                raise ValueError('Output exists with a different immutable protocol')
        else:
            out.mkdir(parents=True);write_json(out/'manifest.json',manifest)
            for name in sources:atomic_exclusive_bytes(out/'source_snapshot'/name,contained(ROOT/name,ROOT).read_bytes())
        with (out/'pipeline.lock').open('a') as output_lock:
            fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            verify_hashes(ROOT,sources)
            norm=ObservationNormalizer(7685,settings['eps']);base=out/'normalization';p=completed(base)
            if p is None:
                p=new_attempt(base);norm.fit(d['features'],source_split='train')
                write_tensors(p/'normalizer.safetensors',norm.state_dict())
                write_json(p/'metadata.json',dict(schema=SCHEMA,source_split='train',observations=len(ids)*7,
                    fitted_without_labels=True,train_dataset_sha256=dm['file_sha256'],k_over_80_unchanged=True))
                commit(base,p)
            else:norm.load_state_dict(load_file(str(p/'normalizer.safetensors')))
            if cli.stage=='prepare':
                print(json.dumps(dict(status='prepared',**summary,CUDA_initialized=torch.cuda.is_initialized(),wandb_initialized=False)));return
            from dotenv import load_dotenv
            env=contained(ROOT/'.env',ROOT)
            load_dotenv(env,override=False)
            if settings['wandb_mode']=='online' and not os.environ.get('WANDB_API_KEY'):
                raise ValueError('Online tracking requires key from .env or environment')
            report=dict(summaries=dict(validation=baselines))
            train(out,manifest,d,norm,report,train_items=items,validation_items=valid_items)

if __name__=='__main__':main()
