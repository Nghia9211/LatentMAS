#!/usr/bin/env python3
"""Prepare/run/verify the frozen MedQA A-latent -> B-text profiling pilot.

--dry-run prepares hash-locked manifests only; never loads model weights.
--resume preserves previous attempts and requires unchanged source/protocol.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
import yaml
from safetensors.torch import load_file
from controller.profile import (GRID,PHI_SCHEMA,PHI_NAMES,EPS,TrajectoryCollector,
    file_hash,tensor_hash,load_official_medqa,make_subset,phi4,tie_targets,
    write_json,write_jsonl,write_tensors,archive_coverage,reconstruct_prefix)

SOURCES = ['controller/__init__.py','controller/profile.py','scripts/profile_controller.py','latent_batch.py',
           'models.py','methods/latent_mas.py','prompts.py','utils.py',
           'scripts/audit_saved_loop_budgets.py']


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    template = yaml.safe_load((ROOT/'configs/profile_medqa_pilot.yaml').read_text())
    if not isinstance(config,dict) or set(config)!=set(template):
        raise ValueError('Profile config must have exactly the documented pilot fields')
    for key,value in config.items():
        example = template[key]
        if key=='model_revision':
            if value is not None and not isinstance(value,str):
                raise ValueError('model_revision must be a pinned hash or null')
        elif type(value) is not type(example):
            raise ValueError(f'Wrong config type: {key}')
    fixed = dict(model_name='Qwen/Qwen3-4B',task='medqa',method='latent_mas',num_agents=2,
        prompt='sequential',think=False,latent_space_realign=True,latent_only=False,
        sequential_info_only=False,temperature=0.0,max_new_tokens=4096,latent_steps=80,
        checkpoints=list(GRID),seed=42,generate_bs=1,pilot_samples=32,split='train',
        use_vllm=True,use_second_HF_model=True,vllm_dtype='bfloat16',tensor_parallel_size=1,
        worker_extension_cls='controller.profile.ProfileWorkerInspector',
        reset_cache_before_each_b=True,capture='all_layer_every_step',phi_schema=PHI_SCHEMA,
        parity_samples=2,parity_checkpoints=[10,20,40,80])
    for key,value in fixed.items():
        if config[key]!=value:
            raise ValueError(f'Pilot protocol mismatch: {key} must be {value!r}')
    if not 0<config['gpu_memory_utilization']<1 or not 0<config['top_p']<=1:
        raise ValueError('Invalid memory utilization/top_p')
    return config


def pin_revision(config):
    if config['model_revision'] is None:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(config['model_name'],'config.json')
        if not isinstance(cached,str) or 'snapshots' not in Path(cached).parts:
            raise ValueError('No locally cached model revision; set a pinned model_revision')
        parts = Path(cached).parts
        config['model_revision'] = parts[parts.index('snapshots')+1]
    revision = config['model_revision']
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Use a40-character model commit hash, not main')


def source_hashes():
    return {name:file_hash(ROOT/name) for name in SOURCES}


def feature_schema():
    return dict(schema_version='latent_controller_profile_v1',phi_schema=PHI_SCHEMA,
        phi_names=PHI_NAMES,phi_dtype='float32',epsilon=EPS,
        hidden_convention='h0 last real prompt position; h_t after forward of e_t',
        latent_convention='e_t=realign(h_(t-1)); stored actual BF16 input, entry0=e1',
        observation=['current_hidden','anchor_h0','delta_previous_checkpoint','phi4','k/80'],
        raw_observation_dimension='3d+5',policy_metadata=['k/80'],availability_masks=False,
        history='actual previous checkpoints only, no future states, reset per question',
        trend_at_k10=0,checkpoints=list(GRID),labels_separate=True,
        reward='strict correctness only; CONTINUE0, terminal correct1/incorrect0; gamma1',
        tie_break='earliest best correctness; all-correct/all-wrong ->K10',
        timing='report only, instrumented, not controller features or reward')


def event(root, **fields):
    fields['utc'] = datetime.now(timezone.utc).isoformat()
    with (root/'events.jsonl').open('a',encoding='utf8') as handle:
        handle.write(json.dumps(fields,ensure_ascii=False,allow_nan=False)+'\n')
        handle.flush()
    print(json.dumps(fields,ensure_ascii=False),flush=True)


def prepare(config,root,resume):
    data_dir=Path(config['data_dir'])
    if not data_dir.is_absolute(): data_dir=ROOT/data_dir
    config['data_dir']=str(data_dir.resolve())
    pin_revision(config)
    items,metadata=load_official_medqa(data_dir,config['split'])
    current_hashes=source_hashes()
    if resume:
        manifest=json.loads((root/'run_manifest.json').read_text())
        if manifest['config']!=config or manifest['source_sha256']!=current_hashes:
            raise ValueError('Resume protocol/source differs; use a new run directory')
        saved=json.loads((root/'split_manifest.json').read_text())
        if saved['dataset_sha256']!=metadata['sha256']:
            raise ValueError('Resume dataset differs')
        lookup={item['source_id']:item for item in items}
        subset=[lookup[source_id] for source_id in saved['source_ids']]
        if subset!=make_subset(items,32,42):
            raise ValueError('Resume subset/order differs')
        if json.loads((root/'feature_schema.json').read_text())!=feature_schema():
            raise ValueError('Resume feature schema differs')
        return subset
    root.mkdir(parents=True,exist_ok=False)
    subset=make_subset(items,32,42)
    split=dict(split='train',seed=42,dataset_sha256=metadata['sha256'],
        selection='random.Random(seed).sample(first-row-per-NFKC-casefold-whitespace-stem,32)',
        source_ids=[item['source_id'] for item in subset],
        question_sha256=[item['question_sha256'] for item in subset],
        count=len(subset),dataset_manifest_sha256=file_hash(data_dir/'manifest.json'))
    write_json(root/'split_manifest.json',split)
    write_json(root/'feature_schema.json',feature_schema())
    write_json(root/'run_manifest.json',dict(config=config,source_sha256=current_hashes,
        prepared_utc=datetime.now(timezone.utc).isoformat(),split_manifest_sha256=file_hash(root/'split_manifest.json'),
        feature_schema_sha256=file_hash(root/'feature_schema.json'),
        initial_command=[sys.executable]+sys.argv,
        expected_main_b_completions=256,expected_parity_b_completions=16,
        historical_test300_used=False,training_started=False))
    for name in SOURCES:
        from controller.profile import atomic_exclusive_bytes
        atomic_exclusive_bytes(root/'source_snapshot'/name,(ROOT/name).read_bytes())
    event(root,stage='prepared',samples=32,main_b_completions=256,model_revision=config['model_revision'])
    return subset


@torch.no_grad()
def capture_a(wrapper,args,item):
    from prompts import build_agent_message_sequential_latent_mas as builder
    collector=TrajectoryCollector()
    prompts={}
    strings={}
    for role,key in [('planner','A'),('judger','B')]:
        messages=builder(role=role,question=item['question'],context='',method='latent_mas',args=args)
        text=wrapper.render_chat(messages,add_generation_prompt=True)
        strings[key]=dict(messages=messages,rendered=text)
        encoded=wrapper.tokenizer([text],return_tensors='pt',padding=True,add_special_tokens=False)
        prompts[key+'_input_ids']=encoded['input_ids'][0].cpu()
        prompts[key+'_attention_mask']=encoded['attention_mask'][0].cpu()
    a_ids=prompts['A_input_ids'][None].to(wrapper.HF_device)
    a_mask=prompts['A_attention_mask'][None].to(wrapper.HF_device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    started=time.perf_counter()
    cache,embeddings=wrapper.generate_latent_batch_hidden_state(a_ids,a_mask,latent_steps=80,observer=collector)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    seconds=time.perf_counter()-started
    states=collector.tensors()
    length=int(a_mask.sum())
    prompts['A_prompt_embeddings_actual']=embeddings[0][:length].detach().cpu().contiguous()
    if not torch.equal(embeddings[0][length:].cpu(),states['latent_inputs_actual']):
        raise ValueError('Actual latent inputs disagree with production prefix')
    b_ids=prompts['B_input_ids'][None].to(wrapper.HF_device)
    b_mask=prompts['B_attention_mask'].bool()
    prompts['B_prompt_nonlatent_embeddings']=wrapper.embedding_layer(b_ids)[0,b_mask].detach().cpu().contiguous()
    marker='<|im_start|>user\n'
    index=strings['B']['rendered'].find(marker)
    if index<0: raise ValueError('Missing Qwen insertion marker')
    insertion=len(wrapper.tokenizer(strings['B']['rendered'][:index+len(marker)],add_special_tokens=False)['input_ids'])
    prompts['A_position_ids']=torch.arange(length,dtype=torch.long)
    prompts['latent_position_ids']=torch.arange(length,length+80,dtype=torch.long)
    del cache,embeddings
    if states['hidden_last_position_all_layers'].dtype!=torch.bfloat16:
        raise ValueError('A trajectory must retain actual BF16 dtype')
    return dict(states=states,prompts=prompts,strings=strings,insertion=insertion,
        A_instrumented_seconds=seconds,layer_count=states['hidden_last_position_all_layers'].shape[1],
        hidden_dim=states['hidden_last_position_all_layers'].shape[-1])


def b_input(capture,k,args):
    from latent_batch import insert_prefix
    prompt=capture['prompts']['B_prompt_nonlatent_embeddings'][None].to(args.device)
    mask=torch.ones(prompt.shape[:2],dtype=torch.long,device=prompt.device)
    prefix=reconstruct_prefix(capture['prompts'],capture['states'],k).to(args.device)
    return insert_prefix(prompt,mask,[prefix],[capture['insertion']])[0]


def decode(wrapper,args,item,k,payload,root,purpose):
    from vllm import SamplingParams
    from scripts.audit_saved_loop_budgets import repetition
    from utils import extract_gsm8k_answer,normalize_answer,medqa_scoring_metadata
    if args.reset_cache_before_each_b and not wrapper.vllm_engine.reset_prefix_cache():
        raise RuntimeError('Cannot reset private engine prefix cache before label generation')
    digest=tensor_hash(payload['prompt_embeds'])
    event(root,stage='B_start',source_id=item['source_id'],k=k,purpose=purpose,prompt_sha256=digest)
    started=time.perf_counter()
    output=wrapper.vllm_engine.generate([payload],SamplingParams(temperature=0,top_p=args.top_p,
        seed=args.seed,max_tokens=args.max_new_tokens),use_tqdm=False)[0].outputs[0]
    seconds=time.perf_counter()-started
    ids=list(output.token_ids)
    pred=normalize_answer(extract_gsm8k_answer(output.text))
    metadata=medqa_scoring_metadata(output.text,item['gold'],pred,pred==item['gold'])
    strict=metadata['medqa_boxed_v1']
    row=dict(source_id=item['source_id'],source_split=item['source_split'],k=k,repeat_id=0,
        decoder_seed=42,gold=item['gold'],raw_prediction=output.text,generated_token_ids=ids,
        output_tokens=len(ids),finish_reason=output.finish_reason,stop_reason=output.stop_reason,
        strict_prediction=strict['prediction'],strict_correct=strict['correct'],
        strict_parse_failed=strict['parse_failed'],legacy_prediction=pred,
        legacy_correct=metadata['legacy_correct'],metric_version=metadata['metric_version'],
        truncation_flag=output.finish_reason=='length',repetition=repetition(ids),
        repetition_diagnostics_version='audit_saved_loop_budgets_v1',
        prompt_embedding_tokens=len(payload['prompt_embeds']),prompt_sha256=digest,
        report_only={'B_call_seconds':seconds},status='complete')
    event(root,stage='B_complete',source_id=item['source_id'],k=k,purpose=purpose,
        tokens=len(ids),strict_correct=row['strict_correct'],finish_reason=output.finish_reason,
        loop_flag=row['repetition']['loop_flag'],seconds=round(seconds,3))
    return row


def parity(wrapper,args,items,root):
    """Compare observer A80 prefixes against the actual existing fixed-K method."""
    from methods.latent_mas import LatentMASMethod
    report_path=root/'parity_report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if not report['passed']: raise RuntimeError('Previous parity failed; preserve evidence/use a new run')
        verify_hashes=report['artifact_checksums']
        from controller.profile import verify_file_manifest
        verify_file_manifest(root,verify_hashes)
        return {}
    captures,rows,checksums={},[],{}
    parity_base=root/'parity'
    attempt_index=0
    while (parity_base/f'attempt_{attempt_index:03d}').exists(): attempt_index+=1
    parity_base=parity_base/f'attempt_{attempt_index:03d}'
    for item in items[:args.parity_samples]:
        capture=capture_a(wrapper,args,item)
        captures[item['source_id']]=capture
        dest=parity_base/item['source_id']
        dest.mkdir(parents=True,exist_ok=False)
        write_tensors(dest/'states.safetensors',capture['states'])
        write_tensors(dest/'prompts.safetensors',capture['prompts'])
        write_json(dest/'prompts.json',dict(**capture['strings'],insertion=capture['insertion']))
        for path in dest.iterdir(): checksums[str(path.relative_to(root))]=file_hash(path)
        for k in args.parity_checkpoints:
            payload=b_input(capture,k,args)
            reused=decode(wrapper,args,item,k,payload,root,'parity_reused')
            write_json(dest/f'reused_K{k}.json',reused)
            base_engine=wrapper.vllm_engine
            class RecordingEngine:
                def generate(self,inputs,params):
                    self.input=inputs[0]['prompt_embeds'].detach().cpu().clone()
                    self.completion=base_engine.generate(inputs,params,use_tqdm=False)[0]
                    return [self.completion]
            recording=RecordingEngine()
            if not base_engine.reset_prefix_cache(): raise RuntimeError('Parity cache reset failed')
            wrapper.vllm_engine=recording
            event(root,stage='B_start',source_id=item['source_id'],k=k,purpose='parity_fixed')
            try:
                method=LatentMASMethod(wrapper,latent_steps=k,judger_max_new_tokens=args.max_new_tokens,
                    temperature=0,top_p=args.top_p,generate_bs=1,args=args)
                fixed=method.run_batch_vllm([item])[0]
            finally:
                wrapper.vllm_engine=base_engine
            out=recording.completion.outputs[0]
            event(root,stage='B_complete',source_id=item['source_id'],k=k,purpose='parity_fixed',
                tokens=len(out.token_ids),finish_reason=out.finish_reason)
            actual=payload['prompt_embeds'].detach().cpu()
            shape_match=actual.shape==recording.input.shape
            prefix_exact=shape_match and torch.equal(actual,recording.input)
            row=dict(source_id=item['source_id'],k=k,prefix_shape=list(actual.shape),
                prefix_exact=prefix_exact,prefix_max_abs_diff=float((actual.float()-recording.input.float()).abs().max()) if shape_match else None,
                reused_prompt_sha256=tensor_hash(actual),fixed_prompt_sha256=tensor_hash(recording.input),
                token_ids_exact=reused['generated_token_ids']==list(out.token_ids),
                raw_text_exact=reused['raw_prediction']==out.text,
                scorer_exact=reused['strict_correct']==fixed['medqa_boxed_v1']['correct'])
            row['passed']=all(row[key] for key in ('prefix_exact','token_ids_exact','raw_text_exact','scorer_exact'))
            write_json(dest/f'fixed_K{k}.json',fixed)
            rows.append(row)
            for name in (f'reused_K{k}.json',f'fixed_K{k}.json'):
                path=dest/name; checksums[str(path.relative_to(root))]=file_hash(path)
            event(root,stage='parity_case',**row)
    report=dict(passed=all(row['passed'] for row in rows),rows=rows,artifact_checksums=checksums,
        note='Same BF16 model/prompts/greedy/cold-private-cache protocol; independent existing production fixed-K path without observer.')
    write_json(report_path,report)
    if not report['passed']: raise RuntimeError('GPU parity failed; full pilot not released')
    return captures


def capture_sample(wrapper,args,item,root,capture=None):
    base=root/'trajectories'/item['source_id']
    if (base/'complete.json').exists():
        archive_coverage(root,[item['source_id']])
        event(root,stage='skip_verified_complete',source_id=item['source_id'])
        return
    base.mkdir(parents=True,exist_ok=True)
    index=0
    while (base/f'attempt_{index:03d}').exists(): index+=1
    dest=base/f'attempt_{index:03d}'
    dest.mkdir()
    try:
        capture=capture if capture is not None else capture_a(wrapper,args,item)
        write_tensors(dest/'states.safetensors',capture['states'])
        write_tensors(dest/'prompts.safetensors',capture['prompts'])
        write_json(dest/'prompts.json',dict(**capture['strings'],insertion=capture['insertion'],
            protocol='B prefix=A prompt embeddings+e1..eK, inserted after Qwen user marker',
            chat_template_enable_thinking='unchanged/default; think flag does not set this'))
        checkpoints=[]
        hidden=capture['states']['hidden_last_position_all_layers'][:,-1]
        for k in GRID:
            checkpoints.append(dict(source_id=item['source_id'],k=k,normalized_step=k/80,
                hidden_index=k,latent_prefix_end_exclusive=k,phi_schema=PHI_SCHEMA,
                phi=phi4(hidden,k).tolist(),status='complete'))
        write_jsonl(dest/'checkpoints.jsonl',checkpoints)
        rows=[]
        for k in GRID:
            row=decode(wrapper,args,item,k,b_input(capture,k,args),root,'main')
            write_json(dest/f'B_K{k}.json',row)
            rows.append(row)
        write_jsonl(dest/'b_outcomes.jsonl',rows)
        write_json(dest/'train_targets.json',tie_targets([r['strict_correct'] for r in rows]))
        write_json(dest/'manifest.json',dict(source_id=item['source_id'],source_split='train',
            actual_dtype=str(hidden.dtype),hidden_dim=capture['hidden_dim'],
            hidden_state_indices=capture['layer_count'],
            layer_mapping='index0 embedding; indices1..N-1 decoder outputs; indexN final-norm last-layer output',
            hidden_shape=list(capture['states']['hidden_last_position_all_layers'].shape),
            latent_shape=list(capture['states']['latent_inputs_actual'].shape),
            capture='last real position only, all indices every step, no full KV dumps',
            report_only={'A_instrumented_seconds':capture['A_instrumented_seconds']}))
        checksums={path.name:file_hash(path) for path in dest.iterdir() if path.is_file()}
        write_json(base/'complete.json',dict(source_id=item['source_id'],attempt=dest.name,checksums=checksums,status='complete'))
        event(root,stage='sample_complete',source_id=item['source_id'],strict_correct=sum(r['strict_correct'] for r in rows))
    except Exception as error:
        write_json(dest/'failure.json',dict(type=type(error).__name__,message=str(error),source_id=item['source_id']))
        raise


def finish(root,items):
    coverage=archive_coverage(root,[item['source_id'] for item in items])
    write_json(root/f'coverage_{time.time_ns()}.json',coverage)
    if not coverage['all_complete']: raise RuntimeError('Incomplete pilot coverage')
    summary={str(k):dict(samples=0,correct=0,tokens=0,capped=0,parse_failures=0,loops=0) for k in GRID}
    for item in items:
        base=root/'trajectories'/item['source_id']
        marker=json.loads((base/'complete.json').read_text())
        for row in map(json.loads,(base/marker['attempt']/'b_outcomes.jsonl').read_text().splitlines()):
            acc=summary[str(row['k'])]
            acc['samples']+=1; acc['correct']+=int(row['strict_correct']); acc['tokens']+=row['output_tokens']
            acc['capped']+=int(row['truncation_flag']); acc['parse_failures']+=int(row['strict_parse_failed'])
            acc['loops']+=int(row['repetition']['loop_flag'])
    for row in summary.values(): row['accuracy']=row['correct']/row['samples']
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    completed=[row for row in events if row.get('stage')=='B_complete']
    payload=dict(status='complete',coverage=coverage,per_budget=summary,main_b_completions=256,
        observed_completed_b_calls=len(completed),parity_passed=json.loads((root/'parity_report.json').read_text())['passed'],
        training_started=False,initialization_excluded_from_request_timings=True,
        timing_note='instrumented sequential profiling, not adaptive latency or throughput claim')
    if not (root/'finished.json').exists(): write_json(root/'finished.json',payload)
    event(root,stage='pilot_complete',samples=32,main_b_completions=256,all_observed_b_calls=len(completed))
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/profile_medqa_pilot.yaml')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify',action='store_true')
    options=parser.parse_args()
    root=options.out.resolve()
    if root==ROOT or len(root.parts)<4: raise ValueError('Use a dedicated run directory')
    config=load_config(options.config)
    items=prepare(config,root,options.resume or options.verify)
    if options.verify:
        if not (root/'finished.json').exists(): raise RuntimeError('Pilot not finished')
        print(json.dumps(archive_coverage(root,[item['source_id'] for item in items]),indent=2))
        return
    if options.dry_run: return
    if (root/'finished.json').exists():
        archive_coverage(root,[item['source_id'] for item in items])
        print('Already complete and checksum-verified; no GPU allocation.',flush=True)
        return
    if not torch.cuda.is_available(): raise RuntimeError('Pilot requires CUDA')
    from models import ModelWrapper
    from utils import set_seed
    args=SimpleNamespace(**config)
    set_seed(args.seed)
    write_json(root/f'invocation_{time.time_ns()}.json',dict(command=[sys.executable]+sys.argv,
        env={key:os.environ.get(key) for key in ('PYTHONHASHSEED','VLLM_BATCH_INVARIANT','VLLM_WORKER_MULTIPROC_METHOD',
            'VLLM_NO_USAGE_STATS','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES')}))
    started=time.perf_counter()
    wrapper=ModelWrapper(args.model_name,torch.device(args.device),use_vllm=True,args=args)
    worker=wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
    for row in worker:
        if row['embedding_dtype']!='torch.bfloat16' or row['head_dtype']!='torch.bfloat16' or row['quantization'] is not None:
            raise ValueError('B must retain unquantized BF16 backbone/head')
        if row['model_revision']!=args.model_revision: raise ValueError('B revision differs from locked model')
    if wrapper.HF_model.config._commit_hash!=args.model_revision: raise ValueError('A revision differs')
    matrix,target=wrapper._ensure_latent_realign_matrix(wrapper.HF_model,torch.device(args.device2),args)
    if not (root/'shared/realignment.safetensors').exists():
        write_tensors(root/'shared/realignment.safetensors',dict(matrix=matrix,target_norm=target))
    else:
        previous=load_file(str(root/'shared/realignment.safetensors'))
        if not torch.equal(previous['matrix'],matrix.cpu()) or not torch.equal(previous['target_norm'],target.cpu()):
            raise ValueError('Resume realignment differs')
    versions={p:importlib.metadata.version(p) for p in ('torch','transformers','vllm','safetensors')}
    write_json(root/f'preflight_{time.time_ns()}.json',dict(versions=versions,worker=worker,
        hf_dtype=str(next(wrapper.HF_model.parameters()).dtype),hf_revision=wrapper.HF_model.config._commit_hash,
        initialization_seconds=time.perf_counter()-started,chat_template=wrapper.tokenizer.chat_template,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),realignment_sha256=file_hash(root/'shared/realignment.safetensors'),
        caching='enabled per config, reset private engine cache before each B label',
        prompt_embeddings=True,fp32_output_head=False))
    event(root,stage='engine_ready',seconds=round(time.perf_counter()-started,3))
    captures=parity(wrapper,args,items,root)
    for item in items: capture_sample(wrapper,args,item,root,captures.pop(item['source_id'],None))
    finish(root,items)


if __name__=='__main__':
    main()
