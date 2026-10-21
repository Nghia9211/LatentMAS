#!/usr/bin/env python3
"""Independent CPU reader: verify archive checksums/features/prefixes/scoring.

Never loads LLM weights or calls CUDA/vLLM. Writes only a new evidence report.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from safetensors.torch import load_file
from controller.profile import file_hash,tensor_hash,write_json,load_official_medqa,make_subset


def reference_phi(hidden,k):
    # Intentionally independent of controller.profile.phi4.
    h=hidden[:k+1].float()
    eps=1e-8
    lengths=[float(torch.linalg.vector_norm(row)) for row in h]
    cosine=float(((h[k]/max(lengths[k],eps))*(h[k-1]/max(lengths[k-1],eps))).sum())
    checkpoint=float(torch.linalg.vector_norm(h[k]-h[k-10]))/(lengths[k-10]+eps)
    steps=[float(torch.linalg.vector_norm(h[t]-h[t-1]))/(lengths[t-1]+eps) for t in range(1,k+1)]
    mean=sum(steps[k-10:k])/10
    previous=sum(steps[k-20:k-10])/10 if k>=20 else mean
    return torch.tensor([cosine,checkpoint,mean,mean-previous])


def verify(root,allow_incomplete=False):
    manifest=json.loads((root/'run_manifest.json').read_text())
    split=json.loads((root/'split_manifest.json').read_text())
    schema=json.loads((root/'feature_schema.json').read_text())
    assert file_hash(root/'split_manifest.json')==manifest['split_manifest_sha256']
    assert file_hash(root/'feature_schema.json')==manifest['feature_schema_sha256']
    assert schema['policy_metadata']==['k/80'] and schema['availability_masks'] is False
    assert len(schema['phi_names'])==4 and schema['phi_schema']=='phi4_checkpoint_v1'
    for name,digest in manifest['source_sha256'].items():
        assert file_hash(root/'source_snapshot'/name)==digest,(name,'source snapshot changed')
    config=manifest['config']
    all_items,metadata=load_official_medqa(config['data_dir'],'train')
    assert metadata['sha256']==split['dataset_sha256']
    items=make_subset(all_items,32,42)
    assert [item['source_id'] for item in items]==split['source_ids']
    preflights=sorted(root.glob('preflight_*.json'))
    assert preflights,'No successful GPU preflight'
    for path in preflights:
        pre=json.loads(path.read_text())
        assert pre['hf_dtype']=='torch.bfloat16' and not pre['fp32_output_head']
        assert file_hash(root/'shared/realignment.safetensors')==pre['realignment_sha256']
        for worker in pre['worker']:
            assert worker['head_dtype']==worker['embedding_dtype']=='torch.bfloat16'
            assert worker['model_revision']==config['model_revision']
    parity=json.loads((root/'parity_report.json').read_text())
    assert parity['passed'] and len(parity['rows'])==8
    for name,digest in parity['artifact_checksums'].items():assert file_hash(root/name)==digest
    for row in parity['rows']:
        assert row['passed'] and row['prefix_max_abs_diff']==0
        assert row['reused_prompt_sha256']==row['fixed_prompt_sha256']
        assert row['token_ids_exact'] and row['scorer_exact'] and row['raw_text_exact']
    summaries={str(k):dict(samples=0,correct=0,tokens=0,capped=0,parse_failures=0,loops=0) for k in range(10,81,10)}
    verified=[];missing=[];shapes=set();max_phi_diff=0.0;repeatability=[]
    # Resolve checked parity artifacts from the report rather than guessing the
    # latest attempt, which may be incomplete after an interrupted run.
    parity_reused={}
    for name in parity['artifact_checksums']:
        path=Path(name)
        if path.name.startswith('reused_K') and path.suffix=='.json':
            reused=json.loads((root/path).read_text())
            parity_reused[(reused['source_id'],reused['k'])]=reused
    for item in items:
        base=root/'trajectories'/item['source_id']
        if not (base/'complete.json').exists():
            missing.append(item['source_id']);continue
        marker=json.loads((base/'complete.json').read_text())
        assert marker['source_id']==item['source_id']
        dest=base/marker['attempt']
        for name,digest in marker['checksums'].items():assert file_hash(dest/name)==digest
        states=load_file(str(dest/'states.safetensors'))
        prompts=load_file(str(dest/'prompts.safetensors'))
        hidden=states['hidden_last_position_all_layers'];latent=states['latent_inputs_actual']
        assert hidden.ndim==3 and hidden.shape[0]==81
        assert latent.shape==(80,hidden.shape[-1])
        assert hidden.dtype==latent.dtype==torch.bfloat16
        assert torch.isfinite(hidden).all() and torch.isfinite(latent).all()
        shapes.add(tuple(hidden.shape))
        texts=json.loads((dest/'prompts.json').read_text())
        assert item['question'] in texts['A']['rendered'] and item['question'] in texts['B']['rendered']
        checkpoints=[json.loads(line) for line in (dest/'checkpoints.jsonl').read_text().splitlines()]
        rows=[json.loads(line) for line in (dest/'b_outcomes.jsonl').read_text().splitlines()]
        assert [row['k'] for row in rows]==list(range(10,81,10))
        assert [row['k'] for row in checkpoints]==list(range(10,81,10))
        for checkpoint,row in zip(checkpoints,rows):
            k=row['k']
            assert checkpoint['normalized_step']==k/80 and checkpoint['hidden_index']==k
            assert set(checkpoint)=={'source_id','k','normalized_step','hidden_index','latent_prefix_end_exclusive','phi_schema','phi','status'}
            assert len(checkpoint['phi'])==4 and checkpoint['phi_schema']=='phi4_checkpoint_v1'
            reference=reference_phi(hidden[:,-1],k)
            measured=torch.tensor(checkpoint['phi'])
            torch.testing.assert_close(measured,reference,atol=1e-6,rtol=1e-5)
            max_phi_diff=max(max_phi_diff,float((reference-measured).abs().max()))
            assert row['source_id']==item['source_id'] and row['gold']==item['gold']
            assert row['status']=='complete' and row['decoder_seed']==42
            assert row['output_tokens']==len(row['generated_token_ids'])<=4096
            assert row['truncation_flag']==(row['finish_reason']=='length')
            boxes=re.findall(r'\\boxed\{([^}]*)\}',row['raw_prediction'])
            answer=boxes[-1].strip().lower() if boxes else None
            answer=answer if answer in ('a','b','c','d') else None
            assert row['strict_prediction']==answer
            assert row['strict_parse_failed']==(answer is None)
            assert row['strict_correct']==(answer is not None and answer==item['gold'])
            prefix=torch.cat([prompts['A_prompt_embeddings_actual'],latent[:k]])
            b=prompts['B_prompt_nonlatent_embeddings'];insertion=texts['insertion']
            reconstructed=torch.cat([b[:insertion],prefix,b[insertion:]])
            assert row['prompt_sha256']==tensor_hash(reconstructed)
            assert row['prompt_embedding_tokens']==len(reconstructed)
            reused=parity_reused.get((item['source_id'],k))
            if reused is not None:
                assert row['prompt_sha256']==reused['prompt_sha256']
                repeatability.append(dict(source_id=item['source_id'],k=k,
                    prefix_exact=True,
                    token_ids_exact=row['generated_token_ids']==reused['generated_token_ids'],
                    raw_text_exact=row['raw_prediction']==reused['raw_prediction'],
                    strict_score_exact=row['strict_correct']==reused['strict_correct']))
            tally=summaries[str(k)]
            tally['samples']+=1;tally['correct']+=int(row['strict_correct']);tally['tokens']+=row['output_tokens']
            tally['capped']+=int(row['truncation_flag']);tally['parse_failures']+=int(row['strict_parse_failed'])
            tally['loops']+=int(row['repetition']['loop_flag'])
        target=json.loads((dest/'train_targets.json').read_text())
        correct=[int(row['strict_correct']) for row in rows]
        optimal=min(row['k'] for row in rows if int(row['strict_correct'])==max(correct))
        assert target['oracle_budget']==optimal and target['oracle_correct']==max(correct)
        for i,row in enumerate(rows[:-1]):
            expected='HANDOFF' if correct[i]>=max(correct[i+1:]) else 'CONTINUE_10'
            assert target['checkpoint_actions'][str(row['k'])]==expected
        verified.append(item['source_id'])
    if not allow_incomplete:
        assert not missing and (root/'finished.json').exists(),'Incomplete pilot'
        finished=json.loads((root/'finished.json').read_text())
        for k,tally in summaries.items():
            tally['accuracy']=tally['correct']/tally['samples']
            assert tally==finished['per_budget'][k]
    return dict(passed=not missing,verified_completed_samples=len(verified),missing_samples=len(missing),
        completed_b_outcomes=len(verified)*8,expected_b_outcomes=256,all_layer_shapes=[list(shape) for shape in sorted(shapes)],
        max_phi_absolute_diff=max_phi_diff,checksums_verified=True,source_snapshots_verified=True,
        independently_recomputed_phi=bool(verified),independently_reconstructed_b_prefix_hashes=bool(verified),
        independently_rescored_strict_answers=bool(verified),train_only=True,all_bf16=True,
        metadata_only_k_over_80=True,no_availability_masks=True,parity_exact_cases=8,
        same_engine_repeated_b_cases=repeatability,
        same_engine_repeated_b_exact=sum(all(row[key] for key in ('token_ids_exact','raw_text_exact','strict_score_exact')) for row in repeatability),
        repeatability_scope='Existing parity/main calls with identical cold-cache B prefixes in one engine; not independent seeds or engine restarts. Output variation is reported separately from archive integrity.',
        per_budget=summaries,missing_source_ids=missing,
        verifier_sha256=file_hash(Path(__file__)),gpu_used_by_verifier=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--allow-incomplete',action='store_true')
    options=parser.parse_args()
    root=options.run.resolve()
    report=verify(root,options.allow_incomplete)
    path=root/f'independent_verification_{time.time_ns()}.json'
    write_json(path,report)
    print(json.dumps({'report':str(path),**report},indent=2))


if __name__=='__main__':main()
