"""CPU loader for verified variable-size raw three-seed profile datasets."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from controller.profile import file_hash
from scripts.migrate_b_seed_repeats import contained,verify_hashes
from scripts.run_matched_controller_ppo import completed

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='medqa_train_pool_expansion_v1'


def validate_expanded_tensors(data,meta,split,expected_ids):
    if split not in ('train','validation') or meta['split']!=split:
        raise ValueError('Only declared train/validation datasets')
    if (meta['source_ids']!=list(expected_ids) or len(set(expected_ids))!=len(expected_ids)
            or any(f':{split}:' not in sid for sid in expected_ids)
            or meta['allowed_for_gradient_updates']!=(split=='train')
            or meta['features_are_raw'] is not True or meta['seeds']!=[42,43,44]):
        raise ValueError('Dataset split/identity/schema firewall failed')
    required={'features','correctness','correctness_samples'}
    if split=='train': required|={'actions','q_targets'}
    if set(data)!=required:
        raise ValueError('Unexpected fields, including training targets in validation')
    x,q,samples=data['features'],data['correctness'],data['correctness_samples']
    n=len(expected_ids)
    if (x.ndim!=3 or x.shape[:2]!=(n,7) or x.shape[-1]<8 or x.dtype!=torch.float32 or (x.shape[-1]-5)%3
            or q.shape!=(n,8) or samples.shape!=(n,8,3)
            or not torch.isfinite(x).all() or not torch.isfinite(q).all()
            or not ((samples==0)|(samples==1)).all()):
        raise ValueError('Invalid feature/outcome tensors')
    expected_k=torch.arange(1,8,dtype=x.dtype,device=x.device)/8
    if not torch.equal(x[...,-1],expected_k.expand(n,7)):
        raise ValueError('Raw k/80 metadata changed')
    torch.testing.assert_close(q,samples.float().mean(-1),rtol=0,atol=0)
    if split=='train':
        future=q.flip(1).cummax(1).values.flip(1)
        targets=torch.stack((q[:,:-1],future[:,1:]),dim=-1)
        torch.testing.assert_close(data['q_targets'],targets,rtol=0,atol=0)
        if not torch.equal(data['actions'],(targets[:,:,0]<targets[:,:,1]).long()):
            raise ValueError('Wrong fraction/tie action targets')
    return data


def load_expanded_dataset(profile,split):
    if split not in ('train','validation'):
        raise ValueError('Only declared train/validation datasets')
    profile=contained(profile,ROOT)
    finish=json.loads((profile/'finished.json').read_text())
    plan=json.loads((profile/'plan.json').read_text())
    if (finish.get('status')!='complete' or finish.get('verified') is not True
            or plan['schema']!=SCHEMA or finish['plan_sha256']!=file_hash(profile/'plan.json')):
        raise ValueError('Require complete verified versioned expansion')
    verify_hashes(profile/'source_snapshot',plan['source_sha256'])
    path=completed(profile/split/'dataset')
    if path is None: raise ValueError('Dataset is not committed')
    path=contained(path,profile)
    meta=json.loads((path/'manifest.json').read_text())
    if meta['schema']!=SCHEMA or meta['plan_sha256']!=finish['plan_sha256']:
        raise ValueError('Dataset lineage differs')
    if (file_hash(path/'dataset.safetensors')!=meta['file_sha256']
            or meta['file_sha256']!=finish['exports'][split]['sha256']):
        raise ValueError('Dataset checksum differs')
    ids=plan['selection']['source_ids'] if split=='train' else plan['validation_ids']
    data=load_file(str(path/'dataset.safetensors'))
    return validate_expanded_tensors(data,meta,split,ids),meta
