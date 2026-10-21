"""Controller data/feature/observer/archive primitives for frozen A->B profiling.

No controller training and no change to the legacy MedQA loader.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unicodedata

import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file

GRID = tuple(range(10, 81, 10))
PHI_SCHEMA = 'phi4_checkpoint_v1'
PHI_NAMES = ['cosine_prev_step', 'relative_l2_prev_checkpoint',
             'mean_relative_l2_block10', 'trend_mean_relative_l2_block10']
EPS = 1e-8


class ProfileWorkerInspector:
    """Read-only named RPC extension; no pickle serialization of callbacks."""
    def profile_inspect_weights(self):
        model = self.model_runner.model
        config = self.vllm_config.model_config
        return dict(embedding_dtype=str(model.model.embed_tokens.weight.dtype),
            head_dtype=str(model.lm_head.weight.dtype),engine_dtype=str(config.dtype),
            model_revision=getattr(config.hf_config,'_commit_hash',None),
            model_source=config.model,quantization=config.quantization,
            logits_processor=model.logits_processor.__class__.__name__)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(list(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def normalized_question(question):
    return ' '.join(unicodedata.normalize('NFKC', question).casefold().split())


def load_official_medqa(data_dir, split='train'):
    """Read/hash-check downloaded official exports; render options exactly once."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Expected an official MedQA split')
    root = Path(data_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    metadata = manifest['splits'][split]
    path = root/metadata['file']
    if file_hash(path) != metadata['sha256']:
        raise ValueError(f'Dataset checksum differs: {path}')
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != metadata['rows']:
        raise ValueError('Dataset row count differs')
    items, ids = [], set()
    for index, row in enumerate(rows):
        options = row['options']
        label = row['answer_idx']
        if (set(options) != set('ABCD') or label not in options or
                row['answer'] != options[label] or row['source_split'] != split or
                row['source_row'] != index or row['source_id'] in ids):
            raise ValueError(f'Invalid MedQA schema/label at row{index}')
        ids.add(row['source_id'])
        stem = row['question'].strip()
        question = stem + '\n\n' + '\n'.join(f'{key}. {options[key]}' for key in 'ABCD')
        items.append(dict(source_id=row['source_id'], source_split=split, source_row=index,
                          question=question, question_stem=stem,
                          question_sha256=row['question_sha256'], gold=label.lower(),
                          solution=label.lower()))
    return items, metadata


def make_subset(items, count=32, seed=42):
    distinct = {}
    for item in items:
        distinct.setdefault(normalized_question(item['question_stem']), item)
    pool = list(distinct.values())
    if not 1 <= count <= len(pool):
        raise ValueError('Invalid distinct question count')
    return random.Random(seed).sample(pool, count)


def phi4(hidden, k):
    """Causal FP32 features from last-layer h[0..k]; input remains unchanged."""
    if k not in GRID or hidden.ndim != 2 or len(hidden) < k+1:
        raise ValueError('Expected [steps,hidden_dim], checkpoint10..80, coverage through k')
    h = hidden[:k+1].detach().float()
    if not torch.isfinite(h).all():
        raise ValueError('Non-finite hidden states')
    norms = torch.linalg.vector_norm(h, dim=-1)
    changes = torch.linalg.vector_norm(h[1:]-h[:-1], dim=-1)/(norms[:-1]+EPS)
    mean = changes[k-10:k].mean()
    trend = mean-changes[k-20:k-10].mean() if k>=20 else mean.new_zeros(())
    result = torch.stack([F.cosine_similarity(h[k],h[k-1],dim=0,eps=EPS),
        torch.linalg.vector_norm(h[k]-h[k-10])/(norms[k-10]+EPS),mean,trend])
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite phi4')
    return result


def raw_observation(hidden, k):
    features = phi4(hidden,k)
    h = hidden[:k+1].detach().float()
    return torch.cat([h[k],h[0],h[k]-h[k-10],features,features.new_tensor([k/80])])


def tie_targets(correctness):
    if len(correctness) != len(GRID) or any(q not in (0,1,False,True) for q in correctness):
        raise ValueError('Need all8 observed correctness outcomes, no missing labels')
    values = list(map(int,correctness))
    future_best = values[:]
    for i in range(6,-1,-1):
        future_best[i] = max(values[i],future_best[i+1])
    actions = ['HANDOFF' if values[i]>=future_best[i+1] else 'CONTINUE_10' for i in range(7)]
    best = min(k for k,q in zip(GRID,values) if q==max(values))
    return dict(oracle_budget=best, oracle_correct=max(values),
                checkpoint_actions=dict(zip(map(str,GRID[:-1]),actions)),
                note='train-only hindsight targets; not inference inputs')


class TrajectoryCollector:
    """Single-request, read-only last-position capture; never retain autograd."""
    def __init__(self):
        self.states, self.latent_inputs = [], []

    def __call__(self, *, step, hidden_last_position, latent_input):
        if step != len(self.states) or hidden_last_position.ndim != 3 or hidden_last_position.shape[0] != 1:
            raise ValueError('Collector expects ordered steps and batch1 [1,layers,dim]')
        self.states.append(hidden_last_position[0].detach().to('cpu').clone().contiguous())
        if step == 0:
            if latent_input is not None:
                raise ValueError('Step0 has no latent input')
        else:
            if latent_input is None or latent_input.shape != (1,hidden_last_position.shape[-1]):
                raise ValueError('Missing/malformed actual latent input')
            self.latent_inputs.append(latent_input[0].detach().to('cpu').clone().contiguous())

    def tensors(self):
        if len(self.states) !=81 or len(self.latent_inputs)!=80:
            raise ValueError('Incomplete h0..h80/e1..e80 trajectory')
        states, inputs = torch.stack(self.states), torch.stack(self.latent_inputs)
        if not torch.isfinite(states).all() or not torch.isfinite(inputs).all():
            raise ValueError('Non-finite trajectory')
        return dict(hidden_last_position_all_layers=states, latent_inputs_actual=inputs)


def atomic_exclusive_bytes(path, payload):
    """Commit a new artifact without overwriting an existing user file."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp,path)  # FileExistsError is intentional, including on resume.
    finally:
        os.unlink(temp)


def write_json(path, payload):
    atomic_exclusive_bytes(path,(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())


def write_jsonl(path, rows):
    atomic_exclusive_bytes(path,(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows)).encode())


def write_tensors(path, tensors):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp = tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    os.close(fd)
    try:
        save_file({key:value.detach().cpu().contiguous() for key,value in tensors.items()},temp)
        os.link(temp,path)
    finally:
        os.unlink(temp)


def verify_file_manifest(root, checksums):
    for name,expected in checksums.items():
        if file_hash(Path(root)/name) != expected:
            raise ValueError(f'Artifact checksum mismatch: {name}')


def reconstruct_prefix(prompts, states, k):
    if k not in GRID or len(states['latent_inputs_actual'])<k:
        raise ValueError('Missing checkpoint prefix')
    return torch.cat([prompts['A_prompt_embeddings_actual'],states['latent_inputs_actual'][:k]],dim=0)


def archive_coverage(root, source_ids):
    complete, missing = [], []
    for source_id in source_ids:
        path = Path(root)/'trajectories'/source_id/'complete.json'
        if not path.exists():
            missing.append(source_id)
            continue
        marker = json.loads(path.read_text())
        attempt = path.parent/marker['attempt']
        verify_file_manifest(attempt,marker['checksums'])
        with (attempt/'b_outcomes.jsonl').open() as handle:
            outcomes = [json.loads(line) for line in handle]
        if [r['k'] for r in outcomes] != list(GRID) or any(r['status']!='complete' for r in outcomes):
            raise ValueError(f'Incomplete grid at {source_id}')
        tensors = load_file(str(attempt/'states.safetensors'))
        if tensors['hidden_last_position_all_layers'].shape[0]!=81 or tensors['latent_inputs_actual'].shape[0]!=80:
            raise ValueError('Incomplete state coverage')
        complete.append(source_id)
    return dict(complete=len(complete),missing=len(missing),complete_source_ids=complete,
                missing_source_ids=missing,expected_b_completions=len(source_ids)*8,
                completed_b_completions=len(complete)*8,all_complete=not missing)
