"""Schema-specific inference loading without fitting statistics."""
import torch
from controller.ppo import PPOController
from controller.normalized import load_normalized_checkpoint, SCHEMA as NORMALIZED
from controller.uniform_explore import load_uniform_checkpoint, SCHEMA as UNIFORM
from controller.simple_state import load_simple_checkpoint, SCHEMA as SIMPLE
from controller.cosine_state import load_cosine_checkpoint, SCHEMA as COSINE


def load_controller_checkpoint(path, device='cpu', *, expected_schema=None):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    state = saved['model']
    if 'uniform_budget_version' in state:
        schema,loader = UNIFORM,load_uniform_checkpoint
    elif 'normalizer.cosine_mean' in state:
        schema,loader = COSINE,load_cosine_checkpoint
    elif 'normalizer.l2_mean' in state:
        schema,loader = SIMPLE,load_simple_checkpoint
    elif any(k.startswith('normalizer.') for k in state):
        schema,loader = NORMALIZED,load_normalized_checkpoint
    else:
        schema,loader = 'raw',None
    if expected_schema is not None and expected_schema != schema:
        raise ValueError('Checkpoint schema differs from frozen run')
    if loader is not None:
        return loader(path,device)
    width,dim = state['encoder.0.weight'].shape
    model = PPOController(dim,width)
    model.load_state_dict(state)
    return model.to(device).eval()
