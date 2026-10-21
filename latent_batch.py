"""Padding-safe HF latent rollout and ragged vLLM prompt construction."""
import torch
import time


def synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def rollout(wrapper, input_ids=None, attention_mask=None, *, latent_steps, past_key_values=None,
            return_embeddings=True, inputs_embeds=None, timings=None, observer=None, stopping_policy=None):
    model = getattr(wrapper, 'HF_model', None)
    if model is None:
        model = wrapper.model
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError('Provide exactly one of input_ids or inputs_embeds')
    inputs = input_ids if input_ids is not None else inputs_embeds
    expected_ndim = 2 if input_ids is not None else 3
    if inputs.ndim != expected_ndim or latent_steps < 0:
        raise ValueError('Expected batched token IDs/embeddings and nonnegative latent_steps')
    shape = inputs.shape[:2]
    mask = (torch.ones(shape, dtype=torch.long, device=inputs.device) if attention_mask is None
            else attention_mask.to(device=inputs.device))
    if mask.shape != shape or not ((mask == 0) | (mask == 1)).all() or not mask.any(dim=1).all():
        raise ValueError("Each prompt needs a binary mask with at least one real token")
    mask = mask.long()
    batch = inputs.shape[0]
    if stopping_policy is not None and (batch != 1 or not callable(stopping_policy)):
        raise ValueError('Online stopping requires a callable and batch1')
    prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    if prefix_len:
        prefix_mask = getattr(past_key_values, "_latent_attention_mask", None)
        offsets = getattr(past_key_values, "_latent_next_position", None)
        if prefix_mask is None or offsets is None:
            raise ValueError("Continuing latent rollout requires saved mask/position metadata")
        prefix_mask, offsets = prefix_mask.to(mask.device), offsets.to(mask.device)
        if prefix_mask.shape != (batch, prefix_len):
            raise ValueError("Cache and prefix mask dimensions differ")
    else:
        prefix_mask = mask[:, :0]
        offsets = torch.zeros(batch, dtype=torch.long, device=mask.device)
    positions = offsets[:, None] + mask.long().cumsum(-1) - 1
    positions = positions.masked_fill(mask == 0, 0)
    full_mask = torch.cat([prefix_mask, mask], dim=1)
    if timings is not None:
        synchronize(inputs.device)
        started = time.perf_counter()
    outputs = model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=full_mask,
                               position_ids=positions, past_key_values=past_key_values,
                               use_cache=True, output_hidden_states=True, return_dict=True)
    if timings is not None:
        synchronize(inputs.device)
        timings['prefill_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
    past = outputs.past_key_values
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = indices.masked_fill(mask == 0, -1).max(-1).values
    hidden = outputs.hidden_states[-1][torch.arange(batch, device=mask.device), last]
    def notify(step, current_outputs, current_last, latent_input=None):
        if observer is None:
            return
        # Independent last-position snapshots, never full prefill activations.
        rows = torch.arange(batch, device=mask.device)
        states = torch.stack([h[rows, current_last].detach() for h in current_outputs.hidden_states], dim=1)
        result = observer(step=step, hidden_last_position=states.clone(),
                          latent_input=None if latent_input is None else latent_input.detach().clone())
        if result is not None:
            raise ValueError('An observer must return None; stopping is not an observer action')
    notify(0, outputs, last)
    # Strip padding per sample; vLLM accepts a list of different-length tensors.
    original_embeddings = inputs_embeds if inputs_embeds is not None else outputs.hidden_states[0]
    embeddings = ([[original_embeddings[i, mask[i].bool()].detach()] for i in range(batch)]
                  if return_embeddings else None)
    next_position = offsets + mask.sum(-1).long()
    for step in range(1, latent_steps + 1):
        latent = wrapper._apply_latent_realignment(hidden, model).unsqueeze(1)
        full_mask = torch.cat([full_mask, full_mask.new_ones((batch, 1))], dim=1)
        outputs = model(inputs_embeds=latent, attention_mask=full_mask,
                                   position_ids=next_position[:, None], past_key_values=past,
                                   use_cache=True, output_hidden_states=True, return_dict=True)
        past, hidden = outputs.past_key_values, outputs.hidden_states[-1][:, -1]
        notify(step, outputs, torch.zeros(batch, dtype=torch.long, device=mask.device), latent[:, 0])
        if embeddings is not None:
            for i in range(batch):
                embeddings[i].append(latent[i].detach())
        next_position = next_position + 1
        if stopping_policy is not None:
            should_stop = stopping_policy(step=step, hidden_last_position=hidden.detach().clone())
            if type(should_stop) is not bool:
                raise ValueError('Stopping policy must return bool')
            if should_stop:
                break
    past._latent_attention_mask = full_mask.detach()
    past._latent_next_position = next_position.detach()
    if timings is not None:
        synchronize(inputs.device)
        timings['latent_seconds'] = time.perf_counter() - started
    return past, ([torch.cat(parts, dim=0) for parts in embeddings] if embeddings is not None else None)


def insert_prefix(prompt_embeddings, prompt_mask, prefixes, insertion_indices):
    if not (len(prompt_embeddings) == len(prefixes) == len(insertion_indices) == len(prompt_mask)):
        raise ValueError("Prompt/prefix batch lengths differ")
    result = []
    for emb, mask, prefix, pos in zip(prompt_embeddings, prompt_mask, prefixes, insertion_indices):
        clean = emb[mask.to(emb.device).bool()]
        if not 0 <= pos <= len(clean):
            raise ValueError("Insertion index outside unpadded prompt")
        prefix = prefix.to(device=clean.device, dtype=clean.dtype)
        result.append({"prompt_embeds": torch.cat([clean[:pos], prefix, clean[pos:]], dim=0)})
    return result
