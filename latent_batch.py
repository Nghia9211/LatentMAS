"""Padding-safe HF latent rollout and ragged vLLM prompt construction."""
import torch


@torch.no_grad()
def rollout(wrapper, input_ids, attention_mask=None, *, latent_steps, past_key_values=None,
            return_embeddings=True):
    model = getattr(wrapper, 'HF_model', None)
    if model is None:
        model = wrapper.model
    if input_ids.ndim != 2 or latent_steps < 0:
        raise ValueError("Expected [batch, tokens] inputs and nonnegative latent_steps")
    mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask.to(input_ids.device)
    if mask.shape != input_ids.shape or not ((mask == 0) | (mask == 1)).all() or not mask.any(dim=1).all():
        raise ValueError("Each prompt needs a binary mask with at least one real token")
    batch = input_ids.shape[0]
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
    outputs = model(input_ids=input_ids, attention_mask=full_mask,
                               position_ids=positions, past_key_values=past_key_values,
                               use_cache=True, output_hidden_states=True, return_dict=True)
    past = outputs.past_key_values
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = indices.masked_fill(mask == 0, -1).max(-1).values
    hidden = outputs.hidden_states[-1][torch.arange(batch, device=mask.device), last]
    # Strip padding per sample; vLLM accepts a list of different-length tensors.
    embeddings = ([[outputs.hidden_states[0][i, mask[i].bool()].detach()] for i in range(batch)]
                  if return_embeddings else None)
    next_position = offsets + mask.sum(-1).long()
    for _ in range(latent_steps):
        latent = wrapper._apply_latent_realignment(hidden, model).unsqueeze(1)
        full_mask = torch.cat([full_mask, full_mask.new_ones((batch, 1))], dim=1)
        outputs = model(inputs_embeds=latent, attention_mask=full_mask,
                                   position_ids=next_position[:, None], past_key_values=past,
                                   use_cache=True, output_hidden_states=True, return_dict=True)
        past, hidden = outputs.past_key_values, outputs.hidden_states[-1][:, -1]
        if embeddings is not None:
            for i in range(batch):
                embeddings[i].append(latent[i].detach())
        next_position = next_position + 1
    past._latent_attention_mask = full_mask.detach()
    past._latent_next_position = next_position.detach()
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
