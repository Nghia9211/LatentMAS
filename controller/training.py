"""Shared controller update primitives, independent of retired benchmark CLIs."""
import statistics

import torch
from torch.nn import functional as F

from controller.ppo import sample_episodes, ppo_loss
from controller.warmstart import check_gradients, warmstart_loss


def train_update(model, optimizer, x, actions, targets, config):
    optimizer.zero_grad(set_to_none=True)
    actor, q = model(x)
    loss, _ = warmstart_loss(actor, q, actions, targets, config['q_weight'])
    loss.backward()
    # Check the true un-clipped gradients. These finite/nonzero checks and the
    # scalar reads are part of the instrumented timing, consistently for all.
    if not torch.isfinite(loss) or not check_gradients(model):
        raise FloatingPointError('Non-finite loss or zero gradient')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def replay_iteration(model,optimizer,x,correctness,ties,config):
    episodes=sample_episodes(model,x,correctness)
    diagnostics=[]
    for _ in range(config['optimization_epochs']):
        order=torch.randperm(len(x),device=x.device)
        for start in range(0,len(x),config['minibatch_questions']):
            index=order[start:start+config['minibatch_questions']]
            sample={key:value[index] for key,value in episodes.items()}
            optimizer.zero_grad(set_to_none=True)
            loss,parts=ppo_loss(model,x[index],sample,ties[index],config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            loss.backward()
            if not check_gradients(model):
                raise FloatingPointError('Zero PPO gradients')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            diagnostics.append(dict(loss=float(loss.detach()),unclipped_gradient_norm=float(norm),
                **{key:float(value.detach()) for key,value in parts.items()}))
    return episodes,dict(sampled_correct=int(episodes['rewards'].sum()),
        sampled_mean_budget=float(((episodes['stop_indices']+1)*10).float().mean()),
        visited_decisions=int(episodes['visited'].sum()),updates=len(diagnostics),
        **{key:statistics.mean(row[key] for row in diagnostics) for key in diagnostics[0]})


def logged_ppo_update(model, optimizer, x, episode, teacher_x, ties, cfg, on_step=None):
    pieces = []
    local = dict(cfg, tie_weight=0)
    for epoch in range(cfg['optimization_epochs']):
        order = torch.randperm(32, device=x.device)
        for start in range(0, 32, cfg['minibatch_questions']):
            index = order[start:start + cfg['minibatch_questions']]
            data = {k: v[index] for k, v in episode.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, parts = ppo_loss(model, x[index], data, torch.zeros_like(data['visited']), local)
            # A zero coefficient removes teacher forwards and label access entirely.
            tie = loss.new_zeros(())
            if cfg['tie_weight']:
                logits, _ = model(teacher_x[index])
                mask = ties[index]
                tie = F.cross_entropy(logits[mask], torch.zeros(int(mask.sum()), dtype=torch.long,
                    device=x.device)) if mask.any() else logits.sum() * 0
                loss = loss + cfg['tie_weight'] * tie
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            loss.backward()
            if not check_gradients(model):
                raise FloatingPointError('Zero PPO gradients')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            value = float(loss.detach())
            pieces.append(value)
            if on_step is not None:
                on_step(len(pieces), dict(
                    **{'loss/total': value, 'loss/policy': float(parts['actor'].detach()),
                       'loss/value': float(parts['value'].detach()),
                       'loss/tie': float(tie.detach()),
                       'loss/entropy_term': -cfg['entropy_weight'] * float(parts['entropy'].detach()),
                       'ppo/entropy': float(parts['entropy'].detach()),
                       'ppo/approx_kl': float(parts['approx_kl'].detach()),
                       'ppo/clip_fraction': float(parts['clip_fraction'].detach()),
                       'ppo/gradient_norm': float(norm.detach()),
                       'ppo/epoch': epoch + 1,
                       'ppo/minibatch': start // cfg['minibatch_questions'] + 1}))
    return sum(pieces) / len(pieces), len(pieces)
