"""Aggregate causal rollout diagnostics; only scalar summaries leave the server."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from controller.profile import verify_file_manifest


def tensor_stats(prefix, values):
    values = values.detach().float().reshape(-1)
    if not len(values):
        return {}
    if not torch.isfinite(values).all():
        raise ValueError('Non-finite rollout diagnostic')
    return {f'{prefix}/{name}': float(value) for name, value in dict(
        mean=values.mean(), std=values.std(unbiased=False), min=values.min(), max=values.max()).items()}


def rollout_metrics(episode):
    visited = episode['visited'].detach().bool()
    actions = episode['actions'].detach()
    if visited.ndim != 2 or visited.shape[1] != 7 or not visited[:, 0].all():
        raise ValueError('Expected causal K10..70 decision mask')
    if ((actions[visited] < 0) | (actions[visited] > 1)).any():
        raise ValueError('Expected binary controller actions')
    n = len(visited)
    budgets = torch.full((n,), 80, device=actions.device, dtype=torch.long)
    for t in range(7):
        stop = visited[:, t] & (actions[:, t] == 0)
        budgets[stop] = (t + 1) * 10
        expected = budgets > (t + 1) * 10
        if t < 6 and not torch.equal(visited[:, t + 1], expected):
            raise ValueError('Non-causal visited mask or post-handoff decisions')
    metrics = {'rollout/visited_decisions': int(visited.sum()),
               'rollout/episodes': n, 'budget/mean_K': float(budgets.float().mean())}
    for k in range(10, 81, 10):
        count = int((budgets == k).sum())
        metrics.update({f'budget/K{k}_count': count, f'budget/K{k}_fraction': count / n})
    selected_p = episode['old_logprobs'].detach().float().exp()
    handoff_p = torch.where(actions == 0, selected_p, 1 - selected_p)
    for t in range(7):
        mask = visited[:, t]
        count = int(mask.sum())
        k = (t + 1) * 10
        metrics[f'handoff/K{k}_reached'] = count
        if count:
            metrics[f'handoff/K{k}_mean_probability'] = float(handoff_p[mask, t].mean())
            metrics[f'handoff/K{k}_observed_fraction'] = float((actions[mask, t] == 0).float().mean())
    metrics['handoff/K80_reached'] = int((budgets == 80).sum())
    metrics['handoff/K80_forced_count'] = metrics['handoff/K80_reached']
    for field, prefix in [('old_values', 'rollout/value'), ('advantages', 'rollout/advantage'),
                          ('returns', 'rollout/return')]:
        metrics.update(tensor_stats(prefix, episode[field][visited]))
    returns = episode['returns'][visited].detach().float()
    values = episode['old_values'][visited].detach().float()
    variance = returns.var(unbiased=False)
    if float(variance) > 0:
        metrics['rollout/value_explained_variance'] = float(1 - (returns - values).var(unbiased=False) / variance)
    return metrics


def outcome_metrics(rows):
    if not rows:
        return {}
    if any(r['source_split'] != 'train' for r in rows):
        raise ValueError('Train diagnostics cannot consume test/validation outcomes')
    tokens = torch.tensor([r['output_tokens'] for r in rows], dtype=torch.float64)
    n = len(rows)
    return {'output/B_tokens_total': int(tokens.sum()), 'output/B_tokens_mean': float(tokens.mean()),
            'output/B_tokens_p50': float(torch.quantile(tokens, .5)),
            'output/B_tokens_p95': float(torch.quantile(tokens, .95)),
            'output/B_tokens_max': int(tokens.max()),
            'output/cap_rate': sum(r['finish_reason'] == 'length' for r in rows) / n,
            'output/parse_failure_rate': sum(r['strict_parse_failed'] for r in rows) / n,
            'output/loop_rate': sum(r['repetition']['loop_flag'] for r in rows) / n,
            'data/episodes_this_iteration': n,
            'data/unique_questions_this_iteration': len({r['source_id'] for r in rows})}


def committed_path(base):
    base = Path(base)
    marker = base / 'complete.json'
    if not marker.exists():
        return None
    meta = json.loads(marker.read_text())
    path = base / meta['attempt']
    verify_file_manifest(path, meta['checksums'])
    return path


def read_outcomes(path):
    rows = []
    for file in sorted(Path(path).glob('cohort*/outcomes.jsonl')):
        rows.extend(json.loads(s) for s in file.read_text().splitlines())
    return rows


def prior_coverage(training, through=None):
    ids, episodes = set(), 0
    for n in range(1, (200 if through is None else through) + 1):
        path = committed_path(Path(training) / 'train' / f'iteration{n:03d}')
        if path is None:
            break
        rows = read_outcomes(path)
        if any(r['source_split'] != 'train' for r in rows):
            raise ValueError('Non-train episodes in coverage')
        ids.update(r['source_id'] for r in rows)
        episodes += len(rows)
    return ids, episodes


def saved_diagnostics(training):
    """Backfill only diagnostics actually recoverable from committed artifacts."""
    training = Path(training)
    manifest = json.loads((training / 'manifest.json').read_text())
    aux_ids = manifest.get('train_manifest', {}).get('source_ids', [])
    events = [json.loads(s) for s in (training / 'events.jsonl').read_text().splitlines()]
    b_times, pending = {}, []
    for event in events:
        if event['stage'] == 'B_cohort_complete' and event.get('purpose') == 'live_train_reward':
            pending.append(event)
        if event['stage'] == 'PPO_iteration_complete':
            b_times[event['iteration']] = pending
            pending = []
    result, seen, episodes = [], set(), 0
    for n in range(1, 201):
        path = committed_path(training / 'train' / f'iteration{n:03d}')
        if path is None:
            break
        tensor = load_file(str(path / 'episodes.safetensors'))
        rows = read_outcomes(path)
        details = json.loads((path / 'result.json').read_text())
        if len(rows) != details['episodes']:
            raise ValueError('Incomplete committed outcomes')
        seen.update(r['source_id'] for r in rows)
        episodes += len(rows)
        metrics = dict(iteration=n, **rollout_metrics(tensor), **outcome_metrics(rows))
        metrics.update({'data/episodes_seen': episodes, 'data/unique_PPO_questions_seen': len(seen),
                        'data/unique_auxiliary_questions': len(set(aux_ids)),
                        'validation/available': 0, 'time/PPO_update_timing_available': 0})
        timings = b_times.get(n, [])
        if len(timings) == 2 and sum(e['tokens'] for e in timings) == sum(r['output_tokens'] for r in rows):
            b_seconds = sum(e['seconds'] for e in timings)
            metrics['time/B_generate_seconds'] = b_seconds
            metrics['time/non_B_seconds'] = details['seconds_including_live_A_B'] - b_seconds
        for k in ('cohort_physical_steps',):
            if k in details:
                metrics['budget/cohort_physical_steps_max'] = max(details[k])
        result.append(metrics)
    return result


def budget_patterns(samples, grid):
    counts = {str(k): dict(first_correct=0, uniquely_correct=0, oracle_selected=0) for k in grid}
    categories = dict(all_correct=0, all_wrong=0, mixed=0)
    transitions = dict(wrong_to_correct=0, correct_to_wrong=0)
    action_labels = dict(HANDOFF=0, CONTINUE_10=0)
    details = []
    for sample in samples:
        bits = [int(row['strict_correct']) for row in sample['rows']]
        assert len(bits) == len(grid) and all(bit in (0, 1) for bit in bits)
        correct_budgets = [k for k, bit in zip(grid, bits) if bit]
        category = 'all_correct' if all(bits) else 'all_wrong' if not any(bits) else 'mixed'
        categories[category] += 1
        oracle = correct_budgets[0] if correct_budgets else grid[0]
        counts[str(oracle)]['oracle_selected'] += 1
        if correct_budgets:
            counts[str(correct_budgets[0])]['first_correct'] += 1
        if len(correct_budgets) == 1:
            counts[str(correct_budgets[0])]['uniquely_correct'] += 1
        for low, high in zip(bits, bits[1:]):
            transitions['wrong_to_correct'] += int(low == 0 and high == 1)
            transitions['correct_to_wrong'] += int(low == 1 and high == 0)
        for i, bit in enumerate(bits[:-1]):
            action_labels['HANDOFF' if bit >= max(bits[i+1:]) else 'CONTINUE_10'] += 1
        details.append(dict(source_id=sample['source_id'], correctness=bits,
                            correct_budgets=correct_budgets, category=category,
                            hindsight_budget=oracle))
    n = len(samples)
    return dict(categories=categories, per_budget=counts, adjacent_transitions=transitions,
                train_only_checkpoint_action_labels=action_labels,
                hindsight_correct=n-categories['all_wrong'],
                hindsight_accuracy=(n-categories['all_wrong'])/n if n else None,
                hindsight_mean_steps=sum(row['hindsight_budget'] for row in details)/n if n else None,
                per_question=details)
