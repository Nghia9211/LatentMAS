"""Immutable archive relocation and train-only three-seed targets for new runs."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from controller.profile import file_hash
from scripts.migrate_b_seed_repeats import contained, verify_hashes, relocate_config, runtime_fingerprint

SCHEMA = 'h100_train_three_seed_v1'


def repeat_targets(rows, selections):
    """Aggregate by identity/K; validation has no action or Q training targets."""
    grids = {s: {} for s in ('train', 'validation')}
    for split in grids:
        ids = selections[split]['source_ids']
        if len(ids) != len(set(ids)) or any(f':{split}:' not in sid for sid in ids):
            raise ValueError('Invalid selected source IDs')
        grids[split] = {sid: {} for sid in ids}
    for row in rows:
        split, sid, k = row['source_split'], row['source_id'], row['k']
        if split not in grids or sid not in grids[split] or k not in range(10, 81, 10):
            raise ValueError('Unexpected repeat identity/split/K')
        values = row['correctness']
        if (row['seeds'] != [42, 43, 44] or row['trials'] != 3 or len(values) != 3
                or any(v not in (0, 1) for v in values)
                or row['correct_count'] != sum(values)
                or abs(row['success_fraction'] - sum(values)/3) > 1e-12
                or k in grids[split][sid]):
            raise ValueError('Invalid or duplicate three-seed outcome')
        grids[split][sid][k] = sum(values)/3
    result = {}
    for split, by_id in grids.items():
        if any(set(v) != set(range(10, 81, 10)) for v in by_id.values()):
            raise ValueError('Incomplete repeat grid')
        q = torch.tensor([[by_id[sid][k] for k in range(10,81,10)]
                          for sid in selections[split]['source_ids']], dtype=torch.float32)
        result[split] = dict(correctness=q)
        if split == 'train':
            best = q.flip(1).cummax(1).values.flip(1)
            targets = torch.stack((q[:, :-1], best[:, 1:]), dim=-1)
            result[split].update(q_targets=targets, actions=(targets[:,:,0] < targets[:,:,1]).long())
    return result


def repeat_evidence(repo, profile, pm, campaign):
    repo = Path(repo).resolve()
    profile, campaign = contained(profile, repo), contained(campaign, repo)
    plan = json.loads(contained(campaign/'plan.json', repo).read_text())
    if (contained(plan['profile'], repo) != profile
            or plan['selected_source_ids'] != {s:pm['selections'][s]['source_ids'] for s in ('train','validation')}
            or plan['seeds'] != [43,44] or plan['grid'] != list(range(10,81,10))):
        raise ValueError('Repeat/profile lineage mismatch')
    if plan.get('migration',{}).get('historical_profile_manifest_sha256') != file_hash(profile/'manifest.json'):
        raise ValueError('Repeat parent manifest changed')
    for key in ('model_name','model_revision','vllm_dtype','temperature','top_p','max_new_tokens',
                'generate_bs','seed','think','latent_space_realign','num_agents','prompt','method'):
        if plan['config'][key] != pm['config'][key]:
            raise ValueError('Repeat numerical protocol mismatch: '+key)
    verify_hashes(profile/'source_snapshot', pm['source_sha256'])
    verify_hashes(campaign/'source_snapshot', plan['source_sha256'])
    verify_hashes(profile, plan['reference_files_sha256'])
    finished = json.loads(contained(campaign/'finished.json', repo).read_text())
    if finished.get('status') != 'complete' or finished.get('verified') is not True:
        raise ValueError('Require completed verified repeats')
    hashes = {'comparison.json':finished['comparison_sha256'],
              'per_question_K_successes.jsonl':finished['per_question_sha256'],
              'report.md':finished['report_sha256']}
    verify_hashes(campaign, hashes)
    from scripts.profile_b_seed_repeats import validate_new_rows
    rows = [json.loads(line) for line in (campaign/'per_question_K_successes.jsonl').read_text().splitlines()]
    targets = repeat_targets(rows, pm['selections'])
    # Check aggregates against committed rows, not just against their own summary.
    from scripts.run_matched_controller_ppo import completed
    for split in ('train','validation'):
        for j,k in enumerate(range(10,81,10)):
            truth = []
            for g in range(len(pm['selections'][split]['source_ids'])//16):
                batches = []
                for seed in (42,43,44):
                    base = profile/split/'profile' if seed == 42 else campaign/f'seed{seed}'/'primary'/split
                    path = completed(base/f'K{k}'/f'cohort{g:03d}')
                    if path is None:
                        raise ValueError('Missing committed seed outcome')
                    path = contained(path,repo)
                    saved_rows = list(map(json.loads,contained(path/'outcomes.jsonl',repo).read_text().splitlines()))
                    expected_ids = pm['selections'][split]['source_ids'][g*16:(g+1)*16]
                    if [r['source_id'] for r in saved_rows] != expected_ids:
                        raise ValueError('Committed cohort identity/order mismatch')
                    if seed == 42:
                        reference_rows = saved_rows
                    else:
                        validate_new_rows(saved_rows, reference_rows, seed)
                    batches.append([int(r['strict_correct']) for r in saved_rows])
                truth.extend(sum(values)/3 for values in zip(*batches))
            torch.testing.assert_close(targets[split]['correctness'][:,j], torch.tensor(truth),rtol=0,atol=0)
    evidence = dict(schema=SCHEMA, campaign=str(campaign),
        plan_sha256=file_hash(campaign/'plan.json'), finished_sha256=file_hash(campaign/'finished.json'),
        result_sha256=hashes, historical_source_sha256=pm['source_sha256'],
        runtime=runtime_fingerprint(), mixed_GB10_H100=True, hardware_parity_established=False,
        supervision='train mean(strict_correct seed42/43/44); H=q(k), C=max future mean; ties HANDOFF',
        live_PPO_reward='binary actual live B strict correctness; no time/step penalty',
        train_unique_questions=len(targets['train']['correctness']),
        validation_labels_used_for_gradients=False, archive_rewritten=False)
    return targets,evidence


def current_sources(repo, config):
    """Freeze current execution separately from archived source snapshots."""
    repo = Path(repo).resolve()
    names = set()
    for sub in ('controller','scripts','methods','tests'):
        for p in contained(repo/sub,repo).glob('*.py'):
            names.add(str(contained(p,repo).relative_to(repo)))
    names.update(str(p.relative_to(repo)) for p in repo.glob('*.py'))
    names.add(str(contained(config,repo).relative_to(repo)))
    return {n:file_hash(contained(repo/n,repo)) for n in sorted(names)}


def migrate_training(repo, profile, pm, settings, config, data):
    repo = Path(repo).resolve()
    m = settings['migration']
    required = {'schema','repeat_campaign','data_root','hf_device','gpu_memory_utilization'}
    if not required <= set(m) or set(m) - required - {'vllm_options','hf_memory_reservation_gib'} or m['schema'] != SCHEMA:
        raise ValueError('Unexpected training migration schema')
    targets,evidence = repeat_evidence(repo,profile,pm,repo/m['repeat_campaign'])
    cfg = relocate_config(pm['config'],repo,contained(repo/m['data_root'],repo),
                          m['gpu_memory_utilization'],m['hf_device'])
    cfg['device'] = cfg['device2']  # Controller observations remain on A/HF GPU.
    if 'hf_memory_reservation_gib' in m:
        size=m['hf_memory_reservation_gib']
        if isinstance(size,bool) or not isinstance(size,(int,float)) or not 0 < size <= 32:
            raise ValueError('Invalid HF memory reservation')
        cfg['hf_memory_reservation_gib']=size
    options = m.get('vllm_options',{})
    if set(options) - {'max_model_len','max_num_seqs','max_num_batched_tokens'}:
        raise ValueError('Unknown vLLM runtime option')
    for key,value in options.items():
        if type(value) is not int or value < 1:
            raise ValueError('Invalid vLLM runtime option')
        cfg['vllm_'+key] = value
    d = dict(data)
    d.update(targets['train'])
    baseline = {str(k):dict(accuracy=float(targets['validation']['correctness'][:,j].mean()))
                for j,k in enumerate(range(10,81,10))}
    sources = current_sources(repo,config)
    evidence.update(data_root=cfg['data_dir'], archived_realignment_sha256=file_hash(Path(profile)/'shared/realignment.safetensors'),
        baseline_source='Verified three-seed fixed archived A, mixed GB10/H100; different engine lifetimes',
        fixed_validation_baselines=baseline)
    return cfg,d,sources,evidence


def bind_archived_realignment(wrapper, profile, expected_hash):
    path = Path(profile)/'shared/realignment.safetensors'
    if file_hash(path) != expected_hash:
        raise ValueError('Archived realignment changed')
    archive = load_file(str(path))
    matrix,norm = wrapper._ensure_latent_realign_matrix(wrapper.HF_model,wrapper.HF_device,wrapper.args)
    if (matrix.shape != archive['matrix'].shape or norm.shape != archive['target_norm'].shape
            or not torch.isfinite(archive['matrix']).all() or not torch.isfinite(archive['target_norm']).all()):
        raise ValueError('Invalid archived realignment')
    exact = torch.equal(matrix.cpu(),archive['matrix']) and torch.equal(norm.cpu(),archive['target_norm'])
    wrapper._latent_realign_matrices[id(wrapper.HF_model)] = (
        archive['matrix'].to(wrapper.HF_device),archive['target_norm'].to(wrapper.HF_device))
    return dict(archived_realignment_reused=True,destination_recomputed_realignment_exact=exact,
                recomputed_matrix_max_abs_error=float((matrix.cpu()-archive['matrix']).abs().max()),
                hardware_parity_established=False)
