#!/usr/bin/env python3
"""Evaluate one validation-selected frozen controller on the ordered MedQA300."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from controller.profile import (file_hash, write_json, write_jsonl, write_tensors,
    load_official_medqa, normalized_question, atomic_exclusive_bytes)
from controller.matched import (historical_style, render_batch, CohortDecisions,
    online_rollout, prefixes_for_b)
from controller.checkpoints import load_controller_checkpoint
from controller.server_training import current_sources, bind_archived_realignment
from scripts.migrate_b_seed_repeats import contained, verify_hashes
from scripts.run_matched_controller_ppo import completed, new_attempt, commit, reset_cache, event


def match_items(historical, official, reference_rows):
    if len(historical) != len(reference_rows):
        raise ValueError('Reference row count mismatch')
    lookup = defaultdict(list)
    for item in official:
        lookup[normalized_question(item['question_stem'])].append(item)
    selected = []
    for index, (row, reference) in enumerate(zip(historical, reference_rows)):
        stem = row['question'].rstrip()
        options = '\n'.join(row['options'])
        while stem.endswith(options):
            stem = stem[:-len(options)].rstrip()
        matches = [item for item in lookup[normalized_question(stem)]
                   if item['question'].rsplit('\n\n', 1)[-1] == options]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous/unmatched historical row {index}')
        item = dict(matches[0])
        if row['options']['abcd'.index(item['gold'])][3:] != row['answer']:
            raise ValueError(f'Historical gold mismatch at {index}')
        question = historical_style(item)
        if question != reference['question'] or item['gold'] != reference['gold']:
            raise ValueError(f'Historical reference mismatch at {index}')
        item.update(question=question, historical_index=index)
        selected.append(item)
    if len({item['source_id'] for item in selected}) != len(selected):
        raise ValueError('Repeated official test identity')
    if any(item['source_split'] != 'test' for item in selected):
        raise ValueError('Expected official test identities')
    return selected


def cohort_items(items):
    return [items[start:start + 16] for start in range(0, len(items), 16)]


def selected_checkpoint(training, iteration, expected_hash):
    finished = json.loads((training / 'training_finished.json').read_text())
    if (finished.get('status') != 'complete' or finished.get('test300_used') is not False
            or finished['best_validation']['iteration'] != iteration):
        raise ValueError('Use the completed run best-validation checkpoint selected before test')
    path = completed(training / 'train' / f'iteration{iteration:03d}')
    if path is None:
        raise ValueError('Requested checkpoint is uncommitted')
    result = json.loads((path / 'result.json').read_text())
    if result['iteration'] != iteration or result['train_only'] is not True or result['episodes'] != 32:
        raise ValueError('Checkpoint training provenance mismatch')
    checkpoint = path / 'checkpoint.pt'
    if file_hash(checkpoint) != expected_hash:
        raise ValueError('Requested checkpoint hash mismatch')
    return checkpoint


@torch.no_grad()
def decode_test_cohort(wrapper, args, items, payloads, budgets, root, path):
    from vllm import SamplingParams
    from utils import extract_gsm8k_answer, normalize_answer, medqa_scoring_metadata
    from scripts.audit_saved_loop_budgets import repetition
    count = len(items)
    if not 1 <= count <= 16 or len(payloads) != count or len(budgets) != count:
        raise ValueError('Invalid test cohort size')
    if any(item['source_split'] != 'test' for item in items):
        raise ValueError('Test-only decoder')
    params = SamplingParams(temperature=.6, top_p=.95, top_k=0, seed=42, max_tokens=4096)
    event(root, stage='B_cohort_start', purpose='test300_PPO', samples=count, budgets=budgets)
    started = time.perf_counter()
    outputs = wrapper.vllm_engine.generate(payloads, params, use_tqdm=False)
    seconds = time.perf_counter() - started
    if len(outputs) != count:
        raise RuntimeError('Incomplete test cohort')
    rows = []
    from controller.profile import tensor_hash
    for item, payload, out, k in zip(items, payloads, outputs, budgets):
        output = out.outputs[0]
        ids = list(output.token_ids)
        pred = normalize_answer(extract_gsm8k_answer(output.text))
        metric = medqa_scoring_metadata(output.text, item['gold'], pred, pred == item['gold'])
        strict = metric['medqa_boxed_v1']
        rows.append(dict(source_id=item['source_id'], source_split='test',
            historical_index=item['historical_index'], k=k, gold=item['gold'],
            raw_prediction=output.text, generated_token_ids=ids, output_tokens=len(ids),
            finish_reason=output.finish_reason, stop_reason=output.stop_reason,
            strict_prediction=strict['prediction'], strict_correct=strict['correct'],
            strict_parse_failed=strict['parse_failed'], legacy_prediction=pred,
            legacy_correct=metric['legacy_correct'], repetition=repetition(ids),
            temperature=.6, decoder_seed=42, batch=count,
            prompt_sha256=tensor_hash(payload['prompt_embeds']),
            cached_tokens=getattr(out, 'num_cached_tokens', None)))
    write_jsonl(path / 'outcomes.jsonl', rows)
    event(root, stage='B_cohort_complete', samples=count, correct=sum(r['strict_correct'] for r in rows),
          seconds=seconds, tokens=sum(r['output_tokens'] for r in rows))
    return rows, seconds


@torch.no_grad()
def evaluate(wrapper, args, root, model, items, reference_rows, iteration):
    reset_cache(wrapper)
    rows = []
    for index, group in enumerate(cohort_items(items)):
        base = root / 'evaluation' / f'cohort{index:03d}'
        if (base / 'complete.json').exists():
            raise ValueError('Unexpected previous cohort; no automatic retry/resume')
        path = new_attempt(base)
        texts, encoded = render_batch(wrapper, args, group)
        for j, item in enumerate(group):
            reference = reference_rows[item['historical_index']]
            for key, role_index in [('A', 0), ('B', 1)]:
                if texts[key][j] != reference['agents'][role_index]['input']:
                    raise ValueError(f'Rendered {key} prompt mismatch at {item["historical_index"]}')
        torch.cuda.synchronize()
        started = time.perf_counter()
        decisions = CohortDecisions(model, len(group), args.device, False)
        past, prefixes, physical = online_rollout(wrapper,
            encoded['A']['input_ids'].to(args.device2),
            encoded['A']['attention_mask'].to(args.device2), decisions,
            preserve_arithmetic=True)
        budgets = decisions.budgets.cpu().tolist()
        del past
        write_tensors(path / 'controller_observations.safetensors',
            dict(features=decisions.x, visited=decisions.visited, actions=decisions.actions,
                 old_logprobs=decisions.logprobs, values=decisions.values))
        payloads = prefixes_for_b(wrapper, texts, encoded, prefixes)
        torch.cuda.synchronize()
        a_seconds = time.perf_counter() - started
        write_json(path / 'prompts.json', texts)
        outcomes, b_seconds = decode_test_cohort(wrapper, args, group, payloads, budgets, root, path)
        if [r['source_id'] for r in outcomes] != [i['source_id'] for i in group]:
            raise ValueError('Test identity/order mismatch')
        summary = dict(cohort=index, samples=len(group), correct=sum(r['strict_correct'] for r in outcomes),
            mean_K=sum(budgets)/len(group), selected_budgets=budgets,
            cohort_physical_A_steps=physical, A_instrumented_seconds=a_seconds, B_seconds=b_seconds)
        write_json(path / 'summary.json', summary)
        commit(base, path)
        rows.extend(outcomes)
        event(root, stage='test_cohort_committed', committed_samples=len(rows), **summary)
        del payloads, prefixes, decisions
    if len(rows) != 300 or [r['source_id'] for r in rows] != [i['source_id'] for i in items]:
        raise ValueError('Expected exactly 300 ordered test outcomes')
    summary = dict(status='complete', checkpoint_iteration=iteration, samples=len(rows),
        actual_B_completions=len(rows), strict_correct=sum(r['strict_correct'] for r in rows),
        accuracy=sum(r['strict_correct'] for r in rows)/len(rows),
        legacy_correct=sum(r['legacy_correct'] for r in rows),
        mean_K=sum(r['k'] for r in rows)/len(rows),
        K_counts=dict(sorted(Counter(r['k'] for r in rows).items())),
        total_B_tokens=sum(r['output_tokens'] for r in rows),
        caps=sum(r['finish_reason']=='length' for r in rows),
        loops=sum(r['repetition']['loop_flag'] for r in rows),
        parse_failures=sum(r['strict_parse_failed'] for r in rows),
        no_optimizer_created=True, test_used_for_training_or_checkpoint_selection=False,
        checkpoint_selection='Iteration locked from validation before this test evaluation',
        scope='Historical MedQA300; previously exposed first16 and historical reference available',
        cache_protocol='Cold at evaluation start, warm across ordered cohorts; 18x16 plus final12',
        timing_caveat='Instrumented cohort throughput, not production request latency')
    write_jsonl(root / 'outcomes.jsonl', rows)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--iteration', type=int, required=True)
    ap.add_argument('--checkpoint-sha256', required=True)
    ap.add_argument('--stage', choices=['prepare', 'evaluate'], default='prepare')
    ap.add_argument('--dry-run', action='store_true')
    cli = ap.parse_args()
    training, out = contained(cli.training, ROOT), contained(cli.out, ROOT)
    if out == ROOT or out == training or training in out.parents or out in training.parents:
        raise ValueError('Separate evaluation output required')
    with (training / 'pipeline.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest = json.loads((training / 'manifest.json').read_text())
        verify_hashes(training / 'source_snapshot', manifest['source_sha256'])
        verify_hashes(ROOT, manifest['source_sha256'])
        checkpoint = contained(selected_checkpoint(training, cli.iteration, cli.checkpoint_sha256), ROOT)
        cfg = manifest['config']
        if (cfg['temperature'] != .6 or cfg['generate_bs'] != 16 or cfg['max_new_tokens'] != 4096
                or cfg['num_agents'] != 2 or not cfg['latent_space_realign']):
            raise ValueError('Unexpected inherited protocol')
        data = contained(Path(cfg['data_dir']), ROOT)
        for name in ['manifest.json', 'test.jsonl', 'train.jsonl', 'validation.jsonl']:
            contained(data / name, ROOT)
        historical = contained(ROOT / 'data/medqa.json', ROOT)
        reference = contained(Path(cfg['reference']), ROOT)
        official, metadata = load_official_medqa(data, 'test')
        reference_rows = json.loads(reference.read_text())['predictions']
        items = match_items(json.loads(historical.read_text()), official, reference_rows)
        if len(items) != 300:
            raise ValueError('Expected full historical test300')
        stems = {normalized_question(i['question_stem']) for i in items}
        for split in ['train', 'validation']:
            other, _ = load_official_medqa(data, split)
            if stems & {normalized_question(i['question_stem']) for i in other}:
                raise ValueError('Train/validation-test stem overlap')
        ids = [i['source_id'] for i in items]
        if set(ids) & (set(manifest['train_ids']) | set(manifest['validation_ids'])):
            raise ValueError('Train/validation-test identity overlap')
        sources = current_sources(ROOT, Path(__file__))
        frozen = dict(schema='expanded_controller_test300_v1', training=str(training),
            training_manifest_sha256=file_hash(training / 'manifest.json'),
            training_finished_sha256=file_hash(training / 'training_finished.json'),
            checkpoint=str(checkpoint), checkpoint_sha256=cli.checkpoint_sha256,
            checkpoint_iteration=cli.iteration, config=cfg, source_sha256=sources,
            historical_test300_sha256=file_hash(historical), official_test_sha256=metadata['sha256'],
            reference_sha256=file_hash(reference), test_source_ids=ids,
            expected_B_completions=300, cohort_sizes=[len(c) for c in cohort_items(items)],
            padding_rows=0, no_optimizer_created=True, no_wandb_writes=True,
            controller_decoding='argmax', realignment_profile=manifest['realignment_profile'],
            realignment_sha256=manifest['migration']['archived_realignment_sha256'],
            cache='Cold initially, warm across fixed original-order cohorts',
            test_used_for_training_or_checkpoint_selection=False)
        if cli.dry_run:
            print(json.dumps(dict(status='dry_run', samples=300, checkpoint_iteration=cli.iteration,
                checkpoint_sha256=cli.checkpoint_sha256, cohort_sizes=frozen['cohort_sizes'],
                source_files=len(sources), train_validation_overlap=False,
                CUDA_initialized=torch.cuda.is_initialized()))); return
        if out.exists():
            if json.loads((out / 'manifest.json').read_text()) != frozen:
                raise ValueError('Evaluation output protocol changed')
        else:
            out.mkdir(parents=True)
            write_json(out / 'manifest.json', frozen)
            for name in sources:
                atomic_exclusive_bytes(out / 'source_snapshot' / name, contained(ROOT / name, ROOT).read_bytes())
        with (out / 'pipeline.lock').open('a') as output_lock:
            fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_hashes(ROOT, sources)
            verify_hashes(out / 'source_snapshot', sources)
            if cli.stage == 'prepare':
                print(json.dumps(dict(status='prepared', samples=300, checkpoint_iteration=cli.iteration,
                    checkpoint_sha256=cli.checkpoint_sha256, source_files=len(sources),
                    CUDA_initialized=torch.cuda.is_initialized()))); return
            if (out / 'evaluation_started.json').exists():
                raise ValueError('Evaluation already started; inspect before any explicit recovery')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['iteration'] != cli.iteration:
                raise ValueError('Checkpoint payload iteration differs')
            del saved
            write_json(out / 'evaluation_started.json', dict(checkpoint_sha256=cli.checkpoint_sha256,
                checkpoint_iteration=cli.iteration, expected_B_completions=300))
            os.environ.pop('VLLM_BATCH_INVARIANT', None)
            from models import ModelWrapper
            from utils import set_seed
            set_seed(cfg['seed'])
            args = SimpleNamespace(**cfg)
            event(out, stage='initializing', checkpoint_iteration=cli.iteration)
            wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
            workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
            if any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
                    or w['quantization'] is not None or w['model_revision'] != args.model_revision for w in workers):
                raise ValueError('B precision/revision mismatch')
            if wrapper.HF_model.config._commit_hash != args.model_revision or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16:
                raise ValueError('A precision/revision mismatch')
            realignment = bind_archived_realignment(wrapper, contained(Path(manifest['realignment_profile']), ROOT),
                manifest['migration']['archived_realignment_sha256'])
            model = load_controller_checkpoint(checkpoint, args.device,
                expected_schema=cfg['controller_normalization']).eval()
            model.requires_grad_(False)
            write_tensors(out / 'frozen_controller.safetensors', model.state_dict())
            write_json(out / 'preflight.json', dict(workers=workers, realignment=realignment,
                hf_dtype='torch.bfloat16', no_optimizer_created=True, normalization_refitted=False))
            summary = evaluate(wrapper, args, out, model, items, reference_rows, cli.iteration)
            if file_hash(checkpoint) != cli.checkpoint_sha256 or file_hash(training/'manifest.json') != frozen['training_manifest_sha256']:
                raise ValueError('Training artifact changed during evaluation')
            verify_hashes(ROOT, sources)
            summary.update(checkpoint_sha256=cli.checkpoint_sha256, training_artifacts_unchanged=True)
            write_json(out / 'evaluation_finished.json', summary)
            event(out, stage='evaluation_complete', **summary)
            print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
