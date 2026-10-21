#!/usr/bin/env python3
"""CPU-only archive verifier for matched train/validation full-grid profiles."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from controller.profile import file_hash, tensor_hash, write_json, load_official_medqa, normalized_question
from scripts.run_matched_controller_ppo import completed
from scripts.verify_controller_profile import reference_phi


def verify(root, allow_incomplete=False, data_root=None):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        if file_hash(root / 'source_snapshot' / name) != digest:
            raise ValueError('Source snapshot checksum mismatch')
    preflights = sorted(root.glob('preflight_*.json'))
    if not preflights:
        raise ValueError('No successful GPU preflight')
    for file in preflights:
        pre = json.loads(file.read_text())
        if pre['hf_dtype'] != 'torch.bfloat16' or file_hash(root / 'shared/realignment.safetensors') != pre['realignment_sha256']:
            raise ValueError('Shared realignment/preflight mismatch')
        for worker in pre['workers']:
            if (worker['head_dtype'] != 'torch.bfloat16' or worker['embedding_dtype'] != 'torch.bfloat16'
                    or worker['model_revision'] != manifest['config']['model_revision'] or worker['quantization'] is not None):
                raise ValueError('B preflight mismatch')
    summaries, missing = {}, []
    max_phi_error = 0.
    split_stems = {}
    for split, selection in manifest['selections'].items():
        ids = selection['source_ids']
        if split not in ('train', 'validation') or any(f':{split}:' not in i for i in ids):
            raise ValueError('Invalid profile split')
        official, metadata = load_official_medqa(data_root if data_root is not None else manifest['config']['data_dir'], split)
        if metadata['sha256'] != selection['dataset_sha256']:
            raise ValueError('Official split checksum changed')
        lookup = {i['source_id']: i for i in official}
        if len(set(ids)) != len(ids) or any(i not in lookup for i in ids):
            raise ValueError('Invalid/duplicate official identity')
        split_stems[split] = {normalized_question(lookup[i]['question_stem']) for i in ids}
        parity_file = root / split / 'prefix_parity.json'
        if not parity_file.exists():
            missing.append(str(parity_file.relative_to(root)))
            continue
        parity = json.loads(parity_file.read_text())
        if not parity['passed'] or len(parity['rows']) != 144 or any(not r['exact'] for r in parity['rows']):
            raise ValueError('Prefix parity failed/incomplete')
        tallies = {str(k): dict(samples=0, correct=0, tokens=0, caps=0, parse_failures=0, loops=0)
                   for k in range(10, 81, 10)}
        for g in range(len(ids) // 16):
            teacher = completed(root / split / 'teachers' / f'cohort{g:03d}')
            if teacher is None:
                missing.append(f'{split}/teacher/{g}')
                continue
            t = load_file(str(teacher / 'teacher.safetensors'))
            grid = load_file(str(teacher / 'feature_grid.safetensors'))
            reconstruct = load_file(str(teacher / 'reconstruction.safetensors'))
            cohort_meta = json.loads((teacher / 'cohort.json').read_text())
            texts = json.loads((teacher / 'prompts.json').read_text())
            cohort_ids = ids[g * 16:(g + 1) * 16]
            if cohort_meta['source_ids'] != cohort_ids or cohort_meta['split'] != split:
                raise ValueError('Teacher source identity mismatch')
            if (t['hidden'].shape != (16, 81, 37, 2560) or t['latent'].shape != (16, 80, 2560)
                    or t['hidden'].dtype != torch.bfloat16 or t['latent'].dtype != torch.bfloat16
                    or not torch.isfinite(t['hidden']).all() or not torch.isfinite(t['latent']).all()):
                raise ValueError('Invalid rich all-layer BF16 trajectory')
            if grid['features'].shape != (16, 8, 7685) or not torch.isfinite(grid['features']).all():
                raise ValueError('Invalid causal feature shape/finiteness')
            for i in range(16):
                stem = lookup[cohort_ids[i]]['question_stem']
                if stem not in texts['A'][i] or stem not in texts['B'][i]:
                    raise ValueError('Archived prompt identity mismatch')
                h = t['hidden'][i, :, -1].float()
                prompt_length = int(t['A_mask'][i].sum())
                if not torch.equal(t[f'prefix_{i}'][prompt_length:], t['latent'][i]):
                    raise ValueError('Actual latent-input/prefix mismatch')
                for j, k in enumerate(range(10, 81, 10)):
                    x = grid['features'][i, j]
                    ref = reference_phi(h, k)
                    torch.testing.assert_close(grid['phi4'][i, j], ref, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(x[-5:-1], ref, atol=1e-6, rtol=1e-5)
                    if (not torch.equal(x[:2560], h[k]) or not torch.equal(x[2560:5120], h[0])
                            or not torch.equal(x[5120:7680], h[k] - h[k - 10]) or x[-1] != k / 80):
                        raise ValueError('Raw observation schema/causality mismatch')
                    max_phi_error = max(max_phi_error, float((x[-5:-1] - ref).abs().max()))
            for k in range(10, 81, 10):
                path = completed(root / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
                if path is None:
                    missing.append(f'{split}/K{k}/cohort{g:03d}')
                    continue
                rows = [json.loads(s) for s in (path / 'outcomes.jsonl').read_text().splitlines()]
                if len(rows) != 16 or [r['source_id'] for r in rows] != cohort_ids:
                    raise ValueError('Incomplete/reordered outcome cohort')
                for i, row in enumerate(rows):
                    if row['source_split'] != split or row['k'] != k or row['temperature'] != .6 or row['batch'] != 16:
                        raise ValueError('Outcome protocol/split mismatch')
                    if row['gold'] != lookup[cohort_ids[i]]['gold'] or row['decoder_seed'] != 42:
                        raise ValueError('Gold/seed mismatch')
                    tokens = row['generated_token_ids']
                    if row['output_tokens'] != len(tokens) or not 0 <= len(tokens) <= 4096:
                        raise ValueError('B token count mismatch')
                    boxes = re.findall(r'\\boxed\{([^}]*)\}', row['raw_prediction'])
                    pred = boxes[-1].strip().lower() if boxes else None
                    pred = pred if pred in ('a', 'b', 'c', 'd') else None
                    if (row['strict_prediction'] != pred or row['strict_parse_failed'] != (pred is None)
                            or row['strict_correct'] != (pred is not None and pred == row['gold'])):
                        raise ValueError('Independent strict boxed score mismatch')
                    prompt = reconstruct[f'B_prompt_{i}']
                    insertion = int(reconstruct[f'insertion_{i}'])
                    length = int(t['A_mask'][i].sum())
                    payload = torch.cat([prompt[:insertion], t[f'prefix_{i}'][:length + k], prompt[insertion:]])
                    if tensor_hash(payload) != row['prompt_sha256']:
                        raise ValueError('Reconstructed B prompt prefix hash mismatch')
                    tally = tallies[str(k)]
                    tally['samples'] += 1
                    tally['correct'] += int(row['strict_correct'])
                    tally['tokens'] += len(tokens)
                    tally['caps'] += int(row['finish_reason'] == 'length')
                    tally['parse_failures'] += int(pred is None)
                    tally['loops'] += int(row['repetition']['loop_flag'])
        dataset = completed(root / split / 'dataset')
        if dataset is None:
            missing.append(f'{split}/dataset')
        else:
            meta = json.loads((dataset / 'manifest.json').read_text())
            if (meta['source_ids'] != ids or meta['split'] != split
                    or meta['allowed_for_gradient_updates'] != (split == 'train')):
                raise ValueError('Exported dataset split/provenance mismatch')
            d = load_file(str(dataset / 'dataset.safetensors'))
            if d['features'].shape != (len(ids), 7, 7685) or d['correctness'].shape != (len(ids), 8):
                raise ValueError('Exported dataset shape mismatch')
            if split == 'validation' and set(d) != {'features', 'correctness'}:
                raise ValueError('Validation export must not contain training action/auxiliary labels')
            for g in range(len(ids) // 16):
                teacher = completed(root / split / 'teachers' / f'cohort{g:03d}')
                expected_x = load_file(str(teacher / 'feature_grid.safetensors'))['features'][:, :7]
                if not torch.equal(d['features'][g * 16:(g + 1) * 16], expected_x):
                    raise ValueError('Exported feature grid mismatch')
                for j, k in enumerate(range(10, 81, 10)):
                    path = completed(root / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
                    if path is None:
                        raise ValueError('Dataset exported before full grid')
                    rows = [json.loads(s) for s in (path / 'outcomes.jsonl').read_text().splitlines()]
                    expected_q = torch.tensor([float(r['strict_correct']) for r in rows])
                    if not torch.equal(d['correctness'][g * 16:(g + 1) * 16, j], expected_q):
                        raise ValueError('Exported correctness mismatch')
            if split == 'train':
                q = d['correctness']
                future = q.flip(1).cummax(1).values.flip(1)
                targets = torch.stack([q[:, :-1], future[:, 1:]], dim=-1)
                actions = (targets[:, :, 0] < targets[:, :, 1]).long()
                if not torch.equal(d['q_targets'], targets) or not torch.equal(d['actions'], actions):
                    raise ValueError('Independent supervision/tie-break mismatch')
        for tally in tallies.values():
            if tally['samples']:
                tally['accuracy'] = tally['correct'] / tally['samples']
                tally['mean_B_tokens'] = tally['tokens'] / tally['samples']
        summaries[split] = tallies
    if split_stems.get('train', set()) & split_stems.get('validation', set()):
        raise ValueError('Train/validation stem overlap')
    if missing and not allow_incomplete:
        raise ValueError(f'Incomplete profile archive: {len(missing)} components missing')
    return dict(passed=not missing, missing=missing, summaries=summaries,
        max_phi_error=max_phi_error, CPU_only_verification=True, training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--data-root', type=Path, help='Relocated dataset root; original manifest stays unchanged')
    cli = parser.parse_args()
    report = verify(cli.run, cli.allow_incomplete, data_root=cli.data_root)
    if cli.out:
        write_json(cli.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
