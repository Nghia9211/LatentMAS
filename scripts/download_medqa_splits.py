"""Download pinned English four-option MedQA; preserve the legacy eval file.

Raw upstream Parquet files remain intact. JSONL is a schema-normalized export,
not a reshuffle or deduplication. No model execution or production-loader edit.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import re
import unicodedata

REPO = 'bigbio/med_qa'
CONFIG = 'med_qa_en_4options_source'
REVISION = '484a6c066fe8e75c83edea0c88b5169316714fcd'
ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'train': 10178, 'validation': 1272, 'test': 1273}


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def question_hash(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def legacy_stem(row):
    """Remove only exact appended option blocks, not arbitrary clinical text."""
    text = row['question'].rstrip()
    block = '\n'.join(row['options']).rstrip()
    removed = 0
    while block and text.endswith(block):
        text = text[:-len(block)].rstrip()
        removed += 1
    return text, removed


def build_overlap_audit(split_rows, legacy):
    before = file_hash(legacy)
    duplicates = {}
    for split, rows in split_rows.items():
        counts = Counter(r['question_sha256'] for r in rows)
        duplicates[split] = dict(duplicate_question_groups=sum(n > 1 for n in counts.values()),
                                extra_duplicate_question_rows=sum(n - 1 for n in counts.values()))
    cross_split = []
    for a, b in itertools.combinations(EXPECTED, 2):
        left = defaultdict(list)
        right = defaultdict(list)
        for r in split_rows[a]: left[r['question_sha256']].append(r['source_id'])
        for r in split_rows[b]: right[r['question_sha256']].append(r['source_id'])
        shared = sorted(left.keys() & right.keys())
        cross_split.append(dict(left=a, right=b, shared_normalized_questions=len(shared),
            matches=[dict(question_sha256=h, left_ids=left[h], right_ids=right[h]) for h in shared]))
    by_question = defaultdict(list)
    for rows in split_rows.values():
        for row in rows: by_question[row['question_sha256']].append(row)
    old = json.loads(legacy.read_text())
    matches = []
    counts = Counter()
    unmatched = []
    removed_counts = Counter()
    for index, row in enumerate(old):
        stem, removed = legacy_stem(row)
        removed_counts[removed] += 1
        found = by_question.get(question_hash(stem), [])
        if not found:
            unmatched.append(index + 1)
            continue
        old_options = [re.sub(r'^\s*[A-Da-d][.):]\s*', '', o).strip() for o in row['options']]
        records = []
        for r in found:
            counts[r['source_split']] += 1
            records.append(dict(source_id=r['source_id'], source_split=r['source_split'],
                option_multiset_matches=sorted(map(normalize, old_options)) == sorted(map(normalize, r['options'].values())),
                gold_text_matches=normalize(row['answer']) == normalize(r['answer'])))
        matches.append(dict(legacy_position=index + 1, legacy_idx=row.get('idx'),
                            removed_appended_option_blocks=removed, matches=records))
    assert file_hash(legacy) == before
    return dict(normalization='Remove exact repeated appended legacy option blocks; then NFKC + casefold + whitespace collapse on stems; compare options/gold separately',
        within_split_duplicates=duplicates, cross_split_overlap=cross_split,
        legacy_file=str(legacy), legacy_sha256=before, legacy_rows=len(old),
        legacy_option_blocks_removed_histogram=dict(removed_counts),
        legacy_matched_rows=len(matches), legacy_unmatched_positions=unmatched,
        legacy_source_match_counts=dict(counts), legacy_matches=matches,
        note='Official splits and row order are preserved, not deduplicated. Stem matches establish exposure risk, not a clinical semantic equivalence proof. Counts are source matches and can exceed unique legacy rows if upstream duplicate stems exist.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/medqa_usmle_4options')
    parser.add_argument('--legacy', type=Path, default=ROOT/'data/medqa.json')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Preserve existing artifacts; select a fresh output directory: {out}')
    legacy = args.legacy.resolve()
    legacy_before = file_hash(legacy)
    out.mkdir(parents=True, exist_ok=True)
    raw = out/'raw'
    filenames = ['README.md'] + [f'{CONFIG}/{s}-00000-of-00001.parquet' for s in EXPECTED]
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    def download(filename):
        return hf_hub_download(REPO, filename, repo_type='dataset', revision=REVISION, local_dir=raw)

    print('Downloading', REPO, CONFIG, 'revision', REVISION, flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, filenames))

    split_rows = {}
    split_meta = {}
    duplicates = {}
    for split, count in EXPECTED.items():
        src = raw/f'{CONFIG}/{split}-00000-of-00001.parquet'
        original = pq.read_table(src).to_pylist()
        assert len(original) == count, (split, len(original), count)
        result = []
        for index, row in enumerate(original):
            options = row['options']
            if isinstance(options, list):
                keys = [r['key'] for r in options]
                assert len(keys) == len(set(keys)), (split, index, 'duplicate option keys')
                options = {r['key']: r['value'] for r in options}
            assert set(options) == set('ABCD'), (split, index, list(options))
            assert all(isinstance(v, str) and v.strip() for v in options.values())
            answer_idx = row['answer_idx']
            assert answer_idx in options and row['answer'] == options[answer_idx], (split, index, 'gold mismatch')
            assert isinstance(row['question'], str) and row['question'].strip(), (split, index, 'empty question')
            result.append(dict(source_id=f'medqa_usmle_4options:{split}:{index}', source_split=split,
                source_row=index, question=row['question'], options={k: options[k] for k in 'ABCD'},
                answer_idx=answer_idx, answer=row['answer'], meta_info=row.get('meta_info'),
                question_sha256=question_hash(row['question'])))
        path = out/f'{split}.jsonl'
        with path.open('x', encoding='utf-8') as f:
            for row in result:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        split_rows[split] = result
        counts = Counter(r['question_sha256'] for r in result)
        duplicates[split] = dict(duplicate_question_groups=sum(n > 1 for n in counts.values()),
                                extra_duplicate_question_rows=sum(n - 1 for n in counts.values()))
        split_meta[split] = dict(rows=len(result), file=path.name, sha256=file_hash(path),
            raw_file=str(src.relative_to(out)), raw_sha256=file_hash(src))
        print(split, len(result), 'rows exported', flush=True)

    assert file_hash(legacy) == legacy_before
    audit = build_overlap_audit(split_rows, legacy)
    write_json(out/'overlap_audit.json', audit)
    manifest = dict(dataset=REPO, config=CONFIG, revision=REVISION,
        source_url=f'https://huggingface.co/datasets/{REPO}/tree/{REVISION}/{CONFIG}',
        original_dataset_url='https://github.com/jind11/MedQA',
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        raw_files=[dict(file=str((raw/f).relative_to(out)), sha256=file_hash(raw/f)) for f in filenames],
        splits=split_meta, total_rows=sum(EXPECTED.values()),
        row_id_note='source_id is derived from the pinned dataset/config/split and zero-based upstream row order, not a native upstream ID.',
        schema='question is stem only; options is A/B/C/D mapping; answer_idx is the gold letter; answer is the gold option text.',
        license_note='The BigBio dataset card marks dataset license unknown; repository code license is not inferred to cover all dataset content.',
        legacy_file_preserved=dict(path=str(legacy), sha256=legacy_before),
        loader_note='No production loader changed; load_medqa still reads packaged medqa.json, not these new JSONL splits.',
        audit_file='overlap_audit.json', audit_sha256=file_hash(out/'overlap_audit.json'),
        package_versions={m: importlib.metadata.version(m) for m in ['huggingface_hub', 'pyarrow']})
    write_json(out/'manifest.json', manifest)
    print('Legacy overlap:', audit['legacy_source_match_counts'], 'matched rows', audit['legacy_matched_rows'],
          'unmatched', len(audit['legacy_unmatched_positions']), flush=True)
    print('Done:', out, flush=True)


if __name__ == '__main__':
    main()
