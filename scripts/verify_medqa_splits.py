"""Verify downloaded bytes/export schema and re-audit legacy stem exposure."""
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from download_medqa_splits import EXPECTED, ROOT, REVISION, build_overlap_audit, file_hash, normalize, question_hash, write_json


def save_or_verify(path, data):
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(data)), f'Existing evidence differs: {path}'
    else:
        write_json(path, data)


def main():
    out = ROOT/'data/medqa_usmle_4options'
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['revision'] == REVISION
    for raw in manifest['raw_files']:
        assert file_hash(out/raw['file']) == raw['sha256']
    split_rows = {}
    ids = set()
    for split, count in EXPECTED.items():
        meta = manifest['splits'][split]
        path = out/meta['file']
        assert file_hash(path) == meta['sha256']
        rows = [json.loads(line) for line in path.open()]
        original = pq.read_table(out/meta['raw_file']).to_pylist()
        assert len(rows) == len(original) == meta['rows'] == count
        for i, (r, o) in enumerate(zip(rows, original)):
            opts = {v['key']: v['value'] for v in o['options']}
            assert r['question'] == o['question'] and r['options'] == opts
            assert r['answer'] == o['answer'] == opts[r['answer_idx']]
            assert r['answer_idx'] == o['answer_idx'] and set(opts) == set('ABCD')
            assert r['source_split'] == split and r['source_row'] == i
            assert r['source_id'] not in ids
            ids.add(r['source_id'])
            assert r['question_sha256'] == question_hash(r['question'])
        split_rows[split] = rows
    legacy = ROOT/'data/medqa.json'
    assert file_hash(legacy) == manifest['legacy_file_preserved']['sha256']
    audit = build_overlap_audit(split_rows, legacy)
    audit['supersedes'] = 'overlap_audit.json: initial check used option-appended legacy question, not the stem; this verified audit replaces its exposure conclusion.'
    save_or_verify(out/'overlap_audit_stem.json', audit)
    old = json.loads(legacy.read_text())
    lookup = {r['source_id']: r for rows in split_rows.values() for r in rows}
    # Independent extraction from the actual inference query, rather than the
    # repeated-option metadata field used by build_overlap_audit.
    for match in audit['legacy_matches']:
        r = old[match['legacy_position'] - 1]
        marker = '\n' + r['options'][0]
        assert marker in r['query']
        query_stem = r['query'].split(marker, 1)[0]
        stripped = [re.sub(r'^\s*[A-Da-d][.):]\s*', '', v).strip() for v in r['options']]
        for source in match['matches']:
            target = lookup[source['source_id']]
            assert normalize(query_stem) == normalize(target['question'])
            assert source['option_multiset_matches'] == (sorted(map(normalize, stripped)) == sorted(map(normalize, target['options'].values())))
            assert source['gold_text_matches'] == (normalize(r['answer']) == normalize(target['answer']))
    summary = dict(passed=True, rows={k:len(v) for k,v in split_rows.items()}, total_rows=len(ids),
        raw_file_hashes_verified=True, export_matches_raw=True, legacy_file_unchanged=True,
        legacy_matched_rows=audit['legacy_matched_rows'], legacy_match_counts=audit['legacy_source_match_counts'],
        independent_query_stem_check=True,
        option_mismatches=sum(not m['option_multiset_matches'] for r in audit['legacy_matches'] for m in r['matches']),
        gold_mismatches=sum(not m['gold_text_matches'] for r in audit['legacy_matches'] for m in r['matches']),
        within_split_duplicates=audit['within_split_duplicates'],
        cross_split_shared_questions=[dict(left=r['left'],right=r['right'],count=r['shared_normalized_questions']) for r in audit['cross_split_overlap']],
        audit_file='overlap_audit_stem.json', audit_sha256=file_hash(out/'overlap_audit_stem.json'))
    save_or_verify(out/'verification.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
