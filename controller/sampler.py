"""Train-only question pools and restartable shuffled PPO iteration plans.

Question selection has its own RNG: it never consumes policy/torch RNG. Only
identities and epoch position are sampled here, not actions or B rewards.
"""
import hashlib
import random

from controller.matched import historical_style, require_train
from controller.profile import load_official_medqa, normalized_question, make_subset


def train_pool(data_dir, count=256, seed=42, required_ids=()):
    """Include the existing train-only rehearsal questions, then fill the pool.

    Held-out splits are read ONLY for duplicate-stem exclusion, never for labels
    or selection by outcomes. Returned examples all originate from official train.
    """
    if type(count) is not int or count < 32 or count % 32:
        raise ValueError('Pool size must be a positive multiple of 32, at least 32')
    items, meta = load_official_medqa(data_dir, 'train')
    forbidden = set()
    split_hashes = {'train': meta['sha256']}
    for split in ('validation', 'test'):
        heldout, split_meta = load_official_medqa(data_dir, split)
        split_hashes[split] = split_meta['sha256']
        forbidden.update(normalized_question(i['question_stem']) for i in heldout)
    lookup = {i['source_id']: i for i in items}
    required_ids = list(required_ids)
    if len(set(required_ids)) != len(required_ids) or len(required_ids) > count:
        raise ValueError('Required train IDs must be distinct and fit in the pool')
    selected, stems = [], set()
    for source_id in required_ids:
        item = lookup.get(source_id)
        if item is None:
            raise ValueError('Required ID is not official train')
        stem = normalized_question(item['question_stem'])
        if stem in forbidden or stem in stems:
            raise ValueError('Required question overlaps held-out data or another train stem')
        selected.append(item)
        stems.add(stem)
    candidates = []
    for item in items:
        stem = normalized_question(item['question_stem'])
        if stem not in stems and stem not in forbidden:
            candidates.append(item)
            stems.add(stem)
    needed = count - len(selected)
    if needed > len(candidates):
        raise ValueError('Insufficient distinct eligible official-train questions')
    selected.extend(random.Random(seed).sample(candidates, needed))
    if any(i['source_split'] != 'train' or ':train:' not in i['source_id'] for i in selected):
        raise ValueError('Train source firewall failed')
    selected = [dict(i, question=historical_style(i)) for i in selected]
    return selected, dict(split='train', count=count, seed=seed,
        dataset_sha256=meta['sha256'], split_sha256=split_hashes,
        source_ids=[i['source_id'] for i in selected], required_source_ids=required_ids,
        heldout_question_overlap=False, unique_normalized_stems=count,
        heldout_used_for_training=False)


class ShuffledQuestionSampler:
    """A pure mapping from 1-based LOCAL PPO iteration to 32 unique questions.

    Each epoch is a permutation of the whole pool. No replacement within an
    epoch; a question may reappear across the epoch boundary. Restart from the
    next committed iteration to recover the exact plan without mutable RNG state.
    """
    def __init__(self, source_ids, seed=42):
        self.source_ids = tuple(source_ids)
        if len(self.source_ids) < 32 or len(self.source_ids) % 32:
            raise ValueError('Expected a pool size divisible by 32')
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError('Duplicate question IDs in sampler pool')
        self.seed = seed
        self.iterations_per_epoch = len(self.source_ids) // 32

    def plan(self, iteration):
        if type(iteration) is not int or iteration < 1:
            raise ValueError('Iteration must be a positive 1-based integer')
        epoch, batch = divmod(iteration - 1, self.iterations_per_epoch)
        # Versioned per-epoch seed, independent of Python hash randomization and
        # all RNG consumed by LLM initialization, PPO updates or diagnostics.
        digest = hashlib.sha256(f'train-pool-v1:{self.seed}:{epoch}'.encode()).digest()
        order = list(range(len(self.source_ids)))
        random.Random(int.from_bytes(digest, 'big')).shuffle(order)
        indices = order[batch * 32:(batch + 1) * 32]
        return dict(local_iteration=iteration, epoch=epoch + 1,
            iteration_in_epoch=batch + 1, iterations_per_epoch=self.iterations_per_epoch,
            pool_size=len(self.source_ids), seed=self.seed, indices=indices,
            source_ids=[self.source_ids[i] for i in indices], cohort_sizes=[16, 16])


def fresh_train_items(data_dir, old_ids, seed=42):
    pool, metadata = load_official_medqa(data_dir, 'train')
    lookup = {item['source_id']: item for item in pool}
    if len(old_ids) != 32 or len(set(old_ids)) != 32 or any(i not in lookup for i in old_ids):
        raise ValueError('Expected32 distinct official-train parent IDs')
    forbidden = {normalized_question(lookup[i]['question_stem']) for i in old_ids}
    for split in ('validation', 'test'):
        heldout, _ = load_official_medqa(data_dir, split)
        forbidden.update(normalized_question(i['question_stem']) for i in heldout)
    eligible = [i for i in pool if i['source_id'] not in old_ids
                and normalized_question(i['question_stem']) not in forbidden]
    items = [dict(i, question=historical_style(i)) for i in make_subset(eligible, 32, seed)]
    require_train(items)
    assert not set(old_ids) & {i['source_id'] for i in items}
    return items, dict(dataset_sha256=metadata['sha256'], seed=seed,
        source_ids=[i['source_id'] for i in items], split='train', count=32,
        parent_source_overlap=False, parent_question_overlap=False,
        heldout_question_overlap=False)
