import json
from pathlib import Path
import tempfile
import unittest
from scripts.evaluate_matched_ppo_checkpoint import latest_checkpoint
from scripts.run_matched_controller_ppo import commit
from controller.profile import write_json, atomic_exclusive_bytes


class CheckpointSelectionTests(unittest.TestCase):
    def test_latest_committed_not_unfinished_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for n in (1, 2):
                base = root / 'train' / f'iteration{n:03d}'
                attempt = base / 'attempt_000'
                attempt.mkdir(parents=True)
                write_json(attempt / 'result.json', dict(iteration=n, train_only=True, episodes=32))
                atomic_exclusive_bytes(attempt / 'checkpoint.pt', b'checkpoint-placeholder')
                commit(base, attempt)
            unfinished = root / 'train' / 'iteration003' / 'attempt_000'
            unfinished.mkdir(parents=True)
            atomic_exclusive_bytes(unfinished / 'checkpoint.pt', b'uncommitted')
            n, checkpoint = latest_checkpoint(root)
            self.assertEqual(n, 2)
            self.assertEqual(checkpoint.parent.parent.name, 'iteration002')
            (checkpoint.parent / 'result.json').write_text(json.dumps(dict(iteration=2, train_only=False, episodes=32)))
            with self.assertRaises(ValueError):
                latest_checkpoint(root)

    def test_no_commits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                latest_checkpoint(Path(directory))


if __name__ == '__main__':
    unittest.main()
