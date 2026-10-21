import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from scripts import profile_b_seed_repeats as profile
from scripts import schedule_b_seed_repeats as schedule


class BSeedRepeatTests(unittest.TestCase):
    def test_only_decoder_seed_changes(self):
        for seed in (43,44):
            self.assertEqual(profile.sampling_config(seed),
                dict(temperature=.6,top_p=.95,top_k=0,seed=seed,max_tokens=4096))
        with self.assertRaises(ValueError):
            profile.sampling_config(42)

    def test_decoder_receives_seed_and_preserves_pairing(self):
        for seed in (43,44):
            params = SimpleNamespace(**profile.sampling_config(seed))
            payloads = [dict(prompt_embeds=torch.ones(3,2,dtype=torch.bfloat16)) for _ in range(16)]
            refs = [dict(source_id=f'x:train:{i}',source_split='train',gold='a',k=10)
                    for i in range(16)]
            class Engine:
                def generate(self,inputs,actual_params,use_tqdm):
                    self.params = actual_params
                    return [SimpleNamespace(outputs=[SimpleNamespace(text=r'\boxed{A}',
                        token_ids=[1,2],finish_reason='stop',stop_reason=None)]) for _ in inputs]
            engine = Engine()
            rows,_ = profile.decode(engine,payloads,refs,seed,params)
            self.assertIs(engine.params,params)
            self.assertTrue(all(r['decoder_seed']==seed and r['strict_correct'] for r in rows))
            for ref,row in zip(refs,rows):
                ref['prompt_sha256'] = row['prompt_sha256']
            profile.validate_new_rows(rows,refs,seed)
            rows[0]['source_split'] = 'test'
            with self.assertRaises(ValueError):
                profile.validate_new_rows(rows,refs,seed)
            params.temperature = 0.
            with self.assertRaises(ValueError):
                profile.decode(engine,payloads,refs,seed,params)

    def test_resume_requires_contiguous_commits(self):
        paths = [Path(str(i)) for i in range(4)]
        with patch.object(profile,'completed',side_effect=[Path('a'),None,None,None]):
            self.assertEqual(profile.first_missing(paths),1)
        with patch.object(profile,'completed',side_effect=[Path('a'),None,Path('c'),None]):
            with self.assertRaises(ValueError):
                profile.first_missing(paths)

    def test_rest_requires_full_hour_on_both_clocks(self):
        self.assertEqual(schedule.rest_remaining(100,1000,3699,4599),1)
        self.assertEqual(schedule.rest_remaining(100,1000,3700,4600),0)
        self.assertEqual(schedule.rest_remaining(100,1000,200,9000),3500)
        self.assertGreater(schedule.rest_remaining(100,1000,5000,900),3600)

    def test_finish_does_not_accept_partial_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'seed43').mkdir()
            p = root/'seed43'/'finished.json'
            self.assertFalse(schedule.verified_finish(root,43))
            p.write_text(json.dumps(dict(status='complete',verified=True,
                decoder_seed=43,main_B_completions=256)))
            with self.assertRaises(ValueError):
                schedule.verified_finish(root,43)
            p.write_text(json.dumps(dict(status='complete',verified=True,
                decoder_seed=43,main_B_completions=3072)))
            self.assertTrue(schedule.verified_finish(root,43))


if __name__ == '__main__':
    unittest.main()
