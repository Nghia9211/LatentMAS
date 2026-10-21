from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from scripts import pilot_greedy_b as pilot


class GreedyBPilotTests(unittest.TestCase):
    def payload_fixture(self):
        t={'A_mask':torch.ones(16,2,dtype=torch.long)}
        rec={}
        for i in range(16):
            t[f'prefix_{i}']=torch.full((82,3),i,dtype=torch.bfloat16)
            rec[f'B_prompt_{i}']=torch.full((5,3),-1,dtype=torch.bfloat16)
            rec[f'insertion_{i}']=torch.tensor(2)
        return t,rec

    def test_exact_reconstruction_and_immutable_archived_tensors(self):
        t,rec=self.payload_fixture()
        payloads=pilot.reconstruct_payloads(t,rec,10)
        self.assertEqual(len(payloads),16)
        self.assertEqual(payloads[0]['prompt_embeds'].shape,(17,3))
        self.assertTrue(payloads[0]['prompt_embeds'][2:14].eq(0).all())
        self.assertTrue(rec['B_prompt_0'].eq(-1).all())
        with self.assertRaises(ValueError):
            pilot.reconstruct_payloads(t,rec,0)

    def test_strict_last_box(self):
        self.assertTrue(pilot.score_text(r'\boxed{a}', 'a')['medqa_boxed_v1']['correct'])
        self.assertFalse(pilot.score_text('A is correct', 'a')['medqa_boxed_v1']['correct'])
        self.assertFalse(pilot.score_text(r'\boxed{A} \boxed{B}', 'a')['medqa_boxed_v1']['correct'])

    def test_pairing_rejects_prompt_or_split_changes(self):
        t,rec=self.payload_fixture()
        payloads=pilot.reconstruct_payloads(t,rec,10)
        ids=[f'x:train:{i}' for i in range(16)]
        rows=[dict(source_id=s,source_split='train',k=10,temperature=.6,batch=16,
                   decoder_seed=42,prompt_sha256=pilot.tensor_hash(p['prompt_embeds']),
                   raw_prediction=r'\boxed{A}',gold='a',strict_correct=True) for s,p in zip(ids,payloads)]
        pilot.check_pairing(payloads,rows,ids,'train',10)
        with self.assertRaises(ValueError):
            pilot.check_pairing(payloads,rows,ids,'validation',10)
        rows[0]['prompt_sha256']='corrupted'
        with self.assertRaises(ValueError):
            pilot.check_pairing(payloads,rows,ids,'train',10)

    def test_greedy_decoder_passes_params_and_reports_real_outputs(self):
        t,rec=self.payload_fixture()
        payloads=pilot.reconstruct_payloads(t,rec,20)
        rows=[dict(source_id=f'x:train:{i}',source_split='train',gold='a') for i in range(16)]
        params=SimpleNamespace(temperature=0.)
        class Engine:
            seen=None
            def generate(self,payloads,actual_params,use_tqdm):
                self.seen=actual_params
                return [SimpleNamespace(outputs=[SimpleNamespace(text=r'\boxed{A}',token_ids=[1,2],
                        finish_reason='stop',stop_reason=None)]) for _ in payloads]
        engine=Engine()
        out,_=pilot.decode_greedy(engine,payloads,rows,20,params)
        self.assertIs(engine.seen,params)
        self.assertEqual(len(out),16)
        self.assertTrue(all(r['strict_correct'] and r['temperature']==0 and r['output_tokens']==2 for r in out))
        self.assertEqual(pilot.tally(out)['correct'],16)
        with self.assertRaises(ValueError):
            pilot.decode_greedy(engine,payloads,rows,20,SimpleNamespace(temperature=.6))


if __name__=='__main__':
    unittest.main()
