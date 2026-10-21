import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from controller.profile import GRID, PHI_SCHEMA, phi4, tie_targets, write_json, write_jsonl, write_tensors
from controller.training import train_update
from controller.warmstart import (CANDIDATES, WarmStartController, build_dataset,
                                 check_gradients, loss_diagnostics, supervision, warmstart_loss)

ROOT = Path(__file__).resolve().parents[1]



def fixture(root, invert=False):
    ids = [f'fixture:train:{i}' for i in range(32)]
    write_json(root/'split_manifest.json', dict(source_ids=ids))
    write_json(root/'feature_schema.json', dict(phi_schema=PHI_SCHEMA,policy_metadata=['k/80'],availability_masks=False))
    generator = torch.Generator().manual_seed(42)
    for i, source_id in enumerate(ids):
        base = root/'trajectories'/source_id
        dest = base/'attempt_000'
        h = torch.randn(81, 2, 4, generator=generator).bfloat16()
        write_tensors(dest/'states.safetensors', dict(hidden_last_position_all_layers=h))
        bits = [int((j+i)%3 == 0) ^ int(invert) for j in range(8)]
        write_jsonl(dest/'b_outcomes.jsonl', [dict(k=k,source_id=source_id,source_split='train',strict_correct=bool(q),
            gold='NEVER_FEED_THIS_LABEL',raw_prediction='NEVER_FEED_B_TEXT') for k,q in zip(GRID,bits)])
        write_jsonl(dest/'checkpoints.jsonl', [dict(k=k,phi=phi4(h[:,-1],k).tolist()) for k in GRID])
        write_json(dest/'train_targets.json', tie_targets(bits))
        write_json(base/'complete.json', dict(attempt='attempt_000'))


class SupervisionTests(unittest.TestCase):
    def test_all_ties_stop(self):
        for correct in (0,1):
            a,q = supervision([correct]*8)
            self.assertEqual(a.tolist(), [0]*7)
            torch.testing.assert_close(q, torch.full((7,2),float(correct)))

    def test_only_cap_correct(self):
        a,q = supervision([0]*7+[1])
        self.assertEqual(a.tolist(), [1]*7)
        torch.testing.assert_close(q, torch.tensor([[0.,1.]]*7))

    def test_mixed_matches_archived_ties(self):
        bits = [0,1,0,1,0,1,1,0]
        a,q = supervision(bits)
        self.assertEqual(a.tolist(), [1,0,1,0,1,0,0])
        self.assertEqual(q[-1].tolist(), [1.,0.])

    def test_invalid_full_grid(self):
        for bits in ([0]*7, [0]*7+[2], [0]*7+[float('nan')]):
            with self.assertRaises(ValueError):
                supervision(bits)


class DatasetTests(unittest.TestCase):
    def test_labels_never_change_policy_features(self):
        with tempfile.TemporaryDirectory() as folder:
            left,right = Path(folder)/'left',Path(folder)/'right'
            fixture(left);fixture(right,invert=True)
            x,a,q,meta = build_dataset(left)
            y,b,r,other = build_dataset(right)
            self.assertEqual(tuple(x.shape),(32,7,17))
            self.assertEqual(x.dtype,torch.float32)
            self.assertFalse(x.requires_grad)
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertFalse(torch.equal(q,r))
            self.assertEqual(meta['features_sha256'],other['features_sha256'])
            self.assertEqual(meta['metadata'],['k/80'])
            self.assertEqual(x[0,:,-1].tolist(),[k/80 for k in GRID[:-1]])
            self.assertEqual(a.numel(),224)

    def test_future_hidden_does_not_change_current_observation(self):
        from controller.profile import raw_observation
        h = torch.randn(81,8).bfloat16()
        changed = h.clone();changed[21:] = 100*changed[21:]
        torch.testing.assert_close(raw_observation(h,20),raw_observation(changed,20),rtol=0,atol=0)


class ArchitectureTests(unittest.TestCase):
    def test_shapes_causal_and_question_reset(self):
        torch.manual_seed(42)
        x = torch.randn(3,7,17)
        for candidate in CANDIDATES:
            model = WarmStartController(candidate,17,width=8).eval()
            with torch.no_grad():
                actor,q = model(x)
                changed = x.clone();changed[:,3:] += 100
                future_actor,future_q = model(changed)
                single_actor,single_q = model(x[:1])
                repeat_actor,repeat_q = model(x[:1])
            self.assertEqual(tuple(actor.shape),(3,7,2))
            torch.testing.assert_close(actor[:,:3],future_actor[:,:3],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(q[:,:3],future_q[:,:3],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(actor[:1],single_actor,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(q[:1],single_q,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(single_actor,repeat_actor,atol=0,rtol=0)
            self.assertEqual(next(model.parameters()).dtype,torch.float32)

    def test_loss_and_gradients(self):
        actor = torch.zeros(2,7,2,requires_grad=True)
        q = torch.zeros_like(actor,requires_grad=True)
        target = torch.ones_like(q)
        total,pieces = warmstart_loss(actor,q,torch.zeros(2,7,dtype=torch.long),target)
        self.assertAlmostEqual(float(total.detach()),math.log(2)+.5,places=6)
        total.backward()
        self.assertTrue(torch.isfinite(actor.grad).all())
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_train_all_architectures_loss_decreases(self):
        x = torch.randn(8,7,17)
        a = torch.zeros(8,7,dtype=torch.long)
        q = torch.ones(8,7,2)
        config = dict(q_weight=1.,grad_clip=1.)
        for candidate in CANDIDATES:
            torch.manual_seed(42)
            model = WarmStartController(candidate,17,width=8)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=0)
            initial = loss_diagnostics(model,x,a,q,1.)['loss']
            for _ in range(25):
                train_update(model,optimizer,x,a,q,config)
            self.assertTrue(check_gradients(model))
            self.assertLess(loss_diagnostics(model,x,a,q,1.)['loss'],initial)
            self.assertIsNone(x.grad)

    def test_nonfinite_and_shape_fail(self):
        model = WarmStartController('raw_linear',17)
        with self.assertRaises(FloatingPointError):
            loss_diagnostics(model,torch.full((1,7,17),float('nan')),torch.zeros(1,7,dtype=torch.long),torch.ones(1,7,2),1.)
        with self.assertRaises(ValueError):
            warmstart_loss(torch.zeros(1,7,2),torch.zeros(1,7,2),torch.zeros(1,8,dtype=torch.long),torch.ones(1,7,2))




if __name__ == '__main__':
    unittest.main()
