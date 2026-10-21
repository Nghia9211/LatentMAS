import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from controller.metrics import rollout_metrics, outcome_metrics, saved_diagnostics, budget_patterns
from scripts import validate_controller_wandb as validation
from scripts.run_ppo_with_wandb import PPOLoggingSession
from controller.ppo import PPOController, sample_episodes

ROOT = Path(__file__).resolve().parents[1]


def episode():
    visited = torch.tensor([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0],
                            [1,1,1,1,1,1,1], [1,1,1,0,0,0,0]], dtype=torch.bool)
    actions = visited.long()
    actions[0,0] = actions[1,1] = actions[3,2] = 0
    probability = torch.where(actions == 0, .2, .8)
    values = torch.full((4,7), .5)
    returns = torch.tensor([1.,0.,1.,0.])[:,None].expand(4,7)
    return dict(visited=visited, actions=actions, old_logprobs=probability.log(),
                old_values=values, returns=returns, advantages=(returns-values)*visited)


def outcomes(start=0, count=16):
    return [dict(source_split='train', source_id=f'train:{i}', output_tokens=100+i,
                 finish_reason='stop', strict_parse_failed=False,
                 repetition={'loop_flag': False}) for i in range(start, start+count)]


class MetricTests(unittest.TestCase):
    def test_probabilities_budgets_and_observed_support(self):
        data = episode()
        m = rollout_metrics(data)
        for k in (10,20,30,80):
            self.assertEqual(m[f'budget/K{k}_count'], 1)
        self.assertEqual(m['handoff/K10_reached'], 4)
        self.assertEqual(m['handoff/K20_reached'], 3)
        self.assertEqual(m['handoff/K30_reached'], 2)
        self.assertAlmostEqual(m['handoff/K10_mean_probability'], .2, places=6)
        self.assertAlmostEqual(m['handoff/K20_mean_probability'], .2, places=6)
        self.assertEqual(m['handoff/K80_forced_count'], 1)
        self.assertNotIn('handoff/K80_mean_probability', m)

    def test_padded_slots_ignored_and_no_fake_unreached_probability(self):
        data = episode()
        data['old_values'][~data['visited']] = float('nan')
        m = rollout_metrics(data)
        self.assertEqual(m['rollout/value/mean'], .5)
        data['visited'] = torch.zeros(4,7,dtype=torch.bool)
        data['visited'][:,0] = True
        data['actions'] = torch.zeros(4,7,dtype=torch.long)
        data['old_values'] = torch.ones(4,7)
        m = rollout_metrics(data)
        self.assertEqual(m['handoff/K20_reached'], 0)
        self.assertNotIn('handoff/K20_mean_probability', m)

    def test_bad_history_rejected_and_constant_returns_not_fake_variance(self):
        data = episode()
        data['visited'][0,1] = True
        with self.assertRaises(ValueError):
            rollout_metrics(data)
        data = episode()
        data['returns'] = torch.ones(4,7)
        self.assertNotIn('rollout/value_explained_variance', rollout_metrics(data))

    def test_output_health_and_no_raw_content(self):
        rows = outcomes(count=4)
        rows[0]['raw_prediction'] = 'private response'
        rows[1]['finish_reason'] = 'length'
        rows[2]['strict_parse_failed'] = True
        rows[3]['repetition']['loop_flag'] = True
        m = outcome_metrics(rows)
        self.assertEqual(m['output/cap_rate'], .25)
        self.assertEqual(m['output/parse_failure_rate'], .25)
        self.assertEqual(m['output/loop_rate'], .25)
        self.assertTrue(all(isinstance(v,(int,float)) for v in m.values()))
        rows[0]['source_split'] = 'validation'
        with self.assertRaises(ValueError):
            outcome_metrics(rows)

    def test_live_coverage_counts_parent_and_fresh_questions(self):
        config = dict(optimization_epochs=4,minibatch_questions=8,clip=.2,
                      value_weight=.5,entropy_weight=.01,tie_weight=.1,grad_clip=1)
        torch.manual_seed(42)
        model = PPOController(9,16)
        x = torch.randn(32,7,9)
        e = sample_episodes(model,x,torch.randint(0,2,(32,8)))
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0003)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_ids = [f'parent:{i}' for i in range(32)]
            (root/'manifest.json').write_text(json.dumps(dict(parent_config={},parent_iteration=7,
                training='/parent',parent_train_ids=old_ids)))
            opt = SimpleNamespace(wandb_mode='disabled',wandb_project='test',wandb_entity=None,wandb_name=None)
            session = PPOLoggingSession(root,opt,'fresh')
            session.measured('reset_cache',lambda *args: True,None)
            for group in range(2):
                q = outcomes(group*16)
                session.measured('decode_cohort',lambda *args,_rows=q: (_rows,.1),
                                  None,None,None,None,None,None,None,'live_train_reward')
            with patch('scripts.run_ppo_with_wandb.prior_coverage',return_value=(set(old_ids),224)):
                session.update(model,optimizer,x,e,x,torch.zeros(32,7,dtype=torch.bool),config)
            session.event(lambda *args,**kwargs: None,root,stage='one_iteration_complete',
                total_controller_iterations=8,correct=20,mean_selected_K=15.,seconds_including_live_A_B=1.)
            session.close(0)
            last=json.loads((root/'training_metrics.jsonl').read_text().splitlines()[-1])
            self.assertEqual(last['data/episodes_seen'],256)
            self.assertEqual(last['data/unique_PPO_questions_seen'],64)
            self.assertEqual(last['data/unique_auxiliary_questions'],32)
            self.assertEqual(last['data/episodes_this_iteration'],32)
            self.assertEqual(last['validation/available'],0)
            self.assertNotIn('validation/PPO/accuracy',last)
            self.assertIn('time/PPO_update_seconds',last)

    def test_actual_history_backfill_only_available_metrics(self):
        root=ROOT/'matched_runs/ppo_temp06_batch16_train32_20261015_v3'
        if not root.exists():
            self.skipTest('Historical run not installed')
        rows=saved_diagnostics(root)
        self.assertEqual(len(rows),7)
        self.assertEqual(rows[-1]['data/episodes_seen'],224)
        self.assertEqual(rows[-1]['data/unique_PPO_questions_seen'],32)
        self.assertNotIn('iteration/loss_total',rows[-1])
        self.assertNotIn('time/PPO_update_seconds',rows[-1])
        self.assertEqual(rows[-1]['validation/available'],0)


class ValidationTests(unittest.TestCase):
    def test_real_validation_selection_never_loads_test(self):
        original=validation.load_official_medqa
        with patch.object(validation,'load_official_medqa',wraps=original) as loader:
            items,meta=validation.validation_items(ROOT/'data/medqa_usmle_4options',16)
        self.assertEqual([c.args[1] for c in loader.call_args_list],['validation','train'])
        self.assertEqual(len(items),16)
        self.assertTrue(all(i['source_split']=='validation' for i in items))
        self.assertFalse(meta['test300_loaded'])
        with self.assertRaises(ValueError):
            validation.validation_items(ROOT/'data/medqa_usmle_4options',17)

    def test_rejects_test_and_train_before_model_calls(self):
        for split in ('test','train'):
            with self.assertRaises(ValueError):
                validation.evaluate_validation(None,None,None,[{'source_split':split}]*16,None,7)

    def test_mock_four_methods_restores_rng_mode_and_weights(self):
        model=torch.nn.Linear(2,2).train()
        weights={k:v.clone() for k,v in model.state_dict().items()}
        cpu=torch.get_rng_state();py=random.getstate();np_state=np.random.get_state()
        encoded={'A':{'input_ids':torch.ones(16,2,dtype=torch.long),'attention_mask':torch.ones(16,2,dtype=torch.long)}}
        args=SimpleNamespace(device='cpu',device2='cpu')
        items=[dict(source_split='validation',source_id=f'validation:{i}') for i in range(16)]
        def decode(wrapper,args,cohort,payloads,budgets,root,path,purpose):
            torch.rand(2);random.random();np.random.rand()
            return [dict(strict_correct=i<15,k=k,output_tokens=10,finish_reason='stop',
                         strict_parse_failed=False,repetition={'loop_flag':False})
                    for i,k in enumerate(budgets)],.1
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(validation,'render_batch',return_value=({},encoded)), \
                 patch.object(validation,'CohortDecisions',return_value=SimpleNamespace(budgets=torch.full((16,),10))), \
                 patch.object(validation,'online_rollout',return_value=(None,[],10)), \
                 patch.object(validation,'rollout',return_value=(None,[])), \
                 patch.object(validation,'prefixes_for_b',return_value=[{}]*16), \
                 patch.object(validation,'reset_cache') as resets, \
                 patch.object(validation,'decode_cohort',side_effect=decode) as calls:
                metrics=validation.evaluate_validation(None,args,model,items,Path(directory),7)
        self.assertEqual(calls.call_count,4)
        self.assertEqual(resets.call_count,4)
        self.assertEqual(metrics['validation/PPO/accuracy'],15/16)
        self.assertEqual(metrics['validation/PPO_minus_K10_accuracy'],0)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))
        self.assertTrue(all(torch.equal(v,weights[k]) for k,v in model.state_dict().items()))


class PatternTests(unittest.TestCase):
    def test_ties_and_mixed_correctness(self):
        grid = [10, 20, 30]
        samples = [dict(source_id=str(i), rows=[dict(strict_correct=value) for value in bits])
                   for i, bits in enumerate([[1, 1, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])]
        result = budget_patterns(samples, grid)
        self.assertEqual(result['categories'], dict(all_correct=1, all_wrong=1, mixed=3))
        self.assertEqual(result['hindsight_accuracy'], 4/5)
        self.assertEqual(result['hindsight_mean_steps'], 16)
        self.assertEqual(result['per_budget']['10'], dict(first_correct=2, uniquely_correct=0, oracle_selected=3))
        self.assertEqual(result['per_budget']['20'], dict(first_correct=1, uniquely_correct=1, oracle_selected=1))
        self.assertEqual(result['per_budget']['30'], dict(first_correct=1, uniquely_correct=1, oracle_selected=1))
        self.assertEqual(result['adjacent_transitions'], dict(wrong_to_correct=3, correct_to_wrong=2))
        self.assertEqual(result['train_only_checkpoint_action_labels'], dict(HANDOFF=6, CONTINUE_10=4))

    def test_empty_does_not_claim_accuracy(self):
        result = budget_patterns([], [10, 20])
        self.assertIsNone(result['hindsight_accuracy'])
        self.assertIsNone(result['hindsight_mean_steps'])

    def test_incomplete_or_invalid_budget_outcomes_are_rejected(self):
        for rows in ([dict(strict_correct=True)], [dict(strict_correct=2)] * 2):
            with self.subTest(rows=rows), self.assertRaises(AssertionError):
                budget_patterns([dict(source_id='x', rows=rows)], [10, 20])


if __name__=='__main__':
    unittest.main()
