import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import unittest
from typing import Dict, List


class SummaryTests(unittest.TestCase):
    def test_saved_latent_steps(self):
        # Execute only the writer, without importing model/GPU dependencies.
        tree = ast.parse((Path(__file__).parents[1] / 'run.py').read_text())
        writer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'save_results')
        ns = dict(argparse=argparse, datetime=datetime, json=json, os=os, Dict=Dict, List=List)
        exec(compile(ast.Module(body=[writer], type_ignores=[]), 'writer', 'exec'), ns)
        summary = dict(method='latent_mas', model='test', task='medqa', split='test',
                       seed=42, max_samples=1, correct=0, accuracy=0,
                       total_time_sec=1, time_per_sample_sec=1, latent_steps=20)
        args = argparse.Namespace(model_name='test', method='latent_mas', task='medqa')
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                name = ns['save_results']([], summary, args)
                self.assertIn('Latent steps   : 20 (per latent agent)',
                              Path('output', name + '_summary.txt').read_text())
                self.assertEqual(json.loads(Path('output', name + '.json').read_text())
                                 ['summary']['latent_steps'], 20)
        finally:
            os.chdir(cwd)
