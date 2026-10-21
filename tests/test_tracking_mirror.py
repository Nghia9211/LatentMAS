import json
from pathlib import Path
import tempfile
import unittest
from controller.tracking_mirror import read_complete_rows


class MirrorReaderTests(unittest.TestCase):
    def test_missing_and_partial_log_during_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.jsonl"
            self.assertEqual(read_complete_rows(path), [])
            path.write_text('{"iteration":1}\n{"iteration":')
            self.assertEqual(read_complete_rows(path), [{"iteration": 1}])
            with path.open("a") as stream:
                stream.write('2}\n')
            self.assertEqual(read_complete_rows(path), [{"iteration": 1}, {"iteration": 2}])

    def test_malformed_completed_record_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.jsonl"
            path.write_text('invalid\n')
            with self.assertRaises(json.JSONDecodeError):
                read_complete_rows(path)
