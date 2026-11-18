import json
import tempfile
import unittest
from pathlib import Path

from experiment_1.components.generation_trace import GenerationTrace


class GenerationTraceTest(unittest.TestCase):
    def test_partial_and_finished_snapshots_warn_without_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            messages = []
            trace = GenerationTrace(Path(directory), 2, "meeting/unsafe-id", messages.append)
            event = {"stage": "chunk_kv_receiver", "tokens": 256, "text": "repeated output",
                     "raw_tail": "raw", "tail_token_ids": [1], "repeated_tail": True,
                     "eos_token_ids": [2], "finished": False, "stop_reason": None}
            trace(event)
            trace({**event, "tokens": 257, "text": "final text", "finished": True, "stop_reason": "eos"})
            folder = Path(directory) / "live/00002"
            self.assertEqual((folder / "chunk_kv_receiver.partial.txt").read_text(encoding="utf-8"), "final text")
            status = json.loads((folder / "chunk_kv_receiver.status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["tokens"], 257)
            self.assertEqual(status["stop_reason"], "eos")
            self.assertNotIn("text", status)
            self.assertEqual(len(messages), 1)
            self.assertFalse(list(folder.glob("*.tmp")))
