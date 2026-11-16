import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment_1.components.generation_diagnostic import run


class DiagnosticTest(unittest.TestCase):
    def test_full_then_routed_keep_same_record_and_config(self):
        calls = []

        class Runner:
            def __init__(self, config, progress):
                self.tokenizer = SimpleNamespace(eos_token_id=2)
                self.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2]))

            def run(self, sample_id, context, **kwargs):
                full = kwargs["diagnostic_full_source"]
                calls.append((sample_id, context, full, kwargs["variants"], kwargs["review"]))
                return {"answers": {"routed_kv": "full answer" if full else "routed answer"},
                        "diagnostic_full_source": full, "source_positions": 2,
                        "selected_source_positions": [0, 1] if full else [0],
                        "variants": {"routed_kv": {"generated_tokens": 2}}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "input.jsonl"
            data.write_text(json.dumps({"id": "repeat", "transcript": "A: content", "summary": "ref"}), encoding="utf-8")
            args = Namespace(input=data, sample_id="repeat", sample_index=0, condition="both", output_dir=root / "out",
                             model="fake", device="cpu", seed=12, role_budget_ratio=.25, span_tokens=32,
                             query_batch_tokens=8, speaker_neighbor_blocks=1, vector_routing=False,
                             embedding_model="fake", vector_weight=.3)
            with patch("experiment_1.components.generation_diagnostic.ReceiverKVRunner", Runner), \
                 patch("experiment_1.components.generation_diagnostic.prompt_parts", return_value=([1, 2], [3], [])):
                directory = run(args)
            self.assertEqual(calls, [("repeat", "A: content", True, ("routed_kv",), False),
                                     ("repeat", "A: content", False, ("routed_kv",), False)])
            self.assertEqual((directory / "full/summary.txt").read_text(), "full answer")
            status = json.loads((directory / "routed/status.json").read_text())
            self.assertFalse(status["diagnostic_full_source"])
            self.assertEqual(status["retained_positions"], 1)
            self.assertFalse((directory / "summary.json").exists())
