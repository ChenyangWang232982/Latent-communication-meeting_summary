import json
import tempfile
import unittest
from pathlib import Path

from experiment_1.components.generation_comparison import compare
from experiment_1.components.generation_trace import GenerationTrace


class GenerationComparisonTest(unittest.TestCase):
    def test_first_divergence_uses_shared_prefix_and_rejects_different_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = "routed_kv_receiver"
            traces = []
            for name, condition, selected, tokens in (
                ("normal", "refill", [0, 2], [5, 6, 7]),
                ("formal", "routed", [0, 2], [5, 8, 9]),
            ):
                trace = GenerationTrace(root / name, 0, "meeting", lambda _: None)
                trace({"event": "input", "stage": stage, "sample_id": "meeting",
                       "prefix_token_ids": [10, 11, 12], "suffix_token_ids": [13],
                       "selected_positions": selected, "cache_condition": condition,
                       "eos_token_ids": [2], "config": {"seed": 12}})
                path = trace.directory / f"{stage}.tokens.jsonl"
                path.write_text("".join(json.dumps({"step": i, "token_id": token,
                                                    "top_logits": [{"token_id": token, "logit": 3.0}]}) + "\n"
                                        for i, token in enumerate(tokens)), encoding="utf-8")
                traces.append(path)
            result = compare(*traces)
            self.assertEqual(result["status"], "diverged")
            self.assertEqual(result["first_divergence"]["zero_based_step"], 1)
            self.assertEqual(result["first_divergence"]["reference"]["token_id"], 6)
            self.assertTrue(result["same_selected_positions"])
            traces[1].write_text(json.dumps({"step": 0, "token_id": 5}) + "\n", encoding="utf-8")
            self.assertEqual(compare(*traces)["status"], "matching_prefix_only")
            identity = traces[1].with_name(f"{stage}.input.json")
            value = json.loads(identity.read_text())
            value["suffix_token_sha256"] = "different"
            identity.write_text(json.dumps(value))
            self.assertEqual(compare(*traces)["status"], "incompatible_inputs")
