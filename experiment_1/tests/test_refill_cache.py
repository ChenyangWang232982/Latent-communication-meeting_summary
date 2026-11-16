"""Run in latent-agents; no downloaded model or GPU required."""
import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "requires the model environment")
class RefillCacheTest(unittest.TestCase):
    def test_seeded_sampling_is_repeatable_filtered_and_observer_neutral(self):
        import torch
        from receiver_kv_workflow.cache import generate_from_prefill

        class Tokenizer:
            eos_token_id = 4
            def decode(self, ids, **kwargs):
                return str(ids)

        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=4))
        logits = torch.tensor([[[2.0, 1.9, 1.8, 1.7, -10.0]]])
        original = logits.clone()
        for seed in range(8):
            observed = []
            torch.manual_seed(seed)
            a = generate_from_prefill(model, Tokenizer(), SimpleNamespace(logits=logits),
                                     next_position=1, context_limit=2, temperature=.7, top_p=.8, top_k=3,
                                     on_token=lambda raw, token, step: observed.append((raw.clone(), token)))
            torch.rand(17)
            torch.manual_seed(seed)
            b = generate_from_prefill(model, Tokenizer(), SimpleNamespace(logits=logits),
                                     next_position=1, context_limit=2, temperature=.7, top_p=.8, top_k=3)
            self.assertEqual(a, b)
            self.assertIn(observed[0][1], (0, 1, 2))
            torch.testing.assert_close(observed[0][0], original[:, -1])
            torch.testing.assert_close(logits, original)
        greedy = generate_from_prefill(model, Tokenizer(), SimpleNamespace(logits=logits),
                                       next_position=1, context_limit=2, temperature=0, top_p=.01, top_k=1)
        self.assertEqual(greedy[0], "[0]")

    def test_token_observation_preserves_decode_and_saves_eos_logits(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        import receiver_kv_workflow.cache as cache
        from experiment_1.components.generation_trace import GenerationTrace

        class Tokenizer:
            eos_token_id = 2
            def decode(self, ids, **kwargs):
                return str(ids)

        def output(token):
            logits = torch.zeros(1, 1, 4)
            logits[0, 0, token] = 3
            return SimpleNamespace(logits=logits, past_key_values=None)

        def forward(model, token, past, position, **kwargs):
            return output(2 if position >= 3 else 1)

        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
        with tempfile.TemporaryDirectory() as directory, patch.object(cache, "forward_tokens", forward):
            trace = GenerationTrace(Path(directory), 0, "meeting", lambda _: None)
            stage = "routed_kv_receiver"
            trace({"event": "input", "stage": stage, "sample_id": "meeting", "prefix_token_ids": [10, 11],
                   "suffix_token_ids": [12], "selected_positions": [0], "cache_condition": "routed", "eos_token_ids": [2]})
            expected = cache.generate_from_prefill(model, Tokenizer(), output(1), next_position=1, context_limit=8)
            actual = cache.generate_from_prefill(model, Tokenizer(), output(1), next_position=1, context_limit=8,
                        on_token=lambda logits, token, step: trace.token(stage, logits, token, step))
            trace.flush()
            self.assertEqual(expected, actual)
            path = trace.directory / f"{stage}.tokens.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], list(range(actual[1])))
            self.assertEqual(rows[-1]["token_id"], 2)
            self.assertEqual(rows[-1]["eos_logits"], [{"token_id": 2, "logit": 3.0}])
            self.assertEqual(rows[0]["top1_top2_margin"], 3.0)

    def test_generated_only_penalty_breaks_loop_without_mutating_raw_logits(self):
        from unittest.mock import patch
        import torch
        import receiver_kv_workflow.cache as cache
        class Tokenizer:
            eos_token_id = 2
            def decode(self, ids, **kwargs):
                return str(ids)
        logits = torch.tensor([[[10., 9.8, 9.7]]])
        output = SimpleNamespace(logits=logits, past_key_values=None)
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
        observed = []
        with patch.object(cache, "forward_tokens", return_value=output):
            baseline = cache.generate_from_prefill(model, Tokenizer(), output, next_position=1, context_limit=5)
            penalized = cache.generate_from_prefill(model, Tokenizer(), output, next_position=1,
                context_limit=5, repetition_penalty=1.1,
                on_token=lambda raw, token, step: observed.append(raw.clone()))
        self.assertEqual(baseline[0], "[0, 0, 0, 0]")
        self.assertEqual(penalized[0], "[0, 1]")
        self.assertFalse(penalized[2])  # natural EOS remains available
        for raw in observed:
            torch.testing.assert_close(raw, logits[:, -1])
        for invalid in (0, .9, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                cache.generate_from_prefill(model, Tokenizer(), output, next_position=1,
                    context_limit=5, repetition_penalty=invalid)

    def test_uses_exact_selected_ids_and_original_positions_without_old_cache(self):
        import torch
        from receiver_kv_workflow.cache import refill_selected_cache
        seen = {}

        def model(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(past_key_values="recomputed")

        prefix = torch.tensor([[10, 11, 12, 13, 14, 15]])
        result = refill_selected_cache(model, prefix, [0, 3, 5])
        self.assertEqual(result, "recomputed")
        self.assertEqual(seen["input_ids"].tolist(), [[10, 13, 15]])
        self.assertEqual(seen["position_ids"].tolist(), [[0, 3, 5]])
        self.assertEqual(seen["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertIsNone(seen["past_key_values"])
        for invalid in ([], [3, 0], [0, 0], [-1], [6]):
            with self.assertRaises(ValueError):
                refill_selected_cache(model, prefix, invalid)
