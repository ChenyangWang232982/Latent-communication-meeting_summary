"""Run in latent-agents; no downloaded model or GPU required."""
import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "requires the model environment")
class RefillCacheTest(unittest.TestCase):
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
