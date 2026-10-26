import unittest

from .selection import select_spans, union_positions


class SelectionTests(unittest.TestCase):
    def test_summary_prompt_preserves_pending_coordination(self):
        from .prompts import EVENT_QUERY, ROLE_QUERIES, SUMMARY_REQUEST, TIMELINE_REQUEST
        self.assertIn("coordination dependency", SUMMARY_REQUEST)
        self.assertIn("attendance confirmation", SUMMARY_REQUEST)
        self.assertIn("coordination", ROLE_QUERIES)
        self.assertIn("exact times", ROLE_QUERIES["coordination"])
        self.assertIn("middle of that event's discussion", TIMELINE_REQUEST)
        self.assertIn("time unavailable", TIMELINE_REQUEST)
        self.assertIn("middle of each event discussion", EVENT_QUERY)

    def test_selected_evidence_prompt_keeps_verbatim_timestamp_context(self):
        import re
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from .prompts import prompt_parts, selected_evidence_prompt
        template = "{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        transcript = "[10:30] Toufique will send the invitation. Zhang confirmation is pending."
        # Make a tiny but readable tokenizer for a deterministic offset test.
        from .prompts import SUMMARY_REQUEST, SYSTEM
        vocabulary = {"<unk>": 0, "<eos>": 1}
        for token in re.findall(r"\w+|[^\w\s]", SYSTEM + SUMMARY_REQUEST + transcript + "system user assistant"):
            vocabulary.setdefault(token, len(vocabulary))
        backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
        tokenizer.chat_template = template
        prefix, _, _ = prompt_parts(tokenizer, transcript)
        selected = list(range(len(prefix)))
        prompt, evidence_tokens, count = selected_evidence_prompt(tokenizer, transcript, selected)
        rendered = tokenizer.decode(prompt, skip_special_tokens=False)
        self.assertIn("Toufique will send the invitation", rendered)
        self.assertIn("Zhang confirmation is pending", rendered)
        self.assertGreater(evidence_tokens, 0)
        self.assertGreaterEqual(count, 1)

    def test_budget_and_order(self):
        selected = select_spans([0, 1, 9, 8, 0, 0], 4, span_size=2, protected=[0, 5])
        self.assertEqual(selected, [0, 2, 3, 5])

    def test_full_and_union(self):
        self.assertEqual(select_spans([1] * 7, 7), list(range(7)))
        self.assertEqual(union_positions([[0, 3, 4], [0, 1, 4]]), [0, 1, 3, 4])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            select_spans([1, 2], 1, protected=[0, 1])
        with self.assertRaises(ValueError):
            select_spans([float("nan")], 1)


class QwenCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        cls.torch = torch
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=256, eos_token_id=63)
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.model.set_attn_implementation("sdpa")

    def setUp(self):
        from .cache import forward_tokens
        self.ids = self.torch.tensor([[1, 5, 7, 8, 9, 11, 3, 4]])
        with self.torch.inference_mode():
            self.source = forward_tokens(self.model, self.ids, None, 0, logits_to_keep=1).past_key_values

    def test_exact_reuse_matches_full_prefill(self):
        from .cache import forward_tokens, select_cache
        torch = self.torch
        suffix = torch.tensor([[6, 12, 4]])
        with torch.inference_mode():
            reused = forward_tokens(self.model, suffix, select_cache(self.source, range(8)), 8).logits
            fresh = forward_tokens(self.model, torch.cat([self.ids, suffix], dim=1), None, 0).logits[:, -3:]
        torch.testing.assert_close(reused, fresh, atol=1e-5, rtol=1e-5)

    def test_sparse_cache_equals_masked_full_cache(self):
        from .cache import forward_tokens, select_cache
        torch = self.torch
        selected = [0, 2, 3, 7]
        suffix = torch.tensor([[6, 12]])
        mask = torch.zeros((1, 10), dtype=torch.long)
        mask[:, selected] = 1
        mask[:, 8:] = 1
        with torch.inference_mode():
            sparse = forward_tokens(self.model, suffix, select_cache(self.source, selected), 8).logits
            masked = self.model(input_ids=suffix, past_key_values=select_cache(self.source, range(8)),
                                attention_mask=mask, position_ids=torch.tensor([[8, 9]]), use_cache=True).logits
        torch.testing.assert_close(sparse, masked, atol=1e-5, rtol=1e-5)
        self.assertEqual(self.source.get_seq_length(), 8)

    def test_query_batching_and_immutability(self):
        from .cache import pairs
        from .scoring import query_scores
        torch = self.torch
        original = [(k.clone(), v.clone()) for k, v in pairs(self.source)]
        query = torch.tensor([[6, 7, 12, 13, 2]])
        once = query_scores(self.model, self.source, query, query_batch=5)
        batched = query_scores(self.model, self.source, query, query_batch=2)
        torch.testing.assert_close(torch.tensor(once), torch.tensor(batched), atol=1e-5, rtol=1e-5)
        for before, after in zip(original, pairs(self.source)):
            for a, b in zip(before, after):
                self.assertTrue(torch.equal(a, b))
        self.assertEqual(self.model.config._attn_implementation, "sdpa")

    def test_generation_cap_and_context_guard(self):
        from .cache import generate, select_cache
        class Tokenizer:
            eos_token_id = 63
            def decode(self, tokens, **kwargs):
                return " ".join(map(str, tokens))
        result = generate(self.model, Tokenizer(), select_cache(self.source, [0, 7]),
                          self.torch.tensor([[5]]), absolute_start=8, max_new_tokens=3, context_limit=32)
        self.assertLessEqual(result[1], 3)
        with self.assertRaises(ValueError):
            generate(self.model, Tokenizer(), self.source, self.torch.tensor([[5]]),
                     absolute_start=8, max_new_tokens=3, context_limit=10)

    def test_local_cli_all_variants(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from .run import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            with self.torch.no_grad():
                self.model.lm_head.weight.zero_()
            self.model.config.eos_token_id = 0
            self.model.generation_config.eos_token_id = 0
            self.model.save_pretrained(model_dir)
            backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 63}, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
            tokenizer.chat_template = "{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
            tokenizer.save_pretrained(model_dir)
            (root / "meeting.txt").write_text("Alice: I will send the draft on Friday. Bob: Agreed.", encoding="utf-8")
            command = ["run", "--transcript", str(root / "meeting.txt"), "--model", str(model_dir),
                       "--device", "cpu", "--max-new-tokens", "2", "--role-budget-ratio", "0.7",
                       "--selection-strategy", "sentence",
                       "--timeline", "--output", str(root / "meeting_output.txt"), "--metrics-dir", str(root / "metrics")]
            with patch("sys.argv", command):
                main()
            report = json.loads(next((root / "metrics").glob("*/metrics.json")).read_text())
            self.assertEqual(set(report["variants"]), {"routed_kv", "full_kv", "selected_text", "specialist_text", "natural_text"})
            self.assertTrue((root / "meeting_output.txt").is_file())
            self.assertTrue((root / "meeting_output.full_kv.txt").is_file())
            self.assertTrue((root / "meeting_output.timeline.txt").is_file())
            self.assertIsNone(report["variants"]["routed_kv"]["rouge_l_f1"])
            self.assertNotIn("answers", report)
            self.assertEqual(report["settings"]["selection_strategy"], "sentence")
            self.assertLessEqual(len(report["selected_source_positions"]), report["baseline_selected_positions"])


if __name__ == "__main__":
    unittest.main()
