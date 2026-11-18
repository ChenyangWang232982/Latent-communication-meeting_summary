import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from experiment_1.components.semantic_quality import add_bertscore


class SemanticQualityTest(unittest.TestCase):
    def sample(self):
        return {"id": "meeting", "reference": "a b", "result": {"answers": {"chunk_kv": "a", "chunk_text": "b"}},
                "quality": {"chunk_kv": {}, "chunk_text": {}}}

    def test_pair_mapping_and_explicit_overlength_rejection(self):
        seen = []
        class Tokenizer:
            model_max_length = 8
            def __call__(self, text, **kwargs):
                self.assert_no_truncation = kwargs["truncation"] is False
                return {"input_ids": [0] * (len(text.split()) + 2)}
        class Scorer:
            hash = "test-hash"
            def __init__(self, **kwargs):
                self._tokenizer = Tokenizer()
                seen.append(kwargs)
            def score(self, predictions, references, **kwargs):
                seen.append((predictions, references))
                return (SimpleNamespace(tolist=lambda: [.1, .2]),
                        SimpleNamespace(tolist=lambda: [.3, .4]), SimpleNamespace(tolist=lambda: [.5, .6]))
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(max_position_embeddings=8, model_type="test")))
        with patch.dict(sys.modules, {"bert_score": SimpleNamespace(BERTScorer=Scorer), "transformers": fake_transformers}), \
             patch("experiment_1.components.semantic_quality.importlib.metadata.version", return_value="test"):
            sample = self.sample()
            info = add_bertscore([sample], "encoder", 2, "cpu")
            self.assertEqual(sample["quality"]["chunk_kv"]["bertscore_f1"], .5)
            self.assertEqual(sample["quality"]["chunk_text"]["bertscore_f1"], .6)
            self.assertEqual(info["hash"], "test-hash")
            self.assertEqual(seen[1], (["a", "b"], ["a b", "a b"]))
            count = len(seen)
            sample["result"]["answers"]["chunk_kv"] = "word " * 10
            with self.assertRaisesRegex(ValueError, "No text was truncated"):
                add_bertscore([sample], "encoder", 2, "cpu")
            self.assertEqual(len(seen), count)

    def test_longformer_sentinel_is_replaced_on_scorer_tokenizer(self):
        class Tokenizer:
            model_max_length = 10 ** 30
            def __call__(self, text, **kwargs):
                assert kwargs["truncation"] is False
                assert kwargs["add_prefix_space"] is True
                assert text == text.strip()
                return {"input_ids": [0] * (len(text.split()) + 2)}
        class Scorer:
            hash = "sentinel-test"
            def __init__(self, **kwargs):
                self._tokenizer = Tokenizer()
            def score(self, *args, **kwargs):
                assert self._tokenizer.model_max_length == 4096
                return tuple(SimpleNamespace(tolist=lambda: [.5, .5]) for _ in range(3))
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(
                max_position_embeddings=4098, model_type="longformer")))
        with patch.dict(sys.modules, {"bert_score": SimpleNamespace(BERTScorer=Scorer), "transformers": fake_transformers}), \
             patch("experiment_1.components.semantic_quality.importlib.metadata.version", return_value="test"):
            sample = self.sample()
            info = add_bertscore([sample], "longformer", 12, "cpu")
            self.assertEqual(info["encoder_context_limit"], 4096)
            sample["reference"] = "word " * 4095
            with self.assertRaisesRegex(ValueError, "4097>4096"):
                add_bertscore([sample], "longformer", 12, "cpu")
