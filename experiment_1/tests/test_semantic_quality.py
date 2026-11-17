import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from experiment_1.components.semantic_quality import add_bertscore


class SemanticQualityTest(unittest.TestCase):
    def sample(self):
        return {"id": "meeting", "reference": "a b", "result": {"answers": {"routed_kv": "a", "specialist_text": "b"}},
                "quality": {"routed_kv": {}, "specialist_text": {}}}

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
            self.assertEqual(sample["quality"]["routed_kv"]["bertscore_f1"], .5)
            self.assertEqual(sample["quality"]["specialist_text"]["bertscore_f1"], .6)
            self.assertEqual(info["hash"], "test-hash")
            self.assertEqual(seen[1], (["a", "b"], ["a b", "a b"]))
            count = len(seen)
            sample["result"]["answers"]["routed_kv"] = "word " * 10
            with self.assertRaisesRegex(ValueError, "No text was truncated"):
                add_bertscore([sample], "encoder", 2, "cpu")
            self.assertEqual(len(seen), count)
