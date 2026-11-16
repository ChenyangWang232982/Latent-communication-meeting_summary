import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class PrecisionTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("transformers"), "requires transformers")
    def test_qwen_forward_keeps_kv_bf16_and_returns_fp32_logits(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from receiver_kv_workflow.cache import pairs
        from receiver_kv_workflow.precision import enable_fp32_lm_head
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        enable_fp32_lm_head(model)
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([[3, 4, 5]]), use_cache=True, logits_to_keep=1)
        self.assertEqual(output.logits.dtype, torch.float32)
        self.assertEqual(model.get_input_embeddings().weight.dtype, torch.bfloat16)
        self.assertTrue(all(key.dtype == value.dtype == torch.bfloat16 for key, value in pairs(output.past_key_values)))

    def test_fp32_projection_resolves_bf16_rounding_tie(self):
        import torch
        from receiver_kv_workflow.precision import FP32OutputHead
        original = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            original.weight.copy_(torch.tensor([[16, 6.71875], [16, 6.75]], dtype=torch.bfloat16))
        hidden = torch.tensor([[1, 1]], dtype=torch.bfloat16)
        before = original(hidden)
        head = FP32OutputHead(original)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            after = head(hidden)
        self.assertEqual(before[0, 0], before[0, 1])
        self.assertEqual(before.argmax().item(), 0)
        self.assertEqual(after.dtype, torch.float32)
        self.assertEqual(after.argmax().item(), 1)
        torch.testing.assert_close(after, torch.tensor([[22.71875, 22.75]]))
        self.assertEqual(head.last_output_dtype, "torch.float32")

    def test_tied_embedding_is_not_upcast_or_changed(self):
        import torch
        from receiver_kv_workflow.precision import enable_fp32_lm_head

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(3, 2, dtype=torch.bfloat16)
                self.head = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
                self.head.weight = self.embedding.weight
            def get_output_embeddings(self):
                return self.head
            def set_output_embeddings(self, head):
                self.head = head

        model = Model()
        original = model.embedding.weight.clone()
        pointer = model.embedding.weight.data_ptr()
        enable_fp32_lm_head(model)
        self.assertEqual(model.embedding.weight.dtype, torch.bfloat16)
        self.assertEqual(model.embedding.weight.data_ptr(), pointer)
        torch.testing.assert_close(model.embedding.weight, original)
        self.assertEqual(model.head.weight.dtype, torch.float32)
        self.assertNotEqual(model.head.weight.data_ptr(), pointer)
