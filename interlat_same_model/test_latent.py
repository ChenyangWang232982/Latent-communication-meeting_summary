import unittest

import torch

from interlat_same_model.latent import HiddenStateAdapter, LatentCompressor


class DummyReceiver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8, "pad_token_id": 0})()
        self.embedding = torch.nn.Embedding(32, 8)

    def get_input_embeddings(self):
        return self.embedding


class InterlatComponentTest(unittest.TestCase):
    def test_same_model_adapter_preserves_trajectory_shape(self):
        adapter = HiddenStateAdapter(source_size=16, target_size=16, num_heads=4)
        states = torch.randn(2, 11, 16)
        self.assertEqual(adapter(states).shape, states.shape)

    def test_compressor_reduces_trajectory_length(self):
        compressor = LatentCompressor(hidden_size=16, num_heads=4, output_length=3)
        result = compressor(torch.randn(2, 11, 16))
        self.assertEqual(result.shape, (2, 3, 16))

    def test_latents_follow_the_receiver_prompt(self):
        from interlat_same_model.latent import InterlatReceiver

        receiver = DummyReceiver()
        model = InterlatReceiver(receiver, source_hidden_size=8, num_heads=2)
        prompt = torch.tensor([[3, 4, 5]])
        target = torch.tensor([[6, 7]])
        embeds, _, labels, latent = model.build_inputs(
            torch.randn(1, 4, 8), bop_id=1, eop_id=2, prompt_ids=prompt, target_ids=target
        )
        self.assertEqual(embeds.shape[1], 3 + 1 + 4 + 1 + 2)
        self.assertTrue(torch.equal(embeds[:, :3], receiver.embedding(prompt)))
        self.assertTrue(torch.equal(embeds[:, 3], model.boundary_embeddings[0].view(1, -1)))
        self.assertTrue(torch.equal(embeds[:, 8], model.boundary_embeddings[1].view(1, -1)))
        self.assertTrue(torch.equal(labels[:, :9], torch.full((1, 9), -100)))
        self.assertEqual(latent.shape, (1, 4, 8))


if __name__ == "__main__":
    unittest.main()
