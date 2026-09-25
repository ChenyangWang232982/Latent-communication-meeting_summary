"""Small CPU tests for paper-stage Interlat mechanics."""

from __future__ import annotations

import unittest

import torch

from interlat_paper.model import IGNORE_INDEX, InterlatActor
from interlat_paper.objectives import plan_alignment_loss, random_contrast_loss


class DummyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8, "pad_token_id": 0})()
        self.embedding = torch.nn.Embedding(32, 8)

    def get_input_embeddings(self):
        return self.embedding


class PaperComponentTest(unittest.TestCase):
    def test_curriculum_keeps_latent_prefix_and_plan_suffix(self):
        actor = DummyActor()
        model = InterlatActor(actor, source_hidden_size=8, num_heads=2).eval()
        latent = torch.randn(1, 4, 8)
        plan_ids = torch.tensor([[3, 4, 5, 6]])
        mixed = model.mix_plan_tokens(latent, plan_ids, replacement_rate=0.5)
        self.assertTrue(torch.equal(mixed[:, :2], latent[:, :2]))
        self.assertTrue(torch.equal(mixed[:, 2:], actor.embedding(plan_ids)[:, 2:]))

    def test_adapter_clamps_inserted_latent_range(self):
        actor = DummyActor()
        model = InterlatActor(actor, source_hidden_size=8, num_heads=2).eval()
        with torch.no_grad():
            model.adapter.adaptive_projection.scale.fill_(100.0)
            model.adapter.adaptive_projection.output_scale.fill_(1.0)
        output = model.adapt_latents(torch.randn(1, 4, 8))
        self.assertLessEqual(float(output.max()), 10.0)
        self.assertGreaterEqual(float(output.min()), -10.0)

    def test_build_inputs_uses_vocab_boundary_embeddings_and_masks_prefix(self):
        actor = DummyActor()
        model = InterlatActor(actor, source_hidden_size=8, num_heads=2)
        prompt = torch.tensor([[3, 4]])
        assistant_prefix = torch.tensor([[7]])
        target = torch.tensor([[5, 6]])
        message = torch.randn(1, 3, 8)
        inputs, _, labels = model.build_inputs(prompt, assistant_prefix, target, message, bop_id=1, eop_id=2)
        self.assertEqual(inputs.shape, (1, 2 + 1 + 3 + 1 + 1 + 2, 8))
        self.assertTrue(torch.equal(inputs[:, 2], actor.embedding(torch.tensor([1]))))
        self.assertTrue(torch.equal(inputs[:, 7], actor.embedding(assistant_prefix).squeeze(1)))
        self.assertTrue(torch.equal(labels[:, :8], torch.full((1, 8), IGNORE_INDEX)))
        self.assertTrue(torch.equal(labels[:, 8:], target))

    def test_distribution_losses_handle_distinct_prefix_lengths(self):
        # Each labels tensor masks a different prefix, as latent and textual
        # messages often have different sequence lengths.
        latent_logits = torch.randn(1, 7, 11, requires_grad=True)
        plan_logits = torch.randn(1, 9, 11)
        mismatch_logits = torch.randn(1, 8, 11)
        latent_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3, 4]])
        plan_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3, 4]])
        mismatch_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3, 4]])
        align = plan_alignment_loss(latent_logits, latent_labels, plan_logits, plan_labels)
        separate = random_contrast_loss(
            latent_logits, latent_labels, mismatch_logits, mismatch_labels
        )
        self.assertTrue(torch.isfinite(align))
        self.assertTrue(torch.isfinite(separate))
        (align + separate).backward()
        self.assertIsNotNone(latent_logits.grad)


if __name__ == "__main__":
    unittest.main()
