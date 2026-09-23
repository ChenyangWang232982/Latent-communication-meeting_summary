import unittest

import torch

from interlat_same_model.latent import HiddenStateAdapter, LatentCompressor


class InterlatComponentTest(unittest.TestCase):
    def test_same_model_adapter_preserves_trajectory_shape(self):
        adapter = HiddenStateAdapter(source_size=16, target_size=16, num_heads=4)
        states = torch.randn(2, 11, 16)
        self.assertEqual(adapter(states).shape, states.shape)

    def test_compressor_reduces_trajectory_length(self):
        compressor = LatentCompressor(hidden_size=16, num_heads=4, output_length=3)
        result = compressor(torch.randn(2, 11, 16))
        self.assertEqual(result.shape, (2, 3, 16))


if __name__ == "__main__":
    unittest.main()
