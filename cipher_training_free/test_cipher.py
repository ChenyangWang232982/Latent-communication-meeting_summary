"""Small shape-level tests for training-free CIPHER utilities."""

import unittest

import torch

from cipher_training_free.cipher import collect_last_generation_states


class CollectLastGenerationStatesTest(unittest.TestCase):
    def test_collects_final_position_from_each_step(self):
        first_step_last_layer = torch.tensor([[[1.0], [2.0]]])
        second_step_last_layer = torch.tensor([[[3.0]]])
        hidden = collect_last_generation_states(
            (
                (torch.zeros_like(first_step_last_layer), first_step_last_layer),
                (torch.zeros_like(second_step_last_layer), second_step_last_layer),
            )
        )
        self.assertEqual(hidden.shape, (1, 2, 1))
        self.assertEqual(hidden.flatten().tolist(), [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
