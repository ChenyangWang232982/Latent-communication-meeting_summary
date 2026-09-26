import torch

from .bridge import StateBridge


def test_alignment_returns_input_embedding_space_shape_and_finite_values():
    torch.manual_seed(0)
    vocab = torch.randn(32, 8)
    states = torch.randn(6, 8)
    token_ids = torch.tensor([1, 2, 3, 4, 5, 6])
    bridge = StateBridge(vocab, snap_ratio=0.3, vocab_chunk_size=8)

    aligned = bridge.align(states, token_ids)

    assert aligned.shape == states.shape
    assert torch.isfinite(aligned).all()
