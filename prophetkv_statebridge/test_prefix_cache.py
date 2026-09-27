import torch

from prophetkv_statebridge.prefix_cache import cache_nbytes, prefill_prefix_streaming
from prophetkv_statebridge.run_prefix_cache import common_prefix_length


def test_common_prefix_stops_before_role_specific_suffix():
    prompts = [
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[1, 2, 3, 9]]),
    ]
    assert common_prefix_length(prompts) == 3


def test_legacy_cache_size_is_counted_once():
    key = torch.zeros(1, 2, 3)
    value = torch.zeros(1, 2, 3)
    assert cache_nbytes(((key, value),)) == key.nbytes + value.nbytes


def test_streaming_prefill_rejects_invalid_chunk_size():
    try:
        prefill_prefix_streaming(None, torch.ones((1, 3), dtype=torch.long), chunk_tokens=0)
    except ValueError as error:
        assert "chunk_tokens" in str(error)
    else:
        raise AssertionError("Expected invalid chunk size to fail")
