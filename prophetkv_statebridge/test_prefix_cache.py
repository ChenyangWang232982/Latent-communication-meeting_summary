import torch

from prophetkv_statebridge.prefix_cache import cache_nbytes
from prophetkv_statebridge.run_prefix_cache import role_suffix, shared_prefix


def test_shared_prefix_is_task_independent():
    assert "Question:" not in shared_prefix("A meeting transcript")
    assert "Source transcript:" in shared_prefix("A meeting transcript")
    assert "Question: What changed?" in role_suffix({"question": "What changed?"}, "facts")


def test_legacy_cache_size_is_counted_once():
    key = torch.zeros(1, 2, 3)
    value = torch.zeros(1, 2, 3)
    assert cache_nbytes(((key, value),)) == key.nbytes + value.nbytes
