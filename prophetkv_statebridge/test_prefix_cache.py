import torch

from prophetkv_statebridge.prefix_cache import cache_nbytes
from prophetkv_statebridge.run_prefix_cache import align_handoffs, common_prefix_length, specialist_user


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


def test_per_message_alignment_preserves_order_and_uses_separate_sender_calls():
    class RecordingBridge:
        def __init__(self):
            self.lengths = []

        def align(self, states, token_ids):
            self.lengths.append(len(token_ids))
            return states + token_ids.unsqueeze(-1)

    segments = [
        {"states": torch.tensor([[1.0], [2.0]]), "token_ids": torch.tensor([10, 20])},
        {"states": torch.tensor([[3.0]]), "token_ids": torch.tensor([30])},
    ]
    bridge = RecordingBridge()
    aligned = align_handoffs(bridge, segments, "per-message")

    assert bridge.lengths == [2, 1]
    assert aligned[:, 0].tolist() == [11.0, 22.0, 33.0]

    bridge = RecordingBridge()
    align_handoffs(bridge, segments, "joint")
    assert bridge.lengths == [3]


def test_unfiltered_meeting_handoff_does_not_request_quotes_or_limit_items():
    prompt = specialist_user(
        "Meeting transcript", {"question": "Summarize the meeting"}, "actions",
        meeting_mode=True, evidence_policy="off",
    )
    assert "supporting quote" not in prompt.lower()
    assert "at most" not in prompt.lower()
    assert "do not write a full meeting summary" in prompt.lower()
