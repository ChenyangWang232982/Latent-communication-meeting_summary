import torch
from types import SimpleNamespace

from prophetkv_statebridge.prefix_cache import cache_nbytes, generate_after_prefix_cache
from prophetkv_statebridge.run_prefix_cache import (
    align_handoffs,
    common_prefix_length,
    log_progress,
    receiver_prompt,
    split_handoff_states,
    split_structured_handoff_states,
    specialist_user,
)


def test_common_prefix_stops_before_role_specific_suffix():
    prompts = [
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[1, 2, 3, 9]]),
    ]
    assert common_prefix_length(prompts) == 3


def test_common_prefix_handles_long_prompts_and_shorter_roles():
    shared = torch.arange(15_000)
    prompts = [
        torch.cat((shared, torch.tensor([1]))).unsqueeze(0),
        torch.cat((shared, torch.tensor([2]))).unsqueeze(0),
        shared.unsqueeze(0),
    ]
    assert common_prefix_length(prompts) == 15_000
    prompts[1][0, 7_500] = -1
    assert common_prefix_length(prompts) == 7_500


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


def test_handoff_windows_retain_a_tail_budget_per_window():
    states = torch.arange(384, dtype=torch.float32).unsqueeze(-1)
    token_ids = torch.arange(384)

    segments = split_handoff_states(
        states, token_ids, state_budget=96, chunk_tokens=192,
    )

    assert [segment["states"].size(0) for segment in segments] == [96, 96]
    assert segments[0]["token_ids"][0].item() == 96
    assert segments[0]["token_ids"][-1].item() == 191
    assert segments[1]["token_ids"][0].item() == 288
    assert segments[1]["token_ids"][-1].item() == 383


def test_handoff_without_window_preserves_legacy_single_tail_budget():
    states = torch.arange(384, dtype=torch.float32).unsqueeze(-1)
    token_ids = torch.arange(384)

    segments = split_handoff_states(
        states, token_ids, state_budget=96, chunk_tokens=None,
    )

    assert len(segments) == 1
    assert segments[0]["token_ids"][0].item() == 288


def test_structured_handoff_samples_each_status_item_including_its_start_and_end():
    message = "CONFIRMED_DECISION: Alpha\nCOMMITTED_ACTION: Beta"
    token_ids = torch.arange(len(message))
    states = token_ids.float().unsqueeze(-1)

    class CharacterTokenizer:
        def __call__(self, text, **kwargs):
            assert text == message
            return {
                "input_ids": list(range(len(text))),
                "offset_mapping": [(index, index + 1) for index in range(len(text))],
            }

    segments = split_structured_handoff_states(
        CharacterTokenizer(), message, states, token_ids, ratio=0.5,
    )

    assert len(segments) == 2
    assert segments[0]["token_ids"][0].item() == 0
    assert segments[0]["token_ids"][-1].item() == len("CONFIRMED_DECISION: Alpha") - 1
    assert segments[1]["token_ids"][0].item() == len("CONFIRMED_DECISION: Alpha\n")
    assert segments[1]["token_ids"][-1].item() == len(message) - 1


def test_unfiltered_meeting_handoff_does_not_request_quotes_or_limit_items():
    prompt = specialist_user(
        "Meeting transcript", {"question": "Summarize the meeting"}, "actions",
        meeting_mode=True, evidence_policy="off",
    )
    assert "supporting quote" not in prompt.lower()
    assert "at most" not in prompt.lower()
    assert "do not write a full meeting summary" in prompt.lower()
    assert "COMMITTED_ACTION:" in prompt


def test_structured_meeting_handoff_limits_items_without_padding():
    prompt = specialist_user(
        "Meeting transcript", {"question": "Summarize the meeting"}, "actions",
        meeting_mode=True, evidence_policy="off", max_items=8,
    )
    assert "at most 8 self-contained items" in prompt
    assert "never pad or invent items" in prompt


def test_meeting_receiver_routes_explicit_status_labels():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return " ".join(message["content"] for message in messages)

    prompt = receiver_prompt(
        Tokenizer(), "Summarize", False, meeting_mode=True,
    )
    assert "Status labels are authoritative routing signals" in prompt
    assert "must NOT cancel a labelled confirmed decision" in prompt


def test_progress_logging_is_opt_in(capsys):
    log_progress(SimpleNamespace(debug_progress=False), "block 1/2")
    assert capsys.readouterr().out == ""
    log_progress(SimpleNamespace(debug_progress=True), "block 1/2")
    assert "block 1/2" in capsys.readouterr().out


def test_specialist_generation_reports_progress_without_eos():
    class Model:
        def __call__(self, **kwargs):
            logits = torch.tensor([[[0.0, 1.0, 0.0]]])
            return SimpleNamespace(
                logits=logits,
                past_key_values=(),
                hidden_states=(torch.ones(1, 1, 2),),
            )

    counts = []
    ids, states = generate_after_prefix_cache(
        Model(), SimpleNamespace(eos_token_id=2, pad_token_id=2), (), 1,
        torch.tensor([[0]]), max_context_tokens=132, progress_callback=counts.append,
    )
    assert ids.shape == (1, 130)
    assert states.shape == (130, 2)
    assert counts == [128]


def test_specialist_generation_honors_optional_max_new_tokens():
    class Model:
        def __call__(self, **kwargs):
            return SimpleNamespace(
                logits=torch.tensor([[[0.0, 1.0, 0.0]]]),
                past_key_values=(),
                hidden_states=(torch.ones(1, 1, 2),),
            )

    ids, states = generate_after_prefix_cache(
        Model(), SimpleNamespace(eos_token_id=2, pad_token_id=2), (), 1,
        torch.tensor([[0]]), max_context_tokens=200, max_new_tokens=7,
    )
    assert ids.shape == (1, 7)
    assert states.shape == (7, 2)
