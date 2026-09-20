from unittest.mock import patch

from meeting_latent_workflow.config import WorkflowConfig
from meeting_latent_workflow.graph import build_workflow, split_transcript


class FakeTokenizer:
    def encode(self, text, **_kwargs):
        return text.split()

    def decode(self, token_ids, **_kwargs):
        return " ".join(token_ids)


class FakeTextAgent:
    tokenizer = FakeTokenizer()

    def __init__(self, *_args):
        pass

    def generate(self, prompt):
        return "APPROVED" if "Reply with either" in prompt else "note"


class FakeCipherAgent:
    def __init__(self, *_args):
        pass

    def generate(self, *_args):
        return "note"


def test_split_transcript_preserves_token_budget():
    chunks = split_transcript(
        "one two three four five six seven eight nine ten eleven twelve",
        FakeTokenizer(),
        chunk_tokens=5,
        overlap_tokens=1,
    )
    assert len(chunks) == 3
    assert all(len(chunk["text"].split()) <= 5 for chunk in chunks)


def test_hierarchical_graph_collects_all_chunk_reports():
    with (
        patch("meeting_latent_workflow.graph.load_system", return_value=object()),
        patch("meeting_latent_workflow.graph.TextAgent", FakeTextAgent),
        patch("meeting_latent_workflow.graph.CipherSummaryAgent", FakeCipherAgent),
    ):
        graph = build_workflow(
            WorkflowConfig(
                chunk_tokens=5,
                chunk_overlap_tokens=1,
                reduce_group_size=2,
            )
        )
        result = graph.invoke({
            "transcript": "one two three four five six seven eight nine ten eleven twelve"
        })

    assert result["chunk_count"] == 3
    assert len(result["chunk_reports"]) == 3
    assert result["final_summary"] == "note"
