import operator
from typing import Annotated, TypedDict


class TranscriptChunk(TypedDict):
    index: int
    text: str


class ChunkReport(TypedDict):
    index: int
    topics: str
    decisions: str
    actions: str


class MeetingWorkflowState(TypedDict, total=False):
    """State exchanged by the hierarchical LangGraph meeting workflow."""

    transcript: str
    chunk: TranscriptChunk
    chunks: list[TranscriptChunk]
    chunk_count: int
    chunk_reports: Annotated[list[ChunkReport], operator.add]
    consolidated_evidence: str
    draft_summary: str
    critique: str
    final_summary: str
    refinement_round: int
    needs_refinement: bool
