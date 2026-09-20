from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from meeting_latent_workflow.agents import CipherSummaryAgent, TextAgent, load_system
from meeting_latent_workflow.config import WorkflowConfig
from meeting_latent_workflow.state import ChunkReport, MeetingWorkflowState, TranscriptChunk


def _specialist_prompt(role: str, transcript: str) -> str:
    return (
        f"You are the {role} in a meeting-analysis team. Extract only factual "
        f"evidence from this transcript chunk. Use concise bullet points and do "
        f"not invent details.\n\nTranscript chunk:\n{transcript}"
    )


def _split_units(text: str) -> list[str]:
    """Prefer speaker/paragraph and sentence boundaries over raw token cuts."""
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    units: list[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        units.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return units or [text.strip()]


def split_transcript(
    transcript: str,
    tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[TranscriptChunk]:
    """Create token-bounded chunks while retaining a small textual overlap."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("chunk_overlap_tokens must be in [0, chunk_tokens)")

    chunks: list[TranscriptChunk] = []
    current_ids: list[int] = []
    for unit in _split_units(transcript):
        remaining_ids = tokenizer.encode(unit, add_special_tokens=False)
        while remaining_ids:
            available = chunk_tokens - len(current_ids)
            if available == 0:
                chunks.append({
                    "index": len(chunks),
                    "text": tokenizer.decode(current_ids, skip_special_tokens=True),
                })
                current_ids = current_ids[-overlap_tokens:] if overlap_tokens else []
                available = chunk_tokens - len(current_ids)

            take = min(available, len(remaining_ids))
            current_ids.extend(remaining_ids[:take])
            remaining_ids = remaining_ids[take:]

            if remaining_ids:
                chunks.append({
                    "index": len(chunks),
                    "text": tokenizer.decode(current_ids, skip_special_tokens=True),
                })
                current_ids = current_ids[-overlap_tokens:] if overlap_tokens else []

    if current_ids:
        chunks.append({
            "index": len(chunks),
            "text": tokenizer.decode(current_ids, skip_special_tokens=True),
        })
    return chunks


def format_chunk_report(report: ChunkReport) -> str:
    return (
        f"Chunk {report['index'] + 1}\n"
        f"Topics and evidence:\n{report['topics']}\n"
        f"Decisions:\n{report['decisions']}\n"
        f"Action items:\n{report['actions']}"
    )


def build_workflow(config: WorkflowConfig):
    """Build a token-bounded hierarchical meeting-summarization workflow."""
    if config.reduce_group_size < 2:
        raise ValueError("reduce_group_size must be at least 2")
    if config.chunk_tokens >= config.max_input_tokens:
        raise ValueError("chunk_tokens must leave room for the agent instruction")

    system = load_system(config)
    text_agent = TextAgent(system, config)
    cipher_agent = CipherSummaryAgent(system, config)

    def segment_transcript(state: MeetingWorkflowState):
        chunks = split_transcript(
            state["transcript"],
            text_agent.tokenizer,
            config.chunk_tokens,
            config.chunk_overlap_tokens,
        )
        if not chunks:
            raise ValueError("Transcript contains no usable text")
        return {"chunk_count": len(chunks), "chunks": chunks}

    def send_chunks(state: MeetingWorkflowState):
        return [
            Send("process_chunk", {"chunk": chunk})
            for chunk in state["chunks"]
        ]

    def process_chunk(state: MeetingWorkflowState):
        chunk = state["chunk"]
        clean_prompt = (
            "Rewrite this ASR transcript chunk into clean readable meeting text. "
            "Remove fillers and exact repetitions, but preserve decisions, action "
            "items, speakers when present, and uncertainty.\n\nTranscript chunk:\n"
            f"{chunk['text']}"
        )
        cleaned_chunk = text_agent.generate(clean_prompt)
        report: ChunkReport = {
            "index": chunk["index"],
            "topics": text_agent.generate(_specialist_prompt(
                "topic and evidence extractor", cleaned_chunk
            )),
            "decisions": text_agent.generate(_specialist_prompt(
                "decision extractor", cleaned_chunk
            )),
            "actions": text_agent.generate(_specialist_prompt(
                "action-item extractor", cleaned_chunk
            )),
        }
        return {"chunk_reports": [report]}

    def consolidate_reports(state: MeetingWorkflowState):
        reports = sorted(state["chunk_reports"], key=lambda report: report["index"])
        current_reports = [format_chunk_report(report) for report in reports]
        while len(current_reports) > 1:
            next_reports = []
            for offset in range(0, len(current_reports), config.reduce_group_size):
                group = current_reports[offset:offset + config.reduce_group_size]
                prompt = (
                    "Merge these adjacent meeting reports into one concise factual "
                    "evidence report. Preserve chronology, decisions, unresolved "
                    "issues, owners, and action items. Do not write a polished final "
                    "summary and do not invent missing information.\n\n"
                    + "\n\n".join(group)
                )
                next_reports.append(text_agent.generate(prompt))
            current_reports = next_reports
        return {"consolidated_evidence": current_reports[0]}

    def create_draft(state: MeetingWorkflowState):
        receiver_prompt = (
            "Write a faithful, concise meeting summary from the received "
            "specialist evidence. Include decisions, open issues, and action "
            "items only when they are supported by the evidence."
        )
        if config.communication_mode == "cipher":
            draft = cipher_agent.generate(
                state["consolidated_evidence"], receiver_prompt
            )
        else:
            draft = text_agent.generate(
                f"{receiver_prompt}\n\n{state['consolidated_evidence']}"
            )
        return {"draft_summary": draft, "refinement_round": 0}

    def critique_draft(state: MeetingWorkflowState):
        prompt = (
            "Check the draft meeting summary against the evidence below. "
            "Reply with either APPROVED or a short list of concrete factual "
            "corrections for omissions, hallucinations, or unsupported claims.\n\n"
            f"Evidence:\n{state['consolidated_evidence']}\n\n"
            f"Draft:\n{state['draft_summary']}"
        )
        critique = text_agent.generate(prompt)
        return {
            "critique": critique,
            "needs_refinement": "APPROVED" not in critique.upper(),
        }

    def refine_draft(state: MeetingWorkflowState):
        prompt = (
            "Revise the draft meeting summary using the critic feedback. Keep "
            "only supported facts, be concise, and return only the final summary.\n\n"
            f"Evidence:\n{state['consolidated_evidence']}\n\n"
            f"Draft:\n{state['draft_summary']}\n\nCritique:\n{state['critique']}"
        )
        return {
            "final_summary": text_agent.generate(prompt),
            "refinement_round": state["refinement_round"] + 1,
        }

    def accept_draft(state: MeetingWorkflowState):
        return {"final_summary": state["draft_summary"]}

    def route_after_critique(state: MeetingWorkflowState):
        if (
            state["needs_refinement"]
            and state["refinement_round"] < config.max_refinement_rounds
        ):
            return "refine"
        return "accept"

    graph = StateGraph(MeetingWorkflowState)
    graph.add_node("segment_transcript", segment_transcript)
    graph.add_node("process_chunk", process_chunk)
    graph.add_node("consolidate_reports", consolidate_reports)
    graph.add_node("create_draft", create_draft)
    graph.add_node("critique_draft", critique_draft)
    graph.add_node("refine_draft", refine_draft)
    graph.add_node("accept_draft", accept_draft)

    graph.add_edge(START, "segment_transcript")
    graph.add_conditional_edges("segment_transcript", send_chunks, ["process_chunk"])
    graph.add_edge("process_chunk", "consolidate_reports")
    graph.add_edge("consolidate_reports", "create_draft")
    graph.add_edge("create_draft", "critique_draft")
    graph.add_conditional_edges(
        "critique_draft",
        route_after_critique,
        {"refine": "refine_draft", "accept": "accept_draft"},
    )
    graph.add_edge("refine_draft", END)
    graph.add_edge("accept_draft", END)
    return graph.compile()
