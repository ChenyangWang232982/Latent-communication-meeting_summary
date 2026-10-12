"""Run exact cross-agent Prefix KV reuse with StateBridge aggregation.

One shared source prefix is prefetched once. Specialist Agents append their
role-specific tasks after a cloned KV cache, so the long source is not encoded
again for every Agent. This is a standard exact-prefix cache baseline, not
ProphetKV's selective recomputation or cross-chunk cache fusion.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from statebridge_qasper.bridge import StateBridge
from statebridge_qasper.run import (
    chat_prompt,
    decode,
    generate_from_ids,
    generate_from_prefix,
    load_records,
    remove_thinking_tokens,
    trim_context,
)

from .prefix_cache import cache_nbytes, generate_after_prefix_cache, prefill_prefix, prefill_prefix_streaming
from .run import ROLE_INSTRUCTIONS
from .selection import chunk_document, rank_chunks
from .state_selection import STATE_SELECTION_STRATEGIES, select_state_tokens


MEETING_ROLE_CONTRACTS = {
    "facts": (
        "List only high-confidence context needed for the minutes. Omit introductions, icebreakers, personal "
        "details, meeting-start acknowledgements, and speculative design ideas. Do not infer a decision, action, owner, "
        "or deadline from a job title."
    ),
    "decisions": (
        "List only decisions explicitly accepted, agreed, chosen, approved, rejected, or committed to in the transcript. "
        "A discussion topic, observation, problem, question, or suggestion is NOT a decision. If no such statement exists, "
        "output None explicitly stated. The supporting quote must itself contain a commitment cue such as 'decided', "
        "'agreed', 'approved', 'chosen', or 'we have settled on'. A project brief or a statement of planned work is not "
        "automatically a decision made in this meeting."
    ),
    "actions": (
        "List only explicit commitments or assigned follow-ups. An action needs an explicit task and a commitment or assignment "
        "such as 'will', 'assigned', 'responsible', 'need to', or a stated deadline. Include an owner or deadline only when the "
        "transcript explicitly states one. The supporting quote must contain both the task and its commitment or assignment; "
        "do not infer work from a person's role or from vague phrases such as 'you know what to do'."
    ),
    "risks": (
        "List only explicitly stated risks, blockers, constraints, uncertainties, disagreements, or unresolved questions. "
        "A normal feature discussion is not a risk. Do not invent a risk from ordinary project context."
    ),
    "discussion": (
        "List agenda items, proposals, alternatives, brainstorming, and ordinary discussion topics that were raised but not "
        "settled. Do not call an item a decision merely because the group said it needs discussion. Do not turn a suggestion "
        "into an action item."
    ),
    "questions": (
        "List only explicit questions that need an answer, confirmation, investigation, or later resolution. Preserve the "
        "uncertainty in the claim. A rhetorical question or a question already answered in the same quoted exchange is not a "
        "pending question."
    ),
    "next_steps": (
        "List only explicit meeting logistics, sequencing, stage transitions, or follow-up arrangements, such as when the "
        "group will reconvene or what process stage comes next. A task with an explicit owner or commitment belongs in actions, "
        "not next_steps. A vague intention belongs in discussion."
    ),
}

QASPER_DEFAULT_ROLES = ["facts", "decisions", "actions", "risks"]
MEETING_DEFAULT_ROLES = ["facts", "decisions", "actions", "discussion", "risks", "questions", "next_steps"]
MEETING_SECTIONS = (
    "Key facts\nConfirmed decisions\nConfirmed action items\nDiscussion topics\n"
    "Risks and open questions\nPending questions\nNext steps"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--data", type=Path, help="QASPER-style JSONL input")
    input_source.add_argument(
        "--meeting-data",
        type=Path,
        help="Meeting JSONL input with context, question, and optional reference answer; enables meeting roles and receiver.",
    )
    input_source.add_argument("--transcript", type=Path, help="One UTF-8 meeting transcript")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--num-samples", type=int, default=1, help="0 means all records")
    parser.add_argument(
        "--meeting-question",
        default="Create an evidence-grounded meeting summary covering the most important facts.",
        help="Shared objective used when --transcript is supplied.",
    )
    parser.add_argument("--source-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=0,
        help="Append the shared prefix in chronological chunks; 0 performs one full prefill.",
    )
    parser.add_argument(
        "--backfill-rounds",
        type=int,
        default=0,
        help="Fixed number of query-aware local evidence review passes per role.",
    )
    parser.add_argument("--backfill-chunk-tokens", type=int, default=256)
    parser.add_argument(
        "--backfill-neighbor-chunks",
        type=int,
        default=1,
        help="Chronological neighbors included around each deterministically selected review chunk.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=sorted(ROLE_INSTRUCTIONS),
        default=None,
        help="Specialists to run. Defaults to seven meeting roles for --transcript and four research-QA roles for --data.",
    )
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=192)
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=64)
    parser.add_argument(
        "--state-selection-strategy",
        "--strategy",
        choices=STATE_SELECTION_STRATEGIES,
        default="tail",
        help="Choose which generated handoff states occupy each fixed-size StateBridge prefix.",
    )
    parser.add_argument("--snap-ratio", type=float, default=0.3)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--communication-metrics-only",
        action="store_true",
        help="Generate specialist handoffs for a real communication count, but write only token/state metrics and skip final answers.",
    )
    parser.add_argument("--variants", nargs="+", choices=["statebridge", "text", "no_comm"], default=["statebridge", "text", "no_comm"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def specialist_user(
    source: str,
    record: dict[str, Any],
    role: str,
    *,
    meeting_mode: bool,
    previous_handoff: str | None = None,
    review_packet: str | None = None,
    review_round: int | None = None,
    compact_token_budget: int | None = None,
) -> str:
    task = ROLE_INSTRUCTIONS[role]
    format_instruction = ""
    if meeting_mode:
        task = MEETING_ROLE_CONTRACTS[role]
        if compact_token_budget is None:
            format_instruction = (
                " Return at most three short bullets. Every bullet must use exactly this format: "
                "- [timestamp] \"verbatim transcript quotation\" -> claim. The quotation must be sufficient on its own to support "
                "the claim. A timestamp alone is not evidence. If no supported item exists, output exactly: None explicitly stated."
            )
        else:
            format_instruction = (
                " Inside each priority marker, use an evidence-backed bullet in the format "
                "- [timestamp] fact -> claim, preserving names, numbers, negation, uncertainty, owners, and deadlines. "
                "If no supported item exists, mark exactly: <P2>None explicitly stated.</P2>"
            )
    compact_instruction = ""
    if compact_token_budget is not None:
        compact_instruction = (
            " Produce a detailed candidate handoff and assign every item exactly one inline priority marker: "
            "<P2>essential explicit fact</P2>, <P1>useful supporting detail</P1>, or "
            "<P0>redundant, speculative, or outside this role</P0>. Do not place factual content outside these markers. "
            f"The communication layer will retain at most {compact_token_budget} marked content states, preferring P2, "
            "then P1, then P0."
        )
    user = (
        f"Source transcript:\n{source}\n\nQuestion: {record['question']}\n\n"
        f"Task: {task} Give a concise factual handoff for another agent. "
        "Only use the transcript. Include specific facts and state missing information rather than guessing."
        f"{compact_instruction}{format_instruction}"
    )
    if review_packet is not None and previous_handoff is not None and review_round is not None:
        user += (
            f"\n\nFixed review pass {review_round}: reconsider the previous handoff using the following chronological "
            "evidence packet. Correct over-classification and remove unsupported claims.\n\n"
            f"Previous handoff:\n{previous_handoff}\n\nEvidence packet:\n{review_packet}"
        )
    return user


def specialist_prompt(
    tokenizer,
    source: str,
    record: dict[str, Any],
    role: str,
    enable_thinking: bool,
    *,
    meeting_mode: bool,
    previous_handoff: str | None = None,
    review_packet: str | None = None,
    review_round: int | None = None,
    compact_token_budget: int | None = None,
) -> str:
    """Build a normal Qwen chat prompt; cache only its token-identical prefix."""
    user = specialist_user(
        source,
        record,
        role,
        meeting_mode=meeting_mode,
        previous_handoff=previous_handoff,
        review_packet=review_packet,
        review_round=review_round,
        compact_token_budget=compact_token_budget,
    )
    return chat_prompt(
        tokenizer,
        "You are a careful meeting and research-document analyst.",
        user,
        enable_thinking,
    )


def compact_specialist_prompt(
    tokenizer,
    source: str,
    record: dict[str, Any],
    role: str,
    candidate_handoff: str,
    enable_thinking: bool,
    *,
    meeting_mode: bool,
    token_budget: int,
) -> str:
    """Ask the sender to turn a candidate into one continuous latent message."""
    task = MEETING_ROLE_CONTRACTS[role] if meeting_mode else ROLE_INSTRUCTIONS[role]
    user = (
        f"Source transcript:\n{source}\n\nQuestion: {record['question']}\n\n"
        f"Task: {task} Dynamic compact pass. Internally rank every candidate item as: "
        "P2 = essential explicit fact for this role, P1 = useful supporting detail, "
        "P0 = redundant, speculative, or outside this role. Rewrite only P2 items, then "
        "P1 items if space remains. Preserve names, numbers, negation, uncertainty, owners, "
        "deadlines, and evidence wording when present. The source may only be used to verify the "
        "candidate; do not introduce an item absent from the candidate. Do not output priority "
        "labels or analysis. "
        f"Return one concise continuous handoff of at most {token_budget} tokens.\n\n"
        f"Candidate handoff:\n{candidate_handoff}"
    )
    return chat_prompt(
        tokenizer,
        "You are a careful meeting and research-document analyst.",
        user,
        enable_thinking,
    )


def common_prefix_length(token_batches: list[torch.Tensor]) -> int:
    """Find the token-identical prefix shared by every specialist prompt."""
    if not token_batches:
        raise ValueError("At least one specialist prompt is required")
    sequences = [tokens[0] for tokens in token_batches]
    limit = min(sequence.numel() for sequence in sequences)
    for index in range(limit):
        value = sequences[0][index]
        if any(sequence[index] != value for sequence in sequences[1:]):
            return index
    return limit


def review_packet(ranked_chunks, round_index: int, neighbors: int) -> str:
    """Select a deterministic ranked chunk and its chronological neighbors."""
    if not ranked_chunks:
        return "No additional evidence is available."
    selected, _ = ranked_chunks[round_index % len(ranked_chunks)]
    all_chunks = {chunk.chunk_id: chunk for chunk, _ in ranked_chunks}
    packet = []
    for chunk_id in range(max(0, selected.chunk_id - neighbors), selected.chunk_id + neighbors + 1):
        chunk = all_chunks.get(chunk_id)
        if chunk is not None:
            packet.append(f"[review chunk {chunk.chunk_id}]\n{chunk.text}")
    return "\n\n".join(packet)


@torch.inference_mode()
def generate_specialist_from_cache(
    model,
    tokenizer,
    prefix_cache,
    prefix_length,
    suffix_ids: torch.Tensor,
    *,
    max_new_tokens: int,
):
    message_ids, states = generate_after_prefix_cache(
        model,
        tokenizer,
        prefix_cache,
        prefix_length,
        suffix_ids,
        max_new_tokens=max_new_tokens,
    )
    message_ids, states = remove_thinking_tokens(tokenizer, message_ids, states)
    return decode(tokenizer, message_ids), states, message_ids[0]


def find_token_marker(
    sequence: list[int],
    patterns: list[list[int]],
    start: int = 0,
    *,
    last: bool = False,
) -> tuple[int, int] | None:
    """Return a matching marker position and token length."""
    matches = []
    for pattern in patterns:
        if not pattern:
            continue
        for index in range(start, len(sequence) - len(pattern) + 1):
            if sequence[index : index + len(pattern)] == pattern:
                matches.append((index, len(pattern)))
                if not last:
                    break
    if not matches:
        return None
    return max(matches) if last else min(matches)


def marker_patterns(tokenizer, marker: str) -> list[list[int]]:
    """Cover the whitespace contexts commonly emitted before an exact marker."""
    return [
        tokenizer.encode(text, add_special_tokens=False)
        for text in (marker, f"\n{marker}", f" {marker}")
    ]


def priority_spans(tokenizer, sequence: list[int], priority: int) -> list[tuple[int, int]]:
    """Locate content spans enclosed by one priority marker."""
    opening_patterns = marker_patterns(tokenizer, f"<P{priority}>")
    closing_patterns = marker_patterns(tokenizer, f"</P{priority}>")
    spans = []
    cursor = 0
    while cursor < len(sequence):
        opening = find_token_marker(sequence, opening_patterns, start=cursor)
        if opening is None:
            break
        content_start = opening[0] + opening[1]
        closing = find_token_marker(sequence, closing_patterns, start=content_start)
        if closing is None:
            break
        content_end = closing[0]
        while content_start < content_end and not tokenizer.decode([sequence[content_start]]).strip():
            content_start += 1
        while content_end > content_start and not tokenizer.decode([sequence[content_end - 1]]).strip():
            content_end -= 1
        if content_end > content_start:
            spans.append((content_start, content_end))
        cursor = closing[0] + closing[1]
    return spans


def extract_prioritized_handoff(tokenizer, states, token_ids, budget: int):
    """Select up to K original generated states using model-emitted priorities."""
    sequence = token_ids.tolist()
    spans_by_priority = {
        priority: priority_spans(tokenizer, sequence, priority)
        for priority in (2, 1, 0)
    }
    selected_indices = []
    selected_counts = {"P2": 0, "P1": 0, "P0": 0}
    for priority in (2, 1, 0):
        for start, end in spans_by_priority[priority]:
            remaining = budget - len(selected_indices)
            if remaining <= 0:
                break
            chosen = list(range(start, min(end, start + remaining)))
            selected_indices.extend(chosen)
            selected_counts[f"P{priority}"] += len(chosen)
        if len(selected_indices) >= budget:
            break

    if not selected_indices:
        selected_states, selected_ids = select_state_tokens(states, token_ids, budget, "tail")
        return decode(tokenizer, selected_ids.unsqueeze(0)), selected_states, selected_ids, False, selected_counts

    # Priority determines admission; chronological ordering remains intact for the receiver.
    selected_indices = sorted(selected_indices)
    index_groups = [[selected_indices[0]]]
    for index in selected_indices[1:]:
        if index == index_groups[-1][-1] + 1:
            index_groups[-1].append(index)
        else:
            index_groups.append([index])
    selected_message = "\n".join(
        tokenizer.decode([sequence[index] for index in group], skip_special_tokens=True).strip()
        for group in index_groups
    ).strip()
    index_tensor = torch.tensor(selected_indices, device=states.device, dtype=torch.long)
    selected_states = states.index_select(0, index_tensor)
    selected_ids = token_ids.index_select(0, index_tensor)
    return selected_message, selected_states, selected_ids, True, selected_counts


@torch.inference_mode()
def specialist_from_cache(
    model,
    tokenizer,
    prefix_cache,
    encoded_prefix: torch.Tensor,
    full_prompt_ids: torch.Tensor,
    args,
    *,
    source: str,
    record: dict[str, Any],
    role: str,
    meeting_mode: bool,
):
    prefix_length = encoded_prefix.size(1)
    single_pass_compact = args.state_selection_strategy == "dynamic_compact"
    message, states, message_ids = generate_specialist_from_cache(
        model,
        tokenizer,
        prefix_cache,
        prefix_length,
        full_prompt_ids[:, prefix_length:],
        max_new_tokens=args.agent_max_new_tokens,
    )
    if single_pass_compact:
        selected_message, selected_states, selected_ids, marker_found, priority_counts = extract_prioritized_handoff(
            tokenizer,
            states,
            message_ids,
            args.prefix_tokens_per_agent,
        )
        if not marker_found:
            print(f"warning: {role} did not produce usable priority markers; using tail fallback.", flush=True)
        return selected_message, selected_states, selected_ids, {
            "mode": "priority_markers" if marker_found else "tail_fallback",
            "generated_tokens": int(message_ids.size(0)),
            "selected_states": int(selected_states.size(0)),
            "priority_state_counts": priority_counts,
        }

    if args.state_selection_strategy == "dynamic_compact_two_pass":
        compact_ids = tokenizer(
            compact_specialist_prompt(
                tokenizer,
                source,
                record,
                role,
                message,
                args.enable_thinking,
                meeting_mode=meeting_mode,
                token_budget=args.prefix_tokens_per_agent,
            ),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(args.device).input_ids
        if not torch.equal(compact_ids[:, :prefix_length], encoded_prefix):
            raise RuntimeError("Dynamic compact prompt no longer shares the cached transcript prefix.")
        compact_message, compact_states, compact_token_ids = generate_specialist_from_cache(
            model,
            tokenizer,
            prefix_cache,
            prefix_length,
            compact_ids[:, prefix_length:],
            max_new_tokens=args.prefix_tokens_per_agent,
        )
        if compact_states.size(0) == 0:
            raise RuntimeError("Two-pass dynamic compact selection produced no usable handoff states.")
        if compact_states.size(0) > args.prefix_tokens_per_agent:
            raise RuntimeError("Two-pass dynamic compact selection exceeded the configured state budget.")
        return compact_message, compact_states, compact_token_ids, {
            "mode": "two_pass_compact",
            "generated_tokens": int(message_ids.size(0) + compact_token_ids.size(0)),
            "selected_states": int(compact_states.size(0)),
        }

    selected_states, selected_ids = select_state_tokens(
        states,
        message_ids,
        args.prefix_tokens_per_agent,
        args.state_selection_strategy,
    )
    return message, selected_states, selected_ids, {
        "mode": args.state_selection_strategy,
        "generated_tokens": int(message_ids.size(0)),
        "selected_states": int(selected_states.size(0)),
    }


def receiver_prompt(tokenizer, question: str, enable_thinking: bool, reports: str | None = None, *, meeting_mode: bool = False) -> str:
    if meeting_mode:
        user = (
            f"Objective: {question}\n\nCreate a concise meeting summary with exactly these sections:\n"
            f"{MEETING_SECTIONS}\n"
            "Keep each item in its proper section: agenda items and proposals are not decisions; unassigned process "
            "arrangements are not action items. Under each section, write 'None explicitly stated.' when the specialist "
            "findings provide no supported item. Only name an owner or deadline when explicitly supported by the specialist findings."
        )
    else:
        user = f"Question: {question}\n\nGive a concise factual answer."
    if reports is not None:
        user += f"\n\nSpecialist handoffs:\n{reports}"
    return chat_prompt(
        tokenizer,
        "You aggregate specialist findings into an evidence-grounded response. Do not invent facts. "
        "Treat a suggestion or discussion topic as neither a decision nor an action unless the quoted evidence states a commitment.",
        user,
        enable_thinking,
    )


@torch.inference_mode()
def answer_variant(model, tokenizer, bridge, record, reports, states, token_ids, variant, args, *, meeting_mode: bool):
    if variant == "text":
        prompt = receiver_prompt(tokenizer, record["question"], args.enable_thinking, reports, meeting_mode=meeting_mode)
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        _, answer_ids = generate_from_ids(model, tokenizer, encoded.input_ids, encoded.attention_mask, args.receiver_max_new_tokens)
        return decode(tokenizer, answer_ids)
    prompt = receiver_prompt(tokenizer, record["question"], args.enable_thinking, meeting_mode=meeting_mode)
    encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
    if variant == "no_comm":
        _, answer_ids = generate_from_ids(model, tokenizer, encoded.input_ids, encoded.attention_mask, args.receiver_max_new_tokens)
        return decode(tokenizer, answer_ids)
    prefix = bridge.align(states, token_ids)
    return decode(tokenizer, generate_from_prefix(model, tokenizer, encoded.input_ids, prefix, args.receiver_max_new_tokens))


def main() -> None:
    args = parse_args()
    if args.prefix_tokens_per_agent < 2:
        raise ValueError("--prefix-tokens-per-agent must be at least 2")
    if (
        args.state_selection_strategy == "dynamic_compact"
        and args.agent_max_new_tokens <= args.prefix_tokens_per_agent
    ):
        raise ValueError(
            "Single-pass dynamic selection needs --agent-max-new-tokens greater than "
            "--prefix-tokens-per-agent so it can generate a long candidate before the marked handoff."
        )
    if args.backfill_rounds < 0:
        raise ValueError("--backfill-rounds cannot be negative")
    if args.backfill_neighbor_chunks < 0:
        raise ValueError("--backfill-neighbor-chunks cannot be negative")
    torch.manual_seed(args.seed)
    meeting_mode = args.transcript is not None or args.meeting_data is not None
    if args.roles is None:
        args.roles = MEETING_DEFAULT_ROLES if meeting_mode else QASPER_DEFAULT_ROLES
    if args.transcript is not None:
        if not args.transcript.is_file():
            raise FileNotFoundError(f"Transcript not found: {args.transcript}")
        transcript = args.transcript.read_text(encoding="utf-8").strip()
        if not transcript:
            raise ValueError(f"Transcript is empty: {args.transcript}")
        records = [{
            "id": args.transcript.stem,
            "context": transcript,
            "question": args.meeting_question,
            "answer": "",
        }]
    else:
        data_path = args.meeting_data or args.data
        assert data_path is not None
        records = load_records(data_path, args.num_samples)
    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    bridge = StateBridge(model.get_input_embeddings().weight, regularization=args.regularization, snap_ratio=args.snap_ratio, vocab_chunk_size=args.vocab_chunk_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as jsonl, args.output.open("w", encoding="utf-8") as report:
        run_kind = "Meeting" if meeting_mode else "QASPER"
        communication_totals = {"text_tokens": 0, "latent_states": 0}
        report.write(f"Cross-agent Prefix KV Cache + StateBridge {run_kind} run: {datetime.now().isoformat(timespec='seconds')}\n")
        prefill_mode = "single prefill" if args.prefill_chunk_tokens == 0 else f"streaming chunks of {args.prefill_chunk_tokens} tokens"
        report.write(
            f"model={args.model}, source_max_tokens={args.source_max_tokens}, roles={','.join(args.roles)}, "
            f"state_selection_strategy={args.state_selection_strategy}\n"
        )
        sender_passes = 2 if args.state_selection_strategy == "dynamic_compact_two_pass" else 1
        report.write(f"Sender generation passes per specialist round: {sender_passes}.\n")
        report.write(f"Each source prefix is prefetched once ({prefill_mode}) and reused by cloned per-agent KV caches.\n\n")
        for index, record in enumerate(records, start=1):
            source = trim_context(tokenizer, record["context"], args.source_max_tokens)
            source_chunks = chunk_document(
                tokenizer,
                source,
                chunk_tokens=args.backfill_chunk_tokens,
                overlap_tokens=0,
            )
            compact_token_budget = (
                args.prefix_tokens_per_agent if args.state_selection_strategy == "dynamic_compact" else None
            )
            role_inputs = [
                tokenizer(
                    specialist_prompt(
                        tokenizer,
                        source,
                        record,
                        role,
                        args.enable_thinking,
                        meeting_mode=meeting_mode,
                        compact_token_budget=compact_token_budget,
                    ),
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(args.device).input_ids
                for role in args.roles
            ]
            # A one-role quality check still needs a boundary before the
            # role-specific instruction. Add an unused alternate prompt only
            # to locate that exact common boundary.
            prefix_inputs = role_inputs
            if len(role_inputs) == 1:
                alternate_role = next(role for role in ROLE_INSTRUCTIONS if role not in args.roles)
                alternate_input = tokenizer(
                    specialist_prompt(
                        tokenizer,
                        source,
                        record,
                        alternate_role,
                        args.enable_thinking,
                        meeting_mode=meeting_mode,
                        compact_token_budget=compact_token_budget,
                    ),
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(args.device).input_ids
                prefix_inputs = [role_inputs[0], alternate_input]
            common_length = common_prefix_length(prefix_inputs)
            if common_length < 2 or any(input_ids.size(1) == common_length for input_ids in role_inputs):
                raise RuntimeError("Specialist prompts do not have a usable shared prefix and suffix.")
            encoded_prefix = role_inputs[0][:, :common_length]
            started = time.perf_counter()
            if args.prefill_chunk_tokens:
                prefix_cache, prefill_chunks = prefill_prefix_streaming(
                    model, encoded_prefix, chunk_tokens=args.prefill_chunk_tokens
                )
            else:
                prefix_cache = prefill_prefix(model, encoded_prefix)
                prefill_chunks = 1
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - started
            prefix_bytes = cache_nbytes(prefix_cache)

            specialists = []
            all_states, all_ids = [], []
            for role, full_prompt_ids in zip(args.roles, role_inputs):
                message, states, token_ids, selection = specialist_from_cache(
                    model,
                    tokenizer,
                    prefix_cache,
                    encoded_prefix,
                    full_prompt_ids,
                    args,
                    source=source,
                    record=record,
                    role=role,
                    meeting_mode=meeting_mode,
                )
                reviews = []
                ranking_query = f"{record['question']} {MEETING_ROLE_CONTRACTS.get(role, ROLE_INSTRUCTIONS[role])}"
                ranked = rank_chunks(source_chunks, ranking_query)
                for review_round in range(args.backfill_rounds):
                    packet = review_packet(ranked, review_round, args.backfill_neighbor_chunks)
                    review_ids = tokenizer(
                        specialist_prompt(
                            tokenizer,
                            source,
                            record,
                            role,
                            args.enable_thinking,
                            meeting_mode=meeting_mode,
                            previous_handoff=message,
                            review_packet=packet,
                            review_round=review_round + 1,
                            compact_token_budget=compact_token_budget,
                        ),
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(args.device).input_ids
                    if not torch.equal(review_ids[:, :common_length], encoded_prefix):
                        raise RuntimeError("Review prompt no longer shares the cached transcript prefix.")
                    message, states, token_ids, selection = specialist_from_cache(
                        model,
                        tokenizer,
                        prefix_cache,
                        encoded_prefix,
                        review_ids,
                        args,
                        source=source,
                        record=record,
                        role=role,
                        meeting_mode=meeting_mode,
                    )
                    reviews.append({"round": review_round + 1, "packet": packet})
                specialists.append({
                    "role": role,
                    "message": message,
                    "state_selection": selection,
                    "backfill_reviews": reviews,
                })
                all_states.append(states)
                all_ids.append(token_ids)
            reports = "\n\n".join(f"[{item['role']}]\n{item['message']}" for item in specialists)
            states, token_ids = torch.cat(all_states), torch.cat(all_ids)
            text_handoff_tokens = len(tokenizer(reports, add_special_tokens=False).input_ids)
            latent_states = int(states.size(0))
            communication_delta = latent_states - text_handoff_tokens
            communication_percent = 0.0 if text_handoff_tokens == 0 else communication_delta * 100.0 / text_handoff_tokens
            communication_totals["text_tokens"] += text_handoff_tokens
            communication_totals["latent_states"] += latent_states
            answers = {}
            if not args.communication_metrics_only:
                answers = {
                    variant: answer_variant(
                        model, tokenizer, bridge, record, reports, states, token_ids, variant, args, meeting_mode=meeting_mode
                    )
                    for variant in args.variants
                }
            result = {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record.get("answer", ""),
                "prefix_tokens": int(encoded_prefix.size(1)),
                "prefix_cache_bytes": prefix_bytes,
                "prefix_prefill_seconds": prefill_seconds,
                "prefix_prefill_chunks": prefill_chunks,
                "statebridge_tokens": latent_states,
                "state_selection_strategy": args.state_selection_strategy,
                "sender_generation_passes_per_specialist_round": sender_passes,
                "communication": {
                    "text_handoff_tokens": text_handoff_tokens,
                    "latent_state_tokens": latent_states,
                    "latent_minus_text_tokens": communication_delta,
                    "latent_minus_text_percent": communication_percent,
                },
                "specialists": [] if args.communication_metrics_only else specialists,
                "backfill_rounds": args.backfill_rounds,
                "answers": answers,
            }
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            report.write("=" * 80 + f"\nSample {index}\nId: {record['id']}\n\nQuestion:\n{record['question']}\n\n")
            report.write(
                f"Shared prefix: {result['prefix_tokens']} tokens in {prefill_chunks} chronological cache updates; "
                f"KV cache: {prefix_bytes / 2**20:.1f} MiB; prefill: {prefill_seconds:.2f}s\n"
            )
            report.write(f"StateBridge specialist prefix: {states.size(0)} tokens\n\n")
            report.write(
                "Communication cost (specialists -> receiver): "
                f"text={text_handoff_tokens} tokens; latent={latent_states} states; "
                f"latent-text={communication_delta:+d} tokens ({communication_percent:+.1f}%). "
                "Negative means latent saved communication units; positive means latent used more.\n\n"
            )
            if not args.communication_metrics_only:
                for item in specialists:
                    selection = item["state_selection"]
                    report.write(
                        f"{item['role']} selection: {selection['mode']}; "
                        f"generated={selection['generated_tokens']} tokens; "
                        f"selected={selection['selected_states']} states\n"
                    )
                    if "priority_state_counts" in selection:
                        counts = selection["priority_state_counts"]
                        report.write(
                            f"{item['role']} selected priorities: "
                            f"P2={counts['P2']}, P1={counts['P1']}, P0={counts['P0']}\n"
                        )
                    report.write(f"{item['role']} handoff (debug only):\n{item['message']}\n\n")
                    for review in item["backfill_reviews"]:
                        report.write(f"{item['role']} fixed backfill pass {review['round']} evidence:\n{review['packet']}\n\n")
                for variant, answer in answers.items():
                    report.write(f"{variant} answer:\n{answer}\n\n")
                if not meeting_mode:
                    report.write(f"Gold answer:\n{record.get('answer', '')}\n\n")
            report.flush()
            print(f"completed {index}/{len(records)}: {record['id']}", flush=True)
        total_text = communication_totals["text_tokens"]
        total_latent = communication_totals["latent_states"]
        total_delta = total_latent - total_text
        total_percent = 0.0 if total_text == 0 else total_delta * 100.0 / total_text
        report.write("=" * 80 + "\nCommunication summary\n")
        report.write(
            f"samples={len(records)}; text={total_text} tokens; latent={total_latent} states; "
            f"latent-text={total_delta:+d} tokens ({total_percent:+.1f}%). "
            "Negative means latent saved communication units; positive means latent used more.\n"
        )


if __name__ == "__main__":
    main()
