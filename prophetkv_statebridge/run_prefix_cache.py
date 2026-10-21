"""Run exact cross-agent Prefix KV reuse with StateBridge aggregation.

One shared source prefix is prefetched once. Specialist Agents append their
role-specific tasks after a cloned KV cache, so the long source is not encoded
again for every Agent. This is a standard exact-prefix cache baseline, not
ProphetKV's selective recomputation or cross-chunk cache fusion.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

from .prefix_cache import (
    cache_nbytes,
    generate_after_prefix_cache,
    prefill_prefix,
)
from .evidence import (
    apply_retrieval_verdicts,
    claim_packet,
    parse_grounded_claims,
    parse_retrieval_claims,
    parse_verdicts,
    retrieved_claim_packet,
    verified_handoff,
)
from .run import ROLE_INSTRUCTIONS


MEETING_ROLE_CONTRACTS = {
    "decisions": (
        "List only decisions explicitly accepted, agreed, chosen, approved, rejected, or committed to in the transcript. "
        "Read later turns before deciding whether an earlier proposal was accepted, reversed, or left open. "
        "A discussion topic, observation, problem, question, or suggestion is NOT a decision. If no such statement exists, "
        "output None explicitly stated. A project brief or a statement of planned work is not automatically a decision "
        "made in this meeting."
    ),
    "actions": (
        "List only explicit commitments or assigned follow-ups. An action needs an explicit task and a commitment or assignment "
        "such as 'will', 'assigned', 'responsible', 'need to', or a stated deadline. Include an owner or deadline only when the "
        "transcript explicitly states one. Do not infer work from a person's role or from vague phrases such as "
        "'you know what to do'."
    ),
    "context": (
        "List only important background, constraints, discussion topics, proposals, and alternatives needed to understand the "
        "meeting. Exclude accepted decisions, assigned actions, and unresolved questions or risks; those belong to other roles. "
        "Omit introductions, icebreakers, and personal details."
    ),
    "open_issues": (
        "List only unresolved questions, risks, blockers, constraints, disagreements, or uncertainties that still need a "
        "decision, answer, or follow-up at the end of the meeting. Check later turns for resolution before listing an issue. "
        "Do not include a resolved decision or an assigned action."
    ),
}

QASPER_DEFAULT_ROLES = ["facts", "decisions", "actions", "risks"]
MEETING_DEFAULT_ROLES = ["decisions", "actions", "open_issues", "context"]
MEETING_SECTIONS = (
    "Key context and discussion\nConfirmed decisions\nAction items\nOpen issues and risks"
)
MEETING_STATUS_LABELS = {
    "decisions": "CONFIRMED_DECISION",
    "actions": "COMMITTED_ACTION",
    "open_issues": "UNRESOLVED",
    "context": "CONTEXT",
}
STATUS_ITEM_PATTERN = re.compile(
    r"(?m)^(?P<label>CONFIRMED_DECISION|COMMITTED_ACTION|UNRESOLVED|CONTEXT):"
)
STATUS_CONTROL_TEXT = {
    "CONFIRMED_DECISION": "[decision]",
    "COMMITTED_ACTION": "[action]",
    "UNRESOLVED": "[issue]",
    "CONTEXT": "[context]",
}


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
    parser.add_argument(
        "--source-max-tokens",
        type=int,
        default=4096,
        help="Maximum source tokens for the legacy research-QA path. Meeting transcripts are read as one source.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=sorted(ROLE_INSTRUCTIONS),
        default=None,
        help="Specialists to run. Defaults to four meeting roles for --transcript and four research-QA roles for --data.",
    )
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=64)
    parser.add_argument(
        "--handoff-chunk-tokens",
        type=int,
        default=None,
        help=(
            "Optional token window for each specialist handoff. Each window sends up to "
            "--prefix-tokens-per-agent latent states; omit to retain the legacy one-tail-per-specialist behavior."
        ),
    )
    parser.add_argument(
        "--max-items-per-specialist",
        type=int,
        default=None,
        help=(
            "Enable structured meeting handoffs with at most this many ranked status-labelled items per specialist. "
            "Each item contributes a contiguous trailing fraction of its hidden states."
        ),
    )
    parser.add_argument(
        "--item-state-ratio",
        type=float,
        default=0.5,
        help="Trailing fraction of each structured handoff item's hidden states to transmit (default: 0.5).",
    )
    parser.add_argument(
        "--specialist-max-new-tokens",
        type=int,
        default=None,
        help="Optional cap per specialist generation. Independent of the per-specialist StateBridge K.",
    )
    parser.add_argument(
        "--alignment-scope",
        choices=["joint", "per-message"],
        default="joint",
        help="Align all specialist states jointly (baseline) or align each sender message independently.",
    )
    parser.add_argument(
        "--evidence-policy",
        choices=["strict", "retrieval", "off"],
        default="strict",
        help="strict requires copied quotes; retrieval verifies retrieved evidence spans; off sends raw handoffs for an ablation.",
    )
    parser.add_argument("--snap-ratio", type=float, default=0.3)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--debug-progress",
        action="store_true",
        help="Print source-prefill, specialist, and final-answer progress immediately to the console.",
    )
    parser.add_argument(
        "--communication-metrics-only",
        action="store_true",
        help="Generate specialist handoffs for a real communication count, but write only token/state metrics and skip final answers.",
    )
    parser.add_argument("--variants", nargs="+", choices=["statebridge", "text", "no_comm"], default=["statebridge", "text", "no_comm"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def log_progress(args, message: str) -> None:
    if getattr(args, "debug_progress", False):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def specialist_user(
    source: str,
    record: dict[str, Any],
    role: str,
    *,
    meeting_mode: bool,
    evidence_policy: str,
    max_items: int | None = None,
) -> str:
    task = ROLE_INSTRUCTIONS[role]
    format_instruction = ""
    item_limit = (
        f" Return at most {max_items} items, ranked by importance. "
        "If fewer supported items exist, return fewer; never pad or invent items."
        if max_items is not None else ""
    )
    if meeting_mode:
        task = MEETING_ROLE_CONTRACTS[role]
    if meeting_mode and evidence_policy == "strict":
        format_instruction = (
            " Return ONLY a JSON array. Each object must have exactly these fields: "
            "category, claim, evidence_quote, timestamp, owner, deadline, confidence. category must be '"
            f"{role}'. evidence_quote must copy a contiguous, self-sufficient passage from the source transcript; "
            "do not paraphrase it, omit filler words, or combine distant turns. "
            "Use an empty string for unknown timestamp, owner, or deadline. If no supported item exists, return []."
            f"{item_limit}"
        )
    elif meeting_mode and evidence_policy == "retrieval":
        format_instruction = (
            " Return ONLY a JSON array. Each object must have exactly these fields: "
            "category, claim, evidence_keywords, timestamp, owner, deadline, confidence. category must be '"
            f"{role}'. evidence_keywords must be a JSON list of 2 to 6 distinctive words or short phrases from the "
            "transcript that can locate supporting evidence; do not copy a full quote. Use an empty list when no "
            "reliable anchor exists, and use an empty string for unknown timestamp, owner, or deadline. "
            "If no supported item exists, return []."
            f"{item_limit}"
        )
    elif meeting_mode:
        format_instruction = (
            f" Begin each item on a new line with {MEETING_STATUS_LABELS[role]}: "
            "Describe its final status at the end of the meeting. Return concise items for this role only. "
            "Do not write a full meeting summary, repeat another role's items, or add headings, quotations, "
            f"or unsupported details.{item_limit}"
        )
    user = (
        f"Source transcript:\n{source}\n\nQuestion: {record['question']}\n\n"
        f"Task: {task} Give a concise factual handoff for another agent. "
        f"Only use the transcript. Include specific facts and state missing information rather than guessing.{format_instruction}"
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
    evidence_policy: str,
    max_items: int | None = None,
) -> str:
    """Build a normal Qwen chat prompt; cache only its token-identical prefix."""
    user = specialist_user(
        source,
        record,
        role,
        meeting_mode=meeting_mode,
        evidence_policy=evidence_policy,
        max_items=max_items,
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
    if len(sequences) == 1:
        return limit
    baseline = sequences[0][:limit]
    matching = torch.ones(limit, dtype=torch.bool, device=baseline.device)
    for sequence in sequences[1:]:
        matching &= baseline.eq(sequence[:limit])
    mismatches = (~matching).nonzero(as_tuple=True)[0]
    return int(mismatches[0].item()) if mismatches.numel() else limit


def align_handoffs(bridge, segments: list[dict[str, Any]], scope: str) -> torch.Tensor:
    """Align sender messages without changing their order or state budget."""
    def with_control_prefix(aligned: torch.Tensor, segment: dict[str, Any]) -> torch.Tensor:
        control_ids = segment.get("control_token_ids")
        if control_ids is None or control_ids.numel() == 0:
            return aligned
        control = bridge.vocab_embeddings.index_select(0, control_ids.to(bridge.vocab_embeddings.device))
        return torch.cat((control.to(device=aligned.device, dtype=aligned.dtype), aligned))

    if scope == "per-message":
        return torch.cat([
            with_control_prefix(bridge.align(item["states"], item["token_ids"]), item)
            for item in segments
        ])
    if scope == "joint":
        aligned = bridge.align(
            torch.cat([item["states"] for item in segments]),
            torch.cat([item["token_ids"] for item in segments]),
        )
        offset = 0
        aligned_segments = []
        for item in segments:
            length = item["states"].size(0)
            aligned_segments.append(with_control_prefix(aligned[offset:offset + length], item))
            offset += length
        return torch.cat(aligned_segments)
    raise ValueError(f"Unsupported alignment scope: {scope}")


def split_handoff_states(
    states: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    state_budget: int,
    chunk_tokens: int | None,
) -> list[dict[str, torch.Tensor]]:
    """Retain a tail budget from each chronological handoff window.

    The default single window preserves the original StateBridge behavior. A
    configured window prevents a long specialist response from contributing
    only its final generated tokens.
    """
    if states.size(0) != token_ids.size(0):
        raise ValueError("Handoff states and token IDs must have the same length")
    window_size = chunk_tokens or token_ids.size(0)
    segments = []
    for start in range(0, token_ids.size(0), window_size):
        end = min(start + window_size, token_ids.size(0))
        count = min(state_budget, end - start)
        segments.append({
            "states": states[end - count:end],
            "token_ids": token_ids[end - count:end],
        })
    return segments


def status_item_ranges(message: str) -> list[tuple[int, int, str]]:
    """Return character spans for newline-delimited structured handoff items."""
    matches = list(STATUS_ITEM_PATTERN.finditer(message))
    return [
        (match.start(), matches[index + 1].start() if index + 1 < len(matches) else len(message), match.group("label"))
        for index, match in enumerate(matches)
    ]


def trailing_indices(indices: torch.Tensor, ratio: float) -> torch.Tensor:
    """Keep a contiguous trailing fraction of one semantic handoff item."""
    count = min(indices.numel(), max(1, math.ceil(indices.numel() * ratio)))
    return indices[-count:]


def generated_token_character_spans(tokenizer, token_ids: torch.Tensor) -> tuple[str, list[tuple[int, int]]]:
    """Map generated token positions to their decoded, trimmed handoff text.

    Retokenizing decoded Qwen text is not guaranteed to reproduce the exact
    generation IDs around whitespace and special tokens. Prefix decoding keeps
    the mapping tied to the IDs whose hidden states are being transferred.
    """
    sequence = token_ids.tolist()
    decoded = tokenizer.decode(sequence, skip_special_tokens=True)
    leading_whitespace = len(decoded) - len(decoded.lstrip())
    message = decoded.strip()
    spans = []
    previous_end = 0
    for index in range(1, len(sequence) + 1):
        prefix = tokenizer.decode(sequence[:index], skip_special_tokens=True)
        current_end = min(len(message), max(0, len(prefix) - leading_whitespace))
        spans.append((previous_end, current_end))
        previous_end = current_end
    return message, spans


def split_structured_handoff_states(
    tokenizer,
    states: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    ratio: float,
) -> list[dict[str, torch.Tensor]]:
    """Sample each status-labelled conclusion independently and uniformly.

    Prefix-decoded spans allow item boundaries in text to select matching,
    context-aware hidden states from the original specialist output. Each
    item keeps a contiguous trailing fraction, preserving the local latent
    trajectory expected by the receiver.
    """
    decoded_message, spans = generated_token_character_spans(tokenizer, token_ids)
    ranges = status_item_ranges(decoded_message)
    if not ranges:
        return [{
            "states": states.index_select(0, trailing_indices(torch.arange(states.size(0), device=states.device), ratio)),
            "token_ids": token_ids.index_select(0, trailing_indices(torch.arange(token_ids.size(0), device=token_ids.device), ratio)),
        }]

    segments = []
    for start, end, label in ranges:
        positions = torch.tensor(
            [index for index, (token_start, token_end) in enumerate(spans) if token_start < end and token_end > start],
            device=token_ids.device,
            dtype=torch.long,
        )
        if positions.numel() == 0:
            continue
        selected = trailing_indices(positions, ratio)
        control_ids = tokenizer(STATUS_CONTROL_TEXT[label], add_special_tokens=False)["input_ids"]
        segments.append({
            "states": states.index_select(0, selected),
            "token_ids": token_ids.index_select(0, selected),
            "control_token_ids": torch.tensor(control_ids, device=token_ids.device, dtype=torch.long),
        })
    if not segments:
        raise RuntimeError("Structured handoff labels did not map to generated tokens.")
    return segments


def segment_communication_tokens(segments: list[dict[str, Any]]) -> int:
    """Count aligned latent states plus explicit structured control tokens."""
    return sum(
        segment["states"].size(0) + int(segment.get("control_token_ids", torch.empty(0)).numel())
        for segment in segments
    )


@torch.inference_mode()
def specialist_from_cache(model, tokenizer, prefix_cache, prefix_length, suffix_ids: torch.Tensor, args, *, progress_callback=None):
    message_ids, states = generate_after_prefix_cache(
        model,
        tokenizer,
        prefix_cache,
        prefix_length,
        suffix_ids,
        max_context_tokens=args.model_context_tokens,
        max_new_tokens=args.specialist_max_new_tokens,
        progress_callback=progress_callback,
    )
    message_ids, states = remove_thinking_tokens(tokenizer, message_ids, states)
    return decode(tokenizer, message_ids), states, message_ids[0], int(message_ids.size(1))


@torch.inference_mode()
def states_for_handoff(model, tokenizer, handoff: str, args) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode verified text so StateBridge receives only grounded information."""
    encoded = tokenizer(handoff, return_tensors="pt", add_special_tokens=False).to(args.device)
    outputs = model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1][0], encoded.input_ids[0], int(encoded.input_ids.size(1))


@torch.inference_mode()
def semantically_supported_claims(model, tokenizer, claims: list[dict[str, Any]], args) -> tuple[list[dict[str, Any]], str]:
    """Use the same model to verify claim support from supplied evidence spans."""
    if not claims:
        return [], "[]"
    retrieved = "evidence_candidates" in claims[0]
    if retrieved:
        user = (
            "For every item below, determine whether one candidate span explicitly supports the claim and its category. "
            "For decisions, require an explicit accepted, agreed, chosen, approved, rejected, or committed outcome. "
            "For actions, require an explicit assignment, commitment, or deadline. Return ONLY a JSON array of objects "
            "with fields index, verdict, and candidate_index. verdict must be exactly supported, unsupported, or uncertain. "
            "candidate_index must name the supporting candidate when verdict is supported; otherwise use -1. "
            "Do not infer unstated owners, deadlines, decisions, or actions.\n\n"
            f"Items:\n{retrieved_claim_packet(claims)}"
        )
    else:
        user = (
            "For every item below, determine whether evidence_quote explicitly supports claim. "
            "Return ONLY a JSON array of objects with fields index and verdict. verdict must be exactly supported, "
            "unsupported, or uncertain. Do not infer unstated owners, deadlines, decisions, or actions.\n\n"
            f"Items:\n{claim_packet(claims)}"
        )
    prompt = chat_prompt(
        tokenizer,
        "You are a strict evidence verifier for meeting minutes.",
        user,
        args.enable_thinking,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
    _, verdict_ids = generate_from_ids(
        model,
        tokenizer,
        encoded.input_ids,
        encoded.attention_mask,
        available_generation_tokens(args, encoded.input_ids.size(1)),
    )
    verdict_message = decode(tokenizer, verdict_ids)
    if retrieved:
        return apply_retrieval_verdicts(verdict_message, claims), verdict_message
    supported = parse_verdicts(verdict_message, len(claims))
    return [claim for index, claim in enumerate(claims) if index in supported], verdict_message


@torch.inference_mode()
def process_source(model, tokenizer, record: dict[str, Any], source: str, args, *, meeting_mode: bool, progress_label: str = "") -> dict[str, Any]:
    """Analyze one source with one shared Prefix KV cache."""
    log_progress(args, f"{progress_label}: tokenizing {len(args.roles)} specialist prompts")
    role_inputs = [
        tokenizer(
            specialist_prompt(
                tokenizer, source, record, role, args.enable_thinking,
                meeting_mode=meeting_mode, evidence_policy=args.evidence_policy,
                max_items=args.max_items_per_specialist,
            ),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(args.device).input_ids
        for role in args.roles
    ]
    longest_prompt_tokens = max(input_ids.size(1) for input_ids in role_inputs)
    if longest_prompt_tokens >= args.model_context_tokens:
        raise RuntimeError(
            "The context is too long for the loaded model: the longest specialist prompt uses "
            f"{longest_prompt_tokens} tokens, but the model context limit is "
            f"{args.model_context_tokens} tokens. Use a shorter transcript or a model with a larger context window."
        )
    prefix_inputs = role_inputs
    if len(role_inputs) == 1:
        available_roles = MEETING_ROLE_CONTRACTS if meeting_mode else ROLE_INSTRUCTIONS
        alternate_role = next(role for role in available_roles if role not in args.roles)
        alternate_input = tokenizer(
            specialist_prompt(
                tokenizer, source, record, alternate_role, args.enable_thinking,
                meeting_mode=meeting_mode, evidence_policy=args.evidence_policy,
                max_items=args.max_items_per_specialist,
            ),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(args.device).input_ids
        prefix_inputs = [role_inputs[0], alternate_input]
    common_length = common_prefix_length(prefix_inputs)
    log_progress(args, f"{progress_label}: shared prefix comparison complete ({common_length} tokens)")
    if common_length < 2 or any(input_ids.size(1) == common_length for input_ids in role_inputs):
        raise RuntimeError("Specialist prompts do not have a usable shared prefix and suffix.")

    encoded_prefix = role_inputs[0][:, :common_length]
    log_progress(args, f"{progress_label}: prefill started ({encoded_prefix.size(1)} tokens)")
    started = time.perf_counter()
    prefix_cache = prefill_prefix(model, encoded_prefix)
    prefill_chunks = 1
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    prefix_bytes = cache_nbytes(prefix_cache)
    log_progress(args, f"{progress_label}: prefill complete ({prefill_seconds:.2f}s)")

    specialists, specialist_token_metrics, all_states, all_ids, segments = [], [], [], [], []
    for role, full_prompt_ids in zip(args.roles, role_inputs):
        completed = len(specialists)
        log_progress(
            args,
            f"{progress_label}: specialist {role} started "
            f"({completed}/{len(args.roles)} complete, {len(args.roles) - completed} remaining)",
        )
        specialist_started = time.perf_counter()
        raw_message, states, token_ids, generated_tokens = specialist_from_cache(
            model, tokenizer, prefix_cache, encoded_prefix.size(1), full_prompt_ids[:, common_length:], args,
            progress_callback=(
                lambda count: log_progress(args, f"{progress_label}: specialist {role} generated {count} tokens so far")
                if args.debug_progress else None
            ),
        )
        log_progress(args, f"{progress_label}: specialist {role} generated {generated_tokens} tokens; preparing handoff")
        verified_claims: list[dict[str, Any]] = []
        rejection_reasons: list[str] = []
        verifier_output = ""
        if meeting_mode and args.evidence_policy in {"strict", "retrieval"}:
            claim_parser = parse_grounded_claims if args.evidence_policy == "strict" else parse_retrieval_claims
            parsed_claims, rejection_reasons = claim_parser(
                raw_message, role=role, source=source, block_id=0
            )
            verified_claims, verifier_output = semantically_supported_claims(model, tokenizer, parsed_claims, args)
            if len(verified_claims) < len(parsed_claims):
                rejection_reasons.append("semantic verifier did not mark one or more claims as supported")
            if args.max_items_per_specialist is not None:
                verified_claims = verified_claims[:args.max_items_per_specialist]
            # Communication receives compact verified claims.  The longer
            # evidence-rich form stays available for audit but is not allowed
            # to crowd the fixed latent-state budget.
            status_label = MEETING_STATUS_LABELS[role] if args.max_items_per_specialist is not None else None
            message = verified_handoff(role, verified_claims, status_label=status_label)
            evidence_message = verified_handoff(role, verified_claims, include_evidence=True)
            states, token_ids, handoff_tokens = states_for_handoff(model, tokenizer, message, args)
        else:
            message = raw_message
            evidence_message = raw_message
            handoff_tokens = generated_tokens
        if args.max_items_per_specialist is not None:
            handoff_segments = split_structured_handoff_states(
                tokenizer,
                states,
                token_ids,
                ratio=args.item_state_ratio,
            )
        else:
            handoff_segments = split_handoff_states(
                states,
                token_ids,
                state_budget=args.prefix_tokens_per_agent,
                chunk_tokens=args.handoff_chunk_tokens,
            )
        selected_states = sum(segment["states"].size(0) for segment in handoff_segments)
        control_tokens = sum(int(segment.get("control_token_ids", torch.empty(0)).numel()) for segment in handoff_segments)
        specialists.append(
            {
                "role": role,
                "message": message,
                "evidence_message": evidence_message,
                "claims": verified_claims,
                "raw_message": raw_message,
                "rejection_reasons": rejection_reasons,
                "verifier_output": verifier_output,
            }
        )
        specialist_token_metrics.append({
            "role": role,
            "handoff_tokens": handoff_tokens,
            "generated_tokens": generated_tokens,
            "verified_claims": len(verified_claims),
            "selected_states": selected_states,
            "control_tokens": control_tokens,
            "communicated_tokens": selected_states + control_tokens,
            "truncated_tokens": max(0, handoff_tokens - selected_states),
            "handoff_windows": len(handoff_segments),
            "selection": "structured_items" if args.max_items_per_specialist is not None else "token_windows",
        })
        all_states.extend(segment["states"] for segment in handoff_segments)
        all_ids.extend(segment["token_ids"] for segment in handoff_segments)
        segments.extend(handoff_segments)
        completed = len(specialists)
        log_progress(
            args,
            f"{progress_label}: specialist {role} complete in {time.perf_counter() - specialist_started:.2f}s "
            f"({completed}/{len(args.roles)} complete, {len(args.roles) - completed} remaining)",
        )

    return {
        "source_tokens": len(tokenizer(source, add_special_tokens=False).input_ids),
        "prefix_tokens": int(encoded_prefix.size(1)),
        "prefix_cache_bytes": prefix_bytes,
        "prefix_prefill_seconds": prefill_seconds,
        "prefix_prefill_chunks": prefill_chunks,
        "specialists": specialists,
        "specialist_token_metrics": specialist_token_metrics,
        "states": torch.cat(all_states),
        "token_ids": torch.cat(all_ids),
        "segments": segments,
    }


def receiver_prompt(tokenizer, question: str, enable_thinking: bool, reports: str | None = None, *, meeting_mode: bool = False) -> str:
    if meeting_mode:
        user = (
            f"Objective: {question}\n\nCreate a concise meeting summary with exactly these sections:\n"
            f"{MEETING_SECTIONS}\n"
            "Status labels are authoritative routing signals. Copy every complete CONFIRMED_DECISION item into Confirmed decisions, "
            "and every complete COMMITTED_ACTION item into Action items. Copy UNRESOLVED items into Open issues and risks, and "
            "CONTEXT items into Key context and discussion. A contextual mention of an earlier proposal, uncertainty, or a future "
            "review must NOT cancel a labelled confirmed decision or action. Exclude a labelled decision only when a handoff explicitly "
            "states that the same decision was reversed or withdrawn. Avoid needless repetition, but preserve both a confirmed item "
            "and a genuinely separate unresolved follow-up when both are present. Under each section, write 'None explicitly stated.' "
            "only when no handoff contains an item routed to that section. For latent handoffs, [decision], [action], [issue], and "
            "[context] are the corresponding authoritative control labels. "
            "Only name an owner or deadline when the specialist handoff provides it."
        )
    else:
        user = f"Question: {question}\n\nGive a concise factual answer."
    if reports is not None:
        user += f"\n\nSpecialist handoffs:\n{reports}"
    return chat_prompt(
        tokenizer,
        "You aggregate specialist findings into an evidence-grounded response. Use only information conveyed by the specialist "
        "handoff; do not invent facts. When status labels are present, route each item by its label rather than re-classifying it. "
        "Treat an unlabelled suggestion or discussion topic as neither a decision nor an action unless it states a commitment.",
        user,
        enable_thinking,
    )


@torch.inference_mode()
def answer_variant(model, tokenizer, bridge, record, reports, states, token_ids, variant, args, *, meeting_mode: bool, segments=None):
    if variant == "text":
        prompt = receiver_prompt(tokenizer, record["question"], args.enable_thinking, reports, meeting_mode=meeting_mode)
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        _, answer_ids = generate_from_ids(
            model, tokenizer, encoded.input_ids, encoded.attention_mask,
            available_generation_tokens(args, encoded.input_ids.size(1)),
        )
        return decode(tokenizer, answer_ids)
    prompt = receiver_prompt(tokenizer, record["question"], args.enable_thinking, meeting_mode=meeting_mode)
    encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
    if variant == "no_comm":
        _, answer_ids = generate_from_ids(
            model, tokenizer, encoded.input_ids, encoded.attention_mask,
            available_generation_tokens(args, encoded.input_ids.size(1)),
        )
        return decode(tokenizer, answer_ids)
    prefix = (
        align_handoffs(bridge, segments, args.alignment_scope)
        if segments is not None
        else bridge.align(states, token_ids)
    )
    return decode(
        tokenizer,
        generate_from_prefix(
            model, tokenizer, encoded.input_ids, prefix,
            available_generation_tokens(args, encoded.input_ids.size(1) + prefix.size(0)),
        ),
    )


def model_context_tokens(model, tokenizer) -> int:
    """Return the usable context limit declared by the loaded model/tokenizer."""
    candidates = [getattr(model.config, "max_position_embeddings", None)]
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    # Some tokenizers use a huge sentinel instead of a real context limit.
    if isinstance(tokenizer_limit, int) and 32 <= tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)
    limits = [value for value in candidates if isinstance(value, int) and value >= 32]
    if not limits:
        raise ValueError("Could not determine the loaded model's maximum context length.")
    return min(limits)


def available_generation_tokens(args, occupied_tokens: int) -> int:
    remaining_tokens = args.model_context_tokens - occupied_tokens
    if remaining_tokens < 1:
        raise RuntimeError("The prompt fills the model context window; no output tokens remain.")
    return remaining_tokens


def main() -> None:
    args = parse_args()
    if args.prefix_tokens_per_agent < 2:
        raise ValueError("--prefix-tokens-per-agent must be at least 2")
    if args.handoff_chunk_tokens is not None and args.handoff_chunk_tokens < args.prefix_tokens_per_agent:
        raise ValueError("--handoff-chunk-tokens must be at least --prefix-tokens-per-agent")
    if args.max_items_per_specialist is not None and args.max_items_per_specialist < 1:
        raise ValueError("--max-items-per-specialist must be positive")
    if not 0 < args.item_state_ratio <= 1:
        raise ValueError("--item-state-ratio must be in (0, 1]")
    if args.max_items_per_specialist is not None and args.handoff_chunk_tokens is not None:
        raise ValueError("Use either --max-items-per-specialist or --handoff-chunk-tokens, not both")
    if args.specialist_max_new_tokens is not None and args.specialist_max_new_tokens < 1:
        raise ValueError("--specialist-max-new-tokens must be positive")
    torch.manual_seed(args.seed)
    meeting_mode = args.transcript is not None or args.meeting_data is not None
    if args.roles is None:
        args.roles = MEETING_DEFAULT_ROLES if meeting_mode else QASPER_DEFAULT_ROLES
    if meeting_mode and any(role not in MEETING_ROLE_CONTRACTS for role in args.roles):
        raise ValueError(f"Meeting roles must be chosen from: {', '.join(MEETING_ROLE_CONTRACTS)}")
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
    log_progress(args, f"startup: loaded {len(records)} record(s); checking device and dtype ({args.device})")
    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    log_progress(args, f"startup: loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_progress(args, f"startup: loading model weights ({dtype})")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    log_progress(args, f"startup: model weights loaded; transferring to {args.device}")
    model = model.to(args.device).eval()
    log_progress(args, "startup: model ready; initializing StateBridge")
    bridge = StateBridge(model.get_input_embeddings().weight, regularization=args.regularization, snap_ratio=args.snap_ratio, vocab_chunk_size=args.vocab_chunk_size)
    log_progress(args, "startup: StateBridge ready")
    args.model_context_tokens = model_context_tokens(model, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as jsonl, args.output.open("w", encoding="utf-8") as report:
        run_kind = "Meeting" if meeting_mode else "QASPER"
        communication_totals = {"text_tokens": 0, "latent_states": 0}
        report.write(f"Cross-agent Prefix KV Cache + StateBridge {run_kind} run: {datetime.now().isoformat(timespec='seconds')}\n")
        source_mode = "full meeting transcript" if meeting_mode else f"legacy first {args.source_max_tokens} source tokens"
        report.write(
            f"model={args.model}, source_mode={source_mode}, roles={','.join(args.roles)}, "
            f"evidence_policy={args.evidence_policy}, alignment_scope={args.alignment_scope}, "
            f"specialist_max_new_tokens={args.specialist_max_new_tokens}, "
            f"max_items_per_specialist={args.max_items_per_specialist}, "
            f"item_state_ratio={args.item_state_ratio}\n"
        )
        report.write(
            "The source prefix is prefetched once and reused by cloned per-agent KV caches.\n\n"
        )
        for index, record in enumerate(records, start=1):
            source = record["context"] if meeting_mode else trim_context(tokenizer, record["context"], args.source_max_tokens)
            if not source.strip():
                raise ValueError(f"Source is empty after tokenization: {record['id']}")
            log_progress(
                args,
                f"sample {index}/{len(records)} ({record['id']}): one source prefix, {len(args.roles)} specialist(s)",
            )
            source_result = process_source(
                model, tokenizer, record, source, args, meeting_mode=meeting_mode,
                progress_label=f"sample {index}/{len(records)}",
            )
            specialists = source_result["specialists"]
            specialist_token_metrics = source_result["specialist_token_metrics"]
            reports = "\n\n".join(f"[{item['role']}]\n{item['message']}" for item in specialists)
            states = source_result["states"]
            token_ids = source_result["token_ids"]
            segments = source_result["segments"]
            prefix_tokens = source_result["prefix_tokens"]
            prefix_bytes = source_result["prefix_cache_bytes"]
            prefill_seconds = source_result["prefix_prefill_seconds"]
            prefill_chunks = source_result["prefix_prefill_chunks"]
            text_handoff_tokens = len(tokenizer(reports, add_special_tokens=False).input_ids)
            latent_states = segment_communication_tokens(segments)
            total_text_tokens = text_handoff_tokens
            total_latent_states = latent_states
            communication_delta = total_latent_states - total_text_tokens
            communication_percent = 0.0 if total_text_tokens == 0 else communication_delta * 100.0 / total_text_tokens
            communication_totals["text_tokens"] += total_text_tokens
            communication_totals["latent_states"] += total_latent_states
            answers = {}
            if not args.communication_metrics_only:
                for variant in args.variants:
                    log_progress(args, f"sample {index}/{len(records)}: final {variant} answer started")
                    answers[variant] = answer_variant(
                        model, tokenizer, bridge, record, reports, states, token_ids, variant, args,
                        meeting_mode=meeting_mode, segments=segments
                    )
                    log_progress(args, f"sample {index}/{len(records)}: final {variant} answer complete")
            result = {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record.get("answer", ""),
                "gold_answers": record.get("answers") or [record.get("answer", "")],
                "prefix_tokens": prefix_tokens,
                "prefix_cache_bytes": prefix_bytes,
                "prefix_prefill_seconds": prefill_seconds,
                "prefix_prefill_chunks": prefill_chunks,
                "source_tokens": source_result["source_tokens"],
                "statebridge_tokens": total_latent_states,
                "alignment_scope": args.alignment_scope,
                "communication": {
                    "text_handoff_tokens": total_text_tokens,
                    "latent_state_tokens": total_latent_states,
                    "latent_minus_text_tokens": communication_delta,
                    "latent_minus_text_percent": communication_percent,
                },
                "specialist_token_metrics": specialist_token_metrics,
                "specialists": [] if args.communication_metrics_only else specialists,
                "answers": answers,
            }
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            report.write("=" * 80 + f"\nSample {index}\nId: {record['id']}\n\nQuestion:\n{record['question']}\n\n")
            report.write(
                f"Source: {result['source_tokens']} tokens; shared prefix: {result['prefix_tokens']} tokens; "
                f"one KV prefill: {prefix_bytes / 2**20:.1f} MiB; {prefill_seconds:.2f}s\n"
            )
            report.write(
                f"StateBridge specialist prefix: {latent_states} communicated states/tokens\n\n"
            )
            report.write(
                "Communication cost (specialists to receiver): "
                f"text={total_text_tokens} tokens; latent={total_latent_states} states; "
                f"latent-text={communication_delta:+d} tokens ({communication_percent:+.1f}%). "
                "Negative means latent saved communication units; positive means latent used more.\n\n"
            )
            if not args.communication_metrics_only:
                for item, token_metric in zip(specialists, specialist_token_metrics):
                    debug_message = item.get("evidence_message", item["message"])
                    report.write(f"{item['role']} handoff (debug only):\n{debug_message}\n\n")
                    report.write(
                        f"handoff tokens: {token_metric['handoff_tokens']}; "
                        f"selection: {token_metric['selection']}; "
                        f"handoff segments: {token_metric['handoff_windows']}; "
                        f"selected states: {token_metric['selected_states']}; "
                        f"control tokens: {token_metric['control_tokens']}; "
                        f"communicated: {token_metric['communicated_tokens']}; "
                        f"truncated tokens: {token_metric['truncated_tokens']}\n\n"
                    )
                    if meeting_mode:
                        report.write(
                            f"verified claims: {len(item['claims'])}; rejected: "
                            f"{'; '.join(item['rejection_reasons']) or 'none'}\n\n"
                        )
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
