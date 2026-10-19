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

from .prefix_cache import (
    cache_nbytes,
    generate_after_embedding_prefix,
    generate_after_prefix_cache,
    prefill_prefix,
)
from .evidence import claim_packet, parse_grounded_claims, parse_verdicts, verified_handoff
from .run import ROLE_INSTRUCTIONS
from .selection import Chunk, chunk_document


MEETING_ROLE_CONTRACTS = {
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
    "context": (
        "List only important background, constraints, discussion topics, proposals, and alternatives needed to understand the "
        "meeting. Exclude accepted decisions, assigned actions, and unresolved questions or risks; those belong to other roles. "
        "Omit introductions, icebreakers, and personal details."
    ),
    "open_issues": (
        "List only unresolved questions, risks, blockers, constraints, disagreements, or uncertainties that still need a "
        "decision, answer, or follow-up. Do not include a resolved decision or an assigned action."
    ),
}

QASPER_DEFAULT_ROLES = ["facts", "decisions", "actions", "risks"]
MEETING_DEFAULT_ROLES = ["decisions", "actions", "open_issues", "context"]
MEETING_SECTIONS = (
    "Key context and discussion\nConfirmed decisions\nAction items\nOpen issues and risks"
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
    parser.add_argument(
        "--source-max-tokens",
        type=int,
        default=4096,
        help="Maximum source tokens for the legacy research-QA path. Meeting inputs use automatic context-aware blocks.",
    )
    parser.add_argument(
        "--document-chunk-overlap-tokens",
        type=int,
        default=0,
        help="Token overlap between adjacent automatically sized meeting blocks.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=sorted(ROLE_INSTRUCTIONS),
        default=None,
        help="Specialists to run. Defaults to seven meeting roles for --transcript and four research-QA roles for --data.",
    )
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=64)
    parser.add_argument(
        "--evidence-policy",
        choices=["strict", "off"],
        default="strict",
        help="strict validates and filters meeting claims; off sends raw specialist handoffs for an ablation.",
    )
    parser.add_argument(
        "--block-summary-prefix-tokens",
        type=int,
        default=None,
        help="StateBridge K used when each block summary is sent to the final merger. Defaults to --prefix-tokens-per-agent.",
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
    evidence_policy: str,
) -> str:
    task = ROLE_INSTRUCTIONS[role]
    format_instruction = ""
    if meeting_mode:
        task = MEETING_ROLE_CONTRACTS[role]
    if meeting_mode and evidence_policy == "strict":
        format_instruction = (
            " Return ONLY a JSON array, with at most three objects. Each object must have exactly these fields: "
            "category, claim, evidence_quote, timestamp, owner, deadline, confidence. category must be '"
            f"{role}'. evidence_quote must copy a contiguous, self-sufficient passage from the source transcript; "
            "do not paraphrase it, omit filler words, or combine distant turns. "
            "Use an empty string for unknown timestamp, owner, or deadline. If no supported item exists, return []."
        )
    elif meeting_mode:
        format_instruction = (
            " Return concise items for this role only. Do not write a full meeting summary, repeat another role's items, "
            "or add headings, quotations, or unsupported details."
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
) -> str:
    """Build a normal Qwen chat prompt; cache only its token-identical prefix."""
    user = specialist_user(
        source,
        record,
        role,
        meeting_mode=meeting_mode,
        evidence_policy=evidence_policy,
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


@torch.inference_mode()
def specialist_from_cache(model, tokenizer, prefix_cache, prefix_length, suffix_ids: torch.Tensor, args):
    message_ids, states = generate_after_prefix_cache(
        model,
        tokenizer,
        prefix_cache,
        prefix_length,
        suffix_ids,
        max_context_tokens=args.model_context_tokens,
    )
    message_ids, states = remove_thinking_tokens(tokenizer, message_ids, states)
    count = min(args.prefix_tokens_per_agent, message_ids.size(1))
    return decode(tokenizer, message_ids), states[-count:], message_ids[0, -count:], int(message_ids.size(1))


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
    count = min(args.prefix_tokens_per_agent, encoded.input_ids.size(1))
    return outputs.hidden_states[-1][0, -count:], encoded.input_ids[0, -count:], int(encoded.input_ids.size(1))


@torch.inference_mode()
def semantically_supported_claims(model, tokenizer, claims: list[dict[str, Any]], args) -> tuple[list[dict[str, Any]], str]:
    """Use the same model to verify whether each located quote entails its claim."""
    if not claims:
        return [], "[]"
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
    supported = parse_verdicts(verdict_message, len(claims))
    return [claim for index, claim in enumerate(claims) if index in supported], verdict_message


@torch.inference_mode()
def process_source_block(model, tokenizer, record: dict[str, Any], source: str, block: Chunk, args, *, meeting_mode: bool) -> dict[str, Any]:
    """Analyze one bounded source block using one shared Prefix KV cache.

    A block cache is only reused by specialists reading that exact block.  We
    deliberately return latent handoffs rather than attempting to concatenate
    independent KV caches from different blocks.
    """
    role_inputs = [
        tokenizer(
            specialist_prompt(
                tokenizer, source, record, role, args.enable_thinking,
                meeting_mode=meeting_mode, evidence_policy=args.evidence_policy,
            ),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(args.device).input_ids
        for role in args.roles
    ]
    prefix_inputs = role_inputs
    if len(role_inputs) == 1:
        alternate_role = next(role for role in ROLE_INSTRUCTIONS if role not in args.roles)
        alternate_input = tokenizer(
            specialist_prompt(
                tokenizer, source, record, alternate_role, args.enable_thinking,
                meeting_mode=meeting_mode, evidence_policy=args.evidence_policy,
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
    prefix_cache = prefill_prefix(model, encoded_prefix)
    prefill_chunks = 1
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    prefix_bytes = cache_nbytes(prefix_cache)

    specialists, specialist_token_metrics, all_states, all_ids = [], [], [], []
    for role, full_prompt_ids in zip(args.roles, role_inputs):
        raw_message, states, token_ids, generated_tokens = specialist_from_cache(
            model, tokenizer, prefix_cache, encoded_prefix.size(1), full_prompt_ids[:, common_length:], args
        )
        verified_claims: list[dict[str, Any]] = []
        rejection_reasons: list[str] = []
        verifier_output = ""
        if meeting_mode and args.evidence_policy == "strict":
            parsed_claims, rejection_reasons = parse_grounded_claims(
                raw_message, role=role, source=source, block_id=block.chunk_id
            )
            verified_claims, verifier_output = semantically_supported_claims(model, tokenizer, parsed_claims, args)
            if len(verified_claims) < len(parsed_claims):
                rejection_reasons.append("semantic verifier did not mark one or more claims as supported")
            # Communication receives compact verified claims.  The longer
            # evidence-rich form stays available for audit but is not allowed
            # to crowd the fixed latent-state budget.
            message = verified_handoff(role, verified_claims)
            evidence_message = verified_handoff(role, verified_claims, include_evidence=True)
            states, token_ids, handoff_tokens = states_for_handoff(model, tokenizer, message, args)
        else:
            message = raw_message
            evidence_message = raw_message
            handoff_tokens = generated_tokens
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
            "block_id": block.chunk_id,
            "role": role,
            "handoff_tokens": handoff_tokens,
            "generated_tokens": generated_tokens,
            "verified_claims": len(verified_claims),
            "selected_states": int(states.size(0)),
        })
        all_states.append(states)
        all_ids.append(token_ids)

    return {
        "block_id": block.chunk_id,
        "source_tokens": block.token_count,
        "prefix_tokens": int(encoded_prefix.size(1)),
        "prefix_cache_bytes": prefix_bytes,
        "prefix_prefill_seconds": prefill_seconds,
        "prefix_prefill_chunks": prefill_chunks,
        "specialists": specialists,
        "specialist_token_metrics": specialist_token_metrics,
        "states": torch.cat(all_states),
        "token_ids": torch.cat(all_ids),
    }


def receiver_prompt(tokenizer, question: str, enable_thinking: bool, reports: str | None = None, *, meeting_mode: bool = False) -> str:
    if meeting_mode:
        user = (
            f"Objective: {question}\n\nCreate a concise meeting summary with exactly these sections:\n"
            f"{MEETING_SECTIONS}\n"
            "Use each item once only: accepted outcomes belong in Confirmed decisions; explicit owner commitments belong in "
            "Action items; unresolved questions, blockers, and uncertainty belong in Open issues and risks; everything else "
            "important belongs in Key context and discussion. Under each section, write 'None explicitly stated.' when the "
            "specialist findings provide no supported item. Only name an owner or deadline when explicitly supported by the specialist findings."
        )
    else:
        user = f"Question: {question}\n\nGive a concise factual answer."
    if reports is not None:
        user += f"\n\nSpecialist handoffs:\n{reports}"
    return chat_prompt(
        tokenizer,
        "You aggregate specialist findings into an evidence-grounded response. Do not invent facts. "
        "Treat a suggestion or discussion topic as neither a decision nor an action unless the evidence states a commitment. "
        "Do not repeat an item across sections.",
        user,
        enable_thinking,
    )


def block_receiver_prompt(tokenizer, question: str, block_id: int, enable_thinking: bool) -> str:
    """Ask a receiver to summarize one block without promoting local claims globally."""
    user = (
        f"Objective: {question}\n\n"
        f"You are summarizing chronological meeting block {block_id}. Create a concise block-local summary with exactly these sections:\n"
        f"{MEETING_SECTIONS}\n"
        "Use only supported specialist findings. Keep proposals and discussions out of Confirmed decisions, and do not repeat "
        "an item across sections. Do not infer owners, deadlines, or agreement. This is a local block summary, not the final meeting summary."
    )
    return chat_prompt(
        tokenizer,
        "You aggregate specialist findings into an evidence-grounded block summary. Do not invent facts.",
        user,
        enable_thinking,
    )


def final_merge_prompt(tokenizer, question: str, enable_thinking: bool, reports: str | None = None) -> str:
    user = (
        f"Objective: {question}\n\nCreate the final evidence-grounded meeting summary with exactly these sections:\n"
        f"{MEETING_SECTIONS}\n"
        "Merge the chronological block summaries. Deduplicate repeated facts. When an earlier proposal conflicts with a later "
        "confirmed outcome, use the later outcome. Do not invent facts, owners, deadlines, decisions, or action items."
    )
    if reports is not None:
        user += f"\n\nChronological block summaries:\n{reports}"
    return chat_prompt(
        tokenizer,
        "You merge evidence-grounded block summaries into final meeting minutes.",
        user,
        enable_thinking,
    )


@torch.inference_mode()
def answer_variant(model, tokenizer, bridge, record, reports, states, token_ids, variant, args, *, meeting_mode: bool):
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
    prefix = bridge.align(states, token_ids)
    return decode(
        tokenizer,
        generate_from_prefix(
            model, tokenizer, encoded.input_ids, prefix,
            available_generation_tokens(args, encoded.input_ids.size(1) + prefix.size(0)),
        ),
    )


@torch.inference_mode()
def statebridge_block_summary(model, tokenizer, bridge, record, block_result, args) -> dict[str, Any]:
    """Turn one block's specialist messages into a local summary and bridge-ready states."""
    prompt = block_receiver_prompt(tokenizer, record["question"], block_result["block_id"], args.enable_thinking)
    encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
    specialist_prefix = bridge.align(block_result["states"], block_result["token_ids"])
    summary_ids, summary_states = generate_after_embedding_prefix(
        model,
        tokenizer,
        encoded.input_ids,
        specialist_prefix,
        max_context_tokens=args.model_context_tokens,
    )
    summary_ids, summary_states = remove_thinking_tokens(tokenizer, summary_ids, summary_states)
    if summary_ids.size(1) < 2:
        raise RuntimeError(f"Block {block_result['block_id']} summary produced fewer than two usable tokens.")
    count = min(args.block_summary_prefix_tokens, summary_ids.size(1))
    return {
        "block_id": block_result["block_id"],
        "summary": decode(tokenizer, summary_ids),
        "states": summary_states[-count:],
        "token_ids": summary_ids[0, -count:],
        "summary_tokens": int(summary_ids.size(1)),
        "selected_states": count,
    }


@torch.inference_mode()
def merge_block_summaries(model, tokenizer, bridge, record, block_summaries, variant, args) -> str:
    """Produce the final answer from chronological block summaries.

    The statebridge variant receives latent states produced by the block receivers;
    the text variant receives the same summaries as natural language.
    """
    reports = "\n\n".join(
        f"[block {item['block_id']}]\n{item['summary']}" for item in block_summaries
    )
    if variant == "text":
        prompt = final_merge_prompt(tokenizer, record["question"], args.enable_thinking, reports)
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        _, answer_ids = generate_from_ids(model, tokenizer, encoded.input_ids, encoded.attention_mask, available_generation_tokens(args, encoded.input_ids.size(1)))
        return decode(tokenizer, answer_ids)
    prompt = final_merge_prompt(tokenizer, record["question"], args.enable_thinking)
    encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
    if variant == "no_comm":
        _, answer_ids = generate_from_ids(model, tokenizer, encoded.input_ids, encoded.attention_mask, available_generation_tokens(args, encoded.input_ids.size(1)))
        return decode(tokenizer, answer_ids)
    states = torch.cat([item["states"] for item in block_summaries])
    token_ids = torch.cat([item["token_ids"] for item in block_summaries])
    prefix = bridge.align(states, token_ids)
    return decode(tokenizer, generate_from_prefix(model, tokenizer, encoded.input_ids, prefix, available_generation_tokens(args, encoded.input_ids.size(1) + prefix.size(0))))


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


def automatic_meeting_block_tokens(model, tokenizer) -> int:
    """Choose a conservative source budget from the loaded model context limit."""
    return max(32, int(model_context_tokens(model, tokenizer) * 0.75))


def available_generation_tokens(args, occupied_tokens: int) -> int:
    remaining_tokens = args.model_context_tokens - occupied_tokens
    if remaining_tokens < 1:
        raise RuntimeError("The prompt fills the model context window; no output tokens remain.")
    return remaining_tokens


def main() -> None:
    args = parse_args()
    if args.prefix_tokens_per_agent < 2:
        raise ValueError("--prefix-tokens-per-agent must be at least 2")
    if args.block_summary_prefix_tokens is None:
        args.block_summary_prefix_tokens = args.prefix_tokens_per_agent
    if args.block_summary_prefix_tokens < 2:
        raise ValueError("--block-summary-prefix-tokens must be at least 2")
    if args.document_chunk_overlap_tokens < 0:
        raise ValueError("--document-chunk-overlap-tokens cannot be negative")
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
    args.model_context_tokens = model_context_tokens(model, tokenizer)
    meeting_block_tokens = automatic_meeting_block_tokens(model, tokenizer) if meeting_mode else 0
    if meeting_mode and args.document_chunk_overlap_tokens >= meeting_block_tokens:
        raise ValueError("--document-chunk-overlap-tokens must be smaller than the automatic meeting block size")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as jsonl, args.output.open("w", encoding="utf-8") as report:
        run_kind = "Meeting" if meeting_mode else "QASPER"
        communication_totals = {"text_tokens": 0, "latent_states": 0}
        report.write(f"Cross-agent Prefix KV Cache + StateBridge {run_kind} run: {datetime.now().isoformat(timespec='seconds')}\n")
        source_mode = (
            f"full document in chronological blocks of {meeting_block_tokens} tokens (75% of loaded model context; {args.document_chunk_overlap_tokens}-token overlap)"
            if meeting_mode
            else f"legacy first {args.source_max_tokens} source tokens"
        )
        report.write(
            f"model={args.model}, source_mode={source_mode}, roles={','.join(args.roles)}, "
            f"evidence_policy={args.evidence_policy}\n"
        )
        report.write(
            "Each source block is prefetched once and reused by cloned per-agent KV caches. "
            "For multi-block meetings, each block becomes a StateBridge summary and those summaries are merged through a second StateBridge handoff.\n\n"
        )
        for index, record in enumerate(records, start=1):
            if meeting_mode:
                document_blocks = chunk_document(
                    tokenizer,
                    record["context"],
                    chunk_tokens=meeting_block_tokens,
                    overlap_tokens=args.document_chunk_overlap_tokens,
                )
            else:
                source = trim_context(tokenizer, record["context"], args.source_max_tokens)
                source_tokens = len(tokenizer(source, add_special_tokens=False).input_ids)
                document_blocks = [Chunk(0, source, source_tokens)]
            if not document_blocks:
                raise ValueError(f"Source is empty after tokenization: {record['id']}")

            block_results = [
                process_source_block(model, tokenizer, record, block.text, block, args, meeting_mode=meeting_mode)
                for block in document_blocks
            ]
            specialists = [
                {
                    "block_id": block_result["block_id"],
                    **specialist,
                }
                for block_result in block_results
                for specialist in block_result["specialists"]
            ]
            specialist_token_metrics = [
                item
                for block_result in block_results
                for item in block_result["specialist_token_metrics"]
            ]
            reports = "\n\n".join(
                f"[block {item['block_id']} | {item['role']}]\n{item['message']}" for item in specialists
            )
            states = torch.cat([block_result["states"] for block_result in block_results])
            token_ids = torch.cat([block_result["token_ids"] for block_result in block_results])
            prefix_tokens = sum(block_result["prefix_tokens"] for block_result in block_results)
            prefix_bytes = max(block_result["prefix_cache_bytes"] for block_result in block_results)
            prefill_seconds = sum(block_result["prefix_prefill_seconds"] for block_result in block_results)
            prefill_chunks = sum(block_result["prefix_prefill_chunks"] for block_result in block_results)
            text_handoff_tokens = len(tokenizer(reports, add_special_tokens=False).input_ids)
            latent_states = int(states.size(0))
            hierarchical_merge = meeting_mode and len(block_results) > 1
            block_summaries = []
            if hierarchical_merge:
                block_summaries = [
                    statebridge_block_summary(model, tokenizer, bridge, record, block_result, args)
                    for block_result in block_results
                ]
            block_summary_text_tokens = sum(item["summary_tokens"] for item in block_summaries)
            block_summary_latent_states = sum(item["selected_states"] for item in block_summaries)
            total_text_tokens = text_handoff_tokens + block_summary_text_tokens
            total_latent_states = latent_states + block_summary_latent_states
            communication_delta = total_latent_states - total_text_tokens
            communication_percent = 0.0 if total_text_tokens == 0 else communication_delta * 100.0 / total_text_tokens
            communication_totals["text_tokens"] += total_text_tokens
            communication_totals["latent_states"] += total_latent_states
            answers = {}
            if not args.communication_metrics_only:
                if hierarchical_merge:
                    answers = {
                        variant: merge_block_summaries(model, tokenizer, bridge, record, block_summaries, variant, args)
                        for variant in args.variants
                    }
                else:
                    answers = {
                        variant: answer_variant(
                            model, tokenizer, bridge, record, reports, states, token_ids, variant, args,
                            meeting_mode=meeting_mode
                        )
                        for variant in args.variants
                    }
            result = {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record.get("answer", ""),
                "gold_answers": record.get("answers") or [record.get("answer", "")],
                "prefix_tokens": prefix_tokens,
                "prefix_cache_bytes": prefix_bytes,
                "prefix_prefill_seconds": prefill_seconds,
                "prefix_prefill_chunks": prefill_chunks,
                "document_blocks": [
                    {
                        "block_id": block_result["block_id"],
                        "source_tokens": block_result["source_tokens"],
                        "prefix_tokens": block_result["prefix_tokens"],
                        "prefix_prefill_seconds": block_result["prefix_prefill_seconds"],
                        "prefix_prefill_chunks": block_result["prefix_prefill_chunks"],
                    }
                    for block_result in block_results
                ],
                "statebridge_tokens": total_latent_states,
                "communication": {
                    "text_handoff_tokens": total_text_tokens,
                    "latent_state_tokens": total_latent_states,
                    "latent_minus_text_tokens": communication_delta,
                    "latent_minus_text_percent": communication_percent,
                    "specialist_to_block": {
                        "text_handoff_tokens": text_handoff_tokens,
                        "latent_state_tokens": latent_states,
                    },
                    "block_to_final": {
                        "text_handoff_tokens": block_summary_text_tokens,
                        "latent_state_tokens": block_summary_latent_states,
                    },
                },
                "specialist_token_metrics": specialist_token_metrics,
                "specialists": [] if args.communication_metrics_only else specialists,
                "hierarchical_block_merge": hierarchical_merge,
                "block_summaries": [
                    {
                        "block_id": item["block_id"],
                        "summary_tokens": item["summary_tokens"],
                        "selected_states": item["selected_states"],
                        "summary": item["summary"],
                    }
                    for item in block_summaries
                ],
                "answers": answers,
            }
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            report.write("=" * 80 + f"\nSample {index}\nId: {record['id']}\n\nQuestion:\n{record['question']}\n\n")
            report.write(
                f"Document blocks: {len(block_results)}; combined prefixes: {result['prefix_tokens']} tokens; "
                f"one KV prefill per block; peak block KV cache: {prefix_bytes / 2**20:.1f} MiB; "
                f"total prefill: {prefill_seconds:.2f}s\n"
            )
            report.write(
                f"StateBridge specialist prefix: {states.size(0)} latent states\n\n"
            )
            if hierarchical_merge:
                report.write(
                    f"Hierarchical merge: {len(block_summaries)} block summaries; "
                    f"text={block_summary_text_tokens} tokens; latent={block_summary_latent_states} states.\n\n"
                )
            report.write(
                "Communication cost (all StateBridge hops): "
                f"text={total_text_tokens} tokens; latent={total_latent_states} states; "
                f"latent-text={communication_delta:+d} tokens ({communication_percent:+.1f}%). "
                "Negative means latent saved communication units; positive means latent used more.\n\n"
            )
            if not args.communication_metrics_only:
                for item in block_summaries:
                    report.write(f"block {item['block_id']} summary (debug only):\n{item['summary']}\n\n")
                for item in specialists:
                    debug_message = item.get("evidence_message", item["message"])
                    report.write(f"block {item['block_id']} | {item['role']} handoff (debug only):\n{debug_message}\n\n")
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
