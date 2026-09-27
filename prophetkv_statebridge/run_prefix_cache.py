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

from .prefix_cache import cache_nbytes, generate_after_prefix_cache, prefill_prefix
from .run import ROLE_INSTRUCTIONS


MEETING_ROLE_CONTRACTS = {
    "facts": (
        "List only high-confidence factual meeting context. Do not infer a decision, action, owner, "
        "or deadline from a participant's job title."
    ),
    "decisions": (
        "List only decisions explicitly accepted, agreed, chosen, approved, or rejected in the transcript. "
        "Do not turn a project description, goal, or proposal into a decision."
    ),
    "actions": (
        "List only explicit commitments or assigned follow-ups. An action needs an explicit task; include an "
        "owner or deadline only when the transcript explicitly states one. Do not infer work from a person's role."
    ),
    "risks": (
        "List only explicitly stated risks, blockers, uncertainties, disagreements, or unresolved questions. "
        "Do not invent a risk from normal project context."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--data", type=Path, help="QASPER-style JSONL input")
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
    parser.add_argument("--roles", nargs="+", choices=sorted(ROLE_INSTRUCTIONS), default=["facts", "decisions", "actions", "risks"])
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=192)
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=64)
    parser.add_argument("--snap-ratio", type=float, default=0.3)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=["statebridge", "text", "no_comm"], default=["statebridge", "text", "no_comm"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def specialist_prompt(
    tokenizer, source: str, record: dict[str, Any], role: str, enable_thinking: bool, *, meeting_mode: bool
) -> str:
    """Build a normal Qwen chat prompt; cache only its token-identical prefix."""
    task = ROLE_INSTRUCTIONS[role]
    format_instruction = ""
    if meeting_mode:
        task = MEETING_ROLE_CONTRACTS[role]
        format_instruction = (
            " Return at most four short bullets. Every bullet must contain a direct supporting quotation or "
            "timestamp from the transcript. If no supported item exists, output exactly: None explicitly stated."
        )
    user = (
        f"Source transcript:\n{source}\n\nQuestion: {record['question']}\n\n"
        f"Task: {task} Give a concise factual handoff for another agent. "
        f"Only use the transcript. Include specific facts and state missing information rather than guessing.{format_instruction}"
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
        max_new_tokens=args.agent_max_new_tokens,
    )
    message_ids, states = remove_thinking_tokens(tokenizer, message_ids, states)
    count = min(args.prefix_tokens_per_agent, message_ids.size(1))
    return decode(tokenizer, message_ids), states[-count:], message_ids[0, -count:]


def receiver_prompt(tokenizer, question: str, enable_thinking: bool, reports: str | None = None, *, meeting_mode: bool = False) -> str:
    if meeting_mode:
        user = (
            f"Objective: {question}\n\nCreate a concise meeting summary with exactly these sections:\n"
            "Decisions\nAction items\nRisks and open questions\n"
            "Only name an owner or deadline when explicitly supported by the specialist findings."
        )
    else:
        user = f"Question: {question}\n\nGive a concise factual answer."
    if reports is not None:
        user += f"\n\nSpecialist handoffs:\n{reports}"
    return chat_prompt(
        tokenizer,
        "You aggregate specialist findings into an evidence-grounded response. Do not invent facts.",
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
    torch.manual_seed(args.seed)
    meeting_mode = args.transcript is not None
    if meeting_mode:
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
        records = load_records(args.data, args.num_samples)
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
        report.write(f"Cross-agent Prefix KV Cache + StateBridge {run_kind} run: {datetime.now().isoformat(timespec='seconds')}\n")
        report.write(f"model={args.model}, source_max_tokens={args.source_max_tokens}, roles={','.join(args.roles)}\n")
        report.write("Each source prefix is prefetched once and reused by cloned per-agent KV caches.\n\n")
        for index, record in enumerate(records, start=1):
            source = trim_context(tokenizer, record["context"], args.source_max_tokens)
            role_inputs = [
                tokenizer(
                    specialist_prompt(
                        tokenizer, source, record, role, args.enable_thinking, meeting_mode=meeting_mode
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
                        tokenizer, source, record, alternate_role, args.enable_thinking, meeting_mode=meeting_mode
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
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - started
            prefix_bytes = cache_nbytes(prefix_cache)

            specialists = []
            all_states, all_ids = [], []
            for role, full_prompt_ids in zip(args.roles, role_inputs):
                message, states, token_ids = specialist_from_cache(
                    model,
                    tokenizer,
                    prefix_cache,
                    encoded_prefix.size(1),
                    full_prompt_ids[:, common_length:],
                    args,
                )
                specialists.append({"role": role, "message": message})
                all_states.append(states)
                all_ids.append(token_ids)
            reports = "\n\n".join(f"[{item['role']}]\n{item['message']}" for item in specialists)
            states, token_ids = torch.cat(all_states), torch.cat(all_ids)
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
                "statebridge_tokens": int(states.size(0)),
                "specialists": specialists,
                "answers": answers,
            }
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            report.write("=" * 80 + f"\nSample {index}\nId: {record['id']}\n\nQuestion:\n{record['question']}\n\n")
            report.write(f"Shared prefix: {result['prefix_tokens']} tokens; KV cache: {prefix_bytes / 2**20:.1f} MiB; prefill: {prefill_seconds:.2f}s\n")
            report.write(f"StateBridge specialist prefix: {states.size(0)} tokens\n\n")
            for item in specialists:
                report.write(f"{item['role']} handoff (debug only):\n{item['message']}\n\n")
            for variant, answer in answers.items():
                report.write(f"{variant} answer:\n{answer}\n\n")
            if not meeting_mode:
                report.write(f"Gold answer:\n{record.get('answer', '')}\n\n")
            report.flush()
            print(f"completed {index}/{len(records)}: {record['id']}", flush=True)


if __name__ == "__main__":
    main()
