"""Paper-style Interlat main-stage training.

This is the first stage from Interlat: the Sender is frozen, while the Actor
and communication adapter are trained jointly with task, separation, and
text-plan alignment losses. Compression is deliberately a later stage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlat_paper.model import InterlatActor
from interlat_paper.objectives import plan_alignment_loss, random_contrast_loss
from interlat_same_model.data import HiddenStateDataset


CURRICULUM_RATES = tuple(step / 10 for step in range(1, 10))


def parse_args():
    parser = argparse.ArgumentParser(description="Interlat paper main-stage Actor training")
    parser.add_argument("--train-hidden", type=Path, required=True)
    parser.add_argument("--val-hidden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actor-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--attention-implementation",
        default="flash_attention_2",
        help="Paper setting. Pass an empty string only when FlashAttention 2 is unavailable.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--max-plan-tokens", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--deepspeed-config",
        type=Path,
        help="Optional DeepSpeed JSON config. Required in practice for full 7B training on a 48GB card.",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def chat_prompt(tokenizer, content: str) -> str:
    """Render an Actor instruction for both base and chat-tokenizer variants."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
    return f"Instruction:\n{content}\n\nResponse:\n"


def prompt_ids(tokenizer, question: str, device: torch.device, max_tokens: int) -> torch.Tensor:
    content = "Answer the question concisely using the received message.\n\nQuestion: " + question
    return tokenizer(
        chat_prompt(tokenizer, content), return_tensors="pt", truncation=True, max_length=max_tokens
    ).input_ids.to(device)


def target_ids(tokenizer, answer: str, device: torch.device, max_tokens: int) -> torch.Tensor:
    ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return torch.tensor([ids], dtype=torch.long, device=device)


def draft_ids(tokenizer, draft: str, device: torch.device, max_tokens: int) -> torch.Tensor:
    ids = tokenizer(draft, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
    if not ids:
        ids = [tokenizer.eos_token_id]
    return torch.tensor([ids], dtype=torch.long, device=device)


def make_scheduler(optimizer, total_updates: int, warmup_ratio: float):
    warmup_updates = max(1, int(total_updates * warmup_ratio))

    def multiplier(update: int) -> float:
        if update < warmup_updates:
            return float(update + 1) / warmup_updates
        remaining = max(total_updates - warmup_updates, 1)
        return max(0.0, float(total_updates - update) / remaining)

    return LambdaLR(optimizer, multiplier)


def same_length_negative(states: torch.Tensor, negative_states: torch.Tensor) -> torch.Tensor:
    """Crop or repeat the cross-task state sequence as in the official code."""
    target_length = states.size(1)
    if negative_states.size(1) >= target_length:
        return negative_states[:, :target_length]
    repeats = math.ceil(target_length / negative_states.size(1))
    repeated = negative_states.repeat(1, repeats, 1)[:, :target_length]
    return repeated + torch.randn_like(repeated) * 0.01


def sample_metrics(model, record, negative_record, tokenizer, bop_id, eop_id, device, args, train):
    dtype = next(model.parameters()).dtype
    states = record["sender_states"].unsqueeze(0).to(device=device, dtype=dtype)
    negative_states = negative_record["sender_states"].unsqueeze(0).to(device=device, dtype=dtype)
    negative_states = same_length_negative(states, negative_states)
    prompt = prompt_ids(tokenizer, record["question"], device, args.max_prompt_tokens)
    target = target_ids(tokenizer, record["answer"], device, args.max_answer_tokens)
    plan = draft_ids(tokenizer, record["sender_draft"], device, args.max_plan_tokens)
    replacement_rate = random.choice(CURRICULUM_RATES) if train else 0.5

    adapted = model.adapt_latents(states)
    mixed_message = model.mix_plan_tokens(adapted, plan, replacement_rate)
    matched, matched_labels = model.forward_message(prompt, target, mixed_message, bop_id, eop_id)

    # Eq. (3): textual reasoning is a teacher signal, not input to the Actor at inference.
    with torch.no_grad():
        textual, textual_labels = model.forward_message(
            prompt, target, model.actor.get_input_embeddings()(plan).to(dtype), bop_id, eop_id
        )
        mismatched_message = model.mix_plan_tokens(
            model.adapt_latents(negative_states), plan, random.choice(CURRICULUM_RATES) if train else 0.5
        )
        mismatched, mismatched_labels = model.forward_message(prompt, target, mismatched_message, bop_id, eop_id)

    task_loss = matched.loss
    contrast_loss = random_contrast_loss(
        matched.logits, matched_labels, mismatched.logits, mismatched_labels
    )
    alignment_loss = plan_alignment_loss(
        matched.logits, matched_labels, textual.logits, textual_labels
    )
    # These adaptive coefficients are the ones used in the authors' released implementation.
    lambda_align = 0.01 + min(max(float(alignment_loss.detach()) / 4.0, 0.0), 1.0) * 0.04
    lambda_contrast = 0.01 + min(max(float(contrast_loss.detach()) / 0.69, 0.0), 1.0) * 0.49
    total = task_loss + lambda_contrast * contrast_loss + lambda_align * alignment_loss
    return {
        "loss": total,
        "task_loss": task_loss,
        "contrast_loss": contrast_loss,
        "alignment_loss": alignment_loss,
        "lambda_contrast": torch.tensor(lambda_contrast, device=device),
        "lambda_align": torch.tensor(lambda_align, device=device),
    }


def run_epoch(
    model,
    loader,
    dataset,
    tokenizer,
    bop_id,
    eop_id,
    device,
    args,
    epoch,
    optimizer=None,
    scheduler=None,
    deepspeed_engine=None,
):
    train = optimizer is not None
    model.train(train)
    totals = {name: 0.0 for name in ("loss", "task_loss", "contrast_loss", "alignment_loss")}
    count = 0
    if train:
        if deepspeed_engine:
            deepspeed_engine.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    for batch_index, batch in enumerate(loader):
        for record in batch:
            if train:
                # The paper's L_sep uses a cross-task message. With a small
                # per-device batch, sample that negative from the same split.
                negative_record = record
                while negative_record is record and len(dataset) > 1:
                    negative_record = dataset[random.randrange(len(dataset))]
            else:
                negative_record = dataset[(count + 1) % len(dataset)]
            context = nullcontext() if train else torch.inference_mode()
            with context:
                metrics = sample_metrics(
                    model, record, negative_record, tokenizer, bop_id, eop_id, device, args, train
                )
            if train:
                scaled_loss = metrics["loss"] / (len(batch) * args.gradient_accumulation)
                if deepspeed_engine:
                    deepspeed_engine.backward(scaled_loss)
                else:
                    scaled_loss.backward()
            for name in totals:
                totals[name] += float(metrics[name].detach().float())
            count += 1
        if train and (batch_index + 1) % args.gradient_accumulation == 0:
            if deepspeed_engine:
                deepspeed_engine.step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
    if train and len(loader) % args.gradient_accumulation:
        if deepspeed_engine:
            deepspeed_engine.step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return {name: value / max(count, 1) for name, value in totals.items()}


def main():
    args = parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("batch-size and gradient-accumulation must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    train_data = HiddenStateDataset(args.train_hidden, args.train_limit)
    val_data = HiddenStateDataset(args.val_hidden, args.val_limit)
    source_size = train_data[0]["sender_states"].shape[-1]

    tokenizer = AutoTokenizer.from_pretrained(args.actor_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<bop>", "<eop>"]})
    bop_id = tokenizer.convert_tokens_to_ids("<bop>")
    eop_id = tokenizer.convert_tokens_to_ids("<eop>")
    load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.attention_implementation:
        load_kwargs["attn_implementation"] = args.attention_implementation
    actor = AutoModelForCausalLM.from_pretrained(args.actor_model, **load_kwargs)
    actor.resize_token_embeddings(len(tokenizer))
    actor.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable()
        actor.config.use_cache = False
    model = InterlatActor(actor, source_size, args.num_heads)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=lambda rows: rows)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=lambda rows: rows)
    total_updates = args.epochs * math.ceil(len(train_loader) / args.gradient_accumulation)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = make_scheduler(optimizer, total_updates, args.warmup_ratio)
    deepspeed_engine = None
    if args.deepspeed_config:
        try:
            import deepspeed
        except ImportError as error:
            raise RuntimeError("Install deepspeed to use --deepspeed-config.") from error
        deepspeed_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            config=str(args.deepspeed_config),
        )
        model = deepspeed_engine.module
        device = deepspeed_engine.device
        scheduler = None
    else:
        model = model.to(device=device, dtype=dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    metadata = vars(args) | {"source_hidden_size": source_size, "bop_id": bop_id, "eop_id": eop_id}
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"device": str(device), "dtype": str(dtype), "train_records": len(train_data), "val_records": len(val_data), "parameters": sum(p.numel() for p in model.parameters())}), flush=True)

    best_task_loss = float("inf")
    without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, train_data, tokenizer, bop_id, eop_id, device, args, epoch,
            optimizer, scheduler, deepspeed_engine,
        )
        val_metrics = run_epoch(model, val_loader, val_data, tokenizer, bop_id, eop_id, device, args, epoch)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["task_loss"] < best_task_loss:
            best_task_loss = val_metrics["task_loss"]
            without_improvement = 0
            checkpoint = {
                "metadata": metadata,
                "best_validation_task_loss": best_task_loss,
            }
            if deepspeed_engine:
                deepspeed_engine.save_checkpoint(
                    str(args.output_dir / "deepspeed"), tag="best", client_state=checkpoint
                )
                torch.save(checkpoint, args.output_dir / "best_metadata.pt")
            else:
                checkpoint["actor"] = model.actor.state_dict()
                checkpoint["adapter"] = model.adapter.state_dict()
                torch.save(checkpoint, args.output_dir / "best.pt")
            saved_path = args.output_dir / ("deepspeed/best" if deepspeed_engine else "best.pt")
            print(f"saved checkpoint: {saved_path}", flush=True)
        else:
            without_improvement += 1
            if args.early_stopping_patience and without_improvement >= args.early_stopping_patience:
                print("early stopping: validation task_loss did not improve", flush=True)
                break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
