"""Supervised same-model Interlat training on collected long-context plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlat_same_model.data import HiddenStateDataset
from interlat_same_model.latent import InterlatReceiver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hidden", type=Path, required=True)
    parser.add_argument("--val-hidden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--compressed-latent-len", type=int, default=0)
    parser.add_argument("--plan-similarity-weight", type=float, default=0.5)
    parser.add_argument("--random-contrast-weight", type=float, default=0.1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop after this many epochs without validation task-loss improvement; 0 disables it.",
    )
    parser.add_argument("--unfreeze-receiver", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def receiver_prompt(tokenizer, question: str, device: str) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Answer the question using the received latent plan.\n\nQuestion: " + question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)


def answer_ids(tokenizer, answer: str, max_tokens: int, device: str) -> torch.Tensor:
    ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
    ids.append(tokenizer.eos_token_id)
    return torch.tensor([ids], dtype=torch.long, device=device)


def run_epoch(model, loader, tokenizer, bop_id, eop_id, optimizer, device, args, train: bool):
    totals = {"loss": 0.0, "task_loss": 0.0, "plan_loss": 0.0, "contrast_loss": 0.0}
    count = 0
    if train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
    for step, batch in enumerate(loader, start=1):
        for record in batch:
            states = record["sender_states"].unsqueeze(0).to(device=device, dtype=next(model.adapter.parameters()).dtype)
            prompt = receiver_prompt(tokenizer, record["question"], device)
            target = answer_ids(tokenizer, record["answer"], args.max_answer_tokens, device)
            context = torch.enable_grad() if train else torch.inference_mode()
            with context:
                metrics = model(sender_states=states, bop_id=bop_id, eop_id=eop_id, prompt_ids=prompt, target_ids=target)
            if train:
                (metrics["loss"] / (len(batch) * args.gradient_accumulation)).backward()
            for name in totals:
                totals[name] += float(metrics[name].detach().float())
            count += 1
        if train and step % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if train and len(loader) % args.gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {name: value / max(count, 1) for name, value in totals.items()}


def main():
    args = parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("batch-size and gradient-accumulation must be positive")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    train_data = HiddenStateDataset(args.train_hidden, args.train_limit)
    val_data = HiddenStateDataset(args.val_hidden, args.val_limit)
    source_size = train_data[0]["sender_states"].shape[-1]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<bop>", "<eop>"]})
    bop_id, eop_id = tokenizer.convert_tokens_to_ids("<bop>"), tokenizer.convert_tokens_to_ids("<eop>")
    receiver = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    receiver.resize_token_embeddings(len(tokenizer))
    receiver.config.pad_token_id = tokenizer.pad_token_id
    if not args.unfreeze_receiver:
        for parameter in receiver.parameters():
            parameter.requires_grad = False
        # Boundary embeddings must adapt even in adapter-only training.
        receiver.get_input_embeddings().weight.requires_grad = True
    model = InterlatReceiver(
        receiver, source_hidden_size=source_size, num_heads=args.num_heads,
        compressed_latent_len=args.compressed_latent_len or None,
        plan_similarity_weight=args.plan_similarity_weight,
        random_contrast_weight=args.random_contrast_weight,
    ).to(device=device, dtype=dtype)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    collate = lambda rows: rows
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tokenizer").mkdir(exist_ok=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    best_task_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, tokenizer, bop_id, eop_id, optimizer, device, args, train=True)
        val_metrics = run_epoch(model, val_loader, tokenizer, bop_id, eop_id, optimizer, device, args, train=False)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        # The auxiliary losses make latent representations easier to consume,
        # but answer quality is represented by task_loss. Select checkpoints by
        # task_loss so a decreasing alignment term cannot hide worse answers.
        if val_metrics["task_loss"] < best_task_loss:
            best_task_loss = val_metrics["task_loss"]
            epochs_without_improvement = 0
            checkpoint = {
                "model_name": args.model, "source_hidden_size": source_size, "num_heads": args.num_heads,
                "compressed_latent_len": args.compressed_latent_len, "plan_similarity_weight": args.plan_similarity_weight,
                "random_contrast_weight": args.random_contrast_weight, "unfreeze_receiver": args.unfreeze_receiver,
                "best_validation_task_loss": best_task_loss,
                "adapter": model.adapter.state_dict(),
                "compressor": model.compressor.state_dict() if model.compressor else None,
                "boundary_embeddings": receiver.get_input_embeddings().weight.detach()[[bop_id, eop_id]].cpu(),
            }
            if args.unfreeze_receiver:
                checkpoint["receiver"] = receiver.state_dict()
            torch.save(checkpoint, args.output_dir / "best.pt")
            print(f"saved checkpoint: {args.output_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                print("early stopping: validation task_loss did not improve", flush=True)
                break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
