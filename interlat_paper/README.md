# Paper-Style Interlat

This directory is a separate implementation of Interlat's main training stage.
It does not reuse the adapter-only objective in `interlat_same_model`.

## What is implemented

- Frozen instruction-tuned Sender trajectories from `interlat_same_model.collect`.
- A trainable Qwen Actor plus the paper's MHA communication adapter.
- `<bop>` / `<eop>` token embeddings around the latent message.
- Curriculum mixing follows the released implementation: `r` is sampled from `{0.1, ..., 0.9}`, and the transmitted message is a latent prefix followed by the corresponding suffix of Sender-plan token embeddings.
- `L_task + w_contrast L_contrast + w_plan L_plan`, where `L_contrast = max(0, 0.69 - JS)` for a cross-task latent message and `L_plan` is KL plus cosine alignment to the text-plan output distribution. The two weights are dynamically derived from their current loss values, as in the authors' implementation.
- An optional DeepSpeed configuration for larger models or multi-GPU environments.

The paper's compression reasoning model is intentionally not included in this stage. It is trained only after the full Actor has converged, with the Actor and adapter frozen.

## Main-stage command

First collect plans with an instruction-tuned Sender. Use the Oracle-evidence JSONL split for the controlled experiment:

```bash
python -m interlat_same_model.collect --data data/qasper_oracle_evidence/train.jsonl --output interlat_paper/data/qasper_oracle_train.pt --model Qwen/Qwen2.5-3B-Instruct --source-max-tokens 4096 --sender-max-new-tokens 1024
python -m interlat_same_model.collect --data data/qasper_oracle_evidence/validation.jsonl --output interlat_paper/data/qasper_oracle_validation.pt --model Qwen/Qwen2.5-3B-Instruct --source-max-tokens 4096 --sender-max-new-tokens 1024
```

For a 48GB GPU, use the same 3B model for the full Actor. Without a DeepSpeed configuration, the script uses ordinary full-parameter AdamW and keeps parameters, gradients, and optimizer states on the GPU:

```bash
python -m interlat_paper.train --train-hidden interlat_paper/data/qasper_oracle_train.pt --val-hidden interlat_paper/data/qasper_oracle_validation.pt --output-dir interlat_paper/checkpoints/qasper_oracle_qwen3b --actor-model Qwen/Qwen2.5-3B-Instruct --epochs 3 --batch-size 1 --gradient-accumulation 16 --learning-rate 1e-5 --warmup-ratio 0.03 --gradient-checkpointing
```

The authors use BF16 and FlashAttention 2. The command therefore requests `flash_attention_2`; install a CUDA-compatible `flash-attn` wheel on the Linux training machine before this run. If that cannot be installed, pass `--attention-implementation ""` for a slower compatibility run, which is no longer the exact performance setting.

The supplied DeepSpeed JSON remains an optional alternative for larger models or multi-GPU environments. Do not reuse a hidden-state `.pt` collected with the 7B Sender for this same-model 3B run: recollect it with the 3B Sender commands above.

## Evaluation

The default 3B all-GPU training checkpoint is already consolidated:

```bash
python -m interlat_paper.evaluate --hidden-data interlat_paper/data/qasper_oracle_validation.pt --checkpoint interlat_paper/checkpoints/qasper_oracle_qwen3b/best.pt --output interlat_paper/output/qasper_oracle.txt --num-samples 8 --include-text-control
```
