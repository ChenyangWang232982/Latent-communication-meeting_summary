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

The latent message is inserted after the user turn and before the assistant generation prefix. Checkpoints trained before this insertion rule was fixed are not compatible with this implementation and must be retrained.

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

To diagnose the paper curriculum before changing training, compare a mostly-latent
message with the pure-latent endpoint. `r` is the latent proportion: `0` is text
plan only and `1` is latent only.

```bash
python -m interlat_paper.evaluate --hidden-data interlat_paper/data/qasper_oracle_validation.pt --checkpoint interlat_paper/checkpoints/qasper_oracle_qwen3b_v3/best.pt --output interlat_paper/output/qasper_oracle_qwen3b_v3_mixes.txt --num-samples 8 --include-text-control --mix-ratios 0.5 0.9 1.0
```

## QASPER Pure-Latent Endpoint Adaptation

The command below is an explicitly labelled adaptation experiment, not an
unchanged paper reproduction. It starts from the paper-curriculum checkpoint,
keeps the `{0.1, ..., 0.9}` curriculum for half of the samples, and makes the
other half pure latent (`r=1`). Validation and checkpoint selection also use
pure latent messages. The original `qasper_oracle_qwen3b_v3` checkpoint remains
the paper-curriculum control.

```bash
python -m interlat_paper.train --train-hidden interlat_paper/data/qasper_oracle_train.pt --val-hidden interlat_paper/data/qasper_oracle_validation.pt --output-dir interlat_paper/checkpoints/qasper_oracle_qwen3b_v3_pure_latent --init-checkpoint interlat_paper/checkpoints/qasper_oracle_qwen3b_v3/best.pt --actor-model Qwen/Qwen2.5-3B-Instruct --epochs 4 --batch-size 1 --gradient-accumulation 16 --learning-rate 5e-6 --warmup-ratio 0.03 --gradient-checkpointing --pure-latent-start-epoch 1 --pure-latent-probability 0.5 --validation-replacement-rate 1.0 --early-stopping-patience 2
```

Evaluate the adapted checkpoint with the same controls. Improvement must be
judged from `r=1`, not from the text or mixed controls.

```bash
python -m interlat_paper.evaluate --hidden-data interlat_paper/data/qasper_oracle_validation.pt --checkpoint interlat_paper/checkpoints/qasper_oracle_qwen3b_v3_pure_latent/best.pt --output interlat_paper/output/qasper_oracle_qwen3b_v3_pure_latent.txt --num-samples 8 --include-text-control --mix-ratios 0.5 0.9 1.0
```

## Pure-Latent-Only Adaptation

Use this only after the mixed endpoint adaptation has failed. Every training and
validation message is pure latent (`r=1`); no Sender-plan text remains in the
Actor input. This is a QASPER-specific ablation, not the original paper
curriculum. Keep `qasper_oracle_qwen3b_v3` as the control checkpoint, but remove
the failed mixed-adaptation directory before this run if disk capacity is tight.

```bash
rm -rf interlat_paper/checkpoints/qasper_oracle_qwen3b_v3_pure_latent

python -m interlat_paper.train --train-hidden interlat_paper/data/qasper_oracle_train.pt --val-hidden interlat_paper/data/qasper_oracle_validation.pt --output-dir interlat_paper/checkpoints/qasper_oracle_qwen3b_v3_all_pure_latent --init-checkpoint interlat_paper/checkpoints/qasper_oracle_qwen3b_v3/best.pt --actor-model Qwen/Qwen2.5-3B-Instruct --epochs 6 --batch-size 1 --gradient-accumulation 16 --learning-rate 1e-5 --warmup-ratio 0.03 --gradient-checkpointing --pure-latent-start-epoch 1 --pure-latent-probability 1.0 --validation-replacement-rate 1.0 --early-stopping-patience 3
```

```bash
python -m interlat_paper.evaluate --hidden-data interlat_paper/data/qasper_oracle_validation.pt --checkpoint interlat_paper/checkpoints/qasper_oracle_qwen3b_v3_all_pure_latent/best.pt --output interlat_paper/output/qasper_oracle_qwen3b_v3_all_pure_latent.txt --num-samples 8 --include-text-control --mix-ratios 0.5 0.9 1.0
```
