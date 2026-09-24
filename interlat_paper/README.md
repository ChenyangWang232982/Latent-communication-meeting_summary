# Paper-Style Interlat

This directory is a separate implementation of Interlat's main training stage.
It does not reuse the adapter-only objective in `interlat_same_model`.

## What is implemented

- Frozen instruction-tuned Sender trajectories from `interlat_same_model.collect`.
- A trainable Qwen Base Actor plus the paper's MHA communication adapter.
- `<bop>` / `<eop>` token embeddings around the latent message.
- Curriculum mixing follows the released implementation: `r` is sampled from `{0.1, ..., 0.9}`, and the transmitted message is a latent prefix followed by the corresponding suffix of Sender-plan token embeddings.
- `L_task + w_contrast L_contrast + w_plan L_plan`, where `L_contrast = max(0, 0.69 - JS)` for a cross-task latent message and `L_plan` is KL plus cosine alignment to the text-plan output distribution. The two weights are dynamically derived from their current loss values, as in the authors' implementation.
- A separate ZeRO-3 CPU-offload configuration for 7B full-parameter training on one 48GB GPU.

The paper's compression reasoning model is intentionally not included in this stage. It is trained only after the full Actor has converged, with the Actor and adapter frozen.

## Main-stage command

First collect plans with an instruction-tuned Sender. Use the Oracle-evidence JSONL split for the controlled experiment:

```bash
python -m interlat_same_model.collect --data data/qasper_oracle_evidence/train.jsonl --output interlat_paper/data/qasper_oracle_train.pt --model Qwen/Qwen2.5-7B-Instruct --source-max-tokens 4096 --sender-max-new-tokens 1024
python -m interlat_same_model.collect --data data/qasper_oracle_evidence/validation.jsonl --output interlat_paper/data/qasper_oracle_validation.pt --model Qwen/Qwen2.5-7B-Instruct --source-max-tokens 4096 --sender-max-new-tokens 1024
```

For a 48GB GPU, install the updated environment and run the full Actor with ZeRO-2 and optimizer-only CPU offload:

```bash
python -m interlat_paper.train --train-hidden interlat_paper/data/qasper_oracle_train.pt --val-hidden interlat_paper/data/qasper_oracle_validation.pt --output-dir interlat_paper/checkpoints/qasper_oracle_qwen7b --actor-model Qwen/Qwen2.5-7B-Instruct --epochs 3 --batch-size 1 --gradient-accumulation 16 --learning-rate 1e-5 --warmup-ratio 0.03 --gradient-checkpointing --deepspeed-config interlat_paper/deepspeed_zero2_optimizer_offload.json
```

The authors use BF16 and FlashAttention 2. The command therefore requests `flash_attention_2`; install a CUDA-compatible `flash-attn` wheel on the Linux training machine before this run. If that cannot be installed, pass `--attention-implementation ""` for a slower compatibility run, which is no longer the exact performance setting.

The current ZeRO JSON sets DeepSpeed accumulation to one because the script owns the paper-style global-batch accumulation. It keeps model parameters on the GPU and offloads only optimizer states, avoiding the large CPU initialization peak of ZeRO-3 parameter offload.

`deepspeed_zero3_cpu.json` is kept as an alternative for machines with ample per-container CPU memory; it is not the recommended choice for the current hosted 48GB setup.

## Evaluation

ZeRO checkpoints must first be consolidated:

```bash
python -m interlat_paper.export_deepspeed --checkpoint-dir interlat_paper/checkpoints/qasper_oracle_qwen7b/deepspeed --metadata interlat_paper/checkpoints/qasper_oracle_qwen7b/best_metadata.pt --output interlat_paper/checkpoints/qasper_oracle_qwen7b/best.pt
python -m interlat_paper.evaluate --hidden-data interlat_paper/data/qasper_oracle_validation.pt --checkpoint interlat_paper/checkpoints/qasper_oracle_qwen7b/best.pt --output interlat_paper/output/qasper_oracle.txt --num-samples 8 --include-text-control
```
