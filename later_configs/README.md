# LaTER Single-GPU Configs

`sft_config_qwen3_4b_stage1_smoke.yaml` is a 50-step feasibility run for one
48 GB GPU. It is not an official LaTER reproduction: it uses Qwen3-4B, a 2048
token limit, 256 training records, and stage 1 only. Stage 1 freezes the
transformer blocks and trains the latent projector, input embeddings, and LM
head; stage 2 full-parameter training is disabled.

Copy it into the external LaTER checkout before running:

```bash
cp later_configs/sft_config_qwen3_4b_stage1_smoke.yaml \
  external/LaTER/later/src/config/
```

From `external/LaTER`, first validate the data path and collator:

```bash
PYTHONPATH=$PWD python -m later.src.train.train \
  --config later/src/config/sft_config_qwen3_4b_stage1_smoke.yaml \
  --dry_run
```

Only after a successful dry run, start the 50-step smoke training:

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc_per_node=1 \
  -m later.src.train.train \
  --config later/src/config/sft_config_qwen3_4b_stage1_smoke.yaml
```
