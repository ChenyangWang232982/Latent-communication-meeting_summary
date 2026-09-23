# Training-Free CIPHER Baseline

This folder is intentionally independent from `llm_latent/`. It implements a
same-model, no-training CIPHER baseline using the long-context decoder-only
instruction model `Qwen/Qwen2.5-1.5B-Instruct`.

## Method boundary

For each QA example:

1. Sender reads `context + question` through its chat template and greedily
   creates an internal draft.
2. The final hidden state for each Sender decoding step goes through the
   Sender's original `lm_head`.
3. Softmax vocabulary probabilities weight the Receiver's original input
   embedding table.
4. These continuous embeddings are prepended to the Receiver's causal prompt.
5. Receiver generates an answer.

The Sender draft is saved only for inspection. It is never supplied to the
Receiver as token IDs or text. There is no adapter, projector, compressor,
learned query, optimizer, or checkpoint.

## Run

Prepare QASPER first with the existing script, then run the validation split:

```bash
python -m cipher_training_free.run \
  --data data/qasper_latent/validation.jsonl \
  --num-samples 8
```

The Hugging Face model downloads automatically on first run. Results are saved
under `cipher_training_free/output/`.

## Controls

Use `--sender-max-new-tokens` to control latent-message length. Start at 128.
`--source-max-tokens 4096` is the sender's long-context budget. Both agents are
identical copies of the selected decoder-only model and stay in evaluation mode.
