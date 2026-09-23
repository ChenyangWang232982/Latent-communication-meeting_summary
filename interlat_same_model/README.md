# Same-Model Interlat for Long Text

This is an independent implementation of the learned **Interlat** method from
ACL 2026, adapted to the existing long-context QASPER JSONL data. It is not a
training-free CIPHER variant.

```text
long context + question
  -> frozen Qwen Sender generates an internal evidence/reasoning plan
  -> Sender final-layer states, one vector per generated plan token
  -> <bop> + learned latent integration (+ optional compression) + <eop>
  -> same-family Qwen Receiver
  -> answer / meeting-summary target
```

The Receiver never receives the Sender plan as text. It only sees its own
question prompt and the continuous hidden-state sequence.

## Paper Correspondence

- **Latent message:** Sender final-layer hidden states from its generated plan.
- **Conditional separation:** the Receiver has only its task prompt; `<bop>` and
  `<eop>` delimit the inserted latent trajectory.
- **Training:** task cross-entropy plus plan-alignment and shuffled-plan
  contrastive losses.
- **Same model:** `Qwen/Qwen2.5-1.5B-Instruct` is used for both roles, so the
  dimensional projection is an identity mapping. The receiver-side attention
  integration module is still learned.
- **Compression:** `--compressed-latent-len K` replaces a full trajectory with
  K learned attention-pooled vectors. Start with `0` (uncompressed) for the
  first experiment.

The original paper evaluates ALFWorld and MATH; this folder adapts its
communication mechanism, rather than claiming the paper evaluated QASPER or
meeting summarization.

## Data

The existing prepared long-text split is used directly:

```text
data/qasper_latent/train.jsonl
data/qasper_latent/validation.jsonl
```

Each row must provide `id`, `context`, `question`, and `answer`.

## 1. Collect Frozen Sender Trajectories

Run this once per split. The output contains only the generated Sender plan and
its last-layer states; it can be several hundred MB for a full split.

```powershell
python -m interlat_same_model.collect --data data/qasper_latent/train.jsonl --output interlat_same_model/data/qasper_train_hidden.pt --model Qwen/Qwen2.5-1.5B-Instruct

python -m interlat_same_model.collect --data data/qasper_latent/validation.jsonl --output interlat_same_model/data/qasper_validation_hidden.pt --model Qwen/Qwen2.5-1.5B-Instruct
```

Use `--limit 30` for a pipeline smoke test. `--source-max-tokens 4096` and
`--sender-max-new-tokens 256` are the defaults.

## 2. Train the Receiver-Side Interlat Module

Start uncompressed and train only the latent module plus the two boundary-token
embeddings. This is the least expensive first check. `best.pt` is selected by
validation task loss, not total loss, because the latent-alignment auxiliary
loss can improve while answer quality worsens.

```powershell
python -m interlat_same_model.train --train-hidden interlat_same_model/data/qasper_train_hidden.pt --val-hidden interlat_same_model/data/qasper_validation_hidden.pt --output-dir interlat_same_model/checkpoints/qasper_qwen15b --epochs 10 --batch-size 1 --gradient-accumulation 8
```

For a more paper-like, higher-cost run that also updates the Receiver:

```powershell
python -m interlat_same_model.train --train-hidden interlat_same_model/data/qasper_train_hidden.pt --val-hidden interlat_same_model/data/qasper_validation_hidden.pt --output-dir interlat_same_model/checkpoints/qasper_qwen15b_full --epochs 10 --batch-size 1 --gradient-accumulation 8 --unfreeze-receiver
```

Try learned compression only after the uncompressed channel produces coherent
answers:

```powershell
python -m interlat_same_model.train --train-hidden interlat_same_model/data/qasper_train_hidden.pt --val-hidden interlat_same_model/data/qasper_validation_hidden.pt --output-dir interlat_same_model/checkpoints/qasper_qwen15b_k32 --compressed-latent-len 32
```

## 3. Latent-Only Evaluation

The evaluation command deliberately has no `context` or Sender plan argument.
It gives the Receiver only stored hidden states and the question.

```powershell
python -m interlat_same_model.evaluate --hidden-data interlat_same_model/data/qasper_validation_hidden.pt --checkpoint interlat_same_model/checkpoints/qasper_qwen15b/best.pt --output interlat_same_model/output/qasper_validation.txt --num-samples 8
```

## Checks

```powershell
python -m unittest interlat_same_model.test_latent -v
```
