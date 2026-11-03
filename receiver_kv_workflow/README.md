# Receiver-Conditioned KV Meeting Workflow

This is a training-free meeting-summarization workflow. It compares two matched
communication representations under the same model, transcript, role queries,
speaker-block routing, output budget, and final summary prompt.

```text
speaker-labelled transcript
  -> one full-source KV prefill
  -> role-conditioned attention scoring
  -> complete seed speaker blocks + adjacent speaker context
  -> routed_kv: selected source KV -> receiver
     specialist_text: same source evidence -> text sender -> receiver
  -> meeting summary (and optional timeline)
```

`routed_kv` is the main method: selected source K/V tensors are supplied to the
receiver without decoding them to text. `specialist_text` is the natural-language
control: it receives the same routed evidence, but a sender writes one handoff
before the receiver summarizes it.

The workflow requires line-start speaker labels such as `Alice: ...`,
`[10:30] Alice: ...`, or `Alice (Team)  1:10:30`. Consecutive turns from the
same speaker become one block. The router selects complete seed blocks and then
keeps `--speaker-neighbor-blocks` blocks on both sides.

## Run

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --model Qwen/Qwen3-4B \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --variants routed_kv specialist_text
```

Use `--output output/meeting1.txt` to place `routed_kv` at that exact path;
the text baseline then uses a sibling `meeting1.specialist_text.txt` file.
Use `--timeline` for a separately generated chronology and `--routing-audit`
for an offline source-coordinate audit. Neither audit text nor dropped source
text is passed to the receiver. `--timeline` and `--decision-evolution` each
use the selected source KV and the generated `routed_kv` summary, and write
separate `.timeline.txt` and `.decisions.txt` files. They require `routed_kv`.
Their generation costs are recorded separately from summary-pipeline totals;
sharing context does not guarantee consistency.

## Important limits

- The full transcript is prefetched once and is never silently truncated.
  Generation stops at EOS or when the remaining model context is exhausted;
  the run fails if the prompt leaves no output room.
- It supports dense full-attention Qwen3 models with standard RoPE only.
- Smaller receiver KV does not imply lower peak GPU memory: the full source KV
  remains resident while routing and generation occur.
- KV positions, KV bytes, logical token events, and runtime are reported as
  separate quantities; they are not interchangeable measures of cost.

## Tests

```bash
python -m unittest receiver_kv_workflow.test_workflow receiver_kv_workflow.test_speaker_blocks -v
```
