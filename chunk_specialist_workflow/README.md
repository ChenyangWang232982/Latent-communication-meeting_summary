# Selected-KV window specialists / 选中 KV 分块专家

```text
全文 prefill 一次 → 原有五类 selector + 向量辅助 + 邻块扩张
 → selected KV → 完整发言块组成的重叠窗口
 → 每窗口一个独立 specialist → 局部自然语言报告
 → 原 selected KV + 按原文顺序排列的报告 → 最终摘要
```

Source framing and routing are unchanged. No extra anchors, fact-reference lists,
topic discovery, JSON fact extraction or FRAME stages remain. The immutable source
cache is reused; each specialist gets a separate writable copy of only its selected
window. Agents share model weights and execute sequentially to avoid concurrent
backend/RNG interference; independent agents do not imply parallel GPU execution.
Local reports are unverified observations, not final meeting-wide conclusions.
The final agent also reads the original selected KV to interpret revisions and conditions.
This is hybrid communication: source context uses KV, local reports use text.
Generated specialist KV is not concatenated at synthesis.

`--specialist-parts 8` targets a window of 1/8 of selected source tokens, excluding
protected framing. `--specialist-overlap 0.25` targets 1/4 overlap. Speaker boundaries
make sizes approximate; long turns stay intact and total windows can exceed eight.
No unselected evidence is restored. Absolute source positions and chronological order
are preserved. Overlapping reports are deduplicated semantically by synthesis, not
discarded by string matching. Full source KV stays resident; no peak-memory claim.

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/chunk_specialists/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --variants routed_kv \
  --chunk-specialists \
  --specialist-parts 8 \
  --specialist-overlap 0.25 \
  --specialist-temperature 0.3 \
  --specialist-top-p 0.9 \
  --specialist-repetition-penalty 1.1 \
  --seed 12 \
  --review-seed 42 \
  --review-temperature 0.3 \
  --review-top-p 0.9
```

`--specialist-*` controls local generation; `--review-*` controls final synthesis.
Each local seed is `seed + window_index`. `--summary-thinking` optionally enables
Qwen thinking ONLY for final synthesis, retaining the same selected KV and reports;
local specialists stay non-thinking. Default is off for comparison. Only the answer
after `</think>` becomes the final summary. Reasoning is saved separately in
`.chunk_summary.thinking.txt` and `thinking_outputs` in the specialists artifact.
Stage metrics split reasoning, delimiter, answer and terminal tokens; all generated
tokens remain included in output totals. No fixed output-token/item
limit is introduced; context exhaustion raises an error. `--review` does not add
another review pass in this mode. Without `--chunk-specialists`, the original
workflow still runs. Paired evaluation with specialist_text is not registered for
this mode, so do not call it a representation-only comparison.

Outputs include the final summary, `.specialists.json` with ordered reports/window
coordinates/stage metrics, and `.chunk_specialist_000.partial.txt` plus `.status.json`
for live generation (also for subsequent windows and `chunk_summary`). Global metrics
count source prefill and probes once, all specialist prompt/output events, and final
summary prompt/output. Reused KV positions are reported separately, not re-counted
as prefill tokens. Logical payload sums all KV handoffs and serialized text reports;
this is not measured network traffic or provider billing. Embedding tokens stay separate.

```bash
python -m unittest discover -s chunk_specialist_workflow/tests
```
