# Global selected KV + transcript-part specialists

每个 specialist 的输入：**全文的 selected KV＋当前块原文＋分析要求及会议位置**。
这里是经路由选中的全局 KV，不是 full KV，也不是只发送当前块的 KV。

```text
Full transcript → one source KV prefill → attention + optional vector routing
                                              ↓
                                      global selected KV
                                              ↓ copied independently for each stage
Original transcript → chronological parts → specialist (global selected KV + current part + location)
                                              ↓ ordered local text reports
                    final synthesis (global selected KV + reports; optional thinking)
```

Parts cover the original full transcript, including turns omitted from selected KV.
Adjacent turns by the same speaker remain together. `--specialist-parts 8` targets
1/8 of original transcript tokens per part; `--specialist-overlap 0.25` targets
1/4 overlap at speaker boundaries. Long turns stay intact, so actual part counts
may exceed eight. There is no fixed output-item or generation-token limit.

Specialists are told their part number and total, original speaker-block range,
and approximate percentage through the transcript by text length. This percentage
is not an elapsed meeting time. They analyze the current part and use global
context for relevant references, conditions and later revisions. They must not
repeat unrelated global topics in every report.

Each stage receives a disposable copy of the same selected source KV. Source KV
is immutable; model weights are shared and agents execute sequentially. Reports
are natural language, not concatenated generated KV. Final synthesis receives
the same global selected KV plus local reports in original order. No separate
review pass is added in this mode. The full source KV remains resident.

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
  --summary-thinking \
  --seed 12 \
  --review-seed 42 \
  --review-temperature 0.3 \
  --review-top-p 0.9
```

`--review-*` controls final synthesis here. Local seeds are `seed + part_index`.
`--summary-thinking` enables thinking only at final synthesis; local specialists
remain non-thinking. The final answer excludes reasoning, which is saved separately.
Context exhaustion raises an error instead of silently truncating input.

`experiment_1 --chunk-specialists` also supplies a text control: the global selected
context is rendered as verbatim text, while current-part original text, location,
instructions and decoding settings match the KV path. Final synthesis receives
global selected text plus its local reports. This compares workflows; contextualized
source KV and freshly encoded selected text have different positional histories.

`.specialists.json` saves `shared_selected_positions`, original part text and its
coordinates, ordered reports, stage tokens and timings. Window `positions` describe
the current original part; they are not the subset of KV sent to that specialist.
Every stage's `source_kv_positions` counts the full shared selected KV.
Live partial/status files are saved per stage.

Logical model-token totals include source prefill/probes once per standalone
workflow, every local prompt (including current-part text) and output, and final
synthesis prompt/output including thinking. KV rows are reused rather than counted
as new input tokens. Embedding encoder tokens are separate. Logical payload sums
stage KV tensors or global evidence UTF-8 bytes, current-part UTF-8 bytes and report
UTF-8 bytes. These are not measured network traffic or provider billing.

```bash
python -m unittest discover -s chunk_specialist_workflow/tests
```
