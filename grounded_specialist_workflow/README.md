# Topic specialist workflow / 动态话题工作流

入口名称仍为 `grounded_specialist_workflow`。流程改为：

```text
全文 + 发言块编号 → 一次原文 KV prefill → 全文话题发现
                                            ↓
                         每话题 selector：注意力 + 语义匹配
                                            ↓
                      同组话题 KV 取并集、去重、按原文顺序排列
                                            ↓
                     联合分析 specialist（默认全部话题一组）→ 结构化报告
                                            ↓
                                    最终摘要 agent
```

Topic discovery reads the complete source KV and returns topic titles/scopes in
first-appearance order. Each topic becomes a selector query, not a fixed action or
decision role. Topic and source-block embeddings are computed in one encoder call;
source vectors are reused. Attention and topic similarity rank complete speaker
blocks; neighbor expansion follows. Topics may share blocks. All selected positions
retain original order and absolute positions within the annotated source prompt.

`[B0]`, `[B1]`, ... markers are added after speaker headers before prefill, so the
model can identify block order. Original words remain intact. This changes prompt
tokens relative to the baseline and must be accounted for in comparisons.

No fixed topic count. The discovery model determines topic granularity and can
still miss or split topics; there is no recursive topic creation or review stage.
By default, one specialist receives ALL discovered topics and the chronological,
deduplicated union of their selected KV positions. Each topic retains an independent
selector. `--topics-per-specialist N` optionally batches N topics in discovery order
(not semantic clusters); 1 reproduces individual-topic scheduling. Reports are held in memory and saved, then merged in discovery
order regardless of completion order. Final synthesis receives structured text.
This is a hybrid workflow, not KV-only communication or independently verified output.

Specialist prompts request key facts (including explicit dates and full time ranges),
one discussion paragraph per topic, distinct decisions, concrete
actions (content, explicit owners and deadline), distinct open issues and a brief
final status. Repeated turns are synthesized rather than enumerated. Missing owners
use `[]` and missing deadlines use `null`; speaking does not imply task ownership.
Per-item IDs and block citations are omitted; routing artifacts retain source blocks.
These are prompt instructions, not additional format checks or item/token limits.

## 使用 / Run

```bash
conda activate latent-agents
python -m grounded_specialist_workflow \
  --transcript input/meeting1.txt \
  --output output/topic_specialists/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda \
  --topic-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --topics-per-specialist 0 \
  --specialist-workers 1 \
  --seed 12 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repetition-penalty 1.1
```

- `--topic-budget-ratio`: initial positions per topic; neighbor expansion can exceed it.
  Reuses the baseline config's internal `role_budget_ratio` field. The old CLI
  `--role-budget-ratio` has been replaced in this entry point only.
- `--topics-per-specialist`: 0 sends all topics together; positive values set group size.
  The union can approach the full source length; this change does not guarantee cache savings.
- `--specialist-workers`: parallel groups, not the number of topics. With one group,
  only one specialist runs regardless of worker count. One shared
  model, independent writable KV and RNG per task; more workers need more memory.
- `--routing-recency-weight`: optional initial ranking position bias, default 0.
  Later statements do not automatically override earlier decisions.
- `--specialist-thinking`: enables Qwen thinking only for specialist analysis.
  Discovery and final synthesis remain non-thinking. Default is off. To compare,
  keep all other arguments identical and use separate output paths.
  `specialist_group_*.thinking.txt` saves reasoning; `.raw.txt` preserves raw output.
  Only the answer after `</think>` is parsed and sent to synthesis. Stage metrics
  separate thinking, delimiter, answer and terminal tokens; output totals include all.
- Generation uses default model precision. No fixed output-item
  or token cap; EOS/model context ends generation. Oversize context raises an error.

Selectors execute before parallel generation because attention scoring temporarily
changes the model backend. This keeps the shared model safe from concurrent backend
changes. Worker threads may not yield GPU speedup; measure wall time.

## 输出 / Outputs

`--output` is the final summary. `<stem>.artifacts/<timestamp>/` contains:

- `topics.json`, `topic_discovery.txt`: discovered topics and raw discovery output.
- `source_blocks.json`, `routing.json`: source text/coordinates and per-topic selections.
- `groups.json`: group membership and deduplicated source positions/block IDs.
- `specialist_group_*.input.txt`, `specialist_group_*.txt`, `report_group_*.json`: prompts and reports.
  Reports use a `reports` array with one entry per assigned topic; source coordinates
  remain in routing/group artifacts rather than the final synthesis prompt.
- `topic_reports.json`: all topic reports in discovery order, ready for synthesis.
- `summary.input.txt`, `summary.txt`, `*.partial.txt`: prompts/results and live output.
- `config.json`, `stages.json`, `metrics.json`, `status.json`: settings and accounting.

There are G+2 generation calls for G groups: discovery, G specialists, synthesis.
The default uses three calls when topics are present. All N selectors still run.
All are included in decoder token totals; embedding tokens are separate. Concurrent
stage times overlap; use pipeline wall time, excluding model loading. Source block
IDs in routing/group artifacts indicate supplied context, not verified claim-level citations.
Only basic JSON parsing and model-context checks remain; no content verdict filter.

```bash
python -m unittest discover -s grounded_specialist_workflow/tests
```

Tests cover variable topic counts, ordered KV selections/reports, cache and RNG
isolation, block markers, accounting and failure audit. Actual GPU quality requires
running the example command. Existing baseline/Experiment 1 entry points are separate.
