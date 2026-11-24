# Three AMI endpoint cases / 三条 AMI 首尾关联案例

This is a small authored diagnostic pilot, not a real-meeting benchmark or a
proof of attention robustness. It keeps each original AMI transcript intact and
adds exactly one opening turn and one closing turn. The same target arrangement
is referenced at those two endpoints only. The closing turn uses a related phrase
rather than repeating the topic verbatim. The expected status is supported by
explicit wording; reviewers should not infer consensus merely from position.

这是三条人工插入案例，不是原始会议事实。原文一字不改，只在首尾各加一条发言。
开头和结尾通过语义回指关联，中间不添加该话题，也不在模型要求中提示关注它。
该设计先测试首尾跨窗口关联；首尾本来可能较易被模型关注，因此不能由此证明解决了
中间位置遗忘或普遍注意力问题。三个案例同样不支持统计泛化结论。

| Original AMI ID in current input | Inserted arrangement | Final target |
| --- | --- | --- |
| 82 | Configuration for a six-unit loan batch | Charging dock replaces the earlier disposable-cell/no-dock proposal; scope is that batch |
| 38 | Owner of the return kit for two evaluation units | Industrial Designer replaces Marketing; main design roles are unchanged |
| 101 | Three-person hallway trial gated on consent | Forms are still missing; trial not authorized and date unconfirmed |

IDs above describe the first three records in the existing AMI export; preparation
uses input order rather than hardcoding those IDs. It does not select sources
based on which method previously performed best.

## Prepare / 生成案例

Run once in the server project root:

```bash
python -m experiment_2.challenge_samples.prepare \
  --input input/experiment_1_unfiltered_cap100/ami.jsonl \
  --output-dir input/ami_endpoint_challenge_3
```

Alternatively, use the original experiment checkpoints with `--source-run` instead
of `--input`. This is how the local pilot was constructed from the supplied files:

```bash
python -m experiment_2.challenge_samples.prepare \
  --source-run test_result/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986 \
  --output-dir input/ami_endpoint_challenge_3
```

The output contains three readable `.txt` files, inference-only `meetings.jsonl`,
`manifest.json`, and separate `answer_key.json`/`ANSWER_KEY.md`. Existing dataset
directories are not overwritten. Inspect the text and key before inference. If a
case needs revision, freeze a new version in a new directory before formal tests.
The original reference is retained only in the key for provenance; it does not
describe the additions and must not be used as a complete reference for this test.

输出在根目录 `input/ami_endpoint_challenge_3/`。不要把答案文件或原参考摘要放入 prompt。
若本地生成目录未同步（`input/` 默认被 Git 忽略），在服务器运行上述生成指令即可。

## Run all three conditions / 三种条件

Run after syncing code; the generation model loads once per condition, processing
all three meetings. Target-specific instructions are **not** added.

```bash
conda activate latent-agents

for condition in direct_text attention_kv semantic_kv; do
  python -m experiment_2.challenge_samples.run \
    --dataset input/ami_endpoint_challenge_3 \
    --condition "$condition" \
    --output-dir test_result/ami_endpoint_challenge_3 \
    --model Qwen/Qwen3-8B \
    --device cuda \
    --seed 12 \
    --summary-seed 42 \
    --summary-temperature 0.3 \
    --summary-top-p 0.9 \
    --summary-thinking \
    --role-budget-ratio 0.25 \
    --speaker-neighbor-blocks 1
done
```

`--condition` determines whether semantic routing runs; no separate
`--vector-routing` flag is needed. Other workflow defaults match the preceding
experiments. This pilot measures target behavior, not comparative efficiency.

## Inspect / 检查结果

```bash
find test_result/ami_endpoint_challenge_3 -name summary.txt -print -exec cat {} \;
```

Each condition saves:

- `manifest.json` and `status.json` — settings and actual completion counts.
- `00000_<id>/summary.txt` — final summary, without the thinking text.
- `00000_<id>/result.json` — raw result including stage measurements/reasoning.
- `00000_<id>/specialists.json` — local notes/windows for KV conditions.
- `00000_<id>/coverage.json` — explicit opening/closing KV-row retention and local
  window membership. It is not itself a factual-correctness score.
- `manual_review.csv` — expected result, source coverage, and blank verdict/notes.

Fill verdicts manually as `correct`, `omitted`, `incorrect`, or `unclear`. A vague
statement without enough information to identify the target should not receive
full credit. Record missing conditions or scope errors in notes. No LLM judge or
automatic lexical correctness classifier is used, and no ratings are invented.

人工判断重点：最终状态是否正确，是否把局部批次决策泛化成主产品决策，是否混淆
责任人，是否保留未满足的条件。先看状态和原文依据，再填表；不能只看是否出现关键词。

For KV conditions, coverage maps the fixed character spans to actual source
token positions. Full retention means every overlapping non-protected source
token is selected; partial retention is reported separately. Final synthesis
receives the global selected positions, while local agents receive windows.
Retained vectors may encode contextual influences from the full prefill, so a
dropped explicit row is not proof of complete semantic absence. Direct text has
no selector: coverage is marked not applicable, not a routing success score.

两条关键发言都在 KV 中、但摘要仍错误，说明错误发生在后续理解或生成链路；
若关键修正行没有保留，则存在明确行级的选择遗漏，但不能由此断言原因只有选择。
即使三例全部成功，也不能据此声称真实会议事实准确率为 100%。
