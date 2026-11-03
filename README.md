# 基于 KV 通信的会议摘要 / Meeting Summarization with KV Communication

[中文](#中文) | [English](#english)

## 中文

本项目使用无需额外训练的 receiver-conditioned KV routing 生成会议摘要，并与自然语言通信方式进行对照。输入是带说话人标签的会议转录文本；输出包括摘要、可选时间线以及评测指标。

```text
会议转录 → 全文 KV prefill → 角色查询与 speaker block 选择
                                  ├─ routed_kv：选中的 KV → 摘要模型
                                  └─ specialist_text：相同证据 → 文本 handoff → 摘要模型
```

`routed_kv` 是主方法，直接把选中的原文 KV 缓存交给摘要接收方；`specialist_text` 是自然语言对照，使用相同的路由证据，先由模型生成文本 handoff，再交给摘要接收方。文本对照因此包含额外的生成步骤。两种方式使用相同模型、源数据和最终摘要要求，但不能认为两条路径的计算完全相同。

## 目录

```text
input/                  输入转录和已准备的评测数据
output/                 会议摘要和可选 audit
test_result/            评测报告和运行指标
receiver_kv_workflow/   摘要工作流
data_preprocess/        评测数据预处理
meeting_evaluation/     评测入口、指标与报告
requirements.txt        pip 依赖
requirements.yml        Conda 环境配置
```

## 环境准备

在项目根目录执行。下面的多行示例使用 Linux / Bash；PowerShell 请使用反引号续行，或将命令写成一行。

```bash
conda create -n latent-agents python=3.11 -y
conda activate latent-agents
pip install -r requirements.txt
```

也可以使用 `conda env create -f requirements.yml` 创建环境。两份配置使用的 PyTorch / CUDA wheel 版本不同：`requirements.txt` 为 CUDA 12.8，`requirements.yml` 为 CUDA 12.6；按机器环境选择其中一种。默认模型是 `Qwen/Qwen3-4B`，首次运行需能访问模型文件。

## 输入格式

### 单场会议：UTF-8 TXT

将会议转录保存到例如 `input/meeting1.txt`。每次换人发言时在行首标出说话人，冒号后留空格：

```text
Alice: 我们先确认项目进度。
Alice: 目前设计方案已经完成。
Bob: 我会在周五之前发送测试报告。
Alice: 下周二再讨论测试结果。
```

也可以使用 `Speaker_0: ...` 等匿名标签。连续的相同说话人会被合并成一个 speaker block，再根据相关性选择。当前程序不会从音频自动识别说话人；纯音频或无说话人标签的文本需要先在外部转录并分离说话人。

### 批量评测：UTF-8 JSONL

一行一个 JSON 对象，`transcript` 内用 `\n` 分隔发言轮次。例如：

```jsonl
{"id":"meeting_001","transcript":"Alice: 项目进度如何？\nBob: 我会在周五发送报告。","summary":"Bob 将于周五发送报告。"}
```

- `id`：样本 ID，建议唯一。
- `transcript`：带说话人标签的会议文本；也可使用 `context` 字段。
- `summary`：人工参考摘要；也可使用 `answer`。缺少参考摘要时无法计算有效的参考摘要 ROUGE-L。
- `question`：可选字段。当前工作流使用固定的会议摘要请求，该字段不会改变路由或摘要任务。

目前内置 ROUGE-L 面向英文分词，不适合直接评价中文摘要。

## 生成会议摘要

只生成 KV 通信摘要：

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/meeting1.txt \
  --model Qwen/Qwen3-4B \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --variants routed_kv \
  --device cuda
```

输出摘要为 `output/meeting1.txt`，运行指标位于 `test_result/receiver_kv/<时间戳>/metrics.txt` 和 `metrics.json`。不写 `--output` 时，摘要默认写入 `output/receiver_kv/<时间戳>/`。

要同时输出自然语言对照，将最后的 `--variants routed_kv` 改成：

```text
--variants routed_kv specialist_text
```

此时自然语言版本写入 `output/meeting1.specialist_text.txt`。也可以添加 `--routing-audit` 输出被选中的原文范围等审计文件。审计文件可能包含会议原文，请按原文的敏感程度保存。

添加 `--review` 后，同一模型先仅根据初稿合并重复行动项，再使用 Qwen3 embedding 按整理后的条目检索原文发言块，将选中 KV 与整理稿交给模型核查。每条取最相近的两个块，按 `--speaker-neighbor-blocks` 扩展并合并重叠位置。`--output output/meeting1.txt` 保留初稿，另外输出 `meeting1.reviewed.txt`、`meeting1.reviewed.organized.txt` 和 `meeting1.reviewed.changes.txt`。变更记录包含文本差异及候选原文片段，不代表逐条已核验的引用。未启用 `--vector-routing` 时审校也会加载 `--embedding-model`。若任一步截断或缺少初稿中已有的标准章节，则跳过审校并保留初稿；章节检查不保证章内每一项完整。

证据核查阶段要求模型说明每处修改的理由和证据依据；程序不再要求 JSON 或逐条匹配格式。能识别的理由会写入变更记录，否则记录未识别到独立理由。理由和依据由模型生成，并未独立验证，也不能保证模型为每处修改提供理由；第一步整理的变化仍以文本差异记录。

默认关闭思考模式；添加 `--enable-thinking` 仅对最终摘要模型开启。两次对照请使用相同输入、参数和随机种子，并设置不同的 `--output`。思考内容不会写入摘要，运行指标单独记录 thinking/answer token；若思考未结束，摘要为空并标记不完整。思考和答案共用模型剩余上下文空间。

使用 `--seed 7` 指定摘要种子，`--review-seed 42` 单独指定审校种子，二者默认均为 7。当前生成使用贪心解码，通常改变种子不会改变输出；不同 GPU、依赖版本等仍可能影响结果。评测命令的 `--seed` 默认也为 7。



## 评测

单个数据集评测会比较 `routed_kv` 和 `specialist_text`，报告摘要质量和通信效率：

```bash
python -m meeting_evaluation.test \
  --benchmark meetingbank \
  --input input/meetingbank_speaker_test.jsonl \
  --output-dir test_result \
  --num-samples 25 \
  --model Qwen/Qwen3-4B \
  --receiver-kv-role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --device cuda
```

自有 JSONL 也可以作为 `--input`；`--benchmark` 用于标记报告名称，可选 `ami`、`icsi`、`qmsum_committee`、`meetingbank`。`--num-samples 0` 表示使用全部记录。只需要 token、缓存和耗时指标时，加 `--communication-metrics-only`；程序仍会生成摘要来统计输出 token，但不保存摘要或计算质量分数。

报告保存在 `test_result/YYYYMMDD_HHMM/`，同一 benchmark 生成 `.txt` 和 `.json`。单次评测若在同一分钟重复运行同一 benchmark，可能覆盖同名报告。

四个预处理后的数据集可以一次运行：

```bash
python -m meeting_evaluation.run_all \
  --num-samples 25 \
  --speaker-neighbor-blocks 1 \
  --receiver-kv-role-budget-ratio 0.25 \
  --model Qwen/Qwen3-4B \
  --device cuda
```

默认寻找 `input/ami_test.jsonl`、`input/icsi_test.jsonl`、`input/qmsum_committee_test.jsonl` 和 `input/meetingbank_speaker_test.jsonl`。加 `--skip-missing` 可跳过缺失的数据集。准备方法见 [data_preprocess/README.md](data_preprocess/README.md)：AMI、ICSI、Committee 来自 QMSum 的不同领域，MeetingBank 使用官方带说话人转录的议程条目。QMSum 的 `all` 目录抽样不应称为官方 held-out test 结果。

## 参数与结果口径

| 参数 | 作用 |
| --- | --- |
| `--role-budget-ratio` / `--receiver-kv-role-budget-ratio` | 每个角色查询用于初步选择 KV 位置的比例。角色结果会取并集，之后还会扩张邻居块，因此最终保留比例可能大于该值。 |
| `--speaker-neighbor-blocks` | 选中的 speaker block 前后各保留多少个相邻 block，默认 1；0 表示不扩张。 |
| `--span-tokens` / `--receiver-kv-span-tokens` | 角色查询初步选择连续 token 位置时的 span 大小，默认 32；最终选择仍以完整 speaker block 为单位。 |
| `--query-batch-tokens` / `--receiver-kv-query-batch-tokens` | 注意力查询计算的批次大小，默认 8。 |

ROUGE-L 是与参考摘要的词序重叠指标，不等于事实准确率；source-support 是词汇重叠代理指标，不是人工事实核查。KV positions、KV 字节、logical model-token events、实际耗时是不同口径，不能直接互换。

## 当前边界

全文会先做一次 prefill；每次生成一直运行到模型输出结束标记或用尽模型上下文的剩余容量。如果输入已没有生成空间，程序会报错，不会自动分块总结或静默截断。`cap hits` 表示生成达到上下文边界；没有人为输出上限时，耗时和显存占用可能增加。当前 cache 适配器针对标准 RoPE、全注意力的 Qwen3 模型；全文 KV 在路由期间仍会占用显存，接收端 KV 变小不等于整个进程的峰值显存同比下降。

更多实现说明见 [receiver_kv_workflow/README.md](receiver_kv_workflow/README.md) 和 [meeting_evaluation/README.md](meeting_evaluation/README.md)。

---

## English

This project uses training-free receiver-conditioned KV routing to generate meeting summaries and compare KV communication with a natural-language handoff. It takes a speaker-labelled transcript as input and produces summaries and evaluation metrics.

```text
transcript → full-source KV prefill → role queries and speaker-block selection
                                      ├─ routed_kv: selected KV → summary receiver
                                      └─ specialist_text: selected evidence → text handoff → summary receiver
```

`routed_kv` sends the selected source KV cache directly to the receiver. `specialist_text` uses the same routed evidence but first generates a natural-language handoff for the receiver. The variants use the same model, source, and final summary requirements; the text variant includes an additional generation step, so their computation is not identical.

### Project layout

```text
input/                  Transcripts and prepared evaluation data
output/                 Summaries and optional audit
test_result/            Evaluation reports and runtime metrics
receiver_kv_workflow/   Summarization workflow
data_preprocess/        Benchmark preparation
meeting_evaluation/     Evaluation entry points, metrics, and reports
requirements.txt        pip dependencies
requirements.yml        Conda environment configuration
```

### Environment setup

Run the commands from the project root. Multiline examples use Linux / Bash; in PowerShell, use backticks for line continuation or enter the command on one line.

```bash
conda create -n latent-agents python=3.11 -y
conda activate latent-agents
pip install -r requirements.txt
```

Alternatively, create the environment with `conda env create -f requirements.yml`. The files specify different PyTorch CUDA wheels: CUDA 12.8 in `requirements.txt` and CUDA 12.6 in `requirements.yml`. Choose the one appropriate for your machine. The default model is `Qwen/Qwen3-4B`; its files must be accessible on the first run.

### Input formats

#### One meeting: UTF-8 TXT

Save the transcript as, for example, `input/meeting1.txt`. Start each new speaker turn with a label followed by a colon and a space:

```text
Alice: Let's check the project status.
Alice: The design proposal is complete.
Bob: I will send the test report by Friday.
Alice: We will discuss the results next Tuesday.
```

Anonymous labels such as `Speaker_0: ...` are also accepted. Consecutive turns from the same speaker are grouped into one speaker block, and blocks are selected by relevance. The program does not perform speech recognition or speaker diarization: audio and unlabelled transcripts must be transcribed and labelled externally first.

#### Batch evaluation: UTF-8 JSONL

Use one JSON object per line and separate speaker turns within `transcript` with `\n`:

```jsonl
{"id":"meeting_001","transcript":"Alice: How is the project?\nBob: I will send the report on Friday.","summary":"Bob will send the report on Friday."}
```

- `id`: sample identifier; unique IDs are recommended.
- `transcript`: speaker-labelled transcript; `context` is also accepted.
- `summary`: human reference summary; `answer` is also accepted. A meaningful reference-based ROUGE-L score requires a reference summary.
- `question`: optional field. The current workflow uses a fixed meeting-summary request; this field does not change routing or the summary task.

The built-in ROUGE-L implementation uses English-oriented tokenization and is not suitable for directly evaluating Chinese summaries.

### Generate a meeting summary

Generate only the KV-communication summary:

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/meeting1.txt \
  --model Qwen/Qwen3-4B \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --variants routed_kv \
  --device cuda
```

The summary is written to `output/meeting1.txt`. Runtime metrics are written to `test_result/receiver_kv/<timestamp>/metrics.txt` and `metrics.json`. Without `--output`, summaries go to `output/receiver_kv/<timestamp>/`.

To generate the natural-language control as well, replace `--variants routed_kv` with:

```text
--variants routed_kv specialist_text
```

The text control is then written to `output/meeting1.specialist_text.txt`. Add `--routing-audit` to save selected source ranges and other audit information. Audit files may contain transcript text, so protect them as you would the original transcript.

With `--review`, the same model first merges duplicate action items using only the draft. Qwen3 embeddings then retrieve the two closest speaker blocks per organized item, expand them by `--speaker-neighbor-blocks`, and the model reviews the organized draft using the selected source KV. The draft remains at `--output`; separate `.reviewed.txt`, `.reviewed.organized.txt` and `.reviewed.changes.txt` files contain the reviewed summary, organized draft, and textual changes with candidate source excerpts. These excerpts are not verified claim-level citations. Review loads `--embedding-model` even without `--vector-routing`. If either generation is truncated or empty, review is skipped and the draft is preserved. Potentially missing standard sections are reported in metrics without blocking output; this check does not guarantee that each item is preserved.

The evidence-review model is asked to explain each change and its evidence. No JSON or exact item-matching format is required; detected explanations are written to the change log. Missing or incomplete explanations do not prevent output, and model-provided evidence is not independently verified. Organization-stage changes are recorded as textual differences.

Thinking is off by default. Add `--enable-thinking` to enable it for the final summary receiver. Use identical input, settings and seed and separate output paths for comparison. Reasoning text is not saved in the summary; metrics separately count thinking and answer tokens. An unfinished thinking section produces an empty summary and is flagged as incomplete. Thinking and the answer share the remaining model context.

Use `--seed 7` for summary generation and `--review-seed 42` for review; both default to 7. Decoding is greedy, so changing seeds normally does not change the output. Results can still vary across GPU hardware or dependency versions. The evaluation command also defaults to seed 7.



### Evaluation

Evaluate both `routed_kv` and `specialist_text` on one dataset:

```bash
python -m meeting_evaluation.test \
  --benchmark meetingbank \
  --input input/meetingbank_speaker_test.jsonl \
  --output-dir test_result \
  --num-samples 25 \
  --model Qwen/Qwen3-4B \
  --receiver-kv-role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --device cuda
```

Your own JSONL can also be supplied via `--input`. `--benchmark` labels the report and accepts `ami`, `icsi`, `qmsum_committee`, or `meetingbank`. Set `--num-samples 0` to process every record. Add `--communication-metrics-only` for token, cache, and timing metrics only; summaries are still generated to count output tokens, but their text is discarded and quality is not scored.

Reports are written as `.txt` and `.json` under `test_result/YYYYMMDD_HHMM/`. Repeating the same benchmark within the same minute can overwrite a report with the same name.

Run all four prepared datasets with:

```bash
python -m meeting_evaluation.run_all \
  --num-samples 25 \
  --speaker-neighbor-blocks 1 \
  --receiver-kv-role-budget-ratio 0.25 \
  --model Qwen/Qwen3-4B \
  --device cuda
```

The default inputs are `input/ami_test.jsonl`, `input/icsi_test.jsonl`, `input/qmsum_committee_test.jsonl`, and `input/meetingbank_speaker_test.jsonl`. Add `--skip-missing` to skip unavailable datasets. See [data_preprocess/README.md](data_preprocess/README.md) for preparation: AMI, ICSI, and Committee are separate QMSum domains, while MeetingBank uses official agenda-item segments with speaker-labelled transcripts. Samples drawn from QMSum's `all` directories should not be reported as official held-out test results.

### Parameters and metrics

| Parameter | Meaning |
| --- | --- |
| `--role-budget-ratio` / `--receiver-kv-role-budget-ratio` | Initial KV-position budget per role query. Role selections are unioned and neighbouring blocks are then added, so the final retained fraction can exceed this value. |
| `--speaker-neighbor-blocks` | Number of adjacent speaker blocks retained on each side of a selected block; default 1, or 0 to disable expansion. |
| `--span-tokens` / `--receiver-kv-span-tokens` | Span size used in initial role-query position selection; default 32. Final selection uses complete speaker blocks. |
| `--query-batch-tokens` / `--receiver-kv-query-batch-tokens` | Batch size for attention-query computation; default 8. |

ROUGE-L measures word-order overlap with a reference, not factual accuracy. Source-support is a lexical-overlap proxy, not a human factuality check. KV positions, KV bytes, logical model-token events, and elapsed time are different measures and cannot be directly converted into one another.

### Current limits

The complete transcript is prefetched once. Each generation continues until an EOS token or the remaining model context is exhausted. The program raises an error if the prompt leaves no output room; it does not automatically summarize document chunks or silently truncate the input. `cap hits` counts outputs that reach the context boundary. Without a user-set output limit, inference may take longer and use more memory. The cache adapter targets dense full-attention Qwen3 models with standard RoPE. The full source KV remains resident during routing, so a smaller receiver KV does not imply a proportional reduction in peak process memory.

For implementation details, see [receiver_kv_workflow/README.md](receiver_kv_workflow/README.md) and [meeting_evaluation/README.md](meeting_evaluation/README.md).

## Optional semantic block routing / 可选向量辅助块选择

See [vector/README.md](vector/README.md) for embedding model download,
`--vector-routing`, inference/evaluation commands and accounting details.
向量模块辅助选择发言块，最终仍传原始 KV；默认不启用，便于保留原方法对照。
