# 语义辅助 KV 通信会议摘要 / Semantic-Assisted KV Meeting Summarization

## 中文

本项目采用无需额外训练的混合通信工作流：全文 prefill 一次，通过注意力与可选
向量辅助路由选择证据，再将选中的证据按发言块顺序组成重叠窗口；每个 specialist
只接收自己的窗口并生成局部报告；最终 agent 结合全部选中证据与有序报告生成摘要。

| 方法 | 每个 specialist 的输入 | 最终汇总 |
|---|---|---|
| `chunk_kv` | 当前选中窗口的 KV cache | 全部 selected KV＋有序局部报告 |
| `chunk_text` | 相同窗口坐标对应的原文文本 | 全部选中文本＋有序局部报告 |

局部报告采用自然语言。两条路径使用相同路由、分块、指令和生成设置，但 KV 包含
原始全文的历史影响，选中文本则重新编码，因此属于工作流对照，不声称完全隔离
通信表示的因果效应。旧版直接 `routed_kv`、单次 `specialist_text` handoff 和额外
review 已删除；历史结果说明见 [experiment_1/HISTORY.md](experiment_1/HISTORY.md)。

### 环境准备

从项目根目录运行，命令使用 Linux/Bash 续行格式。PowerShell 请改用反引号续行或一行执行。

```bash
conda activate latent-agents
pip install -r requirements.txt
```

需要新环境时可使用 `conda env create -f requirements.yml`。默认模型为 Qwen3-8B；
首次使用未缓存模型需要下载。向量辅助使用 Qwen3-Embedding-0.6B：

```bash
python -m vector.download
```

下载参数和本地目录设置见 [vector/README.md](vector/README.md)。

### 输入形式

单会议为 UTF-8 TXT，行首含说话人标签。相邻同一说话人的发言合并：

```text
Alice: 我们讨论一下测试计划。
Alice: 当前方案已经准备好。
Bob: 我会在周五之前发送测试报告。
```

也支持 `Speaker_0` 等匿名标签以及原文时间戳。程序不执行音频转录或说话人分离。
评测数据为 UTF-8 JSONL，一行一个样本：

```jsonl
{"id":"meeting_001","transcript":"Alice: What is next?\nBob: I will send the report on Friday.","summary":"Bob will send the report on Friday."}
```

`context` 可替代 `transcript`，`answer` 可替代 `summary`。正式实验需要唯一 ID 和非空
人工参考摘要。`question` 不改变固定的全会议摘要任务。数据预处理默认不按长度筛选，
每个数据集最多100条，保留原文token数量用于分组。准备指令
见 [data_preprocess/README.md](data_preprocess/README.md)。ROUGE 使用英文向分词。

### 单会议运行

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --variants chunk_kv \
  --seed 12 \
  --summary-seed 42
```

两种通信一起运行时，使用 `--variants chunk_kv chunk_text`。
默认已经启用分块 specialists，不再需要 `--chunk-specialists`。
最终汇总默认开启 thinking；用 `--no-summary-thinking` 同时关闭两种方法的最终思考。
局部 specialist 默认不思考。最终参数是 `--summary-*`，不再使用 `--review-*`。

输出包括最终摘要、`.specialists.json`（汇总KV坐标、窗口坐标、局部报告和逐阶段
计量）、独立 thinking 文本以及实时 partial/status 文件。指标在
`test_result/receiver_kv/<timestamp>/metrics.txt` 和 `.json`；`--routing-audit`
可额外导出选中证据坐标。没有人工输出条数或生成 token 上限；上下文不足则报错。

### 实验1：先测3个样本

```bash
python -m experiment_1 \
  --input input/experiment_1_unfiltered_cap100/meetingbank.jsonl \
  --benchmark meetingbank \
  --output-dir test_result/experiment_1_chunk_3_seed12 \
  --num-samples 3 \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --seed 12 \
  --summary-seed 42
```

取当前文件前3个样本，每条分别运行 `chunk_kv` 与 `chunk_text`，不是每个数据集各3条。
输出包含 ROUGE、逻辑 token、耗时、显存、重复诊断、逐样本摘要和中间报告。
窗口目标数为 `max(1, floor(全文token数/1024 + 0.5))`，保留完整发言块；25%重叠
不增加窗口数量。少于目标数的可用发言块按实际数量处理。`overview.*` 提供总览和
`[0,4000)`、`[4000,8000)`、`[8000,12000)`、`[12000,16000)`、`[16000,+∞)`分组。
BERTScore 使用保存后的结果离线补评，不重新生成摘要；不使用 AI judge。
详细指令、指标定义和断点续跑见 [experiment_1/README.md](experiment_1/README.md)。

### 目录与结果口径

| 目录 | 用途 |
|---|---|
| `receiver_kv_workflow/` | 模型、路由、cache、共享参数与单会议入口 |
| `chunk_specialist_workflow/` | 全文分块、局部分析和最终汇总 |
| `vector/` | embedding 下载与语义辅助路由 |
| `data_preprocess/` | 数据规范化、token计数和每数据集上限100抽样 |
| `experiment_1/` | 配对质量与成本实验、离线 BERTScore |
| `meeting_evaluation/` | 通用批量指标工具 |
| `meeting_stream/` | 向 AG-UI 提供 WebSocket 状态和最终摘要 |
| `input/`, `output/`, `test_result/` | 输入、摘要、实验记录 |

逻辑模型 token、KV positions、KV tensor bytes 与 UTF-8 bytes 是不同指标。
token 节省不等于 FLOPs、API计费或网络流量节省；embedding token 单独报告。
全文 KV 仍驻留，接收缓存变小不代表进程峰值显存同比下降。ROUGE/BERTScore 不等于
事实准确率，词面 source-support proxy 也不是事实核查。转录缺失视觉信息或内容有歧义
时，摘要可能出错。当前适配标准RoPE、全注意力 Qwen3。

## English

This training-free workflow prefills the full transcript once and selects global
evidence using role-conditioned attention and optional embedding assistance.
Independent specialists receive only their chronological window of selected source
context. No extra original-part text or full global context is supplied locally.
Final synthesis receives all selected context and ordered natural-language local reports.

`chunk_kv` transfers each local window as selected KV; `chunk_text` supplies
verbatim text from the same window coordinates. Windows, instructions, stage seeds
and decoding settings match. Source KV retains contextual and positional history
that re-encoded selected text does not, so this compares workflows rather than a
pure representation-only intervention. The old direct-summary, single-handoff and
review paths are retired.

Use UTF-8 speaker-labelled TXT for one meeting, or JSONL with `id`,
`transcript`/`context` and human `summary`/`answer` for evaluation. Adjacent turns
from the same speaker stay together. No ASR or diarization is provided. Inputs are
not silently truncated. Window count targets `max(1, floor(raw_transcript_tokens/1024 + 0.5))`,
preserving whole speaker blocks (fewer blocks means fewer windows). Quarter overlap
does not add specialists. Preprocessing has no default length filter. Reports include
five half-open token-length groups and an overall view in `overview.*`.

The Bash commands above are the current entry points. Defaults: Qwen3-8B, source
budget per role0.25, speaker neighbors +/-1, local sampling0.3/0.9 with repetition
penalty1.1, final sampling0.3/0.9 with thinking enabled. `--summary-*` controls final
synthesis and `--no-summary-thinking` disables its reasoning. `--output` selects
the primary answer path; other variants use sibling files. Thinking is separately
saved and excluded from answer quality scoring, but included in generated-token
costs. No separate review or arbitrary output-token cap is applied.

Experiment1 runs both variants for each selected record, saving summaries, stage
reports, timings, CUDA allocator peaks, logical decoder token events and reference
overlap. Three records means three paired comparisons from one input file.
BERTScore can be added offline using the same saved answers. There is no LLM judge.
See [experiment documentation](experiment_1/README.md),
[workflow architecture](chunk_specialist_workflow/README.md),
[data preparation](data_preprocess/README.md) and
[WebSocket service](meeting_stream/README.md).

KV tensor bytes, positions, text UTF-8 bytes and model-token events have separate
definitions; they are not measured network traffic or billed-token/FLOP savings.
The full source cache stays resident. Embedding tokens are accounted separately.
Automatic reference overlap and lexical support do not establish factual accuracy.

## Experiment 2 / 实验2：全文直接概括基线

`experiment_2` runs one full-transcript summary agent using the shared Qwen3
loader, summary requirements, and final decoding settings. It does not run
selectors, embeddings, local specialists, or review. One model is loaded per
batch; defaults select three records. Saved results support offline BERTScore and
paired comparisons against existing Experiment 1 checkpoints.

实验2用于比较完整工作流与“全文原文直接交给同一模型”的质量和资源消耗。
复用现有 AMI JSONL，默认先测三条；输出、断点续跑、离线评分和配对比较命令见
[experiment_2/README.md](experiment_2/README.md)。尚未实现常规分块汇总基线或 challenge 集。
