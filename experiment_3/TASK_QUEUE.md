# AMI100 串行任务队列

队列执行既有实验入口，不改生成工作流：

```text
固定 Qwen3-8B 快照
  → baseline + semantic，各 100 条
  → direct_text，100 条
  → random seed 7，100 条
  → 统一 ROUGE/BERTScore、配对比较和长度分组
```

默认只有随机 seed 7；若一开始就传 `--random-seeds 7 19 43`，三个随机组会依次执行，再统一评分。
这是启动后持续执行的串行队列，没有设置某个时刻启动，也没有并行 GPU 生成。
每次新队列重新生成这些组；不会自动导入之前手动运行的目录。

## 启动（Linux，项目根目录）

先同步代码、等待其他 GPU 实验结束，再执行：

```bash
conda activate latent-agents
cd /root/autodl-tmp/Latent-communication-meeting_summary

python -m pip install -r experiment_1/requirements-quality.txt

nohup python -u -m experiment_3.task_queue \
  --input input/experiment_1_unfiltered_cap100/ami.jsonl \
  --num-samples 100 \
  --output-dir test_result/ami100_task_queue \
  --random-seeds 7 \
  --device cuda \
  >> ami100_queue.console.log 2>&1 < /dev/null &

echo "Queue PID: $!"
```

`nohup` 允许 SSH 连接断开后继续执行，不代表服务器关机后会自动恢复。
不要反复执行启动命令创建多个新队列，它们可能争用同一 GPU。
也可以不加 nohup，直接 `python -m experiment_3.task_queue ...` 前台运行。

## 查看进度

```bash
tail -f ami100_queue.console.log
```

Ctrl+C 只是退出上述日志查看，不会结束 nohup 后台队列。

```bash
QUEUE_DIR=$(find test_result/ami100_task_queue \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

cat "$QUEUE_DIR/QUEUE_REPORT.md"
```

每次输出独立 UTC 时间戳目录，包括：

| 文件/目录 | 内容 |
|---|---|
| input.jsonl | 原输入的冻结副本；各步骤使用同一份，不受原文件后续改动影响 |
| queue.json | 保存配置、当前任务、命令、尝试次数、状态、UTC 时间和结果路径 |
| QUEUE_REPORT.md | 可读的任务完成/失败数和结果目录 |
| logs/ | 每个任务每次尝试的独立日志 |
| runs/00_model/ | 固定模型快照的路径记录，不复制大模型到队列目录 |
| runs/01_workflow/ | 原窗口、V3、选择审计和阶段数据 |
| runs/02_direct/ | 单模型结果 |
| runs/03_random_seed7/ | 随机对照、token 匹配偏差、局部报告和摘要 |
| runs/04_evaluation/ | summary.txt/json、per_sample.csv、length_groups.csv、评分溯源文件 |

生成入口有自己的时间戳子目录；统一评分目录直接使用 `attempt_001` 等尝试目录。
最后的控制台输出会打印 `Scores: .../summary.txt`，也可从 `queue.json` 的 `evaluation` 字段定位。

## 失败与续跑

- 批次完成但部分样本失败：状态为 `completed_with_failures`，记录失败数量后继续。
  单模型入口因个别样本失败返回非零时，只要完整批次产物已落盘，也允许继续。
- 整步崩溃、没有完整产物或全部样本都失败：队列停止，保留日志；不会给失败样本补零分。
- 统一评分只使用所有传入组的共同成功样本，并报告实际配对数。
- 模型、tokenizer 使用同一固定快照；缺少 BERTScore 依赖会在 CLI 启动时直接提示安装。

修复原因后：

```bash
nohup python -u -m experiment_3.task_queue \
  --resume "$QUEUE_DIR" \
  >> ami100_queue.console.log 2>&1 < /dev/null &
```

续跑继承 `queue.json` 的原配置；不需要重新传 input、模型或随机 seed。
同一队列目录有运行锁，正在运行时不能再启动它的第二个副本。

**续跑粒度是整步任务。**已经完成且产物仍在的步骤跳过；正在执行时中断的步骤会新建
`attempt_002` 等目录，重新运行该步骤，不承诺从中断的某条会议继续。早一步产物若丢失而需重跑，
它后面的步骤也会重新执行，避免把新旧依赖混用。旧尝试不会被删除。

## 指标边界

沿用 [AMI100.md](AMI100.md) 的定义：ROUGE/BERTScore 是参考相似度，不是事实正确率；
未标注真实 AMI 的必要证据，不给这 100 条自动计算挑战集同窗覆盖率；随机 replay 的完整流水线
耗时为 N/A，不把省略 selection/embedding 后的速度与完整工作流直接比较。
目标决策、负责人和条件是否正确仍需另行核对，队列不会调用 AI judge 自动判为通过。
