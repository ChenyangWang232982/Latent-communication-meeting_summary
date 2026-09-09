# 模型流程说明：从会议音频到摘要输出

这份文档解释当前项目里 embedding-level latent communication 框架的完整流程。目标是让你能看懂：数据从哪里来，音频怎么被预处理，speech agent 怎么把语音变成 latent tokens，summary agent 怎么根据 latent tokens 生成会议摘要，以及训练时模型到底在学什么。

当前框架对应的是：

```text
会议音频
  -> 音频预处理和分块
  -> Whisper speech encoder
  -> SpeechEmbeddingComm latent 压缩模块
  -> FLAN-T5 summary model
  -> 会议摘要
```

这里的关键点是：两个 agent 之间没有传递普通文本 transcript，而是传递一组连续向量，也就是 latent embeddings。

## 1. 数据文件

当前主要使用 AMI 会议数据。

音频文件位置：

```text
data/toy_meetings/audio/
```

摘要标注文件来源：

```text
ami/abstractive/*.abssumm.xml
```

训练用的总 metadata 文件：

```text
data/toy_meetings/metadata.jsonl
```

训练、验证、测试划分后的文件：

```text
data/toy_meetings/splits/train.jsonl
data/toy_meetings/splits/val.jsonl
data/toy_meetings/splits/test.jsonl
```

metadata 里每一行大概长这样：

```json
{
  "meeting_id": "ES2002a",
  "audio_path": "data/toy_meetings/audio/ES2002a.wav",
  "transcript": "",
  "summary": "The project manager introduced ..."
}
```

这里 `transcript` 目前是空的，因为当前实验重点不是“ASR 先转文本再总结”，而是直接把音频编码成 latent communication，再让总结模型生成摘要。

## 2. 数据预处理第一步：构建 metadata

负责脚本：

```text
download_ami_audio.py
```

它做三件事：

1. 从 `ami/abstractive` 里读取每场会议的人工摘要。
2. 检查或下载对应的 `Mix-Headset.wav` 音频。
3. 把音频路径和摘要配对，写入 `data/toy_meetings/metadata.jsonl`。

也就是说，模型训练时不直接读 XML，而是读整理好的 JSONL。

可以把它理解成建立一个表：

```text
会议ID      音频路径                         目标摘要
ES2002a    data/.../ES2002a.wav             summary text
ES2002b    data/.../ES2002b.wav             summary text
...
```

## 3. 数据预处理第二步：划分训练集、验证集、测试集

负责脚本：

```text
speech_embedding/split_metadata.py
```

运行后会把总 metadata 按大约 `5 : 1 : 1` 分成三份：

```text
train.jsonl  用来训练模型
val.jsonl    用来在训练过程中挑选较好的 checkpoint
test.jsonl   最后写论文结果时才使用
```

三者的作用不同：

```text
训练集：模型直接看它，并根据 loss 调整权重。
验证集：模型不根据它更新权重，只用来判断训练效果。
测试集：最终评估用，调参阶段尽量不要频繁看。
```

这样做是为了避免模型只是记住训练数据，而不是学会从音频 latent 中提取摘要信息。

## 4. 数据预处理第三步：音频分块

负责文件：

```text
speech_embedding/chunked_data.py
```

类名：

```python
ChunkedMeetingSpeechSummaryDataset
```

会议音频一般很长，不能一次性全部塞进 Whisper 和 T5。所以当前方法会把音频切成固定长度的小块。

当前默认设置：

```python
chunk_seconds = 30
max_chunks = 16
sampling_rate = 16000
```

含义是：

```text
每个 chunk 是 30 秒。
每场会议最多取 16 个 chunk。
每秒采样 16000 个音频点。
```

所以每个 chunk 的原始音频长度是：

```text
30 * 16000 = 480000 个采样点
```

之前如果只取前 16 个 chunk，就只能看到前 8 分钟，长会议后半部分信息会丢失。现在代码已经改成了“均匀抽块”：

```text
如果会议太长，不是只拿开头，而是在整场会议中均匀选 16 个位置。
```

例如一场会议有 80 个 30 秒 chunk，模型不会只拿：

```text
0, 1, 2, ..., 15
```

而是会类似拿：

```text
0, 5, 11, 16, ..., 79
```

这样至少能覆盖会议开头、中间和结尾。

## 5. Whisper 输入特征

音频 chunk 不是直接喂给 Whisper，而是先经过：

```python
WhisperProcessor
```

它会把 30 秒音频转换成 log-mel spectrogram 特征。

单个 chunk 的特征形状是：

```text
[80, 3000]
```

含义：

```text
80    表示 mel 频率维度
3000  表示时间帧
```

一场会议最多 16 个 chunk，所以一个样本的 `input_features` 形状是：

```text
[16, 80, 3000]
```

训练时加上 batch 维度，如果 `batch_size = 4`，就会变成：

```text
[4, 16, 80, 3000]
```

可以这样读：

```text
4 个会议样本
每个会议 16 个音频块
每个音频块是 [80, 3000] 的 Whisper 输入特征
```

## 6. chunk_attention_mask 是什么

不是每场会议都刚好有 16 个 chunk。

如果某个会议很短，只有 10 个有效 chunk，代码会补 6 个全 0 chunk，让所有样本形状一致。

这时候需要一个 mask 告诉模型哪些 chunk 是真的，哪些只是补齐用的。

例如：

```text
chunk_attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
```

含义：

```text
1 表示真实音频 chunk
0 表示 padding chunk
```

后面模型会把 padding chunk 对应的 latent 向量屏蔽掉。

## 7. Speech Agent：Whisper Encoder

负责文件：

```text
speech_embedding/chunked_speech_to_summary_model.py
```

模型类：

```python
ChunkedSpeechToSummaryLatentModel
```

里面的 speech agent 是：

```python
self.speech_model = WhisperModel.from_pretrained("openai/whisper-base")
```

当前只使用 Whisper 的 encoder：

```python
outputs = self.speech_model.encoder(
    input_features=input_features,
    return_dict=True,
)
speech_hidden_states = outputs.last_hidden_state
```

Whisper encoder 的作用是把音频特征变成一串语音 hidden states。

你可以把它理解成：

```text
原始音频特征：
[80, 3000]

Whisper 理解后的语音表示：
[speech_seq_len, speech_dim]
```

这些 hidden states 还不是文本，也不是 transcript，而是 Whisper 内部对这段语音的连续向量表示。

当前设置里：

```python
freeze_speech = True
```

意思是 Whisper 不训练，只当作一个固定的语音特征提取器。这样显存更省，训练也更稳定。

## 8. Latent Communication 模块

负责文件：

```text
speech_embedding/speech_comm.py
```

类名：

```python
SpeechEmbeddingComm
```

这是当前最核心的 latent communication 模块。

输入：

```text
Whisper encoder 输出的 speech_hidden_states
```

输出：

```text
一小组 latent embeddings
```

当前每个 chunk 压成：

```python
chunk_latent_len = 4
```

也就是说：

```text
30 秒音频 -> 4 个 latent tokens
```

如果一场会议取 16 个 chunk，那么整场会议最多得到：

```text
16 * 4 = 64 个 latent tokens
```

这就是所谓 latent compression：

```text
长音频里的大量语音 hidden states
  -> 被压缩成少量 latent tokens
```

## 9. latent token 是不是一个单词

不是。

在这里：

```text
1 个 latent token 不是 1 个自然语言单词。
```

它更像是一个“可训练的信息槽位”。

普通文本 token 可能对应：

```text
apple
meeting
remote
```

但 latent token 是一个向量，例如：

```text
[0.12, -0.33, 0.08, ...]
```

它没有固定的人类可读含义。训练之后，它可能学会携带：

```text
会议主题
关键决定
任务分配
问题讨论
时间顺序
```

但它不会直接显示成一个词。

## 10. query_tokens 是什么

在 `SpeechEmbeddingComm` 里有：

```python
self.query_tokens = nn.Parameter(
    torch.randn(latent_len, speech_dim) * 0.02
)
```

这表示模型自己维护了几个可训练的查询向量。

如果 `latent_len = 4`，那每个 chunk 有 4 个 query tokens。

你可以把它想成 4 个问题槽：

```text
第 1 个 query：从语音里找主题信息
第 2 个 query：从语音里找决定信息
第 3 个 query：从语音里找行动项信息
第 4 个 query：从语音里找其他重要信息
```

注意：这只是方便理解。模型内部不会真的写着这些中文标签。

真实情况是：这些 query tokens 在训练中通过反向传播不断调整，慢慢学会从 Whisper hidden states 里抓取对摘要有用的信息。

## 11. Attention 如何压缩信息

核心代码：

```python
latent_speech, _ = self.attn(
    query=queries,
    key=speech_hidden_states,
    value=speech_hidden_states,
    key_padding_mask=key_padding_mask,
    need_weights=False,
)
```

这里使用的是 multi-head attention。

简单理解：

```text
query_tokens 会去看整段 speech_hidden_states。
每个 query 会根据相似度，决定更关注语音 hidden states 的哪些位置。
最后每个 query 汇总出一个 latent token。
```

所以压缩不是随便平均，也不是手动规则，而是模型学出来的注意力加权汇总。

## 12. 为什么还要 projector

Whisper 的 hidden state 维度和 FLAN-T5 的 embedding 维度不一定完全适合直接相接。

所以代码用了 projector：

```python
self.projector = nn.Sequential(
    nn.LayerNorm(speech_dim),
    nn.Linear(speech_dim, llm_dim),
    nn.GELU(),
    nn.Linear(llm_dim, llm_dim)
)
```

它的作用是：

```text
把 Whisper 空间里的 latent_speech
转换到 T5 能理解的 embedding 空间
```

可以理解成一个“翻译器”：

```text
语音模型内部向量语言
  -> 总结模型输入 embedding 语言
```

这一步非常重要。因为 latent communication 不是只要拿到一个向量就行，还要让 receiver，也就是 summary model，能读懂这个向量。

## 13. Summary Agent：FLAN-T5

当前总结模型是：

```python
google/flan-t5-small
```

在代码里：

```python
self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)
```

FLAN-T5 是 encoder-decoder 模型。

当前我们不是把 transcript 文本输入给 T5，而是把两部分拼起来输入给 T5 encoder：

```text
latent_embeds + prompt_embeds
```

代码：

```python
prompt_embeds = self.summary_model.get_input_embeddings()(prompt_input_ids)

combined_embeds = torch.cat([latent_embeds, prompt_embeds], dim=1)
combined_mask = torch.cat([latent_mask, prompt_attention_mask], dim=1)
```

含义：

```text
前面放语音 agent 传来的 latent 信息
后面放一个简短任务提示，比如 summarize the meeting:
```

也就是说，T5 看到的不是：

```text
summarize the meeting: Alice said ...
```

而是：

```text
[latent token 1] [latent token 2] ... [latent token 64] summarize the meeting:
```

这就是“用 latent communication 替代普通 prompt/transcript 传递”的核心。

## 14. labels 是什么

训练时需要告诉模型正确答案是什么。

正确答案就是人工会议摘要：

```python
labels = self.encode_summary(item["summary"])
```

T5 生成摘要时，会一步一步预测下一个 token。

例如 gold summary 是：

```text
The project manager introduced ...
```

模型会学习：

```text
给定 latent + prompt，应该输出 The
给定 latent + prompt + The，应该输出 project
给定 latent + prompt + The project，应该输出 manager
...
```

loss 就是模型生成答案和 gold summary 之间的差距。

训练的目标是让 loss 下降。

## 15. 训练时哪些权重会变

训练脚本：

```text
speech_embedding/train_chunked_embedding.py
```

当前设置：

```python
freeze_speech = True
freeze_summary = False
```

所以：

```text
Whisper encoder 不更新。
SpeechEmbeddingComm 会更新。
FLAN-T5 会更新。
```

最重要的是 `SpeechEmbeddingComm` 里的这些部分会被训练：

```text
query_tokens
attention 参数
projector 参数
LayerNorm 参数
```

它们会学习如何把语音 hidden states 压缩成对摘要有用的 latent embeddings。

如果 `freeze_summary = False`，T5 也会一起适应这种新的 latent 输入格式。

## 16. 训练循环在做什么

训练循环核心逻辑：

```python
outputs = model(**batch)
loss = outputs.loss

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

可以按四步理解：

```text
1. forward：
   音频 -> latent -> T5 -> 摘要预测

2. 计算 loss：
   比较预测摘要和 gold summary 的差距

3. backward：
   计算哪些参数导致了错误

4. optimizer.step：
   调整可训练参数，让下次预测更接近 gold summary
```

这就是你之前问的“receiver 怎么恢复信息”：不是人工写规则恢复，而是训练时不断调整权重，让 receiver 逐渐学会如何使用这些 latent 向量。

## 17. 当前训练参数在哪里

文件：

```text
speech_embedding/train_chunked_embedding.py
```

主要参数在 `train()` 里面：

```python
speech_model_name = "openai/whisper-base"
summary_model_name = "google/flan-t5-small"
max_chunks = 16
chunk_latent_len = 4
max_summary_length = 256
epochs = 50
```

DataLoader 里有 batch size：

```python
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
```

优化器里有学习率：

```python
optimizer = AdamW(trainable_params, lr=5e-5)
```

这些参数的含义：

```text
max_chunks：
每场会议最多取多少个 30 秒音频块。

chunk_latent_len：
每个音频块压缩成几个 latent tokens。

max_summary_length：
目标摘要最多保留多少个 token。

epochs：
完整训练集重复训练多少轮。

batch_size：
一次训练多少个会议样本。

lr：
每次参数更新的步子有多大。
```

对 RTX 3070 来说，建议先保守使用：

```text
batch_size = 1 或 2
max_chunks = 16
chunk_latent_len = 4
max_summary_length = 256
```

如果显存够，再慢慢尝试：

```text
max_chunks = 24
```

或者：

```text
chunk_latent_len = 8
```

不要一开始同时把它们都调大，因为显存和训练时间都会明显增加。

## 18. checkpoint 是什么

训练时会保存验证集 loss 最低的模型：

```text
speech_embedding/checkpoint_chunked_embedding_best.pt
```

这个文件保存的是模型权重。

以后 generate 或 evaluate 时，需要加载这个 checkpoint，才能使用训练后的 latent communication 模块。

## 19. 生成摘要时发生什么

生成时不再计算 loss，也不需要 gold summary。

流程是：

```text
音频
  -> 分块
  -> Whisper encoder
  -> SpeechEmbeddingComm
  -> latent embeddings
  -> 拼接 prompt embeddings
  -> T5 generate
  -> 输出 summary
```

训练和生成最大的区别：

```text
训练时：有 gold summary，用它计算 loss 并更新权重。
生成时：没有 gold summary，只让模型自己生成摘要。
```

## 20. 为什么需要 random latent baseline

你之前做过：

```text
正常 latent -> 生成接近会议内容的 summary
随机 latent -> 生成完全无关或重复的 summary
```

这个实验的意义是证明：

```text
模型不是只靠 prompt 在生成。
latent 里面确实带了信息。
```

如果随机 latent 也能生成正确摘要，那说明 latent communication 没有发挥作用，模型可能只是记住了数据或靠 prompt 猜。

## 21. 为什么需要 shuffled latent baseline

shuffled latent 的做法是：

```text
样本 A 使用样本 B 的 latent
样本 B 使用样本 A 的 latent
prompt 保持不变
```

如果输出跟着 latent 走，而不是跟着原始样本走，就说明：

```text
summary model 主要依赖 latent communication 里的信息。
```

你之前观察到：

```text
第一条 shuffle 后像第二条 summary
第二条 shuffle 后像第一条 summary
```

这是一个很好的 sanity check。

它说明 latent 不是装饰，确实控制了生成内容。

## 22. 当前方法的论文表述方式

你可以在论文里把当前方法描述为：

```text
We implement an embedding-level latent communication framework for speech-to-summary multi-agent communication.
The speech agent encodes meeting audio with a pretrained Whisper encoder.
A trainable query-attention module compresses speech hidden states into a fixed number of continuous latent tokens.
The summarization agent receives these latent tokens, rather than text transcripts, and generates meeting summaries with a sequence-to-sequence language model.
```

中文理解：

```text
我们实现了一个 embedding 层面的 latent communication 框架。
语音 agent 使用预训练 Whisper encoder 编码会议音频。
一个可训练的 query-attention 模块把语音 hidden states 压缩成固定数量的连续 latent tokens。
总结 agent 不接收文本 transcript，而是接收这些 latent tokens，并生成会议摘要。
```

## 23. 当前框架的优点和限制

优点：

```text
1. 跑通了 speech agent -> latent communication -> summary agent 的完整链路。
2. 没有依赖 transcript，符合替代 prompt/text communication 的研究目标。
3. 有 random latent 和 shuffled latent baseline，可以证明 latent 是否真的携带信息。
4. 分块后可以处理长会议音频，不局限于 30 秒短音频。
```

限制：

```text
1. 均匀抽块仍然可能漏掉部分细节。
2. 目前每个 chunk 固定压成少量 latent tokens，信息容量有限。
3. FLAN-T5-small 能力有限，后续可能需要更强的 summarizer。
4. 当前还没有加入 ROUGE/BERTScore 等自动评价指标。
5. 当前还没有实现 hidden state 和 KV cache 两种 latent communication 对比实验。
```

## 24. 你现在最应该先看懂的代码顺序

建议按这个顺序读：

```text
1. speech_embedding/chunked_data.py
   看懂音频怎么加载、怎么分块、怎么变成 Whisper input_features。

2. speech_embedding/speech_comm.py
   看懂 SpeechEmbeddingComm 怎么用 query attention 把 speech hidden states 压成 latent tokens。

3. speech_embedding/chunked_speech_to_summary_model.py
   看懂 Whisper、latent module、T5 是怎么串起来的。

4. speech_embedding/train_chunked_embedding.py
   看懂训练循环、loss、optimizer、checkpoint。

5. generate 和 baseline 脚本
   看懂训练后的模型怎么生成摘要，以及怎么验证 latent 是否有用。
```

如果时间紧，最核心的是第 2 和第 3 个文件。

## 25. 一句话总结

当前模型做的事情是：

```text
把长会议音频均匀切成若干 30 秒片段，
用 Whisper 把每个片段编码成语音 hidden states，
用一个可训练的 attention 压缩模块把每段语音变成少量 latent tokens，
再把所有 latent tokens 当作 T5 的输入 embeddings，
让 T5 直接从 latent communication 中生成会议摘要。
```

这就是目前 embedding-level speech-to-summary latent communication 的完整基底。
