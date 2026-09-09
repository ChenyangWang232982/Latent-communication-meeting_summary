# Latent Communication for Meeting Summarization Roadmap

## 0. Thesis Direction

Your thesis direction is:

> Replace traditional text prompt / transcript communication with latent communication between a speech agent and a summarization agent for meeting summarization.

Traditional workflow:

```text
Meeting audio
  -> ASR model
  -> transcript text
  -> prompt: "Summarize this meeting: {transcript}"
  -> LLM summarizer
  -> meeting summary
```

Target workflow:

```text
Meeting audio
  -> speech / ASR agent
  -> latent communication module
  -> LLM summarization agent
  -> meeting summary
```

The communication channel is the main variable:

```text
1. Prompts / transcript text
2. Embedding-level latent communication
3. Hidden-state latent communication
4. KV-level latent communication
```

The first implementation target is:

```text
Embedding-level latent communication
```

## 1. Overall Experimental Structure

Keep the system structure stable:

```text
Audio / transcript input
  -> speech agent
  -> communication module
  -> summary agent
  -> generated summary
```

Only replace the communication module.

Recommended comparison table:

| Method | Communication Form | Main Purpose | Difficulty |
| --- | --- | --- | --- |
| Prompt baseline | Text transcript / text prompt | Standard baseline | Easy |
| Embedding communication | Soft prompt embeddings | First latent method | Medium |
| Hidden-state communication | Intermediate representations | Deeper latent method | Hard |
| KV communication | Speech key-value memory | SpeechKV-inspired method | Hardest |

## 2. Method A: Prompt / Text Communication Baseline

This is the standard pipeline and must be implemented first or in parallel as the baseline.

### Workflow

```text
Meeting audio
  -> Whisper ASR
  -> transcript
  -> "Summarize the following meeting: {transcript}"
  -> FLAN-T5 / other LLM
  -> summary
```

### Why This Baseline Is Needed

This represents traditional agent communication:

```text
Speech agent speaks to summary agent using human-readable text.
```

Your latent methods should be compared against this.

### Implementation Steps

1. Load meeting audio.
2. Run Whisper ASR to get transcript.
3. Build summarization prompt.
4. Feed prompt into summarization model.
5. Evaluate with ROUGE / BERTScore / factual checks.

### Expected Output

```text
Generated meeting summary from transcript prompt.
```

## 3. Method B: Embedding-Level Latent Communication

This is the first method to implement.

### Core Idea

Do not pass transcript text from the speech agent to the summarization agent.

Instead:

```text
Speech hidden states
  -> latent embedding projector
  -> soft prompt embeddings
  -> summarization LLM
```

The summary agent receives vectors directly:

```text
[latent_1, latent_2, ..., latent_K] + [summary instruction tokens]
```

### Minimal Architecture

```text
Meeting audio
  -> Whisper encoder
  -> speech hidden states
  -> EmbeddingComm
  -> projected latent embeddings
  -> FLAN-T5 encoder input embeddings
  -> FLAN-T5 decoder
  -> summary
```

### Tensor Shapes

Example with `latent_len = 32` and `flan-t5-small`:

```text
speech_hidden_states: [batch_size, speech_seq_len, whisper_dim]
latent_embeds:        [batch_size, 32, t5_dim]
prompt_embeds:        [batch_size, prompt_len, t5_dim]
combined_embeds:      [batch_size, 32 + prompt_len, t5_dim]
```

### Recommended First Communication Module

Use query attention pooling:

```text
K learnable query tokens attend to all speech hidden states.
The result is K latent tokens.
Then project them into the LLM embedding dimension.
```

Why this method:

```text
1. It is easier than KV modification.
2. It works naturally with inputs_embeds.
3. It lets the model learn which speech frames matter.
4. It gives a fixed communication budget: K latent tokens.
```

### Implementation Steps

1. Load a small speech encoder, such as `openai/whisper-base` or `openai/whisper-small`.
2. Extract encoder hidden states from audio features.
3. Build `SpeechEmbeddingComm`.
4. Use learnable query attention to compress speech hidden states into `latent_len` vectors.
5. Project from Whisper dimension to T5 dimension.
6. Tokenize a short instruction, such as:

```text
summarize the meeting:
```

7. Convert instruction tokens to T5 embeddings.
8. Concatenate:

```python
combined_embeds = torch.cat([latent_embeds, prompt_embeds], dim=1)
```

9. Feed `combined_embeds` to the T5 encoder with `inputs_embeds`.
10. Train against reference meeting summaries.

### First Version Training Strategy

Start small:

```text
Freeze Whisper encoder.
Freeze most or all of T5 if GPU memory is tight.
Train only:
  - SpeechEmbeddingComm
  - projection layer
  - optionally LoRA on T5
```

For RTX 3070, recommended first version:

```text
Whisper: whisper-base
Summary model: flan-t5-small
latent_len: 16 or 32
batch_size: 1 or 2
mixed precision: fp16
```

### Necessary Baselines

Compare:

```text
1. Transcript prompt baseline
2. Embedding latent communication
3. Random latent embeddings
4. Shuffled latent embeddings
```

Random/shuffled baselines verify that the latent channel really carries information.

### Success Condition

The embedding method is working if:

```text
1. Training loss decreases.
2. Generated summaries mention correct meeting content.
3. Random or shuffled latent embeddings perform much worse.
4. It is competitive with transcript prompt baseline on some metrics or qualitative cases.
```

## 4. Method C: Hidden-State Latent Communication

This is the second latent method to implement after embedding works.

### Core Idea

Instead of converting speech representations into input embeddings, inject speech hidden states deeper into the summarization model.

Possible workflow:

```text
Whisper hidden states
  -> adapter
  -> cross-attention memory / intermediate hidden injection
  -> LLM summarizer
```

### Why It Is Different From Embedding Communication

Embedding communication enters before the LLM starts reasoning:

```text
latent embeddings -> LLM input
```

Hidden-state communication enters during LLM reasoning:

```text
latent states -> intermediate layer / cross-attention
```

### Implementation Options

Option 1: Cross-attention memory

```text
LLM hidden states attend to speech latent memory.
```

Option 2: Layer injection

```text
At layer l, add or concatenate projected speech hidden states.
```

Option 3: Compressed hidden-state prefix

```text
Pool speech hidden states and insert them as hidden representations at an intermediate layer.
```

### Why Do This After Embedding

Hidden-state communication requires modifying model internals.

Do it only after:

```text
1. The data pipeline works.
2. The summary baseline works.
3. Embedding communication works.
```

## 5. Method D: KV-Level Latent Communication

This is the SpeechKV-inspired method.

### Core Idea

The communication object is not text, embedding, or hidden state.

It is:

```text
speech key-value memory inside the LLM attention mechanism
```

SpeechKV's main idea:

```text
Compress the cache, not the speech embedding.
```

That means:

```text
Do not compress speech before the LLM too early.
Let the LLM process full speech representations for several layers.
Then compress only speech K/V in later layers.
```

### SpeechKV-Style Workflow

```text
Audio
  -> speech encoder
  -> adapter to LLM dimension
  -> LLM first layers process full speech tokens
  -> from layer l0 onward:
       compress speech K/V with learned pooling
  -> summary generation attends to compressed speech KV memory
```

### Learned KV Pooling

For compression ratio `R = 4`, every 4 adjacent speech K/V vectors become 1 vector.

For a window:

```text
k1, k2, k3, k4
v1, v2, v3, v4
```

The model learns weights:

```text
alpha_1, alpha_2, alpha_3, alpha_4
```

Then:

```text
k_hat = alpha_1*k1 + alpha_2*k2 + alpha_3*k3 + alpha_4*k4
v_hat = alpha_1*v1 + alpha_2*v2 + alpha_3*v3 + alpha_4*v4
```

### Why KV Is Useful

KV communication is closer to Transformer internals:

```text
K = what each previous position can be matched by
V = what information each previous position provides
```

Compressing speech K/V means:

```text
The summary agent keeps a compact speech memory instead of a text transcript.
```

### Why KV Should Be Last

This requires modifying attention layers and cache behavior.

It is harder because:

```text
1. You need to identify speech token positions.
2. You need to modify K/V only, not Q.
3. Attention masks must remain correct.
4. Generation cache behavior is more complex.
```

### Practical Mini Version

For this project, do not start with full SpeechKV reproduction.

Start with:

```text
1. Decoder-only small LLM, such as GPT-2 small or Qwen2.5-0.5B.
2. Speech prefix embeddings from Whisper.
3. Modify attention to pool only speech prefix K/V.
4. Keep text tokens uncompressed.
```

## 6. Recommended Implementation Order

Do the work in this order:

```text
Step 1: Prompt baseline
Step 2: Embedding latent communication
Step 3: Embedding ablations
Step 4: Hidden-state latent communication
Step 5: SpeechKV-style KV communication
```

Because your current priority is embedding, the immediate order is:

```text
1. Keep current CIPHER toy code as proof of concept.
2. Build speech-to-summary embedding communication.
3. Compare with transcript prompt baseline.
4. Add random and shuffled latent baselines.
5. Move to hidden state and KV only after embedding is stable.
```

## 7. Immediate TODO: Embedding Method

### TODO 1: Create SpeechEmbeddingComm

File idea:

```text
latent_agents/speech_comm.py
```

Class idea:

```text
SpeechEmbeddingComm
```

It should map:

```text
[batch_size, speech_seq_len, whisper_dim]
  -> [batch_size, latent_len, t5_dim]
```

### TODO 2: Create SpeechToSummaryLatentModel

File idea:

```text
latent_agents/speech_to_summary_model.py
```

It should contain:

```text
Whisper encoder
SpeechEmbeddingComm
FLAN-T5 summarizer
```

### TODO 3: Create Data Pipeline

Data format:

```python
{
    "audio_path": "...",
    "summary": "..."
}
```

For the first version, you can also use transcript text and generated audio later. But the final target should accept audio.

### TODO 4: Evaluation

Evaluate:

```text
ROUGE
BERTScore
qualitative summary examples
random latent baseline
shuffled latent baseline
```

## 8. Key Thesis Claim

A clean thesis claim:

> We formulate meeting summarization as communication between a speech agent and a summarization agent. Instead of passing human-readable transcripts or prompts, we investigate latent communication channels, including embedding-level, hidden-state-level, and KV-level communication.

For the first implemented method:

> The embedding-level method transmits compressed speech information as soft prompt embeddings to the summarization model, replacing the transcript prompt communication channel.

