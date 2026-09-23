# Meeting CIPHER Workflow

LangGraph workflow for meeting audio or transcripts. It retains the previous
hierarchical architecture, but replaces the deleted learned bridge with the
strict training-free CIPHER implementation in `cipher_training_free/`.

```text
audio -> Whisper large-v3 -> transcript chunks -> clean text
      -> topic / decision / action specialists -> recursive evidence merge
      -> CIPHER Sender -> continuous latent message -> CIPHER Receiver
      -> critic -> optional one-pass refiner -> final summary
```

`--mode cipher` is the default. Sender and Receiver are identical frozen
decoder-only Qwen models. The Sender draft is converted by its native `lm_head`
and the Receiver's native input embeddings; no Sender text, adapter, learned
projector, compressor, checkpoint, or optimizer is used at the communication
boundary. `--mode text` is the natural-language baseline with the same
chunking, prompts, and downstream critic/refiner.

## Directories

- `input/`: put one `.txt` transcript, audio file, or a folder tree here.
- `model/`: put the local Qwen model folder here. Supply its folder name with
  `--model`.
- `output/`: transcripts from audio and final `.summary.txt` files are written
  here, retaining the input tree structure.

Audio supports `.wav`, `.mp3`, `.flac`, `.m4a`, and `.ogg`; it uses
`openai/whisper-large-v3`. Add `--download-asr-model` on the first audio run
to place that model in `model/whisper-large-v3/`.

## Run

Place a local copy of `Qwen/Qwen2.5-1.5B-Instruct` in
`meeting_latent_workflow/model/Qwen2.5-1.5B-Instruct/`, then put a transcript
at `meeting_latent_workflow/input/example.txt`.

```powershell
python -m meeting_latent_workflow.run example.txt --model Qwen2.5-1.5B-Instruct --mode cipher
```

For an entire input folder:

```powershell
python -m meeting_latent_workflow.run ami_test --model Qwen2.5-1.5B-Instruct --mode cipher
```

For a text-channel baseline, change only `--mode`:

```powershell
python -m meeting_latent_workflow.run example.txt --model Qwen2.5-1.5B-Instruct --mode text
```

Use `--chunk-tokens`, `--chunk-overlap-tokens`, and `--reduce-group-size` to
adjust the long-meeting hierarchy. `--max-concurrency` defaults to `1` because
strict CIPHER keeps two copies of the same model on the GPU.
