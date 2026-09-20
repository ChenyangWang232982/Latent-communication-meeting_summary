# Meeting Latent Workflow

This project uses LangGraph to run a lightweight, bounded meeting-summarization
workflow:

```text
ASR transcript -> transcript cleaner -> parallel topic / decision / action agents
-> summary receiver -> critic -> optional one-pass refiner -> final summary
```

Store each local Hugging Face model in `meeting_latent_workflow/model/` and
select it with `--model`. The recommended backbone is
`long-t5-tglobal-large`; its current workflow input limit is 4096 tokens,
matching the pretrained configuration. Longer meetings should be split before
entering this graph.

Long transcripts are segmented at paragraph/sentence boundaries into 2800-token
chunks with 256-token overlap. LangGraph dynamically sends each chunk to the
cleaner and three specialist extractors, then recursively merges reports in
groups of three before final summarization. Override these settings with
`--chunk-tokens`, `--chunk-overlap-tokens`, and `--reduce-group-size`.
`--max-concurrency` defaults to `1` to avoid concurrent LongT5 generations
exhausting a single GPU's memory.

The ASR boundary remains normal text. The research communication boundary is
between the three specialist senders and the summary receiver.

Audio inputs use `openai/whisper-large-v3`. On the first audio run, download it
to `meeting_latent_workflow/model/whisper-large-v3/` with
`--download-asr-model`. The workflow stores the plain transcript and
timestamped Whisper result beside the final summary in `output/`.

- `--mode text`: concatenates specialist notes as a natural-language baseline.
- `--mode cipher`: converts specialist notes to continuous CIPHER embeddings
  before they reach the receiver. It requires a CIPHER checkpoint trained for
  meeting summarization with matching model and compression settings.

## Install

```powershell
pip install -r requirements.txt
```

## Run the text baseline

Put transcript files in `meeting_latent_workflow/input/`. The command accepts
one file name or one folder name relative to that directory and writes matching
`.summary.txt` files under `meeting_latent_workflow/output/`.

```powershell
python -m meeting_latent_workflow.run example_meeting.txt `
  --model long-t5-tglobal-large `
  --mode text
```

Process every `.txt` transcript below `input/ami_test/`:

```powershell
python -m meeting_latent_workflow.run ami_test `
  --model long-t5-tglobal-large `
  --mode text
```

## Run from audio

Put `.wav`, `.mp3`, `.flac`, `.m4a`, or `.ogg` audio in `input/`. The first
run downloads Whisper large-v3; later runs reuse the local snapshot.

```powershell
python -m meeting_latent_workflow.run ami_audio `
  --model long-t5-tglobal-large `
  --mode text `
  --download-asr-model
```

## Run CIPHER communication

```powershell
python -m meeting_latent_workflow.run `
  example_meeting.txt `
  --model long-t5-tglobal-large `
  --mode cipher `
  --cipher-checkpoint checkpoints/cipher_meeting_uncompressed.pt
```

For a valid comparison, keep the prompts, source transcript, generation
settings, and receiver model fixed; only replace the sender-to-receiver
communication channel. Run text, uncompressed CIPHER, compressed CIPHER, and
zero/shuffled-latent controls on the same meeting split.
