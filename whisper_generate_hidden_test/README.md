# Whisper Generate Hidden State Test

This folder is a small isolated experiment for checking whether HuggingFace
`WhisperForConditionalGeneration.generate` can return decoder hidden states
during generation.

Run from the project root:

```bash
python whisper_generate_hidden_test/test_generate_hidden_states.py
```

The script will:

1. Load one audio file.
2. Run Whisper `generate` with `return_dict_in_generate=True` and
   `output_hidden_states=True`.
3. Print the generated transcript.
4. Print the structure of returned hidden states.
5. Try to collect the last decoder-layer hidden state into a tensor.

Edit `AUDIO_PATH` in the script if you want to test a specific file.
