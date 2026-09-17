from pathlib import Path

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "openai/whisper-base"
AUDIO_PATH = None

SAMPLING_RATE = 16000
MAX_NEW_TOKENS = 64
LANGUAGE = "english"
TASK = "transcribe"


def find_audio_file():
    candidates = []
    for folder in [
        PROJECT_ROOT / "input",
        PROJECT_ROOT / "data" / "toy_meetings" / "audio",
    ]:
        if not folder.exists():
            continue
        candidates.extend(sorted(folder.glob("*.wav")))
        candidates.extend(sorted(folder.glob("*.mp3")))
        candidates.extend(sorted(folder.glob("*.flac")))
        candidates.extend(sorted(folder.glob("*.m4a")))

    if not candidates:
        raise FileNotFoundError(
            "No audio file found. Put a wav/mp3/flac/m4a file under input/ "
            "or set AUDIO_PATH in this script."
        )
    return candidates[0]


def describe_value(name, value, indent=0, max_depth=4):
    prefix = " " * indent
    if isinstance(value, torch.Tensor):
        print(
            f"{prefix}{name}: Tensor("
            f"shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
        )
        return

    if isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: {type(value).__name__}(len={len(value)})")
        if max_depth <= 0:
            return
        for index, item in enumerate(value[:3]):
            describe_value(f"[{index}]", item, indent + 2, max_depth - 1)
        if len(value) > 3:
            print(f"{prefix}  ...")
        return

    if isinstance(value, dict):
        print(f"{prefix}{name}: dict(keys={list(value.keys())})")
        if max_depth <= 0:
            return
        for key, item in list(value.items())[:5]:
            describe_value(str(key), item, indent + 2, max_depth - 1)
        return

    print(f"{prefix}{name}: {type(value).__name__}({value})")


def get_forced_decoder_ids(processor):
    tokenizer = processor.tokenizer
    if hasattr(tokenizer, "get_decoder_prompt_ids"):
        return tokenizer.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    return None


def collect_last_decoder_layer(decoder_hidden_states):
    if decoder_hidden_states is None:
        return None

    collected = []
    for step_hidden_states in decoder_hidden_states:
        if not step_hidden_states:
            continue

        last_layer = step_hidden_states[-1]
        if last_layer.ndim == 3:
            last_token = last_layer[:, -1:, :]
        elif last_layer.ndim == 2:
            last_token = last_layer.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected last layer shape: {last_layer.shape}")

        collected.append(last_token)

    if not collected:
        return None
    return torch.cat(collected, dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_path = Path(AUDIO_PATH) if AUDIO_PATH else find_audio_file()
    if not audio_path.is_absolute():
        audio_path = PROJECT_ROOT / audio_path

    print("device:", device)
    print("model:", MODEL_NAME)
    print("audio:", audio_path)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    inputs = processor(
        audio,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
    )
    input_features = inputs["input_features"].to(device)

    generate_kwargs = {
        "input_features": input_features,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": 1,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "output_scores": True,
    }
    forced_decoder_ids = get_forced_decoder_ids(processor)
    if forced_decoder_ids is not None:
        generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    print("\nGenerated ids:")
    describe_value("sequences", outputs.sequences)

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    print("\nGenerated text:")
    print(text)

    print("\nGenerate output keys:")
    print(list(outputs.keys()))

    print("\nFull hidden-state structure:")
    if hasattr(outputs, "encoder_hidden_states"):
        describe_value("encoder_hidden_states", outputs.encoder_hidden_states)
    if hasattr(outputs, "decoder_hidden_states"):
        describe_value("decoder_hidden_states", outputs.decoder_hidden_states)

    last_decoder_latents = collect_last_decoder_layer(
        getattr(outputs, "decoder_hidden_states", None)
    )
    print("\nCollected last decoder layer:")
    if last_decoder_latents is None:
        print("No decoder hidden states were returned.")
    else:
        describe_value("last_decoder_latents", last_decoder_latents)
        print(
            "Meaning: [batch_or_beam, generated_steps, hidden_dim]. "
            "This is the candidate one-pass decoder latent message."
        )


if __name__ == "__main__":
    main()
