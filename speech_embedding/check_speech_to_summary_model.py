import torch

from speech_embedding.speech_to_summary_model import SpeechToSummaryLatentModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpeechToSummaryLatentModel(
        speech_model_name="openai/whisper-base",
        summary_model_name="google/flan-t5-small",
        latent_len=32,
        freeze_speech=True,
        freeze_summary=False,
    ).to(device)

    batch_size = 1

    input_features = torch.randn(
        batch_size,
        80,
        3000,
        device=device,
    )

    prompt = model.summary_tokenizer(
        "summarize the meeting:",
        padding="max_length",
        max_length=16,
        truncation=True,
        return_tensors="pt",
    )

    target = model.summary_tokenizer(
        "The team discussed project updates and next steps.",
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt",
    )

    labels = target["input_ids"]
    labels[labels == model.summary_tokenizer.pad_token_id] = -100

    outputs = model(
        input_features=input_features,
        prompt_input_ids=prompt["input_ids"].to(device),
        prompt_attention_mask=prompt["attention_mask"].to(device),
        labels=labels.to(device),
    )

    print("loss:", outputs.loss.item())


if __name__ == "__main__":
    main()