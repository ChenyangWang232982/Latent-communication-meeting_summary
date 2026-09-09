import torch
from torch.utils.data import DataLoader

from speech_embedding.data import MeetingSpeechSummaryDataset
from speech_embedding.paths import CHECKPOINT_DIR, DATA_DIR
from speech_embedding.speech_to_summary_model import SpeechToSummaryLatentModel


@torch.no_grad()
def generate_with_latents(model, batch, device, use_random_latent=False, max_new_tokens=256):
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    speech_hidden_states = model.encode_speech(batch["input_features"])
    latent_embeds, latent_mask = model.comm(speech_hidden_states)

    if use_random_latent:
        latent_embeds = torch.randn_like(latent_embeds)

    prompt_embeds = model.summary_model.get_input_embeddings()(
        batch["prompt_input_ids"]
    )

    combined_embeds = torch.cat(
        [latent_embeds, prompt_embeds],
        dim=1,
    )

    combined_mask = torch.cat(
        [latent_mask, batch["prompt_attention_mask"]],
        dim=1,
    )

    encoder_outputs = model.summary_model.get_encoder()(
        inputs_embeds=combined_embeds,
        attention_mask=combined_mask,
        return_dict=True,
    )

    generated_ids = model.summary_model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=combined_mask,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=False,
    )

    return model.summary_tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    checkpoint_path = CHECKPOINT_DIR / "checkpoint_embedding_overfit.pt"

    model = SpeechToSummaryLatentModel(
        speech_model_name="openai/whisper-base",
        summary_model_name="google/flan-t5-small",
        latent_len=32,
        freeze_speech=True,
        freeze_summary=False,
    ).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dataset = MeetingSpeechSummaryDataset(
        metadata_path=DATA_DIR / "metadata.jsonl",
        speech_model_name="openai/whisper-base",
        summary_tokenizer=model.summary_tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    batch = next(iter(loader))

    normal_summary = generate_with_latents(
        model=model,
        batch=batch,
        device=device,
        use_random_latent=False,
    )

    random_summary = generate_with_latents(
        model=model,
        batch=batch,
        device=device,
        use_random_latent=True,
    )

    labels = batch["labels"].clone()
    labels[labels == -100] = model.summary_tokenizer.pad_token_id
    gold_summary = model.summary_tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    print("\nNormal latent summary:")
    print(normal_summary[0])

    print("\nRandom latent summary:")
    print(random_summary[0])

    print("\nGold summary:")
    print(gold_summary[0])


if __name__ == "__main__":
    main()
