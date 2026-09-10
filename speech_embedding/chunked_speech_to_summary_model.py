import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, WhisperModel

from speech_embedding.speech_comm import (
    ReceiverWeightedEmbeddingComm,
    SpeechEmbeddingComm,
)


class ChunkedSpeechToSummaryLatentModel(nn.Module):
    def __init__(
        self,
        speech_model_name="openai/whisper-base",
        summary_model_name="google/flan-t5-small",
        chunk_latent_len=4,
        num_heads=8,
        freeze_speech=True,
        freeze_summary=False,
        comm_method="receiver_weighted_embedding",
        comm_temperature=1.0,
    ):
        super().__init__()

        self.speech_model = WhisperModel.from_pretrained(speech_model_name)
        self.summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)
        self.freeze_speech = freeze_speech

        speech_dim = self.speech_model.config.d_model
        llm_dim = self.summary_model.config.d_model

        if comm_method == "direct_projection":
            self.comm = SpeechEmbeddingComm(
                speech_dim=speech_dim,
                llm_dim=llm_dim,
                latent_len=chunk_latent_len,
                num_heads=num_heads,
            )
        elif comm_method == "receiver_weighted_embedding":
            self.comm = ReceiverWeightedEmbeddingComm(
                speech_dim=speech_dim,
                llm_dim=llm_dim,
                vocab_size=self.summary_model.config.vocab_size,
                latent_len=chunk_latent_len,
                num_heads=num_heads,
                temperature=comm_temperature,
            )
        else:
            raise ValueError(f"Unsupported comm_method: {comm_method}")

        self.comm_method = comm_method
        self.chunk_latent_len = chunk_latent_len

        if freeze_speech:
            for param in self.speech_model.parameters():
                param.requires_grad = False

        if freeze_summary:
            for param in self.summary_model.parameters():
                param.requires_grad = False

    def encode_speech_chunk(self, input_features):
        if self.freeze_speech:
            with torch.no_grad():
                outputs = self.speech_model.encoder(
                    input_features=input_features,
                    return_dict=True,
                )
        else:
            outputs = self.speech_model.encoder(
                input_features=input_features,
                return_dict=True,
            )
        return outputs.last_hidden_state

    def encode_all_chunks(self, input_features, chunk_attention_mask):
        batch_size, num_chunks, num_mels, num_frames = input_features.shape
        chunk_latents = []
        chunk_masks = []

        for chunk_idx in range(num_chunks):
            chunk_features = input_features[:, chunk_idx, :, :]
            speech_hidden_states = self.encode_speech_chunk(chunk_features)
            if getattr(self.comm, "requires_receiver_embedding", False):
                receiver_embedding_weight = (
                    self.summary_model.get_input_embeddings().weight
                )
                latent_embeds, latent_mask = self.comm(
                    speech_hidden_states,
                    receiver_embedding_weight=receiver_embedding_weight,
                )
            else:
                latent_embeds, latent_mask = self.comm(speech_hidden_states)

            active = chunk_attention_mask[:, chunk_idx].view(batch_size, 1, 1)
            latent_embeds = latent_embeds * active.to(latent_embeds.dtype)

            active_mask = chunk_attention_mask[:, chunk_idx].view(batch_size, 1)
            latent_mask = latent_mask * active_mask.to(latent_mask.dtype)

            chunk_latents.append(latent_embeds)
            chunk_masks.append(latent_mask)

        all_latents = torch.cat(chunk_latents, dim=1)
        all_masks = torch.cat(chunk_masks, dim=1)
        return all_latents, all_masks

    def forward(
        self,
        input_features,
        chunk_attention_mask,
        prompt_input_ids,
        prompt_attention_mask,
        labels,
    ):
        latent_embeds, latent_mask = self.encode_all_chunks(
            input_features,
            chunk_attention_mask,
        )

        prompt_embeds = self.summary_model.get_input_embeddings()(prompt_input_ids)

        combined_embeds = torch.cat([latent_embeds, prompt_embeds], dim=1)
        combined_mask = torch.cat([latent_mask, prompt_attention_mask], dim=1)

        encoder_outputs = self.summary_model.get_encoder()(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            return_dict=True,
        )

        outputs = self.summary_model(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs
