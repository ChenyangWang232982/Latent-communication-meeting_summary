import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from speech_embedding.speech_comm import SpeechEmbeddingComm


def collect_last_decoder_layer(decoder_hidden_states):
    """Collect one final-layer state for every Whisper generation step."""
    collected = []
    for step_hidden_states in decoder_hidden_states or ():
        if not step_hidden_states:
            continue
        last_layer = step_hidden_states[-1]
        if last_layer.ndim != 3:
            raise ValueError(
                "Expected Whisper decoder states shaped "
                f"[batch, tokens, hidden], got {tuple(last_layer.shape)}"
            )
        collected.append(last_layer[:, -1:, :])

    if not collected:
        raise RuntimeError("Whisper generate returned no decoder hidden states.")
    return torch.cat(collected, dim=1)


class ChunkedSpeechToSummaryLatentModel(nn.Module):
    def __init__(
        self,
        speech_model_name="openai/whisper-base",
        summary_model_name="google/flan-t5-small",
        chunk_latent_len=64,
        num_heads=8,
        freeze_speech=True,
        freeze_summary=True,
        max_whisper_new_tokens=128,
        language="english",
        task="transcribe",
    ):
        super().__init__()

        self.speech_processor = WhisperProcessor.from_pretrained(speech_model_name)
        self.speech_model = WhisperForConditionalGeneration.from_pretrained(
            speech_model_name
        )
        self.summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)
        self.freeze_speech = freeze_speech
        self.max_whisper_new_tokens = max_whisper_new_tokens
        self.forced_decoder_ids = self._get_forced_decoder_ids(language, task)

        speech_dim = self.speech_model.config.d_model
        llm_dim = self.summary_model.config.d_model

        self.comm = SpeechEmbeddingComm(
            speech_dim=speech_dim,
            llm_dim=llm_dim,
            latent_len=chunk_latent_len,
            num_heads=num_heads,
        )
        self.chunk_latent_len = chunk_latent_len

        if freeze_speech:
            for param in self.speech_model.parameters():
                param.requires_grad = False

        if freeze_summary:
            for param in self.summary_model.parameters():
                param.requires_grad = False

    def _get_forced_decoder_ids(self, language, task):
        tokenizer = self.speech_processor.tokenizer
        if not hasattr(tokenizer, "get_decoder_prompt_ids"):
            return None
        return tokenizer.get_decoder_prompt_ids(language=language, task=task)

    def extract_decoder_latents(self, input_features):
        """Run Whisper once and transmit only final decoder-layer states.

        The generated transcript/token ids remain inside Whisper.  There is no
        call to ``WhisperModel.encoder`` in this decoder-latent experiment.
        """
        if input_features.ndim == 4:
            if input_features.size(1) != 1:
                raise ValueError(
                    "Decoder-latent chunk-block training expects one audio "
                    f"chunk per sample, got {input_features.size(1)}."
                )
            input_features = input_features[:, 0]
        if input_features.ndim != 3:
            raise ValueError(
                "Expected [batch, 80, 3000] input features, got "
                f"{tuple(input_features.shape)}"
            )

        generate_kwargs = {
            "input_features": input_features,
            "max_new_tokens": self.max_whisper_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }
        if self.forced_decoder_ids is not None:
            generate_kwargs["forced_decoder_ids"] = self.forced_decoder_ids

        with torch.no_grad():
            generated = self.speech_model.generate(**generate_kwargs)
        return collect_last_decoder_layer(generated.decoder_hidden_states)

    def encode_all_chunks(self, input_features, chunk_attention_mask=None):
        del chunk_attention_mask
        decoder_hidden_states = self.extract_decoder_latents(input_features)
        return self.comm(decoder_hidden_states)

    def build_receiver_inputs(
        self,
        input_features,
        chunk_attention_mask,
        prompt_input_ids,
        prompt_attention_mask,
    ):
        latent_embeds, latent_mask = self.encode_all_chunks(
            input_features,
            chunk_attention_mask,
        )
        prompt_embeds = self.summary_model.get_input_embeddings()(prompt_input_ids)
        combined_embeds = torch.cat([latent_embeds, prompt_embeds], dim=1)
        combined_mask = torch.cat([latent_mask, prompt_attention_mask], dim=1)
        return combined_embeds, combined_mask

    def forward(
        self,
        input_features,
        chunk_attention_mask,
        prompt_input_ids,
        prompt_attention_mask,
        labels,
    ):
        combined_embeds, combined_mask = self.build_receiver_inputs(
            input_features,
            chunk_attention_mask,
            prompt_input_ids,
            prompt_attention_mask,
        )

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
