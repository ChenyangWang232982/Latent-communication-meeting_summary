import torch
import torch.nn as nn
from transformers import WhisperModel, AutoModelForSeq2SeqLM, AutoTokenizer

from speech_embedding.speech_comm import SpeechEmbeddingComm

class SpeechToSummaryLatentModel(nn.Module):
    def __init__(self,
                 speech_model_name = "openai/whisper-base",
                 summary_model_name = "google/flan-t5-small",
                 latent_len = 32, #The number of latent token from ASR to LLM
                 num_heads = 8,
                 freeze_speech = True,
                 freeze_summary = False):
        super().__init__()
        self.speech_model = WhisperModel.from_pretrained(speech_model_name)
        self.summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)

        speech_dim = self.speech_model.config.d_model
        llm_dim = self.summary_model.config.d_model

        self.comm = SpeechEmbeddingComm(
            speech_dim= speech_dim,
            llm_dim= llm_dim,
            latent_len = latent_len,
            num_heads = num_heads
        )

        self.latent_len = latent_len

        #freeze model
        if freeze_speech:
            for param in self.speech_model.parameters():
                param.requires_grad = False

        if freeze_summary:
            for param in self.summary_model.parameters():
                param.requires_grad = False

    # encode speech
    def encode_speech(self, input_features):
        outputs = self.speech_model.encoder(
            input_features=input_features,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        input_features,
        prompt_input_ids,
        prompt_attention_mask,
        labels,
    ):
        speech_hidden_states = self.encode_speech(input_features)
        # speech_hidden_states: [batch, speech_seq_len, speech_dim]

        latent_embeds, latent_mask = self.comm(speech_hidden_states)
        # latent_embeds: [batch, latent_len, llm_dim]
        # latent_mask: [batch, latent_len]

        prompt_embeds = self.summary_model.get_input_embeddings()(
            prompt_input_ids
        )

        combined_embeds = torch.cat(
            [latent_embeds, prompt_embeds],
            dim=1,
        )

        combined_mask = torch.cat(
            [latent_mask, prompt_attention_mask],
            dim=1,
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
