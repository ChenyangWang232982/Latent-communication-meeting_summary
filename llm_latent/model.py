import torch
import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_latent.cipher_comm import CipherEmbeddingComm


class SameModelCipherSystem(nn.Module):
    def __init__(
        self,
        model_name="google/long-t5-tglobal-large",
        latent_len=8,
        temperature=1.0,
        use_compression=True,
        freeze_sender=True,
        freeze_receiver=False,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Some Transformers versions do not restore LongT5's padding token
        # from tokenizer.json.  <pad> is already in the T5 vocabulary, so
        # assigning it does not resize or otherwise change model weights.
        if self.tokenizer.pad_token is None:
            vocabulary = self.tokenizer.get_vocab()
            if "<pad>" in vocabulary:
                self.tokenizer.pad_token = "<pad>"
            elif self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("The tokenizer has neither a pad token nor an EOS token.")

        self.sender = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.receiver = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.sender.config.pad_token_id = self.tokenizer.pad_token_id
        self.receiver.config.pad_token_id = self.tokenizer.pad_token_id
        self.sender.generation_config.max_length = None
        self.receiver.generation_config.max_length = None

        hidden_size = self.sender.config.d_model
        vocab_size = self.sender.config.vocab_size

        self.comm = CipherEmbeddingComm(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            latent_len=latent_len,
            num_heads=8,
            temperature=temperature,
            use_compression=use_compression,
        )

        if freeze_sender:
            for parameter in self.sender.parameters():
                parameter.requires_grad = False

        if freeze_receiver:
            for parameter in self.receiver.parameters():
                parameter.requires_grad = False

    def encode_sender(self, sender_input_ids, sender_attention_mask):
        """Encode the source transcript inside the sender agent."""
        if not any(parameter.requires_grad for parameter in self.sender.parameters()):
            with torch.no_grad():
                outputs = self.sender.get_encoder()(
                    input_ids=sender_input_ids,
                    attention_mask=sender_attention_mask,
                    return_dict=True,
                )
        else:
            outputs = self.sender.get_encoder()(
                input_ids=sender_input_ids,
                attention_mask=sender_attention_mask,
                return_dict=True,
            )

        return outputs.last_hidden_state

    def build_receiver_inputs(
        self,
        sender_input_ids,
        sender_attention_mask,
        receiver_input_ids,
        receiver_attention_mask,
    ):
        sender_hidden_states = self.encode_sender(
            sender_input_ids,
            sender_attention_mask,
        )

        # CIPHER maps its vocabulary probabilities into the receiver's own
        # embedding space, so the receiver can consume continuous tokens.
        receiver_embedding_weight = self.receiver.get_input_embeddings().weight
        latent_embeds, latent_mask = self.comm(
            sender_hidden_states,
            receiver_embedding_weight,
            sender_attention_mask,
        )

        receiver_prompt_embeds = self.receiver.get_input_embeddings()(
            receiver_input_ids
        )
        combined_embeds = torch.cat(
            [latent_embeds, receiver_prompt_embeds],
            dim=1,
        )
        combined_attention_mask = torch.cat(
            [latent_mask, receiver_attention_mask],
            dim=1,
        )

        return combined_embeds, combined_attention_mask

    def forward(
        self,
        sender_input_ids,
        sender_attention_mask,
        receiver_input_ids,
        receiver_attention_mask,
        labels,
    ):
        combined_embeds, combined_attention_mask = self.build_receiver_inputs(
            sender_input_ids,
            sender_attention_mask,
            receiver_input_ids,
            receiver_attention_mask,
        )

        receiver_encoder_outputs = self.receiver.get_encoder()(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            return_dict=True,
        )

        return self.receiver(
            encoder_outputs=receiver_encoder_outputs,
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True,
        )
