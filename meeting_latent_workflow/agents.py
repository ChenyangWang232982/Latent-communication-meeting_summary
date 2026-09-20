from __future__ import annotations

from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutput

from llm_latent.model import SameModelCipherSystem

from meeting_latent_workflow.config import WorkflowConfig


def resolve_device(configured_device: str) -> str:
    if configured_device != "auto":
        return configured_device
    return "cuda" if torch.cuda.is_available() else "cpu"


class TextAgent:
    """A small instruction-following agent backed by the receiver LLM."""

    def __init__(self, system: SameModelCipherSystem, config: WorkflowConfig):
        self.tokenizer = system.tokenizer
        self.model = system.receiver
        self.device = resolve_device(config.device)
        self.max_input_tokens = config.max_input_tokens
        self.max_new_tokens = config.max_new_tokens

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.15,
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


class CipherSummaryAgent:
    """Transfers specialist notes as CIPHER continuous embeddings to T5."""

    def __init__(self, system: SameModelCipherSystem, config: WorkflowConfig):
        self.system = system
        self.tokenizer = system.tokenizer
        self.device = resolve_device(config.device)
        self.max_input_tokens = config.max_input_tokens
        self.max_new_tokens = config.max_new_tokens

    @torch.inference_mode()
    def generate(self, sender_message: str, receiver_prompt: str) -> str:
        sender = self.tokenizer(
            sender_message,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        receiver = self.tokenizer(
            receiver_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        embeddings, attention_mask = self.system.build_receiver_inputs(
            sender_input_ids=sender.input_ids,
            sender_attention_mask=sender.attention_mask,
            receiver_input_ids=receiver.input_ids,
            receiver_attention_mask=receiver.attention_mask,
        )
        encoder_outputs = self.system.receiver.get_encoder()(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )
        generated = self.system.receiver.generate(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state
            ),
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.15,
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def load_system(config: WorkflowConfig) -> SameModelCipherSystem:
    """Load the shared-model system and optionally a trained CIPHER bridge."""
    device = resolve_device(config.device)
    system = SameModelCipherSystem(
        model_name=config.model_name,
        latent_len=config.compressed_latent_len,
        temperature=config.temperature,
        use_compression=config.use_compression,
        freeze_sender=True,
        freeze_receiver=False,
    ).to(device)

    if config.communication_mode == "cipher":
        if config.cipher_checkpoint is None:
            raise ValueError(
                "cipher mode requires --cipher-checkpoint trained for meeting summarization"
            )
        checkpoint = Path(config.cipher_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CIPHER checkpoint not found: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        system.load_state_dict(state_dict)

    system.eval()
    return system
