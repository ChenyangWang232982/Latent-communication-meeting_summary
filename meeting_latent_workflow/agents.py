"""Qwen agents for text baselines and strict training-free CIPHER transport."""

from __future__ import annotations

import torch

from cipher_training_free.cipher import TrainingFreeCipher
from meeting_latent_workflow.config import WorkflowConfig


class TextAgent:
    """Natural-language baseline using the CIPHER receiver model directly."""

    def __init__(self, cipher: TrainingFreeCipher, config: WorkflowConfig):
        self.cipher = cipher
        self.tokenizer = cipher.tokenizer
        self.model = cipher.receiver
        self.max_input_tokens = config.max_input_tokens
        self.max_new_tokens = config.max_new_tokens

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        inputs = self.cipher._chat_inputs(prompt, self.max_input_tokens)
        prompt_length = inputs["input_ids"].size(1)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()


class CipherSummaryAgent:
    """Transfers a summary draft through native Qwen CIPHER embeddings."""

    def __init__(self, cipher: TrainingFreeCipher, config: WorkflowConfig):
        self.cipher = cipher
        self.config = config

    @torch.inference_mode()
    def generate(self, evidence: str, receiver_prompt: str) -> str:
        result = self.cipher.communicate(
            sender_context=evidence,
            sender_instruction=(
                "Write a concise, factual meeting-summary draft from the evidence. "
                "Include supported decisions, unresolved issues, owners, and action "
                "items. Do not invent details."
            ),
            receiver_instruction=receiver_prompt,
            source_max_tokens=self.config.max_input_tokens,
            sender_max_new_tokens=self.config.sender_max_new_tokens,
            receiver_max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return result.receiver_answer


def load_agents(config: WorkflowConfig) -> tuple[TextAgent, CipherSummaryAgent]:
    """Load identical frozen decoder-only agents without any CIPHER checkpoint."""
    cipher = TrainingFreeCipher(model_name=config.model_name, device=config.device)
    return TextAgent(cipher, config), CipherSummaryAgent(cipher, config)
