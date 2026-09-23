"""Strict training-free CIPHER for same decoder-only Qwen agents."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def collect_last_generation_states(hidden_states):
    """Collect one final-layer state for every Sender generation step."""
    states = []
    for step_layers in hidden_states:
        if not step_layers:
            continue
        states.append(step_layers[-1][:, -1:, :])
    if not states:
        raise RuntimeError("Sender generation returned no hidden states.")
    return torch.cat(states, dim=1)


@dataclass
class CipherResult:
    sender_draft: str
    receiver_answer: str
    latent_token_count: int


class TrainingFreeCipher:
    """Original Last-to-First CIPHER mapping with no learned parameters.

    Both agents are identical decoder-only instruction models. The Sender's
    final hidden states go through its native lm_head; vocabulary probabilities
    weight the Receiver's native token embeddings. Those continuous embeddings
    become a causal prefix before the Receiver question prompt. No sender text,
    adapter, projector, compressor, optimizer, or checkpoint is used.
    """

    def __init__(self, model_name=DEFAULT_MODEL_NAME, device="auto"):
        self.device = resolve_device(device)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        self.sender = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.receiver = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        for model in (self.sender, self.receiver):
            model.config.pad_token_id = self.tokenizer.pad_token_id
            model.config.use_cache = True
            model.to(self.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad = False

    def _chat_inputs(self, user_content, max_length):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            name: value.to(self.device)
            for name, value in self.tokenizer(
                rendered,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).items()
        }

    @torch.inference_mode()
    def answer(
        self,
        context,
        question,
        source_max_tokens=4096,
        sender_max_new_tokens=128,
        receiver_max_new_tokens=128,
        temperature=1.0,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        return self.communicate(
            sender_context=context,
            sender_instruction=(
                "Read the context and answer the question concisely and factually.\n\n"
                f"Question: {question}"
            ),
            receiver_instruction=(
                "Answer the question using the received continuous message. "
                f"Question: {question}"
            ),
            source_max_tokens=source_max_tokens,
            sender_max_new_tokens=sender_max_new_tokens,
            receiver_max_new_tokens=receiver_max_new_tokens,
            temperature=temperature,
        )

    @torch.inference_mode()
    def communicate(
        self,
        sender_context,
        sender_instruction,
        receiver_instruction,
        source_max_tokens=4096,
        sender_max_new_tokens=128,
        receiver_max_new_tokens=128,
        temperature=1.0,
    ):
        """Send a task-specific continuous CIPHER message between the agents.

        ``sender_context`` is visible only to the Sender.  The Receiver sees
        its own instruction and CIPHER embeddings, never the Sender's text.
        """
        sender_inputs = self._chat_inputs(
            f"{sender_instruction}\n\nContext:\n{sender_context}",
            source_max_tokens,
        )
        prompt_length = sender_inputs["input_ids"].size(1)
        sender_output = self.sender.generate(
            **sender_inputs,
            max_new_tokens=sender_max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        final_sender_states = collect_last_generation_states(sender_output.hidden_states)

        # Strict CIPHER: native last-layer logits -> expected first-layer token
        # embeddings. No learned projection or normalization is inserted.
        vocabulary_logits = self.sender.lm_head(final_sender_states)
        probabilities = torch.softmax(vocabulary_logits.float() / temperature, dim=-1)
        receiver_embedding_weight = self.receiver.get_input_embeddings().weight.float()
        latent_embeds = (probabilities @ receiver_embedding_weight).to(self.receiver.dtype)
        latent_mask = torch.ones(
            latent_embeds.shape[:2], dtype=torch.long, device=self.device
        )

        receiver_inputs = self._chat_inputs(receiver_instruction, max_length=256)
        prompt_embeds = self.receiver.get_input_embeddings()(
            receiver_inputs["input_ids"]
        )
        combined_embeds = torch.cat((latent_embeds, prompt_embeds), dim=1)
        combined_mask = torch.cat(
            (latent_mask, receiver_inputs["attention_mask"]), dim=1
        )
        receiver_ids = self.receiver.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=receiver_max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return CipherResult(
            sender_draft=self.tokenizer.decode(
                sender_output.sequences[0, prompt_length:], skip_special_tokens=True
            ).strip(),
            receiver_answer=self.tokenizer.decode(
                receiver_ids[0], skip_special_tokens=True).strip(),
            latent_token_count=latent_embeds.size(1),
        )
