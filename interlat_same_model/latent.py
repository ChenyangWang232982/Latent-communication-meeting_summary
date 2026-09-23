"""Core components for the Interlat hidden-state communication channel."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LatentCompressor(nn.Module):
    """Learn a short latent trajectory by cross-attending to a full one."""

    def __init__(self, hidden_size: int, num_heads: int, output_length: int):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, output_length, hidden_size))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(states.size(0), -1, -1)
        attended, _ = self.attention(queries, states, states, need_weights=False)
        return self.norm(queries + attended)


class HiddenStateAdapter(nn.Module):
    """The learned Interlat receiver-side hidden-state integration module."""

    def __init__(self, source_size: int, target_size: int, num_heads: int):
        super().__init__()
        if target_size % num_heads:
            raise ValueError("target_size must be divisible by num_heads")
        self.project = nn.Identity() if source_size == target_size else nn.Linear(source_size, target_size, bias=False)
        self.attention = nn.MultiheadAttention(target_size, num_heads, batch_first=True)
        self.gate = nn.Linear(target_size, target_size)
        self.norm = nn.LayerNorm(target_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        projected = self.project(states)
        attended, _ = self.attention(projected, projected, projected, need_weights=False)
        return self.norm(projected + torch.sigmoid(self.gate(projected)) * attended)


class InterlatReceiver(nn.Module):
    """Receiver with `<bop> latent trajectory <eop>` input insertion.

    This implements the paper's learned latent channel: Sender states are not
    decoded to text, and the Receiver is trained with task, plan-alignment,
    and shuffled-plan contrastive objectives.
    """

    def __init__(
        self,
        receiver: nn.Module,
        source_hidden_size: int,
        num_heads: int = 8,
        compressed_latent_len: int | None = None,
        plan_similarity_weight: float = 0.5,
        random_contrast_weight: float = 0.1,
    ):
        super().__init__()
        self.receiver = receiver
        target_size = receiver.config.hidden_size
        self.compressor = (
            LatentCompressor(source_hidden_size, num_heads, compressed_latent_len)
            if compressed_latent_len else None
        )
        self.adapter = HiddenStateAdapter(source_hidden_size, target_size, num_heads)
        self.plan_similarity_weight = plan_similarity_weight
        self.random_contrast_weight = random_contrast_weight

    def prepare_latents(self, sender_states: torch.Tensor) -> torch.Tensor:
        if sender_states.ndim == 2:
            sender_states = sender_states.unsqueeze(0)
        if sender_states.ndim != 3:
            raise ValueError("sender_states must be [batch, steps, hidden]")
        if self.compressor is not None:
            sender_states = self.compressor(sender_states)
        return self.adapter(sender_states)

    def build_inputs(self, sender_states, bop_id, eop_id, prompt_ids, target_ids=None):
        device = prompt_ids.device
        token_embedding = self.receiver.get_input_embeddings()
        embedding_dtype = token_embedding.weight.dtype
        latent = self.prepare_latents(sender_states).to(dtype=embedding_dtype)
        batch_size = prompt_ids.size(0)
        bop = token_embedding(torch.full((batch_size, 1), bop_id, dtype=torch.long, device=device))
        eop = token_embedding(torch.full((batch_size, 1), eop_id, dtype=torch.long, device=device))
        parts = [bop, latent, eop, token_embedding(prompt_ids)]
        ignored_length = bop.size(1) + latent.size(1) + eop.size(1) + prompt_ids.size(1)
        if target_ids is not None:
            parts.append(token_embedding(target_ids))
        inputs_embeds = torch.cat(parts, dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        labels = None
        if target_ids is not None:
            labels = torch.cat(
                (
                    torch.full((batch_size, ignored_length), -100, dtype=torch.long, device=device),
                    target_ids,
                ),
                dim=1,
            )
        return inputs_embeds, attention_mask, labels, latent

    def forward(self, *, sender_states, bop_id, eop_id, prompt_ids, target_ids):
        embeds, mask, labels, latent = self.build_inputs(sender_states, bop_id, eop_id, prompt_ids, target_ids)
        output = self.receiver(
            inputs_embeds=embeds,
            attention_mask=mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        latent_length = latent.size(1)
        received_plan = output.hidden_states[-1][:, 1:1 + latent_length]
        plan_loss = F.mse_loss(received_plan.float(), latent.detach().float())
        shuffled = latent.roll(1, dims=1)
        positive = F.cosine_similarity(received_plan.float(), latent.detach().float(), dim=-1).mean()
        negative = F.cosine_similarity(received_plan.float(), shuffled.detach().float(), dim=-1).mean()
        contrast_loss = F.relu(0.2 - positive + negative)
        loss = output.loss + self.plan_similarity_weight * plan_loss + self.random_contrast_weight * contrast_loss
        return {"loss": loss, "task_loss": output.loss.detach(), "plan_loss": plan_loss.detach(), "contrast_loss": contrast_loss.detach()}

    @torch.inference_mode()
    def generate(self, *, sender_states, bop_id, eop_id, prompt_ids, max_new_tokens):
        embeds, mask, _, _ = self.build_inputs(sender_states, bop_id, eop_id, prompt_ids)
        return self.receiver.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.receiver.config.pad_token_id,
        )
