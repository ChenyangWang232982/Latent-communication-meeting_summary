"""Paper-style Interlat communication adapter and latent Actor wrapper."""

from __future__ import annotations

import math

import torch
from torch import nn


IGNORE_INDEX = -100


class AdaptiveProjection(nn.Module):
    """Adaptive projection used by the official Interlat communication adapter."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.proj[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        residual = states * self.scale
        return (residual + self.proj(residual)) * self.output_scale


class CommunicationAdapter(nn.Module):
    """MHA + projection adapter from Interlat Section 3.1 and official code."""

    def __init__(self, source_size: int, target_size: int, num_heads: int = 8):
        super().__init__()
        if target_size % num_heads:
            raise ValueError("target_size must be divisible by num_heads")
        self.input_projector = (
            nn.Linear(source_size, target_size, bias=True) if source_size != target_size else nn.Identity()
        )
        self.pre_ln = nn.LayerNorm(target_size, eps=1e-6)
        self.mha = nn.MultiheadAttention(target_size, num_heads, batch_first=True, dropout=0.1)
        self.post_ln = nn.LayerNorm(target_size, eps=1e-6)
        self.adaptive_projection = AdaptiveProjection(target_size)
        self._init_attention()

    def _init_attention(self) -> None:
        nn.init.xavier_uniform_(self.mha.in_proj_weight, gain=1.0 / math.sqrt(3.0))
        nn.init.zeros_(self.mha.in_proj_bias)
        nn.init.xavier_uniform_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        projected = self.input_projector(states)
        normalized = self.pre_ln(projected)
        attended, _ = self.mha(normalized, normalized, normalized, need_weights=False)
        processed = self.adaptive_projection(self.post_ln(normalized + attended))
        # The released Interlat adapter constrains its inserted latent vectors
        # after projection to keep them in a stable embedding-compatible range.
        return torch.clamp(processed, -10.0, 10.0)


class InterlatActor(nn.Module):
    """A trainable paper-style Actor receiving a delimited latent trajectory."""

    def __init__(self, actor: nn.Module, source_hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.actor = actor
        self.config = actor.config
        self.adapter = CommunicationAdapter(source_hidden_size, actor.config.hidden_size, num_heads)

    def _boundary_embeddings(self, bop_id: int, eop_id: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.actor.get_input_embeddings()
        device = embedding.weight.device
        ids = torch.tensor([bop_id, eop_id], device=device, dtype=torch.long)
        vectors = embedding(ids).to(dtype=embedding.weight.dtype)
        return vectors[0].view(1, 1, -1).expand(batch_size, -1, -1), vectors[1].view(1, 1, -1).expand(batch_size, -1, -1)

    def adapt_latents(self, sender_states: torch.Tensor) -> torch.Tensor:
        return self.adapter(sender_states)

    def mix_plan_tokens(
        self,
        adapted_latents: torch.Tensor,
        plan_ids: torch.Tensor,
        replacement_rate: float,
    ) -> torch.Tensor:
        """Official Interlat curriculum: latent prefix, textual-plan suffix."""
        plan_embeddings = self.actor.get_input_embeddings()(plan_ids).to(adapted_latents.dtype)
        if replacement_rate <= 0.0:
            return plan_embeddings
        if replacement_rate >= 1.0:
            return adapted_latents
        latent_prefix_length = int(round(adapted_latents.size(1) * replacement_rate))
        plan_prefix_length = int(round(plan_embeddings.size(1) * replacement_rate))
        return torch.cat(
            (adapted_latents[:, :latent_prefix_length], plan_embeddings[:, plan_prefix_length:]), dim=1
        )

    def build_inputs(
        self,
        prompt_prefix_ids: torch.Tensor,
        assistant_prefix_ids: torch.Tensor,
        target_ids: torch.Tensor,
        message_embeddings: torch.Tensor,
        bop_id: int,
        eop_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.actor.get_input_embeddings()
        prompt = embedding(prompt_prefix_ids)
        assistant_prefix = embedding(assistant_prefix_ids)
        target = embedding(target_ids)
        batch_size = prompt_prefix_ids.size(0)
        bop, eop = self._boundary_embeddings(bop_id, eop_id, batch_size)
        # Interlat inserts the latent message after the human/user turn and
        # before the assistant generation prefix, not into the answer itself.
        inputs = torch.cat((prompt, bop, message_embeddings, eop, assistant_prefix, target), dim=1)
        ignored = prompt.size(1) + bop.size(1) + message_embeddings.size(1) + eop.size(1) + assistant_prefix.size(1)
        labels = torch.cat(
            (
                torch.full((batch_size, ignored), IGNORE_INDEX, dtype=torch.long, device=inputs.device),
                target_ids,
            ),
            dim=1,
        )
        mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        return inputs, mask, labels

    def forward_message(
        self,
        prompt_prefix_ids: torch.Tensor,
        assistant_prefix_ids: torch.Tensor,
        target_ids: torch.Tensor,
        message_embeddings: torch.Tensor,
        bop_id: int,
        eop_id: int,
    ):
        inputs, mask, labels = self.build_inputs(
            prompt_prefix_ids, assistant_prefix_ids, target_ids, message_embeddings, bop_id, eop_id
        )
        return self.actor(inputs_embeds=inputs, attention_mask=mask, labels=labels, return_dict=True), labels

    @torch.inference_mode()
    def generate(
        self,
        prompt_prefix_ids: torch.Tensor,
        assistant_prefix_ids: torch.Tensor,
        sender_states: torch.Tensor,
        bop_id: int,
        eop_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        latent = self.adapt_latents(sender_states)
        embedding = self.actor.get_input_embeddings()
        prompt = embedding(prompt_prefix_ids)
        assistant_prefix = embedding(assistant_prefix_ids)
        bop, eop = self._boundary_embeddings(bop_id, eop_id, prompt_prefix_ids.size(0))
        inputs = torch.cat((prompt, bop, latent, eop, assistant_prefix), dim=1)
        mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        return self.actor.generate(
            inputs_embeds=inputs,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.actor.config.pad_token_id,
        )
