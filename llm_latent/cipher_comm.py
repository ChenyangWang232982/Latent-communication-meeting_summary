import torch
import torch.nn as nn
import torch.nn.functional as F


class CipherEmbeddingComm(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        latent_len=8,
        num_heads=8,
        temperature=1.0,
        use_compression=True,
    ):
        super().__init__()

        self.latent_len = latent_len
        self.temperature = temperature
        self.use_compression = use_compression

        self.query_tokens = nn.Parameter(
            torch.randn(latent_len, hidden_size) * 0.02
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        self.vocab_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(
        self,
        sender_hidden_states,
        receiver_embedding_weight,
        sender_attention_mask=None,
    ):
        batch_size = sender_hidden_states.size(0)

        if self.use_compression:
            queries = self.query_tokens.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )

            key_padding_mask = None
            if sender_attention_mask is not None:
                key_padding_mask = sender_attention_mask == 0

            latent_states, _ = self.attention(
                query=queries,
                key=sender_hidden_states,
                value=sender_hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            latent_mask = torch.ones(
                batch_size,
                self.latent_len,
                dtype=torch.long,
                device=sender_hidden_states.device,
            )
        else:
            # No compression: one continuous CIPHER message per sender token.
            latent_states = sender_hidden_states
            if sender_attention_mask is None:
                latent_mask = torch.ones(
                    batch_size,
                    sender_hidden_states.size(1),
                    dtype=torch.long,
                    device=sender_hidden_states.device,
                )
            else:
                latent_mask = sender_attention_mask.to(dtype=torch.long)

        vocab_logits = self.vocab_projector(latent_states)

        probs = F.softmax(
            vocab_logits / self.temperature,
            dim=-1,
        )

        latent_embeds = probs @ receiver_embedding_weight

        latent_embeds = F.layer_norm(
            latent_embeds,
            latent_embeds.shape[-1:],
        )
        latent_embeds = latent_embeds * latent_mask.unsqueeze(-1).to(
            latent_embeds.dtype
        )

        return latent_embeds, latent_mask
