import torch
import torch.nn as nn

class SpeechEmbeddingComm(nn.Module):
    def __init__(self, speech_dim, llm_dim, latent_len=32, num_heads=8, dropout=0.1):
        super().__init__()
        if speech_dim % num_heads != 0:
            raise ValueError(f"speech_dim = {speech_dim} must be divisible by num_heads = {num_heads}") #Check dimension
        self.latent_len = latent_len
        self.query_tokens = nn.Parameter(
            torch.randn(latent_len, speech_dim) * 0.02
        )
        self.attn = nn.MultiheadAttention(
            embed_dim = speech_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(speech_dim),
            nn.Linear(speech_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        self.output_norm = nn.LayerNorm(llm_dim)

    def forward(self, speech_hidden_states, speech_attention_mask=None):
        batch_size = speech_hidden_states.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # mask processing
        key_padding_mask = None
        if speech_attention_mask is not None:
            key_padding_mask = speech_attention_mask == 0

        #load speech info
        latent_speech, _ = self.attn(
            query = queries,
            key = speech_hidden_states,
            value = speech_hidden_states,
            key_padding_mask = key_padding_mask,
            need_weights = False
        )

        #project to llm dim
        latent_embeds = self.projector(latent_speech)
        latent_embeds = self.output_norm(latent_embeds)
        #latent_embeds: [batch_size, latent_len, llm_dim]

        #create latent mask
        latent_mask = torch.ones(batch_size, self.latent_len, dtype=torch.long, device=speech_hidden_states.device)

        #RETURN
        return latent_embeds, latent_mask