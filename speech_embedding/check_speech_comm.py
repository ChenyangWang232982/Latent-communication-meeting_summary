import torch

from speech_embedding.speech_comm import SpeechEmbeddingComm

def main():
    batch_size = 2
    speech_seq_len = 100
    speech_dim = 512
    llm_dim = 512
    latent_len = 32

    comm = SpeechEmbeddingComm(
        speech_dim = speech_dim,
        llm_dim = llm_dim,
        latent_len = latent_len,
        num_heads = 8,
    )

    speech_hidden_states = torch.randn(batch_size, speech_seq_len, speech_dim)
    speech_attention_mask = torch.ones(batch_size, speech_seq_len, dtype=torch.long)
    latent_embeds, latent_mask = comm(speech_hidden_states, speech_attention_mask)

    print("latent_embeds:", latent_embeds.shape)  # Expected: [batch_size, latent_len, llm_dim]
    print("latent_mask:", latent_mask.shape)      # Expected: [batch_size, latent_len]

if __name__ == "__main__":
    main()