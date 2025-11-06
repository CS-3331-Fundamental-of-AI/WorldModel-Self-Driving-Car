import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """
    Lightweight Transformer text encoder.
    LLM integration can be added by replacing this module.
    """
    def __init__(self, vocab_size=2000, embed_dim=128, latent_dim=256):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(embed_dim, latent_dim)

    def forward(self, tokens):
        emb = self.embed(tokens)          # [B, T, D]
        emb = emb.transpose(0, 1)         # Transformer needs [T, B, D]
        out = self.transformer(emb)
        pooled = out.mean(dim=0)          # [B, D]
        return self.fc(pooled)
