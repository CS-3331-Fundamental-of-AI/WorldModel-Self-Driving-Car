import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, codebook_size=256, num_layers=4, d_model=128):
        super().__init__()
        total_vocab = codebook_size * num_layers   # e.g., 256 * 4 = 1024
        self.embed = nn.Embedding(total_vocab, d_model)

    def forward(self, token_ids):
        # token_ids already include the offset
        return self.embed(token_ids) # B x L x d_model (128)