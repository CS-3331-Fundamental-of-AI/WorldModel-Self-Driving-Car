import torch.nn as nn

# ------------------------
# Trajectory Encoder (1D conv -> GELU -> FC -> s_traj)
# pass in the tokenizer input 
# ------------------------
class TrajEncoder(nn.Module):
    def __init__(self, traj_dim, conv_channels=64, kernel=3, out_dim=128):
        super().__init__()
        self.conv = nn.Conv1d(traj_dim, conv_channels, kernel_size=kernel, padding=kernel//2)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(conv_channels, out_dim)

    def forward(self, traj, traj_mask=None):
        # traj: [B, T, traj_dim]
        x = traj.permute(0,2,1)  # [B, traj_dim, T]
        x = self.conv(x)         # [B, conv_channels, T]
        x = self.gelu(x)
        x = x.permute(0,2,1)     # [B, T, conv_channels]
        # pool across time (masked)
        if traj_mask is not None:
            # compute masked mean
            mask = traj_mask.unsqueeze(-1).float()  # [B,T,1]
            sum_x = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = sum_x / denom
        else:
            pooled = x.mean(dim=1)
        s_traj = self.fc(pooled)
        return s_traj  # [B, out_dim]

