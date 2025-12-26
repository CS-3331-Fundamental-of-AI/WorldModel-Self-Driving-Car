import torch
import torch.nn as nn
import torch.nn.functional as F

class DeterministicKinematicBicycle(nn.Module):
    def __init__(self, state_dim=64, action_dim=2, k=6, wheelbase=2.5):
        super().__init__()
        self.k = k
        self.low_dim = 4
        self.state_dim = state_dim
        self.wheelbase = wheelbase

        self.expand = nn.Sequential(
            nn.Linear(self.low_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

    def forward(self, action, init_low_state=None):
        B = action.size(0)
        dt = 0.1
        device = action.device

        if init_low_state is None:
            s = torch.zeros(B, 4, device=device)
            s[:, 3] = 0.1
        else:
            s = init_low_state.clone()

        accel = action[:, 0]
        steer = action[:, 1].clamp(-0.5, 0.5)

        seq = []
        for _ in range(self.k):
            x, y, yaw, v = s.unbind(-1)

            beta = torch.atan(torch.tan(steer) / 2)
            dx = v * torch.cos(yaw + beta) * dt
            dy = v * torch.sin(yaw + beta) * dt
            dyaw = (v / self.wheelbase) * torch.tan(steer) * dt
            dv = accel * dt

            s = torch.stack([
                x + dx,
                y + dy,
                (yaw + dyaw + torch.pi) % (2 * torch.pi) - torch.pi,
                (v + dv).clamp(min=0)
            ], dim=-1)

            seq.append(s)

        low_seq = torch.stack(seq, dim=1)
        expanded = self.expand(low_seq.reshape(-1, 4))
        return expanded.reshape(B, self.k, self.state_dim)
