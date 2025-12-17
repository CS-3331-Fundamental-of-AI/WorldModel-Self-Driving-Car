import torch
from config.config import EMA_JEPA1

class LatentBuffer:
    def __init__(self, embed_dim, ema_decay=EMA_JEPA1):
        self.prev_z_c = None
        self.ema_decay = ema_decay

    def store_prev_z_c(self, z_c_pooled):
        self.prev_z_c = z_c_pooled.detach().clone()

    def get_prev_z_c(self):
        return self.prev_z_c

def init_target_from_online(online, target):
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad = False

def ema_update(online, target, decay=EMA_JEPA1):
    with torch.no_grad():
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(decay).add_(p_o.data * (1 - decay))
