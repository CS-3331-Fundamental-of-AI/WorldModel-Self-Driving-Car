import torch
from encoder import FrozenEncoder
from HanoiAgent import HanoiAgent
from types import SimpleNamespace

# ----------------------------
# Dummy config
# ----------------------------
config = SimpleNamespace(
    batch_size=1,
    batch_length=1,
    log_every=1,
    train_ratio=1,
    reset_every=1,
    expl_until=10,
    action_repeat=1,
    embed=128,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    pretrain=1,
    eval_state_mean=False,
    num_actions=3,
    actor={"dist": "onehot_gumble"},
    # --- Required by HanoiWorld ---
    dyn_discrete=True,
    num_discs=3,
    max_steps=100,
    reward_type="sparse",
)

# ----------------------------
# Dummy logger
# ----------------------------
class DummyLogger:
    step = 0
    def scalar(self, name, val): pass
    def write(self, fps=False): pass

logger = DummyLogger()

# ----------------------------
# Initialize encoder and agent
# ----------------------------
encoder = FrozenEncoder(out_dim=config.embed, device=config.device)
agent = HanoiAgent(config, logger, encoder=encoder)

# ----------------------------
# Dummy observation
# ----------------------------
B, H, W, C = 1, 256, 256, 3
dummy_obs = {"image": torch.randint(0, 256, (B, H, W, C), dtype=torch.uint8)}

# ----------------------------
# Check obs preprocessing
# ----------------------------
obs_dict = agent._prepare_obs(dummy_obs, True)
print("Preprocessed obs['image'] shape:", obs_dict["image"].shape)
assert obs_dict["image"].shape[1] in [1, 3], "Channels should be 1 or 3 (C,H,W)"

# ----------------------------
# Forward through encoder
# ----------------------------
z = encoder(obs_dict["image"])
print("Encoder output shape:", z.shape)

# ----------------------------
# RSSM step
# ----------------------------
latent, _ = agent._wm.rssm.obs_step(None, None, z, obs_dict["is_first"])
print("RSSM latent keys:", latent.keys())
for k, v in latent.items():
    print(f"{k} shape: {v.shape}, NaNs: {torch.isnan(v).any().item()}, Infs: {torch.isinf(v).any().item()}")
