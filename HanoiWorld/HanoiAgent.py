import torch
import torch.nn as nn
import numpy as np


class HanoiAgent(nn.Module):
    """
    Minimal JEPA-1 Agent:
    - act() for environment interaction
    - train() for one gradient update
    - exposes latest_losses
    """

    def __init__(self, cfg, encoder, world):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder          # frozen
        self.world = world              # RSSM + actor + critic
        self.latest_losses = {}

    # ====================================================
    # Acting (Environment Interaction)
    # ====================================================
    @torch.no_grad()
    def act(self, obs, state=None, training=True):
        """
        obs: dict from env
        state: (latent, prev_action) or None
        """
        latent, prev_action = state if state is not None else (None, None)

        obs = self._prepare_obs(obs, latent is None)

        if prev_action is None:
            prev_action = torch.zeros(
                obs["image"].shape[0],
                self.cfg.num_actions,
                device=self.cfg.device,
            )

        embed = self.encode_step(obs)
        obs["embed"] = embed

        latent, _ = self.world.rssm.obs_step(
            latent, prev_action, embed, obs["is_first"]
        )

        feat = self.world.get_feat(latent)
        actor = self.world.actor(feat)

        action = actor.sample() if training else actor.mode()
        logprob = actor.log_prob(action)

        return (
            {"action": action, "logprob": logprob},
            (self._detach(latent), action.detach()),
        )

    # ====================================================
    # Learning (One Update Step)
    # ====================================================
    def train(self, batch):
        """
        batch: [B,T,...] already prepared by dataset
        returns: dict of scalar losses
        """

        assert "image" in batch, "JEPA-1 requires images"

        # Encode sequence once (JEPA-1)
        batch["embed"] = self.encode_sequence(batch)

        # World model update
        post, _, model_metrics = self.world.train_model(batch)

        # Behavior learning
        reward_fn = lambda f, s, a: self.world.reward(
            self.world.get_feat(s)
        ).mode()

        _, _, _, _, beh_metrics = self.world.train_behavior(post, reward_fn)

        # Collect losses
        self.latest_losses = self._to_scalars({
            **model_metrics,
            **beh_metrics,
        })

        return self.latest_losses

    # ====================================================
    # Encoders
    # ====================================================
    def encode_sequence(self, batch):
        image = torch.as_tensor(batch["image"], device=self.cfg.device)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        image = image.permute(0, 4, 1, 2, 3)  # [B,3,T,H,W]

        with torch.no_grad():
            embed = self.encoder(pixel_values=image)

        return embed

    def encode_step(self, obs):
        image = obs["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        image = image.unsqueeze(1).permute(0, 4, 1, 2, 3)

        with torch.no_grad():
            embed = self.encoder(pixel_values=image)

        return embed[:, -1]

    # ====================================================
    # Utilities
    # ====================================================
    def _prepare_obs(self, obs, is_first):
        out = {}
    
        for k, v in obs.items():
            t = torch.as_tensor(v, device=self.cfg.device)
    
            # Image: [H,W,C] → [1,H,W,C]
            if k == "image" and t.ndim == 3:
                t = t.unsqueeze(0)
    
            # Scalar → [1,1]
            elif t.ndim == 0:
                t = t.view(1, 1)
    
            # Vector → [1,D]
            elif t.ndim == 1:
                t = t.unsqueeze(0)
    
            out[k] = t
    
        # Ensure required Dreamer flags exist
        if "is_first" not in out:
            out["is_first"] = torch.tensor(
                [[float(is_first)]],
                device=self.cfg.device,
            )
    
        if "is_terminal" not in out:
            out["is_terminal"] = torch.zeros_like(out["is_first"])
    
        if "discount" not in out:
            out["discount"] = torch.ones_like(out["is_first"])
    
        return out

    def _detach(self, latent):
        return {k: v.detach() for k, v in latent.items()}

    def _to_scalars(self, metrics):
        out = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                out[k] = float(v.mean().detach().cpu())
            elif isinstance(v, np.ndarray):
                out[k] = float(v.mean())
            else:
                out[k] = float(v)
        return out