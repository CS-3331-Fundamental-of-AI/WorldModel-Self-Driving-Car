import os
import pathlib
import sys
import numpy as np
import torch
from torch import nn

sys.path.append(str(pathlib.Path(__file__).parent))

import tools
from encoder import FrozenEncoder
from hanoi import HanoiWorld

to_np = lambda x: x.detach().cpu().numpy()

class HanoiAgent(nn.Module):
    """
    Dreamer-style agent for the HanoiWorld setup.
    Uses a frozen pretrained encoder to produce embeddings that feed the RSSM.
    """

    def __init__(self, config, logger, dataset=None, encoder=None):
        super().__init__()
        self._config = config
        self._logger = logger
        self._dataset = dataset
        self._metrics = {}

        batch_steps = config.batch_size * config.batch_length
        self._should_log = tools.Every(config.log_every)
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))

        self._step = logger.step // config.action_repeat
        self._update_count = 0

        # Encoder and world model
        self.encoder = (
            encoder
            if encoder is not None
            else FrozenEncoder(out_dim=config.embed, device=config.device)
        )
        self._wm = HanoiWorld(config)

        #Optional compilation
        if getattr(config, "compile", False) and os.name != "nt":
            self._wm = torch.compile(self._wm)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step

        if training and self._dataset is not None:
            do_train = False

            if self._should_pretrain():
                do_train = True
            elif self._should_train(step):
                do_train = True

            if do_train:
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics.setdefault("update_count", []).append(self._update_count)

            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                self._metrics = {}
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step

        return policy_output, state


    def _policy(self, obs, state, training):
        # state carries (latent, prev_action)
        latent, action = state if state is not None else (None, None)
        print("Initial latent:", latent.shape if latent is not None else None)
        print("Initial action:", action.shape if action is not None else None)
        obs = self._prepare_obs(obs, latent is None)

        # Produce embedding with frozen encoder
        embed = self.encode_images(obs["image"])  # (B, 1, E)
        embed = embed[:, 0]                       # (B, E)
        print("policy image:", obs["image"].shape)
        obs["embed"] = embed

        # --- Debug prints ---
        print("embed:", embed.shape)
        print("is_first:", obs["is_first"].shape)
        # ------------------
        
        if latent is None:
            latent = self._wm.rssm.initial(embed.shape[0])
            latent = {k: v.unsqueeze(1) for k, v in latent.items()}

        if action is None:
            action = torch.zeros(
                embed.shape[0], self._config.num_actions,
                device=embed.device
            )
            
        is_first = obs["is_first"][:, 0]

        latent, _ = self._wm.rssm.obs_step(latent, action, embed, is_first)
        # Remove time dimension for policy
        latent = {k: v[:, 0] for k, v in latent.items()}
        if getattr(self._config, "eval_state_mean", False):
            latent["stoch"] = latent["mean"]
        feat = self._wm.get_feat(latent)
        print("feat:", feat.shape)

        if not training:
            actor = self._wm.actor(feat)
            action = actor.mode()
            logprob = None
        elif self._should_expl(self._step):
            actor = self._wm.actor(feat)
            action = actor.sample()
            logprob = actor.log_prob(action)
        else:
            actor = self._wm.actor(feat)
            action = actor.sample()
            logprob = actor.log_prob(action)

        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        print("action:", action.shape)
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
        print("final action:", action.shape)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        # Prepare embeddings for the batch
        img = torch.tensor(data["image"], device=self._config.device)
        emb = self.encode_images(img)   # (B, T, embed)
        print("train image:", img.shape)
        print("train embed:", emb.shape)
        data["embed"] = emb

        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.reward(self._wm.get_feat(s)).mode()
        _, _, _, _, beh_metrics = self._wm.train_behavior(start, reward)
        metrics.update(beh_metrics)

        for name, value in metrics.items():
            self._metrics.setdefault(name, []).append(value)

    def _prepare_obs(self, obs, is_first_episode):
        obs_dict = {}
        for key, val in obs.items():
            # Accept numpy, torch tensors, or python scalars/lists.
            if isinstance(val, torch.Tensor):
                tensor = val
            elif isinstance(val, np.ndarray):
                # Preserve dtype for uint8 images; otherwise cast to float32.
                dtype = torch.uint8 if val.dtype == np.uint8 else torch.float32
                tensor = torch.as_tensor(val, dtype=dtype)
            elif np.isscalar(val):
                tensor = torch.tensor(val, dtype=torch.float32)
            else:
                # Fallback for lists/tuples; torch handles numeric sequences.
                tensor = torch.as_tensor(val, dtype=torch.float32)

            # Add batch dimension.
            if tensor.ndim == 0:
                tensor = tensor.view(1, 1)
            else:
                tensor = tensor.unsqueeze(0)
            obs_dict[key] = tensor.to(self._config.device)

        if "is_first" not in obs_dict:
            is_first = 1.0 if is_first_episode else 0.0
            obs_dict["is_first"] = torch.tensor(
                [[is_first]], dtype=torch.float32, device=self._config.device
            )
        if "is_terminal" not in obs_dict:
            obs_dict["is_terminal"] = torch.zeros_like(obs_dict["is_first"])
        if "discount" not in obs_dict:
            obs_dict["discount"] = torch.ones_like(obs_dict["is_first"])
        return obs_dict
    
    def encode_images(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
            
        assert x.dim() in (4, 5), f"Unexpected image dim: {x.shape}"
        assert x.shape[-1] in (1, 3), f"Expected channel-last image, got {x.shape}"

        # TRAINING PATH (sequence)
        if x.dim() == 5:
            b, t, h, w, c = x.shape
            x = x.view(b * t, h, w, c)
            with torch.no_grad():
                emb = self.encoder(x)
            return emb.view(b, t, -1)

        # POLICY PATH (single step)
        with torch.no_grad():
            return self.encoder(x)


