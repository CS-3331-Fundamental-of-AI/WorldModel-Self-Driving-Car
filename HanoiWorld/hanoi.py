import copy
import torch

from ActorCritic import MLP
from RSSM import RSSM
import tools


to_np = lambda x: x.detach().cpu().numpy()

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class HanoiWorld(torch.nn.Module):
    """
    DreamerV3-style World Model without an encoder.
    Contains:
        - RSSM dynamics model
        - Actor
        - Critic (Value)
        - Reward model
        - Continuation model
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self._use_amp = True if getattr(config, "precision", 32) == 16 else False
        self.device = config.device

        # -------------------------------
        # Compute RSSM feature size
        # -------------------------------
        if config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter

        # -------------------------------
        # Actor
        # -------------------------------
        self.actor = MLP(
            inp_dim=self.feat_size,
            shape=(config.num_actions,),
            layers=config.actor["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist=config.actor['dist'],
            std=config.actor['std'],
            min_std=config.actor['min_std'],
            max_std=config.actor['max_std'],
            absmax=1.0,
            temp=config.actor['temp'],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        ).to(self.device)

        # -------------------------------
        # Critic (Value)
        # -------------------------------
        self.value = MLP(
            self.feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=self.device,
            name="Value",
        ).to(self.device)

        # Slow-target critic
        if config.critic["slow_target"]:
            self.slow_value = copy.deepcopy(self.value)
            self.value_updates = 0
        else:
            self.slow_value = None
            self.value_updates = 0

        # -------------------------------
        # Reward model
        # -------------------------------
        self.reward = MLP(
            self.feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=self.device,
            name="Reward",
        ).to(self.device)

        # -------------------------------
        # Continuation model
        # -------------------------------
        self.cont = MLP(
            self.feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=self.device,
            name="Cont",
        ).to(self.device)

        # -------------------------------
        # RSSM Dynamics (NO encoder → embed=0)
        # -------------------------------
        self.rssm = RSSM(
            stoch=config.dyn_stoch,
            deter=config.dyn_deter,
            hidden=config.units,
            rec_depth=config.dyn_rec_depth,
            discrete=config.dyn_discrete,
            num_actions=config.num_actions,
            embed=config.embed,                   # <---- No encoder version
            device=self.device,
        ).to(self.device)

        # -------------------------------
        # Optimizers and helpers
        # -------------------------------
        self._model_opt = tools.Optimizer(
            "model",
            self._model_params(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )

        if self.config.reward_EMA:
            self.register_buffer("ema_vals", torch.zeros((2,), device=self.device))
            self.reward_ema = RewardEMA(device=self.device)

        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    # ---------------------------------------------------------------
    # Convenience API (Dreamer-style)
    # ---------------------------------------------------------------

    def initial_state(self, batch_size=1):
        return self.rssm.initial(batch_size)

    def imagine_step(self, prev_state, action):
        """RSSM prior-only step: used for imagination rollout."""
        return self.rssm.img_step(prev_state, action)

    def get_feat(self, state):
        """Return concatenated RSSM latent = [stoch || deter]."""
        return self.rssm.get_feat(state)

    def actor_action(self, feat):
        """Sample or get action distribution from actor."""
        return self.actor(feat)

    def value_estimate(self, feat):
        return self.value(feat)

    def reward_estimate(self, feat):
        return self.reward(feat)

    def cont_estimate(self, feat):
        return self.cont(feat)

    # Optional: update slow target critic
    def update_slow_target(self, tau=0.01):
        if self.slow_value is None:
            return
        for p, p_targ in zip(self.value.parameters(), self.slow_value.parameters()):
            p_targ.data.mul_(1 - tau).add_(p.data * tau)

    # ---------------------------------------------------------------
    # Training utilities (adapted from DreamerV3)
    # ---------------------------------------------------------------

    def preprocess(self, obs):
        obs = {k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in obs.items()}
        if "discount" in obs:
            obs["discount"] *= self.config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def _select_embed(self, data):
        if "embed" in data:
            return data["embed"]
        if "state" in data:
            return data["state"]
        if "observation" in data:
            return data["observation"]
        raise KeyError("Expected an 'embed' (or 'state'/'observation') key in data.")

    def _model_params(self):
        return (
            list(self.rssm.parameters())
            + list(self.reward.parameters())
            + list(self.cont.parameters())
        )

    def _train(self, data):
        data = self.preprocess(data)
        with tools.RequiresGrad(self):
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._use_amp and self.device.type == "cuda",
            ):
                embed = self._select_embed(data)
                post, prior = self.rssm.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self.config.kl_free
                dyn_scale = self.config.dyn_scale
                rep_scale = self.config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.rssm.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                preds = {}
                feat = self.rssm.get_feat(post)
                for name, head in (("reward", self.reward), ("cont", self.cont)):
                    grad_head = name in self.config.grad_heads
                    head_inp = feat if grad_head else feat.detach()
                    preds[name] = head(head_inp)

                losses = {
                    name: -pred.log_prob(data[name]) for name, pred in preds.items()
                }
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self._model_params())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self._use_amp and self.device.type == "cuda",
        ):
            metrics["prior_ent"] = to_np(
                torch.mean(self.rssm.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.rssm.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.rssm.get_feat(post),
                kl=kl_value,
                postent=self.rssm.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def train_behavior(self, start, objective):
        self._update_slow_target_value()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._use_amp and self.device.type == "cuda",
            ):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self.config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self.rssm.get_dist(imag_state).entropy()
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat, imag_action, target, weights, base
                )
                actor_loss -= self.config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._use_amp and self.device.type == "cuda",
            ):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())
                slow_target = self.slow_value(value_input[:-1].detach())
                if self.config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self.config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self.rssm
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if self.config.cont_head:
            inp = self.rssm.get_feat(imag_state)
            discount = self.config.discount * self.cont(inp).mean
        else:
            discount = self.config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        target = torch.stack(target, dim=1)
        if self.config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        else:
            adv = target - base

        if self.config.imag_gradient == "dynamics":
            actor_target = adv
        elif self.config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self.config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self.config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self.config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target_value(self):
        if self.config.critic["slow_target"]:
            if self.value_updates % self.config.critic["slow_target_update"] == 0:
                mix = self.config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self.slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self.value_updates += 1
