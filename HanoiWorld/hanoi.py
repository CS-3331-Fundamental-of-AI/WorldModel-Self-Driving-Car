import copy
import torch
import numpy as np
import tools
from ActorCritic import MLP
from RSSM import RSSM

to_np = lambda x: x.detach().cpu().numpy()

class HanoiWorldJEPA1(torch.nn.Module):
    """
    DreamerV3-style World Model (encoder lives outside).
    Components:
      - RSSM (dynamics)
      - Heads: actor, value, reward, cont
      - Optimizers: model_opt, actor_opt, value_opt
    """

    # -----------------------------
    # Init
    # -----------------------------
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.device = config.device
        self.use_amp = (getattr(config, "precision", 32) == 16)

        # 1) Sizes
        self.feat_size = self._compute_feat_size(config)

        # 2) Modules
        self.actor = self._build_actor()
        self.value = self._build_value()
        self.reward = self._build_reward()
        self.cont = self._build_cont()
        self.rssm = self._build_rssm()

        # 3) Slow target value (optional)
        self.slow_value = copy.deepcopy(self.value) if config.critic["slow_target"] else None
        self.value_updates = 0
        self._use_amp = True if getattr(config, "precision", 32) == 16 else False
        

        # 4) Reward EMA (optional)
        if getattr(self.cfg, "reward_EMA", False):
            self.register_buffer("ema_vals", torch.zeros((2,), device=self.device))
            self.reward_ema = tools.RewardEMA(device=self.device)

        # 5) Loss scales
        self.loss_scales = {
            "reward": config.reward_head["loss_scale"],
            "cont": config.cont_head["loss_scale"],
        }

        # 6) Optimizers
        self.model_opt, self.actor_opt, self.value_opt = self._build_optimizers()

    # -----------------------------
    # Builders
    # -----------------------------
    def _compute_feat_size(self, cfg):
        if cfg.dyn_discrete:
            return cfg.dyn_stoch * cfg.dyn_discrete + cfg.dyn_deter
        return cfg.dyn_stoch + cfg.dyn_deter

    def _model_params(self):
        return (
            list(self.rssm.parameters())
            + list(self.reward.parameters())
            + list(self.cont.parameters())
        )

    def to_time_major(self, x):
        # [B, T, ...] → [T, B, ...]
        return x.transpose(0, 1).contiguous()
        
    def _build_actor(self):
        c = self.cfg
        return MLP(
            inp_dim=self.feat_size,
            shape=(c.num_actions,),
            layers=c.actor["layers"],
            units=c.units,
            act=c.act,
            norm=c.norm,
            dist=c.actor["dist"],
            std=c.actor["std"],
            min_std=c.actor["min_std"],
            max_std=c.actor["max_std"],
            absmax=1.0,
            temp=c.actor["temp"],
            unimix_ratio=c.actor["unimix_ratio"],
            outscale=c.actor["outscale"],
            name="Actor",
        ).to(self.device)

    def _build_value(self):
        c = self.cfg
        return MLP(
            inp_dim=self.feat_size,
            shape=(255,) if c.critic["dist"] == "symlog_disc" else (),
            layers=c.critic["layers"],
            units=c.units,
            act=c.act,
            norm=c.norm,
            dist=c.critic["dist"],
            outscale=c.critic["outscale"],
            device=self.device,
            name="Value",
        ).to(self.device)

    def _build_reward(self):
        c = self.cfg
        return MLP(
            inp_dim=self.feat_size,
            shape=(255,) if c.reward_head["dist"] == "symlog_disc" else (),
            layers=c.reward_head["layers"],
            units=c.units,
            act=c.act,
            norm=c.norm,
            dist=c.reward_head["dist"],
            outscale=c.reward_head["outscale"],
            device=self.device,
            name="Reward",
        ).to(self.device)

    def _build_cont(self):
        c = self.cfg
        return MLP(
            inp_dim=self.feat_size,
            shape=(),
            layers=c.cont_head["layers"],
            units=c.units,
            act=c.act,
            norm=c.norm,
            dist="binary",
            outscale=c.cont_head["outscale"],
            device=self.device,
            name="Cont",
        ).to(self.device)

    def _build_rssm(self):
        c = self.cfg
        return RSSM(
            stoch=c.dyn_stoch,
            deter=c.dyn_deter,
            hidden=c.units,
            rec_depth=c.dyn_rec_depth,
            discrete=c.dyn_discrete,
            num_actions=c.num_actions,
            embed=c.embed,      # encoder supplies this externally
            device=self.device,
        ).to(self.device)

    def _build_optimizers(self):
        c = self.cfg

        model_opt = tools.Optimizer(
            "model",
            self._model_params(),
            c.model_lr,
            c.opt_eps,
            c.grad_clip,
            c.weight_decay,
            opt=c.opt,
            use_amp=self._use_amp,
        )

        kw = dict(wd=c.weight_decay, opt=c.opt, use_amp=self.use_amp)

        actor_opt =  tools.Optimizer(
            "actor",
            self.actor.parameters(),
            c.actor["lr"],
            c.actor["eps"],
            c.actor["grad_clip"],
            **kw,
        )

        value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            c.critic["lr"],
            c.critic["eps"],
            c.critic["grad_clip"],
            **kw,
        )

        return model_opt, actor_opt, value_opt

    # -----------------------------
    # Public helpers / API
    # -----------------------------
    def model_parameters(self):
        return list(self.rssm.parameters()) + list(self.reward.parameters()) + list(self.cont.parameters())

    def initial_state(self, batch_size=1):
        return self.rssm.initial(batch_size)

    def get_feat(self, state):
        return self.rssm.get_feat(state)

    def update_slow_value(self):
        """Dreamer slow target update schedule (same logic as your code)."""
        c = self.cfg
        if self.slow_value is None:
            return
        if self.value_updates % c.critic["slow_target_update"] == 0:
            mix = c.critic["slow_target_fraction"]
            for s, d in zip(self.value.parameters(), self.slow_value.parameters()):
                d.data = mix * s.data + (1 - mix) * d.data
        self.value_updates += 1

    # -----------------------------
    # Preprocess + embed selection
    # -----------------------------
    # def preprocess(self, batch):
    #     # keep original behavior
    #     batch = {k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in batch.items()}
    #     if "discount" in batch:
    #         batch["discount"] *= self.cfg.discount
    #         batch["discount"] = batch["discount"].unsqueeze(-1)
    #     assert "is_first" in batch and "is_terminal" in batch
    #     batch["cont"] = (1.0 - batch["is_terminal"]).unsqueeze(-1)
    #     return batch

    def preprocess(self, batch):
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
    
        # keep float32 for continuous, but don't force everything blindly
        for k in batch:
            if batch[k].dtype in (torch.float16, torch.float32, torch.float64):
                batch[k] = batch[k].float()
    
        # discount is already [B,T,1] from your generator; only reshape if needed
        if "discount" in batch:
            batch["discount"] = batch["discount"] * self.cfg.discount
            if batch["discount"].ndim == 2:      # [B,T] -> [B,T,1]
                batch["discount"] = batch["discount"].unsqueeze(-1)

        assert "is_first" in batch and "is_terminal" in batch
    
        if batch["is_terminal"].ndim == 2:       # [B,T] -> [B,T,1]
            batch["is_terminal"] = batch["is_terminal"].unsqueeze(-1)
    
        batch["cont"] = (1.0 - batch["is_terminal"])   # already [B,T,1]
        return batch
        
    def select_embed(self, batch):
        for k in ("embed", "state", "observation"):
            if k in batch:
                return batch[k]
        raise KeyError("Expected one of: 'embed', 'state', 'observation' in batch.")

    # -----------------------------
    # Training: Model step
    # -----------------------------
    def train_model(self, batch):
        """
        Update RSSM + reward head + cont head.
        Returns: post_state, context, metrics
        """
        c = self.cfg
        batch = self.preprocess(batch)

        with tools.RequiresGrad(self):
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp and self.device.type == "cuda",
            ):
                embed = self.select_embed(batch)
                B = batch["action"].shape[1]
                if embed.shape[1] == 1 and B > 1:
                    embed = embed.expand(-1, B, -1).contiguous()
                # print("DEBUG IN HANOI WORLD")
                # print("embed:", batch["embed"].shape)

                
                embed     = self.to_time_major(embed)          # [T, B, E]
                action    = self.to_time_major(batch["action"])  # [T, B, A]
                is_first  = self.to_time_major(batch["is_first"])# [T, B, 1]
                
                # post, prior = self.rssm.observe(embed, batch["action"], batch["is_first"])

                # print(f"THE SHAPE of embed {embed.shape}")
                # print(f"The Shape of action {action.shape}")
                # print(f"The Shape of is_first {is_first.shape}")
                
                post, prior = self.rssm.observe(embed, action, is_first)
                # print("deter:", prior["deter"].shape)

                kl_loss, kl_value, dyn_loss, rep_loss = self.rssm.kl_loss(
                    post, prior, c.kl_free, c.dyn_scale, c.rep_scale
                )

                # print("THE LOSS")
                # print(f"kl_loss {kl_loss}")
                # print(f"kl_value {kl_value}")
                # print(f"dyn_loss {dyn_loss}")
                # print(f"rep_loss {rep_loss}")

                feat = self.rssm.get_feat(post)

                preds = {}
                for name, head in (("reward", self.reward), ("cont", self.cont)):
                    head_inp = feat if (name in c.grad_heads) else feat.detach()
                    preds[name] = head(head_inp)

                # nll = {name: -pred.log_prob(batch[name]) for name, pred in preds.items()}
                targets = {name: self.to_time_major(batch[name]) for name in preds}
                nll = {name: -pred.log_prob(targets[name]) for name, pred in preds.items()}
                
                scaled = {k: v * self.loss_scales.get(k, 1.0) for k, v in nll.items()}
                model_loss = sum(scaled.values()) + kl_loss

            opt_metrics = self.model_opt(torch.mean(model_loss), self.model_parameters())

        metrics = dict(opt_metrics)
        metrics.update({f"{k}_loss": to_np(v) for k, v in nll.items()})
        metrics.update({
            "kl_free": c.kl_free,
            "dyn_scale": c.dyn_scale,
            "rep_scale": c.rep_scale,
            "dyn_loss": to_np(dyn_loss),
            "rep_loss": to_np(rep_loss),
            "kl": to_np(torch.mean(kl_value)),
        })

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp and self.device.type == "cuda",
        ):
            metrics["prior_ent"] = to_np(torch.mean(self.rssm.get_dist(prior).entropy()))
            metrics["post_ent"] = to_np(torch.mean(self.rssm.get_dist(post).entropy()))
            context = {
                "embed": embed,
                "feat": self.rssm.get_feat(post),
                "kl": kl_value,
                "postent": self.rssm.get_dist(post).entropy(),
            }

        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # -----------------------------
    # Training: Behavior step (actor + value)
    # -----------------------------
    def train_behavior(self, start, objective):
        """
        Same behavior training as before, but clearly separated.
        """
        self.update_slow_value()
        c = self.cfg
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp and self.device.type == "cuda",
            ):
                imag_feat, imag_state, imag_action = self._imagine(start, self.actor, c.imag_horizon)
                reward = objective(imag_feat, imag_state, imag_action)

                actor_ent = self.actor(imag_feat).entropy()
                target, weights, base = self._compute_target(imag_feat, imag_state, reward)

                actor_loss, mets = self._actor_loss(imag_feat, imag_action, target, weights, base)
                actor_loss -= c.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp and self.device.type == "cuda",
            ):
                value = self.value(imag_feat[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())

                if self.slow_value is not None:
                    slow_target = self.slow_value(imag_feat[:-1].detach()).mode().detach()
                    value_loss -= value.log_prob(slow_target)

                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))

        if c.actor["dist"] in ["onehot"]:
            metrics.update(tools.tensorstats(torch.argmax(imag_action, dim=-1).float(), "imag_action"))
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))

        with tools.RequiresGrad(self):
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self.value_opt(value_loss, self.value.parameters()))

        return imag_feat, imag_state, imag_action, weights, metrics

    # -----------------------------
    # Imagination + targets (unchanged logic)
    # -----------------------------
    def _imagine(self, start, policy, horizon):
        dynamics = self.rssm
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            action = policy(feat.detach()).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(step, [torch.arange(horizon)], (start, None, None))
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        c = self.cfg
        if c.cont_head:
            inp = self.rssm.get_feat(imag_state)
            discount = c.discount * self.cont(inp).mean
        else:
            discount = c.discount * torch.ones_like(reward)

        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:], value[:-1], discount[1:], bootstrap=value[-1],
            lambda_=c.discount_lambda, axis=0
        )
        weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
        return target, weights, value[:-1]

    def _actor_loss(self, imag_feat, imag_action, target, weights, base):
        c = self.cfg
        metrics = {}

        policy = self.actor(imag_feat.detach())
        target = torch.stack(target, dim=1)

        if getattr(c, "reward_EMA", False):
            offset, scale = self.reward_ema(target, self.ema_vals)
            adv = (target - offset) / scale - (base - offset) / scale
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        else:
            adv = target - base

        if c.imag_gradient == "dynamics":
            actor_target = adv
        elif c.imag_gradient == "reinforce":
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
                (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif c.imag_gradient == "both":
            reinforce = policy.log_prob(imag_action)[:-1][:, :, None] * (
                (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = c.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * reinforce
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(c.imag_gradient)

        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics