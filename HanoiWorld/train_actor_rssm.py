#!/usr/bin/env python3
"""
Minimal training script for HanoiWorld/HanoiAgent (RSSM + Actor/Critic).

This is a lightweight loop that:
 - loads config from configs.yaml (defaults + chosen profile)
 - builds a single Highway-family environment
 - collects episodes into an in-memory replay
 - samples batches to train the world model + actor/critic via HanoiAgent

It is intentionally simple (single env, no multiprocessing) to keep it easy to
adapt or extend.
"""

import argparse
import pathlib
from collections import OrderedDict

import numpy as np
from .encoder import FrozenEncoder
import torch
from tqdm import tqdm

import tools
from HanoiAgent import HanoiAgent
from evaluate import load_config, make_env  # reuse helpers
from comet_ml import Experiment
import os
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Train RSSM + Actor on HanoiWorld")
    parser.add_argument("--config", type=str, default="highway", help="Config name from configs.yaml")
    parser.add_argument("--logdir", type=str, default="runs/train", help="Logging directory")
    parser.add_argument("--steps", type=int, default=None, help="Override total environment steps")
    parser.add_argument("--render_every", type=int, default=0, help="Render every N env steps during rollout (0 disables)")
    parser.add_argument("--render_training", action="store_true", default=False, help="Enable visualization during training rollout")
    parser.add_argument("--prefill_progress", action="store_true", default=False, help="Show tqdm bar during prefill")
    args = parser.parse_args()

    # Comet initialization (fail-safe) using .env variables
    comet_exp = None
    try:
        comet_exp = Experiment(
            api_key=os.getenv("API_KEY"),
            project_name=os.getenv("PROJECT_NAME", "hanoiworld"),
            workspace=os.getenv("WORK_SPACE"),
            auto_output_logging="simple",
            parse_args=False,
        )
        comet_exp.set_name(f"train_{args.config}")
        comet_exp.log_parameters(vars(args))
    except Exception as e:
        print(f"[WARN] Comet disabled: {e}")
        comet_exp = None

    # Load config and set runtime overrides.
    cfg = load_config([args.config])
    if args.steps is not None:
        cfg.steps = args.steps
    cfg.logdir = pathlib.Path(args.logdir)
    cfg.logdir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        cfg.device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cfg.device = torch.device("mps")
    else:
        cfg.device = torch.device("cpu")
    print(f"Using device: {cfg.device}")
    cfg.embed = 128
    # Environment.
    env = make_env(cfg)

    acts = env.action_space
    if hasattr(acts, "n"):
        cfg.num_actions = acts.n
    elif hasattr(acts, "shape"):
        cfg.num_actions = int(np.prod(acts.shape))
    else:
        raise ValueError(f"Unsupported action space: {acts}")


    env.id = "train_env"  # used as key in replay

    # Replay buffer stored as episodes (OrderedDict of arrays).
    replay = OrderedDict()

    # Dataset generator: sample full episodes then slice to batch_length.
    generator = tools.sample_episodes(replay, cfg.batch_length, seed=cfg.seed)
    dataset = tools.from_generator(generator, cfg.batch_size)

    logger = tools.Logger(cfg.logdir, step=0)
    encoder = FrozenEncoder(
        ckpt_root=cfg.jepa_ckpt_root,
        out_dim=cfg.embed,
        device=cfg.device,
    )
    agent = HanoiAgent(config=cfg, logger=logger, dataset=dataset, encoder=encoder)
    agent_state = None

    # Optional prefill with random actions to avoid empty replay deadlock.
    prefill = int(getattr(cfg, "prefill", 0))
    if prefill > 0:
        print(f"Prefilling replay with {prefill} random steps...")
        prefill_progress = tqdm(total=prefill, desc="Prefill", unit="step")
        done_prefill = True
        obs_prefill = None
        ep_id_prefill = 0
        while prefill > 0:
            if done_prefill:
                ep_id_prefill += 1
                obs_prefill, _ = env.reset()
                add_transition(
                    replay,
                    ep_id_prefill,
                    transition=build_transition(obs_prefill, action=None, reward=0.0, discount=1.0),
                    dataset_size=cfg.dataset_size,
                )
                done_prefill = False
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done_prefill = terminated or truncated
            discount = 0.0 if terminated else 1.0
            add_transition(
                replay,
                ep_id_prefill,
                transition=build_transition(
                    next_obs,
                    action=format_action(action, env.action_space, cfg.num_actions),
                    reward=reward,
                    discount=discount,
                ),
                dataset_size=cfg.dataset_size,
            )
            obs_prefill = next_obs
            prefill -= 1
            prefill_progress.update(1)
        prefill_progress.close()

    step = 0
    episode_idx = 0
    done = True
    obs = None

    try:
        progress = tqdm(total=cfg.steps, desc="Train", unit="step")
        while step < cfg.steps:
            if done:
                # Start new episode and store initial observation.
                episode_idx += 1
                obs, info = env.reset()
                add_transition(
                    replay,
                    episode_idx,
                    transition=build_transition(obs, action=None, reward=0.0, discount=1.0),
                    dataset_size=cfg.dataset_size,
                )
                done = False

            # Agent action.
            act_dict, agent_state = agent(
                {k: np.expand_dims(v, 0) for k, v in obs.items()},
                reset=np.array([False], dtype=bool),
                state=agent_state,
                training=True,
            )
            action = act_dict["action"][0].detach().cpu().numpy()
            action = format_action(action, env.action_space, cfg.num_actions)

            # Step env.
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            discount = 0.0 if terminated else 1.0

            if args.render_training and args.render_every and step % args.render_every == 0:
                try:
                    env.render()
                except Exception:
                    pass

            # Add to replay.
            add_transition(
                replay,
                episode_idx,
                transition=build_transition(next_obs, action=action, reward=reward, discount=discount),
                dataset_size=cfg.dataset_size,
            )

            obs = next_obs
            step += 1
            progress.update(1)

            # Simple logging cadence.
            if step % int(cfg.log_every) == 0:
                logger.scalar("train/steps", step)
                logger.write(step=step)
                if comet_exp:
                    comet_exp.log_metric("train_steps", step, step=step)

        progress.close()

        env.close()
        close_renderer(env)
        print("Training loop finished.")
    except Exception as e:
        print(f"[FAIL-SAFE] Training crashed: {e}")
        if comet_exp:
            comet_exp.log_other("fail_safe_error", str(e))
        # Save partial agent weights if available
        try:
            save_path = cfg.logdir / "crash_checkpoint.pt"
            torch.save(agent.state_dict(), save_path)
            print(f"Saved crash checkpoint to {save_path}")
            if comet_exp:
                comet_exp.log_asset(str(save_path))
        except Exception as ee:
            print(f"[FAIL-SAFE] Could not save crash checkpoint: {ee}")
    finally:
        if comet_exp:
            comet_exp.end()


def build_transition(obs, action, reward, discount):
    """Standardize a transition dict for replay."""
    out = dict(obs)
    out["reward"] = np.array(reward, dtype=np.float32)
    out["discount"] = np.array(discount, dtype=np.float32)
    if action is not None:
        out["action"] = np.array(action)
    return out


def add_transition(replay, episode_id, transition, dataset_size):
    """Append transition to replay, pruning old episodes beyond dataset_size steps."""
    tools.add_to_cache(replay, episode_id, transition)
    tools.erase_over_episodes(replay, dataset_size)


def format_action(action, action_space, num_actions):
    """Normalize actions to consistent shape for replay stacking."""
    if hasattr(action_space, "n"):
        # Discrete -> onehot of length num_actions
        idx = int(np.array(action).item())
        onehot = np.zeros((num_actions,), dtype=np.float32)
        onehot[idx] = 1.0
        return onehot
    return np.array(action, dtype=np.float32)


def close_renderer(env):
    """Attempt to close any underlying viewer windows for highway-env."""
    candidates = [
        getattr(env, "viewer", None),
        getattr(getattr(env, "_env", None), "viewer", None),
        getattr(getattr(getattr(env, "_env", None), "unwrapped", None), "viewer", None),
    ]
    for viewer in candidates:
        if viewer and hasattr(viewer, "close"):
            try:
                viewer.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
