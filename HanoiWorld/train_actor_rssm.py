import os
import random
import traceback
import numpy as np
import torch
from tqdm.auto import tqdm
from collections import defaultdict
from dataclasses import asdict
from evaluate import DummyLogger
from comet_ml import Experiment
import traceback
import tools
from hanoi import HanoiWorldJEPA1
from JEPA.jepa_encoder import FrozenJEPA1VisionEncoder
from HanoiAgent import HanoiAgent
import dataprep as dataprep
from config import cfg, env, env_name, CKPT_ROOT

logger = DummyLogger()
# -------------------------
# Replay + episode state
# -------------------------
replay = {}
valid_eps = set()
episode_len = defaultdict(int)

# -------------------------
# World / agent
# -------------------------
world = HanoiWorldJEPA1(cfg)
froz_enc = FrozenJEPA1VisionEncoder(
    device=cfg.device,
    ckpt_root=CKPT_ROOT,
)
agent = HanoiAgent(cfg, froz_enc, world)

# -------------------------
# Dataset generator
# -------------------------
seq_gen = dataprep.fixed_length_generator(
    replay=replay,
    valid_eps=valid_eps,
    T=cfg.batch_length,
)

dataprep = dataprep.from_generator(
    seq_gen,
    cfg.batch_size,
)

def train():

    T = cfg.batch_length
    B = cfg.batch_size
    MAX_STEPS = 5000 # cfg.steps

    step = 0
    ep_id = 0
    done = True
    obs = None
    agent_state = None

    comet_exp = Experiment(
            api_key=os.getenv("API_KEY"),
            project_name=os.getenv("PROJECT_NAME", "hanoiworld"),
            workspace=os.getenv("WORK_SPACE"),
            auto_output_logging="simple",
            parse_args=False,
        )

    comet_exp.set_name(f"jepa_1_{MAX_STEPS}_{env_name}")
    comet_exp.log_parameters({
        "batch_length": T,
        "batch_size": B,
    })
    comet_exp.log_parameters(tools.cfg_to_dict(cfg))
    CKPT_INTERVAL = 1000  # or 1000
    try:
        with tqdm(total=MAX_STEPS, desc="Train", unit="step") as pbar:

            while step < MAX_STEPS:

                # =====================================================
                # (1) Episode reset
                # =====================================================
                if done:
                    ep_id += 1
                    obs, _ = env.reset()
                    agent_state = None
                    episode_len[ep_id] = 0

                    dataprep.add_transition(
                        replay,
                        ep_id,
                        dataprep.build_transition(
                            obs,
                            action=np.zeros(cfg.num_actions, np.float32),
                            reward=0.0,
                            discount=1.0,
                            is_first=True,
                            is_terminal=False,
                            is_last=False,
                        ),
                        dataset_size=None,
                    )

                    done = False
                    comet_exp.log_metric("episode/start", ep_id, step=step)

                # =====================================================
                # (2) Policy step (NO training yet)
                # =====================================================
                image = dataprep.extract_image(obs)

                policy_out, agent_state = agent.act(
                    obs,
                    agent_state,
                    training=True,
                )

                action = policy_out["action"].cpu().numpy()[0]

                # =====================================================
                # (3) Environment step
                # =====================================================
                next_obs, reward, terminated, truncated, _ = env.step(action)

                is_terminal = bool(terminated)
                is_last = bool(terminated or truncated)
                discount = 0.0 if is_terminal else 1.0
                done = is_last

                is_first = (episode_len[ep_id] == 0)

                dataprep.add_transition(
                    replay,
                    ep_id,
                    dataprep.build_transition(
                        next_obs,
                        action=action,
                        reward=reward,
                        discount=discount,
                        is_first=is_first,
                        is_terminal=is_terminal,
                        is_last=is_last,
                    ),
                    dataset_size=None,
                )

                episode_len[ep_id] += 1
                obs = next_obs

                # mark episode valid once it can produce a sequence
                if episode_len[ep_id] >= T:
                    valid_eps.add(ep_id)

                # =====================================================
                # (4) TRAINING STEP (offline)
                # =====================================================
                avail_seqs = dataprep.count_available_sequences(replay, valid_eps, T)

                if avail_seqs >= B:

                    # ---- sample B episodes ----
                    eps = random.choices(list(valid_eps), k=B)
                    batch = defaultdict(list)

                    for ep in eps:
                        L = len(replay[ep]["reward"])
                        start = random.randint(0, L - T)

                        for k, v in replay[ep].items():
                            batch[k].append(v[start:start+T])

                    # ---- stack to [B, T, ...] ----
                    batch = {k: np.stack(v, axis=0) for k, v in batch.items()}
                    batch = {k: torch.as_tensor(v, device=cfg.device) for k, v in batch.items()}

                    # ---- encode observations ----
                    batch["embed"] = agent.encode_sequence(batch)   # [B,T,D]

                    # ---- preprocess ----
                    batch = dataprep.preprocess(batch, cfg.device)

                    # ---- world model update ----
                    post, _, metrics = agent.world.train_model(batch)

                    # ---- logging ----
                    comet_exp.log_metrics({
                        "train/model_loss": tools.to_scalar(metrics["model_loss"]),
                        "train/kl": tools.to_scalar(metrics["kl"]),
                        "train/reward_loss": tools.to_scalar(metrics["reward_loss"]),
                        "train/cont_loss": tools.to_scalar(metrics["cont_loss"]),
                        "train/dyn_loss": tools.to_scalar(metrics["dyn_loss"]),
                        "train/rep_loss": tools.to_scalar(metrics["rep_loss"]),
                        "train/valid_eps": len(valid_eps),
                        "train/avail_seqs": avail_seqs,
                    }, step=step)

                # =====================================================
                # (5) Progress
                # =====================================================
                if step % CKPT_INTERVAL == 0 and step > 0:
                    ckpt_path = tools.save_ckpt(
                        step=step,
                        agent=agent,
                        cfg=cfg,
                        env_name=env_name,
                    )
                    
                    comet_exp.log_asset(
                        ckpt_path,
                        step=step,
                    )
                step += 1
                pbar.update(1)
                pbar.set_postfix({
                    "ep": ep_id,
                    "valid_eps": len(valid_eps),
                    "avail_seqs": avail_seqs,
                    "reward": f"{reward:.2f}",
                })

    except Exception as e:
        print(f"[FAIL-SAFE] Training crashed: {e}")
        traceback.print_exc()
        comet_exp.log_other("crash", str(e))

    finally:
        comet_exp.end()

if __name__ == "__main__":
    train()
