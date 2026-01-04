from types import SimpleNamespace
env_name="highway" # roundabout, merge, highway
args = SimpleNamespace(
    config=env_name,
    steps=10000,                 # e.g. 500_000
    logdir="runs/rssm_jepa",
    # embed_dim=128,
    device=None,                # "cuda", "cpu", "mps" or None (auto)
)

from tools import Logger
import argparse
import pathlib
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm
from evaluate import load_config, make_env  # reuse helpers

import os
from dotenv import load_dotenv
load_dotenv()

CKPT_ROOT = os.getenv("JEPA_CKPT_ROOT")


# train on highway

if CKPT_ROOT is None:
    raise RuntimeError(
        "JEPA_CKPT_ROOT is not set. "
        "Please define it in .env or environment variables."
    )

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

cfg.traj_len = 8  # use 8 or whatever history length you want
cfg.graph_nodes = 64
cfg.graph_feat  = 13
# Default trajectory and graph sizes
cfg.traj_len     = getattr(cfg, "traj_len", 8)
cfg.graph_nodes  = getattr(cfg, "graph_nodes", 13)   # adjust to your graph builder
# cfg.graph_feat   = getattr(cfg, "graph_feat", 13)      # adjust to your graph builder
# Replay buffer stored as episodes (OrderedDict of arrays).
CKPT_ROOT = os.getenv("JEPA_CKPT_ROOT")
logger = Logger(cfg.logdir, step=0)
cfg.prefill=2500
cfg.dataset_size=20000


