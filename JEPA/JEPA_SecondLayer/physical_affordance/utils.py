import os
import argparse
import math
import time
from pathlib import Path
import yaml
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import math
from .kinematic import load_json_worker

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_params(model):
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:40s}  {p.numel():,}")
            total += p.numel()
    print("-" * 60)
    print(f"Total Trainable Params: {total:,}")
    return total

# ------------------------
# Update config defaults for stability
# ------------------------
config = {
    'vicreg_sim_w': 1.0,   # small to prevent huge loss
    'vicreg_var_w': 1.0,
    'vicreg_cov_w': 0.1,
    'lambda_jepa': 1.0,
    'lambda_reg': 1.0,
    'grad_clip': 1.0,
    'use_amp': True,
    'ema_momentum': 0.995,
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 40,
    'save_dir': './checkpoints_tier2'
}

# ------------------------
# Utilities
# ------------------------
def exists(x): return x is not None

def default(val, d):
    return val if exists(val) else d

def load_config(path):
    if path and os.path.exists(path):
        import yaml
        return yaml.safe_load(open(path, 'r'))
    return None

def safe_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, path):
    torch.save(state, path)

# ------------------------
# VICReg losses (stable version)
# ------------------------
def vicreg_loss(x, y, sim_weight=1.0, var_weight=0.1, cov_weight=0.1, eps=1e-4):
    """
    x, y: [B, D] - normalized embeddings
    VICReg = invariance (MSE) + variance + covariance
    """
    # invariance
    inv = F.mse_loss(x, y)

    # variance: encourage per-dim std >= 1
    def std_loss(z):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return F.relu(1.0 - std).mean()
    var_loss = 0.5 * (std_loss(x) + std_loss(y))

    # covariance: off-diagonal elements
    def cov_loss(z):
        B, D = z.size()
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (B - 1 + eps)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / D
    cov = 0.5 * (cov_loss(x) + cov_loss(y))

    loss = sim_weight * inv + var_weight * var_loss + cov_weight * cov
    return loss, {'inv': inv.item(), 'var': var_loss.item(), 'cov': cov.item()}

def build_x_adj(G, type2id, category2id, layer2id):
    """
    Input:
        G    : networkx graph
    Output:
        x    : [N, 13] node feature matrix
        adj  : [N, N] adjacency matrix
    """
    N = G.number_of_nodes()
    F = 13  # fixed
    
    x = torch.zeros((N, F), dtype=torch.float32)

    for i, (node, data) in enumerate(G.nodes(data=True)):
        # --- Numeric features ---
        pos = data.get("pos", (0,0))
        global_pos = data.get("global_pos", (0,0))
        heading = data.get("heading", 0.0)
        speed = data.get("speed", 0.0)
        accel = data.get("acceleration", 0.0)
        size = data.get("size", [0.0,0.0,0.0])
        
        # --- Categorical IDs ---
        type_id = type2id.get(data.get("type"), 0)
        category_id = category2id.get(data.get("category"), 0)
        layer_id = layer2id.get(data.get("layer"), 0)
        
        x[i] = torch.tensor([
            pos[0], pos[1],
            global_pos[0], global_pos[1],
            heading,
            speed,
            accel,
            size[0], size[1], size[2],
            float(type_id),
            float(category_id),
            float(layer_id)
        ])

    # --- adjacency (dense) ---
    A = nx.to_numpy_array(G)
    adj = torch.tensor(A, dtype=torch.float32)

    return x, adj

def build_graph_batch(graph_list, type2id, category2id, layer2id):
    xs, adjs = [], []
    maxN = max([g.number_of_nodes() for g in graph_list])
    
    for G in graph_list:
        x, adj = build_x_adj(G, type2id, category2id, layer2id)
        
        # pad node features
        padN = maxN - x.size(0)
        if padN > 0:
            x = torch.cat([x, torch.zeros(padN, x.size(1))], dim=0)
            adj = torch.cat([
                torch.cat([adj, torch.zeros(adj.size(0), padN)], dim=1),
                torch.zeros(padN, maxN)
            ], dim=0)
        
        xs.append(x)
        adjs.append(adj)
    
    x_batch = torch.stack(xs, dim=0)     # [B, maxN, 13]
    adj_batch = torch.stack(adjs, dim=0) # [B, maxN, maxN]
    
    return x_batch, adj_batch

def plot_attention(A, title="Attention Weights", vmax=None):
    """
    A: [B, N] attention weights
    """
    A = A.detach().cpu().numpy()
    B, N = A.shape

    fig, axes = plt.subplots(1, B, figsize=(3*B, 4))

    if B == 1:
        axes = [axes]

    for i in range(B):
        ax = axes[i]
        attn = A[i][None, :]       # shape → [1, N] for heatmap

        im = ax.imshow(attn, aspect="auto", cmap="viridis", vmax=vmax)
        ax.set_title(f"{title} (sample {i})")
        ax.set_ylabel("Trajectory Query")
        ax.set_xlabel("Graph Node Index")
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, N-1, 6, dtype=int))

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_sorted_attention(A, title="Sorted Attention"):
    A = A.detach().cpu().numpy()
    B, N = A.shape

    fig, axes = plt.subplots(1, B, figsize=(3*B, 4))

    if B == 1:
        axes = [axes]

    for i in range(B):
        ax = axes[i]
        attn = np.sort(A[i])[None, :]  # sort ascending
        im = ax.imshow(attn, aspect="auto", cmap="viridis")
        ax.set_title(f"{title} (sample {i})")
        ax.set_xlabel("Node Rank")
        ax.set_ylabel("Query")
        ax.set_yticks([])

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def compute_delta_stats(dataloader):
    mean_acc = None
    m2_acc = None
    count = 0
    min_val = None
    max_val = None

    # Wrap dataloader with tqdm for progress bar
    for batch in tqdm(dataloader, desc="Scanning deltas"):
        x = batch["deltas"]          # [B, T, 6]
        x = x.reshape(-1, x.shape[-1])  # flatten to [B*T, 6]
        
        b = x.shape[0]               # number of samples in this batch
        batch_mean = x.mean(dim=0)

        if mean_acc is None:
            # ---------- Initialize accumulators ----------
            mean_acc = batch_mean
            m2_acc = ((x - batch_mean) ** 2).sum(dim=0)
            min_val = x.min(dim=0).values
            max_val = x.max(dim=0).values
            count = b
        else:
            # ---------- Welford online update ----------
            delta = batch_mean - mean_acc
            total = count + b

            mean_acc = mean_acc + delta * (b / total)

            m2_acc = m2_acc + ((x - batch_mean)**2).sum(dim=0) \
                     + delta**2 * count * b / total

            # ---------- min/max update ----------
            min_val = torch.min(min_val, x.min(dim=0).values)
            max_val = torch.max(max_val, x.max(dim=0).values)

            count = total

    # Final std
    std_acc = torch.sqrt(m2_acc / count)

    return mean_acc, std_acc, min_val, max_val

# # -----------------------------------------
# # Run it
# # -----------------------------------------
# mean, std, minv, maxv = compute_delta_stats(loader)

# print("\n=== DELTA STATISTICS ===")
# print("Mean:\n", mean)
# print("Std:\n", std)
# print("Min:\n", minv)
# print("Max:\n", maxv)


def visualize_deltas(batch, sample_idx=0):
    traj = batch["deltas"][sample_idx]  # tensor [T, 6]
    
    # Ensure CPU + numpy
    traj = traj.detach().cpu().numpy()

    plt.figure(figsize=(10,5))

    plt.plot(traj[:,0], label="dx")
    plt.plot(traj[:,1], label="dy")
    plt.plot(traj[:,4], label="ds_forward")
    plt.plot(traj[:,5], label="ds_side")

    plt.title(f"Delta Curves for Sample {sample_idx}")
    plt.xlabel("Frame step (t, t-1, ..., t-8)")
    plt.ylabel("Delta Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# k=0
# for batch in loader:
#     visualize_deltas(batch, sample_idx=0)
#     if (k < 100):
#         k = k + 1
#     else:
#         break
#     # break   # remove this to inspect more


def build_backward_trajectory_windows(frames, window_size=4):
    N = len(frames)
    new_frames = []

    for i in range(N):
        positions = []

        # Collect back in time
        for k in range(window_size):
            idx = i - k
            if idx < 0:
                idx = 0  # clamp

            js = frames[idx]

            # Extract ego fields
            x = js["ego_state"]["position"]["x"]
            y = js["ego_state"]["position"]["y"]
            yaw = js["ego_state"]["heading"]      # radians
            speed = js["ego_state"]["speed"]
            timestamp = js["timestamp"]

            vx = speed * math.cos(yaw)
            vy = speed * math.sin(yaw)

            positions.append({
                "x": x,
                "y": y,
                "yaw": yaw,
                "vx": vx,
                "vy": vy,
                "timestamp": timestamp
            })

        # Attach window to frame
        js = frames[i]
        js["trajectory_window"] = {
            "num_frames": window_size,
            "positions": positions
        }

        new_frames.append(js)

    return new_frames

def load_json_worker_test(path):
    with open(path, "r") as f:
        return json.load(f)

def load_all_metadata_parallel_test(metadata_dir):
    metadata_dir = Path(metadata_dir)
    file_list = sorted(metadata_dir.glob("*.json"))

    if len(file_list) == 0:
        raise RuntimeError("No metadata JSON found in directory.")

    num_workers = min(max(1, multiprocessing.cpu_count() - 1), 16)
    print(f"Loading {len(file_list)} metadata files using {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        json_list = list(executor.map(load_json_worker, file_list))

    print(f"Loaded {len(json_list)} metadata files.")
    return json_list

def group_by_scene_test(json_list):
    scenes = {}
    for js in json_list:
        scene = js["scene_name"]
        scenes.setdefault(scene, []).append(js)
    return scenes

def build_scene_mapping_parallel_test(metadata_dir):
    # 1. Load all metadata
    json_list = load_all_metadata_parallel_test(metadata_dir)

    # 2. Group into scenes
    scenes = group_by_scene_test(json_list)

    # 3. Process each scene
    final_scene_map = {}

    for scene_name, frames in scenes.items():
        # sort by sample_idx (your new metadata’s timeline)
        frames_sorted = sorted(frames, key=lambda js: js["sample_idx"])

        # add 4-frame backward windows
        frames_with_windows = build_backward_trajectory_windows(frames_sorted)

        final_scene_map[scene_name] = frames_with_windows

    print(f"Finished processing {len(final_scene_map)} scenes.")
    return final_scene_map


