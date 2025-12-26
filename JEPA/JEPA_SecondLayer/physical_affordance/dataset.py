import os
import json
import torch
import random
import math
from torch.utils.data import Dataset
from .kinematic import compute_windows_safe, pad_traj, load_gpickle

# ============================================================
#  AUGMENTATION UTILITIES  (Safe for 8×6 delta matrices)
# ============================================================

def aug_small_rotation(deltas, max_deg=10.0):
    """Rotate dx,dy and ds_forward,ds_side by a small BEV angle."""
    theta = math.radians(random.uniform(-max_deg, max_deg))
    c, s = math.cos(theta), math.sin(theta)

    R = torch.tensor([[c, -s],
                      [s,  c]], dtype=deltas.dtype)

    d = deltas.clone()

    # rotate (dx, dy)
    xy = d[:, 0:2] @ R.T
    d[:, 0:2] = xy

    # rotate (ds_forward, ds_side)
    fs = d[:, 4:6] @ R.T
    d[:, 4:6] = fs

    # adjust dyaw slightly
    d[:, 3] += theta

    return d


def aug_gaussian_noise(deltas, sigma=0.01):
    return deltas + sigma * torch.randn_like(deltas)


def aug_time_scale(deltas, low=0.95, high=1.05):
    """Scale displacement-related channels."""
    scale = random.uniform(low, high)
    d = deltas.clone()
    d[:, 0] *= scale      # dx
    d[:, 1] *= scale      # dy
    d[:, 2] *= scale      # dv
    d[:, 4] *= scale      # ds_forward
    d[:, 5] *= scale      # ds_side
    # dyaw unchanged
    return d


def aug_temporal_mask(deltas, mask_prob=0.1):
    d = deltas.clone()
    T = d.shape[0]
    for t in range(T):
        if random.random() < mask_prob:
            d[t] = 0
    return d


def aug_lateral_drift(deltas, amp=0.1):
    """Smooth sinusoidal offset in lateral dimension (ds_side)."""
    d = deltas.clone()
    T = d.shape[0]
    phase = random.uniform(0, math.pi)
    for t in range(T):
        d[t, 5] += amp * math.sin( (math.pi * t / T) + phase )
    return d


def apply_safe_augmentations(deltas):
    """Mix clean and augmented, Hansen-style."""
    # 50% chance: return clean sample (prevents over-regularization)
    if random.random() < 0.5:
        return deltas

    d = deltas

    # Always small noise
    d = aug_gaussian_noise(d)

    # Random rotation
    if random.random() < 0.7:
        d = aug_small_rotation(d)

    # Occasionally time scale
    if random.random() < 0.4:
        d = aug_time_scale(d)

    # Occasionally temporal masking
    if random.random() < 0.3:
        d = aug_temporal_mask(d)

    # Occasionally lateral drift
    if random.random() < 0.25:
        d = aug_lateral_drift(d)

    return d


# ============================================================
#  UPDATED DATASET CLASS WITH AUGMENTATION SUPPORT
# ============================================================

class Tier2Dataset(Dataset):
    """
    Dataset using pre-built scene_map:
        scene_map[scene_token] = [js0, js1, js2, ..., jsN]
    """

    def __init__(self, scene_map, dataset_path, num_frames=9, augment=False):
        """
        augment: boolean → enable/disable trajectory augmentation
        """
        super().__init__()
        self.scene_map = scene_map
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.augment = augment

        # ---------------------------
        # Build flat global index
        # ---------------------------
        self.master_index = []
        for scene_token, frames in scene_map.items():
            for i in range(len(frames)):
                self.master_index.append((scene_token, i))

        print(f"[Tier2Dataset] Total frames: {len(self.master_index)}")
        print(f"[Tier2Dataset] Total scenes: {len(scene_map)}")
        print(f"[Tier2Dataset] Augmentation: {'ON' if augment else 'OFF'}")

    def __len__(self):
        return len(self.master_index)

    def _extract_frame(self, js):
        tw = js["trajectory_window"]
        pos = tw["positions"]
        return pos[0] if len(pos) > 0 else None

    def __getitem__(self, idx):
        TARGET = self.num_frames       # 9 frames
        DELTA_STEPS = TARGET - 1       # 8 deltas
        DIM = 6

        # ------------------------------------
        # Lookup (scene_token, frame index)
        # ------------------------------------
        scene_token, frame_idx = self.master_index[idx]
        seq = self.scene_map[scene_token]

        js_current = seq[frame_idx]
        frame0 = self._extract_frame(js_current)
        if frame0 is None:
            raise RuntimeError("Invalid current frame: no positions[]")

        zero_frame = {k: 0.0 for k in frame0.keys()}
        zero_frame["timestamp"] = 0

        # ------------------------------------
        # Build backward window [t, t-1, ...]
        # ------------------------------------
        raw_frames = []
        for k in range(TARGET):
            t = frame_idx - k
            if t < 0:
                raw_frames.append(zero_frame.copy())
            else:
                fr = self._extract_frame(seq[t])
                raw_frames.append(fr if fr is not None else zero_frame.copy())

        # ------------------------------------
        # Safe delta computation
        # ------------------------------------
        deltas = compute_windows_safe(raw_frames)
        deltas = pad_traj(deltas, DELTA_STEPS, DIM)   # → shape [8,6]
        deltas = torch.tensor(deltas, dtype=torch.float32)
        deltas_aug = None
        # ------------------------------------
        # OPTIONAL AUGMENTATION HERE
        # ------------------------------------
        if self.augment:
            deltas_aug = apply_safe_augmentations(deltas)

        # ------------------------------------
        # Load local graph
        # ------------------------------------
        lg = js_current["local_graph"]
        full_path = os.path.join(self.dataset_path, lg["pickle_file"])
        local_graph = load_gpickle(full_path)

        return (local_graph, deltas, js_current, deltas_aug)
    
def tier2_collate_fn(batch):
    """
    batch = list of (local_graph, deltas_clean, meta, deltas_aug)
    """

    B = len(batch)

    # ===========================
    # 1. Get max graph size
    # ===========================
    node_sizes = [G.number_of_nodes() for (G, _, _, _) in batch]
    max_nodes = max(node_sizes)

    graph_feats = []
    graph_adj   = []
    graph_mask  = []

    clean_deltas_list = []
    aug_deltas_list   = []

    traj_mask_list = []

    graphs = []
    metas  = []

    for (G, deltas_clean, meta, deltas_aug) in batch:

        # ===========================
        # (A) GRAPH FEATURES
        # ===========================
        N = G.number_of_nodes()
        node_list = list(G.nodes())

        feats = []
        for n in node_list:
            d = G.nodes[n]
            feats.append([
                d.get("heading", 0.0),
                d.get("speed", 0.0),
                d.get("acceleration", 0.0),
                d.get("pos", (0,0))[0],
                d.get("pos", (0,0))[1],
                d.get("size", [0,0,0])[0],
                d.get("size", [0,0,0])[1],
                d.get("size", [0,0,0])[2],
            ])

        feats = torch.tensor(feats, dtype=torch.float32)   # [N, F]

        pad_feats = torch.zeros((max_nodes, feats.shape[1]), dtype=torch.float32)
        pad_feats[:N] = feats

        # Adjacency matrix
        A = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)
        for u, v in G.edges():
            ui = node_list.index(u)
            vi = node_list.index(v)
            A[ui, vi] = 1.0
            A[vi, ui] = 1.0

        # Node existence mask
        mask = torch.zeros(max_nodes)
        mask[:N] = 1.0

        graph_feats.append(pad_feats)
        graph_adj.append(A)
        graph_mask.append(mask)

        # ===========================
        # (B) DELTAS: CLEAN VIEW
        # ===========================
        d_clean = deltas_clean
        clean_deltas_list.append(torch.tensor(d_clean, dtype=torch.float32))

        traj_mask = (d_clean.abs().sum(dim=1) > 0).float()
        traj_mask_list.append(traj_mask)

        # ===========================
        # (C) DELTAS: AUGMENTED VIEW
        # ===========================
        if deltas_aug is None:
            # no augmentation → use clean copy
            aug_deltas_list.append(torch.tensor(d_clean, dtype=torch.float32))
        else:
            aug_deltas_list.append(torch.tensor(deltas_aug, dtype=torch.float32))

        graphs.append(G)
        metas.append(meta)

    # ===========================
    # STACK EVERYTHING
    # ===========================
    return {
        "graphs": graphs,
        "graph_feats": torch.stack(graph_feats),   # [B, max_nodes, F]
        "graph_adj": torch.stack(graph_adj),       # [B, max_nodes, max_nodes]
        "graph_mask": torch.stack(graph_mask),     # [B, max_nodes]

        "clean_deltas": torch.stack(clean_deltas_list),     # [B, T, 6]
        "aug_deltas": torch.stack(aug_deltas_list),         # [B, T, 6]
        "traj_mask": torch.stack(traj_mask_list),           # [B, T]

        "meta": metas,
    }

def compare_node(graph_node, json_node, is_local=False):
    mismatches = []

    gx, gy = graph_node["pos"]   # tuple → x,y
    jx = json_node["x"]
    jy = json_node["y"]
    global_x = 0
    global_y = 0
    if is_local:
        global_x = graph_node['global_pos'][0]
        global_y = graph_node['global_pos'][1]
        
        if global_x != json_node['global_x']:
            mismatches.append(('global_x', global_x,  json_node['global_x']))
        if global_y != json_node['global_y']:
            mismatches.append(('global_y', global_y,  json_node['global_y']))

        if "heading" in graph_node and "heading" in json_node:
            if graph_node["heading"] != json_node["heading"]:
                mismatches.append(("heading", graph_node["heading"], json_node["heading"]))

        # acceleration
        if "acceleration" in graph_node and "acceleration" in json_node:
            if graph_node["acceleration"] != json_node["acceleration"]:
                mismatches.append(("acceleration", graph_node["acceleration"], json_node["acceleration"]))
    
        # speed
        if "speed" in graph_node and "speed" in json_node:
            if graph_node["speed"] != json_node["speed"]:
                mismatches.append(("speed", graph_node["speed"], json_node["speed"]))
    
        # size (list)
        if "size" in graph_node and "size" in json_node:
            if graph_node["size"] != json_node["size"]:
                mismatches.append(("size", graph_node["size"], json_node["size"]))

        
    # Compare x,y
    if abs(gx - jx) > 1e-6:
        mismatches.append(("x", gx, jx))

    if abs(gy - jy) > 1e-6:
        mismatches.append(("y", gy, jy))

    # Compare type
    if graph_node.get("type") != json_node["type"]:
        mismatches.append(("type", graph_node.get("type"), json_node["type"]))
    if not is_local:
        # layer
        if graph_node.get("layer") != json_node["layer"]:
            mismatches.append(("layer", graph_node.get("layer"), json_node["layer"]))
    
        # lane token
        if graph_node.get("lane_token") != json_node["lane_token"]:
            mismatches.append(("lane_token",
                               graph_node.get("lane_token"),
                               json_node["lane_token"]))

    
    return mismatches



