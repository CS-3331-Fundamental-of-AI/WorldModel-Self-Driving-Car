#!/usr/bin/env python3
"""
Stage-0: Global GCN self-supervised pretraining
(using Tier3Dataset global graphs)

- Mask node positions
- Reconstruct masked nodes
- Save checkpoint in JEPA-compatible format
"""

import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --------------------------------------------------
# Env
# --------------------------------------------------
load_dotenv(".env")

# --------------------------------------------------
# Device (same logic as JEPA-1)
# --------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"👉 Using device: {device}")

# --------------------------------------------------
# Paths / Checkpoints
# --------------------------------------------------
CKPT_DIR = Path("/kaggle/working/checkpoints/gcn_global")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Dataset (IDENTICAL source as JEPA-3)
# --------------------------------------------------
from Utils.jepa2data import get_scene_map
from Utils.jepa3data import Tier3Dataset, tier3_collate_fn

scene_map = get_scene_map()
dataset = Tier3Dataset(scene_map)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    collate_fn=lambda b: tier3_collate_fn(b, device=device),
    pin_memory=(device.type == "cuda"),
)

# --------------------------------------------------
# Model
# --------------------------------------------------
from JEPA_ThirdLayer.global_encoding.gcn_pretrain import GCNPretrainModel
from Utils.gcn_utils import edges_to_adj, mask_nodes

model = GCNPretrainModel(
    node_dim=3,
    hidden=128,
    out_dim=3,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

# --------------------------------------------------
# Checkpoint helpers (MATCH JEPA-1)
# --------------------------------------------------
def atomic_torch_save(obj, final_path: Path):
    tmp = final_path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(final_path)

def safe_save(model, step, tag="auto"):
    path = CKPT_DIR / f"{tag}_step{step}.pt"
    print(f"\n💾 Saving checkpoint: {path.name}")

    atomic_torch_save(
        {
            "version": 1,
            "step": step,
            "state": model.state_dict(),  # MUST be "state"
        },
        path,
    )

# --------------------------------------------------
# Training
# --------------------------------------------------
EPOCHS = 1
MASK_RATIO = 0.3
global_step = 0

try:
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            global_step += 1

            global_nodes = batch["global_nodes"]   # list[T_i, 3]
            global_edges = batch["global_edges"]   # list[E_i, 2]

            losses = []

            for nodes, edges in zip(global_nodes, global_edges):
                N = nodes.shape[0]

                adj = edges_to_adj(edges, N).to(device)

                nodes = nodes.unsqueeze(0)  # [1, N, 3]
                adj   = adj.unsqueeze(0)    # [1, N, N]

                masked_nodes, mask = mask_nodes(nodes, MASK_RATIO)

                recon = model(masked_nodes, adj)

                loss = F.mse_loss(recon[mask], nodes[mask])
                losses.append(loss)

            loss = torch.stack(losses).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        print(f"Epoch {epoch+1} finished")

except KeyboardInterrupt:
    print("\n⛔ Interrupted")
    safe_save(model, global_step, tag="interrupt")
    raise

except Exception as e:
    print("\n❌ Training crashed:", e)
    safe_save(model, global_step, tag="crash")
    raise

else:
    print("\n🎉 Pretraining completed")
    safe_save(model, global_step, tag="gcn_global_final")
