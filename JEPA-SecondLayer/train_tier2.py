
# -------------------------------------------------------------
# Tier-2 Module: End-to-End Testing Script (Corrected)
# -------------------------------------------------------------
import torch
from pathlib import Path
from jepaTier2 import Tier2Module
from utils import count_params, build_graph_batch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Tier2Dataset, tier2_collate_fn
from kinematic import build_scene_mapping_parallel
import numpy as np
import networkx as nx

dataroot= "/kaggle/input/a-crude-data-set-converted-from-nuscene/metadata"
dataset_path = "/kaggle/input/a-crude-data-set-converted-from-nuscene"

scene_map = build_scene_mapping_parallel(
    "/kaggle/input/a-crude-data-set-converted-from-nuscene/metadata"
)

test_dataset = Tier2Dataset(scene_map=scene_map, dataset_path=dataset_path, augment=True)

# Model + utils
test_tier_2 = Tier2Module()
count_params(test_tier_2)


loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=tier2_collate_fn,
    num_workers=0,
    pin_memory=True
)

# -------------------------------------------------------------
# 1) Fetch a batch
# -------------------------------------------------------------
batch = next(iter(loader))

graphs      = batch["graphs"]        # list of NetworkX graphs
graph_mask  = batch["graph_mask"]    # [B, maxN]
traj        = batch["clean_deltas"]  # [B, T, 6]

# -------------------------------------------------------------
# 2) Build graph batch using your function (takes the category id & the graphs)
# -------------------------------------------------------------


# === UNIQUE VALUES DISCOVERED ===
# type: ['agent', 'road']
# category: ['animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller', 'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris', 'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck']
# layer: ['lane']

type_values = ['agent', 'road']
category_values = ['animal',
 'human.pedestrian.adult',
 'human.pedestrian.child',
 'human.pedestrian.construction_worker',
 'human.pedestrian.personal_mobility',
 'human.pedestrian.police_officer',
 'human.pedestrian.stroller',
 'human.pedestrian.wheelchair',
 'movable_object.barrier',
 'movable_object.debris',
 'movable_object.pushable_pullable',
 'movable_object.trafficcone',
 'static_object.bicycle_rack',
 'vehicle.bicycle',
 'vehicle.bus.bendy',
 'vehicle.bus.rigid',
 'vehicle.car',
 'vehicle.construction',
 'vehicle.emergency.ambulance',
 'vehicle.emergency.police',
 'vehicle.motorcycle',
 'vehicle.trailer',
 'vehicle.truck']
layer_values = ['lane']  # trivial

type2id = {v:i for i,v in enumerate(type_values)}
category2id = {v:i for i,v in enumerate(category_values)}
layer2id = {v:i for i,v in enumerate(layer_values)}

x_batch, adj_batch = build_graph_batch(graphs, type2id, category2id, layer2id)
# x_batch:  [B, maxN, 13]
# adj_batch: [B, maxN, maxN]

# -------------------------------------------------------------
# 3) Forward pass through Tier-2
# -------------------------------------------------------------
result = test_tier_2(
    traj=traj,
    adj=adj_batch,
    x_graph=x_batch,
    traj_mask=batch["traj_mask"],
    graph_mask=graph_mask,
    tau=0.5,
)

print("Keys:", result.keys())
print("traj_emb:   ", result["traj_emb"].shape)
print("graph_emb:  ", result["graph_emb"].shape)
print("fusion:     ", result["fusion"].shape)
print("node_level: ", result["node_level"].shape)
print("attn:       ", result["attn"].shape)