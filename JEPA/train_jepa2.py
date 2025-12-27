#!/usr/bin/env python3
"""
Stage-2a: Train JEPA-2 (Inverse + Physical Affordance) separately

- JEPA-1 provides s_c (frozen)
- JEPA-2 trains with EMA target
- Checkpoints saved for JEPA-3 to load later
"""

import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

# --------------------------------------------------
# Config
# --------------------------------------------------
from config.config import DEVICE, BATCH_SIZE, NUM_WORKERS, EPOCHS, CKPT_DIR
from JEPA_SecondLayer import JEPA_Tier2_InverseAffordance, JEPA_Tier2_PhysicalAffordance
from trainers.trainer_jepa2 import JEPA2Trainer
from Utils.jepa2data import Tier2Dataset, DATASET_PATH, get_scene_map
from pipeline.jepa_adapter import JEPAInputAdapter
from pipeline.jepa_pipeline import JEPAPipeline

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()
CKPT_DIR = Path(CKPT_DIR) / "jepa2"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Device selection
# --------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# Dataset
# --------------------------------------------------
scene_map = get_scene_map()
dataset = Tier2Dataset(scene_map=scene_map, dataset_path=DATASET_PATH, augment=True)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

# --------------------------------------------------
# Models
# --------------------------------------------------
jepa2_pa = JEPA_Tier2_PhysicalAffordance().to(device)
jepa2_ia = JEPA_Tier2_InverseAffordance().to(device)
jepa2_ia_ema = JEPA_Tier2_InverseAffordance().to(device)

# Initialize EMA weights
jepa2_ia_ema.load_state_dict(jepa2_ia.state_dict())
for p in jepa2_ia_ema.parameters():
    p.requires_grad = False
jepa2_ia_ema.eval()

# Optimizer
optimizer = torch.optim.AdamW(
    list(jepa2_pa.parameters()) + list(jepa2_ia.parameters()),
    lr=3e-4,
    weight_decay=1e-2,
)

# --------------------------------------------------
# Graph vocab for JEPA-2
# --------------------------------------------------
TYPE_VALUES = ['agent', 'road']
CATEGORY_VALUES = [
    'animal','human.pedestrian.adult','human.pedestrian.child',
    'human.pedestrian.construction_worker','human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer','human.pedestrian.stroller',
    'human.pedestrian.wheelchair','movable_object.barrier','movable_object.debris',
    'movable_object.pushable_pullable','movable_object.trafficcone',
    'static_object.bicycle_rack','vehicle.bicycle','vehicle.bus.bendy',
    'vehicle.bus.rigid','vehicle.car','vehicle.construction',
    'vehicle.emergency.ambulance','vehicle.emergency.police','vehicle.motorcycle',
    'vehicle.trailer','vehicle.truck'
]
LAYER_VALUES = ['lane']

type2id = {v: i for i, v in enumerate(TYPE_VALUES)}
category2id = {v: i for i, v in enumerate(CATEGORY_VALUES)}
layer2id = {v: i for i, v in enumerate(LAYER_VALUES)}

# --------------------------------------------------
# Adapter + Pipeline
# --------------------------------------------------
adapter = JEPAInputAdapter(
    device=device,
    type2id=type2id,
    category2id=category2id,
    layer2id=layer2id,
)

# Dummy JEPA-1 (frozen) placeholder for pipeline
t1 = None  # This can be a minimal stub if JEPA-1 is not needed here
pipeline = JEPAPipeline(t1=t1, t2=JEPA2Trainer(jepa2_pa, jepa2_ia, jepa2_ia_ema, optimizer), t3=None, adapter=adapter)

# --------------------------------------------------
# Checkpoint helper
# --------------------------------------------------
def save_checkpoint(step):
    path = CKPT_DIR / f"jepa2_step{step}.pt"
    print(f"💾 Saving checkpoint: {path}")
    torch.save(
        {
            "version": 1,
            "step": step,
            "state": {
                "pa_model": jepa2_pa.state_dict(),
                "ia_model": jepa2_ia.state_dict(),
                "ia_ema_model": jepa2_ia_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
        },
        path,
    )

# --------------------------------------------------
# Training loop
# --------------------------------------------------
global_step = 0
try:
    for epoch in range(EPOCHS):
        jepa2_pa.train()
        jepa2_ia.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            global_step += 1

            out = pipeline.t2.step(batch)  # JEPA-2 trainer step
            loss = float(out.get("loss", 0.0))

            pbar.set_postfix({"loss": f"{loss:.5f}"})

            if global_step % 500 == 0:
                save_checkpoint(global_step)

except KeyboardInterrupt:
    print("\n⛔ Interrupted. Saving checkpoint...")
    save_checkpoint(global_step)
    raise

except Exception as e:
    print(f"\n❌ Training crashed: {e}")
    save_checkpoint(global_step)
    raise

else:
    print("\n🎉 Training finished. Saving final checkpoint...")
    save_checkpoint(global_step)
