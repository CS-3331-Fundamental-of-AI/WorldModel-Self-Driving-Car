import os, json, math, random, time
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# ENV
# ==========================================
load_dotenv(".env")
os.environ["COMET_LOG_PACKAGES"] = "0"

from comet_ml import Experiment
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from Utils.dataset import MapDataset
from JEPA.JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update
from ultralytics import YOLO
from config import (
    LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1,
    BETA_1, BETA_2, GAMMA, EMA_DECAY,
    ACCUM_STEPS, USE_BF16
)

# ==========================================
# CONFIG — FAST + SMALL
# ==========================================
EPOCH       = 3
K_MEMBERS   = 5
SHARD_SIZE  = 300
VAL_FRAC    = 0.10
BASE_LR     = 3e-4
WEIGHT_DECAY= 0.01
ETA_BOOST   = 0.7
SEED_BASE   = 1337

# Gradient checkpointing toggle
USE_GRAD_CKPT = True

SAVE_DIR = Path("./soup_members")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# ==========================================
# FAST MASK UPSAMPLE
# ==========================================
def up2_fast(mask32):
    return F.interpolate(mask32.float(), scale_factor=2, mode="nearest").bool()

# ==========================================
# AMP POLICY
# ==========================================
def get_amp_policy():
    use_amp = True
    amp_dtype = (
        torch.bfloat16
        if (USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    return use_amp, amp_dtype, scaler

# ==========================================
# YOLO WEIGHT TRANSFER
# ==========================================
def transfer_yolo_weights(primitive_layer):
    print("🔄 Loading YOLOv8 backbone for weight transfer...")
    yolo = YOLO(os.getenv("YOLO_PATH"))
    full = yolo.model.model
    backbone = full[:10]

    yolo_s1 = backbone[0].conv
    yolo_s2 = backbone[1].conv
    yolo_s3 = backbone[3].conv

    def slice_and_load(my_conv, yo_conv, in_ch, out_ch):
        with torch.no_grad():
            my_conv.weight[:] = yo_conv.weight[:out_ch, :in_ch]
            if my_conv.bias is not None:
                my_conv.bias.zero_()

    ctx = primitive_layer.context_encoder
    slice_and_load(ctx.s1[0].block[0], yolo_s1, 3, 16)
    slice_and_load(ctx.s2[0],          yolo_s2, 16, 32)
    slice_and_load(ctx.s3[0],          yolo_s3, 32, 64)
    ctx.s4[0].weight.data[:] = yolo_s3.weight.data

    tgt = primitive_layer.target_encoder
    slice_and_load(tgt.s1[0].block[0], yolo_s1, 3, 16)
    slice_and_load(tgt.s2[0],          yolo_s2, 16, 32)
    slice_and_load(tgt.s3[0],          yolo_s3, 32, 64)
    tgt.s4[0].weight.data[:] = yolo_s3.weight.data

    print("✅ YOLO → JEPA weight transfer complete.")
    return primitive_layer

# ==========================================
# Freeze policy:
#   - context encoder trainable
#   - target encoder frozen (EMA teacher)
# ==========================================
def set_train_policy(model: PrimitiveLayer):
    for p in model.context_encoder.parameters():
        p.requires_grad = True
    for p in model.target_encoder.parameters():
        p.requires_grad = False
    for p in model.predictor.parameters():
        p.requires_grad = True

# ==========================================
# Forward with optional gradient checkpointing
# ==========================================
def forward_ckpt(model, masked, unmasked, m_e, m_n, m_a):
    if not USE_GRAD_CKPT or not torch.is_grad_enabled():
        return model(masked, unmasked, m_e, m_n, m_a)

    # checkpoint needs at least one input requiring grad
    masked_ckpt = masked.detach().requires_grad_(True)
    unmasked_ckpt = unmasked.detach().requires_grad_(True)

    def fn(masked_in, unmasked_in, me_in, mn_in, ma_in):
        return model(masked_in, unmasked_in, me_in, mn_in, ma_in)

    return checkpoint(fn, masked_ckpt, unmasked_ckpt, m_e, m_n, m_a, use_reentrant=False)

# ==========================================
# FAST VALIDATION
# ==========================================
@torch.no_grad()
def evaluate_val(model, val_loader, device, use_amp, amp_dtype):
    model.eval()
    acc = []
    for bev, mask_emp, mask_non_emp, mask_union, \
        mask_emp_np, mask_non_emp_np, mask_union_np, \
        ph, pw, img in val_loader:

        B = bev.size(0)

        masked = mask_emp.squeeze(1).to(device)
        unmasked = mask_non_emp.squeeze(1).to(device)

        m_e = up2_fast(mask_emp_np.to(device).view(B,1,32,32))
        m_n = up2_fast(mask_non_emp_np.to(device).view(B,1,32,32))
        m_a = up2_fast(mask_union_np.to(device).view(B,1,32,32))

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            z_c, s_c, z_t = model(masked, unmasked, m_e, m_n, m_a)
            z_c = F.normalize(z_c, dim=-1)
            s_c = F.normalize(s_c, dim=-1)
            z_t = F.normalize(z_t, dim=-1)

            loss = compute_jepa_loss(
                s_c=s_c, s_t=z_t, z_c=z_c,
                mask_empty=m_e.view(B,-1),
                mask_nonempty=m_n.view(B,-1),
                alpha0=ALPHA_0, alpha1=ALPHA_1,
                beta1=BETA_1, beta2=BETA_2,
                lambda_jepa=LAMBDA_JEPA,
                lambda_reg=LAMBDA_REG,
                gamma=GAMMA
            )["loss_total"]

        acc.append(float(loss))
    model.train()
    return sum(acc)/len(acc)

# ==========================================
# SHARD MAKER
# ==========================================
def make_shards(indices, shard_size):
    return [indices[i:i+shard_size] for i in range(0, len(indices), shard_size)]

# ==========================================
# WORKER: trains subset of soup members on one GPU
# ==========================================
def train_worker(rank, world_size, val_idx, train_idx, train_pos, map_csv):

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[GPU{rank}] running worker.")

    use_amp, amp_dtype, scaler = get_amp_policy()

    full_ds = MapDataset(map_csv_file=map_csv)

    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=64, shuffle=False,
        num_workers=2, pin_memory=True,
        persistent_workers=True
    )

    exp = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE")
    )
    exp.set_name(f"JEPA-Soup-FAST-GPU{rank}")
    exp.add_tags(["fast", "soup", "boost", f"gpu{rank}", "ctx-train", "tgt-frozen", "grad-ckpt"])

    meta_part = {"rank": rank, "members": []}

    member_ids = [k for k in range(K_MEMBERS) if k % world_size == rank]

    for k in member_ids:
        seed_k = SEED_BASE + k*100
        torch.manual_seed(seed_k)
        random.seed(seed_k)

        print(f"\n[GPU{rank}] ==============================")
        print(f"[GPU{rank}] 🥣 Training soup member {k+1}/{K_MEMBERS}")
        print(f"[GPU{rank}] ==============================")

        model = PrimitiveLayer(embed_dim=128).to(device)
        model = transfer_yolo_weights(model)
        set_train_policy(model)

        # optional compile predictor only
        try:
            model.predictor = torch.compile(model.predictor)
        except Exception as e:
            print(f"[GPU{rank}] torch.compile skipped: {e}")

        lr_k = BASE_LR * (1 + random.uniform(-0.2, 0.2))

        optim = torch.optim.Adam(
            list(model.context_encoder.parameters()) +
            list(model.predictor.parameters()),
            lr=lr_k,
            weight_decay=WEIGHT_DECAY
        )

        weights = torch.ones(len(train_idx)) / len(train_idx)

        shards = make_shards(train_idx, SHARD_SIZE)
        random.Random(seed_k).shuffle(shards)

        model.train()
        step = 0

        for ep in range(EPOCH):
            t = ep / max(1, EPOCH-1)
            model.ema_decay = float(EMA_DECAY + (1-EMA_DECAY)*t)

            for sh, shard in enumerate(shards):
                pos = [train_pos[i] for i in shard]
                shard_w = weights[pos].float().clamp(1e-8, 1)
                sampler = WeightedRandomSampler(shard_w, len(shard), replacement=True)

                loader = DataLoader(
                    Subset(full_ds, shard),
                    batch_size=32,
                    sampler=sampler,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=True,
                    drop_last=True
                )

                losses_list = []
                pbar = tqdm(
                    loader,
                    desc=f"[GPU{rank}] M{k+1} EP{ep+1}/{EPOCH} SH{sh+1}/{len(shards)}",
                    leave=False
                )

                for batch in pbar:
                    bev, mask_emp, mask_non_emp, mask_union, \
                    m_e_np, m_n_np, m_a_np, ph, pw, img = batch

                    B = bev.size(0)
                    masked = mask_emp.squeeze(1).to(device, non_blocking=True)
                    unmasked = mask_non_emp.squeeze(1).to(device, non_blocking=True)

                    m_e = up2_fast(m_e_np.to(device).view(B,1,32,32))
                    m_n = up2_fast(m_n_np.to(device).view(B,1,32,32))
                    m_a = up2_fast(m_a_np.to(device).view(B,1,32,32))

                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        z_c, s_c, z_t = forward_ckpt(model, masked, unmasked, m_e, m_n, m_a)
                        z_c = F.normalize(z_c, dim=-1)
                        s_c = F.normalize(s_c, dim=-1)
                        z_t = F.normalize(z_t, dim=-1)

                        losses = compute_jepa_loss(
                            s_c=s_c, s_t=z_t, z_c=z_c,
                            mask_empty=m_e.view(B,-1),
                            mask_nonempty=m_n.view(B,-1),
                            alpha0=ALPHA_0, alpha1=ALPHA_1,
                            beta1=BETA_1, beta2=BETA_2,
                            lambda_jepa=LAMBDA_JEPA,
                            lambda_reg=LAMBDA_REG,
                            gamma=GAMMA
                        )

                    loss = losses["loss_total"] / ACCUM_STEPS
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (step+1) % ACCUM_STEPS == 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optim)

                        torch.nn.utils.clip_grad_norm_(
                            list(model.context_encoder.parameters()) +
                            list(model.predictor.parameters()),
                            1.0
                        )

                        if scaler.is_enabled():
                            scaler.step(optim); scaler.update()
                        else:
                            optim.step()
                        optim.zero_grad(set_to_none=True)

                        # EMA ctx -> tgt (tgt frozen to grads)
                        ema_update(
                            model.context_encoder,
                            model.target_encoder,
                            model.ema_decay
                        )

                    v = float(losses["loss_total"].detach())
                    losses_list.append(v)
                    pbar.set_postfix(loss=v)
                    step += 1

                shard_mean = sum(losses_list)/len(losses_list)
                bump = math.exp(ETA_BOOST * shard_mean)
                weights[pos] *= bump
                weights /= weights.sum()

        val_loss = evaluate_val(model, val_loader, device, use_amp, amp_dtype)
        print(f"[GPU{rank}] 📌 Member {k+1} val loss = {val_loss:.6f}")

        p = SAVE_DIR / f"predictor_member_{k+1}.pt"
        torch.save(model.predictor.state_dict(), p)

        meta_part["members"].append({
            "k": k+1,
            "seed": seed_k,
            "lr": lr_k,
            "val_loss": val_loss,
            "path": str(p)
        })

        exp.log_metric(f"member{k+1}_val_loss", val_loss)

    part_path = SAVE_DIR / f"soup_meta_part_gpu{rank}.json"
    with open(part_path, "w") as f:
        json.dump(meta_part, f, indent=2)
    print(f"[GPU{rank}] ✅ Saved meta part:", part_path)

    exp.log_asset(str(part_path))
    exp.end()


# ==========================================
# MAIN CONTROLLER
# ==========================================
def main():
    assert torch.cuda.device_count() >= 2, "Need 2 GPUs for parallel soup."
    world_size = 2

    map_csv = './map_files_15k.csv' # os.getenv("MAP_CSV")
    full_ds = MapDataset(map_csv_file=map_csv)

    N = len(full_ds)
    all_idx = list(range(N))
    random.Random(SEED_BASE).shuffle(all_idx)

    val_n = int(VAL_FRAC * N)
    val_idx = all_idx[:val_n]
    train_idx = all_idx[val_n:]

    train_pos = {idx: i for i, idx in enumerate(train_idx)}

    torch.multiprocessing.spawn(
        train_worker,
        args=(world_size, val_idx, train_idx, train_pos, map_csv),
        nprocs=world_size,
        join=True
    )

    members = []
    for r in range(world_size):
        part_path = SAVE_DIR / f"soup_meta_part_gpu{r}.json"
        with open(part_path, "r") as f:
            part = json.load(f)
        members.extend(part["members"])

    final_meta = {
        "val_indices": val_idx,
        "train_indices": train_idx,
        "members": sorted(members, key=lambda m: m["k"])
    }

    final_path = SAVE_DIR / "soup_meta.json"
    with open(final_path, "w") as f:
        json.dump(final_meta, f, indent=2)

    print("✅ Final merged soup_meta.json saved:", final_path)


if __name__ == "__main__":
    main()