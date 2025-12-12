import os, json, copy
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/content/WorldModel-Self-Driving-Car/JEPA-PrimitiveLayer/.env")
os.environ["COMET_LOG_PACKAGES"] = "0"

from comet_ml import Experiment
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Utils.dataset import MapDataset
from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from Utils.losses import compute_jepa_loss
from config import (
    LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1,
    BETA_1, BETA_2, GAMMA, USE_BF16
)

SAVE_DIR = Path("./soup_members")
META_PATH = SAVE_DIR / "soup_meta.json"

def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

def get_amp_policy():
    use_amp = True
    amp_dtype = (
        torch.bfloat16
        if (USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    return use_amp, amp_dtype

@torch.no_grad()
def evaluate_val(model, val_loader, device, use_amp, amp_dtype):
    model.eval()
    acc = []
    for batch in val_loader:
        bev, mask_emp, mask_non_emp, mask_union, \
        mask_emp_np, mask_non_emp_np, mask_union_np, \
        ph, pw, img = batch

        B = bev.shape[0]
        masked_img   = mask_emp.squeeze(1).to(device, non_blocking=True)
        unmasked_img = mask_non_emp.squeeze(1).to(device, non_blocking=True)

        mask_emp_grid = mask_emp_np.to(device, non_blocking=True).view(B,1,32,32).bool()
        mask_non_grid = mask_non_emp_np.to(device, non_blocking=True).view(B,1,32,32).bool()
        mask_any_grid = mask_union_np.to(device, non_blocking=True).view(B,1,32,32).bool()

        mask_emp_up = up2(mask_emp_grid)
        mask_non_up = up2(mask_non_grid)
        mask_any_up = up2(mask_any_grid)

        mask_emp_flat = mask_emp_up.view(B, -1)
        mask_non_flat = mask_non_up.view(B, -1)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            z_c, s_c, z_t = model(masked_img, unmasked_img, mask_emp_up, mask_non_up, mask_any_up)

            z_c_norm = F.normalize(z_c, dim=-1)
            s_c_norm = F.normalize(s_c, dim=-1)
            z_t_norm = F.normalize(z_t, dim=-1)

            losses = compute_jepa_loss(
                s_c=s_c_norm, s_t=z_t_norm, z_c=z_c_norm,
                mask_empty=mask_emp_flat, mask_nonempty=mask_non_flat,
                alpha0=ALPHA_0, alpha1=ALPHA_1,
                beta1=BETA_1, beta2=BETA_2,
                lambda_jepa=LAMBDA_JEPA, lambda_reg=LAMBDA_REG, gamma=GAMMA
            )

        acc.append(float(losses["loss_total"]))

    model.train()
    return sum(acc) / max(1, len(acc))

def average_states(states):
    avg = {}
    keys = states[0].keys()
    for k in keys:
        avg[k] = torch.stack([s[k].float() for s in states], 0).mean(0)
    return avg

def main():
    assert META_PATH.exists(), f"Missing {META_PATH}, run train_soup.py first."

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    use_amp, amp_dtype = get_amp_policy()

    # load dataset + fixed val split from meta
    map_csv = os.getenv("MAP_CSV", "./map_files.csv")
    full_ds = MapDataset(map_csv_file=map_csv)

    val_indices = meta["val_indices"]
    val_loader = DataLoader(
        Subset(full_ds, val_indices),
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

    # comet experiment
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
    )
    experiment.set_name("JEPA-PrimitiveLayer-SoupBuild")
    experiment.add_tags(["model-soup", "greedy-soup"])

    # sort members by best val_loss
    members = sorted(meta["members"], key=lambda m: m["val_loss"])
    print("Member ranking by val loss:")
    for m in members:
        print(f"  M{m['k']}: val={m['val_loss']:.6f}")

    # base model shell (encoders unused here except to be consistent)
    base_model = PrimitiveLayer(embed_dim=128).to(device)
    base_model.train()

    # start soup with best member
    soup_states = [torch.load(members[0]["path"], map_location="cpu")]
    best_state = average_states(soup_states)
    base_model.predictor.load_state_dict(best_state)
    best_val = evaluate_val(base_model, val_loader, device, use_amp, amp_dtype)

    print(f"\n🥣 Start soup with M{members[0]['k']} -> val={best_val:.6f}")
    experiment.log_metric("soup_val", best_val, step=0)

    # greedy add
    step = 1
    for m in members[1:]:
        cand_states = soup_states + [torch.load(m["path"], map_location="cpu")]
        cand_avg = average_states(cand_states)

        base_model.predictor.load_state_dict(cand_avg)
        cand_val = evaluate_val(base_model, val_loader, device, use_amp, amp_dtype)

        print(f"Try add M{m['k']}: cand_val={cand_val:.6f} vs best={best_val:.6f}")

        if cand_val <= best_val:
            print("  ❌ Reject")
        else:
            print("  ✅ Accept")
            soup_states = cand_states
            best_state = cand_avg
            best_val = cand_val

        experiment.log_metric("soup_val", best_val, step=step)
        step += 1

    # save final soup predictor
    soup_path = SAVE_DIR / "predictor_soup.pt"
    torch.save(best_state, soup_path)
    print("\n✅ Final greedy soup saved to:", soup_path)
    print("Final soup val loss:", best_val)

    experiment.log_asset(str(soup_path))
    experiment.log_metric("final_soup_val_loss", best_val)
    experiment.end()

if __name__ == "__main__":
    main()
