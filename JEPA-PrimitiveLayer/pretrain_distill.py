# pretrain_distill.py
import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50

from comet_ml import Experiment

from JEPA_PrimitiveLayer.bev_jepa import BEVJEPAEncoder2D
from Utils.dataset import MapDataset

load_dotenv()

# --------------------------------------------------------
# 1. Load DINO ResNet-50 (from local / Kaggle / download)
# --------------------------------------------------------
def load_dino_resnet50(device):
    """
    Load a DINO-pretrained ResNet-50 backbone.

    Priority:
      1) If DINO_CKPT env var is set and exists -> use that path.
      2) Try some common local / Kaggle paths.
      3) Otherwise, download official DINO ckpt into CWD.
    """
    candidate_paths = []

    # 1) Explicit env var (recommended, esp. on Kaggle)
    env_path = os.getenv("DINO_CKPT")
    if env_path:
        candidate_paths.append(env_path)

    # 2) Common local paths (Mac / local dev)
    cwd = os.getcwd()
    candidate_paths.append(os.path.join(cwd, "dino_resnet50_pretrain.pth"))
    candidate_paths.append(os.path.join(cwd, "dino_resnet50_pretrain", "dino_resnet50_pretrain.pth"))

    # 3) Common Kaggle path (attach your Kaggle Model or Dataset here)
    #    You can adjust this to the exact mounted path you see in Kaggle UI.
    candidate_paths.append("/kaggle/input/dino-resnet50-pretrain/dino_resnet50_pretrain.pth")

    ckpt_path = None
    for p in candidate_paths:
        if p and os.path.isfile(p):
            ckpt_path = p
            break

    if ckpt_path is None:
        # Download into current working directory (one-time)
        url = "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        ckpt_path = os.path.join(cwd, "dino_resnet50_pretrain.pth")
        print(f"📥 DINO checkpoint not found locally. Downloading to {ckpt_path} ...")
        from torch.hub import download_url_to_file
        download_url_to_file(url, ckpt_path)
    else:
        print(f"✅ Using local DINO checkpoint at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # checkpoint may be:
    #  - an nn.Module (if you saved the whole model), or
    #  - a plain state_dict, or
    #  - a dict with key "state_dict".
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip potential "module." prefix
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state[k] = v

        # Standard ResNet-50 backbone with no pretrained weights
        model = resnet50(weights=None)

        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"ℹ️ Loaded DINO ResNet-50 state_dict.")
        print(f"   Missing keys:   {len(missing)}")
        print(f"   Unexpected keys:{len(unexpected)}")
    
    # 🔥 CRITICAL FIX: remove the 1000-class head
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


# --------------------------
# 2. Student + Projection Head
# --------------------------
class StudentWithHead(nn.Module):
    def __init__(self, width_mult=0.5, teacher_dim=2048):
        super().__init__()
        self.encoder = BEVJEPAEncoder2D(width_mult=width_mult)
        C = self.encoder.out_dim  # e.g. 64

        # simple projection: C -> 256 -> teacher_dim
        self.proj = nn.Sequential(
            nn.Linear(C, 256),
            nn.GELU(),
            nn.Linear(256, teacher_dim),
        )

    def forward(self, x):
        """
        x: (B,3,H,W)
        returns:
            s_proj: (B, teacher_dim)  - projected to teacher space
            s_vec:  (B, C)            - pooled encoder feature
        """
        feat = self.encoder(x)  # (B, C, Hc, Wc) or (feat, (Hc, Wc))
        if isinstance(feat, tuple):
            feat, _ = feat
        B, C, Hc, Wc = feat.shape

        # Global average pooling over spatial dims
        s_vec = feat.mean(dim=[2, 3])  # (B, C)
        s_proj = self.proj(s_vec)      # (B, teacher_dim)
        return s_proj, s_vec


# --------------------------
# 3. Training Loop + Fail-safe
# --------------------------
def main():
    # --------------------------
    # Comet setup
    # --------------------------
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
    )
    experiment.set_name("Distill-MobileNet-DINO")
    experiment.log_parameters({
        "alpha_cos": 1.0,
        "alpha_l2":  1.0,
        "lr":        3e-4,
        "weight_decay": 0.05,
        "width_mult": 0.5,
        "teacher_dim": 2048,
    })

    # So we can decide in except whether to save a fail-safe checkpoint
    student = None

    try:
        # --------------------------
        # Device selection
        # --------------------------
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔥 Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🍎 Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("⚠️ Using CPU only (slow)")
        experiment.log_parameter("device", str(device))

        # --------------------------
        # Dataset
        # --------------------------
        map_csv = os.getenv("MAP_CSV")
        ds = MapDataset(map_csv_file=map_csv)

        dataloader = DataLoader(
            ds,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True if device.type == "cuda" else False,
        )

        # --------------------------
        # Teacher & Student
        # --------------------------
        teacher = load_dino_resnet50(device)
        student = StudentWithHead(width_mult=0.5, teacher_dim=2048).to(device)

        # Multi-GPU (if available)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"⚡ Using {torch.cuda.device_count()} GPUs with DataParallel")
            student = nn.DataParallel(student)
            teacher = nn.DataParallel(teacher)

        # --------------------------
        # Optimizer & config
        # --------------------------
        optim = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.05)

        epochs = 5
        alpha_cos = 1.0
        alpha_l2 = 1.0

        # --------------------------
        # Training
        # --------------------------
        global_step = 0
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"[Distill] Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                (
                    bev, mask_emp, mask_non_emp, mask_union,
                    mask_emp_np, mask_non_emp_np, mask_union_np,
                    ph, pw, img
                ) = batch

                img = img.to(device)  # (B,3,H,W)

                # Teacher forward
                with torch.no_grad():
                    t_feat = teacher(img)           # (B, 2048)
                    t_feat = F.normalize(t_feat, dim=-1)

                # Student forward
                s_proj, s_vec = student(img)       # (B, 2048), (B, C)
                s_proj = F.normalize(s_proj, dim=-1)

                # Distill loss
                cos_loss = 1.0 - (s_proj * t_feat).sum(dim=-1).mean()
                l2_loss  = F.mse_loss(s_proj, t_feat)
                loss = alpha_cos * cos_loss + alpha_l2 * l2_loss

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optim.step()

                pbar.set_postfix({
                    "cos": f"{cos_loss.item():.4f}",
                    "l2": f"{l2_loss.item():.4f}",
                    "loss": f"{loss.item():.4f}",
                })

                # Comet logging
                experiment.log_metric("loss", loss.item(), step=global_step)
                experiment.log_metric("cos_loss", cos_loss.item(), step=global_step)
                experiment.log_metric("l2_loss", l2_loss.item(), step=global_step)
                global_step += 1

        # --------------------------
        # Save distilled encoder
        # --------------------------
        if isinstance(student, nn.DataParallel):
            enc_state = student.module.encoder.state_dict()
        else:
            enc_state = student.encoder.state_dict()

        out_path = "bev_mobilenet_dino_init.pt"
        torch.save(enc_state, out_path)
        print(f"✅ Saved distilled MobileNet encoder → {out_path}")

        experiment.log_asset(out_path)

    except Exception as e:
        print("\n❌ DISTILLATION FAILED — entering fail-safe mode...\n")

        # Save whatever we currently have from the student encoder
        if student is not None:
            if isinstance(student, nn.DataParallel):
                enc_state = student.module.encoder.state_dict()
            else:
                enc_state = student.encoder.state_dict()
            fail_path = "distill_fail.pt"
            torch.save(enc_state, fail_path)
            print(f"💾 Saved fail-safe checkpoint → {fail_path}")
            experiment.log_asset(fail_path)

        # Write error log
        err_log = "distill_error_log.txt"
        with open(err_log, "w") as f:
            f.write(str(e))
        experiment.log_asset(err_log)
        experiment.log_text(str(e), metadata={"phase": "distill_fail"})

        # Re-raise so you still see the traceback in the notebook
        raise

    finally:
        experiment.end()


if __name__ == "__main__":
    main()