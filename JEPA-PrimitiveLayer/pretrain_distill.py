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

# ------------------------------------------------------------------
# Global flags / Kaggle detection
# ------------------------------------------------------------------
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# Fewer workers on Kaggle to avoid weird hangs with multiprocessing
DEFAULT_NUM_WORKERS = 2 # 0 if IS_KAGGLE else 
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# On Kaggle, you can optionally cap epochs / steps per epoch
KAGGLE_EPOCHS = int(os.getenv("KAGGLE_EPOCHS", "5"))
KAGGLE_MAX_STEPS = int(os.getenv("KAGGLE_MAX_STEPS", "300"))  # per epoch


# --------------------------------------------------------
# 1. Load DINO ResNet-50 (Kaggle checkpoint)
# --------------------------------------------------------
def load_dino_resnet50(device):
    """
    Load a DINO-pretrained ResNet-50 backbone from Kaggle model path.
    """
    kaggle_ckpt = "/kaggle/input/dino-resnet50-pretrain/pytorch/dino_resnet50_pretrain/1/dino_resnet50_pretrain.pth"

    if os.path.exists(kaggle_ckpt):
        ckpt_path = kaggle_ckpt
        print(f"✅ Using Kaggle DINO checkpoint at: {ckpt_path}")
    else:
        raise FileNotFoundError(
            f"DINO checkpoint not found at expected Kaggle path:\n{kaggle_ckpt}"
        )

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract state_dict (works for both raw and wrapped checkpoints)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip "module." prefix from multi-GPU training checkpoints
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    # Load into torchvision ResNet-50 backbone (no ImageNet weights)
    model = resnet50(weights=None)

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"ℹ️ Loaded DINO ResNet-50 state_dict.")
    print(f"   Missing keys:   {len(missing)}")
    print(f"   Unexpected keys:{len(unexpected)}")

    # 🔥 Remove the 1000-class head → output 2048-dim features
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
        "batch_size": BATCH_SIZE,
        "num_workers": DEFAULT_NUM_WORKERS,
        "is_kaggle": IS_KAGGLE,
    })

    student = None

    try:
        # --------------------------
        # Device selection
        # --------------------------
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔥 Using CUDA: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🍎 Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("⚠️ Using CPU only (slow)")
        experiment.log_parameter("device", str(device))

        # --------------------------
        # Dataset / DataLoader
        # --------------------------
        map_csv = os.getenv("MAP_CSV")
        ds = MapDataset(map_csv_file=map_csv)

        dataloader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=True if device.type == "cuda" else False,
            persistent_workers=False,  # safer on Kaggle
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

        if IS_KAGGLE:
            epochs = KAGGLE_EPOCHS
        else:
            epochs = 5

        alpha_cos = 1.0
        alpha_l2 = 1.0

        # --------------------------
        # Training
        # --------------------------
        global_step = 0
        for epoch in range(epochs):
            print(f"\n🚀 Starting epoch {epoch+1}/{epochs}")
            pbar = tqdm(dataloader, desc=f"[Distill] Epoch {epoch+1}/{epochs}", mininterval=1.0)

            for step, batch in enumerate(pbar):
                # Optional cap for Kaggle so kernel doesn't run forever
                if IS_KAGGLE and step >= KAGGLE_MAX_STEPS:
                    print(f"⏹ Reached KAGGLE_MAX_STEPS={KAGGLE_MAX_STEPS} for this epoch, stopping early.")
                    break

                (
                    bev, mask_emp, mask_non_emp, mask_union,
                    mask_emp_np, mask_non_emp_np, mask_union_np,
                    ph, pw, img
                ) = batch

                img = img.to(device, non_blocking=True)  # (B,3,H,W)

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

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optim.step()

                pbar.set_postfix({
                    "cos": f"{cos_loss.item():.4f}",
                    "l2": f"{l2_loss.item():.4f}",
                    "loss": f"{loss.item():.4f}",
                })

                # Comet logging (light)
                if global_step % 10 == 0:
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

        raise  # so you still see full traceback in Logs

    finally:
        experiment.end()


if __name__ == "__main__":
    main()