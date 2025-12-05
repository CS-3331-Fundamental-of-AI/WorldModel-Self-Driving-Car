# pretrain_distill.py
import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from JEPA_PrimitiveLayer.bev_jepa import BEVJEPAEncoder2D
from Utils.dataset import MapDataset

load_dotenv()

# --------------------------
# 1. Load DINO ResNet-50
# --------------------------
def load_dino_resnet50(device):
    model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    model.eval()
    model.to(device)
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
        returns: projected feature (B, teacher_dim)
        """
        feat = self.encoder(x)         # (B, C, Hc, Wc)
        if isinstance(feat, tuple):
            feat, _ = feat
        B, C, Hc, Wc = feat.shape

        # Global average pooling over spatial dims
        s_vec = feat.mean(dim=[2, 3])  # (B, C)
        s_proj = self.proj(s_vec)      # (B, teacher_dim)
        return s_proj, s_vec

# --------------------------
# 3. Training Loop
# --------------------------
def main():
    from comet_ml import Experiment
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
    )
    experiment.set_name("Distill-MobileNet-DINO")

    # Device selection (CPU/GPU/MPS)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU only (slow)")

    # Dataset — reuse your BEV map dataset
    map_csv = os.getenv("MAP_CSV")
    ds = MapDataset(map_csv_file=map_csv)

    # If you want faster pretraining: subsample dataset here
    # from torch.utils.data import Subset
    # ds = Subset(ds, range(0, 10000))  # first 10k samples

    dataloader = DataLoader(
        ds, batch_size=32, shuffle=True,
        num_workers=2, pin_memory=True if device.type == "cuda" else False
    )

    # Teacher
    teacher = load_dino_resnet50(device)

    # Student + head
    student = StudentWithHead(width_mult=0.5, teacher_dim=2048).to(device)

    if torch.cuda.device_count() > 1:
        print(f"⚡ Using {torch.cuda.device_count()} GPUs with DataParallel")
        student = nn.DataParallel(student)
        teacher = nn.DataParallel(teacher)

    try:
        # Optimizer (only student is trainable)
        optim = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.05)

        # Training config
        epochs = 5      # you can tune this
        alpha_cos = 1.0 # weight for cosine loss
        alpha_l2  = 1.0 # weight for L2 loss

        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"[Distill] Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                # Your MapDataset returns:
                # (bev, mask_emp, mask_non_emp, mask_union,
                #  mask_emp_np, mask_non_emp_np, mask_union_np,
                #  ph, pw, img)
                # We want the raw RGB image "img"
                (
                    bev, mask_emp, mask_non_emp, mask_union,
                    mask_emp_np, mask_non_emp_np, mask_union_np,
                    ph, pw, img
                ) = batch

                img = img.to(device)   # (B,3,H,W), same as JEPA

                # Forward teacher
                with torch.no_grad():
                    # DINO ResNet expects something like 224x224,
                    # but your img is likely 256x256 or similar;
                    # ResNet will still handle it (it is conv-only).
                    t_feat = teacher(img)           # (B, 2048)
                    t_feat = F.normalize(t_feat, dim=-1)

                # Forward student
                s_proj, s_vec = student(img)       # (B, 2048), (B, C)
                s_proj = F.normalize(s_proj, dim=-1)

                # Distill loss: cosine + L2
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

                experiment.log_metric("loss", loss.item())
                experiment.log_metric("cos_loss", cos_loss.item())
                experiment.log_metric("l2_loss", l2_loss.item())

    except Exception as e:
        torch.save(student.module.encoder.state_dict() if isinstance(student, nn.DataParallel) else student.encoder.state_dict(), "distill_fail.pt")
        with open("distill_error_log.txt", "w") as f:
            f.write(str(e))
        experiment.log_asset("distill_fail.pt")
        experiment.log_text(str(e), metadata={"phase": "distill_fail"})
        raise e

    # Save only the BEVJEPAEncoder2D weights
    torch.save(student.module.encoder.state_dict() if isinstance(student, nn.DataParallel) else student.encoder.state_dict(), "bev_mobilenet_dino_init.pt")
    print("✅ Saved distilled MobileNet encoder → bev_mobilenet_dino_init.pt")

    experiment.log_asset("bev_mobilenet_dino_init.pt")
    experiment.end()

if __name__ == "__main__":
    main()