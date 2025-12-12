import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
#  IMPORT YOUR MODELS + DATASET + CONFIG
# ============================================================

from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from Utils.dataset import MapDataset
from config import (
    BATCH_SIZE,
    USE_BF16,
)

# ----------------------------
# Device
# ----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple Silicon MPS backend")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU detected — using CPU")

print(f"👉 Final device used for diagnostics: {device}")


# ============================================================
#  VERSION-SAFE WEIGHT LOADER (copied from your train script)
# ============================================================

def load_checkpoint_version_safe(model, ckpt_path, key=None, device="cpu"):
    print(f"\n🔍 Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and key is not None and key in ckpt:
        print(f"🔢 Checkpoint version: {ckpt.get('version', 'unknown')}")
        state = ckpt[key]
    else:
        state = ckpt

    model_state = model.state_dict()
    loaded, skipped = 0, 0

    for name, param in state.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state, strict=False)

    print(f"   ✔️ Loaded {loaded} params")
    print(f"   ⚠️ Skipped {skipped} params (mismatch or new layer)")

    return loaded, skipped


# ============================================================
#  EMBEDDING COLLECTION
# ============================================================

@torch.no_grad()
def collect_embeddings(
    model,
    loader,
    device,
    target_emb_count: int = 100_000,
    max_batches: int = None,
):
    """
    Run the PrimitiveLayer on a subset of the dataset and collect z_t embeddings.

    We assume model(...) -> (z_c, s_c, z_t)
    - z_t shape can be [B, D] or [B, N, D].
    - We flatten over batch (and optionally tokens) to get [M, D].

    NOTE:
        - This function mirrors your training batch unpacking.
        - It ignores the BEV tensor itself and uses the JEPA outputs (z_t).
    """
    model.eval()

    embs = []
    total = 0

    pbar = tqdm(loader, desc="Collecting embeddings")
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        (
            bev, mask_emp, mask_non_emp, mask_union,
            mask_emp_np, mask_non_emp_np, mask_union_np,
            ph, pw, img
        ) = batch

        B = bev.shape[0]

        mask_emp = mask_emp.to(device, non_blocking=True)
        mask_non_emp = mask_non_emp.to(device, non_blocking=True)
        mask_union = mask_union.to(device, non_blocking=True)

        # upsample helpers as in train script
        def up2(x):
            return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

        mask_emp_grid = mask_emp_np.to(device).view(B, 1, 32, 32).bool()
        mask_non_grid = mask_non_emp_np.to(device).view(B, 1, 32, 32).bool()
        mask_any_grid = mask_union_np.to(device).view(B, 1, 32, 32).bool()

        mask_emp_up = up2(mask_emp_grid)
        mask_non_up = up2(mask_non_grid)
        mask_any_up = up2(mask_any_grid)

        # JEPA forward
        z_c, s_c, z_t = model(
            mask_emp.squeeze(1),
            mask_non_emp.squeeze(1),
            mask_emp_up,
            mask_non_up,
            mask_any_up
        )

        # We treat z_t (target embeddings) as our diagnostic space
        # Normalize as in training
        z_t = F.normalize(z_t, dim=-1)

        # Shapes:
        # - if [B, D]
        # - or [B, N, D]
        if z_t.dim() == 3:
            B, N, D = z_t.shape
            z_flat = z_t.reshape(B * N, D)
        elif z_t.dim() == 2:
            z_flat = z_t
        else:
            raise ValueError(f"Unexpected z_t shape: {z_t.shape}")

        embs.append(z_flat.cpu())
        total += z_flat.shape[0]

        pbar.set_postfix({"collected": total})

        if total >= target_emb_count:
            break

    if not embs:
        raise RuntimeError("No embeddings collected; check loader/model behaviour.")

    Y = torch.cat(embs, dim=0)[:target_emb_count]
    print(f"✅ Collected embeddings: {Y.shape}")
    return Y


# ============================================================
#  BASIC METRICS (SECTION A)
# ============================================================

def compute_basic_metrics(Y: torch.Tensor, out_dir: Path):
    """
    Y: [M, C]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    Y = Y.to(torch.float32)

    M, C = Y.shape
    print(f"🔎 Basic metrics on Y with shape [M={M}, C={C}]")

    # Per-dim stats
    mu = Y.mean(dim=0)          # [C]
    centered = Y - mu
    var = centered.var(dim=0)   # [C]

    var_min = var.min().item()
    var_max = var.max().item()
    var_mean = var.mean().item()

    print(f"Per-dim variance: min={var_min:.4e}, max={var_max:.4e}, mean={var_mean:.4e}")

    # Plot variance histogram
    plt.figure()
    plt.hist(var.cpu().numpy(), bins=50)
    plt.title("Per-dimension variance")
    plt.xlabel("variance")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "var_hist.png")
    plt.close()

    # Norm distribution
    norms = Y.norm(dim=1)
    norms_mean = norms.mean().item()
    norms_std = norms.std().item()

    print(f"Norms: mean={norms_mean:.4f}, std={norms_std:.4f}")

    plt.figure()
    plt.hist(norms.cpu().numpy(), bins=50)
    plt.title("Embedding norm distribution")
    plt.xlabel("||z||2")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "norm_hist.png")
    plt.close()

    # Cosine similarity distribution
    N_pairs = min(5000, M)  # number of random pairs
    if M > 1:
        idx_a = torch.randint(0, M, (N_pairs,))
        idx_b = torch.randint(0, M, (N_pairs,))

        sub_a = Y[idx_a]
        sub_b = Y[idx_b]

        sub_a = sub_a / (sub_a.norm(dim=1, keepdim=True) + 1e-8)
        sub_b = sub_b / (sub_b.norm(dim=1, keepdim=True) + 1e-8)

        cos = (sub_a * sub_b).sum(dim=1).cpu().numpy()

        cos_mean = cos.mean()
        cos_std = cos.std()
    else:
        cos = np.array([0.0])
        cos_mean, cos_std = 0.0, 0.0

    print(f"Cosine similarity: mean={cos_mean:.4f}, std={cos_std:.4f}")

    plt.figure()
    plt.hist(cos, bins=50)
    plt.title("Random pair cosine similarity")
    plt.xlabel("cosine")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "cos_hist.png")
    plt.close()

    return {
        "var_min": var_min,
        "var_max": var_max,
        "var_mean": var_mean,
        "norm_mean": norms_mean,
        "norm_std": norms_std,
        "cos_mean": float(cos_mean),
        "cos_std": float(cos_std),
    }


# ============================================================
#  SPECTRAL METRICS (SECTION B)
# ============================================================

def compute_spectral_metrics(Y: torch.Tensor, out_dir: Path, max_samples: int = 40_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    Y = Y.to(torch.float32)

    M, C = Y.shape
    N = min(M, max_samples)

    # Subsample
    perm = torch.randperm(M)[:N]
    subset = Y[perm]
    subset = subset - subset.mean(dim=0, keepdim=True)

    print(f"📈 Computing covariance on subset [N={N}, C={C}]")
    cov = subset.t() @ subset / (N - 1)    # [C, C]

    # Eigendecomposition (symmetric)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-12).cpu().numpy()
    eigvals = eigvals[::-1]  # sort descending

    total = eigvals.sum()
    explained = eigvals / total
    cum_explained = np.cumsum(explained)

    # Scree / cumulative plots
    plt.figure()
    plt.plot(cum_explained)
    plt.title("Cumulative explained variance")
    plt.xlabel("# components")
    plt.ylabel("cumulative variance")
    plt.tight_layout()
    plt.savefig(out_dir / "cum_explained.png")
    plt.close()

    # Log-log spectrum
    ranks = np.arange(1, len(eigvals) + 1)
    plt.figure()
    plt.plot(np.log(ranks), np.log(eigvals))
    plt.title("Eigenvalue spectrum (log-log)")
    plt.xlabel("log(rank)")
    plt.ylabel("log(eigenvalue)")
    plt.tight_layout()
    plt.savefig(out_dir / "spectrum_loglog.png")
    plt.close()

    # Effective rank
    p = eigvals / eigvals.sum()
    H = -(p * np.log(p)).sum()
    eff_rank = float(np.exp(H))

    # Participation Ratio
    PR = float((eigvals.sum() ** 2) / (np.square(eigvals).sum()))

    # Isotropy index
    iso = float(eigvals.min() / eigvals.max())

    # Spectral gap
    if len(eigvals) >= 2:
        spectral_gap = float(eigvals[0] / eigvals[1])
    else:
        spectral_gap = 1.0

    # Whitening score ||cov - I||_F
    I = torch.eye(C, device=cov.device)
    ws = torch.norm(cov - I).item()

    print(f"Effective rank: {eff_rank:.2f} / {C}")
    print(f"Participation Ratio: {PR:.2f}")
    print(f"Isotropy index (lambda_min/lambda_max): {iso:.4e}")
    print(f"Spectral gap (lambda1/lambda2): {spectral_gap:.4f}")
    print(f"Whitening score ||Cov - I||_F: {ws:.4f}")

    return {
        "eff_rank": eff_rank,
        "participation_ratio": PR,
        "isotropy_index": iso,
        "spectral_gap": spectral_gap,
        "whitening_score": ws,
        "eigvals": eigvals.tolist(),   # raw spectrum if you want later
    }


# ============================================================
#  JEPA-SPECIFIC METRICS (empty-token separation stub)
# ============================================================

@torch.no_grad()
def compute_empty_token_separation(
    model,
    loader,
    device,
    out_dir: Path,
    max_cells: int = 50_000,
):
    """
    This is provided as a *template* because the exact attribute names
    for the empty token + BEV encoder are unknown from the snippet.

    Fill in the parts for:
      - how to get z_empty (empty token embedding)
      - how to get per-cell BEV embeddings + masks

    For now it returns None and prints a warning.
    """
    print("⚠️ compute_empty_token_separation: stub implementation.")
    print("   → Fill this in once you know where the empty token and BEV grid live.")
    return None


# ============================================================
#  HEALTH EVALUATION (PASS / WARN / FAIL)
# ============================================================

def evaluate_health(basic, spectral, dim_C: int):
    """
    Apply heuristic thresholds to classify PASS/WARNING/FAIL.
    You can tweak these based on actual runs.
    """
    status = {}

    # 1. Variance (simple collapse check)
    var_mean = basic["var_mean"]
    if var_mean < 1e-4:
        status["variance"] = "FAIL"
    elif var_mean < 1e-3:
        status["variance"] = "WARN"
    else:
        status["variance"] = "PASS"

    # 2. Effective rank + PR
    eff_rank = spectral["eff_rank"]
    PR = spectral["participation_ratio"]

    def grade_dim(val):
        if val >= 0.5 * dim_C:
            return "PASS"
        elif val >= 0.2 * dim_C:
            return "WARN"
        else:
            return "FAIL"

    status["effective_rank"] = grade_dim(eff_rank)
    status["participation_ratio"] = grade_dim(PR)

    # 3. Isotropy index
    iso = spectral["isotropy_index"]
    if iso >= 0.05:
        status["isotropy"] = "PASS"
    elif iso >= 0.01:
        status["isotropy"] = "WARN"
    else:
        status["isotropy"] = "FAIL"

    # 4. Spectral gap
    gap = spectral["spectral_gap"]
    if gap < 3.0:
        status["spectral_gap"] = "PASS"
    elif gap < 10.0:
        status["spectral_gap"] = "WARN"
    else:
        status["spectral_gap"] = "FAIL"

    # 5. Cosine spread
    cos_std = abs(basic["cos_std"])
    if cos_std < 0.02:
        status["cosine_spread"] = "FAIL"
    elif cos_std < 0.05:
        status["cosine_spread"] = "WARN"
    else:
        status["cosine_spread"] = "PASS"

    return status


# ============================================================
#  MAIN ENTRY POINT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to PrimitiveLayer checkpoint")
    parser.add_argument("--distilled_path", type=str, required=True,
                        help="Path to DINO/BEV distilled init (for PrimitiveLayer init)")
    parser.add_argument("--map_csv", type=str, default=None,
                        help="Path to MAP_CSV (if not using env var)")
    parser.add_argument("--out_dir", type=str, default="embedding_health_outputs",
                        help="Output directory for metrics + plots")
    parser.add_argument("--target_embs", type=int, default=100_000,
                        help="Number of embeddings to collect")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optional limit of batches for embedding collection")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Build model
    # ----------------------------
    print("\n📦 Building PrimitiveLayer model...")
    primitive_layer = PrimitiveLayer(
        embed_dim=128,
        distilled_path=args.distilled_path
    ).to(device)

    # Load checkpoint
    load_checkpoint_version_safe(
        primitive_layer,
        args.ckpt,
        key='state',
        device=device
    )

    # ----------------------------
    # Build dataset + loader
    # ----------------------------
    map_csv = args.map_csv or os.getenv("MAP_CSV")
    if map_csv is None:
        raise ValueError("MAP_CSV not provided (arg or env).")

    print(f"\n🗺  Using MAP dataset from: {map_csv}")
    map_ds = MapDataset(map_csv_file=map_csv)

    loader = DataLoader(
        map_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=False,
        pin_memory=True if device.type == "cuda" else False
    )

    # ----------------------------
    # Collect embeddings
    # ----------------------------
    Y = collect_embeddings(
        primitive_layer,
        loader,
        device,
        target_emb_count=args.target_embs,
        max_batches=args.max_batches
    )

    M, C = Y.shape

    # ----------------------------
    # Compute metrics
    # ----------------------------
    basic_metrics = compute_basic_metrics(Y, plots_dir)
    spectral_metrics = compute_spectral_metrics(Y, plots_dir)

    # JEPA-specific empty-token metric (stub for now)
    empty_token_metrics = compute_empty_token_separation(
        primitive_layer,
        loader,
        device,
        plots_dir
    )

    # ----------------------------
    # Evaluate health
    # ----------------------------
    health_status = evaluate_health(basic_metrics, spectral_metrics, dim_C=C)

    # ----------------------------
    # Save metrics + report
    # ----------------------------
    metrics = {
        "basic": basic_metrics,
        "spectral": spectral_metrics,
        "empty_token": empty_token_metrics,
        "health_status": health_status,
        "M": M,
        "C": C,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Simple text report
    lines = []
    lines.append("=== Tier-1 Embedding Health Check Report ===\n")
    lines.append(f"#Embeddings: {M}, Dim: {C}\n")
    lines.append("== Basic Stats ==\n")
    for k, v in basic_metrics.items():
        lines.append(f"{k}: {v}\n")

    lines.append("\n== Spectral Stats ==\n")
    for k, v in spectral_metrics.items():
        if k == "eigvals":
            continue
        lines.append(f"{k}: {v}\n")

    lines.append("\n== Health Status (PASS/WARN/FAIL) ==\n")
    for k, v in health_status.items():
        lines.append(f"{k}: {v}\n")

    with open(out_dir / "report.txt", "w") as f:
        f.writelines(lines)

    print("\n✅ Saved metrics →", out_dir / "metrics.json")
    print("✅ Saved plots   →", plots_dir)
    print("✅ Saved report  →", out_dir / "report.txt")
    print("\n🎉 Embedding health check completed.")


if __name__ == "__main__":
    main()