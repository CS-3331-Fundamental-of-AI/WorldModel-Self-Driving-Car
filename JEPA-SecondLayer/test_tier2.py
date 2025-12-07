# ============================================================
# Tier-2 Test Script (Environment-Driven, Clean Rewrite)
# ============================================================

import os
from dotenv import load_dotenv
import torch
from comet_ml import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm
from TrajTokenizer import TrajectoryTokenizerFSQ
from LatentCollector import LatentCollector
from EDAIN import EDAINLayer
from dataset import Tier2Dataset, tier2_collate_fn
from losses import total_tokenizer_loss_fsq
from kinematic import build_scene_mapping_parallel
from utils import build_graph_batch
from jepaTier2 import Tier2Module
# ------------------------------------------------------------
# 0. LOAD ENV + DEVICE
# ------------------------------------------------------------
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")


# ------------------------------------------------------------
# 1. BUILD & LOAD MODELS
# ------------------------------------------------------------
train_traj_tokenizer = TrajectoryTokenizerFSQ(
    traj_dim=6,
    T=8,
    enc_latent_dim=128,
    d_q=6,          # FSQ bottleneck
    fsq_levels=7,   # scalar levels per dim
    enc_layers=3,
    dec_hidden=64,
    use_layernorm=True
).to(device)

edain = EDAINLayer(D=6).to(device)

# ---- Load checkpoints (env-driven) ----
TOKENIZER_CKPT = os.getenv("TOKENIZER_CKPT")
EDAIN_CKPT     = os.getenv("EDAIN_CKPT")

if TOKENIZER_CKPT:
    print(f"→ Loading tokenizer from {TOKENIZER_CKPT}")
    train_traj_tokenizer.load_state_dict(
        torch.load(TOKENIZER_CKPT, map_location=device),
        strict=False
    )
else:
    print("⚠️ TOKENIZER_CKPT missing — using fresh weights.")

if EDAIN_CKPT:
    print(f"→ Loading EDAIN from {EDAIN_CKPT}")
    edain.load_state_dict(
        torch.load(EDAIN_CKPT, map_location=device),
        strict=False
    )
else:
    print("⚠️ EDAIN_CKPT missing — using fresh weights.")

train_traj_tokenizer.eval()
edain.eval()
print("✓ Models loaded & set to eval mode.")


# ------------------------------------------------------------
# 2. EXPERIMENT INIT (ENV-BASED)
# ------------------------------------------------------------
COMET_API_KEY = os.getenv("API_KEY")
COMET_PROJECT = os.getenv("PROJECT_NAME", "jepa-tier-2")
COMET_WORKSPACE = os.getenv("WORK_SPACE", "dtj-tran")
COMET_EXPERIMENT_NAME = os.getenv(
    "COMET_EXPERIMENT_NAME",
    "TrajectoryTokenizer-T2-with-EDAIN-FSQ-TEST"
)

if COMET_API_KEY is None:
    raise RuntimeError("Missing API_KEY environment variable.")

experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name=COMET_PROJECT,
    workspace=COMET_WORKSPACE
)

experiment.set_name(COMET_EXPERIMENT_NAME)
experiment.log_parameters({
    "batch_size": int(os.getenv("BATCH_SIZE", 8)),
    "traj_dim": int(os.getenv("TRAJ_DIM", 6)),
    "smooth_lambda": float(os.getenv("SMOOTH_LAMBDA", 0.1)),
    "usage_lambda": float(os.getenv("USAGE_LAMBDA", 0.1)),
    "normalization": os.getenv("NORMALIZATION", "EDAIN"),
    "fsq_levels": train_traj_tokenizer.fsq_levels,
    "fsq_dim": train_traj_tokenizer.d_q,
})

latent_probe_test = LatentCollector(latent_dim=128, max_batches=200)


# ------------------------------------------------------------
# 3. BUILD TEST DATASET
# ------------------------------------------------------------
test_dataset_path = os.getenv("TEST_DATASET_PATH", "/kaggle/input/test-dataset")
metadata_path = os.getenv(
    "TEST_METADATA_PATH",
    os.path.join(test_dataset_path, "metadata")
)

test_scene_map = build_scene_mapping_parallel(metadata_path)

env_test_dataset = Tier2Dataset(
    scene_map=test_scene_map,
    dataset_path=test_dataset_path,
    augment=False
)

val_loader = DataLoader(
    env_test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=tier2_collate_fn
)


# ------------------------------------------------------------
# 4. FSQ TOKEN HISTOGRAM
# ------------------------------------------------------------
L = train_traj_tokenizer.fsq_levels
token_hist = torch.zeros(L, dtype=torch.long)


# ------------------------------------------------------------
# 5. VALIDATION LOOP
# ------------------------------------------------------------
val_bar = tqdm(val_loader, desc="[Test]")

val_loss = val_recon = val_smooth = val_usage = 0.0

with torch.no_grad():
    for batch in val_bar:
        traj_raw = batch["clean_deltas"].to(device)
        traj = edain(traj_raw)

        model_out = train_traj_tokenizer(traj, tau=None)

        # collect latent
        latent_probe_test.add_batch(model_out["z0"])

        # compute tokens
        tokens = train_traj_tokenizer.encode_tokens(traj)
        flat = tokens.view(-1).cpu()
        token_hist.scatter_add_(0, flat, torch.ones_like(flat))

        # compute fsq validation loss
        loss_dict = total_tokenizer_loss_fsq(
            out_clean=model_out,
            out_aug=model_out,
            target_traj=traj,
            fsq_levels=train_traj_tokenizer.fsq_levels,
            lambda_recon=1.0,
            lambda_smooth=0.1,
            lambda_usage=0.1
        )

        lt, lr, ls, lu = (
            loss_dict["loss_total"].item(),
            loss_dict["loss_recon"].item(),
            loss_dict["loss_smooth"].item(),
            loss_dict["fsq_usage"].item(),
        )

        val_loss   += lt
        val_recon  += lr
        val_smooth += ls
        val_usage  += lu

        val_bar.set_postfix({
            "loss": f"{lt:.4f}",
            "recon": f"{lr:.4f}",
            "smooth": f"{ls:.4f}",
            "usage": f"{lu:.4f}",
        })


# ------------------------------------------------------------
# 6. SUMMARY + LOGGING
# ------------------------------------------------------------
N = len(val_loader)

val_loss   /= N
val_recon  /= N
val_smooth /= N
val_usage  /= N

print("\n====== TEST SUMMARY ======")
print(f"Loss Total : {val_loss:.4f}")
print(f"Recon      : {val_recon:.4f}")
print(f"Smooth     : {val_smooth:.4f}")
print(f"Usage      : {val_usage:.4f}")
print("\nFSQ token histogram:")
print(token_hist)

experiment.log_metrics({
    "val/loss_total": val_loss,
    "val/loss_recon": val_recon,
    "val/loss_smooth": val_smooth,
    "val/usage": val_usage,
})

for level in range(L):
    experiment.log_metric(
        f"val/token_level_{level}", token_hist[level].item()
    )

# build tier-2-test-dataset for test set
from torch.utils.data import DataLoader
from tqdm import tqdm

type_set_test = set()
category_set_test = set()
layer_set_test = set()

print("=== SCANNING ALL CATEGORICAL VALUES ===")

for batch in tqdm(val_loader):
    graphs = batch["graphs"]

    for G in graphs:
        for _, data in G.nodes(data=True):

            # type
            t = data.get("type")
            if isinstance(t, str):
                type_set_test.add(t)

            # category
            c = data.get("category")
            if isinstance(c, str):
                category_set_test.add(c)

            # layer
            l = data.get("layer")
            if isinstance(l, str):
                layer_set_test.add(l)

# print("\n=== UNIQUE VALUES DISCOVERED ===")
# print("type:", sorted(type_set_test))
# print("category:", sorted(category_set_test))
# print("layer:", sorted(layer_set_test))


types_value_test = (sorted(type_set_test))
category_values_test = sorted(category_set_test)
layer_values = sorted(layer_set_test)

type2id_test = {v:i for i, v in enumerate(types_value_test) }
category2id_test = {v:i for i,v in enumerate (category_values_test) }
layer2id_test = {v:i for i,v in enumerate (layer_values) }



# -------------------------------------------------------------
# 1) Fetch a batch
# -------------------------------------------------------------
batch = next(iter(val_loader))

graphs      = batch["graphs"]        # list of NetworkX graphs
graph_mask  = batch["graph_mask"]    # [B, maxN]
traj        = batch["clean_deltas"]  # [B, T, 6]

# -------------------------------------------------------------
# 2) Build graph batch using your function (takes the category id & the graphs)
# -------------------------------------------------------------

x_batch, adj_batch = build_graph_batch(graphs, type2id_test, category2id_test, layer2id_test)
# x_batch:  [B, maxN, 13]
# adj_batch: [B, maxN, maxN]

# -------------------------------------------------------------
# 3) Forward pass through Tier-2
# -------------------------------------------------------------
test_tier_2 = Tier2Module()

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

