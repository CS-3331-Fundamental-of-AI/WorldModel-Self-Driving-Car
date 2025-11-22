from torch.utils.data import DataLoader
from Utils.dataset import MapDataset
from config import DEVICE, LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1, BETA_1, BETA_2, GAMMA
import torch
from jepa_1 import PrimitiveLayer
from comet_ml import Experiment
from tqdm import tqdm
import torch.nn.functional as F
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update

map_ds = MapDataset(map_csv_file="/content/map_files.csv")
dataloader = DataLoader(map_ds, batch_size=8, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
primitive_layer = PrimitiveLayer(embed_dim=128).to(device)
optimizer = torch.optim.Adam(primitive_layer.parameters(), lr=1e-4)

# Inspect & Test The Dataset Object
i=100
bev = map_ds[i][0]             # B x C x H x W
bme = map_ds[i][1]             # B x C x H x W
bmne = map_ds[i][2]            # B x C x H x W
bma = map_ds[i][3]             # B x C x H x W
mask_emp_np = map_ds[i][4]     # B x (ph x pw) = 32 x 32


experiment = Experiment(
    api_key="YOUR_API_KEY",
    project_name="jepa-training",
    workspace="YOUR_WORKSPACE",
)

experiment.set_name("JEPA-PrimitiveLayer-v1")
experiment.add_tag("jepa")
experiment.add_tag("primitive-layer")
experiment.add_tag("masked-tokens")

experiment.log_parameters({
    "lambda_jepa": LAMBDA_JEPA,
    "lambda_reg": LAMBDA_REG,
    "alpha0": ALPHA_0,
    "alpha1": ALPHA_1,
    "beta1": BETA_1,
    "beta2": BETA_2,
    "gamma": GAMMA,
    "lr": optimizer.param_groups[0]["lr"],
})

loss_history = {
    "total": [],
    "jepa": [],
    "empty": [],
    "nonempty": [],
    "reg": []
}

primitive_layer.train()

for batch in tqdm(dataloader, desc="Training JEPA"):

    # ----------------------------------------------------------
    # Unpack batch
    # ----------------------------------------------------------
    (
        bev,
        mask_emp,
        mask_non_emp,
        mask_union,
        mask_emp_np,
        mask_non_emp_np,
        mask_union_np,
        ph,
        pw,
        img
    ) = batch

    B = bev.shape[0]

    # ----------------------------------------------------------
    # Move everything to device BEFORE forward()
    # ----------------------------------------------------------
    bev = bev.squeeze(1).to(device)

    mask_emp      = mask_emp.to(device)
    mask_non_emp  = mask_non_emp.to(device)
    mask_union    = mask_union.to(device)

    mask_emp_np      = mask_emp_np.to(device)
    mask_non_emp_np  = mask_non_emp_np.to(device)
    mask_union_np    = mask_union_np.to(device)

    # ----------------------------------------------------------
    # Build 32×32 grid masks
    # ----------------------------------------------------------
    mask_emp_grid  = mask_emp_np.view(B, 1, 32, 32).float()
    mask_non_grid  = mask_non_emp_np.view(B, 1, 32, 32).float()
    mask_any_grid  = mask_union_np.view(B, 1, 32, 32).float()

    # ----------------------------------------------------------
    # Upsample to 64×64 for latent injection
    # ----------------------------------------------------------
    mask_emp_up  = F.interpolate(mask_emp_grid, size=(64, 64), mode="nearest")
    mask_non_up  = F.interpolate(mask_non_grid, size=(64, 64), mode="nearest")
    mask_any_up  = F.interpolate(mask_any_grid, size=(64, 64), mode="nearest")

    # ----------------------------------------------------------
    # Forward through primitive layer
    # ----------------------------------------------------------
    z_c, s_c, z_t = primitive_layer.forward(
        mask_emp.squeeze(1).to(device),
        mask_non_emp.squeeze(1).to(device),
        mask_emp_up,
        mask_non_up,
        mask_any_up
    )

    # ----------------------------------------------------------
    # Normalize latent vectors
    # ----------------------------------------------------------
    z_c_norm = F.normalize(z_c, dim=-1)
    s_c_norm = F.normalize(s_c, dim=-1)
    z_t_norm = F.normalize(z_t, dim=-1)

    # ----------------------------------------------------------
    # Flatten 64×64 masks → reduce to per-batch mask flags
    # ----------------------------------------------------------
    mask_non_flat = mask_non_up.bool()  # (B, 1, 64, 64) → bool
    mask_emp_flat = mask_emp_up.bool()

    # reduce across all channels except batch
    mask_non_flat = mask_non_flat.view(B, -1)
    mask_emp_flat = mask_emp_flat.view(B, -1)

    # ----------------------------------------------------------
    # Compute JEPA loss
    # ----------------------------------------------------------
    losses = compute_jepa_loss(
        s_c=s_c_norm,
        s_t=z_t_norm,
        z_c=z_c_norm,
        mask_empty=mask_emp_flat,
        mask_nonempty=mask_non_flat,
        alpha0=ALPHA_0,
        alpha1=ALPHA_1,
        beta1=BETA_1,
        beta2=BETA_2,
        lambda_jepa=LAMBDA_JEPA,
        lambda_reg=LAMBDA_REG,
        gamma=GAMMA,
    )

    experiment.log_metric("loss_total", losses["loss_total"].item())
    experiment.log_metric("loss_jepa", losses["loss_jepa"].item())
    experiment.log_metric("loss_empty", losses["loss_P_empty"].item())
    experiment.log_metric("loss_nonempty", losses["loss_Q_nonempty"].item())
    experiment.log_metric("loss_reg", losses["loss_reg"].item())

    # ----------------------------------------------------------
    # RECORD LOSSES LOCALY
    # ----------------------------------------------------------
    loss_history["total"].append(losses["loss_total"].item())
    loss_history["jepa"].append(losses["loss_jepa"].item())
    loss_history["empty"].append(losses["loss_P_empty"].item())
    loss_history["nonempty"].append(losses["loss_Q_nonempty"].item())
    loss_history["reg"].append(losses["loss_reg"].item())

    # ----------------------------------------------------------
    # Backprop + optimization
    # ----------------------------------------------------------
    optimizer.zero_grad()
    loss = losses["loss_total"]
    loss.backward()
    optimizer.step()

    # ----------------------------------------------------------
    # EMA update
    # ----------------------------------------------------------
    ema_update(
        primitive_layer.context_encoder,
        primitive_layer.target_encoder,
        primitive_layer.ema_decay
    )

# model saving (checkpoint)
torch.save(primitive_layer.state_dict(), "primitive_layer.pt")
experiment.log_asset("primitive_layer.pt")

experiment.log_metric("final_loss_total", losses["loss_total"].item())
experiment.end() # END EXPERIMENT

# go to this for checking the process: https://www.comet.com/YOUR_WORKSPACE/jepa-training