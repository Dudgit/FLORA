"""
FLORA Slice Visualizer
──────────────────────
Loads the first validation sample, runs it through the model, and saves
one PNG per image:

  ct_slice_z{z}.png         — CT axial slice      (gray)
  ct_latent_ch{c}.png       — CT latent channel   (viridis)
  rsp_latent_ch{c}.png      — RSP latent channel  (plasma)
  pred_rsp_z{z}.png         — Predicted RSP slice (plasma)
  gt_rsp_z{z}.png           — Ground-truth RSP    (plasma)
  error_z{z}.png            — Absolute error      (hot)
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# ── adjust to your project layout ─────────────────────────────────────────
from FLORA.model_CT import FLORA_CT_changedBlock
from FLORA.image_loader import get_train_loader_CT

# ── config ─────────────────────────────────────────────────────────────────
CKPT_PATH   = "checkpoints/PHASE_2_CT_ONLY/last-v1.ckpt"
CONFIG_PATH = "config.yaml"
ODE_STEPS   = 10
N_SLICES    = 6       # evenly spaced axial slices
RSP_SCALE   = 3.2
OUT_DIR     = "article_vis/slices"

CMAP_CT     = "gray"
CMAP_LATENT = "viridis"
CMAP_RSP    = "plasma"
CMAP_ERR    = "hot"

FIG_SIZE    = (4, 4)   # each individual image
DPI         = 200


# ───────────────────────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────────────────────

def load_model(cfg, ckpt_path, device):
    velocity_kwargs = dict(cfg.velocity_kwargs)
    velocity_kwargs["in_channels"]       = 4
    velocity_kwargs["with_conditioning"] = False
    velocity_kwargs.pop("cross_attention_dim", None)

    model = FLORA_CT_changedBlock(
        vaekwgs              = cfg.vae_kwgs,
        velocity_kwargs      = velocity_kwargs,
        detectorproj_kwargs  = cfg.detectorproj_kwargs,
        use_detector_context = False,
    ).to(device)

    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ───────────────────────────────────────────────────────────────────────────
# Inference
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, ct, gt_rsp, device, ode_steps):
    ct     = ct.to(device)
    gt_rsp = gt_rsp.to(device)

    z0, _     = model.vae.encode(ct)
    z_t       = z0 * model.latent_scale
    ct_latent = z_t[0].cpu().numpy()              # (C, D, H, W)

    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t_batch = torch.full((1,), i * dt, device=device)
        v   = model.velocity_net(x=z_t, timesteps=t_batch)
        z_t = z_t + v * dt

    rsp_latent = z_t[0].cpu().numpy()             # (C, D, H, W)

    recon = model.vae.decode(z_t / model.latent_scale)
    if isinstance(recon, tuple):
        recon = recon[0]
    recon = torch.clamp(recon, 0.0, 1.0)

    return (ct_latent,
            rsp_latent,
            recon.squeeze().cpu().numpy(),         # (H, W, D)
            gt_rsp.squeeze().cpu().numpy())        # (H, W, D)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def pick_slices(vol_hwd, n):
    D = vol_hwd.shape[2]
    return np.linspace(int(D * 0.15), int(D * 0.85), n, dtype=int)


def norm_channel(arr_hw):
    mn, mx = arr_hw.min(), arr_hw.max()
    return (arr_hw - mn) / (mx - mn + 1e-8)


def save_img(arr_hw, cmap, path, vmin=0.0, vmax=1.0,
             title=None, title_color="white", colorbar=False):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="black")
    ax.set_facecolor("black")

    im = ax.imshow(arr_hw, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if title:
        ax.set_title(title, color=title_color, fontsize=11,
                     fontweight="bold", pad=6)

    if colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cb.outline.set_edgecolor("white")

    plt.tight_layout(pad=0.3)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  saved {os.path.basename(path)}")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg   = OmegaConf.load(CONFIG_PATH)
    model = load_model(cfg, CKPT_PATH, device)

    _, val_loader = get_train_loader_CT(batch_size=1, num_workers=0)
    batch  = next(iter(val_loader))
    ct_np  = batch["ct"].squeeze().numpy()        # (H, W, D)

    ct_lat, rsp_lat, pred_rsp, gt_rsp = run_inference(
        model, batch["ct"], batch["rsp"], device, ODE_STEPS
    )

    slice_idxs = pick_slices(ct_np, N_SLICES)

    # ── CT slices ──────────────────────────────────────────────────────────
    print("\nCT slices:")
    for z in slice_idxs:
        save_img(ct_np[:, :, z], CMAP_CT,
                 path=os.path.join(OUT_DIR, f"ct_slice_z{z:03d}.png"),
                 title=None, title_color="#4FC3F7",
                 colorbar=False)

    # ── CT latent channels (depth-averaged, normalised per channel) ────────
    print("\nCT latent channels:")
    for c in range(ct_lat.shape[0]):
        ch = norm_channel(ct_lat[c].mean(axis=0))   # mean over D → (H, W)
        save_img(ch, CMAP_LATENT,
                 path=os.path.join(OUT_DIR, f"ct_latent_ch{c}.png"),
                 title=None, title_color="#69F0AE",
                 colorbar=False)

    # ── RSP latent channels ────────────────────────────────────────────────
    print("\nRSP latent channels:")
    for c in range(rsp_lat.shape[0]):
        ch = norm_channel(rsp_lat[c].mean(axis=0))  # mean over D → (H, W)
        save_img(ch, CMAP_RSP,
                 path=os.path.join(OUT_DIR, f"rsp_latent_ch{c}.png"),
                 title=None, title_color="#CE93D8",
                 colorbar=False)

    # ── Predicted RSP slices ───────────────────────────────────────────────
    print("\nPredicted RSP slices:")
    for z in slice_idxs:
        save_img(pred_rsp[:, :, z], CMAP_RSP,
                 path=os.path.join(OUT_DIR, f"pred_rsp_z{z:03d}.png"),
                 title=None, title_color="#CE93D8",
                 colorbar=False)

    # ── Ground-truth RSP slices ────────────────────────────────────────────
    print("\nGround-truth RSP slices:")
    for z in slice_idxs:
        save_img(gt_rsp[:, :, z], CMAP_RSP,
                 path=os.path.join(OUT_DIR, f"gt_rsp_z{z:03d}.png"),
                 title=None, title_color="#CE93D8",
                 colorbar=False)

    total = N_SLICES * 4 + ct_lat.shape[0] * 2
    print(f"\nDone — {total} images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
