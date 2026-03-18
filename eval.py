import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from monai.visualize import matshow3d
from FLORA.model import FLORA
from FLORA.image_loader import get_train_loader

def generate_eval_samples(ckpt_path, num_samples=5):
    print(f"Loading weights from {ckpt_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 3D inference on: {device}")

    # 1. Load the normal Vanilla model
    model = FLORA.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    model.to(device)

    # 2. Crack open the raw checkpoint to extract the EMA weights
    raw_checkpoint = torch.load(ckpt_path, map_location=device)
    if "ema_state_dict" in raw_checkpoint:
        ema_state_dict = raw_checkpoint["ema_state_dict"]
        print("Successfully found and loaded EMA shadow weights!")
    else:
        print("WARNING: No 'ema_state_dict' found in checkpoint. Falling back to Vanilla.")
        ema_state_dict = None

    # Load validation data
    _, val_loader = get_train_loader(batch_size=1, num_workers=2)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
                
            print(f"\n--- Processing volume {i+1}/{num_samples} ---")
            
            # Extract the 32-depth slab
            full_volume = batch["rsp"][0:1]
            slab = full_volume[:, :, :, :, :32].to(device)
            
            # Parse the multi-label condition vector for the filename and title
            cond_tensor = batch["condition"][0] 
            active_indices = torch.where(cond_tensor == 1)[0].tolist()
            cond_str = "-".join(map(str, active_indices)) if active_indices else "Baseline"

            # --- FORWARD PASS 1: The Vanilla Weights ---
            recon_vanilla, _, _ = model.vae(slab)
            
            # --- FORWARD PASS 2: The EMA Weights ---
            if ema_state_dict is not None:
                # Save the vanilla brain
                original_state_dict = {k: v.clone() for k, v in model.vae.state_dict().items()}
                
                # Load the EMA brain and generate
                model.vae.load_state_dict(ema_state_dict)
                recon_ema, _, _ = model.vae(slab)
                
                # Restore the vanilla brain for the next loop
                model.vae.load_state_dict(original_state_dict)
            else:
                recon_ema = recon_vanilla
            
            # Move to CPU and format for MONAI visualizer
            slab_cpu = slab.squeeze(0).cpu().numpy()
            vanilla_cpu = recon_vanilla.squeeze(0).cpu().numpy()
            ema_cpu = recon_ema.squeeze(0).cpu().numpy()
            
            # Stack them: [3, Height, Width, Depth]
            combined = np.concatenate([slab_cpu, vanilla_cpu, ema_cpu], axis=0)
            
            # Made the figure taller (12) to fit 3 beautiful rows
            fig = plt.figure(figsize=(16, 12))
            
            matshow3d(
                volume=combined,
                fig=fig,
                title=f"Sample {i+1} | Cond: {cond_str} | Top: Real | Mid: Vanilla | Bot: EMA",
                vmin=0.0, vmax=1.0, 
                every_n=8,          
                frame_dim=-1,       
                frames_per_row=4,   
                show=False          
            )
            
            os.makedirs("eval_outputs", exist_ok=True)
            save_path = os.path.join("eval_outputs", f"sample_{i+1}_cond_{cond_str}.png")
            
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"Successfully saved {save_path}")

if __name__ == "__main__":
    target_ckpt = os.path.join("checkpoints", "PHASE_1", "last.ckpt")
    generate_eval_samples(target_ckpt, num_samples=5)