
import copy
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
import numpy as np
from monai.visualize import matshow3d

class EMACallback(pl.Callback):
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_state_dict = {}

    def on_fit_start(self, trainer, pl_module):
        # Initialize the shadow weights exactly matching the starting Generator
        self.ema_state_dict = {
            k: v.clone().detach() 
            for k, v in pl_module.vae.state_dict().items()
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Smoothly update the shadow weights after every single batch
        with torch.no_grad():
            for k, v in pl_module.vae.state_dict().items():
                if v.dtype.is_floating_point:
                    self.ema_state_dict[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save the shadow weights to the hard drive
        checkpoint["ema_state_dict"] = self.ema_state_dict

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Restore the shadow weights when the 8-hour job resumes
        if "ema_state_dict" in checkpoint:
            self.ema_state_dict = checkpoint["ema_state_dict"]

class SlabVisualizationCallback(pl.Callback):
    def __init__(self, every_n_epochs=5):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            
            full_volume = batch["rsp"][0:1] 
            slab = full_volume[:, :, :, :, :32].to(pl_module.device)
            cond_tensor = batch["condition"][0] # The 8-element vector
            active_indices = torch.where(cond_tensor == 1)[0].tolist()
            cond_str = f"Conds: {active_indices}" if active_indices else "healthy"
            
            with torch.no_grad():
                pl_module.eval()
                
                # 1. The Noisy Vanilla Pass
                recon_vanilla, _, _ = pl_module.vae(slab)
                
                # 2. Swap to the Smooth EMA Weights
                ema_callback = next((c for c in trainer.callbacks if isinstance(c, EMACallback)), None)
                original_state_dict = {k: v.clone() for k, v in pl_module.vae.state_dict().items()}
                
                if ema_callback is not None:
                    pl_module.vae.load_state_dict(ema_callback.ema_state_dict)
                    recon_ema, _, _ = pl_module.vae(slab) # The EMA Pass!
                else:
                    recon_ema = recon_vanilla # Fallback safety
                    
                # 3. CRITICAL: Restore Vanilla Weights immediately for the next training step
                pl_module.vae.load_state_dict(original_state_dict)
                pl_module.train() 
            
            # 4. Render and beam to WandB (Master GPU only)
            if trainer.is_global_zero: 
                slab_cpu = slab.squeeze(0).cpu().numpy()
                vanilla_cpu = recon_vanilla.squeeze(0).cpu().numpy()
                ema_cpu = recon_ema.squeeze(0).cpu().numpy()
                
                # Combine all 3 into shape: [3, H, W, D]
                combined = np.concatenate([slab_cpu, vanilla_cpu, ema_cpu], axis=0)
                
                fig = plt.figure(figsize=(16, 12)) # Made slightly taller for 3 rows
                
                matshow3d(
                    volume=combined,
                    fig=fig,
                    title=f"Epoch {trainer.current_epoch}| {cond_str}| | Top: Real | Mid: Vanilla | Bot: EMA Recon",
                    vmin=0.0, vmax=1.0, 
                    every_n=8,          
                    frame_dim=-1,       
                    frames_per_row=4,   # 12 total frames / 4 per row = exactly 3 rows!
                    show=False          
                )
                
                trainer.logger.experiment.log({
                    "Validation Reconstruction (Vanilla vs EMA)": wandb.Image(fig)
                })
                
                plt.close(fig)
