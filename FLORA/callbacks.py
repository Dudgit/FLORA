
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

class Stage2VisualizationCallback(pl.Callback):
    def __init__(self, every_n_epochs=5, ode_steps=10):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.ode_steps = ode_steps # Number of steps to draw the straight line from x0 to x1

    def solve_ode(self, pl_module, physics_grid, medical_labels, spatial_shape):
        """
        Runs the Euler integration loop to step from x0 to x1.
        """
        B = physics_grid.shape[0]
        device = physics_grid.device
        
        # CRITICAL FIX: Standardize the physics grid using the module's saved buffers!
        scaled_physics_grid = (physics_grid - pl_module.cond_mean) / pl_module.cond_std
        
        # 1. Process physics conditions using the SCALED grid
        cond_features = pl_module.projection(scaled_physics_grid)
        
        if cond_features.ndim == 5:
            cond_features = cond_features.squeeze(1)
        cond_seq = cond_features.view(B, -1, cond_features.shape[-1])
        
        # 2. Get informed prior (x0) using the medical labels
        x_t = pl_module.generate_informed_x0(medical_labels, spatial_shape)
        
        # 3. Euler Integration Loop from t=0 to t=1
        dt = 1.0 / self.ode_steps
        for i in range(self.ode_steps):
            t_val = i * dt
            t_tensor = torch.full((B,), t_val, device=device)
            t_scaled = t_tensor * 1000.0  # MONAI expects 0-1000 scale
            
            # Predict velocity
            v_t = pl_module.velocity_net(x=x_t, timesteps=t_scaled, context=cond_seq)
            
            # Take a straight step
            x_t = x_t + v_t * dt
            
        # 4. Decode the final latent (x1) back into a 3D physical volume
        recon_volume = pl_module.vae.decode(x_t/pl_module.latent_scale)
        
        # Some MONAI versions return a tuple from decode, so we safely grab the first element
        if isinstance(recon_volume, tuple):
            recon_volume = recon_volume[0]
            
        return recon_volume

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            
            # 1. Unpack the batch
            #full_volume = batch["rsp"][0:1] 
            slab = batch["rsp"][0:1].to(pl_module.device)
            physics_grid = batch["physics_grid"][0:1].to(pl_module.device)
            medical_labels = batch["condition"][0:1].to(pl_module.device)
            
            # Format title string
            active_indices = torch.where(medical_labels[0] == 1)[0].tolist()
            cond_str = f"Conds: {active_indices}" if active_indices else "Healthy"
            
            with torch.no_grad():
                pl_module.eval()
                z_mu, _ = pl_module.vae.encode(slab)
                spatial_shape = z_mu.shape[2:]
                recon_vanilla = self.solve_ode(pl_module, physics_grid, medical_labels, spatial_shape)
                original_state_dict = {k: v.clone() for k, v in pl_module.state_dict().items()}
                pl_module.load_state_dict(original_state_dict)
                pl_module.train() 
            
            # 4. Render and push to W&B
            if trainer.is_global_zero: 
                slab_cpu = slab.squeeze(0).cpu().numpy()
                vanilla_cpu = recon_vanilla.squeeze(0).cpu().numpy()
                #ema_cpu = recon_ema.squeeze(0).cpu().numpy()
                
                combined = np.concatenate([slab_cpu, vanilla_cpu], axis=0)
                
                fig = plt.figure(figsize=(16, 8)) 
                
                matshow3d(
                volume=combined,fig=fig,
                title=f"Epoch {trainer.current_epoch} | {cond_str}\nTop: Ground Truth | Bottom: Generated",
                vmin=0.0, vmax=1.0, every_n=8,frame_dim=-1,frames_per_row=4,show=False)
                
                trainer.logger.experiment.log({"Validation Reconstruction (Vanilla vs EMA)": wandb.Image(fig)})
                plt.close(fig)


class PriorDebuggingCallback(pl.Callback):
    def __init__(self, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Only run on the very first batch of validation
        if batch_idx == 0 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            
            slab = batch["rsp"][0:1].to(pl_module.device)
            medical_labels = batch["condition"][0:1].to(pl_module.device)
            
            active_indices = torch.where(medical_labels[0] == 1)[0].tolist()
            cond_str = f"Conds: {active_indices}" if active_indices else "Healthy"
            
            with torch.no_grad():
                pl_module.eval()
                
                # Get the target spatial shape
                z_mu, _ = pl_module.vae.encode(slab)
                spatial_shape = z_mu.shape[2:]
                
                # --- REPLICATE PRIOR LOGIC SAFELY ---
                batch_size = medical_labels.shape[0]
                latent_dim = pl_module.condition_mu.shape[1]
                
                # --- THE NEW SPATIAL LOGIC ---
                label_counts = medical_labels.sum(dim=1, keepdim=True).clamp(min=1e-8)

                weighted_sum = torch.einsum('bc, cfdhw -> bfdhw', medical_labels.float(), pl_module.condition_mu)

                mu_3d = weighted_sum / label_counts.view(-1, 1, 1, 1, 1)

                scale = getattr(pl_module, "latent_scale", 1.0)
                mu_3d = mu_3d * scale
                
                x0 = mu_3d + torch.randn_like(mu_3d) 
                
                recon_clean_mean = pl_module.vae.decode(mu_3d / scale)
                recon_noisy_x0 = pl_module.vae.decode(x0 / scale)
                
                if isinstance(recon_clean_mean, tuple): recon_clean_mean = recon_clean_mean[0]
                if isinstance(recon_noisy_x0, tuple): recon_noisy_x0 = recon_noisy_x0[0]
                
                pl_module.train()
                
            if trainer.is_global_zero:
                gt_cpu = slab.squeeze(0).cpu().numpy()
                clean_cpu = recon_clean_mean.squeeze(0).cpu().numpy()
                noisy_cpu = recon_noisy_x0.squeeze(0).cpu().numpy()
                
                combined = np.concatenate([gt_cpu, clean_cpu, noisy_cpu], axis=0)
                
                fig = plt.figure(figsize=(16, 12))
                matshow3d(
                    volume=combined,
                    fig=fig,
                    title=f"Prior Debug | {cond_str}\nTop: Ground Truth | Mid: Clean Prior Mean | Bot: Noisy x0",
                    vmin=0.0, vmax=1.0, 
                    every_n=8,          
                    frame_dim=-1,       
                    frames_per_row=4,   
                    show=False
                )
                
                trainer.logger.experiment.log({
                    "Medical Prior Diagnostics": wandb.Image(fig)
                })
                plt.close(fig)

class CTtoPCTVisualizationCallback(pl.Callback):
    def __init__(self, every_n_epochs=5, num_steps=10):
        """
        Args:
            every_n_epochs: How often to run the expensive ODE integration.
            num_steps: Number of Euler integration steps for the Flow Matching ODE.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_steps = num_steps

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 1. Multi-GPU & Epoch Safety Checks
        if not trainer.is_global_zero:
            return
        if batch_idx != 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        device = pl_module.device
        
        # 2. Slice [0:1] to extract ONLY the first patient, maintaining [1, C, H, W, D] shape
        ct_images = batch["ct"][0:1].to(device)
        pct_images = batch["rsp"][0:1].to(device)
        medical_labels = batch["condition"][0:1].to(device)
        
        # 3. Format the condition string for the plot title
        active_indices = torch.where(medical_labels[0] == 1)[0].tolist()
        cond_str = f"Conds: {active_indices}" if active_indices else "Healthy"

        # 4. Context preparation
        if pl_module.use_detector_context:
            physics_grid = batch["physics_grid"][0:1].to(device)
            cond_features = pl_module.projection(physics_grid)
            B = cond_features.shape[0]
            context = cond_features.view(B, -1, cond_features.shape[-1])
        else:
            context = None

        # 5. Flow Matching Inference via Euler ODE Integration
        with torch.no_grad():
            z_0, _ = pl_module.vae.encode(ct_images)
            z_0 = z_0 * pl_module.latent_scale
            
            z_t = z_0.clone()
            dt = 1.0 / self.num_steps
            
            for i in range(self.num_steps):
                t_val = i * dt
                t_tensor = torch.full((z_t.shape[0],), t_val, device=device)
                v_pred = pl_module.velocity_net(x=z_t, timesteps=t_tensor, context=context)
                z_t = z_t + v_pred * dt
            
            # Decode back to image space
            z_1_unscaled = z_t / pl_module.latent_scale
            predicted_pct = pl_module.vae.decode(z_1_unscaled)

        # 6. Prepare NumPy arrays for MONAI (Extract first batch item and first channel)
        ct_cpu = ct_images[0, 0].cpu().numpy()
        true_pct_cpu = pct_images[0, 0].cpu().numpy()
        pred_pct_cpu = predicted_pct[0, 0].cpu().numpy()

        # Calculate the absolute difference (Error Map)
        diff_map_cpu = np.abs(true_pct_cpu - pred_pct_cpu)

        # Concatenate along the vertical height axis to create 4 rows
        combined = np.concatenate([ct_cpu, true_pct_cpu, pred_pct_cpu, diff_map_cpu], axis=0)

        # 7. Render the 3D volume slices using matshow3d
        # Increased figsize height to accommodate the 4th row comfortably
        fig = plt.figure(figsize=(16, 16))
        
        matshow3d(
            volume=combined,
            fig=fig,
            title=f"FLORA | Epoch {trainer.current_epoch} | {cond_str}\nRow 1: CT | Row 2: GT RSP | Row 3: Pred RSP | Row 4: Abs Error",
            vmin=0.0, 
            vmax=1.0, 
            every_n=8,            
            frame_dim=-1,         
            frames_per_row=4,     
            show=False
        )

        # 8. Log safely to Weights & Biases
        if trainer.logger and hasattr(trainer.logger.experiment, "log"):
            trainer.logger.experiment.log({
                "val/medical_reconstruction_diagnostics": wandb.Image(fig)
            })
            
        plt.close(fig)