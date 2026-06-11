import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from monai.visualize import matshow3d
from FLORA.model_stage2 import FLORA_stage2
from FLORA.model_CT import FLORA_CT
import os
from FLORA.image_loader import get_train_loader_stage2, get_train_loader_CT
from tqdm import tqdm
from monai.metrics import SSIMMetric, PSNRMetric

class Evaluator():
    def __init__(self, device,ode_steps=50,cfg_scale=2.0):
        self.device = device
        self.ode_steps = ode_steps
        
        # Initialize MONAI 3D Metrics on the evaluation device
        self.ssim_metric = SSIMMetric(data_range=2.5, spatial_dims=3)
        self.psnr_metric = PSNRMetric(max_val=2.5)
        self.cfg_scale = cfg_scale
        
    def compute_metrics(self, batch, model, vanilla_sd):
        """
        Evaluates a single batch using the model checkpoint and returns physical metrics.
        """
        # 1. Setup Data and move to designated evaluation device
        slab = batch["rsp"][0:1].to(self.device)
        physics_grid = batch["physics_grid"][0:1].to(self.device)
        medical_labels = batch["condition"][0:1].to(self.device)

        # Build condition label string for tracking/file names
        active_indices = torch.where(medical_labels[0] == 1)[0].tolist()
        cond_str = f"Conds_{active_indices}" if active_indices else "Healthy"

        with torch.no_grad():
            model.eval()

            # Find the true spatial target dimension of the VAE latent space
            z_mu, _ = model.vae.encode(slab)
            spatial_shape = z_mu.shape[2:]

            # --- Run Reconstruction Generation ---
            model.load_state_dict(vanilla_sd, strict=False)
            
            recon_vanilla = self.solve_ode_Heun(
                model, physics_grid, medical_labels, spatial_shape
            )
            
            # Compute MONAI Metrics (Calculates and appends to internal accumulation buffers)
            van_ssim = self.ssim_metric(recon_vanilla, slab).mean().item()
            van_psnr = self.psnr_metric(recon_vanilla, slab).mean().item()
            
            # CRITICAL: Reset accumulation buffers to prevent massive memory leaks during loops
            self.ssim_metric.reset()
            self.psnr_metric.reset()
        # 2. Process volumes to standard NumPy arrays for plotting/saving
        slab_cpu = slab.squeeze(0).cpu().numpy()
        vanilla_cpu = recon_vanilla.squeeze(0).cpu().numpy()
        return {"ssim": van_ssim,"psnr": van_psnr,"volumes": (slab_cpu, vanilla_cpu)}
    
    def solve_ode_Heun(self, pl_module, physics_grid, medical_labels, spatial_shape):
        """
        Runs the Heun (2nd-Order) integration loop to step from x0 to x1.
        Includes built-in Classifier-Free Guidance (CFG).
        """
        B = physics_grid.shape[0]
        device = physics_grid.device
        
        # 1. Process Conditioning (The Physics Grid)
        scaled_physics_grid = (physics_grid - pl_module.cond_mean) / pl_module.cond_std
        cond_features = pl_module.projection_mlp(scaled_physics_grid)
        
        if cond_features.ndim == 5:
            cond_features = cond_features.squeeze(1)
        cond_seq = cond_features.view(B, -1, cond_features.shape[-1])
        
        # Generate Unconditional Sequence for CFG (Pure Zeros)
        if self.cfg_scale > 1.0:
            uncond_seq = torch.zeros_like(cond_seq)
        
        # 2. Get the Starting Prior (Ghost + Noise)
        x_t = pl_module.generate_informed_x0(medical_labels, spatial_shape)
        
        # 3. Heun Integration Loop from t=0 to t=1
        dt = 1.0 / self.ode_steps
        
        for i in range(self.ode_steps):
            # Time tensors for current and next step
            t_current = i * dt
            t_next = (i + 1) * dt
            
            # MONAI expects 0-1000 scale
            t_current_scaled = torch.full((B,), t_current * 1000.0, device=device)
            t_next_scaled = torch.full((B,), t_next * 1000.0, device=device)
            
            # ==========================================
            # STEP 1: The Predictor (v1)
            # ==========================================
            if self.cfg_scale > 1.0:
                v1_cond = pl_module.velocity_net(x=x_t, timesteps=t_current_scaled, context=cond_seq)
                v1_uncond = pl_module.velocity_net(x=x_t, timesteps=t_current_scaled, context=uncond_seq)
                v1 = v1_uncond + self.cfg_scale * (v1_cond - v1_uncond)
            else:
                v1 = pl_module.velocity_net(x=x_t, timesteps=t_current_scaled, context=cond_seq)
            
            # Take a temporary Euler step to peek into the future
            x_euler = x_t + v1 * dt
            
            # If we are at the very last step, no need to correct. Just finish!
            if i == self.ode_steps - 1:
                x_t = x_euler
                break
                
            # ==========================================
            # STEP 2: The Corrector (v2)
            # ==========================================
            if self.cfg_scale > 1.0:
                v2_cond = pl_module.velocity_net(x=x_euler, timesteps=t_next_scaled, context=cond_seq)
                v2_uncond = pl_module.velocity_net(x=x_euler, timesteps=t_next_scaled, context=uncond_seq)
                v2 = v2_uncond + self.cfg_scale * (v2_cond - v2_uncond)
            else:
                v2 = pl_module.velocity_net(x=x_euler, timesteps=t_next_scaled, context=cond_seq)
                
            # ==========================================
            # STEP 3: The Heun Update
            # ==========================================
            x_t = x_t + (dt / 2.0) * (v1 + v2)
            
        # 4. Decode the final latent (x1) back into a 3D physical volume
        # IMPORTANT: Do not forget to divide by the latent scale before decoding!
        scale = getattr(pl_module, "latent_scale", 1.0)
        recon_volume = pl_module.vae.decode(x_t / scale)
        
        if isinstance(recon_volume, tuple):
            recon_volume = recon_volume[0]
            
        return recon_volume
    
def visualize_reconstruction(resdict,output_dir,cpkt_version,i,cfg_value=2.0):
    slab_cpu, vanilla_cpu = resdict["volumes"]
    combined = np.concatenate([slab_cpu, vanilla_cpu], axis=0)

    fig = plt.figure(figsize=(16, 8))

    matshow3d(
        volume=combined,
        fig=fig,
        title=f"FLORA Reconstruction\nTop: Ground Truth | Mid: Vanilla",
        vmin=0.0,
        vmax=1.0,
        every_n=8,
        frame_dim=-1,
        frames_per_row=4,
        show=False,
    )

    save_path = os.path.join(
        output_dir, f"flora_reconstruction_{i}_steps_CFG_{cfg_value}_diffODE_{cpkt_version}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)



import pandas as pd
HU_to_RSP = pd.read_csv('HU_to_RSP.csv')
def apply_clinical_hlut_old(ct_hu_tensor):
    """
    A simplified piecewise linear stoichiometric calibration curve (HLUT).
    Maps traditional CT Hounsfield Units (HU) to Relative Stopping Power (RSP).
    """
    hu_arr = ct_hu_tensor.cpu().numpy()
    bins = HU_to_RSP['HU'].values
    
    rsp_means = HU_to_RSP['RSP'].values.astype(np.float32)
    rsp_stds = HU_to_RSP['RSP_std'].values.astype(np.float32)
    
    # --- 1. Find Bins ---
    indices = np.digitize(hu_arr, bins) - 1
    indices[indices < 0] = 0
    # --- 2. Create Base Mean Map ---
    mean_map = rsp_means[indices]
    # --- 3. Create STD Map ---
    std_map = rsp_stds[indices]
    noise = np.random.normal(loc=0.0, scale=1.0, size=hu_arr.shape).astype(np.float32)
    final_rsp = mean_map
    return torch.from_numpy(final_rsp).to(ct_hu_tensor.device)



def apply_clinical_hlut(ct_normalized_tensor, a_min=-1000.0, a_max=3000.0):
    # Denormalize back to HU first
    ct_hu = ct_normalized_tensor * (a_max - a_min) + a_min
    hu_arr = ct_hu.cpu().numpy()
    
    bins = HU_to_RSP['HU'].values
    rsp_values = HU_to_RSP['RSP'].values.astype(np.float32)
    
    indices = np.digitize(hu_arr, bins) - 1
    indices = np.clip(indices, 0, len(rsp_values) - 1)
    rsp_map = rsp_values[indices]
    return torch.from_numpy(rsp_map).to(ct_normalized_tensor.device)

class FlowEvaluator():
    def __init__(self, device, ode_steps=50, rsp_max_scale=3.2):
        self.device = device
        self.ode_steps = ode_steps
        self.rsp_max_scale = rsp_max_scale 
        
        # Initialize MONAI 3D Metrics with explicit data ranges
        self.ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=3)
        self.psnr_metric = PSNRMetric(max_val=1.0)
        
    def generate_patient_mask(self, gt_volume, air_threshold=0.05, remove_bed_height=None):
        """
        Creates a binary mask to isolate the patient's body contour.
        Excludes background air and optionally truncates the CT bed at the bottom.
        """
        # Step 1: Threshold out the ambient air noise
        mask = (gt_volume > air_threshold).float()
        
        # Step 2: Optionally cut off the bottom rows if the bed is inflating errors
        # Expects shape [B, C, H, W, D]
        if remove_bed_height is not None:
            H = mask.shape[2]
            cut_idx = int(H * remove_bed_height)
            mask[:, :, cut_idx:, :, :] = 0.0
            
        return mask



    def compute_metrics(self, batch, model_baseline, model_detector, method="euler", use_masking=True):
        """
        Evaluates a single batch on BOTH models + Clinical HLUT inside the true patient contour or full volume.
        """
        # 1. Setup Data
        ct = batch["ct"][0:1].to(self.device)
        gt_rsp = batch["rsp"][0:1].to(self.device)
        physics_grid = batch["physics_grid"][0:1].to(self.device)

        with torch.no_grad():
            model_baseline.eval()
            model_detector.eval()

            # Generate the true patient mask from the ground truth volume conditionally
            if use_masking:
                # If the bed is visible at the bottom 15% of the vertical height, pass remove_bed_height=0.85
                patient_mask = self.generate_patient_mask(gt_rsp, air_threshold=0.0, remove_bed_height=None)
            else:
                # If masking is off, evaluate the entire 3D volume (all voxels are active)
                patient_mask = torch.ones_like(gt_rsp)

            # Choose solver dynamically
            solver_fn = self.solve_flow_Euler #if method == "euler"# else self.solve_flow_Heun

            # --- 2. Run Pure Conditioned Inference & Clinical Heuristic ---
            pred_baseline = solver_fn(model_baseline, ct, context_grid=None)
            pred_detector = solver_fn(model_detector, ct, context_grid=physics_grid)
            
            # Compute external clinical HLUT curve baseline
            # Note: Ensure apply_clinical_hlut matches the normalized output range [0, 1] of your network
            pred_hlut = apply_clinical_hlut(ct)
            
            # --- 3. Apply Patient Mask to Predictions & Ground Truth ---
            masked_gt = gt_rsp * patient_mask  
            masked_base = pred_baseline * patient_mask  
            masked_det = pred_detector * patient_mask  
            masked_hlut = pred_hlut * patient_mask
            hlut_normalized = torch.clamp(masked_hlut / self.rsp_max_scale, 0.0, 1.0) #/ self.rsp_max_scale
            # --- 4. Compute Structural Metrics Inside/Outside Mask ---
            base_ssim = self.ssim_metric(masked_base, masked_gt).mean().item()
            det_ssim = self.ssim_metric(masked_det, masked_gt).mean().item()
            hlut_ssim = self.ssim_metric(hlut_normalized, masked_gt).mean().item()
            self.ssim_metric.reset()
            
            base_psnr = self.psnr_metric(masked_base, masked_gt).mean().item()
            det_psnr = self.psnr_metric(masked_det, masked_gt).mean().item()
            hlut_psnr = self.psnr_metric(hlut_normalized, masked_gt).mean().item()
            self.psnr_metric.reset()
            
            # --- 5. Compute Physical MAE (Unscaled [0, 3.2] RSP) ---
            physical_gt = masked_gt * self.rsp_max_scale
            physical_base = masked_base * self.rsp_max_scale
            physical_det = masked_det * self.rsp_max_scale
            physical_hlut = masked_hlut #* self.rsp_max_scale

            #base_ssim = self.ssim_metric(physical_base, physical_gt).mean().item()
            #det_ssim = self.ssim_metric(physical_det, physical_gt).mean().item()
            #self.ssim_metric.reset()
            #
            #base_psnr = self.psnr_metric(physical_base, physical_gt).mean().item()
            #det_psnr = self.psnr_metric(physical_det, physical_gt).mean().item()
            #self.psnr_metric.reset()
            
            # Calculate mean only where the mask is active
            num_tissue_voxels = patient_mask.sum()
            if num_tissue_voxels > 0:
                base_mae = (torch.abs(physical_base - physical_gt).sum() / num_tissue_voxels).item()
                det_mae = (torch.abs(physical_det - physical_gt).sum() / num_tissue_voxels).item()
                hlut_mae = (torch.abs(physical_hlut - physical_gt).sum() / num_tissue_voxels).item()
            else:
                base_mae, det_mae, hlut_mae = 0.0, 0.0, 0.0

        # 6. Process volumes to standard NumPy arrays for plotting
        return {
            "metrics": {
                "HLUT_SSIM": hlut_ssim,
                "Baseline_SSIM": base_ssim, 
                "Detector_SSIM": det_ssim,
                
                "HLUT_PSNR": hlut_psnr,
                "Baseline_PSNR": base_psnr, 
                "Detector_PSNR": det_psnr,
                
                "HLUT_MAE_RSP": hlut_mae,
                "Baseline_MAE_RSP": base_mae, 
                "Detector_MAE_RSP": det_mae
            },
            "volumes": (
                ct.squeeze(0).cpu().numpy(), 
                gt_rsp.squeeze(0).cpu().numpy(), 
                pred_baseline.squeeze(0).cpu().numpy(), 
                pred_detector.squeeze(0).cpu().numpy(),
            )
        }

    def solve_flow_Euler(self, pl_module, ct_images, context_grid=None):
        B = ct_images.shape[0]
        device = ct_images.device
        
        z_0, _ = pl_module.vae.encode(ct_images)
        z_t = z_0 * pl_module.latent_scale
        
        # Cleaned context pipeline: Pure conditioning, no unconditional states
        if context_grid is not None and pl_module.use_detector_context:
            context_grid = (context_grid - pl_module.cond_mean) / pl_module.cond_std
            cond_features = pl_module.projection_mlp(context_grid)
            cond_seq = cond_features.view(B, -1, cond_features.shape[-1])
        else:
            cond_seq = None
            
        dt = 1.0 / self.ode_steps
        for i in range(self.ode_steps):
            t_curr_tensor = torch.full((B,), i * dt, device=device)
            
            # Straight inference pass
            v = pl_module.velocity_net(x=z_t, timesteps=t_curr_tensor, context=cond_seq)
            z_t = z_t + v * dt
            
        recon_volume = pl_module.vae.decode(z_t / pl_module.latent_scale)
        if isinstance(recon_volume, tuple):
            recon_volume = recon_volume[0]
        return recon_volume
    

def visualize_ablation(resdict, output_dir, patient_idx):
    ct_cpu, gt_cpu, base_cpu, det_cpu = resdict["volumes"]
    metrics = resdict["metrics"]
    
    # Extract the MAE values to display directly on the plot title
    base_mae = metrics["Baseline_MAE_RSP"]
    det_mae = metrics["Detector_MAE_RSP"]
    
    # Stack them: CT | GT RSP | Baseline Prediction | Detector Prediction
    diff_map = np.abs(base_cpu - gt_cpu)
    diff_map_det = np.abs(det_cpu - gt_cpu)
    combined = np.concatenate([gt_cpu, base_cpu, det_cpu], axis=0)

    fig = plt.figure(figsize=(16, 16))

    matshow3d(
        volume=combined,
        fig=fig,
        title=(f"Patient {patient_idx} |\n"
            f"Row 1: Ground Truth RSP\n"
            f"Row 2: Baseline (CT Only) |"
            f"Row 3: Proposed (CT + Detector)"
            ),
        vmin=0.0,
        vmax=1.0,
        every_n=8,
        frame_dim=-1,
        frames_per_row=4,
        show=False,
    )

    save_path = os.path.join(output_dir, f"flora_ablation_patient_{patient_idx}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    fig2 = plt.figure(figsize=(12, 6))
    matshow3d(
        volume=np.concatenate([gt_cpu, diff_map, diff_map_det], axis=0),
        fig=fig2,
        title=(f"Patient {patient_idx} |\n"
            f"Row 1: Ground Truth RSP\n"
            f"Row 2: Absolute Error of Baseline\n"
            f"Row 3: Absolute Error of Detector"
            ),
        vmin=0.0,
        vmax=1.,
        every_n=8,
        frame_dim=-1,
        frames_per_row=4,
        show=False,
    )
    save_path = os.path.join(output_dir, f"flora_ablation_patient_{patient_idx}_error_maps.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig2) 



import pickle
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running standalone evaluation on device: {device}")

    cpkt_version  = "last-v1"
    checkpint_dir = "checkpoints/PHASE_2_CFG"
    checkpoint_file = f"{checkpint_dir}/{cpkt_version}.ckpt"
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    vanilla_state_dict = checkpoint["state_dict"]

    _, val_loader = get_train_loader_stage2(batch_size=1, num_workers=0)
    model = FLORA_stage2().to(device)
    model.eval()

    evaluator = Evaluator(device=device, ode_steps=50)

    pbar = tqdm(val_loader,desc ="Validation",total=len(val_loader))
    output_dir = "s2_outs/eval"
    cfg_value = 1.0
    metrics_dict = {"ssim": [], "psnr": []}
    for i, batch in enumerate(pbar):
        resDict = evaluator.compute_metrics(batch,model,vanilla_state_dict)
        save_interval = max(1, len(val_loader) // 10)
        if i % save_interval == 0:
            visualize_reconstruction(resDict, output_dir, cpkt_version, i, cfg_value)
        metrics_dict["ssim"].append(resDict["ssim"])
        metrics_dict["psnr"].append(resDict["psnr"])
        pbar.set_postfix({"Avg SSIM": f"{np.mean(metrics_dict['ssim']):.4f}", "Avg PSNR": f"{np.mean(metrics_dict['psnr']):.2f} dB"})
    # Save final metrics to disk
    with open(os.path.join(checkpint_dir, f"evaluation_metrics_cfg{cfg_value}_{cpkt_version}.pkl"), "wb") as f:
        pickle.dump(metrics_dict, f)

from omegaconf import OmegaConf
def main_CT():
    cfg = OmegaConf.load("config.yaml")
    vaekwgs = cfg.vae_kwgs
    velocity_kwargs = cfg.velocity_kwargs
    velocity_kwargs.in_channels = 4
    velocity_kwargs.with_conditioning = True
    velocity_kwargs.cross_attention_dim = 256
    ct_cfg = cfg.ct_train_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running standalone Ablation Evaluation on device: {device}")

    # --- 1. Configuration & Paths ---
    output_dir = "s2_outs/eval_ablation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Update these paths to where your actual .ckpt files are!
    ckpt_baseline_path = "checkpoints/PHASE_2_CT_ONLY/last-v1.ckpt" 
    ckpt_detector_path = "checkpoints/PHASE_2_CT_DETECTOR/last-v1.ckpt"

    ode_steps = 10
    cfg_value = 0.0 # Adjusted to standard 1.5, feel free to change
    method = "EULER"

    # --- 2. Load Models ---
    print("Loading Baseline Model...")
    
    v1 = velocity_kwargs.copy()
    v1["with_conditioning"] = False
    v1.pop('cross_attention_dim', None)  # Remove if exists since baseline doesn't use it

    model_baseline = FLORA_CT(vaekwgs=vaekwgs, velocity_kwargs=v1,lr=ct_cfg.lr,use_detector_context=False).to(device)
    sd_base = torch.load(ckpt_baseline_path, map_location="cpu")["state_dict"]
    model_baseline.load_state_dict(sd_base, strict=False)
    
    print("Loading Detector Model...")

    model_detector = FLORA_CT(vaekwgs=vaekwgs, velocity_kwargs=velocity_kwargs,lr=ct_cfg.lr,use_detector_context=True).to(device)
    sd_det = torch.load(ckpt_detector_path, map_location="cpu")["state_dict"]
    model_detector.load_state_dict(sd_det, strict=False)

    # --- 3. Dataloader & Evaluator ---
    # Ensure num_workers=4 or 8 depending on your interactive session, and batch_size=1
    _, val_loader = get_train_loader_CT(batch_size=1, num_workers=4)
    
    evaluator = FlowEvaluator(device=device, ode_steps=ode_steps, rsp_max_scale=3.2)

    # --- 4. Metrics Tracking Dictionary ---
    history = {
        "Baseline_SSIM": [], "Detector_SSIM": [],
        "Baseline_PSNR": [], "Detector_PSNR": [],
        "Baseline_MAE_RSP": [], "Detector_MAE_RSP": [],
        "HLUT_SSIM": [], "HLUT_PSNR": [], "HLUT_MAE_RSP": []
    }

    # --- 5. Evaluation Loop ---
    pbar = tqdm(val_loader, desc=f"Val | {method.upper()} | {ode_steps} steps", total=len(val_loader))
    save_interval = max(1, len(val_loader) // 10)

    for i, batch in enumerate(pbar):
        # Compute everything
        resDict = evaluator.compute_metrics(batch, model_baseline, model_detector, method=method,use_masking=False)
        
        # Save snapshot
        if i % save_interval == 0:
            visualize_ablation(resDict, output_dir, patient_idx=i)
            
        # Accumulate metrics
        for k in history.keys():
            history[k].append(resDict["metrics"][k])
            
        # Live dashboard update (Focusing on the physical MAE error!)
        pbar.set_postfix({
            "Base_SSIM": f"{np.mean(history['Baseline_SSIM']):.4f}",
            "Det_SSIM": f"{np.mean(history['Detector_SSIM']):.4f}",
            "CLINICAL_SSIM": f"{np.mean(history['HLUT_SSIM']):.4f}",
            "BASE_MAE": f"{np.mean(history['Baseline_MAE_RSP']):.4f}",
            "CLINICAL_MAE": f"{np.mean(history['HLUT_MAE_RSP']):.4f}"
        })

    # --- 6. Save Final Metrics to Disk ---
    pickle_name = f"ablation_metrics_cfg{cfg_value}_{method}_{ode_steps}steps.pkl"
    with open(os.path.join(output_dir, pickle_name), "wb") as f:
        pickle.dump(history, f)
        
    print(f"\nEvaluation complete! Final metrics saved to {pickle_name}")
    print(f"Final Average Baseline MAE: {np.mean(history['Baseline_MAE_RSP']):.4f}")
    print(f"Final Average Detector MAE: {np.mean(history['Detector_MAE_RSP']):.4f}")
    print(f"Final Average Clinical HLUT MAE: {np.mean(history['HLUT_MAE_RSP']):.4f}")
    print("\n\n")
    print("Final Average Baseline SSIM: {:.4f}".format(np.mean(history['Baseline_SSIM'])))
    print("Final Average Detector SSIM: {:.4f}".format(np.mean(history['Detector_SSIM'])))
    print("Final Average Clinical HLUT SSIM: {:.4f}".format(np.mean(history['HLUT_SSIM'])))
    print("\n\n")
    print("Final Average Baseline PSNR: {:.2f} dB".format(np.mean(history['Baseline_PSNR'])))
    print("Final Average Detector PSNR: {:.2f} dB".format(np.mean(history['Detector_PSNR'])))
    print("Final Average Clinical HLUT PSNR: {:.2f} dB".format(np.mean(history['HLUT_PSNR'])))

    

if __name__ == "__main__":
    print('Starting Evaluation Script...')
    main_CT()

    