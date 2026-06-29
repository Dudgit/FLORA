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
        
        self.ssim_metric = SSIMMetric(data_range=2.5, spatial_dims=3)
        self.psnr_metric = PSNRMetric(max_val=2.5)
        self.cfg_scale = cfg_scale
        
    def compute_metrics(self, batch, model, vanilla_sd):
        """
        Evaluates a single batch using the model checkpoint and returns physical metrics.
        """
        slab = batch["rsp"][0:1].to(self.device)
        physics_grid = batch["physics_grid"][0:1].to(self.device)
        medical_labels = batch["condition"][0:1].to(self.device)

        active_indices = torch.where(medical_labels[0] == 1)[0].tolist()
        cond_str = f"Conds_{active_indices}" if active_indices else "Healthy"

        with torch.no_grad():
            model.eval()

            z_mu, _ = model.vae.encode(slab)
            spatial_shape = z_mu.shape[2:]

            model.load_state_dict(vanilla_sd, strict=False)
            
            recon_vanilla = self.solve_ode_Heun(
                model, physics_grid, medical_labels, spatial_shape
            )
            
            van_ssim = self.ssim_metric(recon_vanilla, slab).mean().item()
            van_psnr = self.psnr_metric(recon_vanilla, slab).mean().item()
            
            self.ssim_metric.reset()
            self.psnr_metric.reset()
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
        
        scaled_physics_grid = (physics_grid - pl_module.cond_mean) / pl_module.cond_std
        cond_features = pl_module.projection_mlp(scaled_physics_grid)
        
        if cond_features.ndim == 5:
            cond_features = cond_features.squeeze(1)
        cond_seq = cond_features.view(B, -1, cond_features.shape[-1])
        
        if self.cfg_scale > 1.0:
            uncond_seq = torch.zeros_like(cond_seq)
        
        x_t = pl_module.generate_informed_x0(medical_labels, spatial_shape)
        
        dt = 1.0 / self.ode_steps
        
        for i in range(self.ode_steps):

            t_current = i * dt
            t_next = (i + 1) * dt
            

            t_current_scaled = torch.full((B,), t_current * 1000.0, device=device)
            t_next_scaled = torch.full((B,), t_next * 1000.0, device=device)
            

            if self.cfg_scale > 1.0:
                v1_cond = pl_module.velocity_net(x=x_t, timesteps=t_current_scaled, context=cond_seq)
                v1_uncond = pl_module.velocity_net(x=x_t, timesteps=t_current_scaled, context=uncond_seq)
                v1 = v1_uncond + self.cfg_scale * (v1_cond - v1_uncond)
            else:
                v1 = pl_module.velocity_net(x=x_t, timesteps=t_current_scaled, context=cond_seq)
            
            x_euler = x_t + v1 * dt
            
            if i == self.ode_steps - 1:
                x_t = x_euler
                break
                

            if self.cfg_scale > 1.0:
                v2_cond = pl_module.velocity_net(x=x_euler, timesteps=t_next_scaled, context=cond_seq)
                v2_uncond = pl_module.velocity_net(x=x_euler, timesteps=t_next_scaled, context=uncond_seq)
                v2 = v2_uncond + self.cfg_scale * (v2_cond - v2_uncond)
            else:
                v2 = pl_module.velocity_net(x=x_euler, timesteps=t_next_scaled, context=cond_seq)
                

            x_t = x_t + (dt / 2.0) * (v1 + v2)
            

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
    
    indices = np.digitize(hu_arr, bins) - 1
    indices[indices < 0] = 0
    mean_map = rsp_means[indices]
    std_map = rsp_stds[indices]
    noise = np.random.normal(loc=0.0, scale=1.0, size=hu_arr.shape).astype(np.float32)
    final_rsp = mean_map
    return torch.from_numpy(final_rsp).to(ct_hu_tensor.device)



def apply_clinical_hlut(ct_normalized_tensor, a_min=-1000.0, a_max=3000.0):
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
        
        self.ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=3)
        self.psnr_metric = PSNRMetric(max_val=1.0)
        
    def generate_patient_mask(self, gt_volume, air_threshold=0.05, remove_bed_height=None):
        """
        Creates a binary mask to isolate the patient's body contour.
        Excludes background air and optionally truncates the CT bed at the bottom.
        """
        mask = (gt_volume > air_threshold).float()

        if remove_bed_height is not None:
            H = mask.shape[2]
            cut_idx = int(H * remove_bed_height)
            mask[:, :, cut_idx:, :, :] = 0.0
            
        return mask



    def compute_metrics(self, batch, model_baseline, model_detector, method="euler", use_masking=True):
        """
        Evaluates a single batch on BOTH models + Clinical HLUT inside the true patient contour or full volume.
        """
        ct = batch["ct"][0:1].to(self.device)
        gt_rsp = batch["rsp"][0:1].to(self.device)
        physics_grid = batch["physics_grid"][0:1].to(self.device)

        with torch.no_grad():
            model_baseline.eval()
            model_detector.eval()

        
            solver_fn = self.solve_flow_Euler 

            pred_baseline = solver_fn(model_baseline, ct, context_grid=None)
            pred_detector = solver_fn(model_detector, ct, context_grid=physics_grid)
            
            pred_hlut = apply_clinical_hlut(ct)
            physical_gt = gt_rsp * self.rsp_max_scale
            patient_mask = ((physical_gt > 1.038 + 0.003)).float()  # High density
            
            masked_gt = gt_rsp * patient_mask  
            masked_base = pred_baseline * patient_mask  
            masked_det = pred_detector * patient_mask  
            masked_hlut = pred_hlut * patient_mask
            hlut_normalized = torch.clamp(masked_hlut / self.rsp_max_scale, 0.0, 1.0) 

            base_ssim = self.ssim_metric(masked_base, masked_gt).mean().item()
            det_ssim = self.ssim_metric(masked_det, masked_gt).mean().item()
            hlut_ssim = self.ssim_metric(hlut_normalized, masked_gt).mean().item()
            self.ssim_metric.reset()
            
            base_psnr = self.psnr_metric(masked_base, masked_gt).mean().item()
            det_psnr = self.psnr_metric(masked_det, masked_gt).mean().item()
            hlut_psnr = self.psnr_metric(hlut_normalized, masked_gt).mean().item()
            self.psnr_metric.reset()
            
            physical_gt = masked_gt * self.rsp_max_scale
            physical_base = masked_base * self.rsp_max_scale
            physical_det = masked_det * self.rsp_max_scale
            physical_hlut = masked_hlut 

            num_tissue_voxels = patient_mask.sum()
            if num_tissue_voxels > 0:
                base_mae = (torch.abs(physical_base - physical_gt).sum() / num_tissue_voxels).item()
                det_mae = (torch.abs(physical_det - physical_gt).sum() / num_tissue_voxels).item()
                hlut_mae = (torch.abs(physical_hlut - physical_gt).sum() / num_tissue_voxels).item()
            else:
                base_mae, det_mae, hlut_mae = 0.0, 0.0, 0.0

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
                "Detector_MAE_RSP": det_mae,
    
            },
            "volumes": (
                ct.squeeze(0).cpu().numpy(), 
                physical_gt.squeeze(0).cpu().numpy(), 
                physical_base.squeeze(0).cpu().numpy(), 
                physical_hlut.squeeze(0).cpu().numpy(),
            )
        }

    def solve_flow_Euler(self, pl_module, ct_images, context_grid=None):
        B = ct_images.shape[0]
        device = ct_images.device
        
        z_0, _ = pl_module.vae.encode(ct_images)
        z_t = z_0 * pl_module.latent_scale
        
        if context_grid is not None and pl_module.use_detector_context:
            context_grid = (context_grid - pl_module.cond_mean) / pl_module.cond_std
            cond_features = pl_module.projection_mlp(context_grid)
            cond_seq = cond_features.view(B, -1, cond_features.shape[-1])
        else:
            cond_seq = None
            
        dt = 1.0 / self.ode_steps
        for i in range(self.ode_steps):
            t_curr_tensor = torch.full((B,), i * dt, device=device)
            
            v = pl_module.velocity_net(x=z_t, timesteps=t_curr_tensor, context=cond_seq)
            z_t = z_t + v * dt
            
        recon_volume = pl_module.vae.decode(z_t / pl_module.latent_scale)
        if isinstance(recon_volume, tuple):
            recon_volume = recon_volume[0]
        return recon_volume
    


class VAEEvaluator:
    """
    Evaluates VAE reconstruction quality on the validation set.
 
    Answers the question: how much quality does the VAE compression lose?
    All metrics are computed in physical RSP space [0, 3.2] for MAE/MSE,
    and normalized [0, 1] space for SSIM/PSNR — consistent with FlowEvaluator.
 
    Metrics:
        MAE   — mean absolute error in RSP units (clinically interpretable)
        MSE   — mean squared error in RSP units  (penalizes large errors more)
        PSNR  — peak signal-to-noise ratio        (standard image quality)
        SSIM  — structural similarity             (perceptual quality)
    """
 
    def __init__(self, device, rsp_max_scale=3.2):
        self.device         = device
        self.rsp_max_scale  = rsp_max_scale
 
        self.ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=3)
        self.psnr_metric = PSNRMetric(max_val=1.0)
 
    @torch.no_grad()
    def evaluate_batch(self, batch, vae, latent_scale):
        """
        Encode and decode one pCT volume, return per-sample metrics.
 
        Args:
            batch:         dict with key "rsp" → (1, 1, H, W, D) in [0, 1]
            vae:           AutoencoderKL (frozen)
            latent_scale:  scalar tensor (e.g. 11.1560)
 
        Returns:
            dict of scalar metric values for this batch
        """
        gt = batch["rsp"][0:1].to(self.device)   # (1, 1, H, W, D) in [0, 1]
        vae = vae.to(self.device)
        z, _    = vae.encode(gt)
        z       = z * latent_scale
        recon   = vae.decode(z / latent_scale)
        if isinstance(recon, tuple):
            recon = recon[0]
 
        recon = torch.clamp(recon, 0.0, 1.0)
 
        ssim = self.ssim_metric(recon, gt).mean().item()
        self.ssim_metric.reset()
 
        psnr = self.psnr_metric(recon, gt).mean().item()
        self.psnr_metric.reset()
 
        gt_rsp    = gt    * self.rsp_max_scale
        recon_rsp = recon * self.rsp_max_scale
 
        mae = torch.abs(recon_rsp - gt_rsp).mean().item()
        mse = torch.pow(recon_rsp - gt_rsp, 2).mean().item()
 
        return {"SSIM": ssim, "PSNR": psnr, "MAE_RSP": mae, "MSE_RSP": mse}
 
    def run(self, val_loader, vae, latent_scale, output_dir, save_interval=None):
        """
        Full evaluation loop over the validation set.
 
        Args:
            val_loader:    DataLoader yielding batches with "rsp" key
            vae:           AutoencoderKL — will be set to eval()
            latent_scale:  model.latent_scale buffer
            output_dir:    where to save metrics pickle and visualizations
            save_interval: how often to save a reconstruction image
                           (defaults to ~10 times across the dataset)
        """
        os.makedirs(output_dir, exist_ok=True)
        vae.eval()
 
        if save_interval is None:
            save_interval = max(1, len(val_loader) // 10)
 
        history = {"SSIM": [], "PSNR": [], "MAE_RSP": [], "MSE_RSP": []}
 
        pbar = tqdm(val_loader, desc="VAE Evaluation", total=len(val_loader))
        for i, batch in enumerate(pbar):
            metrics = self.evaluate_batch(batch, vae, latent_scale)
 
            for k, v in metrics.items():
                history[k].append(v)
 

 
            pbar.set_postfix({
                "SSIM":    f"{np.mean(history['SSIM']):.4f}",
                "PSNR":    f"{np.mean(history['PSNR']):.2f}",
                "MAE_RSP": f"{np.mean(history['MAE_RSP']):.4f}",
            })
 
        self._print_summary(history)
        self._save_metrics(history, output_dir)
        return history
    

    @staticmethod
    def _print_summary(history):
        print("\n" + "=" * 50)
        print("VAE RECONSTRUCTION QUALITY SUMMARY")
        print("=" * 50)
        for k, vals in history.items():
            unit = " (RSP)" if "RSP" in k else ""
            print(f"  {k+unit:<20}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print("=" * 50)
 
    @staticmethod
    def _save_metrics(history, output_dir):
        path = os.path.join(output_dir, "vae_evaluation_metrics.pkl")
        with open(path, "wb") as f:
            pickle.dump(history, f)
        print(f"Metrics saved to {path}")
    

def visualize_ablation(resdict, output_dir, patient_idx):
    ct_cpu, gt_cpu, base_cpu, det_cpu = resdict["volumes"]
    metrics = resdict["metrics"]
    
    base_mae = metrics["Baseline_MAE_RSP"]
    det_mae = metrics["Detector_MAE_RSP"]
    
    diff_map = np.abs(base_cpu - gt_cpu)
    diff_map_det = np.abs(det_cpu - gt_cpu)
    combined = np.concatenate([gt_cpu, base_cpu, det_cpu], axis=0)

    fig = plt.figure(figsize=(16, 8))

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
        every_n=16,
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
        every_n=16,
        frame_dim=-1,
        frames_per_row=4,
        show=False,
    )
    save_path = os.path.join(output_dir, f"flora_ablation_patient_{patient_idx}_error_maps.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig2) 

def vis_ablataion_2(resdict,patient_idx,output_dir="article_vis", rsp_max_scale=3.2):
    _, gt_cpu, base_cpu, hlut = resdict["volumes"]
    
    gt_cpu = gt_cpu/rsp_max_scale
    base_cpu = base_cpu/rsp_max_scale
    hlut = hlut/rsp_max_scale

    diff_map = np.abs(base_cpu - gt_cpu)
    diff_map_clinical = np.abs(hlut - gt_cpu)
    combined = np.concatenate([gt_cpu, base_cpu], axis=0)

    fig = plt.figure(figsize=(16, 12))
    matshow3d(
        volume=combined,
        fig=fig,
        title=None,
        vmin=0.0,
        vmax=1.0,
        every_n=16,
        frame_dim=-1,
        frames_per_row=4,
        show=False,cmap="grey"
    )
    fig.suptitle(
        f"Patient {patient_idx}\n"
        f"Row 1: Ground Truth RSP | "
        f"Row 2: Predicted Baseline",
        fontsize=16,fontweight='bold',y=0.80)
    #fig.subplots_adjust(top=0.85, bottom=0.05, left=0.02, right=0.98, hspace=0.05, wspace=0.05)

    save_path = os.path.join(output_dir, f"flora_ablation_patient_{patient_idx}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig2 = plt.figure(figsize=(12, 6))
    combined_error = np.concatenate([diff_map], axis=0)
    matshow3d(
        volume=combined_error,
        fig=fig2,
        title=None, #(f"Patient {patient_idx}\n"f"Normalized Absolute Error"),
        every_n=4,
        frame_dim=-1,
        frames_per_row=4,
        show=False,cmap="magma",
    )
    fig2.suptitle(
        f"Patient {patient_idx}\n"
        f"Normalized Absolute Error Maps",
        fontsize=16,fontweight='bold',y=0.95)
    
    colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap="magma"), ax=fig2.axes, orientation='vertical')
    colorbar.set_label('Normalized Absolute Error (RSP)', rotation=270, labelpad=15)
    save_path = os.path.join(output_dir, f"flora_ablation_patient_{patient_idx}_error_maps.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig2)




import os
import matplotlib.pyplot as plt



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


def vis_for_paper():
    cfg = OmegaConf.load("config.yaml")
    vaekwgs = cfg.vae_kwgs
    velocity_kwargs = cfg.velocity_kwargs
    velocity_kwargs.in_channels = 4
    velocity_kwargs.with_conditioning = True
    velocity_kwargs.cross_attention_dim = 256
    ct_cfg = cfg.ct_train_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running standalone Ablation Evaluation on device: {device}")

    output_dir = "s2_outs/eval_ablation"
    os.makedirs(output_dir, exist_ok=True)
    
    ckpt_baseline_path = "checkpoints/PHASE_2_CT_ONLY/last-v1.ckpt" 
    ckpt_detector_path = "checkpoints/PHASE_2_CT_DETECTOR/last-v1.ckpt"

    ode_steps = 10
    method = "EULER"

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

    _, val_loader = get_train_loader_CT(batch_size=1, num_workers=4)
    pbar = tqdm(val_loader, desc=f"Val | {method.upper()} | {ode_steps} steps", total=len(val_loader))
    save_interval = max(1, len(val_loader) // 10)
    evaluator = FlowEvaluator(device=device, ode_steps=ode_steps, rsp_max_scale=3.2)

    for i, batch in enumerate(pbar):        
        if i % save_interval == 0:
            resDict = evaluator.compute_metrics(batch, model_baseline, model_detector, method=method,use_masking=False)
            vis_ablataion_2(resDict, patient_idx=i)



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

    output_dir = "s2_outs/eval_ablation"
    os.makedirs(output_dir, exist_ok=True)
    
    ckpt_baseline_path = "checkpoints/PHASE_2_CT_ONLY/last-v1.ckpt" 
    ckpt_detector_path = "checkpoints/PHASE_2_CT_DETECTOR/last-v1.ckpt"

    ode_steps = 10
    cfg_value = 0.0 
    method = "EULER"

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

    
    _, val_loader = get_train_loader_CT(batch_size=1, num_workers=4)
    
    evaluator = FlowEvaluator(device=device, ode_steps=ode_steps, rsp_max_scale=3.2)

    history = {
        "Baseline_SSIM": [], "Detector_SSIM": [],
        "Baseline_PSNR": [], "Detector_PSNR": [],
        "Baseline_MAE_RSP": [], "Detector_MAE_RSP": [],
        "HLUT_SSIM": [], "HLUT_PSNR": [], "HLUT_MAE_RSP": []
    }

    pbar = tqdm(val_loader, desc=f"Val | {method.upper()} | {ode_steps} steps", total=len(val_loader))
    save_interval = max(1, len(val_loader) // 10)
    patient_index = 0
    for i, batch in enumerate(pbar):
        resDict = evaluator.compute_metrics(batch, model_baseline, model_detector, method=method,use_masking=True)
        
        if i % save_interval == 0:
            patient_index += 1
            visualize_ablation(resDict, output_dir, patient_idx=patient_index)
            
        for k in history.keys():
            history[k].append(resDict["metrics"][k])
            
        pbar.set_postfix({
            "Avg Baseline MAE": f"{np.mean(history['Baseline_MAE_RSP']):.4f}",
            "Avg HLUT MAE": f"{np.mean(history['HLUT_MAE_RSP']):.4f}"
        })

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
    print("\n\n")

from FLORA.blocks import configure_vae

def main_vae():
    cfg = OmegaConf.load("config.yaml")
    vaekwgs = cfg.vae_kwgs
    vae = configure_vae(vaekwgs)
    _, val_loader = get_train_loader_stage2(batch_size=1, num_workers=1)
    evaluator = VAEEvaluator(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), rsp_max_scale=3.2)
    history = evaluator.run(val_loader,vae,latent_scale=3.2,output_dir="s2_outs/vae_eval",save_interval=None)

if __name__ == "__main__":
    print('Starting Evaluation Script...')
    main_CT()

    