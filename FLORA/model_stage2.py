import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.losses import PerceptualLoss
from monai.losses import PatchAdversarialLoss
import torch
import torch.nn as nn
import contextlib
import torch.nn.functional as F
import os
from FLORA.blocks import ProjectionMLP
default_vae_kwargs = {
    "spatial_dims": 3,      # <--- Make sure this is 3! (Not 4)
    "in_channels": 1,
    "out_channels": 1,
    "num_res_blocks": 2,
    "latent_channels": 4,   # <--- This is the one that should be 4
    "attention_levels": (False, False, False, True),
    "norm_num_groups": 32,
}


default_disc_kwargs = {
    "spatial_dims": 3,
    "num_layers_d": 3,
    "channels": 64,
    "in_channels": 1,
    "out_channels": 1}



    
class FLORA_stage2(pl.LightningModule):
    def __init__(self,latent_channels = 4,vaekwgs = default_vae_kwargs,prior_dir="priors/",lr=1e-4):
        super().__init__()

        self.projection_mlp = ProjectionMLP()
        self.vae = self.vae = AutoencoderKL(**vaekwgs)
        checkpoint = torch.load("checkpoints/Medical_condition/PHASE_1/last.ckpt", map_location="cpu")
        vae_state_dict = {
            k.replace("vae.", ""): v 
            for k, v in checkpoint["state_dict"].items() 
            if k.startswith("vae.")}
        self.vae.load_state_dict(vae_state_dict)
        self.vae.eval()
        self.lr = lr
        for param in self.vae.parameters():
            param.requires_grad = False
        self.velocity_net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=latent_channels,
            out_channels=latent_channels, 
            channels=(64, 128, 256),
            attention_levels=(False, True, True),
            num_res_blocks=2,
            num_head_channels=32,
            with_conditioning=True,
            cross_attention_dim=256  # Must match ProjectionMLP embed_dim
        )
        mu_path = os.path.join(prior_dir, "condition_mu.pt")
        sigma_path = os.path.join(prior_dir, "condition_sigma.pt")
        loaded_mu = torch.load(mu_path)
        loaded_sigma = torch.load(sigma_path)
        
        # Register them as buffers. They are now accessible anywhere in the 
        # class via self.condition_mu and self.condition_sigma
        self.register_buffer("condition_mu", loaded_mu)
        self.register_buffer("condition_sigma", loaded_sigma)
        avg_std = loaded_sigma.mean().clamp(min=1e-5)
        # TODO: Save it into a file
        mean_tensor = torch.tensor([0.008381759747862816, 0.020576655864715576, 0.03292844071984291, 0.0516696535050869, 0.07965941727161407, 0.14866508543491364, 0.455157071352005, 167.81105041503906, 174.27532958984375, 177.35848999023438, 180.73336791992188, 184.35720825195312, 187.68963623046875, 226.7608642578125, 0.04672469571232796, 6.969912528991699, 0.07428985834121704, 0.002276827348396182, 3.0780694484710693, 6.565859317779541, 0.7064194679260254])
        std_tensor = torch.tensor([0.020250756293535233, 0.021754246205091476, 0.025645112618803978, 0.03343982622027397, 0.04642146825790405, 0.06932703405618668, 0.13534820079803467, 36.7294807434082, 40.928672790527344, 42.93149948120117, 45.0711555480957, 47.05894470214844, 48.72496795654297, 49.305912017822266, 0.02668251283466816, 5.117215633392334, 0.13192923367023468, 0.28002825379371643, 0.46335211396217346, 0.7556602954864502, 0.4064233601093292])
        self.register_buffer("cond_mean", mean_tensor)
        self.register_buffer("cond_std", std_tensor)
        self.register_buffer("latent_scale",torch.tensor(11.1560))
        #self.register_buffer("scale_factor",1.0/avg_std)


    def flow_matching_loss(self,x1,condition_emb, labels):
        batch_size = x1.shape[0]
        spatial_shape = x1.shape[2:]
        x0 = self.generate_informed_x0(labels, spatial_shape)

        m = torch.distributions.beta.Beta(torch.tensor([5.0]), torch.tensor([1.0]))
        t = m.sample((batch_size,)).squeeze(-1).to(self.device)
        t_expand = t.view(-1, 1, 1, 1, 1)

        xt = t_expand * x1 + (1.0 - t_expand) * x0
        ut = x1 - x0

        t_scaled = t * 1000.0
        vt = self.velocity_net(x=xt, timesteps=t_scaled, context=condition_emb)
        main_loss = F.mse_loss(vt, ut, )
        x1_pred = xt + (1.0 - t_expand) * vt
        with torch.no_grad():
            vt_det = vt.detach()
            ut_det = ut.detach()
    
            raw_loss = F.mse_loss(vt_det, ut_det, reduction='none').mean(dim=[1, 2, 3, 4])
            t_flat = t.view(-1)
            # Create masks for the specific trajectory phases
            early_mask = t_flat < 0.3
            mid_mask = (t_flat >= 0.3) & (t_flat <= 0.7)
            late_mask = t_flat > 0.7
            
            # Log them to W&B if they exist in the current batch
            early_loss = raw_loss[early_mask].mean() if early_mask.any() else torch.tensor(0.0, device=x1.device)
            mid_loss = raw_loss[mid_mask].mean() if mid_mask.any() else torch.tensor(0.0, device=x1.device)
            late_loss = raw_loss[late_mask].mean() if late_mask.any() else torch.tensor(0.0, device=x1.device)
            self.log("health/loss_t_early", early_loss, sync_dist=True, prog_bar=False)
            self.log("health/loss_t_mid", mid_loss, sync_dist=True, prog_bar=False)
            self.log("health/loss_t_late", late_loss, sync_dist=True, prog_bar=False)
                
            # Log the velocity norms to ensure the network isn't predicting zero
            target_speed = ut.norm(dim=1).mean()
            pred_speed = vt.norm(dim=1).mean()
            self.log("health/target_velocity_norm", target_speed, sync_dist=True, prog_bar=True)
            self.log("health/predicted_velocity_norm", pred_speed, sync_dist=True, prog_bar=True)
        
        return F.mse_loss(vt, ut), x1_pred

    def generate_informed_x0(self, labels, spatial_shape):
        batch_size = labels.shape[0]
        latent_dim = self.condition_mu.shape[1]
        
        label_counts = labels.sum(dim=1, keepdim=True).clamp(min=1e-8) # Prevent div by zero

        weighted_sum = torch.einsum('bc, cfdhw -> bfdhw', labels.float(), self.condition_mu)
        mu_3d = weighted_sum / label_counts.view(-1, 1, 1, 1, 1)
        scale = getattr(self, "latent_scale", 1.0)
        mu_3d = mu_3d * scale
        
        x0 = mu_3d + torch.randn_like(mu_3d)
        
        return x0
    
    def training_step(self,batch,batch_idx):
        slabs = batch["rsp"]                # Example shape: (B, 1, D, H, W)
        physics_grid = batch["physics_grid"]
        medical_labels = batch["condition"] # Example shape: (B, Slices, Angles, 21)
        
        physics_grid = (physics_grid - self.cond_mean) / self.cond_std
        cond_features = self.projection_mlp(physics_grid) 
        B, S, A, D = cond_features.shape
        cond_seq = cond_features.view(B, S * A, D)
        cfg_dropout_prob = 0.15

        #! CFG Training added from here
        drop_mask = torch.rand(B, device=self.device) < cfg_dropout_prob
        mask_labels = drop_mask.view(B, *([1] * (medical_labels.ndim - 1)))
        medical_labels = torch.where(mask_labels, torch.zeros_like(medical_labels), medical_labels)
        mask_seq = drop_mask.view(B, 1, 1)
        cond_seq = torch.where(mask_seq, torch.zeros_like(cond_seq), cond_seq)
        
        
        with torch.no_grad():
            z_mu, z_sigma = self.vae.encode(slabs)
            x1 = (z_mu + z_sigma * torch.randn_like(z_sigma).detach()) * self.latent_scale 
        
        fm_loss, x1_pred = self.flow_matching_loss(x1, cond_seq, medical_labels)
        latent_recon_loss = F.mse_loss(x1_pred, x1)
        self.log("train/flow_matching_loss", fm_loss.detach(), prog_bar=True, sync_dist=True)
        self.log("train/latent_recon_loss", latent_recon_loss.detach(), prog_bar=True, sync_dist=True)
        return fm_loss
    
    def validation_step(self, batch, batch_idx):
        slabs = batch["rsp"]
        physics_grid = batch["physics_grid"] 
        medical_labels = batch["condition"]
        physics_grid = (physics_grid - self.cond_mean) / self.cond_std
        
        with torch.no_grad():
            z_mu, z_sigma = self.vae.encode(slabs)
            x1 = z_mu * self.latent_scale
            cond_features = self.projection_mlp(physics_grid) 
            cond_seq = cond_features.view(x1.shape[0], -1, cond_features.shape[-1])
            fm_loss, x1_pred = self.flow_matching_loss(x1, cond_seq, medical_labels)
            
        latent_recon_loss = torch.nn.functional.mse_loss(x1_pred, x1)
        self.log("val/flow_matching_loss", fm_loss.detach(), prog_bar=True, sync_dist=True)
        self.log("val/latent_recon_loss", latent_recon_loss.detach(), prog_bar=True, sync_dist=True)
        return fm_loss
    
    #def configure_optimizers(self):
    #    opt = torch.optim.Adam(list(self.projection_mlp.parameters()) + list(self.velocity_net.parameters()), lr=self.lr)
    #    return opt
    def configure_optimizers(self):
        fine_tune_lr = 1e-5
        opt = torch.optim.Adam(list(self.projection_mlp.parameters()) + list(self.velocity_net.parameters()), lr= fine_tune_lr)
        
        # Smoothly decay the learning rate down to 1e-6 over the course of training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=50, #self.trainer.max_epochs, # The total number of epochs you are running
            eta_min=1e-6                   # The absolute lowest the LR will go
        )
        
        # Lightning requires this specific dictionary format when using a scheduler
        return { "optimizer": opt,"lr_scheduler": scheduler}