import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.networks.nets import  DiffusionModelUNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from FLORA.blocks import ProjectionMLP, configure_vae, DetectorBackprojector

class FLORA_CT(pl.LightningModule):
    def __init__(self,vaekwgs = None,velocity_kwargs = None,lr=1e-4,use_detector_context=True,use_cfg=False):
        super().__init__()

        self.vae = configure_vae(vaekwgs)
        self.projection_mlp =  ProjectionMLP()
        self.lr = lr
        self.use_detector_context = use_detector_context
        self.use_cfg = use_cfg
        self.velocity_net = DiffusionModelUNet(**velocity_kwargs)
        self.register_buffer("latent_scale",torch.tensor(11.1560))
        mean_tensor = torch.tensor([0.008381759747862816, 0.020576655864715576, 0.03292844071984291, 0.0516696535050869, 0.07965941727161407, 0.14866508543491364, 0.455157071352005, 167.81105041503906, 174.27532958984375, 177.35848999023438, 180.73336791992188, 184.35720825195312, 187.68963623046875, 226.7608642578125, 0.04672469571232796, 6.969912528991699, 0.07428985834121704, 0.002276827348396182, 3.0780694484710693, 6.565859317779541, 0.7064194679260254])
        std_tensor = torch.tensor([0.020250756293535233, 0.021754246205091476, 0.025645112618803978, 0.03343982622027397, 0.04642146825790405, 0.06932703405618668, 0.13534820079803467, 36.7294807434082, 40.928672790527344, 42.93149948120117, 45.0711555480957, 47.05894470214844, 48.72496795654297, 49.305912017822266, 0.02668251283466816, 5.117215633392334, 0.13192923367023468, 0.28002825379371643, 0.46335211396217346, 0.7556602954864502, 0.4064233601093292])
        self.register_buffer("cond_mean", mean_tensor)
        self.register_buffer("cond_std", std_tensor)
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.projection_mlp.parameters()) + list(self.velocity_net.parameters()), lr= self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.trainer.max_epochs,eta_min=1e-6)
        return { "optimizer": opt,"lr_scheduler": scheduler}
    
    def flow_matching_loss(self, ct_images, pct_images, context):
        """
        Computes the Rectified Flow Matching loss between CT and pCT latents.
        """
        with torch.no_grad():
            z_0, _ = self.vae.encode(ct_images) 
            z_1, _ = self.vae.encode(pct_images)
            
            z_0 = z_0 * self.latent_scale
            z_1 = z_1 * self.latent_scale

        B = z_0.shape[0]
        t = torch.rand((B,), device=z_0.device)
        t_expand = t.view(B, 1, 1, 1, 1)
        
        z_t = t_expand * z_1 + (1.0 - t_expand) * z_0
        target_velocity = z_1 - z_0
        pred_velocity = self.velocity_net(x=z_t, timesteps=t) if not self.use_detector_context else self.velocity_net(x=z_t, timesteps=t, context=context)
        
        #Calculating Loss
        loss_per_sample = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3, 4])
        self.health_losses(loss_per_sample,t)
        
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        
        return fm_loss
    
    def health_losses(self,loss_per_sample,t):
        early_mask = t < 0.333
        mid_mask = (t >= 0.333) & (t < 0.666)
        late_mask = t >= 0.666
        
        early_loss = loss_per_sample[early_mask].mean() if early_mask.any() else torch.tensor(0.0, device=t.device)
        mid_loss = loss_per_sample[mid_mask].mean() if mid_mask.any() else torch.tensor(0.0, device=t.device)
        late_loss = loss_per_sample[late_mask].mean() if late_mask.any() else torch.tensor(0.0, device=t.device)

        self.log("health/loss_t_early", early_loss, sync_dist=True, prog_bar=False)
        self.log("health/loss_t_mid", mid_loss, sync_dist=True, prog_bar=False)
        self.log("health/loss_t_late", late_loss, sync_dist=True, prog_bar=False)

    def cfg_prep(self,cond_seq,B):
        cfg_dropout_prob = 0.15
        drop_mask = torch.rand(B, device=self.device) < cfg_dropout_prob
        mask_labels = drop_mask.view(B, *([1] * (medical_labels.ndim - 1)))
        medical_labels = torch.where(mask_labels, torch.zeros_like(medical_labels), medical_labels)
        mask_seq = drop_mask.view(B, 1, 1)
        cond_seq = torch.where(mask_seq, torch.zeros_like(cond_seq), cond_seq)
        return cond_seq

    
    def shared_step(self, batch, batch_idx, stage="train"):
        ct_images = batch["ct"]
        pct_images = batch["rsp"]
        context = None if not self.use_detector_context else batch["physics_grid"]
        context = (context - self.cond_mean) / self.cond_std
        cond_features = self.projection_mlp(context) 
        B, S, A, D = cond_features.shape
        cond_seq = cond_features.view(B, S * A, D)
        print(f"cond_seq shape: {cond_seq.shape}")
        if self.use_cfg:
            cond_seq = self.cfg_prep(cond_seq,B)
        
        fm_loss = self.flow_matching_loss(ct_images, pct_images, context=cond_seq)
        self.log(f"{stage}/flow_matching_loss", fm_loss.detach(), prog_bar=True, sync_dist=True)
        return fm_loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="train") 
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="val")
        
    
class FLORA_CT_changedBlock(pl.LightningModule):
    def __init__(self,vaekwgs = None,velocity_kwargs = None,detectorproj_kwargs = None,lr=1e-4,use_detector_context=True,use_cfg=False):
        super().__init__()

        self.vae = configure_vae(vaekwgs)
        self.projection  =  DetectorBackprojector(**detectorproj_kwargs)
        self.lr = lr
        self.use_detector_context = use_detector_context
        self.use_cfg = use_cfg
        if not use_detector_context:
            velocity_kwargs = dict(velocity_kwargs)
            velocity_kwargs['in_channels'] = 4
        self.velocity_net = DiffusionModelUNet(**velocity_kwargs)
        self.register_buffer("latent_scale",torch.tensor(11.1560))
        mean_tensor = torch.tensor([0.008381759747862816, 0.020576655864715576, 0.03292844071984291, 0.0516696535050869, 0.07965941727161407, 0.14866508543491364, 0.455157071352005, 167.81105041503906, 174.27532958984375, 177.35848999023438, 180.73336791992188, 184.35720825195312, 187.68963623046875, 226.7608642578125, 0.04672469571232796, 6.969912528991699, 0.07428985834121704, 0.002276827348396182, 3.0780694484710693, 6.565859317779541, 0.7064194679260254])
        std_tensor = torch.tensor([0.020250756293535233, 0.021754246205091476, 0.025645112618803978, 0.03343982622027397, 0.04642146825790405, 0.06932703405618668, 0.13534820079803467, 36.7294807434082, 40.928672790527344, 42.93149948120117, 45.0711555480957, 47.05894470214844, 48.72496795654297, 49.305912017822266, 0.02668251283466816, 5.117215633392334, 0.13192923367023468, 0.28002825379371643, 0.46335211396217346, 0.7556602954864502, 0.4064233601093292])
        self.register_buffer("cond_mean", mean_tensor)
        self.register_buffer("cond_std", std_tensor)
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.projection .parameters()) + list(self.velocity_net.parameters()), lr= self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.trainer.max_epochs,eta_min=1e-6)
        return { "optimizer": opt,"lr_scheduler": scheduler}
    
    def flow_matching_loss(self, ct_images, pct_images, context):
        """
        Computes the Rectified Flow Matching loss between CT and pCT latents.
        """
        with torch.no_grad():
            z_0, _ = self.vae.encode(ct_images) 
            z_1, _ = self.vae.encode(pct_images)
            
            z_0 = z_0 * self.latent_scale
            z_1 = z_1 * self.latent_scale

        B = z_0.shape[0]
        t = torch.rand((B,), device=z_0.device)
        t_expand = t.view(B, 1, 1, 1, 1)
        
        z_t = t_expand * z_1 + (1.0 - t_expand) * z_0
        target_velocity = z_1 - z_0
        if self.use_detector_context:
            z_t_conditioned = torch.cat([z_t, context], dim=1)
            pred_velocity = self.velocity_net(x=z_t_conditioned, timesteps=t)
        else:
            pred_velocity = self.velocity_net(x=z_t, timesteps=t)
        
        error = (pred_velocity - target_velocity) ** 2
        with torch.no_grad():
            density_weight = 1.0 + torch.abs(z_1) / (z_1.abs().mean() + 1e-8)
        weighted_error = error * density_weight
        fm_loss = weighted_error.mean()

        #Calculating Loss
        loss_per_sample = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3, 4])
        self.health_losses(loss_per_sample,t)
        
        #fm_loss = F.mse_loss(pred_velocity, target_velocity)
        
        return fm_loss
    
    def health_losses(self,loss_per_sample,t):
        early_mask = t < 0.333
        mid_mask = (t >= 0.333) & (t < 0.666)
        late_mask = t >= 0.666
        
        early_loss = loss_per_sample[early_mask].mean() if early_mask.any() else torch.tensor(0.0, device=t.device)
        mid_loss = loss_per_sample[mid_mask].mean() if mid_mask.any() else torch.tensor(0.0, device=t.device)
        late_loss = loss_per_sample[late_mask].mean() if late_mask.any() else torch.tensor(0.0, device=t.device)

        self.log("health/loss_t_early", early_loss, sync_dist=True, prog_bar=False)
        self.log("health/loss_t_mid", mid_loss, sync_dist=True, prog_bar=False)
        self.log("health/loss_t_late", late_loss, sync_dist=True, prog_bar=False)

    
    def shared_step(self, batch, batch_idx, stage="train"):
        ct_images = batch["ct"]
        pct_images = batch["rsp"]

        context = None
        if self.use_detector_context:
            raw_grid = batch["physics_grid"]
            raw_grid = (raw_grid - self.cond_mean) / self.cond_std
            context = self.projection(raw_grid)   # (B, embed_dim, D, H, W)

        fm_loss = self.flow_matching_loss(ct_images, pct_images, context=context)
        self.log(f"{stage}/flow_matching_loss", fm_loss.detach(), prog_bar=True, sync_dist=True)
        return fm_loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="train") 
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="val")
        
    
