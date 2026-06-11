import pytorch_lightning as pl
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.losses import PerceptualLoss
from monai.losses import PatchAdversarialLoss
import torch
import torch.nn as nn
import contextlib
import torch.nn.functional as F


def fft_loss(recon, target):
    """Penalizes differences in the frequency domain to preserve high-frequency detail."""
    recon_f32 = recon.float()
    target_f32 = target.float()
    fft_recon = torch.fft.fftn(recon_f32, dim=(-3, -2, -1))
    fft_target = torch.fft.fftn(target_f32, dim=(-3, -2, -1))
    return F.l1_loss(torch.abs(fft_recon), torch.abs(fft_target))


def gradient_loss_3d(recon, target):
    """Penalizes blurred edges by comparing spatial gradients along all 3 axes."""
    loss = 0.0
    for dim in (-3, -2, -1):
        grad_recon = torch.diff(recon, n=1, dim=dim)
        grad_target = torch.diff(target, n=1, dim=dim)
        loss += F.l1_loss(grad_recon, grad_target)
    return loss / 3.0


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

class FLORA(pl.LightningModule):
    def __init__(self, vaekwgs= default_vae_kwargs,disc_kwargs = default_disc_kwargs,kl_weight=0.01, adv_weight=0.1, perceptual_weight=0.01, lambda_weight=0.1
                 , bank_size=64,generator_lr=1e-4, discriminator_lr=5e-5,
                 fft_weight=0.1, gradient_weight=0.1):
        super().__init__()
        self.automatic_optimization = False

        self.vae = AutoencoderKL(**vaekwgs)
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.discriminator = PatchDiscriminator(**disc_kwargs)
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)

        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.lambda_weight = lambda_weight
        self.fft_weight = fft_weight
        self.gradient_weight = gradient_weight
        self.bank_size = bank_size
        self.latent_dim = default_vae_kwargs.get("latent_channels",3)
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.register_buffer("latent_bank", torch.randn(self.bank_size, self.latent_dim))
        self.register_buffer("label_bank", torch.zeros(self.bank_size, 18))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
    
    def configure_optimizers(self):
        # Use GAN betas
        opt_g = torch.optim.Adam(self.vae.parameters(), lr=self.generator_lr, betas =(0.5, 0.999),weight_decay=1e-5)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas =(0.5, 0.999))
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.trainer.max_epochs, eta_min=1e-6)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return [opt_g, opt_d], [sched_g, sched_d]
     

    def process_slab(self,slab,labels,optimizer_index=0,prefix="train"):
        loss_dict = {}
        if optimizer_index == 0:
            recon, z_mean, z_log_var = self.vae(slab)
            z_pooled = torch.mean(z_mean, dim=[2, 3, 4])    
            supcon_loss = self.supervised_contrastive_loss(z_pooled, labels)        
            recon_loss = torch.nn.functional.l1_loss(recon, slab)
            kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
            
            logits_fake = self.discriminator(recon.contiguous()) 
            g_adv_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            
            d_loss = torch.tensor(0.0, device=slab.device)
            p_loss = self.perceptual_loss(recon.contiguous(), slab.contiguous())

            # High-frequency preservation losses
            freq_loss = fft_loss(recon, slab)
            grad_loss = gradient_loss_3d(recon, slab)

            return_loss = (recon_loss 
                          + self.kl_weight * kl_loss 
                          + self.adv_weight * g_adv_loss 
                          + self.perceptual_weight * p_loss 
                          + self.lambda_weight * supcon_loss
                          + self.fft_weight * freq_loss
                          + self.gradient_weight * grad_loss)
            loss_dict.update({
                'recon_loss': recon_loss, 'kl_loss': kl_loss, 
                'g_adv_loss': g_adv_loss, 'p_loss': p_loss, 
                'supcon_loss': supcon_loss,
                'fft_loss': freq_loss, 'grad_loss': grad_loss
            })
        
        else:
            with torch.no_grad():
                recon, z_mean, z_log_var = self.vae(slab)            

            logits_real = self.discriminator(slab.contiguous())
            d_loss_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            logits_fake = self.discriminator(recon.detach().contiguous()) 
            d_loss_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            loss_dict.update({'d_loss_real': d_loss_real, 'd_loss_fake': d_loss_fake})
            return_loss = d_loss
        return return_loss, loss_dict
        
    def log_values(self, lossDict, prefix):
        for key, value in lossDict.items():
            self.log(f"{prefix}_{key}", value, prog_bar=True,sync_dist=True)

    def shared_step_phase_one(self, batch, prefix, optimizer_index=0, training=False):
        full_volumes = batch["rsp"]
        labels = batch["condition"]
        slabs = torch.split(full_volumes, split_size_or_sections=32, dim=4)
        final_loss = torch.tensor(0.0, device=full_volumes.device)
        if optimizer_index == 0:
            loss_dict = {'recon_loss': 0.0, 'kl_loss': 0.0, 'g_adv_loss': 0.0, 'p_loss': 0.0, 'supcon_loss': 0.0, 'fft_loss': 0.0, 'grad_loss': 0.0}
        else:
            loss_dict = {'d_loss_real': 0.0, 'd_loss_fake': 0.0}
        for i, slab in enumerate(slabs):
            is_last_slab = (i == len(slabs) - 1)
            
            if training and self.trainer.world_size > 1 and not is_last_slab:
                sync_context = self.trainer.model.no_sync()
            else:
                sync_context = contextlib.nullcontext()
                
            with sync_context:
                slab_loss, loss_dict_slab = self.process_slab(slab, labels, prefix=prefix, optimizer_index=optimizer_index)
                scaled_loss = slab_loss / len(slabs)
                final_loss += scaled_loss.detach()
                for key, value in loss_dict_slab.items():
                    loss_dict[key] += value.detach()
                if training:
                    self.manual_backward(scaled_loss)
                    
        opt_gen = "Generator" if optimizer_index == 0 else "Discriminator"
        loss_dict[f"{prefix}/{opt_gen}_volume_total_loss"] = final_loss
        self.log_values(loss_dict, prefix=prefix)
        return final_loss, loss_dict
            
    def training_step(self,batch,batch_idx):
        opt_g, opt_d = self.optimizers()
        opt_g.zero_grad()
        g_loss = self.shared_step_phase_one(batch, prefix="train", optimizer_index=0,training=True)
        opt_g.step()

        opt_d.zero_grad()
        d_loss = self.shared_step_phase_one(batch, prefix="train", optimizer_index=1,training=True)
        opt_d.step()
    
    def validation_step(self,batch,batch_idx):
        self.shared_step_phase_one(batch, prefix="val", optimizer_index=0)
        self.shared_step_phase_one(batch, prefix="val", optimizer_index=1)
    
    def on_train_epoch_end(self):
        sched_g, sched_d = self.lr_schedulers()
        sched_g.step()
        sched_d.step()

    def supervised_contrastive_loss(self, z_pooled, labels, temperature=0.5):
    # 1. Normalize the incoming vector and the entire memory bank
        z_norm = F.normalize(z_pooled, dim=1)
        bank_norm = F.normalize(self.latent_bank, dim=1)
        
        # 2. Compare the current volume against the 64 historical volumes
        # z_norm: [Batch, D], bank_norm: [Bank_Size, D] -> similarity: [Batch, Bank_Size]
        similarity_matrix = torch.matmul(z_norm, bank_norm.T) / temperature
        
        # 3. Find shared medical conditions with the historical volumes
        labels = labels.float()
        shared_conditions = torch.matmul(labels, self.label_bank.T)
        mask = (shared_conditions > 0).float().to(z_pooled.device)
        
        # 4. Calculate Info-NCE Loss against the memory bank
        exp_logits = torch.exp(similarity_matrix)
        
        # Safety catch: If the bank is mostly empty (first few steps), don't punish the network
        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=z_pooled.device, requires_grad=True)
        else:
            log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
            loss = -mean_log_prob_pos.mean()
            
        # 5. Quietly update the rolling bank with the new volume (Strictly No Gradients!)
        if self.training:
            with torch.no_grad():
                for i in range(z_norm.shape[0]):
                    ptr = int(self.bank_ptr.item())
                    self.latent_bank[ptr] = z_norm[i].detach().clone()
                    self.label_bank[ptr] = labels[i].detach().clone()
                    self.bank_ptr[0] = (ptr + 1) % self.bank_size
                    
        return loss



