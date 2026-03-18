import pytorch_lightning as pl
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.losses import PerceptualLoss
from monai.losses import PatchAdversarialLoss
import torch
import torch.nn as nn

def compute_r1_penalty(discriminator, real_images):
    """R1 gradient penalty for discriminator regularization."""
    real_images = real_images.detach().requires_grad_(True)
    real_pred = discriminator(real_images)
    
    # MONAI discriminators return a list to support multi-scale uniformly.
    # We gracefully unpack the list and sum the actual PyTorch tensors.
    if isinstance(real_pred, (list, tuple)):
        pred_sum = sum(out.sum() for out in real_pred)
    else:
        pred_sum = real_pred.sum()
        
    grad_real = torch.autograd.grad(
        outputs=pred_sum, inputs=real_images, create_graph=True
    )[0]
    
    r1_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return r1_penalty

class FLORA(pl.LightningModule):
    #TODO: Load everything from config.yaml
    #TODO: Modify singularity with omegaconf
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.vae = AutoencoderKL(spatial_dims=3,
                            in_channels=1,
                            out_channels=1,
                            latent_channels=4,
                            num_res_blocks=2,
                            attention_levels=(False, False, False, True),
                            norm_num_groups=32)
    
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        
        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            channels=64,
            in_channels=1,
            out_channels=1)
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3, 
            network_type="squeeze", 
            is_fake_3d=True, 
            fake_3d_ratio=0.2 # Processes 20% of the slices to save GPU memory
        )
        self.perceptual_weight = 0.01 # Start small!
                # Apply spectral norm to discriminator for training stability
        for name, module in self.discriminator.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.utils.spectral_norm(module)

        self.discriminator_warmup_steps = 500
        self.phase = 1
        self.last_loss = {}

        # --- Scheduling hyperparameters ---
        # KL annealing: ramp from 0 to kl_target_weight over kl_warmup_steps
        self.kl_target_weight = 1e-3
        self.kl_warmup_steps = 2000

        # Adversarial weight annealing: ramp from adv_weight_init to adv_weight_max
        self.adv_weight_init = 0.001
        self.adv_weight_max = 0.01
        self.adv_warmup_steps = 5000  # steps after disc warmup to reach max

        # R1 gradient penalty
        self.r1_weight = 10.0
        self.r1_every_n_steps = 16  # apply R1 every N steps to save compute

    def _get_kl_weight(self):
        """Linearly anneal KL weight from 0 to target."""
        return min(1.0, self.global_step / max(1, self.kl_warmup_steps)) * self.kl_target_weight

    def _get_adv_weight(self):
        """Linearly ramp adversarial weight after discriminator warmup."""
        steps_since_warmup = max(0, self.global_step - self.discriminator_warmup_steps)
        frac = min(1.0, steps_since_warmup / max(1, self.adv_warmup_steps))
        return self.adv_weight_init + frac * (self.adv_weight_max - self.adv_weight_init)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.vae.parameters(), lr=1e-4)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=5e-5)
        return [opt_g, opt_d], []
    
    def shared_step_phase_one(self, batch, prefix, is_train=True):
        full_volumes = batch["rsp"]  
        
        slabs = torch.split(full_volumes, split_size_or_sections=32, dim=4)        
        discriminator_enabled = (not is_train) or (self.global_step >= self.discriminator_warmup_steps)

        # Current scheduled weights
        kl_weight = self._get_kl_weight()
        adv_weight = self._get_adv_weight()
        apply_r1 = is_train and (self.global_step % self.r1_every_n_steps == 0)

        if is_train:
            opt_g, opt_d = self.optimizers()
            opt_g.zero_grad()
            if discriminator_enabled:
                opt_d.zero_grad()

        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_d_real_loss = 0.0  
        total_d_fake_loss = 0.0
        total_r1 = 0.0

        for slab in slabs:
            if is_train and not discriminator_enabled:
                for p in self.discriminator.parameters():
                    p.requires_grad = False

            recon, z_mean, z_log_var = self.vae(slab)            
            recon_loss = torch.nn.functional.mse_loss(recon, slab)
            kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())


            logits_fake = self.discriminator(recon.contiguous()) 
            g_adv_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            p_loss = self.perceptual_loss(recon.contiguous(), slab.contiguous())
            slab_g_loss = (recon_loss + (self.perceptual_weight * p_loss) + kl_weight * kl_loss + adv_weight * g_adv_loss) / len(slabs)
            if is_train:
                self.manual_backward(slab_g_loss)

            if is_train and not discriminator_enabled:
                for p in self.discriminator.parameters():
                    p.requires_grad = True

            # --- DISCRIMINATOR LOSS ---
            if discriminator_enabled:
                logits_fake_d = self.discriminator(recon.contiguous().detach())
                loss_d_fake = self.adv_loss(logits_fake_d, target_is_real=False, for_discriminator=True)

                logits_real_d = self.discriminator(slab.contiguous())
                loss_d_real = self.adv_loss(logits_real_d, target_is_real=True, for_discriminator=True)

                slab_d_loss = ((loss_d_fake + loss_d_real) * 0.5) / len(slabs)

                # R1 gradient penalty on real data
                r1_loss = torch.tensor(0.0, device=full_volumes.device)
                if apply_r1:
                    r1_loss = compute_r1_penalty(self.discriminator, slab.contiguous())
                    slab_d_loss = slab_d_loss + (self.r1_weight * r1_loss) / len(slabs)

                if is_train:
                    self.manual_backward(slab_d_loss)
            else:
                device = full_volumes.device
                loss_d_fake = torch.tensor(0.0, device=device)
                loss_d_real = torch.tensor(0.0, device=device)
                slab_d_loss = torch.tensor(0.0, device=device)
                r1_loss = torch.tensor(0.0, device=device)

            total_recon_loss += recon_loss.detach()
            total_kl_loss += kl_loss.detach()
            total_g_loss += g_adv_loss.detach()
            total_d_loss += slab_d_loss.detach() * len(slabs)
            total_d_real_loss += loss_d_real.detach()  
            total_d_fake_loss += loss_d_fake.detach()
            total_r1 += r1_loss.detach()
           
        if is_train:
            opt_g.step()
            if discriminator_enabled:
                opt_d.step()

        var_dict = {
            "recon_loss": total_recon_loss / len(slabs),
            "kl_loss": total_kl_loss / len(slabs),
            "g_adv_loss": total_g_loss / len(slabs),
            "d_loss": total_d_loss / len(slabs),
            "d_real_loss": total_d_real_loss / len(slabs),
            "d_fake_loss": total_d_fake_loss / len(slabs),
            "kl_weight": torch.tensor(kl_weight, device=self.device),
            "adv_weight": torch.tensor(adv_weight, device=self.device),
            "r1_penalty": total_r1 / len(slabs),
        }
        self.last_loss = var_dict
        self.log_step(var_dict, prefix)

   
    def training_step(self, batch, batch_idx):
        if self.phase == 1:
            loss = self.shared_step_phase_one(batch, 'train')
        else:
            loss = self.shared_step_phase_two(batch, batch_idx)
        return loss
    
 

    def on_train_epoch_end(self):
        with open('last_loss.txt', 'w') as f:
            for key, value in self.last_loss.items():
                f.write(f'{key}: {value.item()}\n')
    
    def log_step(self,log_dict,prefix):
        for key, value in log_dict.items():
            self.log(f'{prefix}_{key}', value, prog_bar=True, sync_dist=True)