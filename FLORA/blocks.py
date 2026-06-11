import torch.nn as nn
from monai.networks.nets import AutoencoderKL
import torch
import torch.nn.functional as F

class ProjectionMLP(nn.Module):
    def __init__(self, in_channels=21, hidden_dim=128, embed_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            # 1. Expand the feature space
            nn.Linear(in_channels, hidden_dim),
            
            # 2. Normalize the mixed physics units! (Crucial)
            nn.LayerNorm(hidden_dim),
            
            # 3. Non-linear activation
            nn.GELU(),
            
            # 4. Project to the final embedding grid dimension
            nn.Linear(hidden_dim, embed_dim),
            #nn.LayerNorm(embed_dim)
        )

    def forward(self, condition_grid):
        """
        condition_grid: Tensor of shape (Batch, Slices, Angles, 21)
        Returns: Tensor of shape (Batch, Slices, Angles, embed_dim)
        """
        # The linear layers will independently process the last dimension (21)
        # for every single slice and angle perfectly in parallel.
        return self.network(condition_grid)
    
class DetectorBackprojector(nn.Module):
    def __init__(self, in_channels=21, proj_hidden=128, embed_dim=4, latent_spatial_shape=(4,64,64)):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection_mlp = ProjectionMLP(in_channels, proj_hidden, proj_hidden)
        
        # Final projection to spatial channels — zero-initialized output
        self.to_volume = nn.Linear(proj_hidden, embed_dim)
        nn.init.zeros_(self.to_volume.weight)
        nn.init.zeros_(self.to_volume.bias)
        
        self.latent_spatial_shape = latent_spatial_shape

    def forward(self, condition_grid):
        B, S, A, _ = condition_grid.shape
        D, H, W = self.latent_spatial_shape  # 16, 16, 4

        tokens = self.projection_mlp(condition_grid)   # (B, S, A, proj_hidden)
        tokens = self.to_volume(tokens)                # (B, S, A, embed_dim)

        # Reshape to 5D for trilinear interpolation: treat (S, A, 1) as a tiny volume
        tokens = tokens.permute(0, 3, 1, 2)                      # (B, embed_dim, S, A)
        tokens = tokens.unsqueeze(2)                              # (B, embed_dim, 1, S, A)
        tokens = F.interpolate(
            tokens,
            size=(D, H, W),                                       # (16, 16, 4)
            mode='trilinear',
            align_corners=False
        )                                                         # (B, embed_dim, D, H, W)
        return tokens



def configure_vae(vaekwgs):
    
    vae = AutoencoderKL(**vaekwgs)
    checkpoint = torch.load("checkpoints/Medical_condition/PHASE_1/last.ckpt", map_location="cpu")
    
    vae_state_dict = {
        k.replace("vae.", ""): v 
        for k, v in checkpoint["state_dict"].items() 
        if k.startswith("vae.")}
    vae.load_state_dict(vae_state_dict)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae
