import os
# Put this at the VERY TOP of your script!
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"
os.environ["WANDB_MODE"] = "offline"

from FLORA.model import FLORA
from FLORA.image_loader import get_train_loader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
from pytorch_lightning.plugins.environments import LightningEnvironment
import torch
from FLORA.callbacks import EMACallback, SlabVisualizationCallback
from omegaconf import OmegaConf
torch.set_float32_matmul_precision('medium')
#torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)



import wandb
api_key = os.environ.get("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key, relogin=True)


def train():
    run_name = "PHASE_1_losses_for_bubbles"
    cfg = OmegaConf.load("config.yaml")
    model = FLORA(**cfg )
    train_loader, val_loader = get_train_loader(batch_size=1, num_workers=4)
    
    # 1. Save the best validation model for your final results
    val_callback = ModelCheckpoint(monitor='val_recon_loss',dirpath=os.path.join("checkpoints", run_name),filename='best_val_recon', save_top_k=1,mode='min', enable_version_counter=False )
    
    # 2. Save the absolute latest state for HPC 8-hour chaining
    last_callback = ModelCheckpoint(dirpath=os.path.join("checkpoints", run_name),filename='last',save_last=True)
    ema_callback = EMACallback(decay=0.999)
    vis_callback = SlabVisualizationCallback(every_n_epochs=5)
    wandb_logger = WandbLogger(log_model=True, project="VAE_Train", name=run_name.replace("PHASE_1_",""),entity="FLORAAI",
                               #save_dir="/tmp"
                               )

    trainer = pl.Trainer(
        max_epochs=100,
        precision="bf16-mixed", # Don't forget your memory saver!
        callbacks=[val_callback, last_callback, vis_callback, ema_callback],
        logger=wandb_logger,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        plugins=LightningEnvironment()
    )

    # 3. HPC Auto-Resume Logic
    last_ckpt_path = os.path.join("checkpoints", run_name, "last.ckpt")
    
    if os.path.exists(last_ckpt_path):
        print(f"Resuming 8-hour chain from {last_ckpt_path}...")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt_path)
    else:
        print("Starting fresh training run...")
        trainer.fit(model, train_loader, val_loader)




if __name__ == "__main__":
    
    train()
