import os
# Put this at the VERY TOP of your script!
#os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"
#os.environ["WANDB_MODE"] = "offline"

from FLORA.model_stage2 import FLORA_stage2
from FLORA.model_CT import FLORA_CT, FLORA_CT_changedBlock
from FLORA.image_loader import get_train_loader_stage2, get_train_loader_CT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
from pytorch_lightning.plugins.environments import LightningEnvironment
from FLORA.callbacks import Stage2VisualizationCallback, PriorDebuggingCallback, CTtoPCTVisualizationCallback
import torch
from omegaconf import OmegaConf
torch.set_float32_matmul_precision('medium')

import wandb
api_key = os.environ.get("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key, relogin=True)


def start_training(model,trainer,train_loader,val_loader,run_name):
    last_ckpt_path = os.path.join("checkpoints", run_name, "last.ckpt")
    
    if os.path.exists(last_ckpt_path):
        print(f"Resuming 8-hour chain from {last_ckpt_path}...")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt_path)
    else:
        print("Starting fresh training run...")
        trainer.fit(model, train_loader, val_loader)

def train():
    run_name = "PHASE_2_CFG"
    cfg = OmegaConf.load("config.yaml")
    model = FLORA_stage2()
    
    train_loader, val_loader = get_train_loader_stage2(batch_size=8, num_workers=4)
    
    val_callback = ModelCheckpoint(monitor='val/flow_matching_loss',dirpath=os.path.join("checkpoints", run_name),filename='best_val_flow', save_top_k=1,mode='min', enable_version_counter=False )
    last_callback = ModelCheckpoint(dirpath=os.path.join("checkpoints", run_name),filename='last',save_last=True)
    vis_callback = Stage2VisualizationCallback(every_n_epochs=5)
    prior_callback = PriorDebuggingCallback(every_n_epochs=5)
    
    wandb_logger = WandbLogger(log_model=True, project="FLOW", name=run_name.replace("PHASE_2_",""),entity="FLORAAI",save_dir="/tmp")
    
    trainer = pl.Trainer(
        max_epochs=150,
        precision="bf16-mixed",callbacks=[val_callback, last_callback, vis_callback, prior_callback],logger=wandb_logger,accelerator='gpu',
        log_every_n_steps=5,num_sanity_val_steps=0,plugins=LightningEnvironment())
    start_training(model,trainer,train_loader,val_loader,run_name)

def train_ct_only():
    run_name = "PHASE_2_CT_ONLY"
    cfg = OmegaConf.load("config.yaml")
    vaekwgs = cfg.vae_kwgs
    velocity_kwargs = cfg.velocity_kwargs
    ct_cfg = cfg.ct_train_params
    model = FLORA_CT(vaekwgs=vaekwgs, velocity_kwargs=velocity_kwargs,lr=ct_cfg.lr,use_detector_context=False)
    train_loader, val_loader = get_train_loader_CT(batch_size=8, num_workers=8)
    val_callback = ModelCheckpoint(monitor='val/flow_matching_loss',dirpath=os.path.join("checkpoints", run_name),filename='best_val_flow', save_top_k=1,mode='min', enable_version_counter=False )
    last_callback = ModelCheckpoint(dirpath=os.path.join("checkpoints", run_name),filename='last',save_last=True)
    vis_callback = CTtoPCTVisualizationCallback(every_n_epochs=5)
    wandb_logger = WandbLogger(log_model=True, project="FLOW", name=run_name.replace("PHASE_2_",""),entity="FLORAAI",save_dir="/tmp")
    trainer = pl.Trainer(
        max_epochs=150,precision="bf16-mixed", callbacks=[val_callback, last_callback, vis_callback], logger=wandb_logger,accelerator='gpu', 
        log_every_n_steps=5,plugins=LightningEnvironment(),num_sanity_val_steps=2,
        strategy="ddp_find_unused_parameters_true")
    start_training(model,trainer,train_loader,val_loader,run_name)


def train_ct_detector():
    run_name = "PHASE_2_CT_ONLY_WEIGHTED_LOSS"
    cfg = OmegaConf.load("config.yaml")
    vaekwgs = cfg.vae_kwgs
    velocity_kwargs = cfg.velocity_kwargs
    ct_cfg = cfg.ct_train_params
    detectorproj_kwargs = cfg.detectorproj_kwargs
    model = FLORA_CT_changedBlock(vaekwgs=vaekwgs, velocity_kwargs=velocity_kwargs,lr=ct_cfg.lr,use_detector_context=False,detectorproj_kwargs=detectorproj_kwargs)
    train_loader, val_loader = get_train_loader_CT(batch_size=8, num_workers=8)
    val_callback = ModelCheckpoint(monitor='val/flow_matching_loss',dirpath=os.path.join("checkpoints", run_name),filename='best_val_flow', save_top_k=1,mode='min', enable_version_counter=False )
    last_callback = ModelCheckpoint(dirpath=os.path.join("checkpoints", run_name),filename='last',save_last=True)
    wandb_logger = WandbLogger(log_model=True, project="FLOW", name=run_name.replace("PHASE_2_",""),entity="FLORAAI",save_dir="/tmp")
    trainer = pl.Trainer(
        max_epochs=150,precision="bf16-mixed", callbacks=[val_callback, last_callback], logger=wandb_logger,accelerator='gpu', 
        log_every_n_steps=5,plugins=LightningEnvironment(),num_sanity_val_steps=2,
        strategy="ddp_find_unused_parameters_true")
    start_training(model,trainer,train_loader,val_loader,run_name)


if __name__ == "__main__":
    train_ct_detector()
