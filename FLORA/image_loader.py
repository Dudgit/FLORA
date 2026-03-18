import os
from monai.data import PersistentDataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Spacingd,ResizeWithPadOrCropd, EnsureTyped
import torch
import json
from monai.transforms import ScaleIntensityRanged




def get_train_loader(batch_size=2, num_workers=4,TARGET_SPACING=(2.0, 2.0, 3.0), TARGET_SIZE=(256, 256, 64)):

    transforms = Compose([
        LoadImaged(keys=["rsp"]),             
        EnsureChannelFirstd(keys=["rsp"]),    
        Spacingd(keys=["rsp"],pixdim=TARGET_SPACING,mode="trilinear" ),#from bilinear to trilinear
        ResizeWithPadOrCropd(keys=["rsp"],spatial_size=TARGET_SIZE,mode="constant",constant_values=0.0),
        ScaleIntensityRanged(
            keys=["rsp"], 
            a_min=0.0, a_max=3.2, # The physical RSP range you care about
            b_min=0.0, b_max=1.0, # The neural network range
            clip=True),        
        EnsureTyped(keys=["rsp","condition"], dtype=torch.float32)])

    cache_dir = "/home/nr_fldb/nr_floraai_scratch/monai_cache"
    os.makedirs(cache_dir, exist_ok=True)


    json_path = "/home/nr_fldb/nr_floraai_scratch/utils/train_dicts_clean.json" 
    val_json_path = "/home/nr_fldb/nr_floraai_scratch/utils/val_dicts.json"
    with open(json_path, "r") as f:
        data_list = json.load(f)

    train_ds = PersistentDataset(
        data=data_list, 
        transform=transforms, 
        cache_dir=cache_dir
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    with open(val_json_path, "r") as f:
        val_data_list = json.load(f)
    val_ds = PersistentDataset(
        data=val_data_list, 
        transform=transforms,
        cache_dir=cache_dir
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader