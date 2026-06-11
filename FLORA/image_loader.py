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


conditions_dir = "conditions"

    # Helper function to dynamically discover matching .npy files from the rsp filename
def prepare_paths(list_of_dicts):
    updated_list = []
    for item in list_of_dicts:
        rsp_path = item["rsp"]  # e.g., "volumes/patient_01.nii.gz"
        
        filename = os.path.basename(rsp_path)
        base_name = filename.split('.')[0] 
        npy_path = os.path.join(conditions_dir, f"{base_name}_embedding.npy")
        updated_list.append({
            "rsp": rsp_path,
            "physics_grid": npy_path,
            "condition": item["condition"]
        })
    return updated_list

def get_train_loader_stage2(batch_size=2, num_workers=4,TARGET_SPACING=(2.0, 2.0, 3.0), TARGET_SIZE=(256, 256, 64)):

    transforms = Compose([
        LoadImaged(keys=["rsp","physics_grid"],image_only=True),             
        EnsureChannelFirstd(keys=["rsp"]),    
        Spacingd(keys=["rsp"],pixdim=TARGET_SPACING,mode="trilinear" ),#from bilinear to trilinear
        ResizeWithPadOrCropd(keys=["rsp"],spatial_size=TARGET_SIZE,mode="constant",constant_values=0.0),
        ScaleIntensityRanged(
            keys=["rsp"], 
            a_min=0.0, a_max=3.2, # The physical RSP range you care about
            b_min=0.0, b_max=1.0, # The neural network range
            clip=True),        
        EnsureTyped(keys=["rsp","physics_grid","condition"], dtype=torch.float32,track_meta=False)])

    cache_dir = "/home/nr_fldb/nr_floraai_scratch/monai_cache"
    os.makedirs(cache_dir, exist_ok=True)


    json_path = "/home/nr_fldb/nr_floraai_scratch/utils/train_dicts_conds.json" 
    val_json_path = "/home/nr_fldb/nr_floraai_scratch/utils/val_dicts_conds.json"
    with open(json_path, "r") as f:
        data_list = json.load(f)
    train_data_list = prepare_paths(data_list)

    train_ds = PersistentDataset(
        data=train_data_list, 
        transform=transforms, 
        cache_dir=cache_dir)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    with open(val_json_path, "r") as f:
        val_data_list = json.load(f)
    val_data_list = prepare_paths(val_data_list)
    
    val_ds = PersistentDataset(
        data=val_data_list, 
        transform=transforms,
        cache_dir=cache_dir)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, val_loader


def inject_ct_paths(data_list, ct_root_dir):
    """
    Parses the list of dictionaries and maps each RSP entry to its 
    corresponding deeply nested X-ray CT file path safely.
    """
    updated_list = []
    missing_count = 0
    
    for item in data_list:
        rsp_rel_path = item["rsp"] 
        rsp_filename = os.path.basename(rsp_rel_path)
        
        tokens = rsp_filename.replace(".nii.gz", "").split("_")
        
        if len(tokens) >= 3:
            prefix = tokens[0]       
            group_num = tokens[1]    
            case_letter = tokens[2]  
            
            group_folder = f"{prefix}_{group_num}"               
            case_folder = f"{prefix}_{group_num}_{case_letter}"   
            ct_filename = f"{case_folder}_1.nii.gz"                
            
            ct_absolute_path = os.path.join(
                ct_root_dir, group_folder, case_folder, ct_filename
            )
            #print(ct_absolute_path)  # Debug: Print the constructed CT path for verification
            
            # --- THE SAFETY CHECK ---
            if os.path.exists(ct_absolute_path) and os.path.exists(item["rsp"]):
                item["ct"] = ct_absolute_path
                updated_list.append(item)
            else:
                missing_count += 1
                # Print the first missing path so you can debug the string formatting
                if missing_count == 1:
                    print(f"\n[DEBUG] Example of missing file path:\n -> Tried looking for: {ct_absolute_path}")

    print(f"Successfully paired {len(updated_list)} files. Skipped {missing_count} missing files.")
    return updated_list


def get_train_loader_CT(batch_size=2, num_workers=4,TARGET_SPACING=(2.0, 2.0, 3.0), TARGET_SIZE=(256, 256, 64)):

    transforms = Compose([
        LoadImaged(keys=["rsp","physics_grid","ct"],image_only=True),             
        EnsureChannelFirstd(keys=["rsp","ct"]),    
        Spacingd(keys=["rsp","ct"],pixdim=TARGET_SPACING,mode="trilinear" ),#from bilinear to trilinear
        ResizeWithPadOrCropd(keys=["rsp","ct"],spatial_size=TARGET_SIZE,mode="constant",constant_values=0.0),
        ScaleIntensityRanged(
            keys=["rsp"], 
            a_min=0.0, a_max=3.2, # The physical RSP range you care about
            b_min=0.0, b_max=1.0, # The neural network range
            clip=True),
        ScaleIntensityRanged(
            keys=["ct"],
            a_min=-1000.0, a_max=3000.0, # Typical Hounsfield Unit range for CT scans
            b_min=0.0, b_max=1.0,
            clip=True),        
        EnsureTyped(keys=["rsp","physics_grid","condition","ct"], dtype=torch.float32,track_meta=False)])

    cache_dir = "/home/nr_fldb/nr_floraai_scratch/monai_cache"
    os.makedirs(cache_dir, exist_ok=True)


    ct_root = "/mnt/ct_data"
    json_path = "/home/nr_fldb/nr_floraai_scratch/utils/train_dicts_conds.json" 
    val_json_path = "/home/nr_fldb/nr_floraai_scratch/utils/val_dicts_conds.json"
    with open(json_path, "r") as f:
        data_list = json.load(f)
    train_data_list = prepare_paths(data_list)
    train_data_list = inject_ct_paths(train_data_list, ct_root_dir=ct_root)

    train_ds = PersistentDataset(
        data=train_data_list, 
        transform=transforms, 
        cache_dir=cache_dir)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    with open(val_json_path, "r") as f:
        val_data_list = json.load(f)
    val_data_list = prepare_paths(val_data_list)
    val_data_list = inject_ct_paths(val_data_list, ct_root_dir=ct_root)
    
    val_ds = PersistentDataset(
        data=val_data_list, 
        transform=transforms,
        cache_dir=cache_dir)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, val_loader


import tqdm._tqdm
if __name__ == "__main__":
    train_loader,val_loader = get_train_loader_CT()
    pbar = tqdm.tqdm(train_loader)
    for batch in pbar:
        batch_shape = batch["rsp"].shape
        physics_grid_shape = batch["physics_grid"].shape
        condition_shape = batch["condition"].shape
        ct_shape = batch["ct"].shape
        pbar.set_postfix({
            "rsp_shape": batch_shape,
            "physics_grid_shape": physics_grid_shape,
            "condition_shape": condition_shape,
            "ct_shape": ct_shape})

