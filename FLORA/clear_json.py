import json
import nibabel as nib
from tqdm import tqdm
import os

def scan_and_clean_json(json_path, output_path):
    print(f"Scanning {json_path} for corrupted volumes...")
    
    with open(json_path, "r") as f:
        data_list = json.load(f)
        
    valid_data = []
    corrupted_files = []
    
    # Wrap in tqdm for a nice progress bar
    for item in tqdm(data_list, desc="Checking NIfTI files"):
        file_path = item["rsp"]
        try:
            # Force nibabel to actually read the zipped data array
            img = nib.load(file_path)
            _ = img.get_fdata() 
            valid_data.append(item)
        except EOFError:
            corrupted_files.append(file_path)
        except Exception as e:
            # Catches any other Nibabel read errors
            corrupted_files.append(file_path)
            
    print(f"\nScan Complete for {json_path}!")
    print(f"Found {len(corrupted_files)} corrupted files.")
    
    if len(corrupted_files) > 0:
        print("Saving cleaned JSON...")
        with open(output_path, "w") as f:
            json.dump(valid_data, f, indent=4)
        print("✅ Cleaned dictionary saved!")
        
        # Optional: Print the bad files so you know which ones broke
        for bad_file in corrupted_files:
            print(f"Corrupted: {bad_file}")
    else:
        print("✅ All files are perfectly healthy!")

# --- Run the Scanner ---
if __name__ == "__main__":
    train_json = "/home/nr_fldb/nr_floraai_scratch/utils/train_dicts.json"
    val_json = "/home/nr_fldb/nr_floraai_scratch/utils/val_dicts.json"

    scan_and_clean_json(train_json, train_json.replace(".json", "_clean.json"))
    scan_and_clean_json(val_json, val_json.replace(".json", "_clean.json"))