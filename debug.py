import os
import torch
import torch.distributed as dist

def main():
    # 1. Read the OS-level Slurm IDs
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    
    # 2. Translate Slurm IDs into PyTorch IDs
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Because our bash script gags Singularity, every container only sees 1 GPU.
    os.environ["LOCAL_RANK"] = "0" 
    
    print(f"[Rank {rank}] Waking up. Slurm assigned me to this container.")
    
    try:
        dist.init_process_group("nccl")
        
        # 3. Bind to the single GPU visible to this container
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        
        # 4. The Test
        tensor = torch.ones(10, device=device)
        print(f"[Rank {rank}] SUCCESS! Tensor sum: {tensor.sum().item()}")
        
        dist.destroy_process_group()
    except Exception as e:
        print(f"[Rank {rank}] CRASHED: {e}")

if __name__ == "__main__":
    main()