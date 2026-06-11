#!/bin/bash
#SBATCH --job-name=flora_train
#SBATCH --partition=cpu
#SBATCH --mem=64G 
#SBATCH --cpus-per-task=16            
#SBATCH --time=4:00:00            
#SBATCH --output=logs/build_cache.log

export SCRATCH="/home/nr_fldb/nr_floraai_scratch"
export REAL_SCRATCH=$(readlink -f $SCRATCH)

# 2. Resolve the EXACT deep CT folder
export HDD="/home/nr_fldb/nr_floraai/data/ct_rate_subset/dataset/train_fixed"
export REAL_HDD=$(readlink -f $HDD)

# 3. Mount to the safe /mnt directory
singularity exec --pwd $REAL_SCRATCH -B $REAL_SCRATCH:$REAL_SCRATCH -B $REAL_HDD:/mnt/ct_data FLORA/flora.sif python FLORA/image_loader.py