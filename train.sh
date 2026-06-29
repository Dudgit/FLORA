#!/bin/bash
#SBATCH --job-name=flora_train
#SBATCH --partition=ai
#SBATCH --gres=gpu:4
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=32            
#SBATCH --time=4:00:00            
#SBATCH --output=logs/train_weighted_loss.log

module load singularity

export SCRATCH="/home/nr_fldb/nr_floraai_scratch"
export REAL_SCRATCH=$(readlink -f $SCRATCH)

export HDD="/home/nr_fldb/nr_floraai/data/ct_rate_subset/dataset/train_fixed"
export REAL_HDD=$(readlink -f $HDD)

export SINGULARITYENV_WANDB_API_KEY="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"
singularity exec --nv --pwd $REAL_SCRATCH -B $REAL_SCRATCH:$REAL_SCRATCH -B $REAL_HDD:/mnt/ct_data FLORA/flora.sif python main_stage2.py