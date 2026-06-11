#!/bin/bash
#SBATCH --job-name=flora_train
#SBATCH --partition=ai             
#SBATCH --gres=gpu:4               
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=32         
#SBATCH --mem=256G                  
#SBATCH --time=4:00:00            
#SBATCH --output=logs/train_CT.log

module load singularity

export WORKSPACE="/home/nr_fldb/nr_floraai_scratch"
export REAL_WORKSPACE=$(readlink -f $WORKSPACE)

export SINGULARITYENV_PYTHONPATH=$REAL_WORKSPACE
export SINGULARITYENV_WANDB_API_KEY="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"


export HDD="/home/nr_fldb/nr_floraai/data/ct_rate_subset/dataset/train_fixed"
export REAL_HDD=$(readlink -f $HDD)


cd $REAL_WORKSPACE

singularity exec --nv --pwd $REAL_WORKSPACE -B $REAL_WORKSPACE -B $REAL_HDD:/mnt/ct_data FLORA/flora.sif python main_stage2.py

