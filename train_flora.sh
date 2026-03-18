#!/bin/bash
#SBATCH --job-name=flora_train
#SBATCH --partition=ai             
#SBATCH --gres=gpu:6               
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=32         
#SBATCH --mem=256G                  
#SBATCH --time=2:00:00            
#SBATCH --output=logs/train_%j.log

module load singularity

export WORKSPACE="/home/nr_fldb/nr_floraai_scratch"
export REAL_WORKSPACE=$(readlink -f $WORKSPACE)
export SINGULARITYENV_PYTHONPATH=$REAL_WORKSPACE
export SINGULARITYENV_WANDB_API_KEY="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"

cd $REAL_WORKSPACE

singularity exec --nv --pwd $REAL_WORKSPACE -B $REAL_WORKSPACE FLORA/flora.sif python main.py

