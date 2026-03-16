#!/bin/bash
#SBATCH -N 1                   # 
#SBATCH -n 120                   # 
#SBATCH --mem=100g             # 
#SBATCH -J "RL VEX"   # 
#SBATCH -p short               # 
#SBATCH -t 23:00:00            # 
#SBATCH --gres=gpu:1           # 
#SBATCH -C "H100"              #

module load cuda         # 
source .venv/bin/activate  #
uv run train.py        #