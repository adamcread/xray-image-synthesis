#!/bin/bash
# X number of nodes with Y number of cores in each node.
#SBATCH -N 1
#SBATCH -c 4

# partition time limit and resource limit for the job
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small	
#SBATCH --mem=28g
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00

# job name
#SBATCH --job-name=runner
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../../../venv/bin/activate

python3 composed_dataset.py \
    --file $1 \
    --test $2