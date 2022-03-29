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
#SBATCH -o 'obj_detection_test.txt'

# job name
#SBATCH --job-name=obj_detection_test

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate

python3 tools/test.py \
    './configs/custom/cascade_rcnn_config.py' \
    'checkpoints/epoch_12.pth' \
    --out 'test.pkl' \
    --eval 'bbox' \
    --options "classwise=True"