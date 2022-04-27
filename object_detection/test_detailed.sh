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
#SBATCH --job-name=test_obj_detection

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate

# $1 -> composition test
# $2 -> objdet architecture 
# $3 -> dataset type

python3 tools/test_detailed.py \
    "./configs/custom/cascade_rcnn_config.py" \
    "work_dirs/$3/$2/$1/best.pth" \
    --eval "bbox" \
    --outfile "test.txt"
    --outcsv "test.csv"
    --cfg-options   "classwise=True" \
                    "work_dir=work_dirs/test/$3/$2/$1/" \

