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
#SBATCH --job-name=train_obj_det

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate

if [ $1 = "dbf3" ]
then
    python3 tools/train.py "./configs/custom/$2_config.py" \
        --cfg-options   "work_dir=work_dirs/$1/$2" \
        --gpu-id=0  
else
    python3 tools/train.py  "./configs/custom/$2_config.py" \
            --cfg-options   "work_dir=work_dirs/$3/$2/$1/" \
                            "data.train.img_prefix=../dataset/xray/composed/$3/$1/" \
                            "data.train.ann_file=../dataset/xray/composed/$3/helper/annotation/$1.json" \
            --gpu-id=0  
fi