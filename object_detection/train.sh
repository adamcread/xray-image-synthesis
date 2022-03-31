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
#SBATCH -o 'train_obj_det_$1.txt'

# job name
#SBATCH --job-name=train_obj_det

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate

if [ $1 = "dbf3_crcnn" ] || [ $1 = "dbf3_fsaf" ]
then
    python3 tools/train.py './configs/custom/cascade_rcnn_config.py' \
        --cfg-options   "work_dir=work_dirs/train/$1" \
        --gpu-id=0  
else
    python3 tools/train.py './configs/custom/cascade_rcnn_config.py' \
            --cfg-options   "work_dir=work_dirs/train/$1" \
                            "data.train.img_prefix=../dataset/xray/composed/$2/$1/" \
                            "data.train.ann_file=../dataset/xray/composed/$2/helper/annotation/$1.json" \
            --gpu-id=0  
fi

