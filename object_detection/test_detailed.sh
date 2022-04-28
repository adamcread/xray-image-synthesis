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
    --model_name "cascade_rcnn" \
    "./configs/custom/crcnn_config.py" \
    "./work_dirs/dbf3/crcnn/best.pth" \
    --db "../dataset/" \
    --format-only \
    --outfile "test.png" \
    --predfile "test.bbox.json" \
    --outcsv "test.csv" \
    --dbpath "../dataset/xray/unpaired/resized_128x128/composed_images/" \
    --testimg "../dataset/xray/unpaired/resized_128x128/composed_images/" \
    --testgt "../dataset/xray/unpaired/helper/annotation/dbf3_test_resized.json"
 


# python test.py \
#     --model_name fcos \
#     configs/detr/detr_r50_8x2_150e_coco.py \
#     logs/MultispectralRGB5/detr_r50_stdaug-coco/epoch_140.pth \
#     --db MultispectralRGB5 \
#     --dbpath /home/neel/data/cnn_model/dataset \
#     --format-only \
#     --eval-options "jsonfile_prefix=./statistics/MultispectralRGB5/detr_r50_stdaug-coco/test_results_e140" \
#     --statpath statistics/MultispectralRGB5/detr_r50_stdaug-coco \
#     --stats \
#     --predfile statistics/MultispectralRGB5/detr_r50_stdaug-coco/test_results_e140.bbox.json \
#     --outfile statistics/MultispectralRGB5/detr_r50_stdaug-coco/confmat_bbox_e140.png \
#     --outcsv statistics/MultispectralRGB5/detr_r50_stdaug-coco/confmat_bbox_e140.csv \



