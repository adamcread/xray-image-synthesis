#!/bin/bash

# X number of nodes with Y number of cores in each node.
#SBATCH -N 1
#SBATCH -c 4

# partition time limit and resource limit for the job
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small	
#SBATCH --mem=28g
#SBATCH --qos=long-low-prio
#SBATCH -t 07-00:00:00

# job name
#SBATCH --job-name=unpaired_combined

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate

mode=train
name="train_xray_combined"

datalist="../dataset/xray/unpaired/train_paths_combined.txt"
datalist_test="../dataset/xray/unpaired/test_paths_combined.txt"

exp_name="unpaired_compGAN" # suffix for naming
dataset_mode='comp_decomp_unaligned' # dataset type to choose model type

batch_size=8 # size of each training batch
loadSizeY=256 # size to scale images to
fineSizeY=256 # size image is
loadSizeX=256 # size to scale images to
fineSizeX=256 # size image is

G1_comp=1 # completion on object 1	
G2_comp=0 # completion on object 2
STN_model='deep'
lambda_mask=50
lr=0.00002 # initial learning rate
niter=1000
niter_decay=1000

niterSTN=500
niterCompletion=500

which_epoch=0
which_epoch_completion=0 
which_epoch_STN=0

display_port=8775
display_freq=550
print_freq=30
update_html_freq=550
save_epoch_freq=50
CUDA_ID=0

xray=true

# if checkpoint directory doesn't exist make one
if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

# if log already exists delete it
LOG="./checkpoints/${name}/output.txt"
if [ -f $LOG ]; then
    rm $LOG
fi

# make log
exec &> >(tee -a "$LOG")

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} \
	python3 -u train_composition.py \
	--datalist ${datalist} \
	--datalist_test ${datalist_test} \
	--decomp \
	--name ${name} \
	--dataset_mode ${dataset_mode} \
	--xray ${xray} \
	--no_lsgan \
	--conditional \
	--img_completion \
	--no_flip \
	--pool_size 0 \
	--niterCompletion ${niterCompletion} \
	--which_epoch_completion ${which_epoch_completion} \
	--fineSizeY ${fineSizeY} \
	--loadSizeY ${loadSizeY} \
	--fineSizeX ${fineSizeX} \
	--loadSizeX ${loadSizeX} \
	--G1_completion ${G1_comp} \
	--G2_completion ${G2_comp} \
	--lambda_mask ${lambda_mask} \
	--lr ${lr} \
	--batchSize ${batch_size} \
	--niterSTN ${niterSTN} \
	--STN_model ${STN_model} \
	--which_epoch_STN ${which_epoch_STN} \
	--niter ${niter} \
	--niter_decay ${niter_decay} \
	--which_epoch ${which_epoch} \
	--epoch_count ${which_epoch} \
	--update_html_freq ${update_html_freq} \
	--display_freq ${display_freq} \
	--print_freq ${print_freq} \
	--display_port ${display_port} \
	--save_epoch_freq ${save_epoch_freq} \

 