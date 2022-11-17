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
#SBATCH --job-name=generate_imagery
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate

mode=test
name="test"

datalist_test="./scripts/paths_test.txt"

if [ $1 = "paired" ]
then
	dataset_mode="comp_decomp_aligned"
	datalist="./scripts/paths_train_paired.txt"
elif [ $1 = "unpaired" ]
then
	dataset_mode="comp_decomp_unaligned" # dataset type to choose model type
	datalist="./scripts/paths_train_unpaired.txt"
fi

exp_train="train_"$1"_"$2
name="test_"$1"_"$2
model_train="./checkpoints/${exp_train}/"

how_many=$3
batch_size=1

loadSizeY=128
fineSizeY=128
loadSizeX=128
fineSizeX=128
lambda_L2=500
G1_comp=1
G2_comp=0
thresh1=0.99
thresh2=0.99
STN_model='deep'
niter=1
niter_decay=1
LR=0.00005
EPOCH="best"
display_port=8775
display_freq=1
print_freq=20
update_html_freq=5
n_latest=1
xray=true



if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

for epoch in {0..1000..50}
do
	python3 move_checkpoints.py \
		-i ${model_train} \
		-e ${epoch} \
		-d "./checkpoints/${name}/"


	CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} \
		python3 test_composition.py \
		--datalist ${datalist} \
		--datalist_test ${datalist_test} \
		--xray ${xray} \
		--decomp \
		--name ${name} \
		--dataset_mode ${dataset_mode} \
		--no_flip \
		--conditional \
		--erosion \
		--img_completion \
		--lr ${LR} \
		--lambda_L2 ${lambda_L2} \
		--batchSize ${batch_size} \
		--niter ${niter} \
		--niter_decay ${niter_decay} \
		--fineSizeY ${fineSizeY} \
		--loadSizeY ${loadSizeY} \
		--STN_model ${STN_model} \
		--which_epoch ${EPOCH} \
		--Thresh1 ${thresh1} \
		--Thresh2 ${thresh2} \
		--G1_completion ${G1_comp} \
		--G2_completion ${G2_comp} \
		--display_freq ${display_freq} \
		--display_id 1 \
		--display_port ${display_port} \
		--print_freq ${print_freq} \
		--update_html_freq ${update_html_freq} \
		--n_latest ${n_latest} \
		--how_many ${how_many} \
		--eval


	rm -rf "./results/${name}/test_"${epoch}"/"
	cp -R "./results/${name}/test_best/" "./results/${name}/test_"${epoch}"/"

	python3 composed_dataset.py \
		--file="${name}" \
		--test="test_"${epoch}

	python3 ../object_detection/tools/test.py \
		"../object_detection/configs/custom/crcnn_config.py" \
		"../object_detection/work_dirs/dbf3/crcnn/best.pth" \
		--eval "bbox" \
		--cfg-options   "classwise=True" \
						"work_dir=../object_detection/work_dirs/test/" \
						"data.test.img_prefix=./results/${name}/test_${epoch}/images/" \
						"data.test.ann_file=./results/${name}/annotation/test_${epoch}.json"
done


 
