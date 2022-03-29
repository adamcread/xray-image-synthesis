mode=test
name="test"

datalist="./scripts/unpaired/threat_mask/train_comb_paths.txt"
datalist_test="./scripts/unpaired/test.txt"

exp_name="test_unpaired_combined_focal_tversky_700"
name_train="test_unpaired_combined_focal_tversky_700"
model_train="./checkpoints/${name_train}" 

name="${exp_name}"
dataset_mode='comp_decomp_unaligned'
how_many=640
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


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

cp ${model_train}/*.pth ./checkpoints/${name}/

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} \
	python3 test_composition.py \
	--datalist ${datalist} \
	--datalist_test ${datalist_test} \
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

