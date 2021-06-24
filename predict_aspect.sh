set -eux

#export BATCH_SIZE=16
#export LR=2e-5
#export EPOCH=20
#export CUDA_VISIBLE_DEVICES=0
export NFOLD=5
export PRED_DATA=SE-ABSA16_PHNS
unset CUDA_VISIBLE_DEVICES
for((i=0;i<NFOLD;i++)); do python -m paddle.distributed.launch --gpus "0,1" predict_aspect.py --predict_data $PRED_DATA --param_num $[i];done