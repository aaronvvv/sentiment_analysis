set -eux

#export BATCH_SIZE=16
#export LR=2e-5
#export EPOCH=20
#export CUDA_VISIBLE_DEVICES=0
unset CUDA_VISIBLE_DEVICES

python  -m paddle.distributed.launch --gpus "0,1" predict_opinion.py --predict_data "COTE_BD" --param_num 0
python  -m paddle.distributed.launch --gpus "0,1" predict_opinion.py --predict_data "COTE_BD" --param_num 1
python  -m paddle.distributed.launch --gpus "0,1" predict_opinion.py --predict_data "COTE_BD" --param_num 2
python  -m paddle.distributed.launch --gpus "0,1" predict_opinion.py --predict_data "COTE_BD" --param_num 3
python  -m paddle.distributed.launch --gpus "0,1" predict_opinion.py --predict_data "COTE_BD" --param_num 4