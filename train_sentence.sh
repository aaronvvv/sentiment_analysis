set -eux

#export BATCH_SIZE=16
#export LR=2e-5
#export EPOCH=20
#export CUDA_VISIBLE_DEVICES=0
export NFOLD=5
export TRAIN_DATA=NLPCC14-SC
unset CUDA_VISIBLE_DEVICES
#for((i=0;i<NFOLD;i++)); do python -m paddle.distributed.launch --gpus "0,1" train_sentence.py --fold_index $[i] --nfold 5;done
#for((i=0;i<NFOLD;i++)); do python -m paddle.distributed.launch --gpus "0,1" train_sentence2.py --fold_index $[i] --nfold 5;done
python -m paddle.distributed.launch --gpus "0" train_sentence_dev.py 