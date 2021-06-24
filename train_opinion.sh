set -eux

#export BATCH_SIZE=16
#export LR=2e-5
#export EPOCH=20
#export CUDA_VISIBLE_DEVICES=0
unset CUDA_VISIBLE_DEVICES

# python -m paddle.distributed.launch --gpus "0,1" train_opinion.py \
#                             --fold_index 0 \
#                             --nfold 5
# python -m paddle.distributed.launch --gpus "0,1" train_opinion.py \
#                             --fold_index 1 \
#                             --nfold 5
# python -m paddle.distributed.launch --gpus "0,1" train_opinion.py \
#                             --fold_index 2 \
#                             --nfold 5
# python -m paddle.distributed.launch --gpus "0,1" train_opinion.py \
#                             --fold_index 3 \
#                             --nfold 5
# python -m paddle.distributed.launch --gpus "0,1" train_opinion__.py \
#                             --fold_index 4 \
#                             --nfold 5
python -m paddle.distributed.launch --gpus "0,1" train_opinion.py \
                            --fold_index 1 \
                            --nfold 5
# python -m paddle.distributed.launch --gpus "0,1" train_opinion.py \
#                             --fold_index 2 \
#                             --nfold 5