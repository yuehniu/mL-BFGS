#!/usr/bin/env bash
NPROC_PER_NODE=$1
MASTER_ADDR=$2
MASTER_PORT=$3
MODEL=$4
DATA=$5
LR=$6
WD=$7
MOMENTUM=$8
EPOCH=$9
GPU=${10}
MM=${11}
MMM=${12}
FREQ=${13}
# example
# bash scsript     #gpus                               lr   wd    mom  epoch gpu          mmp/g mmm updatefreq
# bash run-SLIM.sh 1 127.0.0.1 11113 resnet18 cifar100 0.1 0.0002 0.9  150   4            0.999 0.9 50
# bash run-SLIM.sh 1 127.0.0.1 11113 deit     cifar100 0.1 0.0005 0.9  50    1            0.999 0.9 50
# bash run-SLIM.sh 4 127.0.0.1 11113 resnet50 imagenet 0.1 0.0002 0.9  100   "0,1,2,3"    0.999 0.9 50

moreargs=" "
PRINT=100
if [ "${DATA}" == "imagenet" ]; then
  moreargs+="  --distributed --fp16"
  PRINT=500
fi

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./train.py ${moreargs} \
    --model=$MODEL \
    --dataset=$DATA \
    --data=./data/${DATA} \
    --optimizer=SLIMBLOCK \
    --update-freq=${FREQ} \
    --mm-p=${MM} \
    --mm-g=${MM} \
    --mmm=${MMM} \
    --hist-sz=10 \
    --lbfgs-damping=0.01 \
    --grad-clip=1 \
    --max-epoch=${EPOCH} \
    --lr=${LR} \
    --wd=${WD} \
    --momentum=${MOMENTUM} \
    --workers=4 \
    --print-freq=$PRINT \
    --logdir=log/${DATA}-${MODEL}/mLBFGS-blocks/1blocks-lr${LR}-wd${WD}+-damp0.01-upd${FREQ}-hist10-mm${MM}-${MM}-${MMM}-refresh-g \
    --init-bn0 \
    --phases "[{'ep': 0, 'sz': 32, 'bs': 128},
    {'ep': (0, 5), 'lr': (0.02, 0.2)},
    {'ep': (5, 150), 'lr': (0.2, 0.2)},
    {'ep': (150, 225), 'lr': (0.02, 0.02)}]"