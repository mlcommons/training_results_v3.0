#!/bin/bash
export SBATCH_NETWORK=sharp

## DL params
export BATCHSIZE=${BATCHSIZE:-16}
export NUMEPOCHS=${NUMEPOCHS:-6}
export LR=${LR:-0.0001}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-backbone-fusion --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --skip-metric-loss --cuda-graphs-syn --async-coco --enable-sharp --master-weights'}

## System run params
export DGXNNODES=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=40
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
export DATADIR="/home/dataset/ssd/open-images-v6-mlperf"
export BACKBONE_DIR="/home/dataset/ssd"
export CONT="/root/enroot_img/root+nvcr.io+nvdlfwea+mlperfv30+ssd+20230428.pytorch.sqsh"
