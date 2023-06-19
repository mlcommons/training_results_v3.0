#!/bin/bash

## DL params
export BATCHSIZE=${BATCHSIZE:-16}
export NUMEPOCHS=${NUMEPOCHS:-10}
export LR=${LR:-1e-04}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-backbone-fusion --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --skip-metric-loss --cuda-graphs-syn --async-coco --dali-cpu-decode --master-weights'}

## System run params
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=160
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=10
export DGXSOCKETCORES=48  #64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
