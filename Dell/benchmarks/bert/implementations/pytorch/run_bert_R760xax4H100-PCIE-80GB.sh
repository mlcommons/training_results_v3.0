#!/bin/bash

set -x 


source config_R760xax4H100-PCIE-80GB_common.sh
source config_R760xax4H100-PCIE-80GB_bckp.sh

export NEXP=10
#export NEXP=12
export CUDA_VISIBLE_DEVICES=0,1,2,3

export LOGDIR=/home/rakshith/mlperf_training_3.0/bert/scripts/results_R760xax4H100
DGXSYSTEM=R760xax4H100-PCIE-80GB CONT=cab1dc18263c   ./run_with_docker.sh


