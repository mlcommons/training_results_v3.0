#!/bin/bash

set -x 

source config_R760xax4H100PCIE80GB.sh

export DATADIR=/mnt/data1/training_ds/ilsvrc12_passthrough
export LOGDIR=/home/rakshith/mlperf_training_3.0/resnet/20230428/scripts/result_R760xax4xH100
CONT=29bef77babc6 ./run_with_docker.sh
