#!/bin/bash

set -x 

source config_R750xax4H100PCIE80GB.sh

export NEXP=8
export LOGDIR=/home/rakshith/mlperf_training_3.0/resnet/20230428/scripts/result_R750xax4xH100

CONT=0a896854ec1a ./run_with_docker.sh
