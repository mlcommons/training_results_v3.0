#!/bin/bash
set -x 
source config_R760xax4H100-PCIE-80GB.sh

export DATADIR=/mnt/data1/training_ds/openimages_ds/open-images-v6-mlperf
export LOGDIR=`pwd`/results_R760xax4xH100
CONT=9e2d9ebb9d29 ./run_with_docker.sh

