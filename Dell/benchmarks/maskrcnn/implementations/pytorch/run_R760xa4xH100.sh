#!/bin/bash
set -x 

source config_R760xax4xH100.sh

#export PKLPATH=/mnt/data1/training_ds/coco2017/pickled
#export DATADIR=/mnt/data1/training_ds
export PKLPATH=/mnt/data/training_ds/coco2017/pickled
export DATADIR=/mnt/data/training_ds

CONT=815746369624  LOGDIR=/home/rakshith/mlperf_training_3.0/maskrcnn/scripts/results_R760xa4xH100 ./run_with_docker.sh

#CONT=4927e4a625bd  LOGDIR=/home/rakshith/mlperf_training_3.0/maskrcnn/scripts/results_R760xa4xH100 ./run_with_docker.sh # binding 
