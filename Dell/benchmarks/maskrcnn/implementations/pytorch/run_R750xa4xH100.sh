#!/bin/bash
set -x 

source config_R750xax4xH100.sh

export PKLPATH=/mnt/data/training_ds/coco2017/pickled
export DATADIR=/mnt/data/training_ds

CONT=815746369624  LOGDIR=/home/rakshith/mlperf_training_3.0/maskrcnn/scripts/results_R750xa4xH100 ./run_with_docker.sh
