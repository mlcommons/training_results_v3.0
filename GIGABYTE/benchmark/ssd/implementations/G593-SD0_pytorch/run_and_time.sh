#!/bin/bash
cd ../pytorch
source config_G593-SD0.sh
export CONT=mlperf_trainingv3.0-gigabyte:retinanet-20230428
export DATADIR=/path/to/dataset
export LOGDIR=/path/to/logdir
export BACKBONE_DIR=/path/to/dataset/torch-home
./run_with_docker.sh
