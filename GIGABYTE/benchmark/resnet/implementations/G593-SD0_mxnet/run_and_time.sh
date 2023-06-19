#!/bin/bash
cd ../mxnet
source config_G593-SD0.sh
export CONT=mlperf_trainingv3.0-gigabyte:resnet-20230428
export DATADIR=/path/to/dataset
export LOGDIR=/path/to/logdir
./run_with_docker.sh
