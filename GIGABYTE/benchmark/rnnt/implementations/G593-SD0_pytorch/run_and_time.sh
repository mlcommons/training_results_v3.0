#!/bin/bash
cd ../pytorch
source config_G593-SD0.sh
export CONT=mlperf_trainingv3.0-gigabyte:rnnt-20230428
export LOGDIR=/path/to/logdir
export DATADIR=/path/to/dataset
export METADATA_DIR=/path/to/METADATA_DIR
export SENTENCEPIECES_DIR=/path/to/SENTENCEPIECES_DIR

./run_with_docker.sh
