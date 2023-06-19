#!/bin/bash
cd ../hugectr
source config_G593-SD0.sh
export CONT=mlperf_trainingv3.0-gigabyte:dlrm-20230428
export LOGDIR=/path/to/logdir
export DATADIR=/path/to/dataset
./run_with_docker.sh
