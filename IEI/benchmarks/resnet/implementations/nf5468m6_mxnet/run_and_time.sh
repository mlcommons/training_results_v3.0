#!/bin/bash

cd ../mxnet
source config_NF5468M6.sh
DGXSYSTEM="NF5468M6" CONT=mlperf-inspur:resnet DATADIR=/path/to/preprocessed/data LOGDIR=/path/to/logfile ./run_with_docker.sh
