#!/bin/bash

cd ../mxnet
source ./config_R4900G6x2A30-PCIE-24GB.sh
CONT=mlperf-H3C:resnet DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh