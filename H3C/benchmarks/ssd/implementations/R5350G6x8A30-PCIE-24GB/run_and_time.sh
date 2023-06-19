#!/bin/bash

cd ../mxnet
source ./config_R5350G6x8A30-PCIE-24GB.sh
CONT=mlperf-H3C:ssd DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR BACKBONE_DIR=/PATH/TO/BACKBONE_DIR ./run_with_docker.sh