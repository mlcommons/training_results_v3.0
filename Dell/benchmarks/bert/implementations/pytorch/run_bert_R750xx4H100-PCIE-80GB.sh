#!/bin/bash

set -x 


source config_R750xax4H100-PCIE-80GB.sh

#export NEXP=10
export NEXP=12
export CUDA_VISIBLE_DEVICES=0,1,2,3

DGXSYSTEM=R750xax4H100-PCIE-80GB CONT=cab1dc18263c   ./run_with_docker.sh


