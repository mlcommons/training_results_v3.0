#!/bin/bash
set -x 
source config_R750xax4H100-PCIE-80GB.sh

LOGDIR=`pwd`/results_node049
CONT=a4a623e9eb25 ./run_with_docker.sh

