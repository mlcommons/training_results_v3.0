#!/bin/bash

set -x 

export NEXP=10	
source config_R750x4xaH100.sh
DGXSYSTEM=R750x4xaH100  CONT=2d1c8fae0241 DATADIR=/mnt/data1/criteo_1tb_multihot_raw/ ./run_with_docker.sh 
