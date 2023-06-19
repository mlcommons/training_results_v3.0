#!/bin/bash

set -x 
export NEXP=40
source config_R750xax4xH100.sh
DGXSYSTEM=R750xax4xH100 CONT=eab9278fa424 DATADIR=/mnt/data/training_ds/unet3d/ LOGDIR=results_R750xax4H100 ./run_with_docker.sh

