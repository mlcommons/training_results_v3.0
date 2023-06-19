#!/bin/bash

set -x 
export NEXP=40
source config_R760xax4xH100.sh
DGXSYSTEM=R760xax4xH100 CONT=eab9278fa424 DATADIR=/mnt/data/training_ds/unet3d/ LOGDIR=results_R760xax4H100 ./run_with_docker.sh

