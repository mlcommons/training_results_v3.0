#!/bin/bash

set -x 

export DATADIR="/mnt/data/training_ds/rnnt_new"
export METADATA_DIR="/mnt/data/training_ds/rnnt_new/metadatadir/"
export SENTENCEPIECES_DIR="/mnt/data/training_ds/rnnt_new/sentencepieces/"

source config_R750xa4xH100PCIE80GB.sh


CONT=ce31cc506dad ./run_with_docker.sh
