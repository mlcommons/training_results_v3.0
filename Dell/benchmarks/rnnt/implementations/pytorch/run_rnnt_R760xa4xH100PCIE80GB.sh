#!/bin/bash

set -x 

export DATADIR="/mnt/data1/training_ds/rnnt_new"
export METADATA_DIR="/mnt/data1/training_ds/rnnt_new/metadatadir/"
export SENTENCEPIECES_DIR="/mnt/data1/training_ds/rnnt_new/sentencepieces/"

source config_R760xa4xH100PCIE80GB.sh

NEXP=10
CONT=bcc177d89d7a ./run_with_docker_r760xa.sh
