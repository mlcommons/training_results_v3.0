#!/bin/bash
set -x 

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}




## Data Paths
export DATADIR="/mnt/data1/training_ds/bert3.0/hdf5/training-4320/hdf5_4320_shards_varlength"
export DATADIR_PHASE2="/mnt/data1/training_ds/bert3.0/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/mnt/data1/training_ds/bert3.0/hdf5/eval_varlength/"
export CHECKPOINTDIR="./ci_checkpoints"
export RESULTSDIR="./results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/mnt/data1/training_ds/bert3.0/phase1"
#export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"


DATADIR_PHASE2_PACKED="/mnt/data1/training_ds/bert3.0/packed_data/"
export NCCL_SOCKET_IFNAME=
