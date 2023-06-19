#!/bin/bash
cd ../pytorch
source config_G593-SD0_1x8x48x1_pack.sh
export CONT=mlperf_trainingv3.0-gigabyte:bert-20230428
export LOGDIR=/path/to/logdir
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATADIR=/path/to/dataset
export EVALDIR=/path/to/evadir
export CHECKPOINTDIR=/path/to/checkpoingdir
export CHECKPOINTDIR_PHASE1=/path/to/checkpoint_phase1
export UNITTESTDIR=/path/to/unittestdir
export DATADIR_PHASE2=/path/to/datadir_phase2
./run_with_docker.sh
