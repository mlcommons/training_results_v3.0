#!/bin/bash

num_of_run=5

source config_PG_10gpu.sh
logdir=$(realpath ../logs-resnet)

for idx in $(seq 1 $num_of_run); do
    NEXP=1 CONT=nvcr.io/nvdlfwea/mlperfv30/resnet:20230428.mxnet DATADIR=/mnt/data4/work/forMXNet_no_resize  \
        LOGDIR=$logdir DGXSYSTEM=PG_10gpu PULL=0 ./run_with_docker.sh
done


