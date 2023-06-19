#!/bin/bash 
cd ../pytorch
docker build --pull -t mlperf_trainingv3.0-gigabyte:rnnt-20230428 .
