#!/bin/bash 
cd ../pytorch
docker build --pull -t mlperf_trainingv3.0-gigabyte:retinanet-20230428 .
