#!/bin/bash 
cd ../mxnet
docker build --pull -t mlperf_trainingv3.0-gigabyte:resnet-20230428 .
