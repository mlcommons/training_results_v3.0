#!/bin/bash 
cd ../mxnet
docker build --pull -t mlperf_trainingv3.0-gigabyte:unet3d-20230428 .
