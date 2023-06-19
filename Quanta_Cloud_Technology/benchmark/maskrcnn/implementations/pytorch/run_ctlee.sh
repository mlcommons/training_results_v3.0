#!/bin/bash
source config_D74H-7U.sh
DATE=$(date +"%Y%m%d_%H%M%S")
CONT=nvcr.io/nvdlfwea/mlperfv30/maskrcnn:20230428.pytorch DATADIR=/mlperf_data/ctlee_data/maskrcnn/data PKLPATH=/mlperf_data/ctlee_data/maskrcnn/data/coco2017/pkl_coco LOGDIR=/mlperf_data/ctlee_data/maskrcnn/run_train/Result COCOPYTDIR=/mlperf_data/ctlee_data/maskrcnn/data/coco_train2017_pyt ./run_with_docker.sh 2>&1 | tee run_train-${DATE}.log
