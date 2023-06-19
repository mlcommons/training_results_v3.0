#!/bin/bash
source config_DGXH100_1x8x192x1.sh
DATE=$(date +"%Y%m%d_%H%M%S")
NEXP=10 CONT=nvcr.io/nvdlfwea/mlperfv30/rnnt:20230428.pytorch DATADIR=/mlperf_data/ctlee_data/rnnt/data/datasets LOGDIR=/mlperf_data/ctlee_data/rnnt/run_training/Result METADATA_DIR=/mlperf_data/ctlee_data/rnnt/data/tokenized SENTENCEPIECES_DIR=/mlperf_data/ctlee_data/rnnt/data/sentencepieces bash ./run_with_docker.sh 2>&1 | tee run_train-${DATE}.log
#NEXP=10 CONT=mlperf/rnn_speech_recognition DATADIR=/mlperf_data/ctlee_data/rnnt/data/datasets LOGDIR=/mlperf_data/ctlee_data/rnnt/run_training/Result METADATA_DIR=/mlperf_data/ctlee_data/rnnt/data/tokenized SENTENCEPIECES_DIR=/mlperf_data/ctlee_data/rnnt/data/sentencepieces bash ./run_with_docker.sh 2>&1 | tee run_train-${DATE}.log
