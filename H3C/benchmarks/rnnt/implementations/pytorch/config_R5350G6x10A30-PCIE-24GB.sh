# Single-node config (since v2.0)

#export DWU_GROUP_SIZE=$DGXNGPU

## Run specific params
export DATADIR="/raid0/datasets/rnnt/"
export METADATA_DIR="/raid0/datasets/rnnt/metadata/"
export SENTENCEPIECES_DIR="/raid0/datasets/rnnt/sentencepieces/"

export GRAD_ACCUMULATION_STEPS=2

## Opt flag

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_single_node.sh
#source $(dirname ${BASH_SOURCE[0]})/config_data_selene.sh

export BATCHSIZE=256
export EVAL_BATCHSIZE=194
export BATCH_SPLIT_FACTOR=4
export AUDIO_RESAMPLING_DEVICE=gpu

export DELAY_ENCODER=true

source $(dirname ${BASH_SOURCE[0]})/A30_hyperparameters_512.sh