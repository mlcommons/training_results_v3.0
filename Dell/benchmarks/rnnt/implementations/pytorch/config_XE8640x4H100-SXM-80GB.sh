source $(dirname ${BASH_SOURCE[0]})/config_common_XE8640.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_XE8640_benchmark.sh
: ${SINGLE_EXP_WALLTIME:=25}
source $(dirname ${BASH_SOURCE[0]})/config_common_XE8640_single_node.sh

export BATCHSIZE=256
export EVAL_BATCHSIZE=338
export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh

export BATCH_SPLIT_FACTOR=2
export NCCL_SOCKET_IFNAME=eno
export NCCL_MAX_RINGS=4