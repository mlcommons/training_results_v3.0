# Single-node Hopper config (since v2.1)

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx_1.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
: ${SINGLE_EXP_WALLTIME:=25}
source $(dirname ${BASH_SOURCE[0]})/config_common_single_node.sh

export BATCHSIZE=64
export EVAL_BATCHSIZE=192
export DATA_CPU_THREADS=8
export CG_UNROLL_FACTOR=8
export BATCH_SPLIT_FACTOR=8
export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true


source $(dirname ${BASH_SOURCE[0]})/hyperparameters_512.sh
