# Single-node Hopper config (since v2.1)

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
: ${SINGLE_EXP_WALLTIME:=35}
source $(dirname ${BASH_SOURCE[0]})/config_common_single_node.sh

#export BATCHSIZE=192
export BATCHSIZE=384
#export EVAL_BATCHSIZE=338
export EVAL_BATCHSIZE=676
export GRAD_ACCUMULATION_STEPS=2

export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh
