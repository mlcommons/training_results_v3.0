## DL params
export MINIBS=64
export TENSOR_MODEL_PARALLEL=4   #  training.model.tensor_model_parallel_size
export PIPELINE_MODEL_PARALLEL=8 #  training.model.pipeline_model_parallel_size
export DGXNNODES=192
#=======================================================================
## System run parms
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# keep the format days-huour:minutes:seconds -> PyT Lightning parsing
export WALLTIME=00-02:00:00 #$(( WALLTIME_BASE + (${NEXP:-1} * WALLTIME_MINUTES) ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_fp8.sh
export MICRO_BATCH_SIZE=1
source $(dirname ${BASH_SOURCE[0]})/config_tp_h100.sh
