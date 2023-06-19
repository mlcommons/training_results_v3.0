## DL params
export CONFIG="train.py"
export BATCHSIZE=135168
export BATCHSIZE_EVAL=1048576
export LEARNING_RATE=0.0034
export USE_MIXED_PRECISION=true
export SCALER=20480
export SHARDING_PLAN=hier_auto
export MEM_COMM_BW_RATIO=67
export GEN_LOSS_SUMMARY=true

## System run parms
export DGXNNODES=8
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=$(( 5 + ${NEXP:-1} * 10 ))

## network flags
export SBATCH_NETWORK=sharp
export NCCL_COLLNET_ENABLE=1
