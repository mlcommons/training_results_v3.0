## DL params
export CONFIG="train.py"
export BATCHSIZE=55296
export BATCHSIZE_EVAL=262144
export LEARNING_RATE=0.004
export USE_MIXED_PRECISION=true
export SCALER=16348
export SHARDING_PLAN=auto
export MEM_COMM_BW_RATIO=7
export GEN_LOSS_SUMMARY=true

## System run parms
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=$(( 5 + ${NEXP:-1} * 15 ))
