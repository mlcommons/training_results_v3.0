## DL params

## System run parms

DGXNGPU="${DGXNGPU:-8}"
export CUDA_VISIBLE_DEVICES=0,4,2,6,1,5,3,7

export MOUNTS="$PREPROC_DATA:/preproc_data,${SPM}:/workspace/llm/tokenizer.model,${LOAD_CHECKPOINTS_PATH}:/load_checkpoints"

# Data blend - mixing fractions of input
COM_DIR="/preproc_data"
C4_6="${COM_DIR}/c4_en_6_c4_spm_text_document"
C4_7="${COM_DIR}/c4_en_7_c4_spm_text_document"
export BLEND="[0.5,${C4_6},0.5,${C4_7}]"
export VALID_C4="${COM_DIR}/c4_en_validation_subset_c4_spm_text_document"

# Model architecture
export NUM_LAYERS=96
export HIDDEN_SIZE=12288
export FFN_HIDDEN_SIZE=49152
export NUM_ATTENTION_HEADS=96

# DL Params
export MICRO_BATCH_SIZE=2
export INTERLEAVED_PIPELINE=12
export USE_DIST_OPTIMIZER=True
# This is to improve p2p overlap on H100, shouldn't affect A100:
export NVTE_FWD_LAYERNORM_SM_MARGIN=4
export NVTE_BWD_LAYERNORM_SM_MARGIN=4

export NCCL_AVOID_RECORD_STREAMS=1
export NCCL_IB_SL=1
export NCCL_MIN_NCHANNELS=4
export ACT_CKPT_GRANULARITY=null
export ACT_CKPT_NUM_LAYERS=null
export ACT_CKPT_METHOD=null

MODEL_PARALLEL=$((TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL))
WORLD_SIZE=$((DGXNNODES * DGXNGPU))
DATA_PARALLEL=$((WORLD_SIZE / MODEL_PARALLEL))
export GLOBAL_BATCH_SIZE=$((MINIBS * DATA_PARALLEL))

export TP_COMM_OVERLAP=False
export LOG_EVERY_N_STEPS=1
export LIMIT_TRAIN_BATCHES=500
export LIMIT_VAL_BATCHES=1.
export TARGET_LOG_PPL=2.69
# set LIMIT_TEST_BATCHES>0 so NeMo r1.11.0 does not error out
# test will be skipped so this does not matter
export LIMIT_TEST_BATCHES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export LOAD_CHECKPOINT=/load_checkpoints/ckpt4000-consumed_samples=0


## GBS dependent hyperparameters
: "${PROXY_GBS:=$GLOBAL_BATCH_SIZE}"
[[ ${PROXY_GBS} -ne ${GLOBAL_BATCH_SIZE} ]] && IS_PROXY_RUN=1 || IS_PROXY_RUN=0
if [[ ${IS_PROXY_RUN} -eq 1 ]]; then
  echo "Setting hyperparameters for proxy GBS ${PROXY_GBS} (actual GBS is ${GLOBAL_BATCH_SIZE})"
else
  echo "Setting hyperparameters for GBS ${PROXY_GBS}"
fi

# max_steps and warmup_steps are computed from the number of samples using the global batch size according to the reference hyperparams
#export MAX_STEPS_FOR_LR_SCHED=$(( ( 166809600 + PROXY_GBS - 1 ) / PROXY_GBS ))
#export WARMUP_STEPS=$(( ( 407040 + PROXY_GBS - 1 ) / PROXY_GBS ))
# exact floating point version:
export MAX_STEPS_FOR_LR_SCHED=$(printf %.4f\\n "$((10000000 * 166809600 / ${PROXY_GBS} ))e-7")
export WARMUP_STEPS=$(printf %.4f\\n "$((10000000 * 407040 / ${PROXY_GBS} ))e-7")

# if LOAD_MINIMAL_NUM_SAMPLES is 1, we load as little samples as possible (for 500 steps).
# Otherwise, we have to load initial 6144000 samples (used to generate the checkpoint)
# and a lot of future samples to ensure exactly the same data order as in Megatron.
# NOTE: with LOAD_MINIMAL_NUM_SAMPLES=1, LR schedule will start from scratch (LR warmup)
: "${LOAD_MINIMAL_NUM_SAMPLES:=$IS_PROXY_RUN}"

# MAX_STEPS makes sure that we don't repeat dataset samples
export MAX_STEPS=$(( ( 20000000 + PROXY_GBS - 1 ) / PROXY_GBS ))
export INIT_GLOBAL_STEP=$(( ( 4000 * 1536 + PROXY_GBS - 1 ) / PROXY_GBS ))
[[ $(( INIT_GLOBAL_STEP * PROXY_GBS )) == $(( 4000 * 1536 )) ]] || echo "Warning: $(( 4000 * 1536 )) not divisible by GBS ${PROXY_GBS}"

if [[ "${LOAD_MINIMAL_NUM_SAMPLES}" -eq 1 ]]; then
  [[ ${IS_PROXY_RUN} -eq 0 ]] && echo "WARNING: LOAD_MINIMAL_NUM_SAMPLES is on (breaking LR schedule and seed-to-seed equality with Megatron)"
  export MAX_STEPS=500
  export OVERRIDE_ZERO_CONSUMED_SAMPLES=0
  export INIT_GLOBAL_STEP=0
fi

export VAL_CHECK_INTERVAL=$(( ( 24576 + PROXY_GBS - 1 ) / PROXY_GBS ))
[[ $(( $VAL_CHECK_INTERVAL * PROXY_GBS )) == 24576 ]] || echo "Warning: 24576 not divisible by GBS ${PROXY_GBS}"

if [[ ${IS_PROXY_RUN} -eq 1 ]]; then
  export LIMIT_VAL_BATCHES=$(( ( 5662 + PROXY_GBS - 1 ) / PROXY_GBS ))  # instead of full valid (1.0)
fi

if [[ ${PROXY_GBS} -lt 3500 ]]; then
  export LR=2e-05
  export MIN_LR=2e-06
else
  export LR=3e-05
  export MIN_LR=3e-06
fi
