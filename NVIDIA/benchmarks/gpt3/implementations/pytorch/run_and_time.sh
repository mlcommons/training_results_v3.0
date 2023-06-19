#!/bin/bash

# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
set -e

[ "${DEBUG}" = "1" ] && set -x

# Vars without defaults
: "${MICRO_BATCH_SIZE:?MICRO_BATCH_SIZE not set}"
: "${GLOBAL_BATCH_SIZE:?GLOBAL_BATCH_SIZE not set}"
: "${TENSOR_MODEL_PARALLEL:?TENSOR_MODEL_PARALLEL_SIZE not set}"
: "${PIPELINE_MODEL_PARALLEL:?PIPELINE_MODEL_PARALLEL_SIZE not set}"
: "${INTERLEAVED_PIPELINE:?INTERLEAVED_PIPELINE not set}"
: "${SEED:?SEED not set}"
: "${WALLTIME:=?WALLTIME not set}"

# Vars with defaults
: "${LOCAL_RANK:=${SLURM_LOCALID}}"
: "${LOGGER:=""}"
: "${MULTI_NODE:=''}"
: "${OMPI_COMM_WORLD_LOCAL_RANK:=""}"
: "${SLURM_JOB_ID:=$RANDOM}"
: "${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK}}"
: "${SLURM_NODEID:=0}"
: "${SLURM_NTASKS_PER_NODE:=$DGXNGPU}"
: "${THROUGHPUT_RUN:=""}"
: "${UNITTEST:=0}"

: "${NVTX_FLAG:=0}"
: "${TIME_TAGS:=0}"
: "${ACT_CKPT_GRANULARITY:="selective"}"
: "${ACT_CKPT_NUM_LAYERS:=1}"
: "${ACT_CKPT_METHOD:="uniform"}"
: "${SEQ_PARALLEL:=True}"
: "${OVERLAP_PARAM_SYNC:=True}"
: "${OVERLAP_GRAD_SYNC:=True}"
: "${TP_COMM_OVERLAP:=False}"
: "${TP_CONFIG_FILE:="h100tp4pp8mbs1_tp_comm_overlap_cfg.yaml"}"
: "${OVERLAP_P2P_COMM:=True}"
: "${BATCH_P2P_COMM:=False}"
: "${SYNC_BATCH_COMM:=False}"

: "${LOAD_CHECKPOINT:=""}"  # if set, training starts from this checkpoint (see SHARE_RERUNS effects below). Otherwise starts from scratch.
: "${ENABLE_RERUNS:=0}"  # enables saving and loading checkpoints for resuming training.
: "${SHARE_RERUNS:=0}"  # uses `shared_logs` results directory for all runs so that the checkpoints can be shared between different runs

echo "LOAD_CHECKPOINT=${LOAD_CHECKPOINT}"
# In order to share checkpoints between different runs (e.g. of a dryrun):
# 1) set ENABLE_RERUNS=1
# 2) set SHARE_RERUNS=1 so that the checkpoints subdirectory is the same for all runs
# 3) set the same LOGDIR in all runs
# 4) run training with `run.sub` or set NEMO_RESULTS_SUBDIR manually to a fixed value
# 5) run dependent slurm job
# 6) set the same SEED in all runs
# NOTE: a dryrun already meets 3) and 4) criteria.
# NOTE: if SHARE_RERUNS is set and an existing checkpoints directory is detected, LOAD_CHECKPOINT has no effect.
#       This is a convenience to avoid unsetting LOAD_CHECKPOINT when resuming from a further checkpoint
# NOTE: ENABLE_RERUNS and LOAD_CHECKPOINT variables are orthogonal, i.e. it makes sense to turn on or off ENABLE_RERUNS
# either when LOAD_CHECKPOINT is set (starting from iteration 4000) or not (starting from scratch).

: "${MIN_LR:=0.6e-5}"
: "${LR:=0.6e-4}"
: "${TARGET_LOG_PPL:=2.69}"

: "${TRANSFORMER_ENGINE:=True}"
: "${FP8:=False}"
: "${FP8_HYBRID:=False}"
: "${FP8_AMAX_HISTORY:=1}"
: "${FP8_AMAX_ALGO:="most_recent"}"
: "${FP8_REDUCE_AMAX:=True}"
: "${BUCKET_CAP_MB:=100}"
: "${USE_DIST_OPTIMIZER:=True}"
: "${USE_DIST_CHECKPOINTING:=1}"
: "${USE_TWO_STAGE_LOADING:=1}"
: "${USE_TWO_STAGE_CPU_TRANSFER:=0}"
: "${CKPT_EVERY_VALIDATION:=False}"
: "${PRE_VALIDATE:=False}"
: "${WALLTIME_EXIT_MINUTES:=0}"
: "${RUN_WARMUP_ON_SYNTH_DATA:=1}"
: "${WARMUP_TRAIN_STEPS:=1}"
: "${RESET_FP8_STATS_AFTER_WARMUP:=1}"
: "${OVERRIDE_ZERO_CONSUMED_SAMPLES:=1}"
: "${DATASET_MAX_STEPS:=0}"
: "${FORCE_SUCCESS_STATUS:=0}"
: "${LOAD_DIRECTLY_ON_DEVICE:=1}"

: "${CUSTOM_INPUT_PIPELINE:=True}"
: "${EXCHANGE_INDICES_DISTRIBUTED:=True}"

: "${LOGLEVEL:=WARNING}"
: "${ENABLE_PROGRESS_BAR:=False}"
: "${EXTRA_ARGS:=$@}"

echo RANK="${RANK}", LOCAL_RANK="${LOCAL_RANK}", MASTER_ADDR="${MASTER_ADDR}", MASTER_PORT="${MASTER_PORT}", WORLD_SIZE="${WORLD_SIZE}", UCX_NET_DEVICES="${UCX_NET_DEVICES}", NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}", NCCL_IB_HCA="${NCCL_IB_HCA}", NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}", NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING}", SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING="${SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING}", UCX_VFS_ENABLE="${UCX_VFS_ENABLE}"


# LOCAL_RANK is set with an enroot hook for Pytorch containers
# SLURM_LOCALID is set by Slurm
# OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

readonly _explicit_log_dir=/results/${NEMO_RESULTS_SUBDIR:-""}
if [ -n "${LOAD_CHECKPOINT}" ]; then
  if [ ${SHARE_RERUNS:-0} -eq 1 ] && [ -d "${_explicit_log_dir}/checkpoints" ] && [ -n "$(ls -A "${_explicit_log_dir}/checkpoints")" ]
  then
    [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ] && echo \
      "Detected a shared rerun." \
      "Resuming from previous run checkpoint stored in ${_explicit_log_dir}/checkpoints" \
      "instead of the initial checkpoint ${LOAD_CHECKPOINT}"
  else
    EXTRA_ARGS+=" model.resume_from_checkpoint=\"${LOAD_CHECKPOINT}\""
  fi
fi
if [ -n "${NEMO_RESULTS_SUBDIR}" ]; then
  EXTRA_ARGS+=" exp_manager.explicit_log_dir=\"${_explicit_log_dir}\""
fi

[ "$INTERLEAVED_PIPELINE" == "0" ] && INTERLEAVED_PIPELINE=null

if [ -n "${VALID_C4}" ]; then
  DATA_PREFIX="{train: $BLEND, validation:[$VALID_C4], test: [$VALID_C4]}"  # test is dummy but required
else
  DATA_PREFIX="$BLEND"
fi

if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
then
  echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"
  START=$(date +%s)
  START_FMT=$(date +%Y-%m-%d\ %r)
  echo "STARTING TIMING RUN AT ${START_FMT}"
fi

if [ ! -z "$THROUGHPUT_RUN" ]
then
  MAX_STEPS=4
fi

if [ "$USE_DIST_OPTIMIZER" = True ]; then
  readonly _optimizer_name=distributed_fused_adam
else
  readonly _optimizer_name=fused_adam
  # Remove some DistrOptim specific params
  EXTRA_ARGS+=" ~model.optim.bucket_cap_mb"
  EXTRA_ARGS+=" ~model.optim.overlap_grad_sync"
  EXTRA_ARGS+=" ~model.optim.overlap_param_sync"
  EXTRA_ARGS+=" ~model.optim.contiguous_grad_buffer"
  EXTRA_ARGS+=" ~model.optim.grad_sync_dtype"
fi

if [ "$CKPT_EVERY_VALIDATION" = True ]; then
  EXTRA_ARGS+=" exp_manager.checkpoint_callback_params.every_n_epochs=1"
  EXTRA_ARGS+=" exp_manager.checkpoint_callback_params.save_last=False"
fi

if [ "${WALLTIME_EXIT_MINUTES:-0}" -gt 0 ]; then
  #timeleft=`squeue -j ${SLURM_JOBID} --noheader --format=%L`

  timeleft=(`echo $WALLTIME | tr ':-' ' '`)
  max_time_minutes=$((timeleft[1]*60 + timeleft[2] - ${WALLTIME_EXIT_MINUTES}))
  [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ] && echo "Setting max_time to $max_time_minutes minutes"

  EXTRA_ARGS+=" +trainer.max_time=00:00:${max_time_minutes}:00"
fi

if [ ${NVTX_FLAG} -gt 0 ]; then
 NSYSCMD=" nsys profile --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/language_model_pytorch_${DGXNNODES}x${DGXNGPU}x${MINIBATCHSIZE}_${DATESTAMP}_${SLURM_PROCID}_${SYNTH_DATA}.nsys-rep "
fi

# run benchmark
[ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ] && echo "running LLM benchmark"

declare -a CMD

IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 && "${ENABLE_IB_BINDING:-}" == "1" ]]; then
    IB_BIND='--ib=single'
fi
if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( 'bindpcie' '--cpu=exclusive' ${IB_BIND} '--' ${NSYSCMD} 'python' '-u')
else
    # docker or single gpu, no need to bind
    CMD=( ${NSYSCMD} 'python' '-u' )
fi

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"

  # there is a bug in apiLog.sh preventing it from collecting
  # NCCL logs, the workaround is to log a single rank only
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
    echo "Using LOGGER=${LOGGER}"
  else
    LOGGER=""
  fi
fi

# Logging
EXPNAME="${SLURM_JOB_NUM_NODES-1}nodes_${DATETIME}"

[ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ] && echo "Extra args: $EXTRA_ARGS"

${LOGGER:-} ${CMD[@]} /workspace/llm/megatron_gpt_pretraining_custom.py \
	--config-path=/workspace/llm/conf \
	--config-name=megatron_gpt_config \
	model.seed=${SEED} \
	trainer.devices=${SLURM_NTASKS_PER_NODE-$DGXNGPU} \
	trainer.num_nodes=${SLURM_JOB_NUM_NODES-1} \
	trainer.max_steps=${MAX_STEPS} \
	trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
	trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
	+trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES} \
	trainer.max_epochs=1 \
	exp_manager.create_checkpoint_callback=${ENABLE_RERUNS} \
	exp_manager.resume_if_exists=${ENABLE_RERUNS} \
	model.global_batch_size=${GLOBAL_BATCH_SIZE} \
	model.micro_batch_size=${MICRO_BATCH_SIZE} \
	model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL} \
	model.pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL} \
	model.virtual_pipeline_model_parallel_size=${INTERLEAVED_PIPELINE} \
	model.hidden_size=${HIDDEN_SIZE} \
	model.ffn_hidden_size=${FFN_HIDDEN_SIZE} \
	model.num_layers=${NUM_LAYERS} \
	model.num_attention_heads=${NUM_ATTENTION_HEADS} \
	model.data.data_prefix="${DATA_PREFIX}" \
	model.data.no_seqlen_plus_one_input_tokens=${CUSTOM_INPUT_PIPELINE} \
	model.data.exchange_indices_distributed=${EXCHANGE_INDICES_DISTRIBUTED} \
	model.optim.name=${_optimizer_name} \
	model.optim.lr=${LR} \
	model.optim.bucket_cap_mb=${BUCKET_CAP_MB} \
	model.optim.overlap_param_sync=${OVERLAP_PARAM_SYNC} \
	model.optim.overlap_grad_sync=${OVERLAP_GRAD_SYNC} \
	model.optim.sched.warmup_steps=${WARMUP_STEPS} \
	model.optim.sched.min_lr=${MIN_LR} \
	model.optim.sched.max_steps_for_lr_sched=${MAX_STEPS_FOR_LR_SCHED} \
	model.activations_checkpoint_granularity=${ACT_CKPT_GRANULARITY} \
	model.activations_checkpoint_method=${ACT_CKPT_METHOD} \
	model.activations_checkpoint_num_layers=${ACT_CKPT_NUM_LAYERS} \
	model.sequence_parallel=${SEQ_PARALLEL} \
	model.transformer_engine=${TRANSFORMER_ENGINE} \
	model.fp8=${FP8} \
	model.fp8_hybrid=${FP8_HYBRID} \
	model.fp8_amax_history_len=${FP8_AMAX_HISTORY} \
	model.fp8_amax_compute_algo=${FP8_AMAX_ALGO} \
	model.reduce_amax=${FP8_REDUCE_AMAX} \
	model.ub_tp_comm_overlap=${TP_COMM_OVERLAP} \
	model.ub_tp_comm_overlap_cfg=${TP_CONFIG_FILE} \
	model.overlap_p2p_comm=${OVERLAP_P2P_COMM} \
	model.batch_p2p_comm=${BATCH_P2P_COMM} \
	model.sync_batch_comm=${SYNC_BATCH_COMM} \
	model.custom.init_global_step=${INIT_GLOBAL_STEP} \
	model.custom.target_log_ppl=${TARGET_LOG_PPL} \
	model.custom.use_distributed_checkpointing=${USE_DIST_CHECKPOINTING} \
	model.custom.use_two_stage_loading=${USE_TWO_STAGE_LOADING} \
	model.custom.use_two_stage_cpu_transfer=${USE_TWO_STAGE_CPU_TRANSFER} \
	model.custom.run_warmup_on_synth_data=${RUN_WARMUP_ON_SYNTH_DATA} \
	model.custom.reset_fp8_stats_after_warmup=${RESET_FP8_STATS_AFTER_WARMUP} \
	model.custom.pre_validate=${PRE_VALIDATE} \
	model.custom.override_zero_consumed_samples=${OVERRIDE_ZERO_CONSUMED_SAMPLES} \
	model.custom.dataset_max_steps=${DATASET_MAX_STEPS} \
	model.custom.force_success_status=${FORCE_SUCCESS_STATUS} \
	model.custom.load_directly_on_device=${LOAD_DIRECTLY_ON_DEVICE} \
	model.custom.warmup_train_steps=${WARMUP_TRAIN_STEPS} \
	exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
	trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
	trainer.enable_progress_bar=${ENABLE_PROGRESS_BAR} \
	trainer.limit_test_batches=${LIMIT_TEST_BATCHES} \
	$EXTRA_ARGS \
	; ret_code=$?

set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
then
  # End timing
  END=$(date +%s)
  END_FMT=$(date +%Y-%m-%d\ %r)
  echo "ENDING TIMING RUN AT ${END_FMT}"

  # Report result
  RESULT=$(( ${END} - ${START} ))
  RESULT_NAME="large language model"
  echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"
fi
