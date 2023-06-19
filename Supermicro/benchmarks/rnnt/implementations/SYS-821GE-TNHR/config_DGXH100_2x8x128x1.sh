# 4n Hopper config (since v2.1)

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
: ${SINGLE_EXP_WALLTIME:=30}
source $(dirname ${BASH_SOURCE[0]})/config_common_multi_node.sh

export NCCL_CROSS_NIC=1
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=ibp155s0,ibp170s0,ibp187s0,ibp218s0,ibp25s0,ibp41s0,ibp59s0,ibp92s0,^lo,^usb0,^docker0,^enp86s0f0,^enp86s0f1
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_UCX_TLS=dc,cuda_copy,cuda_ipc
export NCCL_IB_GID_INDEX=1
export DGXNNODES=2
export BATCHSIZE=128
export EVAL_BATCHSIZE=85

export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true
export CONT="/root/enroot_img/root+nvcr.io+nvdlfwea+mlperfv30+rnnt+20230428.pytorch.sqsh"
export DATADIR=/home/dataset/rnnt/data
export METADATA_DIR=/home/dataset/rnnt/data/metadata
export SENTENCEPIECES_DIR=/home/dataset/rnnt/data/sentencepieces

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh
