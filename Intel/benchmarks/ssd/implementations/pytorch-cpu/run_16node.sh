#!/bin/bash

# Dataset path
DATA_SOURCE=openimages-mlperf
DATA_PATH=path/to/dataset/${DATA_SOURCE}
BACKBONE="resnext50_32x4d"

export ONEDNN_PRIMITIVE_CACHE_CAPACITY=1024
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

torch_ccl_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $torch_ccl_path/env/setvars.sh

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

# CCL/PSM3 Settings
export CCL_ALLREDUCE=rabenseifner
export CCL_BF16=avx512bf
export CCL_ATL_TRANSPORT=ofi
export CCL_MNIC=local
export CCL_MNIC_COUNT=2
export CCL_WORKER_COUNT=2
export CCL_MNIC_NAME=irdma-cvl01tf2,irdma-cvl02tf2,irdma-cvl11tf2,irdma-cvl12tf2
export FI_PROVIDER=psm3
export PSM3_IDENTIFY=1
export PSM3_ALLOW_ROUTERS=1
export PSM3_RDMA=1
export PSM3_RV_MR_CACHE_SIZE=8192
export PSM3_NIC='irdma-cvl*'

export MASTER_ADDR=`head -1 hostfile`

CORES_PER_SOCKET=`$PREFIX lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
NUM_SOCKETS=`$PREFIX lscpu | grep "Socket(s):" | awk '{print $NF}'`
PPN=4

NUM_THREADS=$(( CORES_PER_SOCKET * NUM_SOCKETS / PPN ))
NUM_WORKER_THREADS=$(( NUM_THREADS - CCL_WORKER_COUNT ))
USE_BC=1
HT_WORKER_OFFSET=$NUM_WORKER_THREADS

if ! which bc >& /dev/null ; then USE_BC=0 ; fi

for I in 0 1 2 3 ; do
SHFT=$(( I ))
if [ $USE_BC -eq 1 ] ; then
PROC_MASK_STR[$I]=`BC_LINE_LENGTH=0 bc <<<"obase=16;(2^${NUM_THREADS} - 1)*(2^${SHFT} )"`
else
PROC_MASK=$(( ( ( 1 << NUM_WORKER_THREADS ) - 1 ) << SHFT ))
PROC_MASK_STR[$I]=`printf "%X" $PROC_MASK`
fi
done
MASKS=( )
for(( I=0; I < PPN ; I++)) ; do
  SHFT=$(( I * NUM_THREADS ))
  IND=$(( SHFT % 4 ))
  if [ $SHFT -lt 4 ] ; then
  ZEROS=""
  else
  ZEROS=`printf "%0*X" $(( SHFT / 4 ))`
  fi
  SMASK=${PROC_MASK_STR[$IND]}${ZEROS}
  MASKS[$I]="0x$SMASK"
  for((P=0;P < CCL_WORKER_COUNT ; P++)); do CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $(( HT_WORKER_OFFSET + I * NUM_THREADS + P ))" ; done
done
export I_MPI_PIN_DOMAIN=[`echo ${MASKS[@]} | tr " " ","`]
export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`

(
# Clear cache
python -c "
from ssd_logger import mllogger
from mlperf_logging.mllog import constants
mllogger.event(key=constants.CACHE_CLEAR, value=True)"

mpiexec.hydra -np 64 -ppn 4 -f hostfile -genv I_MPI_PIN_DOMAIN=$I_MPI_PIN_DOMAIN -genv OMP_NUM_THREADS=$NUM_WORKER_THREADS python -u train.py \
    --batch-size=256 \
    --world-size=64 \
    --dataset ${DATA_SOURCE} \
    --data-path="${DATA_PATH}" \
    --epochs 5 \
    --data-layout="channels_last" \
    --print-freq=20 \
    --device="cpu" \
    --backbone="${BACKBONE}" \
    --lr 0.0001 \
    --num-steps="0" \
    --workers 0 \
    --amp --bf16 --ipex-optimize \
    --fl-optimize --fl-index \
    --distributed \
    --dist-backend="ccl" ) 2>&1 | tee output.log

