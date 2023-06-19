## System config params
export DGXNGPU=2
export DGXSOCKETCORES=32
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
#export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1"
