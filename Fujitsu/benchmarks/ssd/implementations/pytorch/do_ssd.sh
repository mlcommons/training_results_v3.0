source config_PG_10gpu.sh

CONT=nvcr.io/nvdlfwea/mlperfv30/ssd:20230428.pytorch
DATADIR=/mnt/data4/work/ssd-openimages
LOGDIR=$(realpath ../logs-ssd)
TORCH_HOME=$(realpath ./torch-model-cache)
NEXP=1
num_of_run=1

for idx in $(seq 1 $num_of_run); do
    CONT=$CONT DATADIR=$DATADIR LOGDIR=$LOGDIR BACKBONE_DIR=$TORCH_HOME NEXP=$NEXP bash run_with_docker.sh
done
