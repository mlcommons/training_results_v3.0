source ./config_NF5488A5.sh
CONT=nvcr.io/nvdlfwea/mlperfv30/ssd:20230428.pytorch DATADIR=/mlperf/data/ssd/ssd_v20/ LOGDIR=$PWD/submit_logs_NF5488A5_0515  ./run_with_docker.sh

source ./config_NF5468M6.sh
CONT=nvcr.io/nvdlfwea/mlperfv30/ssd:20230428.pytorch DATADIR=/mlperf/data/ssd/ssd_v20/ LOGDIR=$PWD/submit_logs_NF5468M6_0515  ./run_with_docker.sh


