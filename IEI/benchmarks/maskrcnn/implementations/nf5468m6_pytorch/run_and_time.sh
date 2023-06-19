cd ../pytorch
source config_NF5468M6.sh
DGXSYSTEM=NF5468M6 CONT=mlperf-inspur:maskrcnn DATADIR=/path/to/preprocessed/data LOGDIR=/path/to/logfile PKLPATH=/path/to/data/maskrcnn/coco2017/annotations/ ./run_with_docker.sh
