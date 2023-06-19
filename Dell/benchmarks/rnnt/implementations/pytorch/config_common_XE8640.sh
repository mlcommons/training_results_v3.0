## System config params

# ${BASH_SOURCE[1]} assumes this is a depth-1 nested call
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1

export DGXNGPU=4  # can be overwritten
