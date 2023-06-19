if [ ${MICRO_BATCH_SIZE} -gt 1 ]; then
 export TP_CONFIG_FILE='h100tp8pp8mbs2_tp_comm_overlap_cfg.yaml'
else
 export TP_CONFIG_FILE='h100tp4pp8mbs1_tp_comm_overlap_cfg.yaml'
fi
export TP_COMM_OVERLAP=True