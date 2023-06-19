#!/bin/bash
scontrol show hostname > hostfile
HOSTNAMES=$(cat hostfile)
for HOSTNAME in $HOSTNAMES; do
  ssh $HOSTNAME 'bash -s' < ${DIR}/spr_perf_mode.sh
done
