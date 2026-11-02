#!/bin/bash
# 2026-11-02 live A/B (CUDA-graph phase 2): BYT5_CUDAGRAPH=0 (control) vs =1 (graph decode).
# Matched control-variant-control triples, sequential then overlap, one fresh process per cell.
# Same harness/shape as latency_runs/enc_int8_ab (flag=0, int8 encoder default, services off).
set -u
cd "$(dirname "$0")/../.."
PY=shubert_venv/bin/python3
D=latency_runs/cudagraph_ab
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
i=0
for COND in sequential overlap; do
  for Q in 0 1 0; do
    i=$((i+1)); TAG="${i}_${COND}_cudagraph_${Q}"
    echo "=== cell $TAG start $(date +%T) oc3=$(cat /sys/class/hwmon/hwmon3/oc3_event_cnt 2>/dev/null) ==="
    tegrastats --interval 1000 --logfile $D/tegrastats_$TAG.log &
    sleep 2
    BYT5_CUDAGRAPH=$Q $PY latency_runs/latency_probe.py --condition $COND \
        --pass-index $i --seed 1001 --out $D/lat_$TAG.csv > $D/run_$TAG.log 2>&1
    echo "=== cell $TAG exit $? $(date +%T) oc3=$(cat /sys/class/hwmon/hwmon3/oc3_event_cnt 2>/dev/null) ==="
    pkill -u "$USER" -x tegrastats
    sleep 20
  done
done
echo ALL_DONE
