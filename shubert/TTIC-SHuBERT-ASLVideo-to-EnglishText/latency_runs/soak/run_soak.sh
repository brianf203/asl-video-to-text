#!/bin/bash
# 2026-10-21 soak: PYTORCH_NO_CUDA_MEMORY_CACHING=0 for SOAK_HOURS, hunting the NVML assert.
# Alternates sequential (120 utterances/process) and overlap (40) passes, fresh process each,
# new seed each. A crash is recorded and the loop continues.
set -u
cd "$(dirname "$0")/../.."
PY=shubert_venv/bin/python3
D=latency_runs/soak
END=$(( $(date +%s) + ${SOAK_HOURS:-4} * 3600 ))
i=0
while [ "$(date +%s)" -lt "$END" ]; do
  for COND in sequential overlap; do
    [ "$(date +%s)" -ge "$END" ] && break
    i=$((i+1)); TAG="$(printf %02d $i)_$COND"
    CS=latency_runs/clip_set.csv; [ $COND = sequential ] && CS=$D/clip_set_x3.csv
    echo "=== $TAG start $(date +%T)"
    tegrastats --interval 1000 --logfile $D/tegrastats_$TAG.log &
    sleep 2
    PYTORCH_NO_CUDA_MEMORY_CACHING=0 $PY latency_runs/latency_probe.py --condition $COND \
        --pass-index $i --seed $((2000 + i)) --clip-set $CS --out $D/lat_$TAG.csv > $D/run_$TAG.log 2>&1
    RC=$?
    pkill -u "$USER" -x tegrastats
    echo "=== $TAG exit $RC $(date +%T)  nvml=$(grep -c NVML_SUCCESS $D/run_$TAG.log) failed=$(grep -c FAILED $D/run_$TAG.log)"
    sleep 20
  done
done
echo SOAK_DONE
