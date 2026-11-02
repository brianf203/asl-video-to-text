#!/bin/bash
# 2026-10-31 A/B: glibc MALLOC_ARENA_MAX unset (default) vs =2, at the shipping flag=0.
# Matched control-variant-control triples, sequential then overlap, one fresh process per cell.
set -u
cd "$(dirname "$0")/../.."
PY=shubert_venv/bin/python3
D=latency_runs/arena_ab
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
i=0
for COND in sequential overlap; do
  for ARENA in default 2 default; do
    i=$((i+1)); TAG="${i}_${COND}_arena${ARENA}"
    echo "=== cell $TAG start $(date +%T) oc3=$(cat /sys/class/hwmon/hwmon3/oc3_event_cnt 2>/dev/null) ==="
    tegrastats --interval 1000 --logfile $D/tegrastats_$TAG.log &
    sleep 2
    if [ "$ARENA" = default ]; then
      env -u MALLOC_ARENA_MAX $PY latency_runs/latency_probe.py --condition $COND \
          --pass-index $i --seed 1001 --out $D/lat_$TAG.csv > $D/run_$TAG.log 2>&1
    else
      MALLOC_ARENA_MAX=$ARENA $PY latency_runs/latency_probe.py --condition $COND \
          --pass-index $i --seed 1001 --out $D/lat_$TAG.csv > $D/run_$TAG.log 2>&1
    fi
    echo "=== cell $TAG exit $? $(date +%T) oc3=$(cat /sys/class/hwmon/hwmon3/oc3_event_cnt 2>/dev/null) ==="
    pkill -u "$USER" -x tegrastats
    sleep 20
  done
done
echo ALL_DONE
