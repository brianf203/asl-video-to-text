#!/bin/bash
# 2026-11-02 live A/B: MAX_RETAINED_FRAMES 450 (shipping) vs 900, OVERLAP only (sequential never declines).
# Matched control-variant-control, one fresh process per cell, flag=0, int8 encoder default, services
# permanently stopped (services_ab), seed 1001, gap 1.5 s. MIN_AVAILABLE_MB stays 350 as the backstop.
set -u
cd "$(dirname "$0")/../.."
PY=shubert_venv/bin/python3
D=latency_runs/cap_ab
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
i=0
for CAP in 450 900 450; do
  i=$((i+1)); TAG="${i}_overlap_cap${CAP}"
  echo "=== cell $TAG start $(date +%T) oc3=$(cat /sys/class/hwmon/hwmon3/oc3_event_cnt 2>/dev/null) avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) ==="
  tegrastats --interval 1000 --logfile $D/tegrastats_$TAG.log &
  sleep 2
  MAX_RETAINED_FRAMES=$CAP $PY latency_runs/latency_probe.py --condition overlap \
      --pass-index $i --seed 1001 --out $D/lat_$TAG.csv > $D/run_$TAG.log 2>&1
  echo "=== cell $TAG exit $? $(date +%T) oc3=$(cat /sys/class/hwmon/hwmon3/oc3_event_cnt 2>/dev/null) ==="
  pkill -u "$USER" -x tegrastats
  sleep 30
done
echo ALL_DONE
