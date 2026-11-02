#!/bin/bash
# 2026-11-01 services A/B (edomtech step 1). One phase = idle snapshot + sequential cell + overlap cell,
# one fresh process each, same harness/shape as enc_int8_ab (flag=0, int8 encoder default, seed 1001).
# Phases are run ctrl -> stopped -> ctrl; services are toggled between phases by stop/start scripts.
# Usage: run_phase.sh <phase_tag>   e.g. 1_ctrl, 2_stopped, 3_ctrl
set -u
cd "$(dirname "$0")/../.."
PY=shubert_venv/bin/python3
D=latency_runs/services_ab
P="$1"
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
sleep 30   # let the box settle after toggling services
bash $D/snapshot.sh "${P}_idle"
i=0
for COND in sequential overlap; do
  i=$((i+1)); TAG="${P}_${COND}"
  echo "=== cell $TAG start $(date +%T) oc3=$(cat /sys/class/hwmon/hwmon3/oc3_event_cnt 2>/dev/null) ==="
  tegrastats --interval 1000 --logfile $D/tegrastats_$TAG.log &
  sleep 2
  $PY latency_runs/latency_probe.py --condition $COND \
      --pass-index $i --seed 1001 --out $D/lat_$TAG.csv > $D/run_$TAG.log 2>&1
  echo "=== cell $TAG exit $? $(date +%T) oc3=$(cat /sys/class/hwmon/hwmon3/oc3_event_cnt 2>/dev/null) ==="
  pkill -u "$USER" -x tegrastats
  sleep 20
done
echo "PHASE_DONE $P"
