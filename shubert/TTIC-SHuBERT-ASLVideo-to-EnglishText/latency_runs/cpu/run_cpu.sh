#!/bin/bash
# CPU/memory utilisation study, 2026-10-21: one sequential + one overlap pass of the 40-clip set.
set -u
cd "$(dirname "$0")/../.."
PY=shubert_venv/bin/python3
D=latency_runs/cpu
for COND in sequential overlap; do
  tegrastats --interval 500 --logfile $D/tegrastats_$COND.log &
  sleep 2
  CPU_TRACE=$D/trace_$COND.csv $PY latency_runs/cpu_profile.py --condition $COND \
      --pass-index 1 --seed 1001 --out $D/lat_$COND.csv 2>&1 | tee $D/run_$COND.log
  tegrastats --stop 2>/dev/null || pkill -u "$USER" -x tegrastats
  sleep 20
done
echo ALL_DONE
