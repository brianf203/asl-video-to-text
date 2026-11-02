#!/bin/bash
# CPU study rerun at the shipping config (flag=0), with mpstat -P ALL and jtop logged alongside.
set -u
cd "$(dirname "$0")/../.."
PY=shubert_venv/bin/python3
D=latency_runs/cpu_flag0
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
for COND in sequential overlap; do
  tegrastats --interval 500 --logfile $D/tegrastats_$COND.log &
  mpstat -P ALL 1 > $D/mpstat_$COND.txt & S_MP=$!
  python3 latency_runs/jtop_log.py $D/jtop_$COND.csv 0.5 & J=$!
  sleep 2
  CPU_TRACE=$D/trace_$COND.csv $PY latency_runs/cpu_profile.py --condition $COND \
      --pass-index 1 --seed 1001 --out $D/lat_$COND.csv 2>&1 | tee $D/run_$COND.log
  kill $S_MP $J; tegrastats --stop 2>/dev/null || pkill -u "$USER" -x tegrastats
  sleep 20
done
echo ALL_DONE
