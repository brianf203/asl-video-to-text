#!/bin/bash
# 2026-10-21 A/B: PYTORCH_NO_CUDA_MEMORY_CACHING=1 (shipping) vs =0 (caching allocator ON).
# Matched control-variant-control triples, sequential then overlap, one fresh process per cell.
set -u
cd "$(dirname "$0")/../.."
PY=shubert_venv/bin/python3
D=latency_runs/flag_ab
i=0
for COND in sequential overlap; do
  for FLAG in 1 0 1; do
    i=$((i+1)); TAG="${i}_${COND}_flag${FLAG}"
    echo "=== cell $TAG start $(date +%T) ==="
    tegrastats --interval 1000 --logfile $D/tegrastats_$TAG.log &
    sleep 2
    PYTORCH_NO_CUDA_MEMORY_CACHING=$FLAG $PY latency_runs/latency_probe.py --condition $COND \
        --pass-index $i --seed 1001 --out $D/lat_$TAG.csv > $D/run_$TAG.log 2>&1
    echo "=== cell $TAG exit $? $(date +%T) ==="
    pkill -u "$USER" -x tegrastats
    sleep 20
  done
done
echo ALL_DONE
