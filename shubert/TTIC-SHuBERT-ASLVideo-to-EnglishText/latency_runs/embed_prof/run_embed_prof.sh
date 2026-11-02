#!/bin/bash
# 2026-11-02 embed-thread profile (measurement only), shipping config, seed 1001, 40 clips each.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
D=latency_runs/embed_prof
i=0
for COND in sequential overlap; do
  i=$((i+1))
  echo "=== $COND start $(date +%T) ==="
  EMBED_PROFILE=$D/$COND shubert_venv/bin/python3 latency_runs/embed_profile.py --condition $COND \
      --pass-index $i --seed 1001 --out $D/${COND}_lat.csv > $D/${COND}.log 2>&1
  echo "=== $COND exit $? $(date +%T) ==="
  sleep 30
done
echo ALL_DONE
