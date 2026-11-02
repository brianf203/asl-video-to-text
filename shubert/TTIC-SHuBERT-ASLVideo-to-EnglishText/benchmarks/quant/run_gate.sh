#!/bin/bash
# bf16 replay check, then the three torchao variants only if bf16 reproduces the capture text.
cd "$(dirname "$0")/../.."
export EVAL_DIR=eval_set_openasl BYT5_DEVICE=cuda MALLOC_ARENA_MAX=2 PYTHONPATH="$1"
for v in ${VARIANTS:-bf16 dec_int4 enc_int8 both}; do
  echo "=== $v $(date -Is) boot $(cat /proc/sys/kernel/random/boot_id) ==="
  shubert_venv/bin/python3 benchmarks/quant/quant_gate.py --variant $v || { echo "=== $v FAILED rc=$?"; exit 1; }
  if [ $v = bf16 ]; then
    d=$(shubert_venv/bin/python3 -c "import json;print(json.load(open('benchmarks/quant/results/bf16_summary.json'))['differ_vs_bf16'])")
    [ "$d" = 0 ] || { echo "=== bf16 replay differs on $d clips; stopping"; exit 2; }
  fi
done
echo "=== ALL DONE $(date -Is)"
