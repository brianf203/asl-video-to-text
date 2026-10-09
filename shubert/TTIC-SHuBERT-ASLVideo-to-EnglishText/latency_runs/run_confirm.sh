#!/bin/bash
# The two confirming tests the 2026-10-08 report left open, plus the controls they need.
#
# TEST 1 -- why decode measures 77.8 ms/step live against the log's 31.3 ms.
#   PERCEPTION_WORKERS=1 does NOT isolate this: stream.finish() joins the perception and
#   embed threads before ByT5 starts, so perception is idle during decode either way. The
#   isolating test is the project's own committed microbenchmark, run in both cells of the
#   caching flag, because nocache_ab.py already measured 38.8 (unset) vs 99.7 (=1) ms/step
#   and the SHIPPING configuration is =1.
#
# TEST 2 -- whether chunk-boundary re-detection is why the deferred path decodes differently.
#   With PERCEPTION_WORKERS=1, add_frame's (index // PERCEPTION_CHUNK) % 1 is always 0, so
#   one detector sees every frame in order and there are no chunk boundaries -- the same
#   thing video_holistic does on the deferred path. If W1 output == DEFER output and both
#   differ from the 2-worker baseline on the same clips, chunking is the cause.
#
# MAX_LIVE_STREAMS=0 forces every clip onto the deferred path (_take_stream_slot: 0 >= 0).
# Sequential throughout, so queue depth is 0 and queueing cannot confound the comparison.
# The control runs LAST, per the probe's own matched-start-state rule.
set -u
cd "$(dirname "$0")/.."
PY=shubert_venv/bin/python3
LR=latency_runs

echo "################ TEST 1: decode, standalone, both flag cells ################"
for cell in 0 1; do
  echo "--- nocache_ab.py with PYTORCH_NO_CUDA_MEMORY_CACHING=$cell ---"
  PYTORCH_NO_CUDA_MEMORY_CACHING=$cell $PY benchmarks/nocache_ab.py 2>&1 | tail -2
done
echo "--- nocache_ab.py with the flag UNSET (as the 31.3 ms figure was measured) ---"
env -u PYTORCH_NO_CUDA_MEMORY_CACHING $PY benchmarks/nocache_ab.py 2>&1 | tail -2
echo "--- decode_bound.py, shipping flag=1, beam sweep ---"
PYTORCH_NO_CUDA_MEMORY_CACHING=1 $PY benchmarks/decode_bound.py 2>&1 | tail -10

echo "################ TEST 2 + the 2.53x check ################"
run() {  # name, out, extra env...
  local name=$1 out=$2; shift 2
  echo "=== $name ($* ) -> $out ==="
  tegrastats --interval 1000 > "$LR/tegrastats_${name}.log" 2>&1 &
  local ts=$!
  env "$@" $PY $LR/latency_probe.py --condition sequential --pass-index 1 \
      --seed 1001 --out "$out" --limit 0 2>&1 | tee "$LR/run_${name}.log" | grep -E "^\[lat\]"
  local rc=${PIPESTATUS[0]}
  kill $ts 2>/dev/null; wait $ts 2>/dev/null
  echo "=== $name exit $rc ==="
  sleep 20
}

run defer   "$LR/confirm_defer.csv"   MAX_LIVE_STREAMS=0
run w1      "$LR/confirm_w1.csv"      PERCEPTION_WORKERS=1
run control "$LR/confirm_control.csv" FRAME_STRIDE=2
echo "ALL CONFIRM RUNS DONE"
