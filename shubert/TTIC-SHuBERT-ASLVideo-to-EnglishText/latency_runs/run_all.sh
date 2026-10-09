#!/bin/bash
# Drive the whole measurement: 3 passes x 2 conditions, one process per pass.
#
# One process per pass, not one for all three, because live_worker_probe.py's own
# docstring is explicit that a run leaves the box several hundred MB dirtier than it
# found it -- so each pass starts from its own fresh process and its own model load.
#
# Conditions ALTERNATE (seq p1, ovl p1, seq p2, ...) so that any monotonic drift over the
# session -- heat, memory dirt -- is shared between the two conditions instead of landing
# entirely on whichever one ran last.
set -u
cd "$(dirname "$0")/.."
PY=shubert_venv/bin/python3
LR=latency_runs
SEEDS=(1001 1002 1003)
LIMIT="${LIMIT:-0}"

# Close the user-owned desktop apps that can wake up mid-run and take memory or CPU
# (gnome-software pulls updates; tracker indexes the disk). Best effort, user-owned only --
# snapd and packagekitd are root-owned and are left running and noted in the report.
for proc in gnome-software tracker-miner-fs-3 update-notifier; do
  pkill -u "$USER" -f "$proc" && echo "closed $proc" || echo "$proc not running"
done
sleep 3

./$LR/capture_env.sh
echo "env captured -> $LR/env.txt"

for i in 0 1 2; do
  P=$((i + 1))
  for COND in sequential overlap; do
    LOG="$LR/tegrastats_${COND}_pass${P}.log"
    echo "=== $COND pass $P (seed ${SEEDS[$i]}) -> $LR/${COND}.csv ==="
    tegrastats --interval 1000 > "$LOG" 2>&1 &
    TS=$!
    $PY $LR/latency_probe.py --condition "$COND" --pass-index "$P" \
        --seed "${SEEDS[$i]}" --out "$LR/${COND}.csv" --limit "$LIMIT" \
        2>&1 | tee -a "$LR/run_${COND}_pass${P}.log"
    RC=${PIPESTATUS[0]}
    kill $TS 2>/dev/null; wait $TS 2>/dev/null
    echo "=== $COND pass $P exit $RC ==="
    sleep 20   # let the box settle between passes
  done
done

$PY $LR/parse_tegrastats.py | tee $LR/thermal.txt
$PY $LR/analysis.py
