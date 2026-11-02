#!/bin/bash
# Idle memory snapshot: free -m, buddyinfo (Normal zone), top user PSS, root RSS. Usage: snapshot.sh <tag>
D="$(dirname "$0")"; T="$1"; O="$D/snap_$T.txt"
{ echo "# $T $(date +%T)"; free -m; grep Normal /proc/buddyinfo
  grep -E "MemFree|MemAvailable|Cached|SwapFree" /proc/meminfo
  ps -eo rss,user:6,comm --sort=-rss | head -25; } > "$O"
grep -E "^Mem:" "$O"
