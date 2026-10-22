"""Drain-phase detail and GPU load by phase (sequential pass)."""
import csv, re, statistics as st, time
from collections import defaultdict
import sys
sys.path.insert(0, "latency_runs/cpu")
from analyze_cpu import phases, which, group

D = "latency_runs/cpu"
ph = phases(f"{D}/lat_sequential.csv")
rows = [r for r in csv.DictReader(open(f"{D}/lat_sequential.csv")) if r["declined"] != "True"]
# seconds after cut -> group cpu, during drain only
bins = defaultdict(lambda: defaultdict(float)); nb = defaultdict(set)
cuts = [(float(r["cut_ts"]), float(r["cut_ts"]) + float(r["capture_end_to_perception_done_s"] or 0)) for r in rows]
for r in csv.DictReader(open(f"{D}/trace_sequential.csv")):
    if r["kind"] != "thread": continue
    ts = float(r["ts"])
    for c, e in cuts:
        if c <= ts < e:
            b = int(ts - c)
            bins[b][group(r["label"])] += float(r["value"]); nb[b].add(ts)
print("drain: seconds after cut -> cores used (MediaPipe / embed+DINOv2)")
for b in sorted(bins)[:10]:
    w = len(nb[b]) * 0.5
    print(f"  t+{b}s  n={len(nb[b]):3d}  MediaPipe {bins[b]['MediaPipe graph threads (C++)']/w:4.2f}"
          f"  embed {bins[b]['embed thread (crop + DINOv2 launch)']/w:4.2f}")
# GPU by phase from tegrastats
g = defaultdict(list); ram = defaultdict(list)
prev = None; k = 0
for line in open(f"{D}/tegrastats_sequential.log"):
    m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)", line)
    if not m: continue
    k = k + 1 if m.group(1) == prev else 0; prev = m.group(1)
    ts = time.mktime(time.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")) + 0.5 * k
    p = which(ts, ph)
    g[p].append(int(re.search(r"GR3D_FREQ (\d+)%", line).group(1)))
    ram[p].append(int(re.search(r"RAM (\d+)/", line).group(1)))
for p in ["capture", "drain", "byt5", "idle"]:
    if g[p]:
        print(f"{p:8s} GPU busy mean {st.mean(g[p]):5.1f}%  median {st.median(g[p]):4.0f}%   RAM used median {st.median(ram[p])} MB max {max(ram[p])}")
