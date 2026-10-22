"""Summarise cpu_profile.py traces: per-core load and per-thread-group CPU, split by phase.

Phases come from the matching latency CSV (sequential only -- under overlap the phases of
different utterances interleave, so it is reported whole):
  capture : cut_ts - capture_seconds .. cut_ts   (perception + DINOv2 streaming)
  drain   : cut_ts .. cut_ts + drain             (backlog finishing after the cut)
  byt5    : end of drain .. done_ts              (SHuBERT + ByT5 encoder + decode)
  idle    : everything else after warmup
"""
import csv
import re
import statistics as st
import sys
from collections import defaultdict

D = "latency_runs/cpu"


def group(label):
    if label.startswith("[mediapipe"):
        return "MediaPipe graph threads (C++)"
    if label == "[python3]":
        return "native pools (TFLite/XNNPACK, torch, cv2)"
    if label.startswith("perception-") or label.startswith("ThreadPoolExecutor"):
        return "perception Python glue"
    if label == "embed":
        return "embed thread (crop + DINOv2 launch)"
    if "(worker)" in label:
        return "translation worker (SHuBERT + ByT5)"
    if label == "MainThread":
        return "clip feeder (mp4 decode; replay stand-in for camera)"
    if label == "Thread-2":
        return "camera loop + preview"
    return "other (" + label + ")" if not label.startswith("[") else "other native"


def phases(lat_path):
    out = []
    for r in csv.DictReader(open(lat_path)):
        if r["declined"] == "True" or not r["cut_ts"] or not r["done_ts"]:
            continue
        cut, done = float(r["cut_ts"]), float(r["done_ts"])
        cap = float(r["capture_seconds"])
        drain = float(r["capture_end_to_perception_done_s"] or 0)
        out += [("capture", cut - cap, cut), ("drain", cut, cut + drain),
                ("byt5", cut + drain, done)]
    return out


def which(ts, ph):
    for name, a, b in ph:
        if a <= ts < b:
            return name
    return "idle"


def main(cond):
    trace = f"{D}/trace_{cond}.csv"
    ph = phases(f"{D}/lat_{cond}.csv") if cond == "sequential" else []
    start = min(a for _, a, _ in ph) if ph else None
    core = defaultdict(lambda: defaultdict(list))      # phase -> core -> [busy%]
    grp = defaultdict(lambda: defaultdict(float))      # phase -> group -> cpu seconds
    seconds = defaultdict(set)                          # phase -> sample timestamps
    worker_max = defaultdict(list)                      # phase -> per-sample worker cores
    per_ts_worker = defaultdict(float)
    for r in csv.DictReader(open(trace)):
        ts = float(r["ts"])
        if start is not None and ts < start:
            continue
        p = which(ts, ph) if ph else "all"
        if r["kind"] == "core":
            core[p][r["key"]].append(float(r["value"]))
            seconds[p].add(ts)
        else:
            grp[p][group(r["label"])] += float(r["value"])
            if "(worker)" in r["label"]:
                per_ts_worker[(p, ts)] += float(r["value"])
    interval = 0.5
    for (p, ts), v in per_ts_worker.items():
        worker_max[p].append(v / interval)
    print(f"===== {cond} =====")
    for p in ["capture", "drain", "byt5", "idle", "all"]:
        if p not in core:
            continue
        n = len(seconds[p])
        wall = n * interval
        cores = {k: st.mean(v) for k, v in sorted(core[p].items())}
        total = sum(cores.values()) / 100
        print(f"\n-- phase {p}: {wall:.0f}s of samples --")
        print("   per-core busy %: " + "  ".join(f"{k}={v:4.1f}" for k, v in cores.items()))
        print(f"   cores busy on average: {total:.2f} of 6  ({100*total/6:.0f}% of the CPU)")
        for g, s in sorted(grp[p].items(), key=lambda kv: -kv[1]):
            if s / wall >= 0.02:
                print(f"     {s/wall:5.2f} cores  {g}")
        if worker_max[p]:
            w = worker_max[p]
            print(f"   worker thread: median {st.median(w):.2f} cores, "
                  f"{100*sum(x > 0.9 for x in w)/len(w):.0f}% of its samples > 0.9 of a core")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["sequential", "overlap"]:
        main(c)
