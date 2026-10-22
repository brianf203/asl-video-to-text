"""Summarise the PYTORCH_NO_CUDA_MEMORY_CACHING A/B (run_flag_ab.sh): one row per cell."""
import csv
import glob
import os
import re
import statistics as st

D = "latency_runs/flag_ab"


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def tegra(tag):
    swap, lfb = [], []
    for line in open(f"{D}/tegrastats_{tag}.log", errors="replace"):
        m = re.search(r"SWAP (\d+)/", line)
        n = re.search(r"lfb (\d+)x", line)
        if m:
            swap.append(int(m.group(1)))
        if n:
            lfb.append(int(n.group(1)))
    return (max(swap) if swap else None), (min(lfb) if lfb else None)


cells = sorted(glob.glob(f"{D}/lat_*.csv"), key=lambda p: int(os.path.basename(p)[4]))
texts = {}
print(f"{'cell':24s} meas/att decl err  NVML  median  mean   p95    max  decode ms/step  drain  minavail swapmax lfb4MBmin")
for path in cells:
    tag = os.path.basename(path)[4:-4]
    rows = list(csv.DictReader(open(path)))
    m = [r for r in rows if r["declined"] != "True" and r["oom_or_error"] != "True"]
    err = sum(r["oom_or_error"] == "True" for r in rows)
    decl = [r for r in rows if r["declined"] == "True"]
    log = open(f"{D}/run_{tag}.log", errors="replace").read()
    nvml = log.count("NVML_SUCCESS")
    pc = [float(r["post_cut_latency_s"]) for r in m]
    ms = [1000 * float(r["byt5_decode_s"]) / int(r["decode_steps"]) for r in m if r["decode_steps"]]
    dr = [float(r["capture_end_to_perception_done_s"]) for r in m if r["capture_end_to_perception_done_s"]]
    av = [int(r["min_available_mb_during_run"]) for r in rows if r.get("min_available_mb_during_run")]
    swap, lfb = tegra(tag)
    texts[tag] = {r["clip_id"]: r["output_text"] for r in m}
    reasons = {}
    for r in decl:
        k = "frames" if "backlog" in r["decline_reason"] else "MB"
        reasons[k] = reasons.get(k, 0) + 1
    print(f"{tag:24s} {len(m):3d}/{len(rows):<3d} {len(decl):3d}  {err:3d}  {nvml:3d}  "
          + (f"{st.median(pc):6.2f} {st.mean(pc):6.2f} {pct(pc, .95):6.2f} {max(pc):6.2f}" if pc else " " * 27)
          + (f"   {st.median(ms):7.1f}      " if ms else " " * 16)
          + (f"{st.mean(dr):5.2f}" if dr else "   - ")
          + f"   {min(av) if av else '-':>6}  {swap}  {lfb}   declines={reasons}")

print("\ntranslation identity, flag0 vs flag1 (same clips, same order):")
tags = list(texts)
for a in tags:
    for b in tags:
        if a < b and a.split("_")[1] == b.split("_")[1]:
            shared = set(texts[a]) & set(texts[b])
            diff = sum(texts[a][c] != texts[b][c] for c in shared)
            print(f"  {a} vs {b}: {len(shared)} shared, {diff} differ")
