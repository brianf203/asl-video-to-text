#!/usr/bin/env python3
"""Pick the 40-clip latency set: 8 per duration bucket, BY DURATION ONLY.

Reproducible: durations come from ffprobe, the draw uses a fixed seed, and the
chosen ids plus their measured durations are written to clip_set.csv.

Selection is deliberately blind to latency and to translation quality -- the only
input is container duration. Buckets are the ones the measurement plan asked for.
"""
import csv
import os
import random
import subprocess
import sys

SEED = 20261008
CLIP_DIR = "eval_set_openasl/clips"
BUCKETS = [(2.0, 4.0), (4.0, 6.0), (6.0, 9.0), (9.0, 12.0), (12.0, 15.0)]
PER_BUCKET = 8
OUT = "latency_runs/clip_set.csv"


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def main():
    ids = sorted(f[:-4] for f in os.listdir(CLIP_DIR) if f.endswith(".mp4"))
    durs = {cid: duration(os.path.join(CLIP_DIR, f"{cid}.mp4")) for cid in ids}

    rng = random.Random(SEED)
    chosen = []
    for lo, hi in BUCKETS:
        pool = sorted(cid for cid in ids if lo <= durs[cid] < hi)
        if len(pool) < PER_BUCKET:
            sys.exit(f"bucket {lo}-{hi}s has only {len(pool)} clips, need {PER_BUCKET}")
        pick = rng.sample(pool, PER_BUCKET)
        for cid in sorted(pick, key=lambda c: durs[c]):
            chosen.append((f"{lo:g}-{hi:g}", cid, durs[cid], len(pool)))

    os.makedirs("latency_runs", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket_s", "clip_id", "duration_s", "bucket_pool_size"])
        for bucket, cid, d, pool in chosen:
            w.writerow([bucket, cid, f"{d:.3f}", pool])
    print(f"seed={SEED}  wrote {len(chosen)} clips to {OUT}")
    for bucket, cid, d, _ in chosen:
        print(f"  {bucket:>7}  {cid:<20} {d:6.2f}s")


if __name__ == "__main__":
    main()
