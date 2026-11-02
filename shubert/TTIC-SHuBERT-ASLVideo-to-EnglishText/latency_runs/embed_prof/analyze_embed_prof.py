"""Summarise embed_profile.py output: embed-thread time split and where the backlog sits."""
import csv
import collections
import statistics as st
import sys

D = "latency_runs/embed_prof"
for cond in sys.argv[1:] or ["sequential", "overlap"]:
    rows = list(csv.DictReader(open(f"{D}/{cond}_steps.csv")))
    tot = collections.Counter()
    n = collections.Counter()
    for r in rows:
        tot[r["step"]] += float(r["seconds"])
        n[r["step"]] += int(r["n_items"])
    frames = n["crop_hands"]                      # one crop_hands call per flush, n = frames
    busy = sum(tot.values())
    print(f"\n== {cond}: {frames} frames embedded, embed-thread busy {busy:.1f} s "
          f"({1000 * busy / max(frames, 1):.1f} ms/frame -> {frames / max(busy, 1e-9):.1f} fps capacity)")
    for k in ["crop_hands", "crop_face", "preprocess", "lock_wait", "gpu_held"]:
        print(f"  {k:11s} {tot[k]:8.1f} s  {100 * tot[k] / busy:5.1f}%  {1000 * tot[k] / max(frames, 1):6.1f} ms/frame")

    bl = list(csv.DictReader(open(f"{D}/{cond}_backlog.csv")))
    mp, em = [], []
    per_stream = collections.defaultdict(lambda: [0, 0])
    for r in bl:
        a, l, e = int(r["added"]), int(r["landmarked"]), int(r["embedded"])
        mp.append(a - l)
        em.append(l - e)
        s = per_stream[r["stream_id"]]
        s[0] = max(s[0], a - l)
        s[1] = max(s[1], l - e)
    busy_samples = [(m, e) for m, e in zip(mp, em) if m + e > 0]
    if busy_samples:
        msum = sum(m for m, _ in busy_samples)
        esum = sum(e for _, e in busy_samples)
        print(f"  backlog (frame-samples while any backlog): waiting-for-MediaPipe {100 * msum / (msum + esum):.1f}%"
              f"  waiting-for-embed {100 * esum / (msum + esum):.1f}%")
        print(f"  per-clip max backlog, median: MediaPipe {st.median(v[0] for v in per_stream.values()):.0f}"
              f"  embed {st.median(v[1] for v in per_stream.values()):.0f}  frames"
              f"  (max: {max(v[0] for v in per_stream.values())} / {max(v[1] for v in per_stream.values())})")
