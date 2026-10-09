"""Analyse the latency CSVs. Measured numbers only -- nothing smoothed or extrapolated.

Percentiles are numpy's linear interpolation over the measured values. With n around 120 a
p99 sits between the two largest samples, i.e. it IS essentially the max and carries no
information the max does not; that is stated wherever it is printed rather than left for
the reader to infer.
"""
import csv
import os
import sys

import numpy as np

OUT_DIR = "latency_runs"
BUCKETS = [(2, 4), (4, 6), (6, 9), (9, 12), (12, 15)]
# Retained frames arrive at the camera rate divided by FRAME_STRIDE: 30/2 = 15 per second.
# Perception must sustain that to finish inside the utterance and leave a zero drain.
REQUIRED_FPS = 15.0
RAW_CAMERA_FPS = 30.0


def load(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in ("duration_s", "capture_seconds", "post_cut_latency_s", "worker_s",
                      "queue_wait_s", "capture_end_to_perception_done_s", "shubert_s",
                      "byt5_encoder_s", "byt5_decode_s", "byt5_stage_s", "generate_s",
                      "other_s", "perception_busy_seconds_cpu", "embed_busy_seconds"):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
            for k in ("pass", "order_index", "retained_frames_after_trim", "raw_frames",
                      "decode_steps", "output_bytes", "queue_depth_at_cut",
                      "perception_frames_at_cut", "embedded_frames",
                      "min_available_mb_during_utterance", "min_available_mb_during_run"):
                r[k] = int(float(r[k])) if r.get(k) not in (None, "", "None") else None
            for k in ("oom_or_error", "declined", "deferred", "first_after_warmup"):
                r[k] = str(r.get(k, "")).strip().lower() in ("true", "1")
            rows.append(r)
    return rows


def good(rows):
    """Measurements: an utterance that was recorded and produced text."""
    return [r for r in rows if not r["declined"] and not r["oom_or_error"]
            and r["post_cut_latency_s"] is not None]


def stats(vals, label, out):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if not len(v):
        out.append(f"  {label}: no measurements")
        return
    p = np.percentile(v, [50, 90, 95, 99])
    out.append(f"  {label}: n={len(v)}  mean {v.mean():.2f}  median {p[0]:.2f}  "
               f"p90 {p[1]:.2f}  p95 {p[2]:.2f}  p99 {p[3]:.2f}  "
               f"min {v.min():.2f}  max {v.max():.2f}")


def fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, intercept, r2, r


def bucket_of(d):
    for lo, hi in BUCKETS:
        if lo <= d < hi:
            return f"{lo}-{hi}"
    return "out-of-range"


def describe(name, rows, out):
    out.append("")
    out.append("=" * 78)
    out.append(f"CONDITION: {name}")
    out.append("=" * 78)
    g = good(rows)
    declined = [r for r in rows if r["declined"]]
    failed = [r for r in rows if r["oom_or_error"]]
    out.append(f"  utterances attempted      : {len(rows)}")
    out.append(f"  measurements (text out)   : {len(g)}")
    out.append(f"  failures / drops          : {len(failed)}")
    out.append(f"  declined by backlog guard : {len(declined)}")
    for r in failed:
        out.append(f"      FAIL {r['clip_id']} pass{r['pass']}: {r['error'][:110]}")
    for r in declined:
        out.append(f"      DECLINED {r['clip_id']} pass{r['pass']}: {r['decline_reason']}"
                   f" (queue depth {r['queue_depth_at_cut']})")
    if not g:
        return
    deferred = [r for r in g if r["deferred"]]
    out.append(f"  deferred (over stream cap): {len(deferred)}")

    out.append("")
    out.append("-- post-cut latency, seconds (p99 at this n is effectively the max) --")
    stats([r["post_cut_latency_s"] for r in g], "all", out)
    out.append("")
    out.append("-- post-cut latency by utterance-duration bucket (s) --")
    for lo, hi in BUCKETS:
        sel = [r for r in g if lo <= r["duration_s"] < hi]
        stats([r["post_cut_latency_s"] for r in sel], f"{lo:>2}-{hi:<2}s", out)

    out.append("")
    f = fit([r["duration_s"] for r in g], [r["post_cut_latency_s"] for r in g])
    if f:
        out.append(f"-- latency vs utterance duration: latency = {f[0]:.3f} * duration "
                   f"+ {f[1]:.3f}   R^2 {f[2]:.3f}   Pearson r {f[3]:.3f}")
    f2 = fit([r["retained_frames_after_trim"] for r in g],
             [r["post_cut_latency_s"] for r in g])
    if f2:
        out.append(f"-- latency vs retained frames : latency = {f2[0]:.4f} * frames "
                   f"+ {f2[1]:.3f}   R^2 {f2[2]:.3f}   Pearson r {f2[3]:.3f}")

    out.append("")
    out.append("-- fraction of measurements at or below a threshold --")
    lat = np.array([r["post_cut_latency_s"] for r in g])
    for thr in (12, 24, 30):
        out.append(f"  <= {thr:>2}s : {(lat <= thr).sum():3d}/{len(lat)} "
                   f"= {100.0 * (lat <= thr).mean():.1f}%")

    out.append("")
    out.append("-- stage shares of post-cut latency (share computed per utterance) --")
    stage_cols = [("queue wait", "queue_wait_s"), ("drain", "capture_end_to_perception_done_s"),
                  ("SHuBERT", "shubert_s"), ("ByT5 encoder", "byt5_encoder_s"),
                  ("ByT5 decode", "byt5_decode_s"), ("ByT5 stage (total)", "byt5_stage_s"),
                  ("unattributed", "other_s")]
    for label, col in stage_cols:
        sh = [r[col] / r["post_cut_latency_s"] for r in g
              if r[col] is not None and r["post_cut_latency_s"]]
        secs = [r[col] for r in g if r[col] is not None]
        if sh:
            out.append(f"  {label:<19} mean {100 * np.mean(sh):5.1f}%  median "
                       f"{100 * np.median(sh):5.1f}%   (mean {np.mean(secs):5.2f}s, "
                       f"median {np.median(secs):5.2f}s, n={len(secs)})")
        else:
            out.append(f"  {label:<19} NOT MEASURED")

    if name == "overlap":
        out.append("")
        out.append("-- latency vs queue depth at cut --")
        depths = sorted({r["queue_depth_at_cut"] for r in g
                         if r["queue_depth_at_cut"] is not None})
        for d in depths:
            sel = [r["post_cut_latency_s"] for r in g if r["queue_depth_at_cut"] == d]
            stats(sel, f"depth {d}", out)

    out.append("")
    out.append("-- 5 worst utterances by post-cut latency --")
    worst = sorted(g, key=lambda r: -r["post_cut_latency_s"])[:5]
    for r in worst:
        parts = []
        for label, col in (("wait", "queue_wait_s"),
                           ("drain", "capture_end_to_perception_done_s"),
                           ("shubert", "shubert_s"), ("enc", "byt5_encoder_s"),
                           ("decode", "byt5_decode_s"), ("other", "other_s")):
            parts.append(f"{label} {'--' if r[col] is None else format(r[col], '.1f')}")
        dom = max(((r[c] or 0, n) for n, c in
                   (("drain", "capture_end_to_perception_done_s"),
                    ("decode", "byt5_decode_s"), ("queue wait", "queue_wait_s"))))[1]
        out.append(f"  {r['clip_id']:<20} pass{r['pass']} dur {r['duration_s']:5.2f}s "
                   f"frames {r['retained_frames_after_trim']:>3} "
                   f"bytes {r['output_bytes']} steps {r['decode_steps']} "
                   f"lat {r['post_cut_latency_s']:6.2f}s | {' '.join(parts)} "
                   f"| dominated by {dom}"
                   + ("  [DEFERRED: perception not started during capture]"
                      if r["deferred"] else ""))

    out.append("")
    out.append("-- per-pass median latency (drift / throttling check) --")
    for p in sorted({r["pass"] for r in g}):
        sel = [r["post_cut_latency_s"] for r in g if r["pass"] == p]
        out.append(f"  pass {p}: n={len(sel)} median {np.median(sel):.2f}s "
                   f"mean {np.mean(sel):.2f}s")

    out.append("")
    out.append("-- cold start (D) --")
    firsts = [r for r in g if r["first_after_warmup"]]
    rest = [r for r in g if not r["first_after_warmup"]]
    for r in firsts:
        out.append(f"  pass {r['pass']} first utterance after warmup: {r['clip_id']} "
                   f"dur {r['duration_s']:.2f}s latency {r['post_cut_latency_s']:.2f}s "
                   f"(warmup itself {r['warmup_seconds']}s)")
    if rest:
        out.append(f"  steady state (excluding those): n={len(rest)} "
                   f"median {np.median([r['post_cut_latency_s'] for r in rest]):.2f}s")

    out.append("")
    out.append("-- output size --")
    by = [r["output_bytes"] for r in g if r["output_bytes"] is not None]
    st = [r["decode_steps"] for r in g if r["decode_steps"] is not None]
    if by:
        out.append(f"  output bytes: mean {np.mean(by):.1f}  max {max(by)}  "
                   f"at/over the 768-byte cap: {sum(1 for b in by if b >= 768)}")
    if st:
        out.append(f"  decode steps: mean {np.mean(st):.1f}  max {max(st)}  "
                   f"(ByT5 is byte-level, so steps ~ bytes + specials)")

    out.append("")
    out.append("-- perception real-time factor (B), per utterance --")
    sustained = [(r["perception_frames_at_cut"] / r["capture_seconds"], r)
                 for r in g if r["perception_frames_at_cut"] is not None
                 and r["capture_seconds"]]
    if sustained:
        v = np.array([s for s, _ in sustained])
        out.append(f"  sustained perception fps DURING capture: mean {v.mean():.2f}  "
                   f"median {np.median(v):.2f}  min {v.min():.2f}  max {v.max():.2f}  "
                   f"(n={len(v)})")
        need = REQUIRED_FPS / v
        out.append(f"  shortfall vs the {REQUIRED_FPS:.0f} fps needed for a zero drain: "
                   f"mean {need.mean():.2f}x  median {np.median(need):.2f}x  "
                   f"min {need.min():.2f}x  max {need.max():.2f}x")
        out.append(f"  same figure against the raw {RAW_CAMERA_FPS:.0f} fps camera rate: "
                   f"mean {(RAW_CAMERA_FPS / v).mean():.2f}x")
    cpu = [(r["perception_busy_seconds_cpu"] / r["capture_seconds"], r) for r in g
           if r["perception_busy_seconds_cpu"] is not None and r["capture_seconds"]]
    if cpu:
        v = np.array([c for c, _ in cpu])
        out.append(f"  aggregate perception CPU seconds / utterance duration: "
                   f"mean {v.mean():.2f}  median {np.median(v):.2f}  min {v.min():.2f}  "
                   f"max {v.max():.2f}   [AGGREGATE across PERCEPTION_WORKERS=2, so "
                   f"divide by 2 for per-worker: mean {v.mean() / 2:.2f}]")
    return


def decode_length(all_rows, out):
    out.append("")
    out.append("=" * 78)
    out.append("DECODE-LENGTH DEPENDENCE (F)")
    out.append("=" * 78)
    g = [r for r in all_rows if not r["declined"] and not r["oom_or_error"]
         and r["byt5_decode_s"] is not None and r["output_bytes"] is not None]
    if len(g) < 3:
        out.append("  too few measurements")
        return
    f = fit([r["output_bytes"] for r in g], [r["byt5_decode_s"] for r in g])
    out.append(f"  decode_s vs output_bytes : slope {1000 * f[0]:.2f} ms/byte  "
               f"intercept {f[1]:.3f}s  R^2 {f[2]:.3f}  Pearson r {f[3]:.3f}  (n={len(g)})")
    gs = [r for r in g if r["decode_steps"]]
    if gs:
        f2 = fit([r["decode_steps"] for r in gs], [r["byt5_decode_s"] for r in gs])
        out.append(f"  decode_s vs decode_steps : slope {1000 * f2[0]:.2f} ms/step  "
                   f"intercept {f2[1]:.3f}s  R^2 {f2[2]:.3f}  Pearson r {f2[3]:.3f}")
        per = np.array([1000 * r["byt5_decode_s"] / r["decode_steps"] for r in gs])
        out.append(f"  naive decode_s/steps     : mean {per.mean():.2f} ms/step  "
                   f"median {np.median(per):.2f}  min {per.min():.2f}  max {per.max():.2f}")


def determinism(seq, ovl, out):
    out.append("")
    out.append("=" * 78)
    out.append("DETERMINISM (STEP 5)")
    out.append("=" * 78)
    by_clip = {}
    for r in good(seq):
        by_clip.setdefault(r["clip_id"], {})[r["pass"]] = r["output_text"]
    multi = {k: v for k, v in by_clip.items() if len(v) > 1}
    diff = {k: v for k, v in multi.items() if len(set(v.values())) > 1}
    out.append(f"  sequential: {len(multi)} clips measured in more than one pass; "
               f"{len(diff)} differ across passes")
    for k, v in diff.items():
        out.append(f"    DIFFERS {k}:")
        for p, t in sorted(v.items()):
            out.append(f"       pass{p}: {t[:100]}")
    seq_text = {k: next(iter(v.values())) for k, v in by_clip.items()}
    ovl_text = {}
    for r in good(ovl):
        ovl_text.setdefault(r["clip_id"], r["output_text"])
    shared = sorted(set(seq_text) & set(ovl_text))
    mism = [k for k in shared if seq_text[k] != ovl_text[k]]
    out.append(f"  overlap vs sequential: {len(shared)} clips in both; {len(mism)} differ")
    for k in mism:
        out.append(f"    DIFFERS {k}:")
        out.append(f"       sequential: {seq_text[k][:100]}")
        out.append(f"       overlap   : {ovl_text[k][:100]}")


def plot(seq, ovl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for rows, label, color, marker in ((good(seq), "sequential", "#1f77b4", "o"),
                                       (good(ovl), "overlapped", "#d62728", "^")):
        if not rows:
            continue
        x = [r["duration_s"] for r in rows]
        y = [r["post_cut_latency_s"] for r in rows]
        ax.scatter(x, y, s=26, alpha=0.7, c=color, marker=marker,
                   label=f"{label} (n={len(rows)})", edgecolors="none")
        f = fit(x, y)
        if f:
            xs = np.linspace(min(x), max(x), 50)
            ax.plot(xs, f[0] * xs + f[1], c=color, lw=1.8,
                    label=f"{label} fit: {f[0]:.2f}x + {f[1]:.2f} (R^2 {f[2]:.2f})")
    ax.set_xlabel("utterance duration (seconds)")
    ax.set_ylabel("post-cut latency (seconds)")
    ax.set_title("Post-cut latency vs utterance duration\n"
                 "Jetson Orin Nano 8GB, shipping configuration, replayed OpenASL clips")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "latency_vs_duration.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    seq = load(os.path.join(OUT_DIR, "sequential.csv"))
    ovl = load(os.path.join(OUT_DIR, "overlap.csv"))
    out = []
    out.append("ANALYSIS -- measured values only, nothing smoothed or extrapolated.")
    out.append(f"sequential.csv rows: {len(seq)}   overlap.csv rows: {len(ovl)}")
    describe("sequential", seq, out)
    describe("overlap", ovl, out)
    decode_length(seq + ovl, out)
    determinism(seq, ovl, out)
    text = "\n".join(out)
    with open(os.path.join(OUT_DIR, "analysis.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    if good(seq) or good(ovl):
        plot(seq, ovl)


if __name__ == "__main__":
    main()
