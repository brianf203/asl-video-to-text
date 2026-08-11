#!/usr/bin/env python3
"""Paired bootstrap over two run_eval.py results files.

Why paired rather than comparing the two absolute scores: this eval set's absolute
BLEU carries a ~+/-4 point 95% CI at 200 clips (and quadrupling the set only cut it
26%, because corpus BLEU is dominated by a few long clips and its variance does not
scale like an i.i.d. mean). Two overlapping CIs therefore cannot resolve the 1-3 BLEU
a decoding or perception change is worth. When both systems ran the SAME clips, the
per-clip difficulty is common to both and cancels in the difference, so resampling the
DIFFERENCE is far more sensitive than resampling each score.

Corpus BLEU is recomputed inside every resample -- it is not an average of per-clip
scores, so bootstrapping per-clip BLEU values would measure a different statistic.

Usage:
    python3 paired_bootstrap.py A.json B.json [--resamples 2000] [--seed 0]

Reports delta (B - A) for raw BLEU, normalized BLEU and chrF with 95% CIs and
P(delta <= 0), plus the per-clip diff counts and a leave-the-winners-out check --
a corpus delta that vanishes when the top few clips are dropped was one clip, not
a real effect. That failure has happened here twice, so it is built into the output.
"""

import argparse
import json
import random
import sys

import sacrebleu

from run_eval import normalize_numbers, proper_nouns, char_similarity


def load(path):
    with open(path) as f:
        d = json.load(f)
    outputs = {o["id"]: o for o in d["outputs"]}
    return d, outputs


def corpus_scores(pairs):
    """pairs: list of (reference, hypothesis)."""
    refs = [p[0] for p in pairs]
    hyps = [p[1] for p in pairs]
    n_refs = [normalize_numbers(r) for r in refs]
    n_hyps = [normalize_numbers(h) for h in hyps]
    return {
        "bleu_raw": sacrebleu.corpus_bleu(hyps, [refs]).score,
        "bleu": sacrebleu.corpus_bleu(n_hyps, [n_refs]).score,
        "chrf": sacrebleu.corpus_chrf(n_hyps, [n_refs]).score,
    }


def sentence_bleu(ref, hyp):
    return sacrebleu.sentence_bleu(hyp, [ref]).score


def pn_similarity(pairs):
    """Mean char similarity of reference proper nouns to their best hypothesis token."""
    sims = []
    for ref, hyp in pairs:
        want = proper_nouns(ref)
        if not want:
            continue
        got = [t for t in hyp.lower().replace(",", " ").replace(".", " ").split() if t]
        for w in want:
            best = 0.0
            for g in got:
                best = max(best, char_similarity(w, g))
            sims.append(best)
    return (sum(sims) / len(sims) * 100) if sims else 0.0, len(sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline results JSON")
    ap.add_argument("b", help="candidate results JSON")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    da, oa = load(args.a)
    db, ob = load(args.b)

    shared = [i for i in oa if i in ob]
    # Order by the baseline file so runs are compared in manifest order, not dict order.
    shared = [o["id"] for o in da["outputs"] if o["id"] in ob]
    if len(shared) != len(oa) or len(shared) != len(ob):
        print(f"WARNING: {len(oa)} vs {len(ob)} clips, {len(shared)} shared -- "
              f"comparing the shared subset only", file=sys.stderr)
    if not shared:
        sys.exit("no clips in common")

    # A reference mismatch means the two runs scored different footage; pairing would
    # be meaningless, so refuse rather than emit a plausible-looking number.
    for i in shared:
        if oa[i]["reference"] != ob[i]["reference"]:
            sys.exit(f"reference text differs for {i} -- these runs are not comparable")

    def cfg(d):
        keys = ("streaming", "perception_workers", "perception_chunk", "byt5_num_beams",
                "byt5_device", "mediapipe_video_mode", "frame_stride", "no_trim")
        return {k: d.get(k) for k in keys}

    ca, cb = cfg(da), cfg(db)
    print(f"A = {args.a.split('/')[-1]}  tag={da.get('tag')!r}")
    print(f"B = {args.b.split('/')[-1]}  tag={db.get('tag')!r}")
    diffs = {k: (ca[k], cb[k]) for k in ca if ca[k] != cb[k]}
    print(f"config differences (A -> B): {diffs or 'NONE -- is this the same config?'}")
    print(f"clips paired: {len(shared)}")
    sa, sb = da.get("mean_seconds_per_clip"), db.get("mean_seconds_per_clip")
    if sa and sb:
        print(f"mean s/clip: {sa:.1f} -> {sb:.1f}  ({sa / sb:.2f}x)")
    print()

    pa = [(oa[i]["reference"], oa[i]["hypothesis"]) for i in shared]
    pb = [(ob[i]["reference"], ob[i]["hypothesis"]) for i in shared]

    base_a, base_b = corpus_scores(pa), corpus_scores(pb)

    rng = random.Random(args.seed)
    n = len(shared)
    deltas = {k: [] for k in base_a}
    for _ in range(args.resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        ra = corpus_scores([pa[i] for i in idx])
        rb = corpus_scores([pb[i] for i in idx])
        for k in deltas:
            deltas[k].append(rb[k] - ra[k])

    print(f"{'metric':<10} {'A':>8} {'B':>8} {'delta':>8}   {'95% CI':>18}  P(d<=0)")
    print("-" * 68)
    for k in ("bleu_raw", "bleu", "chrf"):
        d = sorted(deltas[k])
        lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d))]
        p = sum(1 for x in d if x <= 0) / len(d)
        print(f"{k:<10} {base_a[k]:>8.2f} {base_b[k]:>8.2f} {base_b[k] - base_a[k]:>+8.2f}"
              f"   [{lo:>+7.2f}, {hi:>+7.2f}]  {p:>6.3f}")

    pna, cnt = pn_similarity(pa)
    pnb, _ = pn_similarity(pb)
    print(f"{'pn_sim':<10} {pna:>8.1f} {pnb:>8.1f} {pnb - pna:>+8.2f}   ({cnt} names)")
    print()

    # Per-clip diff. The standing rule on this project: never read a corpus delta
    # without checking whether it is spread across clips or carried by one of them.
    changed = [i for i in shared if oa[i]["hypothesis"] != ob[i]["hypothesis"]]
    gains = []
    for i in changed:
        ga = sentence_bleu(oa[i]["reference"], oa[i]["hypothesis"])
        gb = sentence_bleu(ob[i]["reference"], ob[i]["hypothesis"])
        gains.append((gb - ga, i))
    better = sum(1 for g, _ in gains if g > 0)
    worse = sum(1 for g, _ in gains if g < 0)
    print(f"text changed on {len(changed)}/{len(shared)} clips: "
          f"{better} better, {worse} worse, {len(changed) - better - worse} even "
          f"(by sentence BLEU)")

    gains.sort(reverse=True)
    for drop in (1, 2, 5):
        if len(gains) >= drop:
            excl = {i for _, i in gains[:drop]}
            keep = [j for j, i in enumerate(shared) if i not in excl]
            ra = corpus_scores([pa[j] for j in keep])
            rb = corpus_scores([pb[j] for j in keep])
            print(f"  drop top {drop} winner(s): BLEU delta "
                  f"{rb['bleu_raw'] - ra['bleu_raw']:+.2f}")

    print("\ntop 5 gains:")
    for g, i in gains[:5]:
        print(f"  {g:+6.1f}  {i}")
    print("top 5 losses:")
    for g, i in gains[-5:][::-1]:
        print(f"  {g:+6.1f}  {i}")


if __name__ == "__main__":
    main()
