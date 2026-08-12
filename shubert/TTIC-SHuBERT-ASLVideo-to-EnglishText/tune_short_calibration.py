"""Choose the statistics the STARTUP calibration fits with, by simulating the outcome.

WHY
---
`fit_thresholds.fit()` pairs the MAXIMUM of the non-signing class against the weakest
signing EVENT peak. That is the right pair when there are 84s of labelled data. It falls
apart on the ~22s the startup calibration can afford: resampling short windows out of
calib.jsonl gives fitted start thresholds from 0.35 to 1.78 (stdev 0.39) against 1.56 from
the full recording, with a median of ONE event per fit. A max over 6s of stillness is one
frame, and "events above that max" is often zero.

So the short fit needs upper-tail statistics that are less extreme, and the honest way to
choose them is not to eyeball the numbers but to run each candidate through v5's actual
state machine and count what the signer would experience: false starts, and how much of
their signing gets captured. Optimising a proxy picked the wrong config once already on
this project (the `block` metric ranked first on margin and captured 39.9% of signing).

    python3 tune_short_calibration.py

Reports the grid; the pair with 0 false starts across every window at the highest coverage
is what fit_thresholds.SHORT_* should hold.
"""
import json
import statistics

import fit_thresholds as ft
import simulate_segmentation as sim

FPS = 30
STILL_SECONDS = 6
SIGN_SECONDS = 9
# Candidate upper-tail statistics. 1.0 is the maximum, i.e. what the long-form fit uses.
NEG_QUANTILES = [0.95, 0.98, 0.99, 0.995, 1.0]
POS_QUANTILES = [0.90, 0.95, 0.98, 0.99, 1.0]


def phase_spans(rows):
    spans = []
    for r in rows:
        if not spans or r["phase"] != spans[-1][0]:
            spans.append((r["phase"], []))
        spans[-1][1].append(r)
    return spans


def short_windows(rows):
    """Every (still slice, signing slice) pair a short calibration might have captured."""
    spans = phase_spans(rows)
    still = [r for name, vals in spans if ft.is_negative(name) for r in vals]
    sign = [r for name, vals in spans if not ft.is_negative(name) for r in vals]
    out = []
    for so in range(0, max(1, len(still) - STILL_SECONDS * FPS), 90):
        for go in range(0, max(1, len(sign) - SIGN_SECONDS * FPS), 150):
            out.append((still[so:so + STILL_SECONDS * FPS],
                        sign[go:go + SIGN_SECONDS * FPS]))
    return out


def fit_short(still_rows, sign_rows, neg_q, pos_q, window=15):
    """Fit from quantiles rather than max-vs-event-peak."""
    neg = ft.smooth([r["current"] for r in still_rows], window)
    pos = ft.smooth([r["current"] for r in sign_rows], window)
    if not neg or not pos:
        return None
    neg_hi = ft.quantile(neg, neg_q)
    pos_hi = ft.quantile(pos, pos_q)
    if pos_hi <= neg_hi:
        # No separation in this sample. Sit just above the non-signing side: refusing to
        # trigger is recoverable (sign again), a threshold underneath it is not.
        return neg_hi * 1.05
    return (neg_hi * pos_hi) ** 0.5


def main():
    rows = sim.load("calib.jsonl")
    windows = short_windows(rows)
    full_start, _, _ = ft.fit_rows(rows, window=15)
    print(f"{len(windows)} short windows ({STILL_SECONDS}s still + {SIGN_SECONDS}s sign); "
          f"full-recording fit start {full_start:.2f}\n")
    print(f"{'neg_q':>6}{'pos_q':>7}{'start med':>11}{'stdev':>8}"
          f"{'cover%':>9}{'false/win':>11}{'clean win':>11}")
    print("-" * 63)

    best = []
    for neg_q in NEG_QUANTILES:
        for pos_q in POS_QUANTILES:
            starts, covers, falses = [], [], []
            for still_rows, sign_rows in windows:
                start = fit_short(still_rows, sign_rows, neg_q, pos_q)
                if start is None:
                    continue
                profile = {"start_threshold": start,
                           "stop_threshold": start * ft.STOP_RATIO, "floor": None}
                clips, _ = sim.simulate(rows, adaptive=True, profile=profile)
                s = sim.score(rows, clips)
                starts.append(start)
                covers.append(s["coverage"])
                falses.append(s["false_starts"])
            if not starts:
                continue
            clean = 100.0 * sum(1 for f in falses if f == 0) / len(falses)
            row = (neg_q, pos_q, statistics.median(starts), statistics.pstdev(starts),
                   statistics.mean(covers), statistics.mean(falses), clean)
            best.append(row)
            print(f"{neg_q:>6}{pos_q:>7}{row[2]:>11.2f}{row[3]:>8.2f}"
                  f"{row[4]:>9.1f}{row[5]:>11.2f}{clean:>10.0f}%")

    # Rank: never false-start first, then capture the most signing, then be stable.
    best.sort(key=lambda r: (-r[6], -r[4], r[3]))
    n_q, p_q, med, sd, cov, fa, clean = best[0]
    print(f"\nbest: neg_q {n_q} / pos_q {p_q} -> median start {med:.2f}, stdev {sd:.2f}, "
          f"{cov:.1f}% coverage, {clean:.0f}% of windows with zero false starts")


if __name__ == "__main__":
    main()
