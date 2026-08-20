"""Fit start/stop thresholds from a guided calibration recording, and show the working.

WHY THIS EXISTS
---------------
`motion_gate.py` infers the start threshold as floor x 9.0, where the floor is the room's
online noise estimate. That measures only the STILL side of the problem and guesses the
signing side from a multiplier fitted in one room. Measured live 2026-08-12: a noisy floor
of 0.785 put the start threshold at 7.06 while the highest smoothed score signing ever
produced in calibration is 1.889 -- unreachable by construction, and the user could not
start a second sentence. Measure both sides instead of inferring one from the other.

WHAT THE CALIBRATION DATA ACTUALLY SAYS (calib.jsonl, 2534 labelled frames, this room)
--------------------------------------------------------------------------------------
The decision is NOT "still vs signing". Those are 4.4x apart and easy:

    genuinely settled stillness   <= 0.36      (2nd still span, after frame 200)
    signing / fingerspell peaks   1.45 - 1.89

The hard pair is SIGNING vs REPOSITIONING -- shifting in the seat, scratching, settling
into frame. That reaches 1.54 in the same recording, overlapping signing almost entirely,
and it is what forced the shipped threshold up to 1.74. The old calibration never labelled
it, so those frames landed in the "SIT STILL" bucket and inflated the still maximum from
0.36 to 1.54, which is the whole reason the still-only fit lands on top of signing.

So the calibration prompts a NEGATIVE motion phase ("move but do not sign") and the fit
separates NEGATIVE (still + non-signing motion) from POSITIVE (signing), which is the
decision the machine actually makes. If they overlap, that is a fact about a global
pixel-difference mean and the fit says so rather than shipping a threshold that cannot
work; the hand-presence veto in hand_trigger.py is what covers the residue.

Run standalone against any recording to see the numbers:

    python3 fit_thresholds.py calib.jsonl
"""
import json
import math

# Seconds discarded at the start of a POSITIVE phase: the signer spends the first moments
# reading the new prompt rather than signing, and the trailing mean still holds the
# previous phase. Those frames would otherwise depress the measured signing peaks.
#
# IT IS DELIBERATELY NOT APPLIED TO NEGATIVE PHASES, and that asymmetry is the whole
# design. Trimming both sides was tried first and produced a false start in simulation:
# the head of a "sit still" phase is where the signer settles into position, which reaches
# 1.539 on calib.jsonl against a 1.440 trimmed maximum, so the trim removed the very
# motion the threshold has to clear and placed it 0.09 underneath. Everything that happens
# while the prompt says do-not-sign is, by definition, motion that must not trigger.
SETTLE_SECONDS = 3.0

# A threshold placement is only reported as safe if the two classes are this far apart.
# Below it, a modest change (lighting, signer distance) crosses the boundary.
MIN_SEPARATION = 1.30

# Stop threshold as a fraction of start. Preserved from the hand-fitted pair validated
# end-to-end in this room (start 1.74, stop 0.29 = 1/6.0). The stop side is far less
# delicate than the start side: STILL_DURATION_SECONDS already protects against cutting on
# a hold, and back-dating protects the head of the clip.
STOP_RATIO = 1.0 / 6.0

# Phases whose motion must NOT trigger a clip. Matched as a prefix of the phase label.
# "SIT STILL" is kept alongside "STAND STILL" so calibration logs recorded before the
# prompt was reworded still fit -- a still phase that fails this test is treated as
# POSITIVE (signing), which silently poisons the fit rather than erroring.
NEGATIVE_PHASES = ("STAND STILL", "SIT STILL", "MOVE")
# The subset of those that is pure stillness. The rest ("MOVE") is deliberate non-signing
# MOTION, and the difference matters in the hand-gated path below: stillness is what a
# pixel threshold can still separate when signing has fallen into the fidget distribution.
STILL_PHASES = ("STAND STILL", "SIT STILL")

# How far above the still maximum the SIGN phase has to reach before the recording counts
# as evidence that anyone signed at all. Below this the two failures are indistinguishable
# -- an empty SIGN phase and a signer whose signing does not register -- and re-prompting
# is worth the 28s. Above it the signer demonstrably WAS signing, so a second identical
# calibration only wastes their time and produces the same failure. That distinction is
# what stopped the 2026-08-16 standing session re-running calibration twice and then
# telling the user to "sign as you normally would", which they had been doing.
SIGNING_EVIDENCE_RATIO = 1.30


def smooth(values, window):
    """Trailing mean, matching MotionGate's deque exactly."""
    out = []
    acc = []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def quantile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def event_peaks(values, threshold, min_gap):
    """Peak of each contiguous run above `threshold`, runs separated by `min_gap` frames.

    A movement phase is not one event: "raise your right arm, then lower it, repeat" is
    several distinct motions with quiet between them. What matters for the start threshold
    is the WEAKEST of those peaks -- the least emphatic movement that must still trigger.
    A phase mean would be dominated by the quiet stretches between events instead.
    """
    peaks = []
    cur = None
    gap = 0
    for v in values:
        if v > threshold:
            cur = v if cur is None else max(cur, v)
            gap = 0
        elif cur is not None:
            gap += 1
            if gap >= min_gap:
                peaks.append(cur)
                cur = None
    if cur is not None:
        peaks.append(cur)
    return peaks


def is_negative(phase):
    return any(phase.upper().startswith(p) for p in NEGATIVE_PHASES)


def is_still(phase):
    return any(phase.upper().startswith(p) for p in STILL_PHASES)


def longest_run_below(values, threshold):
    """Longest contiguous run at or below `threshold`, in frames.

    This is what the stop threshold is actually judged on. A fraction of frames below the
    threshold says nothing: ending a clip needs STILL_DURATION_SECONDS of CONSECUTIVE
    quiet, and holding a sign mid-sentence produces exactly that if the threshold is too
    high. So report the run lengths on both sides and let the caller see the margin.
    """
    run = best = 0
    for v in values:
        run = run + 1 if v <= threshold else 0
        best = max(best, run)
    return best


def fit(negative, positive, fps=30.0):
    """Fit thresholds. `negative`/`positive` are dicts of phase label -> smoothed scores.

    Returns (start, stop, report). `start` is None only if nothing positive was recorded.
    """
    neg_all = [v for vals in negative.values() for v in vals]
    if not neg_all:
        return None, None, {"ok": False, "reason": "no negative (non-signing) frames"}

    # The negative side is summarised by its MAXIMUM, not a quantile: one negative frame
    # above the threshold is one false start, and a p99 tolerates one frame in 100 -- more
    # than three per second at 30fps.
    neg_max = max(neg_all)
    neg_p50 = quantile(neg_all, 0.50)
    neg_by_phase = {k: (max(v) if v else 0.0) for k, v in negative.items()}

    # Positive events are delimited against the negative maximum, so "an event" means
    # "rose above anything non-signing did" -- no second free parameter to tune.
    phase_peaks = {}
    for name, vals in positive.items():
        phase_peaks[name] = event_peaks(vals, neg_max, min_gap=int(0.5 * fps))
    all_peaks = [p for peaks in phase_peaks.values() for p in peaks]

    report = {
        "neg_max": neg_max, "neg_p50": neg_p50, "neg_by_phase": neg_by_phase,
        "phase_peaks": {k: sorted(v) for k, v in phase_peaks.items()},
        "n_events": len(all_peaks),
    }

    if not all_peaks:
        # No signing event cleared the non-signing maximum, so this recording contains no
        # evidence of where the boundary is. REFUSE rather than emit a number.
        #
        # The first version returned sqrt(neg_p50 * pos_peak) here, reasoning that some
        # threshold beats none. Measured on a calibration run where nobody signed: it
        # produced start 0.769 against a non-signing maximum of 2.615 -- a value certain
        # to fire on the very motion it was fitted to reject, presented to the user as a
        # successful calibration. A threshold placed above neg_max instead would be the
        # opposite failure (nothing ever triggers) and just as silent. Both are worse than
        # saying so: the caller falls back and can ask for the calibration again.
        pos_all = [v for vals in positive.values() for v in vals]
        pos_max = max(pos_all) if pos_all else 0.0
        still_all = [v for name, vals in negative.items() if is_still(name) for v in vals]
        still_max = max(still_all) if still_all else neg_max
        report.update({
            "ok": False,
            "weakest_move_peak": pos_max,
            "separation": (pos_max / neg_max) if neg_max > 0 else 1.0,
            "still_max": still_max,
        })

        if pos_max <= still_max * SIGNING_EVIDENCE_RATIO:
            # Nothing in the SIGN phase rose meaningfully above sitting/standing still, so
            # the recording carries no evidence of where the boundary is. Worth asking for
            # again -- the common cause is a signer who missed the prompt.
            # Two very different recordings land here, and telling a signer the wrong one
            # sends them back to repeat what they were already doing. If the STILL phase
            # was nearly as loud as the MOVE phase, the problem is at the other end: the
            # still baseline is contaminated (standing still is genuinely harder than
            # sitting still -- measured on the first of the 2026-08-16 attempts, STAND
            # STILL reached 0.617 against MOVE's 0.687), and every threshold derived from
            # it is inflated.
            contaminated = still_max >= 0.80 * neg_max
            report.update({
                "retry_worthy": True,
                "contaminated_still": contaminated,
                "reason": (
                    (f"the STILL phase recorded almost as much motion ({still_max:.3f}) "
                     f"as the MOVE phase ({neg_max:.3f}) — the still baseline is "
                     f"contaminated, so nothing can rise above it. Settle into position "
                     f"BEFORE the countdown and hold still through it.")
                    if contaminated else
                    (f"no signing rose above ordinary movement (which reached "
                     f"{neg_max:.3f}) — was anyone signing during the SIGN phase?")),
            })
            return None, None, report

        # Signing WAS measured (it cleared stillness by more than SIGNING_EVIDENCE_RATIO)
        # but it never cleared non-signing MOVEMENT. That is not a bad recording, it is a
        # fact about the metric at this distance: a signer far enough from the camera moves
        # a smaller share of the frame with their hands than with their whole body, so
        # shifting your weight outscores a sentence. Measured 2026-08-16, standing:
        # signing peaked at 0.497 smoothed against 0.514 for "move but do not sign".
        #
        # Refusing outright is what used to happen, and the caller's fallback -- floor x
        # START_MULTIPLIER -- then ignored this measurement entirely and produced a start
        # threshold of 1.7 against signing that never exceeded 0.5. Unreachable, which is
        # the same failure as the 7.06 case that motivated this whole module.
        #
        # So emit a pair fitted to the STILL side only, and say plainly that it is only
        # safe behind the hand-presence veto: it WILL fire on ordinary movement (that is
        # unavoidable -- signing sits inside that distribution), and hand presence is what
        # separates the two. On the same recording the veto measured 0 detections in 184
        # non-signing checks against 32 in 101 signing checks.
        start = math.sqrt(still_max * pos_max)
        stop = math.sqrt(quantile(still_all, 0.50) * quantile(pos_all, 0.50))
        report.update({
            "retry_worthy": False,
            "hand_gated": True,
            "reason": ("signing never cleared ordinary movement — this metric cannot "
                       "separate them at this distance; thresholds fitted to stillness "
                       "instead and ONLY valid with the hand-presence veto on"),
            "negative_frames_above_start":
                sum(1 for v in neg_all if v > start) / len(neg_all),
            "still_frames_above_start":
                sum(1 for v in still_all if v > start) / len(still_all) if still_all else 0,
            # The stop side is the half that decides whether a clip ever ENDS. Both runs
            # are reported because the threshold has to sit between them, and when signing
            # falls into the still distribution there may be no gap left at all.
            "stop_still_quiet_frames": longest_run_below(still_all, stop),
            "stop_signing_quiet_frames": longest_run_below(pos_all, stop),
        })
        return start, stop, report

    # The weakest event that must trigger -- a low quantile rather than the raw minimum,
    # so one half-hearted movement cannot drag the ceiling down onto the floor.
    weakest = quantile(all_peaks, 0.10)
    separation = weakest / neg_max if neg_max > 0 else float("inf")

    # Geometric midpoint: equal proportional headroom on both sides. An arithmetic
    # midpoint sits closer to the negative side in ratio terms, and that is the side that
    # produces false starts.
    start = math.sqrt(neg_max * weakest)
    stop = start * STOP_RATIO

    # Concrete consequences of this placement, which is more useful than a pass/fail:
    # how much of the negative side would trigger, and how many positive events would not.
    false_frac = sum(1 for v in neg_all if v > start) / len(neg_all)
    missed = sum(1 for p in all_peaks if p <= start)
    report.update({
        "ok": separation >= MIN_SEPARATION,
        "reason": ("" if separation >= MIN_SEPARATION else
                   f"non-signing motion and signing overlap ({separation:.2f}x < "
                   f"{MIN_SEPARATION}x) -- a global pixel-difference mean cannot separate "
                   f"them in this room; keep LANDMARK_TRIGGER on"),
        "weakest_move_peak": weakest, "separation": separation,
        "negative_frames_above_start": false_frac,
        "events_below_start": missed,
    })
    return start, stop, report


def fit_rows(rows, metric="current", window=15, fps=30.0):
    """Fit directly from calibration rows ({metric, phase, t}), splitting by phase span."""
    spans = []
    for r in rows:
        if not spans or r["phase"] != spans[-1][0]:
            spans.append((r["phase"], []))
        spans[-1][1].append(r[metric])

    negative, positive = {}, {}
    trim = int(SETTLE_SECONDS * fps)
    for name, vals in spans:
        # Smooth WITHIN the span so a phase boundary cannot leak the previous phase's
        # motion into this phase's samples. The settle-in margin comes off positive
        # phases only -- see SETTLE_SECONDS.
        sm = smooth(vals, window)
        if is_negative(name):
            negative.setdefault(name, []).extend(sm)
        elif len(sm) > trim:
            positive.setdefault(name, []).extend(sm[trim:])
    return fit(negative, positive, fps=fps)


def describe(start, stop, rep):
    lines = []
    lines.append(f"non-signing: max {rep['neg_max']:.3f}  median {rep['neg_p50']:.3f}")
    for name, mx in rep.get("neg_by_phase", {}).items():
        lines.append(f"    {name:<34} max {mx:.3f}")
    for name, peaks in rep.get("phase_peaks", {}).items():
        shown = ", ".join(f"{p:.2f}" for p in peaks[:10]) or "(none above non-signing max)"
        lines.append(f"    {name:<34} {len(peaks):>2} events: {shown}")
    if start is None:
        lines.append(f"FAILED: {rep['reason']}")
        return "\n".join(lines)
    if rep.get("hand_gated"):
        # A different fit with different guarantees, so it is reported differently rather
        # than dressed up as a normal one with a poor separation number.
        lines.append(f"DEGRADED (hand-gated): {rep['reason']}")
        lines.append(f"    signing peaked at {rep.get('weakest_move_peak', 0):.3f}, "
                     f"non-signing movement at {rep['neg_max']:.3f}, "
                     f"stillness at {rep.get('still_max', 0):.3f}")
        lines.append(f"    at start {start:.3f}: "
                     f"{100 * rep.get('still_frames_above_start', 0):.1f}% of STILL frames "
                     f"and {100 * rep.get('negative_frames_above_start', 0):.1f}% of all "
                     f"non-signing frames are above it (the hand veto has to reject those)")
        quiet_still = rep.get("stop_still_quiet_frames", 0)
        quiet_sign = rep.get("stop_signing_quiet_frames", 0)
        lines.append(f"    at stop {stop:.3f}: longest quiet run is {quiet_still} frames "
                     f"({quiet_still / 30:.1f}s) when still, {quiet_sign} frames "
                     f"({quiet_sign / 30:.1f}s) while signing "
                     f"— a clip ends after 2.5s (75 frames) of quiet, so this "
                     + ("cuts mid-sentence" if quiet_sign >= 75 else
                        "never ends a clip" if quiet_still < 75 else "should cut cleanly"))
        lines.append(f"=> start {start:.3f}  stop {stop:.3f}")
        return "\n".join(lines)
    lines.append(f"weakest signing event (p10 of {rep['n_events']}): "
                 f"{rep.get('weakest_move_peak', 0):.3f}")
    lines.append(f"separation: {rep.get('separation', 0):.2f}x "
                 f"{'OK' if rep['ok'] else '-- TOO NARROW'}")
    if not rep["ok"]:
        lines.append(f"    {rep['reason']}")
    if "negative_frames_above_start" in rep:
        lines.append(f"predicted at this threshold: "
                     f"{100 * rep['negative_frames_above_start']:.2f}% of non-signing "
                     f"frames above start, {rep['events_below_start']} of "
                     f"{rep['n_events']} signing events below it")
    lines.append(f"=> start {start:.3f}  stop {stop:.3f}")
    return "\n".join(lines)


def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--metric", default="current")
    ap.add_argument("--window", type=int, default=15)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path)]
    start, stop, rep = fit_rows(rows, metric=args.metric, window=args.window)
    print(describe(start, stop, rep))
    print("   (hand-fitted pair validated in this room: start 1.74  stop 0.29)")


if __name__ == "__main__":
    _main()
