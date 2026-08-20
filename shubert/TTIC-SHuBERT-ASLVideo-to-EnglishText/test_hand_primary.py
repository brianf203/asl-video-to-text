"""Tests for the standing-distance segmentation path (2026-08-16).

WHAT BROKE
----------
A standing signer calibrated three times and got a start threshold of 1.7 against signing
that never exceeded 0.5 smoothed. Three separate faults, one per test below:

1. The fit REFUSED (signing never cleared non-signing movement) and the caller fell back
   to floor x START_MULTIPLIER, which ignores the calibration entirely -- the same
   unreachable-threshold failure as the 7.06 case that motivated fit_thresholds.py.
2. The refusal was reported as "was anyone signing?", so calibration re-ran and asked the
   signer to sign as they normally would, which is what they had been doing.
3. The pre-gated hand veto cannot rescue it: a pixel threshold safe against stillness is
   cleared only in short bursts (4 of them, 22.5% of a 9s signing phase), so the detector
   is fed too rarely. Measured on the frozen recording: 4 starts against hand-primary's
   18. Starvation rather than impossibility -- a 0.5s burst is just enough verdicts to
   fill the 5-slot window -- but 4 chances to begin a sentence in 9s of signing is not a
   usable trigger.

The measured recording is a COMMITTED fixture (see fixtures/README.md). It used to read
`last_calibration.jsonl`, which every launch overwrites -- a demo run replaced the analysed
recording an hour after the analysis, and this test then failed against the note's own
numbers. Point CALIB_LOG at a fresh recording to re-check a new room.
The synthetic fixtures below reproduce the same structure so this runs anywhere.
"""
import json
import os
from collections import deque

import fit_thresholds as ft
from motion_gate import ARMING_SECONDS

_HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_LOG = os.environ.get(
    "CALIB_LOG", os.path.join(_HERE, "fixtures", "standing_calibration_20260816.jsonl"))
HAND_NEEDED, HAND_WINDOW = 3, 5


def synthetic_rows(sign_hands=True):
    """A standing-distance calibration: signing sits INSIDE the fidget distribution.

    Levels are taken from the 2026-08-16 recording -- still ~0.145, non-signing movement
    ranging to ~0.51, signing to ~0.50 -- because the whole point is that signing does not
    clear movement while it does clear stillness.
    """
    rows, t = [], 1000.0

    def phase(label, values, hands):
        nonlocal t
        for i, v in enumerate(values):
            rows.append({"current": v, "phase": label, "t": t,
                         "hands": {"t": t, "n_hands": 2 if hands(i) else 0, "wrists": []}})
            t += 1 / 30.0

    phase("STAND STILL", [0.145 + 0.02 * (i % 3) for i in range(180)], lambda i: False)
    # Movement in bursts, as a person shifting their weight actually is.
    phase("MOVE (NO SIGNING)",
          [0.55 if (i // 15) % 2 else 0.16 for i in range(210)], lambda i: False)
    # Signing: peaks below the movement peaks, hands visible ~50% of the time (measured
    # 46.9% standing -- they drop between signs).
    phase("SIGN", [0.50 if (i // 10) % 2 else 0.15 for i in range(270)],
          (lambda i: sign_hands and (i // 10) % 2 == 1))
    return rows


def load_real():
    if not os.path.exists(CALIB_LOG):
        return None
    return [json.loads(line) for line in open(CALIB_LOG)]


def test_overlap_is_hand_gated_not_a_retry():
    """Signing measured but inside the fidget distribution -> degraded pair, no retry."""
    for name, rows in (("synthetic", synthetic_rows()), ("recorded", load_real())):
        if rows is None:
            print("  (no last_calibration.jsonl — skipping the recorded case)")
            continue
        start, stop, rep = ft.fit_rows(rows, window=15)
        assert start is not None, f"{name}: refused instead of fitting a hand-gated pair"
        assert rep.get("hand_gated"), f"{name}: not flagged hand_gated"
        assert not rep.get("retry_worthy"), f"{name}: would re-run calibration pointlessly"
        assert not rep["ok"], f"{name}: a degraded fit must not report as separable"
        # Above stillness (or it would fire constantly), below the movement it cannot
        # separate (or it would never fire at all).
        assert rep["still_max"] < start < rep["neg_max"], (
            f"{name}: start {start:.3f} outside (still {rep['still_max']:.3f}, "
            f"move {rep['neg_max']:.3f})")
        print(f"  {name}: start {start:.3f} stop {stop:.3f}  "
              f"(still {rep['still_max']:.3f}, move {rep['neg_max']:.3f}, "
              f"signing {rep['weakest_move_peak']:.3f})")


def test_no_signing_still_refuses_and_retries():
    """The genuinely-empty SIGN phase must keep its refusal AND ask for another run."""
    rows = [r for r in synthetic_rows() if r["phase"] != "SIGN"]
    still = [r for r in rows if r["phase"] == "STAND STILL"]
    rows += [dict(r, phase="SIGN") for r in still + still]
    start, _, rep = ft.fit_rows(rows, window=15)
    assert start is None, "emitted a threshold from a recording with no signing in it"
    assert rep.get("retry_worthy"), "would not re-prompt a signer who missed the phase"
    print("  refused, retry_worthy=True")


def test_separable_room_unaffected():
    """A recording where signing DOES clear movement must take the normal path."""
    rows, t = [], 1000.0
    for label, values in (("SIT STILL", [0.2] * 180),
                          ("MOVE (NO SIGNING)", [0.5 if (i // 15) % 2 else 0.2
                                                 for i in range(210)]),
                          ("SIGN", [1.8 if (i // 10) % 2 else 0.3 for i in range(270)])):
        for v in values:
            rows.append({"current": v, "phase": label, "t": t})
            t += 1 / 30.0
    start, stop, rep = ft.fit_rows(rows, window=15)
    assert start is not None and not rep.get("hand_gated"), "took the degraded path"
    assert rep["ok"], "a cleanly separable room should report separable"
    print(f"  normal fit: start {start:.3f} stop {stop:.3f} "
          f"separation {rep['separation']:.2f}x")


def simulate_starts(rows, phase_wanted, pre_gate=None, start_threshold=0.0):
    """Replay the IDLE start decision over one phase. `pre_gate=None` is hand-primary.

    `rows` is the WHOLE recording, not the phase, because the arming window below is
    measured from the first frame the program ever saw.
    """
    t0 = rows[0]["t"]
    span = [r for r in rows if r["phase"] == phase_wanted]
    smoothed = ft.smooth([r["current"] for r in span], 15)
    recent, above, starts, last_t = deque(maxlen=HAND_WINDOW), 0, 0, None
    for r, s in zip(span, smoothed):
        hot = True
        if pre_gate:
            hot = s > start_threshold
            above = above + 1 if hot else 0
        if hot:
            h = r.get("hands") or {}
            if h.get("t") != last_t:     # a new detector verdict landed
                last_t = h.get("t")
                recent.append(1 if h.get("n_hands", 0) else 0)
        elif pre_gate:
            recent.append(0)             # note_absent(), at the camera's rate
        # `armed` in v5's IDLE branch: with a profile the thresholds are known at frame 0,
        # so the only wait is ARMING_SECONDS -- whoever launched the program is still
        # settling into frame, hands up, and that is motion at signing levels. Modelled
        # here because it is the ONLY thing standing between hand-primary and a false
        # start on this recording: all 13 of its STAND STILL detections land in the first
        # 1.30s. Note the detector keeps being fed while unarmed, exactly as v5 feeds it
        # regardless of `armed` -- suppressing the verdicts instead would flatter the
        # result by emptying the window.
        armed = (r["t"] - t0) >= ARMING_SECONDS
        due = armed and sum(recent) >= HAND_NEEDED and (not pre_gate or above >= 3)
        if due:
            starts += 1
            recent.clear()
            above = 0
    return starts, len(span) / 30.0


def test_hand_primary_starts_where_the_pre_gate_cannot():
    for name, rows in (("synthetic", synthetic_rows()), ("recorded", load_real())):
        if rows is None:
            continue
        start, _, _ = ft.fit_rows(rows, window=15)
        gated, _ = simulate_starts(rows, "SIGN", pre_gate=True, start_threshold=start)
        primary, secs = simulate_starts(rows, "SIGN")
        assert primary >= 1, f"{name}: hand-primary never started on real signing"
        print(f"  {name}: {secs:.0f}s of signing -> pre-gated {gated} starts, "
              f"hand-primary {primary}")
        for phase in ("STAND STILL", "MOVE (NO SIGNING)"):
            false_starts, secs = simulate_starts(rows, phase)
            assert false_starts == 0, (
                f"{name}: {false_starts} false start(s) during {phase}")
            print(f"  {name}: {secs:.0f}s of {phase:<18} -> 0 false starts "
                  f"(unarmed for the first {ARMING_SECONDS:.0f}s, as v5 is)")


if __name__ == "__main__":
    for test in (test_overlap_is_hand_gated_not_a_retry,
                 test_no_signing_still_refuses_and_retries,
                 test_separable_room_unaffected,
                 test_hand_primary_starts_where_the_pre_gate_cannot):
        print(f"{test.__name__}:")
        test()
    print("\nall hand-primary tests passed")
