#!/usr/bin/env python3
"""Regression tests for MotionGate's floor estimator.

The scenario that matters is the one the calibration recording does NOT contain: a signer
moving while the machine is still IDLE (getting ready, adjusting, a false start that did
not take). The first version sampled every idle frame into the floor, so that motion
raised the floor, which raised the threshold, which made triggering harder -- a runaway
that a live signer hit immediately and every offline test missed.

Run: python3 test_motion_gate.py
"""
import json
import sys
from collections import deque

import motion_gate
from motion_gate import MotionGate

SMOOTHING = 15
FIXED_START, FIXED_STOP = 1.74, 0.29
FPS = 30.0


def load_phases(path="calib.jsonl"):
    rows = [json.loads(line) for line in open(path)]
    buf = deque(maxlen=SMOOTHING)
    for r in rows:
        buf.append(r["current"])
        r["sm"] = sum(buf) / len(buf)
    still = [r["current"] for r in rows if r["phase"] == "SIT STILL"]
    moving = [r["current"] for r in rows if r["phase"] in ("SIGN", "FINGERSPELL")]
    signing_max = max(r["sm"] for r in rows if r["phase"] in ("SIGN", "FINGERSPELL"))
    return still, moving, signing_max


def run(gate, scores, recording=False, t0=0.0):
    t = t0
    for s in scores:
        gate.update(s, recording=recording, now=t)
        t += 1.0 / FPS
    return t


def main():
    still, moving, signing_max = load_phases()
    print(f"signing smoothed max = {signing_max:.3f}  "
          f"(a start threshold above this can never fire)\n")
    failures = []

    # --- 1. quiet room: the floor lands where the fitted config expects -------------
    g = MotionGate(SMOOTHING, FIXED_START, FIXED_STOP)
    run(g, still)
    print(f"[still only]      floor {g.floor:.3f}  start {g.start_threshold:.2f}")
    if not (0.5 < g.start_threshold < signing_max):
        failures.append("still-only start threshold is not in a usable range")

    # --- 2. THE REGRESSION: moving while idle must not run the threshold away -------
    g = MotionGate(SMOOTHING, FIXED_START, FIXED_STOP)
    t = run(g, still)
    before = g.start_threshold
    # 30s of signing-level motion with the machine never recording.
    run(g, moving * 2, recording=False, t0=t)
    after = g.start_threshold
    print(f"[idle but moving] floor {g.floor:.3f}  start {before:.2f} -> {after:.2f}")
    if after > signing_max:
        failures.append(f"RUNAWAY: start threshold {after:.2f} exceeds the highest score "
                        f"signing can produce ({signing_max:.3f}) -- unreachable")
    if after > before * 2.0:
        failures.append(f"floor inflated {after / before:.1f}x under idle motion")

    # --- 3. the old behaviour still reproduces the bug, so the test has teeth -------
    saved = motion_gate.QUIET_ACCEPT_FACTOR
    motion_gate.QUIET_ACCEPT_FACTOR = 1e9      # accept everything = the old estimator
    g = MotionGate(SMOOTHING, FIXED_START, FIXED_STOP)
    t = run(g, still)
    run(g, moving * 2, recording=False, t0=t)
    old = g.start_threshold
    motion_gate.QUIET_ACCEPT_FACTOR = saved
    print(f"[old estimator]   floor {g.floor:.3f}  start {old:.2f}  "
          f"{'(unreachable - the reported bug)' if old > signing_max else ''}")
    if old <= signing_max:
        failures.append("control case did not reproduce the runaway; test proves nothing")

    # --- 4. a genuinely noisier room must still be trackable ------------------------
    # Quiet-only sampling could deadlock: if nothing ever looks quiet the floor can never
    # rise. QUIET_STALE_SECONDS is the escape hatch; check it works.
    g = MotionGate(SMOOTHING, FIXED_START, FIXED_STOP)
    run(g, [s * 4 for s in still])
    print(f"[4x noisier room] floor {g.floor:.3f}  start {g.start_threshold:.2f}")
    if g.floor is None:
        failures.append("no floor established in a noisier room (deadlock)")
    elif g.start_threshold < 1.0:
        failures.append("threshold did not rise with a noisier room")

    # --- 5. recording frames never feed the floor ----------------------------------
    g = MotionGate(SMOOTHING, FIXED_START, FIXED_STOP)
    t = run(g, still)
    quiet_floor = g.floor
    run(g, moving * 2, recording=True, t0=t)
    print(f"[during a clip]   floor {quiet_floor:.3f} -> {g.floor:.3f} (must not move)")
    if abs(g.floor - quiet_floor) > 1e-9:
        failures.append("floor moved while recording")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("all motion gate tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
