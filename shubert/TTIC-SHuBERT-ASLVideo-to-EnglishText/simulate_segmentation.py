#!/usr/bin/env python3
"""Replay the segmentation state machine over labelled calibration data.

The 2026-08-10 calibration established that per-frame metric statistics rank configs
WRONGLY: a safety-margin sweep put `block` smoothed-5 first, and simulating the actual
state machine then gave it 39.9% signing coverage. Only the end-to-end acceptance test
found the right config. So any threshold change gets judged here, on the real decision
rule, before it goes near a camera.

Acceptance metrics (same four as that session):
  false starts  -- clips begun during a labelled SIT STILL phase
  coverage      -- fraction of labelled signing frames inside a kept span
  fragments     -- signing phases broken into more than one clip
  lead-in lost  -- signing time before the first kept frame of each clip

Usage:
    python3 simulate_segmentation.py                   # fixed vs adaptive, this room
    python3 simulate_segmentation.py --scale 0.5 2 4   # simulate other rooms
"""
import argparse
import json
import os

import fit_thresholds
from motion_gate import MotionGate, load_profile

HERE = os.path.dirname(os.path.abspath(__file__))

# Mirrors auto_segment_v5.py. Imported rather than duplicated where possible; these are
# the plain constants the state machine below needs.
import auto_segment_v5 as v5

SIGNING_PHASES = ("SIGN", "FINGERSPELL")


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_hand_presence(path):
    """Per-frame hand presence from analyze_hand_trigger.py, indexed by frame."""
    if not os.path.exists(path):
        return None
    present = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            present[r["i"]] = r["n_hands"] > 0
    return present


def simulate(rows, adaptive, scale=1.0, metric="current", hands=None,
             pregate_factor=1.0, profile=None):
    """Run v5's state machine over the rows. Returns per-clip spans and gate history.

    `hands` enables the hybrid trigger: the pixel threshold is divided by
    `pregate_factor` (so it becomes a sensitive, high-recall pre-gate) and a start also
    requires hand presence confirmed over the rolling window.
    """
    gate = MotionGate(v5.MOTION_SMOOTHING_FRAMES,
                      v5.MOTION_START_THRESHOLD, v5.MOTION_STOP_THRESHOLD,
                      adaptive=adaptive, profile=profile)
    from collections import deque as _deque
    from hand_trigger import HAND_WINDOW, HAND_NEEDED
    hand_recent = _deque(maxlen=HAND_WINDOW)
    state = "IDLE"
    above_start = 0
    record_start_i = None
    last_motion_i = None
    clips = []
    floors = []
    smoothed_hist = []
    trigger_i = None       # where the debounce fired, i.e. the first non-seeded frame

    for i, r in enumerate(rows):
        raw = r[metric] * scale
        t = r["t"]
        # Pass the recording's own clock: the floor's staleness logic is time-based,
        # and wall-clock time would race ahead of the replayed data.
        smoothed = gate.update(raw, recording=(state == "RECORDING"), now=t)
        floors.append(gate.floor)
        smoothed_hist.append(smoothed)

        if state == "IDLE":
            # With the hybrid the pixel test becomes a sensitive pre-gate; hands supply
            # the precision, so the threshold no longer has to sit inside the narrow
            # still-max/signing-max window.
            threshold = gate.start_threshold / pregate_factor
            above_start = above_start + 1 if smoothed > threshold else 0
            if hands is not None:
                # Detection only runs on frames the pre-gate lets through; unchecked
                # frames count as absent, mirroring HandPresenceTrigger.note_absent().
                hand_recent.append(1 if (above_start and hands.get(i, False)) else 0)
            hands_ok = (hands is None or sum(hand_recent) >= HAND_NEEDED)
            # The interlock: in adaptive mode nothing records until a floor exists.
            if (above_start >= v5.START_DEBOUNCE_FRAMES and hands_ok
                    and (not adaptive or gate.ready)):
                above_start = 0
                hand_recent.clear()
                state = "RECORDING"
                # Back-date exactly as the live path does, using v5's own function and
                # the gate's CURRENT stop threshold -- without this the simulation loses
                # the lead-in the real machine recovers, and coverage reads far too low.
                lo = max(0, i - int(v5.PRE_ROLL_MAX_SECONDS * 30))
                ring = [(None, rows[j]["t"], rows[j][metric] * scale)
                        for j in range(lo, i + 1)]
                onset = v5.find_motion_onset(ring, stop_threshold=gate.stop_threshold)
                onset = max(0, onset - int(v5.LEAD_PAD_SECONDS * 30))
                record_start_i = lo + onset
                last_motion_i = i
                trigger_i = i
        else:
            if smoothed > gate.stop_threshold:
                last_motion_i = i
            still_duration = t - rows[last_motion_i]["t"]
            elapsed = t - rows[record_start_i]["t"]
            if still_duration >= v5.STILL_DURATION_SECONDS or elapsed >= v5.MAX_CLIP_SECONDS:
                forced = elapsed >= v5.MAX_CLIP_SECONDS
                end_i = i if forced else last_motion_i
                signing = rows[end_i]["t"] - rows[record_start_i]["t"]
                # Mirror v5's empty-clip test: the fraction of KEPT, non-seeded frames
                # that are actually moving. Frames before the trigger are the back-dated
                # lead-in, which v5 excludes for the same reason.
                lo_f = trigger_i if trigger_i is not None else record_start_i
                window = smoothed_hist[lo_f:end_i + 1]
                moving = (sum(1 for v in window if v > gate.stop_threshold) / len(window)
                          if window else 1.0)
                accepted = (signing >= v5.MIN_CLIP_DURATION_SECONDS
                            and moving >= v5.MIN_MOTION_FRACTION)
                if accepted:
                    clips.append((record_start_i, end_i, forced))
                if forced and accepted:
                    # v5 restarts immediately at the cut point rather than dropping to
                    # IDLE -- the signer is still mid-sentence. Modelling this matters:
                    # without it the simulation re-pays the debounce and loses coverage
                    # the real machine keeps.
                    record_start_i = i
                    last_motion_i = i
                    trigger_i = i
                else:
                    state = "IDLE"
                    above_start = 0

    if state == "RECORDING" and last_motion_i is not None:
        signing = rows[last_motion_i]["t"] - rows[record_start_i]["t"]
        if signing >= v5.MIN_CLIP_DURATION_SECONDS:
            clips.append((record_start_i, last_motion_i, False))
    return clips, floors


def phase_spans(rows):
    """Contiguous runs of the same phase label, as (phase, start_i, end_i)."""
    spans = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["phase"] != rows[start]["phase"]:
            spans.append((rows[start]["phase"], start, i - 1))
            start = i
    return spans


def score(rows, clips):
    spans = phase_spans(rows)
    covered = set()
    for a, b, _ in clips:
        covered.update(range(a, b + 1))

    signing_idx = [i for i, r in enumerate(rows) if r["phase"] in SIGNING_PHASES]
    coverage = (len(covered & set(signing_idx)) / len(signing_idx) * 100
                if signing_idx else 0.0)

    # A false start is a clip containing NO signing at all -- i.e. triggered by noise.
    # Do not test the start index alone: back-dating deliberately reaches into the
    # preceding still phase, and a forced-cut restart begins wherever the cut landed, so
    # an index test flags legitimate clips and makes noisy configs look better than they
    # are.
    false_starts = 0
    for a, b, _ in clips:
        if not any(rows[j]["phase"] in SIGNING_PHASES for j in range(a, b + 1)):
            false_starts += 1

    fragments = 0
    for phase, s, e in spans:
        if phase not in SIGNING_PHASES:
            continue
        n = sum(1 for a, b, _ in clips if a <= e and b >= s)
        if n > 1:
            fragments += n - 1

    lead_lost = []
    for phase, s, e in spans:
        if phase not in SIGNING_PHASES:
            continue
        starts = [a for a, b, _ in clips if a <= e and b >= s]
        if starts:
            first = min(starts)
            if first > s:
                lead_lost.append(rows[first]["t"] - rows[s]["t"])
    return {
        "clips": len(clips),
        "forced": sum(1 for _, _, f in clips if f),
        "coverage": coverage,
        "false_starts": false_starts,
        "fragments": fragments,
        "lead_lost": max(lead_lost) if lead_lost else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=os.path.join(HERE, "calib.jsonl"))
    ap.add_argument("--scale", nargs="*", type=float, default=[1.0],
                    help="multiply every score, to stand in for a different room/camera")
    ap.add_argument("--metric", default="current")
    ap.add_argument("--hands", default=os.path.join(HERE, "hand_trigger.jsonl"),
                    help="per-frame hand presence from analyze_hand_trigger.py")
    ap.add_argument("--profile", default=os.path.join(HERE, "motion_profile.json"),
                    help="calibration profile to add a 'calibrated' row for, if present")
    ap.add_argument("--fit-profile", action="store_true",
                    help="fit a profile from THIS recording and simulate with it. Fitting "
                         "and testing on the same data is circular as a quality claim, but "
                         "it does check the fitted pair drives the state machine sanely")
    ap.add_argument("--pregate", type=float, default=1.0,
                    help="divide the pixel start threshold by this for the hybrid pre-gate. "
                         "1.0 (the shipped setting) means hands act as a pure veto and the "
                         "threshold is unchanged; loosening it was measured worse -- it "
                         "adds a false start while the signer settles into position and "
                         "rejects nothing extra")
    args = ap.parse_args()

    rows = load(args.calib)
    print(f"{len(rows)} frames, {rows[-1]['t'] - rows[0]['t']:.1f}s\n")
    print(f"{'scale':>6} {'config':>9} {'clips':>6} {'cover%':>7} {'false':>6} "
          f"{'frag':>5} {'lead lost':>10} {'floor':>7} {'start thr':>10}")
    print("-" * 78)
    hands = load_hand_presence(args.hands)
    if hands is None:
        print("(no hand_trigger.jsonl -- run analyze_hand_trigger.py for the hybrid row)\n")
    configs = [("fixed", False, None, None), ("adaptive", True, None, None)]
    if hands is not None:
        configs.append(("hybrid", True, hands, None))

    profile = None
    if args.fit_profile:
        start, stop, rep = fit_thresholds.fit_rows(rows, metric=args.metric,
                                                   window=v5.MOTION_SMOOTHING_FRAMES)
        if start is not None:
            profile = {"start_threshold": start, "stop_threshold": stop, "floor": None}
            print(f"(fitted from this recording: start {start:.2f} stop {stop:.2f})")
    else:
        profile = load_profile(args.profile)
    if profile is not None:
        # Thresholds are ABSOLUTE here, so the scale sweep is the honest test of what a
        # profile does NOT do: it cannot follow a room it was not measured in, beyond the
        # +/-40% drift clamp. That is the deliberate trade -- measured beats inferred in
        # the room you calibrated, and re-calibration is the answer elsewhere.
        configs.append(("calibrated", True, hands, profile))

    for scale in args.scale:
        for name, adaptive, hnd, prof in configs:
            clips, floors = simulate(rows, adaptive=adaptive, scale=scale,
                                     metric=args.metric, hands=hnd,
                                     pregate_factor=args.pregate if hnd else 1.0,
                                     profile=prof)
            s = score(rows, clips)
            seen = [f for f in floors if f is not None]
            floor = seen[-1] if seen else float("nan")
            if prof is not None:
                gate_start = prof["start_threshold"]
            elif adaptive and seen:
                gate_start = floor * __import__("motion_gate").START_MULTIPLIER
            else:
                gate_start = v5.MOTION_START_THRESHOLD
            print(f"{scale:>6.2f} {name:>9} "
                  f"{s['clips']:>6} {s['coverage']:>7.1f} {s['false_starts']:>6} "
                  f"{s['fragments']:>5} {s['lead_lost']:>9.2f}s "
                  f"{floor:>7.3f} {gate_start:>10.2f}")
        print()


if __name__ == "__main__":
    main()
