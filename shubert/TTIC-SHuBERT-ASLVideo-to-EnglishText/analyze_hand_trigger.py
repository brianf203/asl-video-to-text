#!/usr/bin/env python3
"""Measure whether hand landmarks separate stillness from signing better than pixels.

The shipped trigger is a global pixel-difference mean. Its usable window is only
1.571 (still smoothed max) to 1.889 (signing smoothed max) -- a 20% margin -- because
anything that shifts every pixel (auto-exposure, background movement) is indistinguishable
from a signer moving. That fragility is why thresholds needed hand-fitting per room.

A semantic trigger should do better, but "should" is not evidence, so measure it on the
same labelled recording the pixel thresholds were fitted on (calib.mp4 + calib.jsonl,
phases SIT STILL / SIGN / FINGERSPELL) before building anything.

Two candidate signals:
  presence -- are hands detected at all
  velocity -- mean per-frame displacement of hand landmarks, in normalised image units
              (so it is resolution independent, unlike a pixel difference)

Writes hand_trigger.jsonl so thresholds can be fitted without re-running detection.

    python3 analyze_hand_trigger.py [--limit N] [--stride N]
"""
import argparse
import json
import os
import statistics as st
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_BASE = ("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/snapshots/"
               "578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
HAND_MODEL = os.path.join(MODELS_BASE, "hand_landmarker.task")
SIGNING = ("SIGN", "FINGERSPELL")


def quantile(values, p):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(HERE, "calib.mp4"))
    ap.add_argument("--labels", default=os.path.join(HERE, "calib.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "hand_trigger.jsonl"))
    ap.add_argument("--stride", type=int, default=1,
                    help="detect every Nth frame (the live trigger can also be rate limited)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--image-mode", action="store_true",
                    help="IMAGE instead of VIDEO mode (measured far worse: hand detection "
                         "during signing 48%% vs 76%%, because VIDEO mode's tracking is "
                         "what recovers frames where detection alone drops the hand)")
    args = ap.parse_args()

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp

    rows = [json.loads(line) for line in open(args.labels)]

    # VIDEO mode by default. The first pass used IMAGE mode on the reasoning that a
    # sporadic trigger has no continuity for tracking to exploit -- that was wrong, and
    # measurably so: IMAGE detects hands in 48%% of signing frames, VIDEO in 76%%, at the
    # same still-frame rate (~8%%) and a similar cost. VIDEO mode only requires
    # monotonically increasing timestamps; it tolerates skipped frames.
    from mediapipe.tasks.python.vision import RunningMode
    kwargs = dict(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=2,
        min_hand_detection_confidence=args.conf,
    )
    if not args.image_mode:
        kwargs["running_mode"] = RunningMode.VIDEO
    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(**kwargs))

    cap = cv2.VideoCapture(args.video)
    out = []
    prev_pts = None
    i = 0
    t_start = time.time()
    detect_times = []
    while True:
        ok, frame = cap.read()
        if not ok or (args.limit and i >= args.limit):
            break
        if i >= len(rows):
            break
        if i % args.stride:
            i += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t0 = time.time()
        res = (detector.detect(mp_image) if args.image_mode
               else detector.detect_for_video(mp_image, i * 33))
        detect_times.append(time.time() - t0)

        hands = res.hand_landmarks or []
        # Normalised coordinates, so this is independent of resolution and framing --
        # unlike the pixel-difference metric, which scales with both.
        pts = None
        if hands:
            pts = np.array([[lm.x, lm.y] for hand in hands for lm in hand], dtype=np.float32)

        velocity = 0.0
        if pts is not None and prev_pts is not None and len(pts) == len(prev_pts):
            velocity = float(np.mean(np.linalg.norm(pts - prev_pts, axis=1)))
        prev_pts = pts

        out.append({"i": i, "phase": rows[i]["phase"], "n_hands": len(hands),
                    "velocity": velocity, "pixel": rows[i]["current"]})
        i += 1
    cap.release()
    detector.close()

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"{len(out)} frames analysed in {time.time() - t_start:.0f}s "
          f"({st.mean(detect_times) * 1000:.0f} ms/frame detection)")
    print(f"wrote {args.out}\n")

    print(f"{'phase':<12} {'n':>5} {'hands seen':>11} {'vel med':>9} {'vel p90':>9} "
          f"{'vel p99':>9}")
    print("-" * 60)
    for phase in ("SIT STILL", "SIGN", "FINGERSPELL"):
        v = [r for r in out if r["phase"] == phase]
        if not v:
            continue
        vel = [r["velocity"] for r in v]
        seen = sum(1 for r in v if r["n_hands"] > 0) / len(v) * 100
        print(f"{phase:<12} {len(v):>5} {seen:>10.1f}% {st.median(vel):>9.4f} "
              f"{quantile(vel, .90):>9.4f} {quantile(vel, .99):>9.4f}")

    still = [r["velocity"] for r in out if r["phase"] not in SIGNING]
    sign = [r["velocity"] for r in out if r["phase"] in SIGNING]
    print()
    print("--- separation, the number that decides whether this is worth building ---")
    print(f"hand velocity : still p99 {quantile(still, .99):.4f} vs "
          f"signing median {st.median(sign):.4f}  -> margin "
          f"{st.median(sign) / max(quantile(still, .99), 1e-9):.2f}x")
    ps = [r["pixel"] for r in out if r["phase"] not in SIGNING]
    pg = [r["pixel"] for r in out if r["phase"] in SIGNING]
    print(f"pixel metric  : still p99 {quantile(ps, .99):.4f} vs "
          f"signing median {st.median(pg):.4f}  -> margin "
          f"{st.median(pg) / max(quantile(ps, .99), 1e-9):.2f}x")
    print()
    print("A margin above 1.0 means a threshold exists that separates them; the pixel")
    print("metric's is below 1.0 per-frame, which is why it needs 15-frame smoothing.")


if __name__ == "__main__":
    main()
