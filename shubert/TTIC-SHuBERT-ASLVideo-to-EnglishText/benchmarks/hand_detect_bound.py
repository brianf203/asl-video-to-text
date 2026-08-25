"""Decompose MediaPipe's hand-detection wall and sweep the two confidence knobs
that have never been tuned.

Hand detection is the per-frame wall of the whole pipeline (~135 ms/frame in VIDEO
mode; pose at 86 ms and face at 51 ms hide fully behind it). Everything cheap has
already been done to it -- num_hands 6->2, IMAGE->VIDEO mode -- and everything
structural has been rejected: downscaling does nothing (MediaPipe resizes
internally), model_complexity=0 costs crop precision, the ONNX/TensorRT port
degraded translation, a GPU build needs bazel from source, and swapping the
extractor is closed by the crop-jitter result.

What was never examined is WHY VIDEO mode only bought 1.41x. Pure frame-to-frame
tracking should be far cheaper than that, which suggests the palm DETECTOR is still
re-firing on most frames. MediaPipe re-runs it whenever it is tracking fewer than
`num_hands` hands, and the measured detection rate is 1.48 hands/frame against
num_hands=2 -- so on a one-handed frame it may be hunting for a second hand that
is not there, every frame.

This script tests that mechanism and sweeps the knobs that control it:

  * min_hand_presence_confidence (0.5 default, NEVER SET in kpe_mediapipe.py) --
    below this the landmark model is judged to have lost the hand and the palm
    detector re-runs.
  * min_tracking_confidence (0.5 default, NEVER SET) -- same idea for tracking.

Lowering either should keep the cheap tracking path alive longer. num_hands=1 is
included as a DIAGNOSTIC CEILING only: it is not shippable (ASL is two-handed),
it is here to prove or kill the mechanism.

    python3 benchmarks/hand_detect_bound.py
    python3 benchmarks/hand_detect_bound.py --clip ../../eval_set/clips/004.mp4 --frames 80

READ THIS BEFORE ACTING ON THE OUTPUT. These are STANDALONE per-detector numbers
with no pose/face contention and no live capture. This project has five recorded
instances of a component measurement evaporating or inverting on the live path, so
a win here is a reason to run `live_worker_probe.py --overlap` as a matched
control-variant-control triple, and then the 200-clip eval -- not a reason to ship.
Any change here moves the landmarks that drive the hand and face CROPS, and the
crop-jitter work established that crop accuracy is exactly what translation
quality is sensitive to. Speed without a quality gate means nothing.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode

MODELS_BASE = ("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/snapshots/"
               "578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
HAND_MODEL = os.path.join(MODELS_BASE, "hand_landmarker.task")


def load_frames(path, n, stride):
    """Match features.py: keep every stride'th frame, convert BGR->RGB."""
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {path}")
    return frames


def run(frames, *, num_hands, detection, presence, tracking, video_mode=True):
    """Time the hand landmarker alone, per frame, recording hands returned."""
    kwargs = dict(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=num_hands,
        min_hand_detection_confidence=detection,
        min_hand_presence_confidence=presence,
        min_tracking_confidence=tracking,
    )
    if video_mode:
        kwargs["running_mode"] = RunningMode.VIDEO
    det = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(**kwargs))

    times, counts = [], []
    try:
        for i, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            t0 = time.perf_counter()
            # Same call shape as HolisticDetector._detect_hand.
            res = det.detect_for_video(mp_image, i * 33) if video_mode else det.detect(mp_image)
            times.append((time.perf_counter() - t0) * 1000.0)
            counts.append(len(res.hand_landmarks) if res.hand_landmarks else 0)
    finally:
        det.close()
    return np.array(times), np.array(counts)


def summarise(label, times, counts, warmup):
    """Drop warmup frames; report cost overall and SPLIT BY HANDS RETURNED.

    The split is the actual diagnostic: if 2-hand frames are much cheaper than
    0/1-hand frames, the palm detector re-firing is confirmed as the cost.
    """
    t, c = times[warmup:], counts[warmup:]
    rate = c.mean()
    row = {"label": label, "mean": t.mean(), "median": np.median(t),
           "p90": np.percentile(t, 90), "rate": rate, "n": len(t)}
    by = {}
    for k in (0, 1, 2):
        sel = t[c == k]
        by[k] = (len(sel), sel.mean() if len(sel) else float("nan"))
    row["by_count"] = by
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(ROOT, "eval_set", "clips", "004.mp4"),
                    help="the fixture the 191->135ms VIDEO-mode result was measured on")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5,
                    help="frames dropped before averaging (first-frame model load)")
    args = ap.parse_args()

    frames = load_frames(args.clip, args.frames, args.stride)
    print(f"clip   : {args.clip}")
    print(f"frames : {len(frames)} at stride {args.stride}, {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"warmup : {args.warmup} frames dropped from every mean\n")

    # (label, num_hands, detection, presence, tracking)
    # Baseline first and repeated LAST as a control -- this box drifts under thermal
    # and background load, and a sweep read without a closing control is not evidence.
    shipping = ("baseline (shipping)", 2, 0.05, 0.5, 0.5)
    configs = [
        shipping,
        ("presence 0.1",            2, 0.05, 0.1, 0.5),
        ("tracking 0.1",            2, 0.05, 0.5, 0.1),
        ("presence+tracking 0.1",   2, 0.05, 0.1, 0.1),
        ("presence+tracking 0.01",  2, 0.05, 0.01, 0.01),
        ("DIAGNOSTIC num_hands=1",  1, 0.05, 0.5, 0.5),
        ("baseline (control)",      2, 0.05, 0.5, 0.5),
    ]

    rows = []
    for label, nh, d, p, tr in configs:
        times, counts = run(frames, num_hands=nh, detection=d, presence=p, tracking=tr)
        rows.append(summarise(label, times, counts, args.warmup))
        print(f"  ran {label}")

    base = rows[0]["mean"]
    print("\n" + "=" * 78)
    print(f"{'config':<26} {'mean ms':>8} {'median':>8} {'p90':>8} {'vs base':>8} {'hands/f':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:<26} {r['mean']:>8.1f} {r['median']:>8.1f} "
              f"{r['p90']:>8.1f} {base / r['mean']:>7.2f}x {r['rate']:>8.2f}")
    print("=" * 78)

    print("\nCost split by hands returned (the palm-detector test):")
    print(f"{'config':<26} {'0 hands':>18} {'1 hand':>18} {'2 hands':>18}")
    print("-" * 82)
    for r in rows:
        cells = []
        for k in (0, 1, 2):
            n, m = r["by_count"][k]
            cells.append(f"{n:>3}f {m:>8.1f}ms" if n else f"{'-':>14}")
        print(f"{r['label']:<26} " + " ".join(f"{c:>18}" for c in cells))
    print("-" * 82)
    print("If 2-hand frames are markedly CHEAPER than 0/1-hand frames, the palm")
    print("detector is re-firing to hunt the missing hand and that is the wall.")
    print("A config is only interesting if it cuts ms/frame WITHOUT cutting hands/f.")


if __name__ == "__main__":
    main()
