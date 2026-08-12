"""Record a short guided session and measure what the motion metrics actually do.

Segmentation thresholds in auto_segment_v5.py (MOTION_START_THRESHOLD=3.0,
MOTION_STOP_THRESHOLD=1.5) were never calibrated against real footage. Replaying v5's
state machine over the recorded eval clips showed signing motion sitting at a median of
0.28-1.22 -- below the stop threshold and far below the start threshold -- with 62-99% of
frames during active signing scoring as "still". That predicts exactly the reported
symptoms: sentences cut mid-fingerspelling, and starts triggered by things that are not
signing.

Those clips are mp4v-compressed though, and compression smooths the sensor noise that
inflates a frame-difference metric, so their absolute scale is biased low against the RAW
frames the live path actually sees. This tool removes that confound by computing the
metrics live, on raw camera frames, in the same order v5 does -- while labelling each
frame with what the signer was actually doing.

The output is the missing half of the picture: the noise floor when nobody is signing.
Without it there is no principled way to choose a start threshold.

    python3 calibrate_motion.py --out calib.jsonl

Phases are prompted on screen. Follow them; the numbers come out at the end.
"""
import argparse
import json
import time

import cv2
import numpy as np

import fit_thresholds
from motion_gate import MotionGate, PROFILE_PATH

# (label, on-screen instruction, seconds). The LABEL is what lands in the data and what
# fit_thresholds.py classifies on -- it is explicit rather than parsed out of the prompt,
# because the old parsing ("SIGN a few".split(" a")[0]) broke on any reworded prompt.
#
# THE "MOVE (NO SIGNING)" PHASE IS THE POINT OF THIS LIST. The machine's real decision is
# not still-vs-signing (4.4x apart, trivial) but signing-vs-fidgeting: shifting in the
# seat and settling into frame reach 1.54 against signing's 1.45-1.89. The old phase list
# never prompted for that, so those frames landed in "SIT STILL", inflated the still
# maximum from 0.36 to 1.54 and pushed the fitted threshold up onto the signing
# distribution. Labelling it is what lets the fit place a threshold between the two -- or
# report honestly that it cannot.
PHASES = [
    ("SIT STILL", "SIT STILL - look at the camera, breathe normally", 12),
    ("MOVE (NO SIGNING)",
     "MOVE but DO NOT SIGN - shift in your seat, scratch your face, adjust your hair", 12),
    ("RIGHT ARM", "RAISE YOUR RIGHT ARM, lower it. Repeat, unhurried", 8),
    ("LEFT ARM", "RAISE YOUR LEFT ARM, lower it. Repeat, unhurried", 8),
    ("SIGN", "SIGN a sentence, pause, then sign another", 18),
    ("FINGERSPELL", "FINGERSPELL a name - slowly, then quickly", 12),
    ("SIT STILL", "SIT STILL again", 10),
]

# Shown before each phase so the signer reads the instruction on a static screen instead
# of during the recorded window. fit_thresholds.SETTLE_SECONDS also trims the head of each
# phase, so this is belt and braces -- but it materially improves what gets captured.
LEAD_IN_SECONDS = 3


def metrics(prev, cur):
    """Every candidate, computed on the same frame pair so they are directly comparable."""
    prev_blur, prev_small = prev
    cur_blur, cur_small = cur

    # 1. What v5 uses today: global mean of the 21x21-blurred greyscale difference.
    m_current = float(np.mean(cv2.absdiff(prev_blur, cur_blur)))

    d = cv2.absdiff(prev_small, cur_small).astype(np.float32)

    # 2. Percent of pixels in localised motion, with the global illumination shift removed.
    #    The median difference IS the whole-frame shift (auto-exposure, gain hunting,
    #    flicker); subtracting it is what separates "the room got brighter" from "a hand
    #    moved". Counting moved pixels rather than averaging magnitude is what keeps a
    #    small fingerspelling hand visible instead of averaging it into nothing.
    shift = float(np.median(d))
    d_local = d - shift
    m_pct = float(100.0 * np.mean(d_local > 8.0))

    # 3. Strongest 8x8 block, same illumination correction. A moving hand lights up one
    #    block hard; noise does not concentrate.
    h, w = d_local.shape
    bh, bw = h // 8, w // 8
    blocks = d_local[:bh * 8, :bw * 8].reshape(8, bh, 8, bw).mean(axis=(1, 3))
    m_block = float(blocks.max())

    # 4. High percentile of the raw difference, no correction -- included as a control, to
    #    show whether the illumination correction in 2 and 3 is doing real work.
    m_p99 = float(np.percentile(d, 99))

    return {"current": m_current, "pct": m_pct, "block": m_block, "p99": m_p99,
            "shift": shift}


def prep(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (cv2.GaussianBlur(gray, (21, 21), 0),
            cv2.GaussianBlur(cv2.resize(gray, (160, 120)), (5, 5), 0))


def write_profile(samples, path, metric="current", smoothing=15, fps=30.0):
    """Fit thresholds from the labelled samples and save them for auto_segment_v5.py.

    The profile also stores the noise floor AS THE SHIPPED ONLINE ESTIMATOR COMPUTES IT,
    not as a summary statistic of the still phases. Those are different quantities -- the
    rolling estimator produces ~0.155 in this room where the global still median is 0.264 --
    and the live gate compares its own floor against this one to decide whether the room
    has drifted. Fitting one estimator and comparing against another silently applies a
    1.7x error to every drift correction.
    """
    start, stop, rep = fit_thresholds.fit_rows(samples, metric=metric,
                                               window=smoothing, fps=fps)
    if start is None:
        print("\nNO PROFILE WRITTEN: " + rep.get("reason", "fit failed"))
        return None

    gate = MotionGate(smoothing, start, stop, fps=fps, adaptive=True)
    for s in samples:
        if fit_thresholds.is_negative(s["phase"]):
            gate.update(s[metric], recording=False, now=s["t"])
    floor = gate.floor

    profile = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric": metric,
        "smoothing_frames": smoothing,
        "fps": fps,
        "start_threshold": start,
        "stop_threshold": stop,
        # May be None if the still phases never produced a sustained quiet run. The gate
        # treats that as "no drift reference" and simply uses the fitted thresholds as-is.
        "floor": floor,
        "separation": rep.get("separation"),
        "separable": rep.get("ok", False),
        "n_frames": len(samples),
        "stats": {
            "neg_max": rep.get("neg_max"),
            "neg_p50": rep.get("neg_p50"),
            "neg_by_phase": rep.get("neg_by_phase"),
            "weakest_move_peak": rep.get("weakest_move_peak"),
            "phase_peaks": rep.get("phase_peaks"),
        },
    }
    with open(path, "w") as fh:
        json.dump(profile, fh, indent=2)

    print("\n" + "=" * 78)
    print("FITTED THRESHOLDS")
    print("=" * 78)
    print(fit_thresholds.describe(start, stop, rep))
    print(f"online floor at calibration: "
          f"{'%.3f' % floor if floor is not None else 'not established'}")
    print(f"wrote {path}")
    if not rep.get("ok"):
        print("\nNOTE: the classes overlap, so this threshold cannot be reliable on its "
              "own.\nKeep the hand-presence veto on (LANDMARK_TRIGGER=1, the default).")
    return profile


def summarise(samples):
    keys = ["current", "pct", "block", "p99"]
    order = []
    for label, _prompt, _secs in PHASES:
        if label not in order:
            order.append(label)

    print("\n" + "=" * 78)
    print("PER-PHASE DISTRIBUTIONS")
    print("=" * 78)
    print(f"{'phase':<22}{'metric':<10}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>9}")
    stats = {}
    for p in order:
        vals = [s for s in samples if s["phase"] == p]
        if not vals:
            continue
        for k in keys:
            a = np.array([v[k] for v in vals])
            stats[(p, k)] = a
            print(f"{p:<22}{k:<10}{np.percentile(a, 50):>9.2f}{np.percentile(a, 90):>9.2f}"
                  f"{np.percentile(a, 95):>9.2f}{np.percentile(a, 99):>9.2f}{a.max():>9.2f}")
        print()

    # The number that decides everything: can a threshold separate "still" from "signing"?
    print("=" * 78)
    print("SEPARATION  (still p99 vs signing p10 -- a metric only works if signing p10 is")
    print("            clearly above still p99, otherwise no threshold can split them)")
    print("=" * 78)
    # "Still" here means every phase that must NOT trigger a clip, which now includes the
    # MOVE (NO SIGNING) phase -- that is the class the threshold actually has to clear.
    still_keys = [p for p in order if fit_thresholds.is_negative(p)]
    sign_keys = [p for p in order if not fit_thresholds.is_negative(p)]
    print(f"{'metric':<10}{'still p99':>12}{'sign p10':>12}{'margin':>12}  verdict")
    for k in keys:
        still = np.concatenate([stats[(p, k)] for p in still_keys if (p, k) in stats]) \
            if any((p, k) in stats for p in still_keys) else np.array([0.0])
        sign = np.concatenate([stats[(p, k)] for p in sign_keys if (p, k) in stats]) \
            if any((p, k) in stats for p in sign_keys) else np.array([0.0])
        s99 = float(np.percentile(still, 99))
        g10 = float(np.percentile(sign, 10))
        margin = g10 - s99
        verdict = "SEPARABLE" if margin > 0 else "overlaps - cannot threshold cleanly"
        print(f"{k:<10}{s99:>12.2f}{g10:>12.2f}{margin:>12.2f}  {verdict}")

    # Fingerspelling is the acute case: report it on its own.
    fs = [p for p in order if p.upper().startswith("FINGERSPELL")]
    if fs and any((fs[0], k) in stats for k in keys):
        print("\nFINGERSPELLING specifically (the case that gets cut off mid-sign):")
        for k in keys:
            if (fs[0], k) not in stats:
                continue
            a = stats[(fs[0], k)]
            still = np.concatenate([stats[(p, k)] for p in still_keys if (p, k) in stats])
            s99 = float(np.percentile(still, 99))
            below = float(np.mean(a <= s99) * 100)
            print(f"  {k:<10} {below:5.1f}% of fingerspelling frames are at or below the "
                  f"still-p99 level ({s99:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib.jsonl")
    ap.add_argument("--video", default="calib.mp4",
                    help="also save the footage, so new metric ideas can be tried later")
    ap.add_argument("--profile", default=PROFILE_PATH,
                    help="where to write the fitted thresholds auto_segment_v5.py loads")
    ap.add_argument("--no-profile", action="store_true",
                    help="measure and print only; leave any existing profile alone")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cap.isOpened():
        raise SystemExit("could not open camera 0")

    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
    samples = []
    prev = None
    fh = open(args.out, "w")

    try:
        for label, prompt, seconds in PHASES:
            phase = label
            # Lead-in: hold the instruction on screen, recording nothing, so the signer
            # reads it before the measured window rather than during it.
            t_lead = time.time() + LEAD_IN_SECONDS
            while time.time() < t_lead:
                ok, frame = cap.read()
                if not ok:
                    break
                prev = prep(frame)
                disp = frame.copy()
                cv2.putText(disp, "NEXT:", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(disp, prompt, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(disp, f"starting in {int(t_lead - time.time()) + 1}", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("motion calibration", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            t_end = time.time() + seconds
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    break
                cur = prep(frame)
                if prev is not None:
                    m = metrics(prev, cur)
                    m["phase"] = phase
                    m["t"] = time.time()
                    samples.append(m)
                    fh.write(json.dumps(m) + "\n")
                prev = cur
                writer.write(frame)

                disp = frame.copy()
                remain = int(t_end - time.time()) + 1
                cv2.putText(disp, prompt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(disp, f"{remain}s left", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if samples:
                    s = samples[-1]
                    cv2.putText(disp, f"cur {s['current']:.2f}  pct {s['pct']:.2f}  "
                                      f"block {s['block']:.2f}", (10, 460),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("motion calibration", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\ninterrupted - summarising what was captured")
    finally:
        fh.close()
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    if samples:
        summarise(samples)
        print(f"\nwrote {args.out} ({len(samples)} frames) and {args.video}")
        if not args.no_profile:
            write_profile(samples, args.profile)


if __name__ == "__main__":
    main()
