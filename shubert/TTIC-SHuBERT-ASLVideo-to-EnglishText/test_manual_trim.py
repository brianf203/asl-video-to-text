"""Checks for the push-to-record head/tail trim.

Two halves:
  1. find_signing_onset + the clamps, driven by REAL motion scores from calib.jsonl
     (the `current` field is the same whole-frame-mean metric the live path uses),
     spliced into the shape a manual clip actually has: press -> reach -> settle ->
     sign -> still -> press.
  2. StreamingPerception.finish(keep, start=...) slicing, including the landmark
     re-keying and embedding alignment, on a hand-built instance.
"""
import json
import sys
import threading
import queue

import numpy as np

sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")

FPS = 15.0          # retained frames are stride-2 off a 30fps camera
DT = 1.0 / FPS


def load_phase(name, count=None):
    rows = [json.loads(l) for l in open(
        "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText/calib.jsonl")]
    scores = [r["current"] for r in rows if r["phase"] == name]
    scores = scores[::2]            # mimic stride 2
    return scores if count is None else scores[:count]


def longest_run(scores, threshold, above):
    """The longest contiguous span on one side of the threshold.

    Fixtures are DERIVED rather than sliced by hand: hand-picked ranges quietly contained
    the opposite class (the SIT STILL phase opens with the signer settling into frame at
    2.0-3.0, the SIGN phase opens with 34 frames of prompt-reading below 0.24), which made
    the first version of this test assert against a reach that never moved.
    """
    best = (0, 0)
    start = None
    for i, x in enumerate(scores + [None]):
        hit = x is not None and ((x > threshold) if above else (x <= threshold))
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return scores[best[0]:best[1]]


def make_clip(segments):
    """segments -> (times, raw_scores). Each segment is a list of scores."""
    scores = [s for seg in segments for s in seg]
    times = [1000.0 + i * DT for i in range(len(scores))]
    return times, scores


def run_trim(times, scores, stop_threshold, last_motion_time,
             max_seconds=2.5, min_clip=1.0):
    """The exact arithmetic auto_segment_v5 runs on a manual stop."""
    import auto_segment_v5 as v5
    onset = v5.find_signing_onset(times, scores, stop_threshold, max_seconds)
    lead_cutoff = times[onset] - v5.LEAD_PAD_SECONDS
    while onset > 0 and times[onset - 1] >= lead_cutoff:
        onset -= 1
    tail_cutoff = max(last_motion_time + v5.TAIL_PAD_SECONDS, times[-1] - max_seconds)
    tail = sum(1 for t in times if t <= tail_cutoff)
    span = times[tail - 1] - times[onset] if tail > onset else 0.0
    if span < min_clip:
        return 0, len(times), span, "skipped"
    return onset, tail, span, "trimmed"


def last_motion(times, scores, stop_threshold):
    lm = times[0]
    for t, s in zip(times, scores):
        if s > stop_threshold:
            lm = t
    return lm


def report(name, times, head, keep, note=""):
    total = len(times)
    print(f"  {name:38s} kept {keep - head:3d}/{total:3d} frames  "
          f"head -{head * DT:.2f}s  tail -{(total - keep) * DT:.2f}s  {note}")
    return (head + (total - keep)) * DT


def main():
    import auto_segment_v5 as v5

    still = load_phase("SIT STILL")
    sign = load_phase("SIGN")
    fingerspell = load_phase("FINGERSPELL")

    # The live gate fits stop = floor * STOP_MULTIPLIER off the room's noise floor. Use the
    # recording's own quiet median as the floor so the threshold is the one this data would
    # have produced, not a constant from another room. Median over the WHOLE still phase,
    # not its opening: the gate only feeds sustained quiet runs into the floor, so the
    # settling-into-frame motion at the start of the phase is excluded by construction --
    # taking it in gives a floor 3x too high and a threshold above signing itself.
    import motion_gate
    floor = float(np.median(still))
    stop_threshold = floor * motion_gate.STOP_MULTIPLIER
    print(f"floor {floor:.3f} -> stop_threshold {stop_threshold:.3f} "
          f"(still median {np.median(still):.2f}, signing median {np.median(sign):.2f})\n")

    # Fixtures are taken from spans VERIFIED against the threshold rather than from the
    # front of each phase. The SIGN phase opens with ~34 frames of the signer reading the
    # prompt (0.14-0.24, i.e. quiet), so slicing sign[:8] for "the reach" silently built a
    # motionless reach -- and the first version of this test then blamed the code for
    # correctly finding the real onset 2.3s later.
    quiet = longest_run(still, stop_threshold, above=False)
    moving = longest_run(sign, stop_threshold, above=True)
    body = sign[34:124]                     # the sentence, from where it actually starts
    fs_body = fingerspell[6:96]             # fingerspelling, from its first moving frame
    reach = moving[:8]          # ~0.5s of hand coming off the keyboard: real motion
    settle = quiet[:6]          # ~0.4s pause before the sign starts
    quiet_tail = quiet[:30]     # ~2.0s of the signer sitting still before pressing
    assert all(x > stop_threshold for x in reach), "fixture: reach is not moving"
    assert all(x <= stop_threshold for x in settle + quiet_tail), "fixture: settle moves"

    print("A. clip WITH a settle between the reach and the sign (the trim should fire)")
    saved = 0.0
    for label, body in (("sign", body), ("fingerspell", fs_body)):
        times, scores = make_clip([reach, settle, body, quiet_tail])
        lm = last_motion(times, scores, stop_threshold)
        head, keep, span, how = run_trim(times, scores, stop_threshold, lm)
        saved += report(f"{label} + still tail", times, head, keep, f"span {span:.1f}s [{how}]")
        expect_head = len(reach) + len(settle)
        assert how == "trimmed", f"{label}: trim did not fire"
        assert 0 < head <= expect_head, f"{label}: head {head} outside (0, {expect_head}]"
        assert keep < len(times), f"{label}: tail not trimmed"
        # The sign itself must survive intact, pad included.
        assert head <= expect_head, f"{label}: trimmed into the sign"

    print("\nB. no boundary to find (reach flows straight into the sign, press while moving)")
    times, scores = make_clip([reach, body, moving[:10]])
    lm = last_motion(times, scores, stop_threshold)
    head, keep, span, how = run_trim(times, scores, stop_threshold, lm)
    report("continuous motion", times, head, keep, f"[{how}]")
    assert head == 0, "trimmed a head that has no quiet boundary"
    assert keep >= len(times) - 2, "trimmed a tail that is still moving"

    print("\nC. guards")
    # C1: a clip that is almost entirely dead air must not be trimmed to a stub.
    times, scores = make_clip([quiet[:20], moving[:6], quiet[:20]])
    lm = last_motion(times, scores, stop_threshold)
    head, keep, span, how = run_trim(times, scores, stop_threshold, lm)
    report("mostly dead air", times, head, keep, f"span {span:.1f}s [{how}]")
    assert how == "skipped" and (head, keep) == (0, len(times)), \
        "left a sub-MIN_CLIP_DURATION stub instead of keeping the clip whole"

    # C2: a sentence signed entirely below the stop threshold must keep its tail. This is
    # the failure the max() clamp exists for -- last_motion_time never advances, so the
    # naive cutoff would trim back to the first frame.
    times, scores = make_clip([quiet[:8], [stop_threshold * 0.5] * 90, quiet[:20]])
    lm = last_motion(times, scores, stop_threshold)
    head, keep, span, how = run_trim(times, scores, stop_threshold, lm)
    report("sentence below threshold", times, head, keep, f"[{how}]")
    # Tolerance is one frame interval: the cap bounds the cutoff TIME, and the last frame
    # at or before it is kept, so the dropped span can exceed the cap by up to one frame.
    assert (len(times) - keep) * DT <= 2.5 + DT + 1e-6, \
        "tail trim exceeded MANUAL_TRIM_MAX_SECONDS"

    # C3: neither end may lose more than MANUAL_TRIM_MAX_SECONDS.
    times, scores = make_clip([quiet * 3, body, quiet * 3])
    lm = last_motion(times, scores, stop_threshold)
    head, keep, span, how = run_trim(times, scores, stop_threshold, lm)
    report("long dead air both ends", times, head, keep, f"[{how}]")
    assert head * DT <= v5.MANUAL_TRIM_MAX_SECONDS + DT + 1e-6, "head trim exceeded the cap"
    assert (len(times) - keep) * DT <= v5.MANUAL_TRIM_MAX_SECONDS + DT + 1e-6, \
        "tail trim exceeded the cap"

    # C4: a pause in the MIDDLE of a sentence must not become the onset. Safe by
    # construction now that the search stops at the FIRST boundary, but the regression
    # this guards against (trimming into the sentence) is the one the test caught.
    times, scores = make_clip([reach, settle, body[:40], quiet[:10], body[40:], quiet_tail])
    lm = last_motion(times, scores, stop_threshold)
    head, keep, span, how = run_trim(times, scores, stop_threshold, lm)
    report("pause mid-sentence", times, head, keep, f"[{how}]")
    assert head <= len(reach) + len(settle), "onset jumped past a mid-sentence pause"

    print("\nD. StreamingPerception.finish(keep, start=...) slicing")
    from streaming_perception import StreamingPerception
    s = StreamingPerception.__new__(StreamingPerception)
    n = 12
    s._frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(n)]
    s._landmarks = {i: f"lm{i}" for i in range(n)}
    s._lock = threading.Lock()
    s._emit_lock = threading.Lock()
    s._ready = {}
    s._queues, s._workers = [], []
    s._embed_queue = queue.Queue()
    s._embed_worker = None
    s._embed_error = None
    s._embed_config = {"dino_hands_model_path": "x", "dino_face_model_path": "y"}
    col = lambda tag: [np.array([[i * 10 + tag]], dtype=np.float32) for i in range(n)]
    s._left_chunks = col(1)
    s._right_chunks = col(2)
    s._face_chunks = col(3)

    frames, landmarks, emb = s.finish(keep=10, start=3)
    assert len(frames) == 7, len(frames)
    assert [int(f[0, 0, 0]) for f in frames] == list(range(3, 10))
    assert landmarks == {i: f"lm{i + 3}" for i in range(7)}, landmarks
    for e, tag in zip(emb, (1, 2, 3)):
        assert e.shape[0] == 7, e.shape
        assert [int(v[0]) for v in e] == [i * 10 + tag for i in range(3, 10)], e
    print("  head+tail slice: frames, landmark keys and all three embedding "
          "streams stay aligned")

    # start=0 must be byte-identical to the old single-argument behaviour.
    s2 = StreamingPerception.__new__(StreamingPerception)
    s2.__dict__.update(s.__dict__)
    f0, l0, e0 = s2.finish(keep=10)
    assert [int(f[0, 0, 0]) for f in f0] == list(range(10))
    assert l0 == {i: f"lm{i}" for i in range(10)}
    assert all(e.shape[0] == 10 for e in e0)
    print("  start=0 path unchanged (tail-only trim behaves exactly as before)")

    print(f"\nALL CHECKS PASSED — dead air removed in case A: {saved / 2:.2f}s per clip "
          f"on average")


if __name__ == "__main__":
    main()
