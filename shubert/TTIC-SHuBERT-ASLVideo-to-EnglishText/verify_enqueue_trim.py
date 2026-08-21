"""Would trimming the head at ENQUEUE time change the translation?

The head trim currently runs at the CUT: StreamingPerception.finish() drains every queue,
then slices [start:end], so head frames are fully processed by MediaPipe and DINOv2 before
being discarded. Measured 2026-08-21, that is 4-27 frames per clip of hand detection
(~135ms each) done for nothing -- and since post-cut latency IS the drained backlog, work
removed there is latency removed.

Moving the decision earlier (never submit those frames) is only free if it does not change
the OUTPUT. It might: dropped head frames still feed MediaPipe's temporal tracking and the
crop-fallback state of the frames that are kept, so the sentence's first frames -- the ones
that matter most -- could land differently. finish()'s own docstring flags this.

Two conditions per clip, same padded input, same kept frames:
    submitted  every frame streamed, finish(keep, start=pad)   -- what ships today
    enqueue    only frames[pad:] streamed, finish(keep)        -- the proposal
Identical text on both means the tracking history contributes nothing here and the change
is safe. Different text means the saving is not free and the difference has to be judged.
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")
os.chdir("/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")

import auto_segment_v5 as v5
from features import SHuBERTProcessor
from streaming_perception import StreamingPerception, stride_from_env

# --all runs the whole 28-clip eval set and scores both conditions against the manifest
# references, because 4 clips cannot say which DIRECTION the difference goes.
# EVAL_DIR picks the set, exactly as run_eval.py does, so settling this on the 200-clip
# OpenASL set is `EVAL_DIR=eval_set_openasl python3 verify_enqueue_trim.py --all` and not a
# code change. NOTE this harness exists BECAUSE run_eval.py cannot answer the question: it
# goes through process_video(), the sequential path, and never touches StreamingPerception,
# which is where the head slice lives.
ALL = "--all" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
EVAL_DIR = os.environ.get("EVAL_DIR", "eval_set")
if ALL:
    import json as _json
    CLIPS = [f"{EVAL_DIR}/{_json.loads(l)['file']}"
             for l in open(f"{EVAL_DIR}/manifest.jsonl")]
else:
    CLIPS = ARGS or [f"{EVAL_DIR}/clips/003.mp4", f"{EVAL_DIR}/clips/004.mp4",
                     f"{EVAL_DIR}/clips/005.mp4", f"{EVAL_DIR}/clips/006.mp4"]
OUT = os.environ.get("ENQUEUE_AB_OUT", f"enqueue_trim_ab_{os.path.basename(EVAL_DIR)}.json")
# The live head trims measured on 2026-08-21 were 4-27 kept frames (0.27-1.80s). 1.0s sits
# in that range and is what a signer's pause after the key press actually looks like.
PAD_SECONDS = 1.0


def read_frames(path, stride):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def run(processor, frames, keep, start, label):
    """Stream `frames`, keep [start:keep], translate. Returns (text, wall, perception)."""
    embed_config = v5.config if v5.STREAM_DINOV2 else None
    stream = StreamingPerception(
        v5.config['mediapipe_face_model_path'],
        v5.config['mediapipe_hands_model_path'],
        embed_config=embed_config)
    t0 = time.time()
    try:
        for f in frames:
            stream.add_frame(f)
        got, landmarks, embeddings = stream.finish(keep, start=start)
        assert len(got) == keep - start, (len(got), keep, start)
        perception = stream.busy_seconds
        processed = stream.processed_frames
        text = processor.process_frames(
            got, landmarks=landmarks,
            mediapipe_seconds=stream.busy_seconds,
            embeddings=embeddings,
            embed_seconds=stream.embed_busy_seconds)
    finally:
        stream.close()
    wall = time.time() - t0
    print(f"  {label:10s} {len(frames):3d} submitted -> {processed:3d} through MediaPipe "
          f"-> {keep - start:3d} kept  ({wall:5.1f}s wall, {perception:5.1f}s perception)"
          f"\n             {text!r}")
    return text, wall, perception


def main():
    stride = stride_from_env()
    processor = SHuBERTProcessor(v5.config)
    t0 = time.time()
    processor.warmup()
    print(f"models loaded in {time.time() - t0:.1f}s\n")

    rows = []
    for clip in CLIPS:
        frames = read_frames(clip, stride)
        pad = int(PAD_SECONDS * 30 / stride)
        # A still head built from the clip's own first frame, so the padding carries this
        # room's real sensor noise rather than a synthetic constant.
        padding = [frames[0].copy() for _ in range(pad)]
        print(f"{clip}: {len(frames)} frames at stride {stride}, head padded with {pad}")
        sub_text, sub_wall, sub_perc = run(processor, padding + frames,
                                           keep=pad + len(frames), start=pad,
                                           label="submitted")
        enq_text, enq_wall, enq_perc = run(processor, frames,
                                           keep=len(frames), start=0,
                                           label="enqueue")
        rows.append((clip, sub_text, enq_text, sub_wall, enq_wall, sub_perc, enq_perc))
        print()

    print("=" * 78)
    same = 0
    for clip, a, b, aw, bw, ap, bp in rows:
        match = a == b
        same += match
        print(f"{os.path.basename(clip):10s} identical: {str(match):5s}  "
              f"perception {ap:5.1f}s -> {bp:5.1f}s ({ap - bp:+.1f}s)  "
              f"wall {aw:5.1f}s -> {bw:5.1f}s ({aw - bw:+.1f}s)")
        if not match:
            print(f"    submitted {a!r}")
            print(f"    enqueue   {b!r}")
    print(f"\n{same}/{len(rows)} identical")

    import json
    with open(OUT, "w") as fh:
        json.dump([{"clip": os.path.basename(c), "submitted": a, "enqueue": b,
                    "wall_submitted": aw, "wall_enqueue": bw,
                    "perception_submitted": ap, "perception_enqueue": bp}
                   for c, a, b, aw, bw, ap, bp in rows], fh, indent=2)
    print(f"wrote {OUT}")

    if ALL:
        import sacrebleu
        refs = {}
        for line in open(f"{EVAL_DIR}/manifest.jsonl"):
            r = json.loads(line)
            refs[os.path.basename(r["file"])] = r["reference"]
        ref = [refs[os.path.basename(c)] for c, *_ in rows]
        for name, idx in (("submitted (ships today)", 1), ("enqueue (proposed)", 2)):
            hyp = [r[idx] for r in rows]
            print(f"  {name:24} BLEU {sacrebleu.corpus_bleu(hyp, [ref]).score:5.2f}  "
                  f"chrF {sacrebleu.corpus_chrf(hyp, [ref]).score:5.2f}")
    if rows:
        ap = sum(r[5] for r in rows) / len(rows)
        bp = sum(r[6] for r in rows) / len(rows)
        print(f"mean perception {ap:.1f}s -> {bp:.1f}s ({100 * (ap - bp) / ap:.0f}% less "
              f"work for a {PAD_SECONDS}s head)")


if __name__ == "__main__":
    main()
