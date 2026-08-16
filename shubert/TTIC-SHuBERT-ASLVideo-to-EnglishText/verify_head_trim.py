"""End-to-end check of finish(keep, start=...) with the real models.

The unit test proves the slices line up structurally on fake arrays. This proves the
pipeline still produces a sensible SENTENCE when frames are dropped from the head -- a
misalignment between frames, landmarks and the three embedding streams would not raise,
it would just decode to garbage, which is exactly what a demo would put on screen.

Three runs over one clip, one process:
  full      finish(keep=N)                  -- the shipped path, start defaults to 0
  padded    the same frames with `pad` extra still frames spliced onto the FRONT, then
            finish(keep=N+pad, start=pad)   -- the head trim removing exactly that padding
  short     the same padded stream finished WITHOUT the head trim, for contrast

`padded` should reproduce `full` closely: it is the same signing frames, differing only in
the MediaPipe tracking history and crop-fallback state the dropped frames left behind.
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

CLIPS = sys.argv[1:] or ["eval_set/clips/003.mp4"]
PAD_SECONDS = 1.5


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
        assert len(landmarks) == len(got), (len(landmarks), len(got))
        if embeddings is not None:
            for e in embeddings:
                assert e.shape[0] == len(got), (e.shape, len(got))
        text = processor.process_frames(
            got, landmarks=landmarks,
            mediapipe_seconds=stream.busy_seconds,
            embeddings=embeddings,
            embed_seconds=stream.embed_busy_seconds)
    finally:
        stream.close()
    print(f"  {label:8s} {len(frames):3d} in -> {keep - start:3d} kept "
          f"({time.time() - t0:5.1f}s)  {text!r}")
    return text


def main():
    stride = stride_from_env()
    processor = SHuBERTProcessor(v5.config)
    t0 = time.time()
    processor.warmup()
    print(f"models loaded in {time.time() - t0:.1f}s\n")

    summary = []
    for clip in CLIPS:
        frames = read_frames(clip, stride)
        # A still head, built from the clip's own first frame so the padding carries this
        # room's real sensor noise rather than a synthetic constant.
        pad = int(PAD_SECONDS * 30 / stride)
        padding = [frames[0].copy() for _ in range(pad)]
        print(f"{clip}: {len(frames)} frames at stride {stride}, "
              f"padding the head with {pad} ({PAD_SECONDS}s)")
        full = run(processor, frames, keep=len(frames), start=0, label="full")
        padded = run(processor, padding + frames, keep=pad + len(frames), start=pad,
                     label="trimmed")
        untrimmed = run(processor, padding + frames, keep=pad + len(frames), start=0,
                        label="dead air")
        summary.append((clip, full, padded, untrimmed))
        print()

    print("=" * 78)
    for clip, full, padded, untrimmed in summary:
        print(f"{os.path.basename(clip)}  trimmed==full: {str(padded == full):5s}  "
              f"deadair==full: {untrimmed == full}")
        print(f"    full     {full!r}")
        print(f"    trimmed  {padded!r}")
        print(f"    dead air {untrimmed!r}")


if __name__ == "__main__":
    main()
