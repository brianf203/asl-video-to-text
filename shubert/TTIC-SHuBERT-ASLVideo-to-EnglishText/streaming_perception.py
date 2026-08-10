"""Run MediaPipe landmark extraction *during* capture instead of after the clip is cut.

Perception is the dominant stage of a clip (~30.7s of a 54.9s clip once ByT5 moved to the
GPU) and runs about 4.5x slower than realtime (~0.146 s/frame of MediaPipe against a
0.066s stride-2 frame interval). Because it cannot keep up with the camera it can never
finish early -- but every second spent signing is a second it can be working, so starting
at the first recorded frame rather than at the cut absorbs the whole capture duration.
For a 14s clip that is ~14s off ~55s.

This is a pure scheduling change: the landmarks are identical, not approximated. MediaPipe
holds temporal tracking state across process() calls, so what matters is that a *fresh*
detector sees the *same frames in the same order* -- exactly what video_holistic() does per
clip, and exactly what this does. The result for frame i depends only on frames 0..i, so
discarding a trimmed tail afterwards gives the same landmarks as never having fed it.

Memory note: streaming replaces auto_segment_v5's raw-BGR `recorded_frames` list, so it
also *lowers* peak RAM. Only every FRAME_STRIDE'th frame is retained, already converted to
RGB -- roughly 193MB for a 419-frame clip against 386MB for the full-rate BGR buffer.
"""
import os
import queue
import threading
import time

import numpy as np

from kpe_mediapipe import HolisticDetector


class StreamingPerception:
    """Feed frames in as they are captured; collect landmarks when the clip is cut.

    Not reusable across clips, by design -- a HolisticDetector must never be shared
    between clips (MediaPipe leaks the previous clip's tracking state and silently
    changes the translation). Build one per clip and call close().
    """

    def __init__(self, face_model_path: str, hand_model_path: str):
        self._detector = HolisticDetector(face_model_path, hand_model_path)
        self._queue: "queue.Queue" = queue.Queue()
        self._frames = []          # RGB, stride-applied, in capture order
        self._landmarks = {}       # index -> landmarks
        self._lock = threading.Lock()
        self._closed = False
        self._error = None
        self._processed = 0
        self._busy_seconds = 0.0
        self._worker = threading.Thread(target=self._run, name="perception", daemon=True)
        self._worker.start()

    # -- producer side (camera thread) ---------------------------------------

    def add_frame(self, frame_rgb: np.ndarray) -> int:
        """Queue one already-RGB frame. Returns its index. Never blocks on inference."""
        with self._lock:
            index = len(self._frames)
            self._frames.append(frame_rgb)
        self._queue.put((index, frame_rgb))
        return index

    @property
    def queued_frames(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def processed_frames(self) -> int:
        return self._processed

    # -- consumer side (perception thread) -----------------------------------

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            index, frame = item
            try:
                t0 = time.time()
                _boxes, landmarks = self._detector.detect_frame_landmarks(frame)
                self._busy_seconds += time.time() - t0
                self._landmarks[index] = landmarks
            except Exception as e:
                # Match process_video_frames(): a bad frame becomes a None entry rather
                # than killing the clip, since downstream already tolerates gaps.
                print(f"[stream] error on frame {index}: {type(e).__name__}: {e}")
                self._landmarks[index] = None
            finally:
                self._processed += 1

    # -- finishing -----------------------------------------------------------

    def finish(self, keep: int = None):
        """Stop accepting frames, drain the backlog, and return (frames, landmarks).

        `keep` truncates to the first N frames (the still-tail trim). Frames past `keep`
        may already have been processed; their results are dropped, which is equivalent
        to never having submitted them.
        """
        self._queue.put(None)
        self._worker.join()

        with self._lock:
            frames = self._frames if keep is None else self._frames[:keep]
        count = len(frames)
        landmarks = {i: self._landmarks.get(i) for i in range(count)}
        return frames, landmarks

    def close(self):
        """Release MediaPipe native resources. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=30)
        self._detector.close()

    @property
    def busy_seconds(self) -> float:
        """Wall time the perception thread actually spent inside MediaPipe."""
        return self._busy_seconds

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def stride_from_env() -> int:
    """The same FRAME_STRIDE features.py applies at video read.

    When frames are streamed in, features.py never sees the video, so the caller has to
    apply the stride itself or the model silently gets 30fps instead of the ~15fps
    SHuBERT expects.
    """
    return max(1, int(os.environ.get("FRAME_STRIDE", "2")))
