"""Confirm a start with hand landmarks instead of trusting pixels alone.

WHY
---
The pixel-difference trigger cannot tell a signer from anything else that moves pixels:
auto-exposure hunting, a person walking behind, a screen changing. Its usable window on
the calibration recording is only 1.571 (still max) to 1.889 (signing max) -- a 20% margin
-- so the threshold has to be fitted per room and is fragile even there.

Hand presence is a semantic signal and separates far better. Measured over the 2534
labelled calibration frames (calib.mp4, MediaPipe HandLandmarker, VIDEO mode, conf 0.3):

    phase          hands detected
    SIT STILL       7.7%   (and 0.0% across the 452-frame post-signing still span)
    SIGN           79.0%
    FINGERSPELL    83.9%

End-to-end over sliding windows, against genuine stillness: **3 of 5 frames gives 0.00%
false triggers and 82.2% detection**, in a 0.17s window.

TWO MEASUREMENTS THAT SHAPED THIS
---------------------------------
1. **VIDEO mode is not optional.** IMAGE mode detects hands in only 48% of signing frames
   versus VIDEO's 79%, at the same ~8% still rate and a similar cost (88 vs 102 ms). The
   first version of this used IMAGE mode, reasoning that a sporadic trigger has no
   continuity for tracking to exploit -- wrong: VIDEO mode only needs monotonically
   increasing timestamps, tolerates skipped frames, and its tracking is precisely what
   recovers the frames where one-shot detection drops the hand. With IMAGE mode there is
   NO operating point at <=2% false / >=80% detection; with VIDEO mode there are several.
2. **Hand VELOCITY is useless here, do not retry it.** Landmark displacement only exists
   when hands are found in two consecutive frames, and its margin is 0.19x (still p99
   0.0248 vs signing median 0.0047) -- worse than the pixel metric it was meant to replace.
   Presence is the signal.

COST, AND WHY THIS IS A PRE-GATED HYBRID
----------------------------------------
Detection is ~102 ms/frame. Running it continuously while idle would burn most of a core
and compete with the translation worker. So the cheap pixel metric stays as the PRE-GATE
and this only runs on frames where something already moved.

The pixel threshold is deliberately NOT loosened -- hands act as a pure VETO. Lowering it
was tried and measured worse: at the shipped threshold the hybrid matches pixel-only
exactly on clean data (4 clips, 97.9% coverage, 0 false starts) while rejecting a 20x
background-motion burst that makes pixel-only false-start; lowering the threshold 3x added
a false start (triggering while the signer settles into position with hands visible) and
bought nothing.

The trigger deliberately covers START only. Landmarks look free during recording since
StreamingPerception computes them anyway, but perception runs ~4.5x slower than realtime
and lags capture by seconds, so it cannot inform a real-time stop. A landmark stop would
need a second live detector competing with the streaming one.

Added latency is not a problem here: v5 back-dates every clip to motion onset from a 2.5s
pre-roll ring, so a slower, more careful trigger costs no lead-in.
"""
import os
import threading
import time
from collections import deque

HAND_CONFIDENCE = float(os.environ.get("HAND_TRIGGER_CONF", "0.3"))
# 3 of 5 measured 0.00% false / 82.2% detected against genuine stillness. 1-of-5 detects
# slightly more (84.9%) at the same 0% false on this recording, but requiring 3 leaves
# margin for a room where the still-frame detection rate is not as close to zero.
HAND_WINDOW = int(os.environ.get("HAND_TRIGGER_WINDOW", "5"))
HAND_NEEDED = int(os.environ.get("HAND_TRIGGER_NEEDED", "3"))


class HandPresenceTrigger:
    """Rolling 'are hands present' verdict over the last HAND_WINDOW checked frames.

    Build lazily and close() when done -- it holds a MediaPipe graph with native memory,
    the same resource the pipeline elsewhere is careful to release per clip.
    """

    def __init__(self, hand_model_path, confidence=None, window=None, needed=None):
        self._model_path = hand_model_path
        self._confidence = HAND_CONFIDENCE if confidence is None else confidence
        self._window = HAND_WINDOW if window is None else window
        self._needed = HAND_NEEDED if needed is None else needed
        self._recent = deque(maxlen=self._window)
        self._detector = None
        # Most recent detection, for callers that want more than the rolling verdict --
        # currently the startup calibration, which logs it so a future landmark-based
        # motion score can be fitted from real labelled data instead of guessed at.
        self._latest = None
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._pending = None
        self._thread = None
        self._stop = threading.Event()
        # VIDEO mode requires strictly increasing timestamps. As elsewhere in this
        # pipeline the value only has to be monotonic, not real.
        self._counter = 0

    def _ensure(self):
        if self._detector is not None:
            return
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import RunningMode
        self._detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self._model_path),
                num_hands=2,
                min_hand_detection_confidence=self._confidence,
                running_mode=RunningMode.VIDEO,
            ))

    def start(self):
        """Start the detection thread. Detection NEVER runs on the capture thread.

        One detection is ~102 ms while the capture loop has a 33 ms budget at 30fps, so
        doing this inline would stall the preview and drop frames of the very sentence
        being started -- the same reason stream.finish() is kept off the camera thread.
        The worker takes only the most recent submitted frame and discards any it could
        not keep up with, so the verdict lags by ~100-200 ms and never queues up. That lag
        costs nothing: clips are back-dated to motion onset from the pre-roll ring.
        """
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="hand-trigger",
                                            daemon=True)
            self._thread.start()

    def _run(self):
        import mediapipe as mp
        self._ensure()
        while not self._stop.is_set():
            with self._cv:
                while self._pending is None and not self._stop.is_set():
                    self._cv.wait(timeout=0.2)
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            try:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                self._counter += 1
                result = self._detector.detect_for_video(image, self._counter * 33)
                found = bool(result.hand_landmarks)
                # Landmark 0 is the wrist. Its normalised position is enough to describe
                # where the hands are and how they move; the full 21 points are not
                # needed for a trigger and would be noise to log.
                latest = {
                    "t": time.time(),
                    "n_hands": len(result.hand_landmarks),
                    "wrists": [[round(h[0].x, 4), round(h[0].y, 4)]
                               for h in result.hand_landmarks],
                }
            except Exception as e:
                print(f"[hand-trigger] detection failed: {type(e).__name__}: {e}")
                found = False
                latest = {"t": time.time(), "n_hands": 0, "wrists": []}
            with self._lock:
                self._recent.append(1 if found else 0)
                self._latest = latest

    def submit(self, frame_rgb):
        """Offer the latest frame for checking. Never blocks; drops frames if busy."""
        with self._cv:
            self._pending = frame_rgb
            self._cv.notify()

    def note_absent(self):
        """Record 'no hands' for a frame that was never checked.

        Called when the pixel pre-gate is quiet. Without it the window would remember
        hits from minutes ago and confirm on the next single detection.
        """
        with self._lock:
            self._recent.append(0)

    @property
    def confirmed(self):
        with self._lock:
            return sum(self._recent) >= self._needed

    def latest(self):
        """The most recent detection dict, or None. Lags capture by ~100-200ms and
        arrives at the detector's ~10fps, not the camera's 30 -- fine for logging a
        distribution, not a per-frame signal."""
        with self._lock:
            return self._latest

    @property
    def recent_hits(self):
        with self._lock:
            return sum(self._recent), len(self._recent)

    def reset(self):
        with self._lock:
            self._recent.clear()

    def close(self):
        self._stop.set()
        with self._cv:
            self._cv.notify()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
