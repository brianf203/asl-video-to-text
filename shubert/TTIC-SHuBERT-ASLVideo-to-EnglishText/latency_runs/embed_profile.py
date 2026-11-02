"""Where does the embed thread's time go, and which stage holds the backlog? Measurement only.

Runs latency_probe.main() in-process (same arguments) with timing wrappers on the embed
thread's steps -- no pipeline file is modified:

  crop        HandExtractor.extract_hand_frames_chunk + FaceExtractor.extract_face_frames_chunk
  preprocess  DINOEmbedder._preprocess_frames_batch  (PIL Resize + ToTensor + Normalize + stack + .to(GPU))
  lock_wait   waiting to acquire gpu_serial (another thread -- ByT5 -- holds the GPU)
  gpu_held    inside gpu_serial: DINOv2 forward + .cpu() sync

Only calls made on the thread named "embed" are counted. A sampler thread also records, every
0.5 s and for every live StreamingPerception, frames added / landmarked / embedded, so the backlog
can be split into "waiting for MediaPipe" (added - landmarked) and "waiting for embed"
(landmarked - embedded).

Usage:
    EMBED_PROFILE=latency_runs/embed_prof/seq shubert_venv/bin/python3 latency_runs/embed_profile.py \
        --condition sequential --pass-index 1 --seed 1001 --out latency_runs/embed_prof/seq_lat.csv
Writes <prefix>_steps.csv (one row per wrapped call) and <prefix>_backlog.csv.
"""
import contextlib
import csv
import os
import sys
import threading
import time
import weakref

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREFIX = os.environ.get("EMBED_PROFILE", "latency_runs/embed_prof/run")
os.makedirs(os.path.dirname(PREFIX), exist_ok=True)
_steps = open(PREFIX + "_steps.csv", "w", newline="")
_sw = csv.writer(_steps)
_sw.writerow(["t", "step", "n_items", "seconds"])
_slock = threading.Lock()


def _log(step, n, sec):
    with _slock:
        _sw.writerow([round(time.time(), 3), step, n, round(sec, 6)])


def _on_embed():
    return threading.current_thread().name == "embed"


def _wrap(cls, name, step, count):
    orig = getattr(cls, name)

    def w(self, *a, **k):
        if not _on_embed():
            return orig(self, *a, **k)
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        _log(step, count(a), time.perf_counter() - t0)
        return r
    setattr(cls, name, w)


def install():
    import crop_hands
    import crop_face
    import dinov2_features
    import streaming_perception

    _wrap(crop_hands.HandExtractor, "extract_hand_frames_chunk", "crop_hands", lambda a: len(a[0]))
    _wrap(crop_face.FaceExtractor, "extract_face_frames_chunk", "crop_face", lambda a: len(a[0]))
    _wrap(dinov2_features.DINOEmbedder, "_preprocess_frames_batch", "preprocess", lambda a: len(a[0]))

    orig_serial = dinov2_features.gpu_serial

    @contextlib.contextmanager
    def timed_serial(*a, **k):
        if not _on_embed():
            with orig_serial(*a, **k):
                yield
            return
        t0 = time.perf_counter()
        with orig_serial(*a, **k):
            t1 = time.perf_counter()
            yield
            t2 = time.perf_counter()
        _log("lock_wait", 0, t1 - t0)
        _log("gpu_held", 0, t2 - t1)
    dinov2_features.gpu_serial = timed_serial

    live = weakref.WeakSet()
    orig_init = streaming_perception.StreamingPerception.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        live.add(self)
    streaming_perception.StreamingPerception.__init__ = init
    return live


def sampler(live, stop):
    with open(PREFIX + "_backlog.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "stream_id", "added", "landmarked", "embedded"])
        while not stop.wait(0.5):
            for sp in list(live):
                try:
                    w.writerow([round(time.time(), 2), id(sp), len(sp._frames),
                                sp._processed, sp._embedded])
                except Exception:
                    pass
            fh.flush()


if __name__ == "__main__":
    live = install()
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(live, stop), daemon=True, name="embed-prof-sampler")
    th.start()
    from latency_runs import latency_probe
    try:
        latency_probe.main()
    finally:
        stop.set()
        th.join(timeout=3)
        _steps.close()
