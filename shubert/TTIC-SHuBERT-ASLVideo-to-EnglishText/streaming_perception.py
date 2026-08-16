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

# Frame-level parallelism. HolisticDetector already runs its three detectors concurrently
# WITHIN a frame (3 threads), but frames themselves were processed one at a time -- so on a
# 6-core box only about half the CPU was ever in use, while perception is the entire
# remaining latency. arXiv 2607.09611, which runs this same MediaPipe -> DINOv2 -> SHuBERT
# -> ByT5 stack, reports 107-111 -> 45-48 ms/frame from exactly this change (two CPU
# perception workers, 10-frame chunks, reorder buffer).
#
# Frames are handed out in CONTIGUOUS CHUNKS, not round-robin per frame, because MediaPipe
# tracks landmarks temporally: a worker that sees frames 0..9 keeps useful tracking state,
# one that sees every other frame does not. The cost is a re-detection at each chunk
# boundary, which is why chunks should not be tiny.
#
# The default is 30, not the paper's 10. Measured 2026-08-11 across a 200-clip OpenASL eval
# at chunks 10/15/20/30 plus a 1-worker control: chunk 10 was the only setting that leaned
# quality-negative (raw BLEU 18.66 vs 19.15/18.98/19.54, and 1-worker control 19.51), which
# is what more re-detections at more chunk boundaries would predict. None of the deltas
# cleared significance on 200 clips, but the change is free: a live-worker probe sweep
# (4 clips per config, control re-run last) found chunk size latency-NEUTRAL -- the whole
# spread was 1.3 s/clip while two runs of the SAME config differed by 0.6 s/clip. Eval
# s/clip had suggested 1.20-1.67x differences; those were artifact, which is exactly why
# score_streaming's docstring says latency conclusions must come from the probe.
PERCEPTION_WORKERS = max(1, int(os.environ.get("PERCEPTION_WORKERS", "2")))
PERCEPTION_CHUNK = max(1, int(os.environ.get("PERCEPTION_CHUNK", "30")))


class StreamingPerception:
    """Feed frames in as they are captured; collect landmarks when the clip is cut.

    Not reusable across clips, by design -- a HolisticDetector must never be shared
    between clips (MediaPipe leaks the previous clip's tracking state and silently
    changes the translation). Build one per clip and call close().
    """

    def __init__(self, face_model_path: str, hand_model_path: str, embed_config: dict = None,
                 chunk_size: int = None, workers: int = None):
        self._face_model_path = face_model_path
        self._hand_model_path = hand_model_path
        self._n_workers = workers or PERCEPTION_WORKERS
        # One detector per worker, built inside the worker thread so the several-hundred-ms
        # construction cost happens in parallel rather than serially at clip start.
        self._detectors = [None] * self._n_workers
        self._queues = [queue.Queue() for _ in range(self._n_workers)]
        self._frames = []          # RGB, stride-applied, in capture order
        self._landmarks = {}       # index -> landmarks
        self._lock = threading.Lock()
        self._closed = False
        self._error = None
        self._processed = 0
        # Aggregate CPU time across workers, NOT wall time -- with N workers the sum can
        # exceed the elapsed wall clock. Reported as-is so the stage breakdown still shows
        # what perception costs in CPU terms.
        self._busy_seconds = 0.0

        # Reorder buffer. Workers finish out of order, but the crop extractors downstream
        # carry fallback state from frame to frame (prev_left_frame and friends), so the
        # embed stage must see frames in strictly increasing index order or the crops
        # silently change. This is the piece that makes parallel perception safe.
        self._ready = {}           # index -> (frame, landmarks) completed but not yet emitted
        self._next_to_emit = 0
        self._emit_lock = threading.Lock()

        # Optional second stage: crop + DINOv2 as landmarks land, pipelined behind
        # perception. DINOv2 is GPU work and MediaPipe is CPU-bound, so this stage runs
        # essentially for free in the shadow of the perception backlog. Disabled unless
        # embed_config is supplied.
        self._embed_config = embed_config
        self._chunk_size = chunk_size or int(os.environ.get("DINOV2_BATCH_SIZE", "32"))
        self._embed_queue: "queue.Queue" = queue.Queue()
        self._embed_busy_seconds = 0.0
        self._embedded = 0
        self._embed_error = None
        self._left_chunks, self._right_chunks, self._face_chunks = [], [], []
        self._embed_worker = None

        self._workers = [
            threading.Thread(target=self._run, args=(i,), name=f"perception-{i}",
                             daemon=True)
            for i in range(self._n_workers)
        ]
        for w in self._workers:
            w.start()
        if self._embed_config is not None:
            self._embed_worker = threading.Thread(target=self._embed_run, name="embed",
                                                  daemon=True)
            self._embed_worker.start()

    # -- producer side (camera thread) ---------------------------------------

    def add_frame(self, frame_rgb: np.ndarray) -> int:
        """Queue one already-RGB frame. Returns its index. Never blocks on inference."""
        with self._lock:
            index = len(self._frames)
            self._frames.append(frame_rgb)
        # Contiguous chunks per worker, so each detector's temporal tracking stays useful.
        worker = (index // PERCEPTION_CHUNK) % self._n_workers
        self._queues[worker].put((index, frame_rgb))
        return index

    @property
    def queued_frames(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def processed_frames(self) -> int:
        return self._processed

    # -- consumer side (perception thread) -----------------------------------

    def _run(self, worker_id: int):
        detector = HolisticDetector(self._face_model_path, self._hand_model_path)
        self._detectors[worker_id] = detector
        my_queue = self._queues[worker_id]
        while True:
            item = my_queue.get()
            if item is None:
                break
            index, frame = item
            try:
                t0 = time.time()
                _boxes, landmarks = detector.detect_frame_landmarks(frame)
                with self._lock:
                    self._busy_seconds += time.time() - t0
                self._landmarks[index] = landmarks
            except Exception as e:
                # Match process_video_frames(): a bad frame becomes a None entry rather
                # than killing the clip, since downstream already tolerates gaps.
                print(f"[stream] error on frame {index}: {type(e).__name__}: {e}")
                self._landmarks[index] = None
            finally:
                with self._lock:
                    self._processed += 1
                if self._embed_config is not None:
                    self._emit_in_order(index, frame)

    def _emit_in_order(self, index, frame):
        """Release completed frames to the embed stage in index order, never out of it.

        Workers complete out of order; the crop extractors downstream fall back to the
        PREVIOUS frame's crop when a hand or face is missing, so feeding them out of order
        would silently corrupt crops with no error raised.
        """
        with self._emit_lock:
            self._ready[index] = (frame, self._landmarks[index])
            while self._next_to_emit in self._ready:
                f, lm = self._ready.pop(self._next_to_emit)
                self._embed_queue.put((f, lm))
                self._next_to_emit += 1

    def _embed_run(self):
        """Crop and embed frames in chunks as their landmarks become available."""
        from crop_hands import HandExtractor
        from crop_face import FaceExtractor
        from dinov2_features import extract_embeddings_from_frames, HANDS_DTYPE, FACE_DTYPE

        hand_extractor = HandExtractor()
        face_extractor = FaceExtractor()
        pending_frames, pending_landmarks = [], []
        done = False

        def flush():
            if not pending_frames:
                return
            t0 = time.time()
            left, right = hand_extractor.extract_hand_frames_chunk(
                pending_frames, pending_landmarks)
            face = face_extractor.extract_face_frames_chunk(
                pending_frames, pending_landmarks)
            self._left_chunks.append(extract_embeddings_from_frames(
                left, self._embed_config['dino_hands_model_path'], dtype=HANDS_DTYPE))
            self._right_chunks.append(extract_embeddings_from_frames(
                right, self._embed_config['dino_hands_model_path'], dtype=HANDS_DTYPE))
            self._face_chunks.append(extract_embeddings_from_frames(
                face, self._embed_config['dino_face_model_path'], dtype=FACE_DTYPE))
            self._embedded += len(pending_frames)
            self._embed_busy_seconds += time.time() - t0
            pending_frames.clear()
            pending_landmarks.clear()

        try:
            while not done:
                item = self._embed_queue.get()
                if item is None:
                    done = True
                else:
                    pending_frames.append(item[0])
                    pending_landmarks.append(item[1])
                if done or len(pending_frames) >= self._chunk_size:
                    flush()
        except Exception as e:
            # Record and stop; finish() reports it rather than returning partial features.
            print(f"[stream] embedding stage failed: {type(e).__name__}: {e}")
            # Include memory state: this stage is where CUDA OOM surfaces first, and
            # without the numbers it is impossible to tell a genuine backlog problem from
            # the box simply having started short of headroom.
            try:
                with open("/proc/meminfo") as f:
                    info = {k.strip(): v for k, v in
                            (line.split(":", 1) for line in f)}
                print(f"[stream] at failure: MemAvailable={info['MemAvailable'].strip()}"
                      f" frames_queued={len(self._frames)}"
                      f" embedded={self._embedded}")
            except Exception:
                pass
            self._embed_error = e

    # -- finishing -----------------------------------------------------------

    def finish(self, keep: int = None, start: int = 0):
        """Stop accepting frames, drain the backlog, and return (frames, landmarks).

        `keep` truncates to the first N frames (the still-tail trim). Frames past `keep`
        may already have been processed; their results are dropped, which is equivalent
        to never having submitted them.

        `start` drops the first N frames (the still-HEAD trim, used by push-to-record,
        where the clip begins at a key press rather than at motion). Unlike the tail this
        is NOT equivalent to never having submitted them: the dropped frames still fed
        MediaPipe's temporal tracking and the crop extractors' previous-frame fallback, so
        a kept frame whose hand went undetected can inherit a crop from a dropped one.
        That is history the model would have had anyway if the signer had started sooner,
        so it is left in deliberately -- but it does mean a head-trimmed clip is not
        reproducible by re-running perception over the kept frames alone.
        """
        for q in self._queues:
            q.put(None)
        for w in self._workers:
            w.join()
        if self._embed_config is not None:
            # Only now is every frame through perception, so only now can the reorder
            # buffer be empty. Sending the sentinel from a worker (as the single-threaded
            # version did) would end the embed stage while other workers were still
            # producing frames it had not seen.
            with self._emit_lock:
                assert not self._ready, (
                    f"reorder buffer still holds {sorted(self._ready)} after all workers "
                    f"finished; next expected {self._next_to_emit}")
            self._embed_queue.put(None)
        if self._embed_worker is not None:
            self._embed_worker.join()
            if self._embed_error is not None:
                raise self._embed_error

        with self._lock:
            end = len(self._frames) if keep is None else keep
            frames = self._frames[start:end]
        count = len(frames)
        stop = start + count
        # Landmarks are keyed by the index the frame had in the stream; re-key to the
        # returned list's own indices, since process_frames() looks them up positionally.
        landmarks = {i - start: self._landmarks.get(i) for i in range(start, stop)}

        embeddings = None
        if self._embed_config is not None:
            # Chunks are appended in perception order, so concatenating then slicing to
            # [start:stop) matches embedding exactly the kept frames: DINOv2 is a per-image
            # encoder with no cross-frame state, so an embedding depends only on its own
            # crop regardless of which neighbours were trimmed.
            embeddings = (
                np.concatenate(self._left_chunks)[start:stop],
                np.concatenate(self._right_chunks)[start:stop],
                np.concatenate(self._face_chunks)[start:stop],
            )
        return frames, landmarks, embeddings

    def close(self):
        """Release MediaPipe native resources. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        for q, w in zip(self._queues, self._workers):
            if w.is_alive():
                q.put(None)
                w.join(timeout=30)
        for d in self._detectors:
            if d is not None:
                d.close()

    @property
    def busy_seconds(self) -> float:
        """Wall time the perception thread actually spent inside MediaPipe."""
        return self._busy_seconds

    @property
    def embed_busy_seconds(self) -> float:
        """Wall time the embedding thread spent cropping and running DINOv2."""
        return self._embed_busy_seconds

    @property
    def embedded_frames(self) -> int:
        return self._embedded

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
