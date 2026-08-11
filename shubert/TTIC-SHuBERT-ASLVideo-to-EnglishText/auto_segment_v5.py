"""Live ASL auto-segmentation with a persistent in-process translation worker.

Supersedes auto_segment_v3.py (subprocess per clip) and
auto_segment_shubert_threaded.py (in-process worker, frame-count thresholds).

v3 shelled out to `python3 run_shubert.py` for every clip, so each translation
paid the full cold-start cost — the ~13s ByT5 checkpoint load, both DINOv2 loads
and the torch import — on top of the actual work. Here the models are loaded once
at startup (SHuBERTProcessor.warmup) and reused by a background worker thread, so
per-clip latency drops to the compute alone. The camera loop never blocks on
translation; clips are handed off through a queue.

Measured on my_please.mp4: ~57s cold -> ~41s warm per clip.
"""
import os
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
from features import SHuBERTProcessor
from streaming_perception import StreamingPerception, stride_from_env

# Run MediaPipe on each frame as it is captured rather than after the cut. Perception is
# ~4.5x slower than realtime so it never finishes early, but it absorbs the whole recording
# duration -- ~14s off a ~55s clip. Landmarks are identical, not approximated: a fresh
# detector sees the same frames in the same order either way. Set STREAM_PERCEPTION=0 to
# fall back to writing an mp4 and processing it after the cut.
STREAM_PERCEPTION = os.environ.get("STREAM_PERCEPTION", "1") not in ("0", "false", "False")

# Second stage: also crop and run DINOv2 as landmarks land, pipelined behind perception.
# DINOv2 is GPU work while MediaPipe is CPU-bound, so it hides almost entirely in the
# shadow of the perception backlog -- it drops to 0.0s on the critical path. Measured over
# 5 clips against the non-streaming path, decoded text identical on 5/5:
#   003 23.6->11.0s (2.14x), 004 49.4->22.2s (2.22x), 005 45.0->19.8s (2.27x),
#   006 51.1->22.8s (2.24x), 001 39.3->18.7s (2.10x).
# Set STREAM_DINOV2=0 to disable.
STREAM_DINOV2 = os.environ.get("STREAM_DINOV2", "1") not in ("0", "false", "False")

# How many clips may be streaming at once. Each live StreamingPerception holds a MediaPipe
# detector, its retained stride-2 RGB frames (~193MB for a 419-frame clip) and its own
# DINOv2 thread -- and a signer does not wait for the translation before starting the next
# sentence. In the first real signing session six streams piled up and CUDA OOM'd 5 of 9
# clips; every prior validation had fed clips sequentially, so it never saw more than one.
# Past this cap a clip still records, it just buffers frames and does its perception after
# the cut (slower for that clip, but bounded). gpu_serial.py bounds the device side.
MAX_LIVE_STREAMS = int(os.environ.get("MAX_LIVE_STREAMS", "2"))
_live_lock = threading.Lock()
_live_streams = [0]


def _take_stream_slot():
    """Reserve one of the MAX_LIVE_STREAMS slots. False if they are all taken."""
    with _live_lock:
        if _live_streams[0] >= MAX_LIVE_STREAMS:
            return False
        _live_streams[0] += 1
        return True


def _release_stream_slot():
    with _live_lock:
        _live_streams[0] -= 1

MODELS_BASE = "/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models"

config = {
    'yolov8_model_path': os.path.join(MODELS_BASE, 'yolov8n.pt'),
    'dino_face_model_path': os.path.join(MODELS_BASE, 'dinov2face.pth'),
    'dino_hands_model_path': os.path.join(MODELS_BASE, 'dinov2hand.pth'),
    'mediapipe_face_model_path': os.path.join(MODELS_BASE, 'face_landmarker_v2_with_blendshapes.task'),
    'mediapipe_hands_model_path': os.path.join(MODELS_BASE, 'hand_landmarker.task'),
    'shubert_model_path': os.path.join(MODELS_BASE, 'checkpoint_836_400000.pt'),
    'slt_model_config': os.path.join(MODELS_BASE, 'byt5_base', 'config.json'),
    'slt_model_checkpoint': os.path.join(MODELS_BASE, 'checkpoint-11625'),
    'slt_tokenizer_checkpoint': os.path.join(MODELS_BASE, 'byt5_base'),
    'temp_dir': 'temp',
}
os.makedirs(config['temp_dir'], exist_ok=True)

# Segmentation thresholds, CALIBRATED 2026-08-10 against a labelled recording of this
# signer and camera (calibrate_motion.py -> calib.jsonl; 85s of still/sign/fingerspell).
#
# The previous values (start 3.0, stop 1.5, still 1.5s) were inherited from v3 and had
# never been measured. Replaying the state machine over that recording, they recorded
# **11% of the time the user was actually signing**, with a false start and a mid-sentence
# cut. Actual signing scores a median of 0.45 and fingerspelling 0.45, so a stop threshold
# of 1.5 classified ~92% of fingerspelling frames as motionless -- which is exactly the
# reported "it cuts off in the middle of fingerspelling", while whole-frame noise
# (auto-exposure hunting) crossing 3.0 is the reported "it thinks I'm starting a sign".
# These values give 95.8% coverage, 0 false starts and 0 fragments on the same recording.
#
# Two structural changes go with them, both necessary -- thresholds alone are not enough:
#   * the score is SMOOTHED over MOTION_SMOOTHING_FRAMES. No instantaneous metric can
#     separate signing from stillness (measured: every candidate overlapped, because
#     signing contains holds and stillness contains twitches). The separation only exists
#     over time.
#   * a start needs START_DEBOUNCE_FRAMES sustained, and the clip is then back-dated to
#     the real motion onset from the pre-roll ring buffer. Without the back-dating the
#     debounce plus the smoothing lag would eat ~2s off the front of every sentence --
#     trading a truncated tail for a truncated head.
#
# NOTE these are specific to this camera, room and lighting. Re-run calibrate_motion.py
# after any of those change; an adaptive noise floor would be the principled fix.
MOTION_START_THRESHOLD = 1.74
MOTION_STOP_THRESHOLD = 0.29
STILL_DURATION_SECONDS = 2.5
MIN_CLIP_DURATION_SECONDS = 1.0
MOTION_SMOOTHING_FRAMES = 15    # 0.5s trailing mean
START_DEBOUNCE_FRAMES = 3       # sustained frames above start threshold before recording

# Pre-roll: how far back the ring buffer can reach when back-dating a clip to motion onset.
PRE_ROLL_MAX_SECONDS = 2.5
# Mirror of TAIL_PAD_SECONDS at the front, so the first handshape is not clipped.
LEAD_PAD_SECONDS = 0.25

# The STILL_DURATION_SECONDS of stillness that *triggers* the cut also gets recorded,
# so every clip used to carry ~45 dead frames at 30fps. Latency is ~0.41s/frame end to
# end (measured across 105/87/149/181-frame clips), so that tail cost ~18s per clip and
# gave ByT5 nothing to translate — a tail-only clip once produced a fluent hallucination.
# Trim back to the last motion, keeping a short pad so a final handshape hold (which
# carries meaning in ASL) isn't clipped off.
TAIL_PAD_SECONDS = 0.25

CAMERA_INDEX = 0

clip_queue = queue.Queue()
models_ready = threading.Event()
state_lock = threading.Lock()
latest_translation = ["Loading models..."]
clip_counter = [0]


def translation_worker(processor):
    """Load models once, then translate queued clips until the sentinel arrives."""
    try:
        t0 = time.time()
        processor.warmup()
        print(f"[worker] Models loaded in {time.time() - t0:.1f}s")
        with state_lock:
            latest_translation[0] = "Ready. Waiting for signing..."
    except Exception as e:
        print(f"[worker] FATAL: model preload failed: {e}")
        with state_lock:
            latest_translation[0] = f"Model load failed: {e}"
        return
    finally:
        models_ready.set()

    while True:
        item = clip_queue.get()
        if item is None:
            break
        # Three shapes arrive here:
        #   ("stream", label, stream, keep)  live StreamingPerception, backlog to drain
        #   ("frames", label, frames)        buffered frames, perception not started (the
        #                                    over-cap fallback; no mp4 round trip)
        #   "<path>.mp4"                     legacy, only when STREAM_PERCEPTION=0
        kind = item[0] if isinstance(item, tuple) else "path"
        label = item[1] if isinstance(item, tuple) else item
        streamed = kind == "stream"
        stream = None
        try:
            print(f"[worker] Translating {label} ...")
            t0 = time.time()
            if streamed:
                _, _, stream, keep = item
                # Drain here rather than on the camera thread: perception runs ~4.5x
                # slower than realtime, so at the cut there is still a backlog worth
                # seconds, and blocking the capture loop on it would freeze the preview
                # and drop frames of whatever the signer does next.
                drain_start = time.time()
                frames, landmarks, embeddings = stream.finish(keep)
                drain = time.time() - drain_start
                print(f"[worker] drained perception backlog in {drain:.1f}s "
                      f"({stream.processed_frames} frames processed, "
                      f"{stream.busy_seconds:.1f}s inside MediaPipe)")
                result = processor.process_frames(
                    frames, landmarks=landmarks,
                    mediapipe_seconds=stream.busy_seconds,
                    embeddings=embeddings,
                    embed_seconds=stream.embed_busy_seconds)
            elif kind == "frames":
                # Over the stream cap: landmarks were never started, so this pays the full
                # perception cost here. Still cheaper than the legacy path, which wrote and
                # re-read an mp4 and buffered full-rate BGR.
                result = processor.process_frames(item[2])
            else:
                result = processor.process_video(item)
            elapsed = time.time() - t0
            with state_lock:
                latest_translation[0] = result
            print(f"[worker] ({elapsed:.1f}s) {result}")
        except Exception as e:
            # Report rather than swallow — v2 discarded stderr and made failures invisible.
            print(f"[worker] Error translating {label}: {type(e).__name__}: {e}")
            with state_lock:
                latest_translation[0] = f"[translation failed: {type(e).__name__}]"
        finally:
            # MediaPipe native objects must be released per clip or they exhaust the
            # Jetson's shared pool after a few clips.
            if stream is not None:
                stream.close()
                _release_stream_slot()
            clip_queue.task_done()
            if kind == "path":
                try:
                    os.remove(item)
                except OSError:
                    pass


def find_motion_onset(ring):
    """Walk back through the pre-roll ring to where this burst of motion began.

    The trigger fires late by construction (a debounce plus a 0.5s trailing mean), so the
    clip has to be back-dated or the front of every sentence is lost. Onset = the end of
    the last sustained quiet stretch in the buffer; anything before that belongs to the
    previous stillness, not to this sentence. This is the mirror image of the tail trim,
    which ends the clip at the last motion plus a pad.

    `ring` is a deque of (frame, timestamp, raw_score). Returns an index into it.
    """
    quiet_needed = max(1, int(0.3 * 30))    # 0.3s of quiet marks a real boundary
    quiet = 0
    for i in range(len(ring) - 1, -1, -1):
        if ring[i][2] <= MOTION_STOP_THRESHOLD:
            quiet += 1
            if quiet >= quiet_needed:
                return min(i + quiet_needed, len(ring) - 1)
        else:
            quiet = 0
    return 0    # the whole buffer is motion; keep all of it


def write_clip(frames, path, fps=30.0):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def main():
    print("Starting persistent translation worker...")
    processor = SHuBERTProcessor(config)
    worker = threading.Thread(target=translation_worker, args=(processor,), daemon=True)
    worker.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}")
        return

    state = "IDLE"
    recorded_frames = []
    frame_times = []
    record_start_time = None
    last_motion_time = None
    prev_gray = None
    stream = None          # StreamingPerception for the clip being recorded
    deferred = False       # recording without a stream, because the cap was reached
    raw_frame_index = 0    # raw frames since RECORDING began, for applying the stride
    motion_history = deque(maxlen=MOTION_SMOOTHING_FRAMES)
    pre_roll = deque(maxlen=int(PRE_ROLL_MAX_SECONDS * 30))
    above_start = 0        # consecutive frames over the start threshold (the debounce)
    stride = stride_from_env()
    if STREAM_PERCEPTION:
        print(f"Streaming perception ON (stride {stride}) — landmarks are extracted "
              f"during recording, not after the cut.")

    print("Auto-segmentation active. Sign naturally, pause briefly between sentences.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        raw_motion = 0.0
        if prev_gray is not None:
            raw_motion = float(np.mean(cv2.absdiff(prev_gray, gray)))
        prev_gray = gray

        # Decisions run on the smoothed score; the raw one is kept for locating motion
        # onset in the pre-roll, where the smoothing lag would blur the boundary.
        motion_history.append(raw_motion)
        motion_score = sum(motion_history) / len(motion_history)
        pre_roll.append((frame, now, raw_motion))

        display_frame = frame.copy()
        with state_lock:
            status_line = latest_translation[0][:70]
        queued_count = clip_queue.qsize()
        if queued_count > 0:
            status_line += f"  ({queued_count} translating/queued)"

        # Don't start recording until the models are up, otherwise clips pile up
        # in the queue while the 2.68GB checkpoint is still loading.
        if not models_ready.is_set():
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        elif state == "IDLE":
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            above_start = above_start + 1 if motion_score > MOTION_START_THRESHOLD else 0
            if above_start >= START_DEBOUNCE_FRAMES:
                above_start = 0
                state = "RECORDING"
                # Back-date to motion onset. The trigger is late by the debounce plus the
                # smoothing window, measured at ~2s on the calibration recording -- all of
                # which is signing, so it must come from the ring buffer rather than being
                # dropped.
                onset = find_motion_onset(pre_roll)
                # Extend back by the lead pad, the mirror of TAIL_PAD_SECONDS: cutting
                # exactly at onset would clip the start of the first handshape.
                onset = max(0, onset - int(LEAD_PAD_SECONDS * 30))
                seed = list(pre_roll)[onset:]

                record_start_time = seed[0][1] if seed else now
                last_motion_time = now
                raw_frame_index = 0
                recorded_frames = []
                frame_times = []
                # Stream only if a slot is free. Over the cap the clip is still recorded,
                # just buffered at stride and translated after the cut — degraded latency
                # for that one clip instead of an OOM that loses it entirely.
                if STREAM_PERCEPTION and _take_stream_slot():
                    # One detector per clip — never reused, see streaming_perception.py.
                    stream = StreamingPerception(
                        config['mediapipe_face_model_path'],
                        config['mediapipe_hands_model_path'],
                        embed_config=config if STREAM_DINOV2 else None,
                    )
                elif STREAM_PERCEPTION:
                    # Buffer stride-2 RGB, the same frames the stream would have retained,
                    # so this fallback costs no more memory than streaming would have.
                    deferred = True
                else:
                    deferred = False

                if not STREAM_PERCEPTION:
                    # Legacy path keeps every frame and strides later, at video read.
                    recorded_frames = [f for f, _, _ in seed]
                    frame_times = [t for _, t, _ in seed]
                else:
                    # Replay the pre-roll through the same stride the live branch uses, so
                    # seeded and live frames form one evenly-sampled sequence.
                    for f, t, _ in seed:
                        if raw_frame_index % stride == 0:
                            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            if stream is not None:
                                stream.add_frame(rgb)
                            else:
                                recorded_frames.append(rgb)
                            frame_times.append(t)
                        raw_frame_index += 1

                extra = "" if stream is not None else (
                    f" (deferred — {MAX_LIVE_STREAMS} streams already live)"
                    if deferred else "")
                print(f">>> RECORDING STARTED{extra} — backdated {len(seed)} frames "
                      f"({(now - record_start_time):.2f}s) to motion onset <<<")

        elif state == "RECORDING":
            # The stride is applied here for both streamed and deferred clips: features.py
            # only strides at video read, which neither of them goes through. Without this
            # the model would get 30fps instead of the ~15fps SHuBERT expects.
            if stream is not None:
                if raw_frame_index % stride == 0:
                    stream.add_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_times.append(now)
                raw_frame_index += 1
            elif deferred:
                if raw_frame_index % stride == 0:
                    recorded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_times.append(now)
                raw_frame_index += 1
            else:
                recorded_frames.append(frame)
                frame_times.append(now)
            if motion_score > MOTION_STOP_THRESHOLD:
                last_motion_time = now

            still_duration = now - last_motion_time
            elapsed = now - record_start_time

            if stream is not None:
                rec_label = (f"RECORDING - {len(frame_times)} frames, {elapsed:.1f}s "
                             f"(perception {stream.processed_frames}/{stream.queued_frames})")
            else:
                rec_label = (f"RECORDING{' [deferred]' if deferred else ''} - "
                             f"{len(recorded_frames)} frames, {elapsed:.1f}s")
            cv2.putText(display_frame, rec_label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, status_line, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if still_duration >= STILL_DURATION_SECONDS:
                # Drop the trailing stillness; keep everything up to the last motion
                # plus TAIL_PAD_SECONDS. Gate on signing time rather than wall-clock
                # elapsed, so a clip that is nothing but the still tail scores ~0s and
                # is rejected instead of being handed to ByT5 to hallucinate over.
                cutoff = last_motion_time + TAIL_PAD_SECONDS
                keep = sum(1 for t in frame_times if t <= cutoff)
                signing_duration = last_motion_time - record_start_time

                if signing_duration >= MIN_CLIP_DURATION_SECONDS:
                    clip_counter[0] += 1
                    if stream is not None:
                        label = f"clip_{clip_counter[0]} (streamed)"
                        # Hand the whole stream over; the worker drains, closes it and
                        # releases the slot.
                        clip_queue.put(("stream", label, stream, keep))
                        print(f"Queued {label} ({keep} frames, {signing_duration:.1f}s "
                              f"signing; trimmed {len(frame_times) - keep} still frames; "
                              f"{stream.processed_frames} already through MediaPipe)")
                        stream = None
                    elif deferred:
                        label = f"clip_{clip_counter[0]} (deferred)"
                        clip_queue.put(("frames", label, recorded_frames[:keep]))
                        print(f"Queued {label} ({keep} frames, {signing_duration:.1f}s "
                              f"signing; trimmed {len(recorded_frames) - keep} still "
                              f"frames; perception not started)")
                    else:
                        clip_path = os.path.join(config['temp_dir'],
                                                 f"clip_{clip_counter[0]}.mp4")
                        write_clip(recorded_frames[:keep], clip_path)
                        clip_queue.put(clip_path)
                        print(f"Queued {clip_path} ({keep} frames, {signing_duration:.1f}s "
                              f"signing; trimmed {len(recorded_frames) - keep} still frames)")
                else:
                    print(f"Too short, ignored ({signing_duration:.1f}s signing, "
                          f"{len(frame_times)} frames).")
                    if stream is not None:
                        stream.close()
                        _release_stream_slot()
                        stream = None

                state = "IDLE"
                deferred = False
                recorded_frames = []
                frame_times = []

        # Two decimals: the thresholds are now 1.74 / 0.29, so one decimal cannot show
        # whether the score is near either of them.
        cv2.putText(display_frame,
                    f"motion: {motion_score:.2f} (raw {raw_motion:.2f})  "
                    f"start {MOTION_START_THRESHOLD} stop {MOTION_STOP_THRESHOLD}",
                    (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Auto ASL Translation (v5 - persistent worker)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Quitting mid-recording leaves a detector holding MediaPipe native resources.
    if stream is not None:
        stream.close()
        _release_stream_slot()

    clip_queue.put(None)
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
