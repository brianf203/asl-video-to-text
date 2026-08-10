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
from features import SHuBERTProcessor
from streaming_perception import StreamingPerception, stride_from_env

# Run MediaPipe on each frame as it is captured rather than after the cut. Perception is
# ~4.5x slower than realtime so it never finishes early, but it absorbs the whole recording
# duration -- ~14s off a ~55s clip. Landmarks are identical, not approximated: a fresh
# detector sees the same frames in the same order either way. Set STREAM_PERCEPTION=0 to
# fall back to writing an mp4 and processing it after the cut.
STREAM_PERCEPTION = os.environ.get("STREAM_PERCEPTION", "1") not in ("0", "false", "False")

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

# Time-based segmentation thresholds, carried over from v3.
MOTION_START_THRESHOLD = 3.0
MOTION_STOP_THRESHOLD = 1.5
STILL_DURATION_SECONDS = 1.5
MIN_CLIP_DURATION_SECONDS = 1.0

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
        # Streamed clips arrive as a live StreamingPerception whose backlog still has to
        # be drained; legacy clips arrive as a path to an mp4 needing full processing.
        streamed = isinstance(item, tuple)
        label = item[0] if streamed else item
        stream = None
        try:
            print(f"[worker] Translating {label} ...")
            t0 = time.time()
            if streamed:
                _, stream, keep = item
                # Drain here rather than on the camera thread: perception runs ~4.5x
                # slower than realtime, so at the cut there is still a backlog worth
                # seconds, and blocking the capture loop on it would freeze the preview
                # and drop frames of whatever the signer does next.
                drain_start = time.time()
                frames, landmarks = stream.finish(keep)
                drain = time.time() - drain_start
                print(f"[worker] drained perception backlog in {drain:.1f}s "
                      f"({stream.processed_frames} frames processed, "
                      f"{stream.busy_seconds:.1f}s inside MediaPipe)")
                result = processor.process_frames(
                    frames, landmarks=landmarks,
                    mediapipe_seconds=stream.busy_seconds)
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
            clip_queue.task_done()
            if not streamed:
                try:
                    os.remove(item)
                except OSError:
                    pass


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
    raw_frame_index = 0    # raw frames since RECORDING began, for applying the stride
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

        motion_score = 0.0
        if prev_gray is not None:
            motion_score = float(np.mean(cv2.absdiff(prev_gray, gray)))
        prev_gray = gray

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
            if motion_score > MOTION_START_THRESHOLD:
                state = "RECORDING"
                record_start_time = now
                last_motion_time = now
                raw_frame_index = 0
                recorded_frames = []
                frame_times = []
                if STREAM_PERCEPTION:
                    # One detector per clip — never reused, see streaming_perception.py.
                    stream = StreamingPerception(
                        config['mediapipe_face_model_path'],
                        config['mediapipe_hands_model_path'],
                    )
                    stream.add_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_times.append(now)
                    raw_frame_index = 1
                else:
                    recorded_frames = [frame]
                    frame_times = [now]
                print(">>> RECORDING STARTED <<<")

        elif state == "RECORDING":
            if STREAM_PERCEPTION:
                # Apply the stride here: features.py only strides at video read, which
                # streamed frames never go through. Without this the model would get
                # 30fps instead of the ~15fps SHuBERT expects.
                if raw_frame_index % stride == 0:
                    stream.add_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_times.append(now)
                raw_frame_index += 1
            else:
                recorded_frames.append(frame)
                frame_times.append(now)
            if motion_score > MOTION_STOP_THRESHOLD:
                last_motion_time = now

            still_duration = now - last_motion_time
            elapsed = now - record_start_time

            if STREAM_PERCEPTION:
                rec_label = (f"RECORDING - {len(frame_times)} frames, {elapsed:.1f}s "
                             f"(perception {stream.processed_frames}/{stream.queued_frames})")
            else:
                rec_label = f"RECORDING - {len(recorded_frames)} frames, {elapsed:.1f}s"
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
                    if STREAM_PERCEPTION:
                        label = f"clip_{clip_counter[0]} (streamed)"
                        # Hand the whole stream over; the worker drains and closes it.
                        clip_queue.put((label, stream, keep))
                        print(f"Queued {label} ({keep} frames, {signing_duration:.1f}s "
                              f"signing; trimmed {len(frame_times) - keep} still frames; "
                              f"{stream.processed_frames} already through MediaPipe)")
                        stream = None
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
                        stream = None

                state = "IDLE"
                recorded_frames = []
                frame_times = []

        cv2.putText(display_frame, f"motion: {motion_score:.1f}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Auto ASL Translation (v5 - persistent worker)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Quitting mid-recording leaves a detector holding MediaPipe native resources.
    if stream is not None:
        stream.close()

    clip_queue.put(None)
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
