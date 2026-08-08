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
        clip_path = clip_queue.get()
        if clip_path is None:
            break
        try:
            print(f"[worker] Translating {clip_path} ...")
            t0 = time.time()
            result = processor.process_video(clip_path)
            elapsed = time.time() - t0
            with state_lock:
                latest_translation[0] = result
            print(f"[worker] ({elapsed:.1f}s) {result}")
        except Exception as e:
            # Report rather than swallow — v2 discarded stderr and made failures invisible.
            print(f"[worker] Error translating {clip_path}: {type(e).__name__}: {e}")
            with state_lock:
                latest_translation[0] = f"[translation failed: {type(e).__name__}]"
        finally:
            clip_queue.task_done()
            try:
                os.remove(clip_path)
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
    record_start_time = None
    last_motion_time = None
    prev_gray = None

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
                recorded_frames = [frame]
                record_start_time = now
                last_motion_time = now
                print(">>> RECORDING STARTED <<<")

        elif state == "RECORDING":
            recorded_frames.append(frame)
            if motion_score > MOTION_STOP_THRESHOLD:
                last_motion_time = now

            still_duration = now - last_motion_time
            elapsed = now - record_start_time

            cv2.putText(display_frame, f"RECORDING - {len(recorded_frames)} frames, {elapsed:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, status_line, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if still_duration >= STILL_DURATION_SECONDS:
                if elapsed >= MIN_CLIP_DURATION_SECONDS:
                    clip_counter[0] += 1
                    clip_path = os.path.join(config['temp_dir'], f"clip_{clip_counter[0]}.mp4")
                    write_clip(recorded_frames, clip_path)
                    clip_queue.put(clip_path)
                    print(f"Queued {clip_path} ({len(recorded_frames)} frames, {elapsed:.1f}s)")
                else:
                    print("Too short, ignored.")

                state = "IDLE"
                recorded_frames = []

        cv2.putText(display_frame, f"motion: {motion_score:.1f}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Auto ASL Translation (v5 - persistent worker)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    clip_queue.put(None)
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
