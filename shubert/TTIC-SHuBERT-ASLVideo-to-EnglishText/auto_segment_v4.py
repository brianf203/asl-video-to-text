import os
import cv2
import numpy as np
import multiprocessing as mp
import time
from collections import deque

MOTION_START_THRESHOLD = 3.0
MOTION_STOP_THRESHOLD = 0.8
STILL_DURATION_SECONDS = 1.5
MIN_CLIP_DURATION_SECONDS = 1.0

CAMERA_INDEX = 0
TEMP_DIR = "temp"


def worker_process(input_queue, output_queue):
    import os
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
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
        'temp_dir': TEMP_DIR,
    }
    os.makedirs(TEMP_DIR, exist_ok=True)

    print("[worker process] Ready. Waiting for clips...")
    output_queue.put(("READY", None))

    while True:
        clip_path = input_queue.get()
        if clip_path is None:
            break
        try:
            result = SHuBERTProcessor(config).process_video(clip_path)
            output_queue.put(("RESULT", result))
        except Exception as e:
            output_queue.put(("ERROR", str(e)))
        finally:
            try:
                os.remove(clip_path)
            except OSError:
                pass


def main():
    os.makedirs(TEMP_DIR, exist_ok=True)

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    worker = mp.Process(target=worker_process, args=(input_queue, output_queue), daemon=True)
    worker.start()

    print("Starting camera while worker process loads models in the background...")

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
    clip_counter = 0
    pending_count = 0
    latest_translation = "Loading SHuBERT models..."

    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        while not output_queue.empty():
            msg_type, payload = output_queue.get()
            if msg_type == "READY":
                latest_translation = "Ready. Waiting for signing..."
            elif msg_type == "RESULT":
                latest_translation = payload
                pending_count = max(0, pending_count - 1)
                print(f"[result] {payload}")
            elif msg_type == "ERROR":
                pending_count = max(0, pending_count - 1)
                print(f"[error] {payload}")

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_score = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = float(np.mean(diff))
        prev_gray = gray

        display_frame = frame.copy()
        status_line = latest_translation[:70]
        if pending_count > 0:
            status_line += f"  ({pending_count} queued)"

        if state == "IDLE":
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
                    clip_counter += 1
                    clip_path = os.path.join(TEMP_DIR, f"clip_{clip_counter}.mp4")
                    h, w = recorded_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(clip_path, fourcc, 30.0, (w, h))
                    for f in recorded_frames:
                        out.write(f)
                    out.release()
                    input_queue.put(clip_path)
                    pending_count += 1
                    print(f"Queued {clip_path} ({len(recorded_frames)} frames, {elapsed:.1f}s)")
                else:
                    print("Too short, ignored.")

                state = "IDLE"
                recorded_frames = []

        cv2.putText(display_frame, f"motion: {motion_score:.1f}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Auto ASL Translation (persistent worker)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    input_queue.put(None)


if __name__ == "__main__":
    main()
