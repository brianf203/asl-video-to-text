import os
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

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

MOTION_START_THRESHOLD = 8.0
MOTION_STOP_THRESHOLD = 3.0
STOP_FRAMES_NEEDED = 45
MIN_CLIP_FRAMES = 40

CAMERA_INDEX = 0

clip_queue = queue.Queue()
latest_translation = ["Waiting for signing..."]
translation_lock = threading.Lock()
clip_counter = [0]


def translation_worker(processor):
    while True:
        clip_path = clip_queue.get()
        if clip_path is None:
            break
        try:
            print(f"[worker] Translating {clip_path} ...")
            result = processor.process_video(clip_path)
            with translation_lock:
                latest_translation[0] = result
            print(f"[worker] Translation: {result}")
        except Exception as e:
            print(f"[worker] Error during translation: {e}")
        finally:
            clip_queue.task_done()
            try:
                os.remove(clip_path)
            except OSError:
                pass


def main():
    print("Loading SHuBERT processor...")
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
    prev_gray = None
    still_count = 0

    print("Auto-segmentation active (threaded). Sign naturally, pause briefly between sentences.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_score = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = float(np.mean(diff))
        prev_gray = gray
        print(f"motion_score={motion_score:.2f} state={state}", end="\r")

        display_frame = frame.copy()

        with translation_lock:
            current_text = latest_translation[0]

        queue_size = clip_queue.qsize()
        status_line = current_text[:70]
        if queue_size > 0:
            status_line += f"  ({queue_size} clip(s) queued)"

        if state == "IDLE":
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if motion_score > MOTION_START_THRESHOLD:
                state = "RECORDING"
                print("\n>>> RECORDING STARTED <<<")
                recorded_frames = [frame]
                still_count = 0

        elif state == "RECORDING":
            recorded_frames.append(frame)
            if motion_score < MOTION_STOP_THRESHOLD:
                still_count += 1
            else:
                still_count = 0

            cv2.putText(display_frame, f"RECORDING - {len(recorded_frames)} frames",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, status_line, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if still_count >= STOP_FRAMES_NEEDED:
                if len(recorded_frames) >= MIN_CLIP_FRAMES:
                    clip_counter[0] += 1
                    clip_path = os.path.join(config['temp_dir'], f"clip_{clip_counter[0]}.mp4")
                    h, w = recorded_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(clip_path, fourcc, 30.0, (w, h))
                    for f in recorded_frames:
                        out.write(f)
                    out.release()
                    clip_queue.put(clip_path)
                    print(f"Queued {clip_path} ({len(recorded_frames)} frames) for translation.")

                state = "IDLE"
                recorded_frames = []

        cv2.imshow('Auto ASL Translation (SHuBERT, threaded)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    clip_queue.put(None)


if __name__ == "__main__":
    main()
