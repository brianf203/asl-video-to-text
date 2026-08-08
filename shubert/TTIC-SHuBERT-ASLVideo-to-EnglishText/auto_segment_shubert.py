import cv2
import numpy as np
import time
import os
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
STOP_FRAMES_NEEDED = 20
MIN_CLIP_FRAMES = 15

CAMERA_INDEX = 0
CLIP_PATH = os.path.join(config['temp_dir'], "auto_clip.mp4")


def main():
    print("Loading SHuBERT processor (this loads models on first clip, not now)...")
    processor = SHuBERTProcessor(config)

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
    last_translation = "Waiting for signing..."

    print("Auto-segmentation active. Start signing - detection is automatic.")
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

        display_frame = frame.copy()

        if state == "IDLE":
            cv2.putText(display_frame, "IDLE - " + last_translation[:60],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if motion_score > MOTION_START_THRESHOLD:
                state = "RECORDING"
                recorded_frames = [frame]
                still_count = 0
                print("Motion detected, recording started...")

        elif state == "RECORDING":
            recorded_frames.append(frame)
            if motion_score < MOTION_STOP_THRESHOLD:
                still_count += 1
            else:
                still_count = 0

            cv2.putText(display_frame, f"RECORDING - {len(recorded_frames)} frames",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if still_count >= STOP_FRAMES_NEEDED:
                if len(recorded_frames) >= MIN_CLIP_FRAMES:
                    print(f"Sign ended, {len(recorded_frames)} frames captured. Translating...")
                    cv2.putText(display_frame, "TRANSLATING...",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.imshow('Auto ASL Translation (SHuBERT)', display_frame)
                    cv2.waitKey(1)

                    h, w = recorded_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(CLIP_PATH, fourcc, 30.0, (w, h))
                    for f in recorded_frames:
                        out.write(f)
                    out.release()

                    try:
                        last_translation = processor.process_video(CLIP_PATH)
                        print(f"Translation: {last_translation}")
                    except Exception as e:
                        print(f"Error during translation: {e}")
                        last_translation = "Error during translation"
                else:
                    print("Too short, ignored.")
                    last_translation = "Waiting for signing..."

                state = "IDLE"
                recorded_frames = []

        cv2.imshow('Auto ASL Translation (SHuBERT)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
