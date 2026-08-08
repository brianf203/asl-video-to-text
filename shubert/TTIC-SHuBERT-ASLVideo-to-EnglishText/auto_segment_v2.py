import os
import cv2
import numpy as np
import subprocess
import time

MOTION_START_THRESHOLD = 3.0
MOTION_STOP_THRESHOLD = 1.5
STOP_FRAMES_NEEDED = 45
MIN_CLIP_FRAMES = 40

CAMERA_INDEX = 0
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

clip_counter = 0
running_processes = []
latest_translation = "Waiting for signing..."


def check_finished_processes():
    global latest_translation
    still_running = []
    for proc, clip_path in running_processes:
        if proc.poll() is None:
            still_running.append((proc, clip_path))
        else:
            stdout, _ = proc.communicate()
            output = stdout.decode(errors='ignore')
            for line in output.splitlines():
                if line.startswith("The English translation is:"):
                    latest_translation = line.replace("The English translation is:", "").strip()
                    print(f"[result] {latest_translation}")
            try:
                os.remove(clip_path)
            except OSError:
                pass
    running_processes[:] = still_running


def main():
    global clip_counter

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

    print("Auto-segmentation active (subprocess-based). Sign naturally, pause briefly between sentences.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        check_finished_processes()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_score = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = float(np.mean(diff))
        prev_gray = gray

        display_frame = frame.copy()
        status_line = latest_translation[:70]
        if running_processes:
            status_line += f"  ({len(running_processes)} translating)"

        if state == "IDLE":
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if motion_score > MOTION_START_THRESHOLD:
                state = "RECORDING"
                recorded_frames = [frame]
                still_count = 0
                print(">>> RECORDING STARTED <<<")

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
                    clip_counter += 1
                    clip_path = os.path.join(TEMP_DIR, f"clip_{clip_counter}.mp4")
                    h, w = recorded_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(clip_path, fourcc, 30.0, (w, h))
                    for f in recorded_frames:
                        out.write(f)
                    out.release()

                    proc = subprocess.Popen(
                        ["python3", "run_shubert.py", clip_path],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                    )
                    running_processes.append((proc, clip_path))
                    print(f"Launched translation subprocess for {clip_path} ({len(recorded_frames)} frames)")

                state = "IDLE"
                recorded_frames = []

        cv2.putText(display_frame, f"motion: {motion_score:.1f}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Auto ASL Translation (subprocess)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
