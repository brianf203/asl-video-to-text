import os
import cv2
import numpy as np
import subprocess
import time
from collections import deque

MOTION_START_THRESHOLD = 3.0
MOTION_STOP_THRESHOLD = 1.5
STILL_DURATION_SECONDS = 1.5
MIN_CLIP_DURATION_SECONDS = 1.0

CAMERA_INDEX = 0
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

clip_counter = 0
pending_clips = deque()
current_process = None
latest_translation = "Waiting for signing..."


def maybe_launch_next():
    global current_process
    if current_process is None and pending_clips:
        clip_path = pending_clips.popleft()
        proc = subprocess.Popen(
            ["python3", "run_shubert.py", clip_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        current_process = (proc, clip_path)
        print(f"[worker] Started translating {clip_path}")


def check_current_process():
    global current_process, latest_translation
    if current_process is None:
        return
    proc, clip_path = current_process
    if proc.poll() is None:
        return

    stdout, _ = proc.communicate()
    output = stdout.decode(errors='ignore')
    found = False
    for line in output.splitlines():
        if line.startswith("The English translation is:"):
            latest_translation = line.replace("The English translation is:", "").strip()
            print(f"[result] {latest_translation}")
            found = True
    if not found:
        print(f"[worker] No translation produced for {clip_path}. Full output:")
        print(output[-1500:])

    try:
        os.remove(clip_path)
    except OSError:
        pass

    current_process = None


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
    record_start_time = None
    last_motion_time = None
    prev_gray = None

    print("Auto-segmentation active (one translation at a time). Sign naturally, pause briefly between sentences.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        check_current_process()
        maybe_launch_next()

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_score = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = float(np.mean(diff))
        prev_gray = gray

        display_frame = frame.copy()
        queued_count = len(pending_clips) + (1 if current_process else 0)
        status_line = latest_translation[:70]
        if queued_count > 0:
            status_line += f"  ({queued_count} translating/queued)"

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
                    pending_clips.append(clip_path)
                    print(f"Queued {clip_path} ({len(recorded_frames)} frames, {elapsed:.1f}s)")
                else:
                    print("Too short, ignored.")

                state = "IDLE"
                recorded_frames = []

        cv2.putText(display_frame, f"motion: {motion_score:.1f}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Auto ASL Translation (v3)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
