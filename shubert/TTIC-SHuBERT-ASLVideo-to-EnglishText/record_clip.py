import cv2
import time
import sys

CAMERA_INDEX = 0
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "my_sign.mp4"

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

is_recording = False
print("Controls:")
print("  [r] - start/stop recording")
print("  [q] - quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    if is_recording:
        out.write(frame)
        cv2.putText(display_frame, "RECORDING", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(display_frame, "Press 'r' to start recording", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Record Sign Clip', display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        if not is_recording:
            is_recording = True
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (640, 480))
            print("Recording started...")
        else:
            is_recording = False
            out.release()
            print(f"Recording stopped. Saved to {OUTPUT_PATH}")

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
