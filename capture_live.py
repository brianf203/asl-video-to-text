import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

CAMERA_INDEX = 0
OUTPUT_PATH = "live_sample.npy"

def extract_frame_landmarks(results):
    frame_landmarks = np.zeros((543, 2))

    if results.pose_landmarks:
        for i in range(33):
            lm = results.pose_landmarks.landmark[i]
            frame_landmarks[i][0] = lm.x
            frame_landmarks[i][1] = lm.y

    if results.right_hand_landmarks:
        for i in range(21):
            lm = results.right_hand_landmarks.landmark[i]
            frame_landmarks[33 + i][0] = lm.x
            frame_landmarks[33 + i][1] = lm.y

    if results.left_hand_landmarks:
        for i in range(21):
            lm = results.left_hand_landmarks.landmark[i]
            frame_landmarks[54 + i][0] = lm.x
            frame_landmarks[54 + i][1] = lm.y

    if results.face_landmarks:
        for i in range(468):
            lm = results.face_landmarks.landmark[i]
            frame_landmarks[75 + i][0] = lm.x
            frame_landmarks[75 + i][1] = lm.y

    return frame_landmarks


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}")
        return

    is_recording = False
    recorded_frames = []

    print("Controls:")
    print("  [r] - start/stop recording a sign")
    print("  [q] - quit")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            display_frame = frame.copy()
            mp_drawing.draw_landmarks(display_frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if is_recording:
                frame_landmarks = extract_frame_landmarks(results)
                recorded_frames.append(frame_landmarks)
                cv2.putText(display_frame, f"RECORDING - {len(recorded_frames)} frames",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Press 'r' to start recording",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('ASL Capture - Stage A', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not is_recording:
                    is_recording = True
                    recorded_frames = []
                    print("Recording started...")
                else:
                    is_recording = False
                    feature = np.array(recorded_frames)
                    np.save(OUTPUT_PATH, feature)
                    print(f"Recording stopped. Saved {feature.shape} to {OUTPUT_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
EOF
