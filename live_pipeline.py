import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import sys

sys.path.append('ASL-citizen-code/ST-GCN')
from architecture.st_gcn import STGCN
from architecture.fc import FC
from architecture.network import Network

torch.set_default_dtype(torch.float64)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

CAMERA_INDEX = 0

# --- Motion detection tuning ---
MOTION_START_THRESHOLD = 0.015   # how much movement counts as "starting a sign"
MOTION_STOP_THRESHOLD = 0.008    # how little movement counts as "done"
STOP_FRAMES_NEEDED = 8           # consecutive still frames before we call it done
MIN_SIGN_FRAMES = 10             # ignore accidental tiny flickers


def extract_frame_landmarks(results):
    frame_landmarks = np.zeros((543, 2))
    if results.pose_landmarks:
        for i in range(33):
            lm = results.pose_landmarks.landmark[i]
            frame_landmarks[i][0], frame_landmarks[i][1] = lm.x, lm.y
    if results.right_hand_landmarks:
        for i in range(21):
            lm = results.right_hand_landmarks.landmark[i]
            frame_landmarks[33 + i][0], frame_landmarks[33 + i][1] = lm.x, lm.y
    if results.left_hand_landmarks:
        for i in range(21):
            lm = results.left_hand_landmarks.landmark[i]
            frame_landmarks[54 + i][0], frame_landmarks[54 + i][1] = lm.x, lm.y
    if results.face_landmarks:
        for i in range(468):
            lm = results.face_landmarks.landmark[i]
            frame_landmarks[75 + i][0], frame_landmarks[75 + i][1] = lm.x, lm.y
    return frame_landmarks


def hand_motion_score(prev_landmarks, curr_landmarks):
    # only look at hand keypoints (indices 33-74) for motion detection
    if prev_landmarks is None:
        return 0.0
    prev_hands = prev_landmarks[33:75]
    curr_hands = curr_landmarks[33:75]
    diff = np.abs(curr_hands - prev_hands)
    return np.mean(diff)


def preprocess(frames_array, max_frames=128):
    data0 = np.array(frames_array)
    length = data0.shape[0]

    if length > max_frames:
        indices = np.linspace(0, length - 1, max_frames).astype(int)
        data0 = data0[indices]
    elif length < max_frames:
        data0 = np.pad(data0, ((0, max_frames - length), (0, 0), (0, 0)))

    shoulder_l = data0[:, 11, :]
    shoulder_r = data0[:, 12, :]
    center = np.mean((shoulder_l + shoulder_r) / 2, axis=0)
    mean_dist = np.mean(np.sqrt(((shoulder_l - shoulder_r) ** 2).sum(-1)))
    if mean_dist != 0:
        data0 = (data0 - center) * (1.0 / mean_dist)

    data0 = data0[:, 0:75, :]
    posedata = data0[:, 0:33, :]
    rhdata = data0[:, 33:54, :]
    lhdata = data0[:, 54:, :]
    data = np.concatenate([posedata, lhdata, rhdata], axis=1)

    keypoints = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
                 58, 59, 62, 63, 66, 67, 70, 71, 74]
    data = data[:, keypoints, :]
    data = np.transpose(data, (2, 0, 1))

    return torch.from_numpy(data).double().unsqueeze(0)


def load_model(n_classes):
    graph_args = {'num_nodes': 27, 'center': 0,
                  'inward_edges': [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5],
                                   [4, 6], [5, 7], [6, 17], [7, 8], [7, 9],
                                   [9, 10], [7, 11], [11, 12], [7, 13], [13, 14],
                                   [7, 15], [15, 16], [17, 18], [17, 19], [19, 20],
                                   [17, 21], [21, 22], [17, 23], [23, 24], [17, 25], [25, 26]]}
    stgcn = STGCN(in_channels=2, graph_args=graph_args, edge_importance_weighting=True)
    fc = FC(n_features=256, num_class=n_classes, dropout_ratio=0.05)
    model = Network(encoder=stgcn, decoder=fc)
    model.load_state_dict(torch.load('models/ASL_citizen_stgcn_weights.pt', map_location='cuda'))
    model.cuda()
    model.eval()
    return model


def predict(model, idx2gloss, frames_array):
    inputs = preprocess(frames_array).cuda()
    with torch.no_grad():
        predictions = model(inputs)
        probs = torch.softmax(predictions, dim=1)
        top5 = torch.topk(probs, 5, dim=1)
    results = []
    for i in range(5):
        idx = top5.indices[0][i].item()
        conf = top5.values[0][i].item()
        results.append((idx2gloss[idx], conf))
    return results


def main():
    with open('gloss_dict.json', 'r') as f:
        gloss_dict = json.load(f)
    idx2gloss = {v: k for k, v in gloss_dict.items()}
    n_classes = len(gloss_dict)

    print("Loading model...")
    model = load_model(n_classes)
    print("Model loaded. Starting camera...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state = "IDLE"  # IDLE -> RECORDING -> (predict) -> IDLE
    recorded_frames = []
    prev_landmarks = None
    still_frame_count = 0
    last_prediction_text = "Waiting for a sign..."

    print("Show your hands and start signing - detection is automatic. Press 'q' to quit.")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            landmarks = extract_frame_landmarks(results)
            motion = hand_motion_score(prev_landmarks, landmarks)
            prev_landmarks = landmarks

            display_frame = frame.copy()
            mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if state == "IDLE":
                if motion > MOTION_START_THRESHOLD:
                    state = "RECORDING"
                    recorded_frames = [landmarks]
                    still_frame_count = 0
                    print("Sign started...")
                cv2.putText(display_frame, "IDLE - " + last_prediction_text,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif state == "RECORDING":
                recorded_frames.append(landmarks)
                if motion < MOTION_STOP_THRESHOLD:
                    still_frame_count += 1
                else:
                    still_frame_count = 0

                cv2.putText(display_frame, f"RECORDING - {len(recorded_frames)} frames",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if still_frame_count >= STOP_FRAMES_NEEDED:
                    if len(recorded_frames) >= MIN_SIGN_FRAMES:
                        print(f"Sign ended, {len(recorded_frames)} frames captured. Predicting...")
                        top5 = predict(model, idx2gloss, recorded_frames)
                        last_prediction_text = f"{top5[0][0]} ({top5[0][1]*100:.1f}%)"
                        print("Top 5:")
                        for gloss, conf in top5:
                            print(f"  {gloss}: {conf*100:.2f}%")
                    else:
                        print("Too short, ignored.")
                        last_prediction_text = "Waiting for a sign..."
                    state = "IDLE"
                    recorded_frames = []

            cv2.imshow('ASL Live Pipeline', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
