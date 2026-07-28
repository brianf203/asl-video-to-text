import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque
import sys
sys.path.append('onnx_models')
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose
from mp_persondet import MPPersonDet
from mp_pose import MPPose

PROVIDERS = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
CAMERA_INDEX = 0
OUTPUT_PATH = "live_sample.npy"


class MPPalmDetORT(MPPalmDet):
    def __init__(self, modelPath):
        self.model_path = modelPath
        self.nms_threshold = 0.3
        self.score_threshold = 0.5
        self.topK = 5000
        self.input_size = np.array([192, 192])
        self.session = ort.InferenceSession(modelPath, providers=PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = self._load_anchors()

    def infer(self, srcimg):
        h, w, _ = srcimg.shape
        input_blob, pad_bias = self._preprocess(srcimg)
        outputs = self.session.run(None, {self.input_name: input_blob})
        return self._postprocess(outputs, np.array([w, h]), pad_bias)


class MPHandPoseORT(MPHandPose):
    def __init__(self, modelPath):
        self.model_path = modelPath
        self.conf_threshold = 0.8
        self.input_size = np.array([224, 224])
        self.PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2]
        self.PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0
        self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2
        self.PALM_BOX_PRE_SHIFT_VECTOR = [0, 0]
        self.PALM_BOX_PRE_ENLARGE_FACTOR = 4
        self.PALM_BOX_SHIFT_VECTOR = [0, -0.4]
        self.PALM_BOX_ENLARGE_FACTOR = 3
        self.HAND_BOX_SHIFT_VECTOR = [0, -0.1]
        self.HAND_BOX_ENLARGE_FACTOR = 1.65
        self.session = ort.InferenceSession(modelPath, providers=PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image, palm):
        input_blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias = self._preprocess(image, palm)
        outputs = self.session.run(None, {self.input_name: input_blob})
        return self._postprocess(outputs, rotated_palm_bbox, angle, rotation_matrix, pad_bias)


class MPPersonDetORT(MPPersonDet):
    def __init__(self, modelPath):
        self.model_path = modelPath
        self.nms_threshold = 0.3
        self.score_threshold = 0.5
        self.topK = 5000
        self.input_size = np.array([224, 224])
        self.session = ort.InferenceSession(modelPath, providers=PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = self._load_anchors()

    def infer(self, image):
        h, w, _ = image.shape
        input_blob, pad_bias = self._preprocess(image)
        outputs = self.session.run(['Identity:0', 'Identity_1:0'], {self.input_name: input_blob})
        return self._postprocess(outputs, np.array([w, h]), pad_bias)


class MPPoseORT(MPPose):
    def __init__(self, modelPath):
        self.model_path = modelPath
        self.conf_threshold = 0.5
        self.input_size = np.array([256, 256])
        self.PERSON_BOX_PRE_ENLARGE_FACTOR = 1
        self.PERSON_BOX_ENLARGE_FACTOR = 1.25
        self.session = ort.InferenceSession(modelPath, providers=PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image, person):
        h, w, _ = image.shape
        input_blob, rotated_person_bbox, angle, rotation_matrix, pad_bias = self._preprocess(image, person)
        outputs = self.session.run(None, {self.input_name: input_blob})
        return self._postprocess(outputs, rotated_person_bbox, angle, rotation_matrix, pad_bias, np.array([w, h]))


def extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector):
    frame_landmarks = np.zeros((543, 2))
    h, w, _ = frame.shape

    persons = person_detector.infer(frame)
    if len(persons) > 0:
        pose_result = pose_detector.infer(frame, persons[0])
        if pose_result is not None:
            pose_landmarks = pose_result[1]
            for i in range(33):
                frame_landmarks[i][0] = pose_landmarks[i][0] / w
                frame_landmarks[i][1] = pose_landmarks[i][1] / h

    palms = palm_detector.infer(frame)
    for palm in palms:
        hand_result = hand_detector.infer(frame, palm)
        if hand_result is not None:
            handedness = hand_result[130]
            landmarks_2d = np.array([[hand_result[4 + i*3] / w, hand_result[4 + i*3 + 1] / h] for i in range(21)])

            if handedness > 0.5:
                for i in range(21):
                    frame_landmarks[33 + i][0] = landmarks_2d[i][0]
                    frame_landmarks[33 + i][1] = landmarks_2d[i][1]
            else:
                for i in range(21):
                    frame_landmarks[54 + i][0] = landmarks_2d[i][0]
                    frame_landmarks[54 + i][1] = landmarks_2d[i][1]

    return frame_landmarks


def main():
    print("Loading ONNX models (TensorRT compiling on first use, may take a while)...")
    palm_detector = MPPalmDetORT("onnx_models/palm_detection_mediapipe_2023feb.onnx")
    hand_detector = MPHandPoseORT("onnx_models/handpose_estimation_mediapipe_2023feb.onnx")
    person_detector = MPPersonDetORT("onnx_models/person_detection_mediapipe_2023mar.onnx")
    pose_detector = MPPoseORT("onnx_models/pose_estimation_mediapipe_2023mar.onnx")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}")
        return

    ret, frame = cap.read()
    if ret:
        _ = extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector)

    is_recording = False
    recorded_frames = []

    print("Controls:")
    print("  [r] - start/stop recording a sign")
    print("  [q] - quit")
    fps_times = deque(maxlen=30)

    while cap.isOpened():
        ret, frame = cap.read()
        frame_start = time.time()
        if not ret:
            print("Failed to read frame from camera.")
            break

        landmarks = extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector)
        fps_times.append(time.time() - frame_start)
        current_fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

        display_frame = frame.copy()
        if is_recording:
            recorded_frames.append(landmarks)
            cv2.putText(display_frame, f"RECORDING - {len(recorded_frames)} frames",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Press 'r' to start recording",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow('ASL Capture - ONNX/TensorRT', display_frame)

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
