import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import sys
import time
sys.path.append('onnx_models')
from mp_persondet import MPPersonDet
from mp_pose import MPPose

PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

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


person_detector = MPPersonDetORT("onnx_models/person_detection_mediapipe_2023mar.onnx")
pose_detector = MPPoseORT("onnx_models/pose_estimation_mediapipe_2023mar.onnx")

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Stand with both shoulders clearly visible. Capturing one frame in 3 seconds...")
time.sleep(3)
ret, frame = cap.read()
cap.release()

h, w, _ = frame.shape

persons = person_detector.infer(frame)
if len(persons) > 0:
    onnx_result = pose_detector.infer(frame, persons[0])
    onnx_landmarks = onnx_result[1]
    print("\nONNX pose landmarks shape:", onnx_landmarks.shape)
    print("ONNX left shoulder (index 11):", onnx_landmarks[11][:2])
    print("ONNX right shoulder (index 12):", onnx_landmarks[12][:2])
else:
    print("No person detected by ONNX model")

with mp_holistic.Holistic(static_image_mode=True, model_complexity=1) as holistic:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    if results.pose_landmarks:
        lm11 = results.pose_landmarks.landmark[11]
        lm12 = results.pose_landmarks.landmark[12]
        print("\nMediaPipe left shoulder (index 11):", [lm11.x * w, lm11.y * h])
        print("MediaPipe right shoulder (index 12):", [lm12.x * w, lm12.y * h])
    else:
        print("No pose detected by MediaPipe")
