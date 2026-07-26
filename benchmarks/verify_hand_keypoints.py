import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import sys
import time
sys.path.append('onnx_models')
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

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

palm_detector = MPPalmDetORT("onnx_models/palm_detection_mediapipe_2023feb.onnx")
hand_detector = MPHandPoseORT("onnx_models/handpose_estimation_mediapipe_2023feb.onnx")
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Hold up one hand clearly, fingers spread. Capturing in 3 seconds...")
time.sleep(3)
ret, frame = cap.read()
cap.release()
h, w, _ = frame.shape

palms = palm_detector.infer(frame)
if len(palms) > 0:
    result = hand_detector.infer(frame, palms[0])
    def get_pt(idx):
        return result[4 + idx*3], result[4 + idx*3 + 1]
    print("\nONNX wrist (0):", get_pt(0))
    print("ONNX thumb tip (4):", get_pt(4))
    print("ONNX index tip (8):", get_pt(8))
else:
    print("No hand detected by ONNX")

with mp_holistic.Holistic(static_image_mode=True, model_complexity=1) as holistic:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    hand_lms = results.right_hand_landmarks or results.left_hand_landmarks
    if hand_lms:
        wrist = hand_lms.landmark[0]
        thumb = hand_lms.landmark[4]
        index_tip = hand_lms.landmark[8]
        print("\nMediaPipe wrist (0):", (wrist.x * w, wrist.y * h))
        print("MediaPipe thumb tip (4):", (thumb.x * w, thumb.y * h))
        print("MediaPipe index tip (8):", (index_tip.x * w, index_tip.y * h))
    else:
        print("No hand detected by MediaPipe")
