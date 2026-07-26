import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
sys.path.append('onnx_models')
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

class MPPalmDetORT(MPPalmDet):
    def __init__(self, modelPath, nmsThreshold=0.3, scoreThreshold=0.5, topK=5000):
        self.model_path = modelPath
        self.nms_threshold = nmsThreshold
        self.score_threshold = scoreThreshold
        self.topK = topK
        self.input_size = np.array([192, 192])
        self.session = ort.InferenceSession(
            modelPath,
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = self._load_anchors()

    def infer(self, srcimg):
        h, w, _ = srcimg.shape
        input_blob, pad_bias = self._preprocess(srcimg)
        outputs = self.session.run(None, {self.input_name: input_blob})
        results = self._postprocess(outputs, np.array([w, h]), pad_bias)
        return results


class MPHandPoseORT(MPHandPose):
    def __init__(self, modelPath, confThreshold=0.8):
        self.model_path = modelPath
        self.conf_threshold = confThreshold
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
        self.session = ort.InferenceSession(
            modelPath,
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image, palm):
        input_blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias = self._preprocess(image, palm)
        outputs = self.session.run(None, {self.input_name: input_blob})
        results = self._postprocess(outputs, rotated_palm_bbox, angle, rotation_matrix, pad_bias)
        return results


palm_detector = MPPalmDetORT("onnx_models/palm_detection_mediapipe_2023feb.onnx")
hand_detector = MPHandPoseORT("onnx_models/handpose_estimation_mediapipe_2023feb.onnx")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Warming up (TensorRT engine compilation, may take a while)...")
ret, frame = cap.read()
palms = palm_detector.infer(frame)
if len(palms) > 0:
    _ = hand_detector.infer(frame, palms[0])
print("Warmup done. Benchmarking for 5 seconds (show your hand to the camera)...")

start = time.time()
count = 0
hands_detected = 0
while time.time() - start < 5:
    ret, frame = cap.read()
    if ret:
        palms = palm_detector.infer(frame)
        if len(palms) > 0:
            hand_result = hand_detector.infer(frame, palms[0])
            hands_detected += 1
        count += 1

print(f"Processed {count} frames in 5 seconds = {count/5:.1f} FPS")
print(f"Hands successfully detected in {hands_detected}/{count} frames")
cap.release()
