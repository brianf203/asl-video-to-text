import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
sys.path.append('onnx_models')
from mp_palmdet import MPPalmDet

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


detector = MPPalmDetORT("onnx_models/palm_detection_mediapipe_2023feb.onnx")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Warming up (first inference compiles TensorRT engine, may take a while)...")
ret, frame = cap.read()
_ = detector.infer(frame)
print("Warmup done. Benchmarking for 5 seconds...")

start = time.time()
count = 0
while time.time() - start < 5:
    ret, frame = cap.read()
    if ret:
        results = detector.infer(frame)
        count += 1

print(f"Processed {count} frames in 5 seconds = {count/5:.1f} FPS")
cap.release()
