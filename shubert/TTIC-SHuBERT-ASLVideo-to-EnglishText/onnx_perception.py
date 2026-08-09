"""ONNX/TensorRT hand and pose detection, drop-in for MediaPipe's in HolisticDetector.

STATUS: opt-in, OFF by default. Set USE_ONNX_PERCEPTION=1 to enable.

It is genuinely faster -- ~1.5x on the perception stage -- but that is only ~13% end to
end (perception is a minority of clip time; ByT5 dominates), and a 5-clip A/B against
MediaPipe showed a real accuracy cost. On dailymoth_examples/rDUefZVPfmU_crop_1, the only
clip in the set with verifiable ground truth (the real Wilmington LA tunnel collapse,
31 workers), this backend lost the count ("31 workers" -> "Wild workers"), inverted the
event ("collapsed" -> "taken down") and relocated Wilmington to "Louisiana, California".
MediaPipe got all three right. The other four clips were: one identical, one draw, and
two minor splits.

The cause is landmark ACCURACY, not detection rate -- worth knowing before anyone retries
the obvious fix. Hand detection on that clip was already 98% single-hand / 27% two-hand
(matching MediaPipe) at the original conf_threshold of 0.8, and lowering it to 0.3
reproduced all three errors unchanged. The ONNX 21-point positions are simply less
precise than MediaPipe's, which propagates into the DINOv2 hand crops.

Worth revisiting if perception ever becomes the dominant cost, or for high-resolution
input, where the advantage grows: the ONNX models resize to fixed inputs (192/224/256)
so their cost is resolution-independent, while MediaPipe's scales with frame size
(1.42-1.61x on the larger dailymoth clips vs 1.41x on 640x480 camera footage).

MediaPipe's hand (139.9 ms/frame) and pose (126.5 ms/frame) detectors set the perception
critical path. The legacy ASL-Citizen pipeline already carries ONNX exports of the same
MediaPipe graphs, which run ~10x faster here under TensorRT. Benchmarked on this Jetson
over my_please.mp4:

    hand path (palm + handpose) : ~28 ms/frame   (MediaPipe 139.9)
    pose path (person + pose)   :  51.5 ms/frame (MediaPipe 126.5)

The three detectors run concurrently, so the wall is max(pose, face, hand) -- both hand
and pose had to improve for it to move. Face stays on MediaPipe: at 41.8 ms it is already
under the new 51.5 ms wall, so converting it would buy nothing.

Output format matches MediaPipe exactly -- normalized [x, y, z] with 21 landmarks per
hand and 33 for pose in standard MediaPipe Pose order -- because `crop_hands.py` and
`body_features.py` consume that shape. Only x and y matter downstream
(`PoseProcessor.coords_per_keypoint == 2`, and crop_hands scales l[0]/l[1] by frame
size), so the differing z semantics between the two stacks are not a problem.
"""
import os
import sys
import threading

import cv2
import numpy as np
import onnxruntime as ort

_DEFAULT_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "legacy_asl_citizen_pipeline", "onnx_models")
)
MODELS_DIR = os.environ.get("ONNX_MODELS_DIR", _DEFAULT_MODELS_DIR)

# TensorRT compiles an engine per model on first use -- 266s total when uncached, which
# would otherwise be paid on every process start and dwarf the worker's ~20s warmup.
CACHE_DIR = os.environ.get(
    "TRT_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trt_cache"),
)
os.makedirs(CACHE_DIR, exist_ok=True)

PROVIDERS = [
    ("TensorrtExecutionProvider", {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": CACHE_DIR,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": CACHE_DIR,
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

# 0.85 is the legacy pipeline's threshold, tuned for a different task; it finds a palm on
# only 33% of frames. Sweeping on my_please.mp4: 0.70->75%, 0.50->84%, 0.30->94%,
# 0.10->100%, at flat cost (11.3-13.0 ms). MediaPipe's own rate is 78% of frames with
# >=1 hand and 27% with both. 0.30 beats it on the former and matches it on the latter;
# 0.10 looks better still but precision was never measured and a false palm hands DINOv2
# a garbage crop, so do not go lower without checking that.
PALM_SCORE_THRESHOLD = float(os.environ.get("ONNX_PALM_SCORE_THRESHOLD", "0.30"))

# Second filter, applied by the handpose model after the palm detector accepts a box --
# why end-to-end hand detection is lower than palm detection alone. Swept on my_please:
# 0.8 -> 71%, 0.5 -> 75%, 0.3 -> 78%, 0.1 -> 84% single-hand, at flat cost. 0.3 matches
# MediaPipe's 78%, so it is the default. Note this did NOT fix the accuracy regression
# described above -- that clip was already at 98% detection; see the module docstring.
HANDPOSE_CONF_THRESHOLD = float(os.environ.get("ONNX_HANDPOSE_CONF_THRESHOLD", "0.30"))

sys.path.insert(0, MODELS_DIR)
from mp_palmdet import MPPalmDet      # noqa: E402
from mp_handpose import MPHandPose    # noqa: E402
from mp_persondet import MPPersonDet  # noqa: E402
from mp_pose import MPPose            # noqa: E402


class _PalmDet(MPPalmDet):
    def __init__(self, model_path, score_threshold):
        self.model_path = model_path
        self.nms_threshold = 0.3
        self.score_threshold = score_threshold
        self.topK = 5000
        self.input_size = np.array([192, 192])
        self.session = ort.InferenceSession(model_path, providers=PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = self._load_anchors()

    def infer(self, srcimg):
        h, w, _ = srcimg.shape
        blob, pad_bias = self._preprocess(srcimg)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, np.array([w, h]), pad_bias)


class _HandPose(MPHandPose):
    def __init__(self, model_path, conf_threshold=HANDPOSE_CONF_THRESHOLD):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
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
        self.session = ort.InferenceSession(model_path, providers=PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image, palm):
        blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias = self._preprocess(image, palm)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, rotated_palm_bbox, angle, rotation_matrix, pad_bias)


class _PersonDet(MPPersonDet):
    def __init__(self, model_path):
        self.model_path = model_path
        self.nms_threshold = 0.3
        self.score_threshold = 0.5
        self.topK = 5000
        self.input_size = np.array([224, 224])
        self.session = ort.InferenceSession(model_path, providers=PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = self._load_anchors()

    def infer(self, image):
        h, w, _ = image.shape
        blob, pad_bias = self._preprocess(image)
        outputs = self.session.run(['Identity:0', 'Identity_1:0'], {self.input_name: blob})
        return self._postprocess(outputs, np.array([w, h]), pad_bias)


class _Pose(MPPose):
    def __init__(self, model_path):
        self.model_path = model_path
        self.conf_threshold = 0.5
        self.input_size = np.array([256, 256])
        self.PERSON_BOX_PRE_ENLARGE_FACTOR = 1
        self.PERSON_BOX_ENLARGE_FACTOR = 1.25
        self.session = ort.InferenceSession(model_path, providers=PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image, person):
        h, w, _ = image.shape
        blob, rotated_person_bbox, angle, rotation_matrix, pad_bias = self._preprocess(image, person)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, rotated_person_bbox, angle, rotation_matrix,
                                 pad_bias, np.array([w, h]))


class OnnxPerception:
    """Hand and pose detection returning MediaPipe-shaped normalized landmarks."""

    def __init__(self, models_dir=MODELS_DIR, palm_score_threshold=PALM_SCORE_THRESHOLD):
        self.palm = _PalmDet(
            os.path.join(models_dir, "palm_detection_mediapipe_2023feb.onnx"),
            palm_score_threshold,
        )
        self.handpose = _HandPose(
            os.path.join(models_dir, "handpose_estimation_mediapipe_2023feb.onnx"))
        self.person = _PersonDet(
            os.path.join(models_dir, "person_detection_mediapipe_2023mar.onnx"))
        self.pose = _Pose(
            os.path.join(models_dir, "pose_estimation_mediapipe_2023mar.onnx"))

    def detect_hands(self, frame_bgr, max_hands=2):
        """-> list of hands, each 21 x [x, y, z] normalized, or None."""
        h, w = frame_bgr.shape[:2]
        palms = self.palm.infer(frame_bgr)
        if palms is None or len(palms) == 0:
            return None

        hands = []
        for palm in palms[:max_hands]:
            result = self.handpose.infer(frame_bgr, palm)
            if result is None:
                continue
            # _postprocess packs [bbox(4), landmarks(21*3), world(21*3), handedness, conf]
            landmarks = np.asarray(result[4:67], dtype=np.float32).reshape(21, 3)
            landmarks[:, 0] /= w
            landmarks[:, 1] /= h
            hands.append(landmarks.tolist())
        return hands or None

    def detect_pose(self, frame_bgr):
        """-> [33 x [x, y, z]] normalized, wrapped in a list, or None."""
        h, w = frame_bgr.shape[:2]
        persons = self.person.infer(frame_bgr)
        if persons is None or len(persons) == 0:
            return None

        result = self.pose.infer(frame_bgr, persons[0])
        if result is None:
            return None
        # _postprocess returns [bbox, landmarks, world, mask, heatmap, conf]; landmarks is
        # 39 x [x, y, z, visibility, presence] -- the first 33 are MediaPipe Pose order.
        landmarks = np.asarray(result[1], dtype=np.float32)[:33, :3].copy()
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h
        return [landmarks.tolist()]


_instance = None
_instance_lock = threading.Lock()


def get_perception():
    """Process-wide singleton -- loading rebuilds/loads TensorRT engines."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = OnnxPerception()
    return _instance


if __name__ == "__main__":
    import time

    clip = sys.argv[1] if len(sys.argv) > 1 else "my_please.mp4"
    cap = cv2.VideoCapture(clip)
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 2 == 0:
            frames.append(f)
        i += 1
    cap.release()

    t0 = time.time()
    p = get_perception()
    print(f"models loaded in {time.time() - t0:.1f}s (cache: {CACHE_DIR})")
    print(f"palm provider: {p.palm.session.get_providers()[0]}")

    n_hand = n_pose = 0
    t_hand = t_pose = 0.0
    for f in frames[5:]:
        t0 = time.time()
        hands = p.detect_hands(f)
        t_hand += time.time() - t0
        t0 = time.time()
        pose = p.detect_pose(f)
        t_pose += time.time() - t0
        n_hand += hands is not None
        n_pose += pose is not None
        if hands:
            arr = np.array(hands[0])
            assert arr.shape == (21, 3), arr.shape
            assert 0 <= arr[:, 0].mean() <= 1, "hand x not normalized"
        if pose:
            arr = np.array(pose[0])
            assert arr.shape == (33, 3), arr.shape
            assert 0 <= arr[:, 0].mean() <= 1, "pose x not normalized"

    n = len(frames) - 5
    print(f"frames: {n}")
    print(f"  hands detected: {n_hand} ({n_hand / n * 100:.0f}%)  {t_hand / n * 1000:.1f} ms/frame")
    print(f"  pose  detected: {n_pose} ({n_pose / n * 100:.0f}%)  {t_pose / n * 1000:.1f} ms/frame")
    print("shapes and normalization OK")
