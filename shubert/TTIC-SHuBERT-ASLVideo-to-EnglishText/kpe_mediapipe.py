import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import json
import time
import concurrent.futures
from pathlib import Path
try:
    import decord
except ImportError:
    decord = None
from typing import Dict, Optional, Tuple, Any


class HolisticDetector:
    """
    A class for detecting face, hand, and pose landmarks in videos using MediaPipe.
    """
    
    def __init__(self, face_model_path: str, hand_model_path: str,
                 min_detection_confidence: float = 0.1,
                 min_hand_detection_confidence: float = 0.05,
                 max_faces: int = 6, max_hands: int = 2):
        """
        Initialize the HolisticDetector with model paths and configuration.
        
        Args:
            face_model_path: Path to the face detection model
            hand_model_path: Path to the hand detection model
            min_detection_confidence: Minimum confidence for pose detection
            min_hand_detection_confidence: Minimum confidence for hand detection
            max_faces: Maximum number of faces to detect
            max_hands: Maximum number of hands to detect. Defaults to 2 (one
                signer's two hands) rather than MediaPipe's default of larger
                values meant for multi-person scenes; the hand detector's
                search cost scales with this, so lowering it materially cuts
                per-frame latency.
        """
        self.face_model_path = face_model_path
        self.hand_model_path = hand_model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.max_faces = max_faces
        self.max_hands = max_hands
        self.use_onnx = os.environ.get("USE_ONNX_PERCEPTION", "1") not in ("0", "false", "False")
        self._onnx = None

        # Run pose/face/hand detection concurrently per frame instead of
        # sequentially. MediaPipe's underlying C++ inference releases the GIL,
        # so this gives real parallelism rather than just interleaving.
        self._executor = None
        self._initialize_detectors()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        # Per-frame (pose, face, hand) timings, for profiling the critical path
        # inside the concurrent detect_frame_landmarks() call.
        self._frame_timings = []

    def close(self):
        """Release the thread pool and MediaPipe native resources.

        Safe to call multiple times. Closing the MediaPipe objects matters:
        a fresh detector is built per clip, and the underlying graphs hold
        native memory that is not reclaimed by Python GC alone — leaking it
        exhausts the Jetson's shared CPU/GPU pool after a few clips.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        for attr in ('mp_holistic', 'face_detector', 'hand_detector'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __del__(self):
        self.close()

    def print_timing_summary(self):
        """Print a per-detector timing breakdown collected across processed frames."""
        n = len(self._frame_timings)
        if n == 0:
            return
        totals = {k: sum(t[k] for t in self._frame_timings) for k in ('pose', 'face', 'hand', 'wall')}
        backend = "ONNX/TRT hand+pose, MediaPipe face" if self.use_onnx else "MediaPipe"
        print(f"\n--- Perception per-frame timing ({backend}, {n} frames) ---")
        for key in ('pose', 'face', 'hand'):
            total = totals[key]
            print(f"  {key:6s}: {total:6.1f}s total, {1000 * total / n:6.1f}ms/frame avg")
        print(f"  {'wall':6s}: {totals['wall']:6.1f}s total, {1000 * totals['wall'] / n:6.1f}ms/frame avg"
              f" (critical path: max(pose,face,hand) + thread overhead)")
    
    def _initialize_detectors(self):
        """Initialize the detectors.

        Face is always MediaPipe. Hand and pose come from the ONNX/TensorRT stack
        unless USE_ONNX_PERCEPTION=0 -- see onnx_perception.py for why (~10x faster
        for those two, which are the ones on the critical path).
        """
        # Initialize face detector -- MediaPipe in both modes. At 41.8 ms/frame it is
        # already under the ONNX path's ~51.5 ms wall, so converting it gains nothing.
        base_options_face = python.BaseOptions(model_asset_path=self.face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=self.max_faces
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(options_face)

        if self.use_onnx:
            from onnx_perception import get_perception
            # Process-wide singleton: loading it maps the TensorRT engines, which is
            # far too expensive to repeat per clip.
            self._onnx = get_perception()
            self.hand_detector = None
            self.mp_holistic = None
            return

        # Initialize hand detector
        base_options_hand = python.BaseOptions(model_asset_path=self.hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options_hand)

        # Initialize holistic model for pose
        self.mp_holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence
        )

    def detect_frame_landmarks(self, image: np.ndarray) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Detect landmarks in a single frame.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (bounding_boxes_count, landmarks_data)
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        def _timed(fn, *args):
            t0 = time.time()
            result = fn(*args)
            return result, time.time() - t0

        frame_start = time.time()
        if self.use_onnx:
            # The ONNX wrappers cvtColor BGR2RGB internally, but `image` is already RGB
            # (features.py converts on read), so hand it BGR or every detection is run
            # on colour-swapped input.
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run pose and hands SEQUENTIALLY in one task rather than as two concurrent
            # ones. MediaPipe's detectors are CPU-bound and genuinely parallelize, so
            # wall was max(pose, face, hand); these two are GPU-bound and serialize on
            # the device regardless. Submitting them separately just adds contention --
            # measured 112.7 ms/frame concurrent vs a 72.8 ms sequential sum. Face is
            # still submitted separately because it is CPU-bound and does overlap.
            def _pose_then_hands():
                t0 = time.time()
                pose_result = self._onnx.detect_pose(frame_bgr)
                pose_elapsed = time.time() - t0
                t0 = time.time()
                hand_result = self._onnx.detect_hands(frame_bgr, self.max_hands)
                return (pose_result, pose_elapsed), (hand_result, time.time() - t0)

            gpu_future = self._executor.submit(_pose_then_hands)
        else:
            pose_future = self._executor.submit(_timed, self.mp_holistic.process, image)
            hand_future = self._executor.submit(_timed, self.hand_detector.detect, mp_image)
        face_future = self._executor.submit(_timed, self.face_detector.detect, mp_image)

        if self.use_onnx:
            (pose_raw, pose_dt), (hand_raw, hand_dt) = gpu_future.result()
        else:
            pose_raw, pose_dt = pose_future.result()
            hand_raw, hand_dt = hand_future.result()
        face_prediction, face_dt = face_future.result()

        # Normalize both backends onto the same shapes before assembling the output:
        #   hands: list of hands, each 21 x [x, y, z] normalized, or None
        #   pose : [33 x [x, y, z]] normalized, or None
        if self.use_onnx:
            onnx_hand_landmarks = hand_raw
            onnx_pose_landmarks = pose_raw
        else:
            onnx_hand_landmarks = onnx_pose_landmarks = None
            hand_prediction = hand_raw
            results = pose_raw

        self._frame_timings.append({
            'pose': pose_dt, 'face': face_dt, 'hand': hand_dt,
            'wall': time.time() - frame_start,
        })

        bounding_boxes = {}
        landmarks_data = {}

        # Process face landmarks
        if face_prediction.face_landmarks:
            bounding_boxes['#face'] = len(face_prediction.face_landmarks)
            landmarks_data['face_landmarks'] = []
            for face in face_prediction.face_landmarks:
                landmarks_face = [[landmark.x, landmark.y, landmark.z] for landmark in face]
                landmarks_data['face_landmarks'].append(landmarks_face)
        else:
            bounding_boxes['#face'] = 0
            landmarks_data['face_landmarks'] = None

        # Process hand landmarks
        if self.use_onnx:
            if onnx_hand_landmarks:
                bounding_boxes['#hands'] = len(onnx_hand_landmarks)
                landmarks_data['hand_landmarks'] = onnx_hand_landmarks
            else:
                bounding_boxes['#hands'] = 0
                landmarks_data['hand_landmarks'] = None
        elif hand_prediction.hand_landmarks:
            bounding_boxes['#hands'] = len(hand_prediction.hand_landmarks)
            landmarks_data['hand_landmarks'] = []
            for hand in hand_prediction.hand_landmarks:
                landmarks_hand = [[landmark.x, landmark.y, landmark.z] for landmark in hand]
                landmarks_data['hand_landmarks'].append(landmarks_hand)
        else:
            bounding_boxes['#hands'] = 0
            landmarks_data['hand_landmarks'] = None

        # Process pose landmarks
        if self.use_onnx:
            if onnx_pose_landmarks:
                bounding_boxes['#pose'] = 1
                landmarks_data['pose_landmarks'] = onnx_pose_landmarks
            else:
                bounding_boxes['#pose'] = 0
                landmarks_data['pose_landmarks'] = None
        elif results.pose_landmarks:
            bounding_boxes['#pose'] = 1
            landmarks_data['pose_landmarks'] = []
            pose_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]
            landmarks_data['pose_landmarks'].append(pose_landmarks)
        else:
            bounding_boxes['#pose'] = 0
            landmarks_data['pose_landmarks'] = None

        return bounding_boxes, landmarks_data

    def process_video(self, video_input, save_results: bool = False, 
                     output_dir: Optional[str] = None, video_name: Optional[str] = None) -> Dict[int, Any]:
        """
        Process a video and extract landmarks from all frames.
        
        Args:
            video_input: Either a path to video file (str) or a decord.VideoReader object
            save_results: Whether to save results to files
            output_dir: Directory to save results (required if save_results=True)
            video_name: Name for output files (required if save_results=True and video_input is VideoReader)
            
        Returns:
            Dictionary containing landmarks for each frame
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If save_results=True but output_dir is None, or if video_name is None when needed
            TypeError: If video_input is neither string nor VideoReader
        """
        if save_results and output_dir is None:
            raise ValueError("output_dir must be provided when save_results=True")
        
        # Handle different input types
        if isinstance(video_input, str):
            # Input is a file path
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            
            try:
                video = decord.VideoReader(str(video_path))
            except Exception as e:
                raise RuntimeError(f"Error loading video {video_input}: {e}")
                
            file_name = video_path.stem
            
        # elif hasattr(video_input, '__len__') and hasattr(video_input, '__getitem__'):
        else:
            # Input is a VideoReader object or similar
            video = video_input
            if save_results and video_name is None:
                raise ValueError("video_name must be provided when save_results=True and video_input is a VideoReader object")
            file_name = video_name or "video"
            
        # else:
        #     raise TypeError("video_input must be either a file path (str) or a VideoReader object")
        
        result_dict = {}
        stats = {}
        self._frame_timings = []

        # Process each frame
        for i in range(len(video)):
            try:
                # frame_rgb = video[i].asnumpy()
                frame_rgb = video[i]
                if hasattr(video, 'seek'):
                    video.seek(0)
                bounding_boxes, landmarks = self.detect_frame_landmarks(frame_rgb)
                result_dict[i] = landmarks
                stats[i] = bounding_boxes
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                result_dict[i] = None
                stats[i] = {'#face': 0, '#hands': 0, '#pose': 0}
        
        # Save results if requested
        if save_results:
            self._save_results(file_name, result_dict, stats, output_dir)

        self.print_timing_summary()
        return result_dict

    def process_video_frames(self, frames: list, save_results: bool = False,
                           output_dir: Optional[str] = None, video_name: str = "video") -> Dict[int, Any]:
        """
        Process a list of frames and extract landmarks.
        
        Args:
            frames: List of frame images as numpy arrays
            save_results: Whether to save results to files
            output_dir: Directory to save results (required if save_results=True)
            video_name: Name for output files
            
        Returns:
            Dictionary containing landmarks for each frame
        """
        if save_results and output_dir is None:
            raise ValueError("output_dir must be provided when save_results=True")
        
        result_dict = {}
        stats = {}
        self._frame_timings = []

        # Process each frame
        for i, frame in enumerate(frames):
            try:
                bounding_boxes, landmarks = self.detect_frame_landmarks(frame)
                result_dict[i] = landmarks
                stats[i] = bounding_boxes
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                result_dict[i] = None
                stats[i] = {'#face': 0, '#hands': 0, '#pose': 0}
        
        # Save results if requested
        if save_results:
            self._save_results(video_name, result_dict, stats, output_dir)

        self.print_timing_summary()
        return result_dict

    def _save_results(self, video_name: str, landmarks_data: Dict, stats_data: Dict, output_dir: str):
        """Save landmarks and stats to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save landmarks
        landmarks_file = output_path / f"{video_name}_pose.json"
        with open(landmarks_file, 'w') as f:
            json.dump(landmarks_data, f)
        
        # Save stats
        stats_file = output_path / f"{video_name}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f)

    def compute_video_stats(self, landmarks_data: Dict) -> Dict[str, Any]:
        """
        Compute statistics from landmarks data.
        
        Args:
            landmarks_data: Dictionary containing landmarks for each frame
            
        Returns:
            Dictionary containing frame-by-frame stats and maximums
        """
        stats = {}
        max_counts = {'#face': 0, '#hands': 0, '#pose': 0}
        
        for frame, landmarks in landmarks_data.items():
            if landmarks is None:
                presence = {'#face': 0, '#hands': 0, '#pose': 0}
            else:
                presence = {
                    '#face': len(landmarks.get('face_landmarks', [])) if landmarks.get('face_landmarks') else 0,
                    '#hands': len(landmarks.get('hand_landmarks', [])) if landmarks.get('hand_landmarks') else 0,
                    '#pose': len(landmarks.get('pose_landmarks', [])) if landmarks.get('pose_landmarks') else 0
                }
            stats[frame] = presence
            
            # Update max counts
            for key in max_counts:
                max_counts[key] = max(max_counts[key], presence[key])
        
        stats['max'] = max_counts
        return stats


# NOTE: deliberately NOT cached across clips. `mp.solutions.holistic.Holistic`
# tracks landmarks temporally across successive process() calls, which is what we
# want *within* a clip but not *between* clips — reusing one detector leaks the
# previous clip's tracking state into the next and produced different translations
# for identical input. Constructing a detector is cheap relative to per-frame
# inference (measured: no difference in the MediaPipe stage), so there is nothing
# to gain by caching it here.
# Convenience function for backward compatibility and simple usage
def video_holistic(video_input, face_model_path: str, hand_model_path: str,
                  save_results: bool = False, output_dir: Optional[str] = None,
                  video_name: Optional[str] = None) -> Dict[int, Any]:
    """
    Convenience function to process a video and extract holistic landmarks.
    
    Args:
        video_input: Either a path to video file (str) or a decord.VideoReader object
        face_model_path: Path to the face detection model
        hand_model_path: Path to the hand detection model
        save_results: Whether to save results to files
        output_dir: Directory to save results
        video_name: Name for output files (required if save_results=True and video_input is VideoReader)
        
    Returns:
        Dictionary containing landmarks for each frame
    """
    detector = HolisticDetector(face_model_path, hand_model_path)
    try:
        return detector.process_video(video_input, save_results, output_dir, video_name)
    finally:
        detector.close()


# Utility functions for batch processing
def load_file(filename: str):
    """Load a pickled and gzipped file."""
    import pickle
    import gzip
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def is_string_in_file(file_path: str, target_string: str) -> bool:
    """Check if a string exists in a file."""
    try:
        with Path(file_path).open("r") as f:
            for line in f:
                if target_string in line:
                    return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main function for command-line usage."""
    import argparse
    import time
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--pose_path', type=str, required=True,
                        help='path to where the pose data will be saved')
    parser.add_argument('--stats_path', type=str, required=True,
                        help='path to where the stats data will be saved')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit')
    parser.add_argument('--files_list', type=str, required=True,
                        help='files list')
    parser.add_argument('--problem_file_path', type=str, required=True,
                        help='problem file path')
    parser.add_argument('--face_model_path', type=str, required=True,
                        help='face model path')
    parser.add_argument('--hand_model_path', type=str, required=True,
                        help='hand model path')

    args = parser.parse_args()
    
    start_time = time.time()

    # Initialize detector
    detector = HolisticDetector(args.face_model_path, args.hand_model_path)

    # Load the files list
    fixed_list = load_file(args.files_list)

    # Create folders if they do not exist
    Path(args.pose_path).mkdir(parents=True, exist_ok=True)
    Path(args.stats_path).mkdir(parents=True, exist_ok=True)

    # Create problem file if it doesn't exist
    if not os.path.exists(args.problem_file_path):
        with open(args.problem_file_path, 'w') as f:
            pass

    # Process videos in batches
    video_batches = [fixed_list[i:i + args.batch_size] for i in range(0, len(fixed_list), args.batch_size)]

    try:
        for video_file in video_batches[args.index]:
            current_time = time.time()
            if current_time - start_time > args.time_limit:
                print("Time limit reached. Stopping execution.")
                break

            # Check if output files already exist
            video_name = Path(video_file).stem
            landmark_json_path = Path(args.pose_path) / f"{video_name}_pose.json"
            stats_json_path = Path(args.stats_path) / f"{video_name}_stats.json"

            if landmark_json_path.exists() and stats_json_path.exists():
                print(f"Skipping {video_file} - output files already exist")
                continue
            elif is_string_in_file(args.problem_file_path, video_file):
                print(f"Skipping {video_file} - found in problem file")
                continue
            else:
                try:
                    print(f"Processing {video_file}")
                    result_dict = detector.process_video(
                        video_file_path=video_file,
                        save_results=True,
                        output_dir=args.pose_path
                    )

                    # Also save stats separately for compatibility
                    stats = detector.compute_video_stats(result_dict)
                    with open(stats_json_path, 'w') as f:
                        json.dump(stats, f)

                    print(f"Successfully processed {video_file}")

                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
                    # Add to problem file
                    with open(args.problem_file_path, "a") as p:
                        p.write(video_file + "\n")
    finally:
        detector.close()


if __name__ == "__main__":
    main()