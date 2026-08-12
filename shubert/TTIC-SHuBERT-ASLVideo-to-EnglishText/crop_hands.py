import cv2
import numpy as np
import os
import pickle
import gzip
from datetime import datetime
from pathlib import Path
try:
    import decord
except ImportError:
    decord = None
import argparse
import json
import time
from typing import Dict, Optional, Tuple, List, Union, Any
from crop_jitter import jitter_xywh
import tempfile


class HandExtractor:
    """
    A class for extracting hand regions from videos based on pose landmarks.
    """
    
    def __init__(self, output_size: Tuple[int, int] = (224, 224), 
                 scale_factor: float = 1.5, distance_threshold: float = 0.1):
        """
        Initialize the HandExtractor.
        
        Args:
            output_size: Size of the output hand frames (width, height)
            scale_factor: Scale factor for bounding box expansion
            distance_threshold: Distance threshold for hand-pose matching
        """
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.distance_threshold = distance_threshold
        self._reset_state()

    def _reset_state(self):
        """Clear the cross-frame fallback state.

        Cropping is sequential: a frame with no detected hand reuses the previous frame's
        crop. Holding that on the instance (rather than in locals) lets the streaming path
        extract crops in chunks as landmarks arrive and still get identical output, since
        state carries across calls. extract_hand_frames() resets first, so the one-shot
        path behaves exactly as before.
        """
        self._prev_left_frame = None
        self._prev_right_frame = None
        self._prev_landmarks = None
        # Frame counter, so crop_jitter's perturbation is a pure function of position in
        # the clip rather than of call order. extract_hand_frames() resets it, and the
        # streaming path continues it across chunks -- same contract as the crop state.
        self._frame_index = 0

    def _blank(self) -> np.ndarray:
        return np.zeros((*self.output_size, 3), dtype=np.uint8)

    def _process_one(self, frame, frame_landmarks):
        """Crop one frame's hands, updating fallback state. Returns (left, right)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index = self._frame_index
        self._frame_index += 1

        if frame_landmarks is None:
            if self._prev_landmarks is not None:
                frame_landmarks = self._prev_landmarks
            else:
                return self._blank(), self._blank()
        else:
            self._prev_landmarks = frame_landmarks

        if frame_landmarks.get('pose_landmarks') is None:
            left = self._prev_left_frame if self._prev_left_frame is not None else self._blank()
            right = self._prev_right_frame if self._prev_right_frame is not None else self._blank()
            return left, right

        left_hand_landmarks, right_hand_landmarks = self.select_hands(
            frame_landmarks['pose_landmarks'][0],
            frame_landmarks.get('hand_landmarks'),
            frame_rgb.shape
        )

        if left_hand_landmarks is not None:
            left_box = self.get_bounding_box(left_hand_landmarks, frame_rgb.shape, self.scale_factor)
            # Measurement hook, no-op unless CROP_JITTER_* is set. Applied BEFORE the
            # clamp so a displaced box is still legal.
            left_box = jitter_xywh(left_box, "left_hand", index)
            left_box = self.adjust_bounding_box(left_box, frame_rgb.shape)
            left_frame = self.resize_frame(self.crop_frame(frame_rgb, left_box), self.output_size)
            self._prev_left_frame = left_frame
        elif self._prev_left_frame is not None:
            left_frame = self._prev_left_frame
        else:
            left_frame = self._blank()

        if right_hand_landmarks is not None:
            right_box = self.get_bounding_box(right_hand_landmarks, frame_rgb.shape, self.scale_factor)
            right_box = jitter_xywh(right_box, "right_hand", index)
            right_box = self.adjust_bounding_box(right_box, frame_rgb.shape)
            right_frame = self.resize_frame(self.crop_frame(frame_rgb, right_box), self.output_size)
            self._prev_right_frame = right_frame
        elif self._prev_right_frame is not None:
            right_frame = self._prev_right_frame
        else:
            right_frame = self._blank()

        return left_frame, right_frame

    def extract_hand_frames_chunk(self, frames, landmarks_list):
        """Crop a chunk of consecutive frames, CONTINUING state from the previous call.

        For the streaming path only. Feed chunks in capture order on one instance and the
        result is identical to one extract_hand_frames() call over the whole clip.
        """
        left_hand_frames, right_hand_frames = [], []
        for frame, frame_landmarks in zip(frames, landmarks_list):
            left_frame, right_frame = self._process_one(frame, frame_landmarks)
            left_hand_frames.append(left_frame)
            right_hand_frames.append(right_frame)
        return left_hand_frames, right_hand_frames


    def resize_frame(self, frame: np.ndarray, frame_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Resize frame to specified size."""
        if frame is not None and frame.size > 0:
            return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        else:
            return None

    def crop_frame(self, image: np.ndarray, bounding_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop frame using bounding box."""
        x, y, w, h = bounding_box
        cropped_frame = image[y:y + h, x:x + w]
        return cropped_frame

    def get_bounding_box(self, landmarks: List[List[float]], image_shape: Tuple[int, int, int], 
                        scale_factor: float = 1.2) -> Tuple[int, int, int, int]:
        """Get bounding box from landmarks."""
        ih, iw, _ = image_shape
        landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
        center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
        xb, yb, wb, hb = cv2.boundingRect(landmarks_px)
        box_size = max(wb, hb)
        half_size = box_size // 2
        x = center_x - half_size
        y = center_y - half_size
        w = box_size
        h = box_size
        
        w_padding = int((scale_factor - 1) * w / 2)
        h_padding = int((scale_factor - 1) * h / 2)
        x -= w_padding
        y -= h_padding
        w += 2 * w_padding
        h += 2 * h_padding    
        
        return x, y, w, h

    def adjust_bounding_box(self, bounding_box: Tuple[int, int, int, int], 
                           image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Adjust bounding box to fit within image boundaries."""
        x, y, w, h = bounding_box
        ih, iw, _ = image_shape

        # Adjust x-coordinate if the bounding box extends beyond the image's right edge
        if x + w > iw:
            x = iw - w
        
        # Adjust y-coordinate if the bounding box extends beyond the image's bottom edge
        if y + h > ih:
            y = ih - h
        
        # Ensure bounding box's x and y coordinates are not negative
        x = max(x, 0)
        y = max(y, 0)    

        return x, y, w, h

    def select_hands(self, pose_landmarks: List[List[float]], hand_landmarks: Optional[List[List[List[float]]]], 
                    image_shape: Tuple[int, int, int]) -> Tuple[Optional[List[List[float]]], Optional[List[List[float]]]]:
        """
        Select left and right hands from detected hand landmarks based on pose wrist positions.
        
        Args:
            pose_landmarks: Pose landmarks from MediaPipe
            hand_landmarks: Hand landmarks from MediaPipe
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Tuple of (left_hand_landmarks, right_hand_landmarks)
        """
        if hand_landmarks is None:
            return None, None
        
        # Get wrist landmarks from pose (indices 15 and 16 for left and right wrists)
        left_wrist_from_pose = pose_landmarks[15]
        right_wrist_from_pose = pose_landmarks[16]
        
        # Get wrist landmarks from hand detections (index 0 is wrist in hand landmarks)
        wrist_from_hand = [hand_landmarks[i][0] for i in range(len(hand_landmarks))]
        
        # Match right hand
        right_hand_landmarks = None
        if right_wrist_from_pose is not None:
            minimum_distance = 100
            best_hand_idx = 0
            for i in range(len(hand_landmarks)):
                distance = np.linalg.norm(np.array(right_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
                if distance < minimum_distance:
                    minimum_distance = distance
                    best_hand_idx = i
            
            if minimum_distance < self.distance_threshold:
                right_hand_landmarks = hand_landmarks[best_hand_idx]
        
        # Match left hand
        left_hand_landmarks = None
        if left_wrist_from_pose is not None:
            minimum_distance = 100
            best_hand_idx = 0
            for i in range(len(hand_landmarks)):
                distance = np.linalg.norm(np.array(left_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
                if distance < minimum_distance:
                    minimum_distance = distance
                    best_hand_idx = i
            
            if minimum_distance < self.distance_threshold:
                left_hand_landmarks = hand_landmarks[best_hand_idx]
        
        return left_hand_landmarks, right_hand_landmarks

    def extract_hand_frames(self, video_input, landmarks_data: Dict[int, Any]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract hand frames from video based on landmarks.
        
        Args:
            video_input: Either a path to video file (str) or a decord.VideoReader object
            landmarks_data: Dictionary containing pose and hand landmarks for each frame
            
        Returns:
            Tuple of (left_hand_frames, right_hand_frames) as lists of numpy arrays
        """
        # Handle different input types
        if isinstance(video_input, str):
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            video = decord.VideoReader(str(video_path))
        # elif hasattr(video_input, '__len__') and hasattr(video_input, '__getitem__'):
        else:
            video = video_input
        # else:
        #     raise TypeError("video_input must be either a file path (str) or a VideoReader object")
        
        left_hand_frames = []
        right_hand_frames = []

        # One-shot over a whole clip: start from clean state so behaviour is unchanged.
        self._reset_state()

        for i in range(len(video)):
            # frame = video[i].asnumpy()
            frame = video[i]
            if hasattr(video, 'seek'):
                video.seek(0)
            left_frame, right_frame = self._process_one(frame, landmarks_data.get(i, None))
            left_hand_frames.append(left_frame)
            right_hand_frames.append(right_frame)

        return left_hand_frames, right_hand_frames

    def extract_and_save_hand_videos(self, video_input, landmarks_data: Dict[int, Any], 
                                   output_dir: str, video_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Extract hand frames and save as video files.
        
        Args:
            video_input: Either a path to video file (str) or a decord.VideoReader object
            landmarks_data: Dictionary containing pose and hand landmarks for each frame
            output_dir: Directory to save the hand videos
            video_name: Name for output videos (auto-generated if not provided)
            
        Returns:
            Tuple of (left_hand_video_path, right_hand_video_path)
        """
        # Handle video input and get FPS
        if isinstance(video_input, str):
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            video = decord.VideoReader(str(video_path))
            if video_name is None:
                video_name = video_path.stem
        # elif hasattr(video_input, '__len__') and hasattr(video_input, '__getitem__'):
        else:
            video = video_input
            if video_name is None:
                video_name = "video"
        # else:
        #     raise TypeError("video_input must be either a file path (str) or a VideoReader object")
        
        fps = video.get_avg_fps() if hasattr(video, 'get_avg_fps') else 30.0
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define output paths
        left_hand_path = output_path / f"{video_name}_hand1.mp4"
        right_hand_path = output_path / f"{video_name}_hand2.mp4"
        
        # Remove existing files
        if left_hand_path.exists():
            left_hand_path.unlink()
        if right_hand_path.exists():
            right_hand_path.unlink()
        
        # Create video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        left_writer = cv2.VideoWriter(str(left_hand_path), fourcc, fps, self.output_size)
        right_writer = cv2.VideoWriter(str(right_hand_path), fourcc, fps, self.output_size)
        
        try:
            # Extract hand frames
            left_frames, right_frames = self.extract_hand_frames(video, landmarks_data)
            
            # Write frames to video files
            for left_frame, right_frame in zip(left_frames, right_frames):
                left_writer.write(left_frame)
                right_writer.write(right_frame)
                
        finally:
            # Clean up
            left_writer.release()
            right_writer.release()
            del left_writer
            del right_writer
        
        return str(left_hand_path), str(right_hand_path)


# Convenience function for backward compatibility
def extract_hand_frames(video_input, landmarks_data: Dict[int, Any], 
                       output_size: Tuple[int, int] = (224, 224)) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Convenience function to extract hand frames from video.
    
    Args:
        video_input: Either a path to video file (str) or a decord.VideoReader object
        landmarks_data: Dictionary containing pose and hand landmarks for each frame
        output_size: Size of the output hand frames (width, height)
        
    Returns:
        Tuple of (left_hand_frames, right_hand_frames) as lists of numpy arrays
    """
    extractor = HandExtractor(output_size=output_size)
    return extractor.extract_hand_frames(video_input, landmarks_data)


def video_holistic(video_file: str, hand_path: str, problem_file_path: str, pose_path: str):
    """
    Original function for backward compatibility with command-line usage.
    """
    try:
        video = decord.VideoReader(video_file)
        fps = video.get_avg_fps()

        video_name = Path(video_file).stem
        clip_hand1_path = Path(hand_path) / f"{video_name}_hand1.mp4"
        clip_hand2_path = Path(hand_path) / f"{video_name}_hand2.mp4"
        landmark_json_path = Path(pose_path) / f"{video_name}_pose.json"

        # Load landmarks
        with open(landmark_json_path, 'r') as rd:
            landmarks_data = json.load(rd)

        # Convert string keys to integers
        landmarks_data = {int(k): v for k, v in landmarks_data.items()}

        # Extract hand videos
        extractor = HandExtractor()
        extractor.extract_and_save_hand_videos(video, landmarks_data, hand_path, video_name)

    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        with open(problem_file_path, "a") as p:
            p.write(video_file + "\n")


# Utility functions for batch processing
def load_file(filename: str):
    """Load a pickled and gzipped file."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit')
    parser.add_argument('--files_list', type=str, required=True,
                        help='files list')
    parser.add_argument('--problem_file_path', type=str, required=True,
                        help='problem file path')
    parser.add_argument('--pose_path', type=str, required=True,
                        help='pose path')
    parser.add_argument('--hand_path', type=str, required=True,
                        help='hand path')

    args = parser.parse_args()
    start_time = time.time()

    # Create directories if they do not exist
    Path(args.hand_path).mkdir(parents=True, exist_ok=True)
    
    # Load files list
    fixed_list = load_file(args.files_list)
    
    # Create problem file if it doesn't exist
    if not os.path.exists(args.problem_file_path):
        with open(args.problem_file_path, "w") as f:
            f.write("")
    
    # Process videos in batches
    video_batches = [fixed_list[i:i + args.batch_size] for i in range(0, len(fixed_list), args.batch_size)]
    
    for video_file in video_batches[args.index]:
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            print("Time limit reached. Stopping execution.")
            break

        video_name = Path(video_file).stem
        clip_hand2_path = Path(args.hand_path) / f"{video_name}_hand2.mp4"
        
        if clip_hand2_path.exists():
            print(f"Skipping {video_file} - output already exists")
            continue
        elif is_string_in_file(args.problem_file_path, video_file):
            print(f"Skipping {video_file} - found in problem file")
            continue
        else:
            try:
                print(f"Processing {video_file}")
                video_holistic(video_file, args.hand_path, args.problem_file_path, args.pose_path)
                print(f"Successfully processed {video_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                with open(args.problem_file_path, "a") as p:
                    p.write(video_file + "\n")


if __name__ == "__main__":
    main()