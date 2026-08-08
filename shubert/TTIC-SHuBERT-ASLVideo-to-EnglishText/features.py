import os
import time
import torch
import numpy as np
import cv2
import torch.nn as nn
import json
import cv2
from kpe_mediapipe import video_holistic
from crop_hands import HandExtractor
from crop_face import FaceExtractor
from dinov2_features import extract_embeddings_from_frames, preload_embedder
from body_features import process_pose_landmarks
# from shubert import SignHubertModel, SignHubertConfig
from inference import test, preload_model
import subprocess



class SHuBERTProcessor:

    def __init__(self, config):
        self.config = config

    def warmup(self):
        """Preload the DINOv2 and ByT5 models into their process-wide caches.

        Model loading (notably ByT5's ~13s / 2.68GB checkpoint) otherwise happens
        lazily on the first clip. In a long-running process — live capture, the
        Gradio app — calling this at startup moves that cost off the first
        translation. Safe to call more than once; subsequent calls are no-ops.
        """
        preload_embedder(self.config['dino_hands_model_path'])
        preload_embedder(self.config['dino_face_model_path'])
        preload_model(
            self.config['slt_model_checkpoint'],
            self.config['slt_tokenizer_checkpoint'],
            self.config['temp_dir'],
        )


    def process_video(self, video_path):
        
        # output_file = f"{output_path}/{os.path.basename(video_file)}"
    
        
        # # Target FPS is 12.5
        # cmd = [
        #     'ffmpeg',
        #     '-i', video_path,
        #     '-filter:v', 'fps=15',
        #     '-c:v', 'libx264',
        #     '-preset', 'medium',  # Balance between speed and quality
        #     '-crf', '23',  # Quality level (lower is better)
        #     '-y',  # Overwrite output file if it exists
        #     video_path
        # ]
        

        # try:
        #     subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #     print(f"Saved to {video_path} at 15 fps")
        # except subprocess.CalledProcessError as e:
        #     print(f"Error reading video {video_path}: {e}")
        
        
        
        timings = {}
        stage_start = time.time()

        # Step 1: Change the fps to 15
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        signer_video = np.array(frames)
        timings['video_read'] = time.time() - stage_start
        stage_start = time.time()

        # Step 2: Extract pose using kpe_mediapipe
        landmarks = video_holistic(
            video_input=signer_video,
            face_model_path=self.config['mediapipe_face_model_path'],
            hand_model_path=self.config['mediapipe_hands_model_path'],
        )
        timings['mediapipe_landmarks'] = time.time() - stage_start
        stage_start = time.time()

        # Step 3: Extract stream features
        hand_extractor = HandExtractor()
        left_hand_frames, right_hand_frames = hand_extractor.extract_hand_frames(signer_video, landmarks)
        timings['hand_crop'] = time.time() - stage_start
        stage_start = time.time()

        left_hand_embeddings = extract_embeddings_from_frames(left_hand_frames, self.config['dino_hands_model_path'])
        right_hand_embeddings = extract_embeddings_from_frames(right_hand_frames, self.config['dino_hands_model_path'])
        del left_hand_frames, right_hand_frames
        timings['dinov2_hands'] = time.time() - stage_start
        stage_start = time.time()

        face_extractor = FaceExtractor()
        face_frames = face_extractor.extract_face_frames(signer_video, landmarks)
        timings['face_crop'] = time.time() - stage_start
        stage_start = time.time()

        face_embeddings = extract_embeddings_from_frames(face_frames, self.config['dino_face_model_path'])
        del face_frames, signer_video
        timings['dinov2_face'] = time.time() - stage_start
        stage_start = time.time()

        pose_embeddings = process_pose_landmarks(landmarks)
        del landmarks
        timings['pose_features'] = time.time() - stage_start
        stage_start = time.time()

        output_text = test(face_embeddings,
             left_hand_embeddings,
             right_hand_embeddings,
             pose_embeddings,
             self.config['slt_model_config'],
             self.config['slt_model_checkpoint'],
             self.config['slt_tokenizer_checkpoint'],
             self.config['temp_dir'])
        timings['byt5_inference'] = time.time() - stage_start

        print("\n--- Stage timing breakdown ---")
        for stage_name, elapsed in timings.items():
            print(f"  {stage_name:20s}: {elapsed:6.1f}s")
        print(f"  {'total':20s}: {sum(timings.values()):6.1f}s")

        return output_text
        
if __name__ == "__main__":
    config = {
        'yolov8_model_path': '/share/data/pals/shester/inference/models/yolov8n.pt',
        'dino_face_model_path': '/share/data/pals/shester/inference/models/dinov2face.pth',
        'dino_hands_model_path': '/share/data/pals/shester/inference/models/dinov2hand.pth',
        'mediapipe_face_model_path': '/share/data/pals/shester/inference/models/face_landmarker_v2_with_blendshapes.task',
        'mediapipe_hands_model_path': '/share/data/pals/shester/inference/models/hand_landmarker.task',
        'shubert_model_path': '/share/data/pals/shester/inference/models/checkpoint_836_400000.pt',
        'temp_dir': '/share/data/pals/shester/inference',
        'slt_model_config': '/share/data/pals/shester/inference/models/byt5_base/config.json',
        'slt_model_checkpoint': '/share/data/pals/shester/inference/models/checkpoint-11625',
        'slt_tokenizer_checkpoint': '/share/data/pals/shester/inference/models/byt5_base',
    }
    
    # input_clip = "/share/data/pals/shester/datasets/openasl/clips_bbox/J-0KHhPS_m4.029676-029733.mp4"  
    # input_clip = "/share/data/pals/shester/inference/recordings/sabrin30fps.mp4"
    input_clip = "/share/data/pals/shester/inference/recordings/sample_sabrina.mp4"
    processor = SHuBERTProcessor(config) 
    output_text = processor.process_video(input_clip)
    print(f"The English translation is: {output_text}")
    
# /home-nfs/shesterg/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py
# /home-nfs/shesterg/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py