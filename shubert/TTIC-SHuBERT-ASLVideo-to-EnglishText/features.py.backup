import os
import torch
import numpy as np
import decord
import torch.nn as nn
import json
import cv2
from kpe_mediapipe import video_holistic
from crop_hands import HandExtractor
from crop_face import FaceExtractor
from dinov2_features import extract_embeddings_from_frames
from body_features import process_pose_landmarks
# from shubert import SignHubertModel, SignHubertConfig
from inference import test
import subprocess



class SHuBERTProcessor:

    def __init__(self, config):
        self.config = config
    
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
        
        
        
        # Step 1: Change the fps to 15
        signer_video = decord.VideoReader(video_path)
        
        signer_video_fps = signer_video.get_avg_fps()
        # target_fps = 12
        # stride = max(1, int(round(signer_video_fps / target_fps)))
        stride = 1
        index_list = list(range(0, len(signer_video), stride))
        signer_video = signer_video.get_batch(index_list)
        signer_video = signer_video.asnumpy()
        
        # Step 2: Extract pose using kpe_mediapipe 
        landmarks = video_holistic(
            video_input=signer_video,
            face_model_path=self.config['mediapipe_face_model_path'],
            hand_model_path=self.config['mediapipe_hands_model_path'],
        )
             
        # Step 3: Extract stream features
        hand_extractor = HandExtractor()
        left_hand_frames, right_hand_frames = hand_extractor.extract_hand_frames(signer_video, landmarks)
        left_hand_embeddings = extract_embeddings_from_frames(left_hand_frames, self.config['dino_hands_model_path'])
        right_hand_embeddings = extract_embeddings_from_frames(right_hand_frames, self.config['dino_hands_model_path'])
        del left_hand_frames, right_hand_frames
        
        face_extractor = FaceExtractor()
        face_frames = face_extractor.extract_face_frames(signer_video, landmarks)
        face_embeddings = extract_embeddings_from_frames(face_frames, self.config['dino_face_model_path'])
        del face_frames, signer_video
          
        pose_embeddings = process_pose_landmarks(landmarks)
        del landmarks

        output_text = test(face_embeddings, 
             left_hand_embeddings, 
             right_hand_embeddings, 
             pose_embeddings, 
             self.config['slt_model_config'], 
             self.config['slt_model_checkpoint'], 
             self.config['slt_tokenizer_checkpoint'], 
             self.config['temp_dir'])

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