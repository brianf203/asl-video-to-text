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
from dinov2_features import (extract_embeddings_from_frames, preload_embedder,
                             HANDS_DTYPE, FACE_DTYPE)
from body_features import process_pose_landmarks
# from shubert import SignHubertModel, SignHubertConfig
from inference import test, preload_model



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
        preload_embedder(self.config['dino_hands_model_path'], dtype=HANDS_DTYPE)
        preload_embedder(self.config['dino_face_model_path'], dtype=FACE_DTYPE)
        preload_model(
            self.config['slt_model_checkpoint'],
            self.config['slt_tokenizer_checkpoint'],
            self.config['temp_dir'],
        )


    def process_video(self, video_path):

        timings = {}
        stage_start = time.time()

        # Step 1: Drop to ~15fps by keeping every FRAME_STRIDE'th frame.
        #
        # SHuBERT expects roughly 15fps. Upstream tried to do this with an ffmpeg
        # `-filter:v fps=15` pass that was commented out and could never have worked
        # anyway (it read video_path and wrote back to video_path), leaving the read
        # loop taking every frame while its "Step 1: Change the fps to 15" comment
        # claimed otherwise. A 30fps camera therefore fed the model twice the frames
        # it wants, paying for it twice over: perception is per-frame, and the extra
        # frames also lengthen the ByT5 encoder input.
        #
        # A/B'd over my_please.mp4 plus 5 dailymoth_examples clips (189-364 frames,
        # both signers), 3 strides each: stride 2 runs at 53% of full-rate latency
        # with no quality cost (2 clips better, 3 draws, 1 marginally worse — on the
        # one clip with checkable ground truth it dropped a hallucinated place name
        # the full-rate run invented). Stride 3 is 36% but degrades: misspelled proper
        # nouns ("Liberia" -> "Libera"), dropped clauses, "collapsed" -> "torn down".
        stride = max(1, int(os.environ.get("FRAME_STRIDE", "2")))
        cap = cv2.VideoCapture(video_path)
        frames = []
        read_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if read_count % stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            read_count += 1
        cap.release()
        signer_video = np.array(frames)
        print(f"[features] stride {stride}: {len(frames)}/{read_count} frames kept")
        video_read_seconds = time.time() - stage_start

        return self.process_frames(signer_video, landmarks=None,
                                   video_read_seconds=video_read_seconds)

    def process_frames(self, signer_video, landmarks=None, video_read_seconds=0.0,
                       mediapipe_seconds=None, embeddings=None, embed_seconds=None):
        """Everything after the frames exist: landmarks, crops, DINOv2, pose, ByT5.

        Split out of process_video() so the live worker can hand over frames whose
        landmarks were already computed *during* capture (see streaming_perception.py).
        Pass `landmarks` to skip the MediaPipe stage entirely; `mediapipe_seconds` is
        then only used to report what that overlapped work cost.
        """
        timings = {'video_read': video_read_seconds}
        if not isinstance(signer_video, np.ndarray):
            signer_video = np.array(signer_video)
        stage_start = time.time()

        # Step 2: Extract pose using kpe_mediapipe (unless the caller already did it).
        if landmarks is None:
            landmarks = video_holistic(
                video_input=signer_video,
                face_model_path=self.config['mediapipe_face_model_path'],
                hand_model_path=self.config['mediapipe_hands_model_path'],
            )
            timings['mediapipe_landmarks'] = time.time() - stage_start
        else:
            # Already done, overlapped with capture. Report it as ~0 on the critical
            # path and show the absorbed cost separately, so the breakdown stays honest.
            timings['mediapipe_landmarks'] = time.time() - stage_start
            if mediapipe_seconds is not None:
                timings['mediapipe_overlapped'] = mediapipe_seconds
        stage_start = time.time()

        # Step 3: Extract stream features (unless the caller already embedded them during
        # capture — see streaming_perception.py's second stage).
        if embeddings is not None:
            left_hand_embeddings, right_hand_embeddings, face_embeddings = embeddings
            del signer_video
            timings['hand_crop'] = 0.0
            timings['dinov2_hands'] = 0.0
            timings['face_crop'] = 0.0
            timings['dinov2_face'] = 0.0
            if embed_seconds is not None:
                timings['dinov2_overlapped'] = embed_seconds
        else:
            hand_extractor = HandExtractor()
            left_hand_frames, right_hand_frames = hand_extractor.extract_hand_frames(signer_video, landmarks)
            timings['hand_crop'] = time.time() - stage_start
            stage_start = time.time()

            left_hand_embeddings = extract_embeddings_from_frames(left_hand_frames, self.config['dino_hands_model_path'], dtype=HANDS_DTYPE)
            right_hand_embeddings = extract_embeddings_from_frames(right_hand_frames, self.config['dino_hands_model_path'], dtype=HANDS_DTYPE)
            del left_hand_frames, right_hand_frames
            timings['dinov2_hands'] = time.time() - stage_start
            stage_start = time.time()

            face_extractor = FaceExtractor()
            face_frames = face_extractor.extract_face_frames(signer_video, landmarks)
            timings['face_crop'] = time.time() - stage_start
            stage_start = time.time()

            face_embeddings = extract_embeddings_from_frames(face_frames, self.config['dino_face_model_path'], dtype=FACE_DTYPE)
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
            note = "  (overlapped with capture, not on the critical path)" \
                if stage_name.endswith('_overlapped') else ""
            print(f"  {stage_name:20s}: {elapsed:6.1f}s{note}")
        # mediapipe_overlapped is reported but excluded: it was already paid during
        # recording, so counting it here would misstate the post-cut latency.
        critical = sum(v for k, v in timings.items() if not k.endswith('_overlapped'))
        print(f"  {'total':20s}: {critical:6.1f}s")

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