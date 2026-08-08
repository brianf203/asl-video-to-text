import time
import os
from features import SHuBERTProcessor

MODELS_BASE = "/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models"

config = {
    'yolov8_model_path': os.path.join(MODELS_BASE, 'yolov8n.pt'),
    'dino_face_model_path': os.path.join(MODELS_BASE, 'dinov2face.pth'),
    'dino_hands_model_path': os.path.join(MODELS_BASE, 'dinov2hand.pth'),
    'mediapipe_face_model_path': os.path.join(MODELS_BASE, 'face_landmarker_v2_with_blendshapes.task'),
    'mediapipe_hands_model_path': os.path.join(MODELS_BASE, 'hand_landmarker.task'),
    'shubert_model_path': os.path.join(MODELS_BASE, 'checkpoint_836_400000.pt'),
    'slt_model_config': os.path.join(MODELS_BASE, 'byt5_base', 'config.json'),
    'slt_model_checkpoint': os.path.join(MODELS_BASE, 'checkpoint-11625'),
    'slt_tokenizer_checkpoint': os.path.join(MODELS_BASE, 'byt5_base'),
    'temp_dir': 'temp',
}

os.makedirs(config['temp_dir'], exist_ok=True)

import sys
input_clip = sys.argv[1] if len(sys.argv) > 1 else "dailymoth_examples/L5hUxT5YbnY_crop_1.mp4"

processor = SHuBERTProcessor(config)
start_time = time.time()
output_text = processor.process_video(input_clip)
elapsed = time.time() - start_time
print(f"\nThe English translation is: {output_text}")
print(f"Total time: {elapsed:.1f} seconds")
