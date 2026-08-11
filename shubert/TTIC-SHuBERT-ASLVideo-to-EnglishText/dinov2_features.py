import torch
import torch.nn as nn

from gpu_serial import gpu_serial
from torchvision import transforms
from PIL import Image
try:
    import decord
except ImportError:
    decord = None
    VideoReader = None
import numpy as np
import os
import pickle
import gzip
from pathlib import Path
import argparse
import json
import csv
import glob
import time 
from typing import List, Union, Optional, Tuple


class DINOEmbedder:
    """
    A class for extracting DINOv2 embeddings from video frames or images.
    """
    
    def __init__(self, dino_model_path: str, batch_size: int = 128, device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize the DINOEmbedder.

        Args:
            dino_model_path: Path to the fine-tuned DINOv2 model
            batch_size: Batch size for processing frames
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None
            dtype: Compute dtype. Defaults to DEFAULT_DTYPE (fp16 on CUDA). Forced to
                fp32 on CPU, where half precision has no fast kernels.
        """
        self.dino_model_path = dino_model_path
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # fp16 is a CUDA-only win here: the Orin has half-precision tensor cores, while
        # on CPU half falls back to slow emulated kernels. Unlike ByT5 (which overflows
        # in fp16 and needs bf16), a ViT encoder is numerically well behaved in fp16.
        dtype = DEFAULT_DTYPE if dtype is None else dtype
        if "cuda" not in str(self.device):
            dtype = torch.float32
        self.dtype = dtype

        # Initialize model
        self.model = self._load_dino_model()
        self.model.eval()
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"DINOEmbedder initialized on device: {self.device} dtype: {self.dtype}")
    
    def _load_dino_model(self) -> nn.Module:
        """Load the fine-tuned DINOv2 model."""
        # Load the original DINOv2 model with the correct architecture
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)
        
        # Load fine-tuned weights
        pretrained = torch.load(self.dino_model_path, map_location=self.device)
        
        # Make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key:
                continue  # Skip dino_head layers
            else:
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value
        
        # Change shape of pos_embed
        pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
        model.pos_embed = pos_embed
        
        # Load state dict
        model.load_state_dict(new_state_dict, strict=True)
        
        # Move model to device
        model.to(device=self.device, dtype=self.dtype)
        return model
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame."""
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
        
        tensor = self.transform(image)
        # Ensure only RGB channels are considered
        return tensor[:3]
    
    def _preprocess_frames_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of frames."""
        batch_tensors = torch.stack([self._preprocess_frame(frame) for frame in frames])
        return batch_tensors.to(device=self.device, dtype=self.dtype)
    
    def extract_embeddings_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract DINOv2 embeddings from a list of frames.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            Numpy array of embeddings with shape (num_frames, embedding_dim)
        """
        all_embeddings = []

        # Process frames in batches
        for idx in range(0, len(frames), self.batch_size):
            batch_frames = frames[idx:idx + self.batch_size]

            # Preprocess batch. Deliberately OUTSIDE the GPU lock below -- this is CPU
            # work, so leaving it unserialised lets it overlap with another thread's
            # device work.
            batch_tensors = self._preprocess_frames_batch(batch_frames)

            # Extract embeddings. Serialised process-wide: on the live path several clips
            # can be embedding at once while the worker runs ByT5, and nothing else bounds
            # the peak on this 8GB shared-memory board. See gpu_serial.py.
            with gpu_serial(), torch.no_grad():
                # Cast back to fp32 so callers keep receiving float32 arrays regardless
                # of the compute dtype -- downstream (SHuBERT/ByT5) sets its own dtype.
                batch_embeddings = self.model(batch_tensors).float().cpu().numpy()
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings
    
    def extract_embeddings_from_video(self, video_input: Union[str, VideoReader], 
                                     target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Extract DINOv2 embeddings from a video.
        
        Args:
            video_input: Either a path to video file (str) or a VideoReader object
            target_size: Target size for video frames (width, height)
            
        Returns:
            Numpy array of embeddings with shape (num_frames, embedding_dim)
        """
        # Handle different input types
        if isinstance(video_input, str):
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            try:
                vr = VideoReader(str(video_path), width=target_size[0], height=target_size[1])
            except Exception as e:
                raise RuntimeError(f"Error loading video {video_input}: {e}")
        # elif hasattr(video_input, 'get_batch'):
        else:
            vr = video_input
        # else:
        #     raise TypeError("video_input must be either a file path (str) or a VideoReader object")
        
        total_frames = len(vr)
        all_embeddings = []
        
        # Process video in batches
        for idx in range(0, total_frames, self.batch_size):
            batch_indices = range(idx, min(idx + self.batch_size, total_frames))
            # batch_frames = vr.get_batch(batch_indices).asnumpy()
            batch_frames = vr[batch_indices]
            
            # Preprocess batch
            batch_tensors = self._preprocess_frames_batch(batch_frames)
            
            # Extract embeddings
            with torch.no_grad():
                # Cast back to fp32 so callers keep receiving float32 arrays regardless
                # of the compute dtype -- downstream (SHuBERT/ByT5) sets its own dtype.
                batch_embeddings = self.model(batch_tensors).float().cpu().numpy()
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings
    
    def extract_embeddings_from_video_and_save(self, video_path: str, output_folder: str) -> str:
        """
        Extract embeddings from video and save to file.
        
        Args:
            video_path: Path to the video file
            output_folder: Folder to save the embeddings
            
        Returns:
            Path to the saved embeddings file
        """
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Extract embeddings
        embeddings = self.extract_embeddings_from_video(video_path)
        
        # Save embeddings
        video_name = Path(video_path).stem
        np_path = Path(output_folder) / f"{video_name}.npy"
        np.save(np_path, embeddings)
        
        return str(np_path)
    
    def extract_embedding_from_single_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract DINOv2 embedding from a single image.
        
        Args:
            image: Image as numpy array or PIL Image
            
        Returns:
            Numpy array of embedding with shape (1, embedding_dim)
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        tensor = self.transform(image).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        # Extract embedding
        with torch.no_grad():
            embedding = self.model(tensor).float().cpu().numpy()

        return embedding


# DINOv2 peak GPU memory is dominated by per-batch activations, not weights (the
# ViT-S/14 backbone is only ~22M params). On the Jetson's 8GB *shared* CPU/GPU
# pool a batch of 128 frames is enough to OOM once anything else is resident
# (e.g. the cached ByT5 model in host RAM), so the live pipeline uses a much
# smaller batch. Throughput impact is minimal — this GPU does not saturate at 128.
# Override with DINOV2_BATCH_SIZE=<n> when tuning against memory pressure.
DEFAULT_BATCH_SIZE = int(os.environ.get("DINOV2_BATCH_SIZE", "32"))

# DINOv2 was the single largest stage in the pipeline (~41% of clip time) and was
# running in fp32 on a GPU with half-precision tensor cores. Override with
# DINOV2_DTYPE=float32 to A/B. Ignored on CPU (see DINOEmbedder.__init__).
DEFAULT_DTYPE = getattr(torch, os.environ.get("DINOV2_DTYPE", "float16"))

# The hand and face streams are separate embedders, so their precision can differ.
# Measured on 50 OpenASL clips: fp16 everywhere is 1.28x but costs ~1 BLEU, and the
# damage showed up as a dropped fingerspelled name -- i.e. hand detail. These let us
# keep hands at full precision while the face stream (coarser: expression and mouth
# shape, not fingertip position) runs fp16. Both fall back to DINOV2_DTYPE.
HANDS_DTYPE = getattr(torch, os.environ.get("DINOV2_HANDS_DTYPE",
                                            os.environ.get("DINOV2_DTYPE", "float16")))
FACE_DTYPE = getattr(torch, os.environ.get("DINOV2_FACE_DTYPE",
                                           os.environ.get("DINOV2_DTYPE", "float16")))

# Process-wide cache so repeated calls with the same model path/batch_size/device
# reuse the already-loaded model instead of reloading weights from disk each time.
_embedder_cache = {}


def _get_cached_embedder(dino_model_path: str, batch_size: int = DEFAULT_BATCH_SIZE,
                          device: Optional[str] = None,
                          dtype: Optional[torch.dtype] = None) -> "DINOEmbedder":
    # dtype is part of the key so an A/B within one process gets two distinct models
    # rather than silently reusing the first one's precision.
    key = (dino_model_path, batch_size, str(device) if device else None,
           str(dtype) if dtype else None)
    embedder = _embedder_cache.get(key)
    if embedder is None:
        embedder = DINOEmbedder(dino_model_path, batch_size, device, dtype)
        _embedder_cache[key] = embedder
    return embedder


def preload_embedder(dino_model_path: str, batch_size: int = DEFAULT_BATCH_SIZE,
                     dtype: Optional[torch.dtype] = None) -> None:
    """Load and cache an embedder ahead of first use, so the first clip does not
    pay the model-load cost. Useful for long-running processes (live capture).

    Pass the same dtype the caller will use at inference time, or warmup populates a
    cache entry under a different key and the first clip pays the load anyway."""
    _get_cached_embedder(dino_model_path, batch_size, dtype=dtype)


# Convenience functions for backward compatibility
def extract_embeddings_from_frames(frames: List[np.ndarray], dino_model_path: str,
                                  batch_size: int = DEFAULT_BATCH_SIZE,
                                  dtype: Optional[torch.dtype] = None) -> np.ndarray:
    """
    Convenience function to extract embeddings from frames.

    Args:
        frames: List of frames as numpy arrays
        dino_model_path: Path to the fine-tuned DINOv2 model
        batch_size: Batch size for processing
        dtype: Compute dtype; defaults to DEFAULT_DTYPE. Callers pass HANDS_DTYPE /
            FACE_DTYPE to set the two streams independently.

    Returns:
        Numpy array of embeddings
    """
    embedder = _get_cached_embedder(dino_model_path, batch_size, dtype=dtype)
    return embedder.extract_embeddings_from_frames(frames)


def extract_embeddings_from_video(video_path: str, dino_model_path: str,
                                 batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """
    Convenience function to extract embeddings from video.

    Args:
        video_path: Path to the video file
        dino_model_path: Path to the fine-tuned DINOv2 model
        batch_size: Batch size for processing

    Returns:
        Numpy array of embeddings
    """
    embedder = _get_cached_embedder(dino_model_path, batch_size)
    return embedder.extract_embeddings_from_video(video_path)


def video_to_embeddings(video_path: str, output_folder: str, dino_path: str, batch_size: int = 128):
    """
    Original function for backward compatibility with command-line usage.
    """
    try:
        embedder = _get_cached_embedder(dino_path, batch_size)
        embedder.extract_embeddings_from_video_and_save(video_path, output_folder)
    except Exception as e:
        print(f'Error processing {video_path}: {e}')


# Utility functions for batch processing
def get_mp4_files(directory: str) -> List[str]:
    """Get all MP4 files in a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
    
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))
    return [os.path.abspath(file) for file in mp4_files]


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
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit in seconds')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='number of videos to process in this batch')
    parser.add_argument('--files_list', type=str, required=True,
                        help='path to the files list file')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='path to the output folder')
    parser.add_argument('--dino_path', type=str, required=True,
                        help='path to the dino model')
    
    args = parser.parse_args()
    start_time = time.time()
    
    # Load files list
    fixed_list = load_file(args.files_list)
    
    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Initialize embedder
    embedder = DINOEmbedder(args.dino_path, batch_size=512)
    
    # Process videos in batches
    video_batches = [fixed_list[i:i + args.batch_size] for i in range(0, len(fixed_list), args.batch_size)]
    print(f"Total number of video batches: {len(video_batches)}")
    
    for video_path in video_batches[args.index]:
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            print("Time limit reached. Stopping execution.")
            break
        
        video_name = Path(video_path).stem
        np_path = Path(args.output_folder) / f"{video_name}.npy"
        
        if np_path.exists():
            print(f"Skipping {video_path} - output already exists")
            continue
        else:
            try:
                print(f"Processing {video_path}")
                embedder.extract_embeddings_from_video_and_save(video_path, args.output_folder)
                print(f"Successfully processed {video_path}")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()