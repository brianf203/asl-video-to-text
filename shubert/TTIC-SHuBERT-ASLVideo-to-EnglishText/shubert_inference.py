import torch
import numpy as np
import csv
import os
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from examples.shubert.models.shubert import SHubertModel, SHubertConfig
from transformers import ByT5Tokenizer, ByT5ForConditionalGeneration


class SHubertProcessor:
    """
    A class for processing multi-modal embeddings through SHubert model.
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize the SHubertProcessor.
        
        Args:
            checkpoint_path: Path to the SHubert model checkpoint
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        self.model = self._load_model()
        
        print(f"SHubertProcessor initialized on device: {self.device}")
    
    def _load_model(self) -> SHubertModel:
        """Load the SHubert model from checkpoint."""
        # Initialize configuration
        cfg = SHubertConfig()
        
        # Initialize the model
        model = SHubertModel(cfg)
        
        # Load the checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load the state dictionary into the model
        model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        model.to(self.device)
        return model
    
    def process_embeddings(self, face_embeddings: np.ndarray, 
                          left_hand_embeddings: np.ndarray,
                          right_hand_embeddings: np.ndarray, 
                          pose_embeddings: np.ndarray) -> np.ndarray:
        """
        Process multi-modal embeddings through SHubert model.
        
        Args:
            face_embeddings: Face embeddings array of shape (num_frames, embedding_dim)
            left_hand_embeddings: Left hand embeddings array of shape (num_frames, embedding_dim)
            right_hand_embeddings: Right hand embeddings array of shape (num_frames, embedding_dim)
            pose_embeddings: Pose embeddings array of shape (num_frames, pose_dim)
            
        Returns:
            Numpy array of SHubert features with shape (num_layers, num_frames, feature_dim)
        """
        # Convert to tensors and move to device
        face = torch.from_numpy(face_embeddings).float().to(self.device)
        left_hand = torch.from_numpy(left_hand_embeddings).float().to(self.device)
        right_hand = torch.from_numpy(right_hand_embeddings).float().to(self.device)
        body_posture = torch.from_numpy(pose_embeddings).float().to(self.device)
        
        length = face.shape[0]
        
        # Prepare input in the format expected by SHubert
        source = [{
            "face": face,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "body_posture": body_posture,
            # Add dummy labels to match the expected input format
            "label_face": torch.zeros((length, 1)).to(self.device),
            "label_left_hand": torch.zeros((length, 1)).to(self.device),
            "label_right_hand": torch.zeros((length, 1)).to(self.device),
            "label_body_posture": torch.zeros((length, 1)).to(self.device)
        }]
        
        # Extract features
        with torch.no_grad():
            result = self.model.extract_features(source, padding_mask=None, kmeans_labels=None, mask=False)
        
        # Extract layer outputs
        layer_outputs = []
        for layer in result['layer_results']:
            # layer_output has shape [T, B, D]
            # Since batch size B is 1, we can squeeze it
            layer_output = layer[-1]
            layer_output = layer_output.squeeze(1)  # Shape: [T, D]
            layer_outputs.append(layer_output.cpu().numpy())  # Convert to NumPy array
        
        # Stack the outputs from all layers to get an array of shape [L, T, D]
        features = np.stack(layer_outputs, axis=0)  # Shape: [L, T, D]
        return features
    
    def process_embeddings_from_files(self, face_path: str, left_hand_path: str, 
                                     right_hand_path: str, pose_path: str) -> np.ndarray:
        """
        Process embeddings loaded from files.
        
        Args:
            face_path: Path to face embeddings .npy file
            left_hand_path: Path to left hand embeddings .npy file
            right_hand_path: Path to right hand embeddings .npy file
            pose_path: Path to pose embeddings .npy file
            
        Returns:
            Numpy array of SHubert features with shape (num_layers, num_frames, feature_dim)
        """
        # Load numpy arrays
        face_embeddings = np.load(face_path)
        left_hand_embeddings = np.load(left_hand_path)
        right_hand_embeddings = np.load(right_hand_path)
        pose_embeddings = np.load(pose_path)
        
        return self.process_embeddings(face_embeddings, left_hand_embeddings, 
                                     right_hand_embeddings, pose_embeddings)
    
    def process_and_save_embeddings(self, face_embeddings: np.ndarray, 
                                   left_hand_embeddings: np.ndarray,
                                   right_hand_embeddings: np.ndarray, 
                                   pose_embeddings: np.ndarray,
                                   output_path: str) -> str:
        """
        Process embeddings and save to file.
        
        Args:
            face_embeddings: Face embeddings array
            left_hand_embeddings: Left hand embeddings array
            right_hand_embeddings: Right hand embeddings array
            pose_embeddings: Pose embeddings array
            output_path: Path to save the output file
            
        Returns:
            Path to the saved file
        """
        # Process embeddings
        features = self.process_embeddings(face_embeddings, left_hand_embeddings, 
                                         right_hand_embeddings, pose_embeddings)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        np.save(output_path, features)
        
        return str(output_path)
    
    def process_from_files_and_save(self, face_path: str, left_hand_path: str, 
                                   right_hand_path: str, pose_path: str, 
                                   output_path: str) -> str:
        """
        Process embeddings from files and save results.
        
        Args:
            face_path: Path to face embeddings .npy file
            left_hand_path: Path to left hand embeddings .npy file
            right_hand_path: Path to right hand embeddings .npy file
            pose_path: Path to pose embeddings .npy file
            output_path: Path to save the output file
            
        Returns:
            Path to the saved file
        """
        # Process embeddings
        features = self.process_embeddings_from_files(face_path, left_hand_path, 
                                                     right_hand_path, pose_path)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        np.save(output_path, features)
        
        return str(output_path)


class SHuBERTTextGenerator:
    """
    A class that combines SHuBERT feature extraction with BYT5 text generation.
    """
    
    def __init__(self, shubert_checkpoint: str, byt5_model_name: str = "google/byt5-base",
                 device: Optional[str] = None):
        """
        Initialize with SHuBERT and BYT5 models.
        
        Args:
            shubert_checkpoint: Path to SHuBERT model checkpoint
            byt5_model_name: Name of BYT5 model (default: "google/byt5-base")
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SHuBERT processor
        self.shubert_processor = SHubertProcessor(shubert_checkpoint, self.device)
        
        # Initialize BYT5 model
        self.tokenizer = ByT5Tokenizer.from_pretrained(byt5_model_name)
        self.model = ByT5ForConditionalGeneration.from_pretrained(byt5_model_name).to(self.device)
    
    def generate_text(self, face_embeddings: np.ndarray, 
                     left_hand_embeddings: np.ndarray,
                     right_hand_embeddings: np.ndarray, 
                     pose_embeddings: np.ndarray,
                     max_length: int = 1024, 
                     num_beams: int = 5) -> str:
        """
        Generate text from multi-modal embeddings.
        
        Args:
            face_embeddings: Face embeddings array
            left_hand_embeddings: Left hand embeddings array
            right_hand_embeddings: Right hand embeddings array
            pose_embeddings: Pose embeddings array
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            
        Returns:
            Generated text string
        """
        # Get SHuBERT features
        features = self.shubert_processor.process_embeddings(
            face_embeddings, left_hand_embeddings, right_hand_embeddings, pose_embeddings)
        
        # Select features from specific layer (default: last layer)
        features = features[-1]  # Shape: [T, D]
        
        # Convert to tensor and add batch dimension
        features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Generate text
        generated_ids = self.model.generate(
            inputs_embeds=features,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        # Decode generated tokens to text
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def generate_text_from_features(face_embeddings: np.ndarray, 
                              left_hand_embeddings: np.ndarray,
                              right_hand_embeddings: np.ndarray, 
                              pose_embeddings: np.ndarray,
                              shubert_checkpoint: str,
                              byt5_model_name: str = "google/byt5-base",
                              max_length: int = 1024,
                              num_beams: int = 5) -> str:
    """
    Convenience function to generate text from features.
    """
    generator = SHuBERTTextGenerator(shubert_checkpoint, byt5_model_name)
    return generator.generate_text(
        face_embeddings, left_hand_embeddings, right_hand_embeddings, pose_embeddings,
        max_length=max_length, num_beams=num_beams
    )


# Convenience functions for backward compatibility
def process_shubert_embeddings(face_embeddings: np.ndarray, 
                              left_hand_embeddings: np.ndarray,
                              right_hand_embeddings: np.ndarray, 
                              pose_embeddings: np.ndarray,
                              checkpoint_path: str) -> np.ndarray:
    """
    Convenience function to process embeddings through SHubert.
    
    Args:
        face_embeddings: Face embeddings array
        left_hand_embeddings: Left hand embeddings array
        right_hand_embeddings: Right hand embeddings array
        pose_embeddings: Pose embeddings array
        checkpoint_path: Path to the SHubert model checkpoint
        
    Returns:
        Numpy array of SHubert features
    """
    processor = SHubertProcessor(checkpoint_path)
    return processor.process_embeddings(face_embeddings, left_hand_embeddings, 
                                      right_hand_embeddings, pose_embeddings)


def process_sample(model: SHubertModel, face_path: str, left_hand_path: str, 
                  right_hand_path: str, body_posture_path: str) -> np.ndarray:
    """
    Original function for backward compatibility with command-line usage.
    """
    # Load numpy arrays
    face_np = np.load(face_path)
    left_hand_np = np.load(left_hand_path)
    right_hand_np = np.load(right_hand_path)
    body_posture_np = np.load(body_posture_path)
    
    face = torch.from_numpy(face_np).float().cuda()
    left_hand = torch.from_numpy(left_hand_np).float().cuda()
    right_hand = torch.from_numpy(right_hand_np).float().cuda()
    body_posture = torch.from_numpy(body_posture_np).float().cuda()
    
    length = face.shape[0]
    
    # Prepare input
    source = [{
        "face": face,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "body_posture": body_posture,
        # Add dummy labels to match the expected input format
        "label_face": torch.zeros((length, 1)).cuda(),
        "label_left_hand": torch.zeros((length, 1)).cuda(),
        "label_right_hand": torch.zeros((length, 1)).cuda(),
        "label_body_posture": torch.zeros((length, 1)).cuda()
    }]
    
    # Extract features
    with torch.no_grad():
        result = model.extract_features(source, padding_mask=None, kmeans_labels=None, mask=False)
    
    # Extract layer outputs
    layer_outputs = []
    for layer in result['layer_results']:
        # layer_output has shape [T, B, D]
        # Since batch size B is 1, we can squeeze it
        layer_output = layer[-1]
        layer_output = layer_output.squeeze(1)  # Shape: [T, D]
        layer_outputs.append(layer_output.cpu().numpy())  # Convert to NumPy array
    
    # Stack the outputs from all layers to get an array of shape [L, T, D]
    features = np.stack(layer_outputs, axis=0)  # Shape: [L, T, D]
    return features


def load_model(checkpoint_path: str) -> SHubertModel:
    """
    Original function for backward compatibility with command-line usage.
    """
    cfg = SHubertConfig()
    
    # Initialize the model
    model = SHubertModel(cfg)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # If the checkpoint is saved with a 'model' key
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.cuda()  # Move to GPU if available
    return model


def main(csv_list: List[List[str]], checkpoint_path: str, output_dir: str, index: int):
    """
    Original main function for backward compatibility with command-line usage.
    """
    model = load_model(checkpoint_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for row in csv_list:
        cues_list = row[0].split('\t')
        face_path, left_hand_path, right_hand_path, body_posture_path = cues_list[0], cues_list[1], cues_list[2], cues_list[3]
        
        output_filename = f"{os.path.basename(face_path).rsplit('.', 1)[0].rsplit('_', 1)[0]}.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        # check if the output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_path} as it already exists")
            continue
        
        # Process the sample
        features = process_sample(model, face_path, left_hand_path, right_hand_path, body_posture_path)
        
        np.save(output_path, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='path to the CSV file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='directory to save output files')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size for processing')
    
    args = parser.parse_args()
    index = args.index
    csv_path = args.csv_path
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    batch_size = int(args.batch_size)
    
    # make output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV data
    fixed_list = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fixed_list.append(row)
    
    # Process in batches
    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    
    csv_list = video_batches[index]
    main(csv_list, checkpoint_path, output_dir, index)