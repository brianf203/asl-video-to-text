import spaces
import gradio as gr
import os
import tempfile
import huggingface_hub
import shutil
import logging
import traceback
from features import SHuBERTProcessor


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set writable cache directories
def setup_cache_directories():
    """Set up cache directories with proper error handling"""
    try:
        cache_dirs = {
            'MPLCONFIGDIR': '/tmp/matplotlib',
            'TRANSFORMERS_CACHE': '/tmp/huggingface',
            'HF_HOME': '/tmp/huggingface',
            'FONTCONFIG_PATH': '/tmp/fontconfig',
            'TORCH_HOME': '/tmp/torch',  # PyTorch cache directory
        }
        
        for env_var, path in cache_dirs.items():
            os.environ[env_var] = path
            os.makedirs(path, exist_ok=True, mode=0o777)
            logger.info(f"Cache directory created: {env_var} = {path}")
        
        # Also set XDG_CACHE_HOME to override default .cache location
        os.environ['XDG_CACHE_HOME'] = '/tmp/cache'
        os.makedirs('/tmp/cache', exist_ok=True, mode=0o777)
        logger.info(f"Cache directory created: XDG_CACHE_HOME = /tmp/cache")
        
        # Clear any existing PyTorch Hub cache to avoid corruption issues
        torch_hub_dir = '/tmp/torch/hub'
        if os.path.exists(torch_hub_dir):
            shutil.rmtree(torch_hub_dir)
            logger.info("Cleared existing PyTorch Hub cache")
        os.makedirs(torch_hub_dir, exist_ok=True, mode=0o777)
        logger.info(f"Created clean PyTorch Hub cache directory: {torch_hub_dir}")
        
        # Copy updated DINOv2 files to torch cache after clearing
        # This ensures they're available when PyTorch Hub downloads the repo
        try:
            src_dir = os.path.dirname(os.path.abspath(__file__))
            target_dir = '/tmp/torch/hub/facebookresearch_dinov2_main/dinov2/layers'
            
            for filename in ['attention.py', 'block.py']:
                src_path = os.path.join(src_dir, filename)
                if os.path.exists(src_path):
                    # We'll copy these after the initial hub download
                    logger.info(f"Found {filename} in project directory - will copy after hub download")
                else:
                    logger.warning(f"Could not find {filename} in project directory")
        except Exception as e:
            logger.warning(f"Error preparing DINOv2 files: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating cache directories: {str(e)}")
        return False

# Configuration for Hugging Face Spaces
MODEL_REPO = "ShesterG/SHuBERT"
TOKEN = os.environ.get('HF_TOKEN')

def validate_environment():
    """Validate required environment variables and setup"""
    if not TOKEN:
        raise ValueError("HF_TOKEN environment variable not set. This is required to access private model repository.")
    
    # Check available disk space
    free_space = shutil.disk_usage('/').free / (1024*1024*1024)  # GB
    logger.info(f"Available disk space: {free_space:.2f} GB")
    
    if free_space < 2:  # Less than 2GB
        logger.warning("Low disk space available. This may cause issues.")
    
    return True

def download_models():
    """Download all required models from Hugging Face Hub with enhanced error handling"""
    logger.info("Starting model download process...")
    
    try:
        # Validate environment first
        validate_environment()
        


        logger.info("Downloading entire models folder...")

        # Download the entire models folder
        models_path = huggingface_hub.snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns="models/*",  # Download everything in models folder
            token=TOKEN,
            cache_dir=os.environ['TRANSFORMERS_CACHE']
        )
        

        # Build config dict with expected file paths
        config = {
            'yolov8_model_path': os.path.join(models_path, "models/yolov8n.pt"),
            'dino_face_model_path': os.path.join(models_path, "models/dinov2face.pth"),
            'dino_hands_model_path': os.path.join(models_path, "models/dinov2hand.pth"),
            'mediapipe_face_model_path': os.path.join(models_path, "models/face_landmarker_v2_with_blendshapes.task"),
            'mediapipe_hands_model_path': os.path.join(models_path, "models/hand_landmarker.task"),
            'shubert_model_path': os.path.join(models_path, "models/checkpoint_836_400000.pt"),
            'slt_model_config': os.path.join(models_path, "models/byt5_base/config.json"),
            'slt_model_checkpoint': os.path.join(models_path, "models/checkpoint-11625"),
            'slt_tokenizer_checkpoint': os.path.join(models_path, "models/byt5_base"),
            'temp_dir': 'temp'
        }

        # Verify all required files and folders exist
        logger.info("Verifying downloaded files...")
        missing_files = []

        for key, path in config.items():
            if key == 'temp_dir':  # Skip temp_dir check
                continue
            
            if not os.path.exists(path):
                missing_files.append(f"{key}: {path}")
                logger.error(f"Missing: {path}")
            else:
                logger.info(f"✓ Found: {path}")

        if missing_files:
            logger.error(f"Missing {len(missing_files)} required files/folders:")
            for missing in missing_files:
                logger.error(f"  - {missing}")
            raise FileNotFoundError(f"Required files not found: {missing_files}")

        logger.info("All models downloaded and verified successfully!")
        logger.info(f"Models root path: {models_path}")

        return config
        
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Additional debugging info
        try:
            cache_contents = os.listdir(os.environ['TRANSFORMERS_CACHE'])
            logger.info(f"Cache directory contents: {cache_contents}")
        except:
            logger.error("Cannot access cache directory")
            
        return None



def download_example_videos():
    """Download example videos from Hugging Face Hub with enhanced error handling"""

    # Download the entire example_videos folder
    example_video_path = huggingface_hub.snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns="example_video/*",  # Download everything in example_videos folder
        token=TOKEN,
        cache_dir=os.environ['TRANSFORMERS_CACHE']
    )

    example_video_path_list = [
        os.path.join(example_video_path, "rDUefZVPfmU_crop_1.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_2.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_3.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_4.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_5.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_6.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_7.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_8.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_9.mp4"),
        os.path.join(example_video_path, "rDUefZVPfmU_crop_10.mp4"),
        os.path.join(example_video_path, "L5hUxT5YbnY_crop_1.mp4"),
        os.path.join(example_video_path, "L5hUxT5YbnY_crop_2.mp4"),
        os.path.join(example_video_path, "L5hUxT5YbnY_crop_3.mp4"),
        os.path.join(example_video_path, "L5hUxT5YbnY_crop_4.mp4"),
        os.path.join(example_video_path, "L5hUxT5YbnY_crop_5.mp4"),
        os.path.join(example_video_path, "L5hUxT5YbnY_crop_6.mp4"),
        os.path.join(example_video_path, "L5hUxT5YbnY_crop_7.mp4"),
    ]

    return example_video_path_list
        


def initialize_processor(config):
    """Initialize SHuBERT processor with error handling"""
    try:
        logger.info("Initializing SHuBERT processor...")
        processor = SHuBERTProcessor(config)
        logger.info("SHuBERT processor initialized successfully!")
        return processor
    except Exception as e:
        logger.error(f"Error initializing SHuBERT processor: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

# Initialize the application
def initialize_app():
    """Initialize the entire application with comprehensive error handling"""
    try:
        # Setup cache directories
        if not setup_cache_directories():
            raise RuntimeError("Failed to setup cache directories")
        
        # Download models
        config = download_models()
        if config is None:
            raise RuntimeError("Failed to download models")
        
        # Initialize processor
        processor = initialize_processor(config)
        if processor is None:
            raise RuntimeError("Failed to initialize SHuBERT processor")
        
        logger.info("Application initialized successfully!")
        return config, processor
        
    except Exception as e:
        error_msg = f"Application initialization failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise RuntimeError(error_msg)

# Global variables for application state
config = None
processor = None
initialization_error = None

try:
    config, processor = initialize_app()
except Exception as e:
    initialization_error = str(e)
    logger.error(f"Startup failed: {initialization_error}")

def copy_dinov2_files_if_needed():
    """Copy updated DINOv2 files after PyTorch Hub download if needed"""
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = '/tmp/torch/hub/facebookresearch_dinov2_main/dinov2/layers'
        
        # Check if PyTorch Hub has downloaded the repository
        hub_main_dir = '/tmp/torch/hub/facebookresearch_dinov2_main'
        
        if os.path.exists(hub_main_dir):
            # Ensure the target directory exists
            os.makedirs(target_dir, exist_ok=True)
            
            files_copied = 0
            for filename in ['attention.py', 'block.py']:
                src_path = os.path.join(src_dir, filename)
                target_path = os.path.join(target_dir, filename)
                
                if os.path.exists(src_path):
                    # Always overwrite with our robust versions
                    shutil.copy2(src_path, target_path)
                    # Make sure it's readable
                    os.chmod(target_path, 0o644)
                    logger.info(f"Replaced {filename} with robust version (numpy/Python 3.8 compatible)")
                    files_copied += 1
                else:
                    logger.error(f"Source file not found: {src_path}")
            
            if files_copied > 0:
                # Clear Python's import cache to ensure new files are used
                import importlib
                import sys
                
                # Remove any cached imports of dinov2 modules
                modules_to_remove = [key for key in sys.modules.keys() if 'dinov2' in key]
                for module in modules_to_remove:
                    del sys.modules[module]
                    logger.info(f"Cleared cached import: {module}")
                
                logger.info(f"Successfully replaced {files_copied} DINOv2 files with robust versions")
                return True
        else:
            logger.info("PyTorch Hub repository not yet downloaded")
            return False
            
    except Exception as e:
        logger.error(f"Error copying DINOv2 files: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
@spaces.GPU
def process_video(video_file):
    """Process uploaded video file with enhanced error handling"""
    # Check if initialization was successful
    if initialization_error:
        return f"Application initialization failed: {initialization_error}\n\nPlease check the logs for more details."
    
    if processor is None:
        return "Error: Model not initialized properly. Please check the logs."
    
    if video_file is None:
        return "Please upload a video file."
    
    logger.info(f"=== Starting video processing ===")
    logger.info(f"Video file input: {video_file}")
    logger.info(f"Video file type: {type(video_file)}")
    
    try:
        # Create temp directory with proper permissions
        temp_dir = config['temp_dir']
        os.makedirs(temp_dir, exist_ok=True, mode=0o777)
        logger.info(f"Temp directory: {temp_dir}")
        
        # Generate unique filename to avoid conflicts
        import time
        timestamp = str(int(time.time() * 1000))
        file_extension = '.mp4'  # Default extension
        
        # Try to get original extension if available
        try:
            if hasattr(video_file, 'name') and video_file.name:
                file_extension = os.path.splitext(video_file.name)[1] or '.mp4'
            elif isinstance(video_file, str):
                file_extension = os.path.splitext(video_file)[1] or '.mp4'
        except:
            pass
            
        temp_video_path = os.path.join(temp_dir, f"video_{timestamp}{file_extension}")
        logger.info(f"Target temp video path: {temp_video_path}")
        
        # Handle Gradio file upload - video_file is typically a string path to temp file
        logger.info(f"Processing video file: {video_file} (type: {type(video_file)})")
        
        if isinstance(video_file, str):
            # Gradio provides a file path string
            source_path = video_file
            
            # Handle both absolute and relative paths
            if not os.path.isabs(source_path):
                # Try current working directory first
                abs_source_path = os.path.abspath(source_path)
                logger.info(f"Converting relative path {source_path} to absolute: {abs_source_path}")
                if os.path.exists(abs_source_path):
                    source_path = abs_source_path
                else:
                    # Try looking in common Gradio temp directories
                    possible_paths = [
                        source_path,
                        os.path.join('/tmp', os.path.basename(source_path)),
                        os.path.join('/tmp/gradio', os.path.basename(source_path)),
                        abs_source_path
                    ]
                    
                    found_path = None
                    for path in possible_paths:
                        logger.info(f"Checking path: {path}")
                        if os.path.exists(path):
                            found_path = path
                            logger.info(f"Found file at: {path}")
                            break
                    
                    if found_path:
                        source_path = found_path
                    else:
                        logger.error(f"Could not find source file in any expected location")
                        logger.error(f"Tried paths: {possible_paths}")
                        raise FileNotFoundError(f"Source video file not found in any expected location: {video_file}")
            
            logger.info(f"Final source file path: {source_path}")
            logger.info(f"Source file exists: {os.path.exists(source_path)}")
            
            if os.path.exists(source_path):
                try:
                    # Check source file permissions and size
                    stat_info = os.stat(source_path)
                    logger.info(f"Source file size: {stat_info.st_size} bytes, mode: {oct(stat_info.st_mode)}")
                    
                    # Try to read the file content
                    with open(source_path, 'rb') as src:
                        content = src.read()
                        logger.info(f"Successfully read {len(content)} bytes from source")
                    
                    # Write to destination (with a different name to avoid conflicts)
                    final_temp_path = os.path.join(temp_dir, f"processed_{timestamp}{file_extension}")
                    with open(final_temp_path, 'wb') as dst:
                        dst.write(content)
                        logger.info(f"Successfully wrote to destination: {final_temp_path}")
                    
                    # Update temp_video_path to the final location
                    temp_video_path = final_temp_path
                        
                except PermissionError as e:
                    logger.error(f"Permission error reading source file: {e}")
                    # Try alternative approach - use a completely different temp location
                    try:
                        import tempfile
                        # Create a new temporary file in system temp directory
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                            alternative_temp_path = tmp.name
                        
                        logger.info(f"Trying alternative temp path: {alternative_temp_path}")
                        
                        # Try to copy using system copy command as fallback
                        import subprocess
                        result = subprocess.run(['cp', source_path, alternative_temp_path], 
                                              capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            logger.info("Successfully copied using system cp command")
                            temp_video_path = alternative_temp_path
                        else:
                            logger.error(f"System cp failed: {result.stderr}")
                            raise PermissionError(f"Cannot read video file due to permission restrictions: {e}")
                            
                    except Exception as e2:
                        logger.error(f"Alternative copy method also failed: {e2}")
                        raise PermissionError(f"Cannot read video file due to permission restrictions: {e}")
            else:
                raise FileNotFoundError(f"Source video file not found: {source_path}")
                
        elif hasattr(video_file, 'read'):
            # If it's a file-like object with read method
            try:
                content = video_file.read()
                with open(temp_video_path, 'wb') as f:
                    f.write(content)
                logger.info(f"Saved video from file object: {temp_video_path} ({len(content)} bytes)")
            except Exception as e:
                logger.error(f"Error reading from file object: {e}")
                raise ValueError(f"Cannot read from file object: {e}")
        else:
            # Handle other cases - try to extract file path or content
            logger.info(f"Attempting to handle unknown file type: {type(video_file)}")
            try:
                # Check if it has a name attribute (common for file objects)
                if hasattr(video_file, 'name'):
                    source_path = video_file.name
                    logger.info(f"Found name attribute: {source_path}")
                    
                    if os.path.exists(source_path):
                        with open(source_path, 'rb') as src:
                            content = src.read()
                        with open(temp_video_path, 'wb') as dst:
                            dst.write(content)
                        logger.info(f"Successfully copied from name attribute")
                    else:
                        raise FileNotFoundError(f"File from name attribute not found: {source_path}")
                else:
                    logger.error(f"Unsupported video file type: {type(video_file)}")
                    raise ValueError(f"Unsupported video file type: {type(video_file)}")
            except Exception as e:
                logger.error(f"Failed to handle unknown file type: {e}")
                raise ValueError(f"Cannot process video file: {e}")
        
        # Set proper permissions on the saved file
        os.chmod(temp_video_path, 0o666)
        
        # Verify file exists and has content
        if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
            raise ValueError("Video file is empty or could not be saved")
        
        # Copy DINOv2 files if needed before processing
        # This needs to happen right after PyTorch Hub downloads but before model loading
        logger.info("Ensuring DINOv2 files are ready for processing...")
        copy_dinov2_files_if_needed()
        
        # Set up a monitoring patch for torch.hub.load to replace files immediately after download
        original_torch_hub_load = None
        try:
            import torch.hub
            original_torch_hub_load = torch.hub.load
            
            def patched_torch_hub_load(*args, **kwargs):
                logger.info(f"PyTorch Hub load called with: {args[0] if args else 'unknown'}")
                
                # Call the original function first
                result = original_torch_hub_load(*args, **kwargs)
                
                # If this was a DINOv2 call, immediately replace the files
                if args and 'dinov2' in str(args[0]):
                    logger.info("DINOv2 downloaded! Immediately replacing with robust versions...")
                    
                    # Try multiple times to ensure files are replaced
                    import time
                    for attempt in range(5):
                        if copy_dinov2_files_if_needed():
                            logger.info("Successfully replaced DINOv2 files!")
                            break
                        else:
                            logger.info(f"Attempt {attempt + 1} failed, retrying in 1 second...")
                            time.sleep(1)
                
                return result
            
            # Temporarily patch torch.hub.load
            torch.hub.load = patched_torch_hub_load
            logger.info("Patched torch.hub.load to replace DINOv2 files after download")
        except Exception as e:
            logger.warning(f"Could not patch torch.hub.load: {e}")
        
        logger.info(f"Processing video: {temp_video_path}")
        try:
            output_text = processor.process_video(temp_video_path)
        finally:
            # Restore original function
            if original_torch_hub_load:
                try:
                    import torch.hub
                    torch.hub.load = original_torch_hub_load
                    logger.info("Restored original torch.hub.load")
                except:
                    pass
        
        logger.info(f"Video processed successfully. Output: {output_text[:100]}...")
        
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info("Temporary video file cleaned up")
        
        return output_text
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error processing video: {str(e)}\n\nPlease check that your video is a valid ASL video under 10 seconds."

# # Create Gradio interface
# def create_interface():
#    """Create the Gradio interface"""
#    description = """
#    Upload an ASL* video to get an English translation.  *Sign languages belonging to the same sign language family as ASL (e.g. Ghanaian Sign Language, as well as others listed in Table 7, Row 1 of https://aclanthology.org/2023.findings-emnlp.664.pdf) might also have non-trivial performance, although the model is trained only on ASL data.


# This app uses TTIC's foundation model SHuBERT (introduced in an ACL 2025 paper, see http://shubert.pals.ttic.edu).
     
#    **Requirements:**
#    - We recommend that videos be under 20 seconds.  Performance for longer videos has not been tested.
#    - The signer should be the main part (e.g. 90% of the area) of the video. Videos recorded from a phone camera, tablet, or personal computer should work well. Studio recordings where the signer is farther from the camera may not work as well.
#    - Supported formats: MP4, MOV

#    **Note:**
#    - This is just a demo of a research project, and should NOT be used to replace an interpreter in any way.
#    - Videos will be deleted after the output is generated.
#    - Inquires or feedback? Please email us at shesterg@ttic.edu
#    """
  
#    if initialization_error:
#        description += f"\n\n:warning: **Initialization Error:** {initialization_error}"
  
    
# #    dailymoth_pathlist = download_example_videos()

#    src_dir = os.path.dirname(os.path.abspath(__file__))
#    dailymoth_pathlist = []
#    L5hUxT5YbnY_crop_1 = "dailymoth_examples/L5hUxT5YbnY_crop_1.mp4"
#    L5hUxT5YbnY_crop_2 = "dailymoth_examples/L5hUxT5YbnY_crop_2.mp4"
#    L5hUxT5YbnY_crop_3 = "dailymoth_examples/L5hUxT5YbnY_crop_3.mp4"
#    L5hUxT5YbnY_crop_4 = "dailymoth_examples/L5hUxT5YbnY_crop_4.mp4"
#    L5hUxT5YbnY_crop_5 = "dailymoth_examples/L5hUxT5YbnY_crop_5.mp4"
#    L5hUxT5YbnY_crop_6 = "dailymoth_examples/L5hUxT5YbnY_crop_6.mp4"
#    L5hUxT5YbnY_crop_7 = "dailymoth_examples/L5hUxT5YbnY_crop_7.mp4"
#    rDUefZVPfmU_crop_1 = "dailymoth_examples/rDUefZVPfmU_crop_1.mp4"
#    rDUefZVPfmU_crop_2 = "dailymoth_examples/rDUefZVPfmU_crop_2.mp4"
#    rDUefZVPfmU_crop_3 = "dailymoth_examples/rDUefZVPfmU_crop_3.mp4"
#    rDUefZVPfmU_crop_4 = "dailymoth_examples/rDUefZVPfmU_crop_4.mp4"
#    rDUefZVPfmU_crop_5 = "dailymoth_examples/rDUefZVPfmU_crop_5.mp4"
#    rDUefZVPfmU_crop_6 = "dailymoth_examples/rDUefZVPfmU_crop_6.mp4"
#    rDUefZVPfmU_crop_7 = "dailymoth_examples/rDUefZVPfmU_crop_7.mp4"
#    rDUefZVPfmU_crop_8 = "dailymoth_examples/rDUefZVPfmU_crop_8.mp4"
#    rDUefZVPfmU_crop_9 = "dailymoth_examples/rDUefZVPfmU_crop_9.mp4"
#    rDUefZVPfmU_crop_10 = "dailymoth_examples/rDUefZVPfmU_crop_10.mp4"
#    dailymoth_filenames = [L5hUxT5YbnY_crop_1, L5hUxT5YbnY_crop_2, L5hUxT5YbnY_crop_3, L5hUxT5YbnY_crop_4, L5hUxT5YbnY_crop_5, L5hUxT5YbnY_crop_6, L5hUxT5YbnY_crop_7, rDUefZVPfmU_crop_1, rDUefZVPfmU_crop_2, rDUefZVPfmU_crop_3, rDUefZVPfmU_crop_4, rDUefZVPfmU_crop_5, rDUefZVPfmU_crop_6, rDUefZVPfmU_crop_7, rDUefZVPfmU_crop_8, rDUefZVPfmU_crop_9, rDUefZVPfmU_crop_10]
   
#    for filename in dailymoth_filenames:
#        src_path = os.path.join(src_dir, filename)
#        if os.path.exists(src_path):
#            dailymoth_pathlist.append(src_path)
#        else:
#            print(f"Warning: File not found at {src_path}")
       
#    with gr.Blocks(title="ASL Video to English Text Translation") as interface:
#        gr.Markdown(f"# ASL Video to English Text Translation\n\n{description}")
    
#    with gr.Row():
#        with gr.Column():
#            video_input = gr.Video(label="ASL Video (under 20 seconds)", format="mp4", height=480, width=640)
#            submit_btn = gr.Button("Translate", variant="primary")
#        with gr.Column():
#            output_text = gr.Textbox(label="English Translation", lines=3)
#            # Add examples section
#            if dailymoth_pathlist:  # Only show examples if we have valid files
#                gr.Examples(
#                    examples=dailymoth_pathlist,
#                    inputs=video_input,
#                    label="Click a video to try an example"
#                )
                
#                # Add attribution note for the examples
#                gr.Markdown("""
#                ---
#                **Example Videos Attribution:**
#                The example videos used in this demo are from [The Daily Moth](https://www.youtube.com/@TheDailyMoth), 
#                a popular ASL news channel made by deaf creators. Specifically, they are from the Previews of [July 10](https://www.youtube.com/watch?v=rDUefZVPfmU) and [July 11](https://www.youtube.com/watch?v=L5hUxT5YbnY) 2025 Top Stories.
#                The videos are only used for illustrative purposes, and all rights to the content belong to The Daily Moth. In this light, we encourage to subscribe to their [channel](https://members.dailymoth.com/about).
#                """)
#            else:
#                gr.Markdown("*No example videos available at this time.*")

# #    video_input = gr.Video(label="ASL Video (under 20 seconds)", format="mp4", height=480, width=640)
# #    text_output = gr.Textbox(label="English Translation", lines=5)
  
  
  
# #    interface = gr.Interface(
# #        fn=process_video,
# #        inputs=video_input,
# #        outputs=text_output,
# #        title="ASL Video to English Text Translation",
# #        description=description,
# #        article="",
# #     #    examples=dailymoth_pathlist,  
# #     #    example_labels=["Officials with an EU force said they are searching for the missing."], 
# #        allow_flagging="never",    
# #    )
   
# #    gr.Examples(
# #         examples=dailymoth_pathlist,
# #         inputs=video_input,
# #         label="Click a video to try an example"
# #     )
#    submit_btn.click(fn=process_video, inputs=video_input, outputs=output_text)
   
#    return interface


def create_interface():
    """Create the Gradio interface"""
    description = """Upload an ASL* video to get an English translation.  *Sign languages belonging to the same sign language family as ASL (e.g. Ghanaian Sign Language, as well as others listed in Table 7, Row 1 of https://aclanthology.org/2023.findings-emnlp.664.pdf) might also have non-trivial performance, although the model is trained only on ASL data.

This app uses TTIC's foundation model SHuBERT (introduced in an ACL 2025 paper, see http://shubert.pals.ttic.edu).

**Requirements:**
- We recommend that videos be under 20 seconds.  Performance for longer videos has not been tested.
- The signer should be the main part (e.g. 90% of the area) of the video. Videos recorded from a phone camera, tablet, or personal computer should work well. Studio recordings where the signer is farther from the camera may not work as well.
- Supported formats: MP4, MOV

**Note:**
- This is just a demo of a research project, and should NOT be used to replace an interpreter in any way.
- Videos will be deleted after the output is generated.
- Inquires or feedback? Please email us at shesterg@ttic.edu"""
   
    if initialization_error:
        description += f"\n\n:warning: **Initialization Error:** {initialization_error}"
   
    src_dir = os.path.dirname(os.path.abspath(__file__))
    dailymoth_pathlist = []
    L5hUxT5YbnY_crop_1 = "dailymoth_examples/L5hUxT5YbnY_crop_1.mp4"
    L5hUxT5YbnY_crop_2 = "dailymoth_examples/L5hUxT5YbnY_crop_2.mp4"
    L5hUxT5YbnY_crop_3 = "dailymoth_examples/L5hUxT5YbnY_crop_3.mp4"
    L5hUxT5YbnY_crop_4 = "dailymoth_examples/L5hUxT5YbnY_crop_4.mp4"
    L5hUxT5YbnY_crop_5 = "dailymoth_examples/L5hUxT5YbnY_crop_5.mp4"
    L5hUxT5YbnY_crop_6 = "dailymoth_examples/L5hUxT5YbnY_crop_6.mp4"
    L5hUxT5YbnY_crop_7 = "dailymoth_examples/L5hUxT5YbnY_crop_7.mp4"
    rDUefZVPfmU_crop_1 = "dailymoth_examples/rDUefZVPfmU_crop_1.mp4"
    rDUefZVPfmU_crop_2 = "dailymoth_examples/rDUefZVPfmU_crop_2.mp4"
    rDUefZVPfmU_crop_3 = "dailymoth_examples/rDUefZVPfmU_crop_3.mp4"
    rDUefZVPfmU_crop_4 = "dailymoth_examples/rDUefZVPfmU_crop_4.mp4"
    rDUefZVPfmU_crop_5 = "dailymoth_examples/rDUefZVPfmU_crop_5.mp4"
    rDUefZVPfmU_crop_6 = "dailymoth_examples/rDUefZVPfmU_crop_6.mp4"
    rDUefZVPfmU_crop_7 = "dailymoth_examples/rDUefZVPfmU_crop_7.mp4"
    rDUefZVPfmU_crop_8 = "dailymoth_examples/rDUefZVPfmU_crop_8.mp4"
    rDUefZVPfmU_crop_9 = "dailymoth_examples/rDUefZVPfmU_crop_9.mp4"
    rDUefZVPfmU_crop_10 = "dailymoth_examples/rDUefZVPfmU_crop_10.mp4"
    dailymoth_filenames = [L5hUxT5YbnY_crop_1, L5hUxT5YbnY_crop_2, L5hUxT5YbnY_crop_3, L5hUxT5YbnY_crop_4, L5hUxT5YbnY_crop_5, L5hUxT5YbnY_crop_6, L5hUxT5YbnY_crop_7, rDUefZVPfmU_crop_1, rDUefZVPfmU_crop_2, rDUefZVPfmU_crop_3, rDUefZVPfmU_crop_4, rDUefZVPfmU_crop_5, rDUefZVPfmU_crop_6, rDUefZVPfmU_crop_7, rDUefZVPfmU_crop_8, rDUefZVPfmU_crop_9, rDUefZVPfmU_crop_10]
    
    for filename in dailymoth_filenames:
        src_path = os.path.join(src_dir, filename)
        if os.path.exists(src_path):
            dailymoth_pathlist.append(src_path)
        else:
            print(f"Warning: File not found at {src_path}")
        
    with gr.Blocks(title="ASL Video to English Text Translation") as interface:
        gr.Markdown(f"# ASL Video to English Text Translation\n\n{description}")
     
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="ASL Video (under 20 seconds)", format="mp4", height=480, width=640)
                submit_btn = gr.Button("Translate", variant="primary")
            with gr.Column():
                output_text = gr.Textbox(label="English Translation", lines=3)
                
                # Add examples section in the right column
                if dailymoth_pathlist:  # Only show examples if we have valid files
                    gr.Examples(
                        examples=dailymoth_pathlist,
                        inputs=video_input,
                        label="Click a video to try an example"
                    )
                     
                    # Add attribution note for the examples
                    gr.Markdown("""
                    ---
                    **Example Videos Attribution:**
                    The example videos used in this demo are from [The Daily Moth](https://www.youtube.com/@TheDailyMoth), 
                    a popular ASL news channel made by deaf creators. Specifically, they are from the Previews of [July 10](https://www.youtube.com/watch?v=rDUefZVPfmU) and [July 11](https://www.youtube.com/watch?v=L5hUxT5YbnY) 2025 Top Stories.
                    The videos are only used for illustrative purposes, and all rights to the content belong to The Daily Moth. In this light, we encourage to subscribe to their [channel](https://members.dailymoth.com/about).
                    """)
                else:
                    gr.Markdown("*No example videos available at this time.*")
                
        # Set up the button click handler AFTER both input and output are defined
        submit_btn.click(fn=process_video, inputs=video_input, outputs=output_text)

    return interface


# Create the demo
demo = create_interface()

if __name__ == "__main__":
    # Launch with better configuration for Hugging Face Spaces
    logger.info("Launching Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )