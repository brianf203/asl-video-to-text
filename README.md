# ASL Video-to-Text Pipeline

This pipeline uses **SHuBERT**, a pretrained ASL foundation model (TTIC, ACL 2025),
to translate ASL video directly into English text.

Reference: http://shubert.pals.ttic.edu

## Overview of the pipeline

```
Video file (.mp4)
    -> MediaPipe extracts hand/face/pose landmarks
    -> Hand and face regions are cropped from each frame
    -> DINOv2 extracts visual features from hand/face crops (GPU)
    -> Pose landmarks are processed into pose features
    -> SHuBERT encoder + ByT5 decoder translate all features into English text (CPU)
```

## 1. Clone this repository

```bash
git clone https://github.com/brianf203/asl-video-to-text.git
cd asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText
```

## 2. Set up a dedicated virtual environment

```bash
python3 -m venv shubert_venv
source shubert_venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs torch, transformers, mediapipe, fairseq, gradio, and related
packages. This will take a while, and `fairseq` will build from source.

## 4. Install the Jetson-specific PyTorch build (for GPU acceleration)

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Verify GPU is available:
```bash
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```
This should print `CUDA: True`.

## 5. Download the SHuBERT model

You'll need a free Hugging Face account and access token (Read access):
```bash
pip install -U huggingface_hub
```
```bash
hf auth login
```
Paste your token (from https://huggingface.co/settings/tokens) when prompted.

Then download the model files:
```bash
python3 -c "
import huggingface_hub
path = huggingface_hub.snapshot_download(repo_id='ShesterG/SHuBERT', allow_patterns='models/*')
print(path)
"
```
Note the printed path, it's needed in the next two steps.

## 6. Convert the ByT5 checkpoint to bf16 (required)

The pipeline loads the ByT5 decoder from `checkpoint-11625-bf16`, which is **not** part of
the Hugging Face download — it is generated locally from the checkpoint you just fetched:

```bash
python3 benchmarks/convert_bf16.py /path/from/step/5
```

Pass the same path Step 5 printed. This reads `checkpoint-11625` and writes
`checkpoint-11625-bf16` beside it (2.68GB -> 1.34GB), and takes a couple of minutes.

**Skipping this step makes Step 7 fail** with a missing-checkpoint error from
`from_pretrained`.

Why it exists: `from_pretrained` materialises the whole fp32 file before casting it to
bf16, leaving ~3.58GB resident for what becomes a 1.34GB model. On a Jetson's unified
memory that repeatedly ran out of memory at `.to("cuda")`. Loading the bf16 copy is about
1.6x faster (22-24s -> 13-15s) and leaves ~850MB more headroom.

The result is **bitwise identical** to what the fp32 checkpoint becomes at load, since the
cast happens either way — not an approximation, and nothing to re-validate. To confirm:

```bash
python3 benchmarks/load_peak.py /path/from/step/5/checkpoint-11625      fp32
python3 benchmarks/load_peak.py /path/from/step/5/checkpoint-11625-bf16 bf16
```

Both print the same `sha256`, computed over every loaded parameter.

To use the fp32 checkpoint instead, without editing any code:

```bash
export BYT5_CKPT=/path/from/step/5/checkpoint-11625
```

## 7. Configure and run

Edit `run_shubert.py` and set `MODELS_BASE` to the path printed in Step 5, e.g.:
```python
MODELS_BASE = "/home/YOUR_USERNAME/.cache/huggingface/hub/models--ShesterG--SHuBERT/snapshots/<hash>/models"
```

Run on a bundled example video:
```bash
python3 run_shubert.py
```

Run on a specific video file:
```bash
python3 run_shubert.py path/to/your_video.mp4
```

## 8. Recording and testing your own videos

Use `record_clip.py` to record a clip from your camera:
```bash
python3 record_clip.py my_sign.mp4
```
- Press `r` to start/stop recording
- Press `q` to quit

Then run it through SHuBERT:
```bash
python3 run_shubert.py my_sign.mp4
```

**Tips for good results:**
- The signer should be the main part of the frame (around 90% of the area)
- Keep clips under 20 seconds
