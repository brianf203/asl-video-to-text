# ASL Video-to-Text Pipeline

This pipeline uses **SHuBERT**, a pretrained ASL foundation model (TTIC, ACL 2025),
to translate ASL video directly into English text. It runs live on an NVIDIA Jetson
Orin Nano (8GB shared memory, JetPack 6.2, ARM64).

Reference: http://shubert.pals.ttic.edu

The translation model is not ours — it is used through its published checkpoints.
What is here is the system around it: live capture, segmentation, and the memory and
scheduling work needed to run it on an 8GB embedded board.

## Overview of the pipeline

```
Video frames (live camera, or an .mp4)
    -> MediaPipe extracts hand/face/pose landmarks
    -> Hand and face regions are cropped from each frame
    -> DINOv2 extracts visual features from hand/face crops (GPU, fp16)
    -> Pose landmarks are processed into pose features
    -> SHuBERT encoder + ByT5 decoder translate all features into English (GPU, bf16)
```

On the live path, landmark extraction and DINOv2 run *during* recording, so what you
wait after a clip ends is only the remainder of the backlog plus translation.

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

## 7. Point the code at your model directory

`MODELS_BASE` is a hardcoded path literal, and it appears in **six** files. Set all of
them to the path printed in Step 5, e.g.:

```python
MODELS_BASE = "/home/YOUR_USERNAME/.cache/huggingface/hub/models--ShesterG--SHuBERT/snapshots/<hash>/models"
```

| file | why you need it |
|---|---|
| `auto_segment_v5.py` | the live camera path — **the primary one** |
| `run_shubert.py` | one-shot translation of a video file |
| `run_eval.py` | the benchmark harness |
| `analyze_hand_trigger.py` | hand-trigger analysis |
| `benchmarks/hand_detect_bound.py` | hand-detector benchmark |
| `benchmarks/byt5_split.py` | ByT5 stage benchmark |

The first two are enough to use the pipeline; the rest only matter if you run the
benchmarks. To find them all:

```bash
grep -rn '^MODELS_BASE\|^MODELS_BASE = (' --include='*.py' .
```

## 8. Set the CUDA memory flag

PyTorch on this Jetson can hit `NVML_SUCCESS == r INTERNAL ASSERT FAILED`. Export this
before running anything in `shubert_venv`:

```bash
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

`auto_segment_v5.py` sets it for itself if it is unset, but the other entry points do not.

## 9. Run it

### Live camera (the primary path)

```bash
python3 auto_segment_v5.py
```

The default is **push-to-record**: a camera window opens, **SPACE** starts a clip and
SPACE ends it. Dead air at both ends of the clip is trimmed automatically. Translations
print to **this terminal**, one `Signer: ...` line per clip — the camera window carries
only short status strings. Press `q` to quit (it drains the queue first, which can take a
couple of minutes; Ctrl-C abandons deliberately).

Models load once at startup (~20s). After that, expect roughly 11s after the cut for a
~6s clip and ~22s for a ~14s one. Clips are force-cut at 15s.

Every translation is also appended to `transcripts/session_YYYYMMDD-HHMMSS.txt` as it is
printed, fsync'd per clip, so a crash or power loss does not take the session's output
with it.

For demos, the same session with everything except the sentences filtered out:

```bash
python3 demo_transcript.py
```

The full log is still written to `demo_full_log.txt`, so a failure is diagnosable
afterwards.

### One-shot on a video file

```bash
BYT5_DTYPE=float32 python3 run_shubert.py path/to/your_video.mp4
```

fp32 is faster here because there is no repeated load to amortise. With no argument it
runs a bundled example clip.

### Recording your own clips

```bash
python3 record_clip.py my_sign.mp4
```
- Press `r` to start/stop recording
- Press `q` to quit

**Tips for good results:**
- The signer should be the main part of the frame (around 90% of the area)
- Keep clips under 15 seconds — that is where the live path cuts them

## Configuration

All of these are environment variables; the defaults are what ships and what has been
validated.

| variable | default | what it does |
|---|---|---|
| `RECORD_MODE` | `manual` | `motion` restores automatic segmentation: the app calibrates at startup and starts clips on motion or hand presence |
| `MANUAL_TRIM` | `1` | trim dead air from both ends of a push-to-record clip; `0` keeps every frame between the presses |
| `MANUAL_TRIM_MAX_SECONDS` | `2.5` | upper bound on what either end can lose |
| `MAX_CLIP_SECONDS` | `15` | force-cut length for a single clip |
| `FRAME_STRIDE` | `2` | keep 1 frame in N. 2 is the validated setting; 1 is full 30fps and no better, 3 degrades accuracy |
| `TRANSCRIPT` | `1` | `0` disables the session transcript. `TRANSCRIPT_DIR` / `TRANSCRIPT_PATH` relocate it |
| `BYT5_DEVICE` | `cuda` | `cpu` reverts the decoder to CPU |
| `BYT5_DTYPE` | `bfloat16` | decoder precision (T5 overflows in fp16) |
| `BYT5_CKPT` | — | override the checkpoint path for every entry point at once |
| `DINOV2_DTYPE` | `float16` | DINOv2 compute precision; measured quality-neutral. Ignored on CPU |
| `DINOV2_BATCH_SIZE` | `32` | lower it under memory pressure |
| `WINDOW_SCALE` | `1.6` | preview window size. Display only — capture stays 640x480 |

Watch the output for `[features] stride 2: N/M frames kept`, `[byt5] device: … dtype: …`
and `DINOEmbedder initialized on device: … dtype: …` to confirm these took effect.

## Notes on this hardware

- Jetson power mode should be maxed: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
- Camera is a Logitech ConferenceCam, typically `/dev/video0`, best in MJPG at 640x480.
- GPU memory is 8GB shared with the whole OS and is genuinely tight. Close memory-heavy
  desktop apps (a browser at 1GB+ RSS is enough) before running, or the first DINOv2 load
  can OOM.
