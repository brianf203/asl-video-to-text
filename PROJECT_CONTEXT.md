# ASL Video-to-Text Project — Context for Claude Code

## Goal
Build a system that translates ASL signing into English text on an NVIDIA
Jetson Orin Nano (8GB shared memory, JetPack 6.2, ARM64/aarch64), ideally with
translations appearing a few seconds after the signer finishes a sentence
(not truly word-by-word live, but not requiring a full stop-and-wait either).

## Repository structure
```
~/asl-video-to-text/
├── legacy_asl_citizen_pipeline/   # Earlier approach, kept for reference, NOT the active path
│   ├── capture_live_onnx.py       # MediaPipe replaced with ONNX/TensorRT (palm/hand/person/pose detection)
│   ├── predict.py                 # ST-GCN classifier trained on ASL Citizen (isolated signs only)
│   └── venv/                      # Its own separate virtual environment
├── shubert/                        # ACTIVE / PRIMARY approach
│   └── TTIC-SHuBERT-ASLVideo-to-EnglishText/
│       ├── shubert_venv/          # Separate virtual environment for this pipeline
│       ├── features.py            # Main orchestrator: SHuBERTProcessor class
│       ├── run_shubert.py         # CLI entry point: `python3 run_shubert.py <video.mp4>`
│       ├── record_clip.py         # Records a manual video clip from the camera
│       ├── auto_segment_v3.py     # Auto-detects sign start/stop via frame differencing,
│       │                          # launches run_shubert.py as a subprocess per clip
│       ├── inference.py           # ByT5 translation logic (currently forced to CPU, see below)
│       ├── dinov2_features.py     # DINOv2 feature extraction (currently reloads model every call — known inefficiency)
│       └── kpe_mediapipe.py       # MediaPipe landmark extraction
└── online_cslr/                    # Abandoned research direction, see "Paths we tried and rejected" below
```

## What's currently working (as of today)
- **Legacy pipeline (ONNX/TensorRT)**: live camera → MediaPipe-equivalent ONNX models
  (palm/hand/person/pose detection, verified to match real MediaPipe output within 1-2px)
  → ST-GCN classifier. Runs ~12-17fps. Accuracy is inconsistent/mediocre — this is why
  we moved to SHuBERT.
- **SHuBERT pipeline (primary)**: video file in → MediaPipe landmarks → DINOv2 visual
  features (hands + face) → SHuBERT encoder → ByT5 decoder → English text out.
  Confirmed working and accurate on both bundled example clips and our own recorded
  footage (e.g., correctly translated "my name is [name]" from a self-recorded clip).
  **Problem: takes ~100+ seconds per clip**, far too slow for anything resembling
  real-time.

## Known environment gotchas (already fixed in the current code, documented in case
they resurface or need reapplying elsewhere)
- `decord` (video reading library) has no ARM64/Jetson wheels. Already replaced with
  `cv2.VideoCapture`-based reading in `features.py` and patched around in other files
  that imported it unnecessarily.
- Jetson-specific PyTorch must be installed from
  `https://pypi.jetson-ai-lab.io/jp6/cu126`, not plain `pip install torch`.
- PyTorch on this Jetson can hit `NVML_SUCCESS == r INTERNAL ASSERT FAILED` — mitigated
  with `export PYTORCH_NO_CUDA_MEMORY_CACHING=1`. This env var should be set before
  running anything in the shubert_venv.
- GPU memory is genuinely tight (8GB shared with the whole OS). Running DINOv2 (3x per
  clip: left hand, right hand, face) AND the full ByT5 model on GPU simultaneously
  reliably causes `CUDA error: out of memory`. Current working config: DINOv2 on GPU
  (fast, ~20-25s total), ByT5 forced to CPU in `inference.py` (`device = torch.device("cpu")`)
  — this is the main reason overall latency is still ~100+ seconds.
- **FIXED (2026-08-07)**: `dinov2_features.py`'s `extract_embeddings_from_frames()` used
  to construct a brand new `DINOEmbedder` (reloading model weights from scratch) on
  every single call (3x per clip). Now cached module-wide by
  `(model_path, batch_size, device)` — see "Suggested order of work" below for the
  measured impact (small, ~4%, since MediaPipe turned out to be the real bottleneck).
- GPU memory is tight enough that having a browser open (e.g. Firefox using ~1GB+ RSS)
  can push the very first, small DINOv2 model load into `CUDA error: out of memory` on
  this 8GB shared-memory board. Close memory-heavy desktop apps before running the
  pipeline if you hit OOM on model load.

## The actual next task (please start here)
We want to make the SHuBERT pipeline meaningfully faster and closer to real-time,
based on a real research paper we found (arXiv 2607.09611, "Toward Real-Time
Sentence-Level Sign Language Translation") — NOTE: **that paper's linked GitHub repo
does not exist (confirmed 404), so treat it as a design reference only, not a
dependency to clone.** Their reported approach (not verified/reproduced by us):

1. **Freeze SHuBERT entirely, fine-tune only a small adapter on top using QLoRA**
   (4-bit quantization + low-rank adapters) targeting ByT5's query/value projections
   plus the SHuBERT→ByT5 projection layer. Trained on a small subset of How2Sign
   (~10k clip-sentence pairs is what they used). This is gloss-free — uses How2Sign's
   English sentence translations only, NOT gloss annotations (which we separately
   confirmed are NOT practically available for How2Sign — see "Paths we tried and
   rejected" below).
2. **Run MediaPipe's face/hand/pose landmark extraction concurrently** (e.g. via a
   thread pool) instead of sequentially — they reported this alone cut perception time
   from ~110ms/frame to ~45ms/frame. This is testable on our existing code right now,
   independent of any retraining.
3. **A sentence-boundary detection state machine** (WAIT → RECORDING → HOLD → FINAL)
   very similar to what `auto_segment_v3.py` already does — worth comparing our
   thresholds against theirs (they used: min utterance 600ms, hands-absent 400ms,
   hands-lowered 500ms, hands-idle 900ms).

### Suggested order of work
1. **DONE (2026-08-07)**: fixed `dinov2_features.py` to cache loaded `DINOEmbedder`
   instances by `(model_path, batch_size, device)` instead of reloading per call.
   Measured on `my_please.mp4`: 113.2s → 108.4s wall time (~5s / ~4% faster). Smaller
   win than hoped since DINOv2-vits14-reg is a small model — kept anyway since it's
   free and correct, but it wasn't the real bottleneck (see profiling below).
2. **Profiling added (2026-08-07)**: `features.py`'s `process_video()` now times each
   stage and prints a breakdown (see bottom of `run_shubert.py` output). On
   `my_please.mp4` (93.3s total post-fix-#1):
   - MediaPipe landmark extraction: 55.2s (**59% of total** — the actual bottleneck)
   - ByT5 inference (CPU): 20.9s (22%)
   - DINOv2 hands: 10.3s (11%)
   - DINOv2 face: 6.5s (7%)
   - video read / hand+face crop / pose feature processing: ~0.4s combined (negligible)

   **This overturned the original assumption that ByT5-on-CPU was the main bottleneck.**
   MediaPipe is nearly 3x costlier than ByT5 inference. Re-prioritize accordingly:
3. **DONE (2026-08-07)**: parallelized MediaPipe extraction in `kpe_mediapipe.py` —
   `HolisticDetector` now runs pose (`mp_holistic.process`), face, and hand detection
   concurrently per frame via a persistent `ThreadPoolExecutor(max_workers=3)`
   (MediaPipe's C++ inference releases the GIL, so this is real parallelism, not just
   interleaving). Added `HolisticDetector.close()` to shut the pool down cleanly;
   `video_holistic()` and the CLI `main()` call it via try/finally. Also added
   per-detector (pose/face/hand) timing to `HolisticDetector` (`_frame_timings` +
   `print_timing_summary()`) to profile the critical path inside the concurrent call.
   Measured on `my_please.mp4`: MediaPipe stage 55.2s → ~29.3s (1.9x), total pipeline
   93.3s → ~68s (~26% faster), translation output unchanged/correct.
4. **DONE (2026-08-07)**: re-profiled after parallelizing and found hand detection was
   ~98% of the remaining per-frame critical path (255.9ms/frame of a 259.9ms/frame
   wall time — pose at 141.2ms and face at 49.5ms were now fully hidden behind it).
   The hand detector was configured with `max_hands=6` (a multi-person-scene default),
   even though clips only ever have one signer's two hands. Changed the
   `HolisticDetector` default to `max_hands=2` in `kpe_mediapipe.py`. Measured: hand
   detector 255.9ms → 154.6ms/frame, MediaPipe stage 29.1s → ~17.5s, **total pipeline
   69.1s → ~56s**. Pose (134.6ms/frame) and hand (154.6ms/frame) are now roughly
   balanced as MediaPipe's internal co-bottlenecks — further within-MediaPipe wins
   would need optimizing both, with diminishing returns; not pursued further for now.

   **Running total so far**: 113.2s (original) → 108.4s (DINOv2 cache) → ~68s
   (MediaPipe parallelized) → ~56s (max_hands=2) — roughly **2x faster overall**.
5. **SCOPED, NOT STARTED (2026-08-07)** — QLoRA fine-tuning. See **`QLORA_SCOPING.md`**
   in the repo root for the full analysis. Headline findings that changed the plan:
   - Split the ByT5 stage and found **13.2s of the ~20.6s is just loading the 2.68 GB
     checkpoint from disk; actual `generate()` compute is only 7.1s (13% of total)**.
     QLoRA targets that 7.1s, so its latency ceiling is ~12.6% even if generation
     became free. Instrumentation for this lives in
     `inference.py::generate_text_from_features`.
   - Perception (MediaPipe + DINOv2) is now **63% of total** (35.6s of 56.5s).
   - Blockers: How2Sign **not downloaded** (only the `download_how2sign.sh` stub in
     `online_cslr/`); ~**99 Jetson-hours** of feature extraction for a 10k-clip subset;
     no viable training hardware until the OSU HPC workshop (~Aug 11, 2026); `peft` /
     `bitsandbytes` not installed and unverified on aarch64; and `transformers==4.30.2`
     is old enough that upgrading for `peft` risks breaking the custom T5 subclasses in
     `inference.py` (use a separate training venv).
   - Good news: the model-side change is small — `LinearAdapter.final_layer` *is* the
     SHuBERT→ByT5 projection the paper targets, and ByT5's q/v projections match stock
     `peft` `target_modules=["q","v"]`.
6. **DONE (2026-08-07) — persistent worker + model caching.** Chose latency over the
   QLoRA path. Changes:
   - **`inference.py`**: added `_model_cache` / `_get_cached_model()` / `preload_model()`.
     The ByT5 checkpoint is now loaded once per process instead of per clip
     (13.2s → 0.0s on every clip after the first).
   - **`inference.py`**: ByT5 now loads in **bfloat16** by default (`BYT5_DTYPE` env
     var to override). This halves resident memory ~2.7GB → ~1.4GB. **This was
     necessary, not cosmetic**: with an fp32-resident ByT5 the box swaps and CUDA then
     fails to map memory for DINOv2 — fp32 OOM'd 2 of 3 clips in the worker even at
     batch 16. bf16 not fp16, because T5 overflows in fp16. `LinearAdapter.forward`
     and the input tensors now follow the model's dtype instead of hardcoding float32.
   - **`dinov2_features.py`**: `DEFAULT_BATCH_SIZE = 32` (was 128), overridable with
     `DINOV2_BATCH_SIZE`. DINOv2 peak memory is dominated by per-batch activations,
     and 128 OOM'd once anything else was resident.
   - **`kpe_mediapipe.py`**: `HolisticDetector.close()` now also releases the MediaPipe
     native objects (`mp_holistic`, `face_detector`, `hand_detector`), not just the
     thread pool. Without this the per-clip detectors leaked native memory and OOM'd
     after ~3 clips.
   - **`features.py`**: added `SHuBERTProcessor.warmup()` to preload DINOv2 + ByT5, so
     a long-running process pays model loading at startup rather than on the first clip.
     `app.py` (Gradio) can use this too.
   - **`auto_segment_v5.py`** (new): live capture with a persistent in-process worker
     thread — supersedes `auto_segment_v3.py` (subprocess per clip) and
     `auto_segment_shubert_threaded.py` (whose worker skeleton it reuses). Keeps v3's
     time-based thresholds and error reporting, adds a models-ready gate so clips
     aren't queued while the checkpoint is still loading.

   **Measured (`my_please.mp4`, 6-clip worker run): warmup 17.8s, then a steady
   ~44s per clip**, identical output every time, no memory growth, clean shutdown.
   Versus the original 113.2s and the ~56s one-shot, i.e. **~2.6x faster** end to end
   for the live path.

   **Do NOT cache the MediaPipe `HolisticDetector` across clips.** Tried it; it is
   wrong. `mp.solutions.holistic.Holistic` tracks landmarks *temporally across
   process() calls*, so a reused detector leaks the previous clip's tracking state and
   silently changes the translation (identical input produced "Hello, my name is..."
   instead of "My name is..."). It also bought nothing measurable — constructing a
   detector is free relative to per-frame inference. `video_holistic()` deliberately
   builds and closes one per call; there's a comment there saying so.

   **Use-case-dependent dtype**: bf16 is right for the *live worker* (stability), but
   it makes `generate()` ~3s slower (no native bf16 CPU kernels on ARM), which the
   worker hides by amortizing the load. For **one-shot CLI** runs there is nothing to
   amortize, so `BYT5_DTYPE=float32 python3 run_shubert.py <clip>` is faster
   (62.7s vs 71.8s). One-shot totals vary ±6s with OS page-cache state on the 2.68GB
   checkpoint.
7. **Known trap — the `patch_*.py` scripts.** `patch_timing.py`, `patch_inference.py`,
   and `patch_dataset.py` rewrite `inference.py` in place via unanchored `str.replace`
   with a *relative* path. The model-loading block they target has now been rewritten,
   so they are permanently inert (they print nothing and silently do nothing). Delete
   them or convert to real diffs; do not run them expecting an effect.
6. Re-integrate with `auto_segment_v3.py` once individual clip latency is meaningfully
   reduced, and re-tune the boundary-detection thresholds. Note: `auto_segment_v3.py`
   launches `run_shubert.py` as a fresh subprocess per clip, so any in-process caching
   (like the DINOv2 fix above) only helps within a single clip, not across clips —
   worth reconsidering that architecture (long-running process vs. per-clip subprocess)
   once latency work otherwise stabilizes.

## Paths we tried and rejected (don't redo this research)
- **Zuo/Wei/Mak "Online CSLR"** (github.com/FangyunWei/SLRT, EMNLP 2024): real,
  working code, but requires gloss-annotated continuous sign video to train a
  prerequisite model (TwoStream-SLR). We could not find a viable ASL gloss dataset:
  How2Sign's gloss annotations are described in their paper but NOT actually available
  in their download pipeline (marked "TODO" in `download_how2sign.sh`), and the
  official GitHub issue asking for them has been open and unanswered since Dec 2021.
  NCSLGR (Boston University) is a real gloss-annotated ASL corpus but far too small
  (~5.3 hours) to be a practical training set on its own. This path is not currently
  viable without a data breakthrough — don't re-attempt without new information.
- **KD-MSLRT** (AAAI 2025): looked promising (lightweight, knowledge distillation) but
  has no public code release.
- Training TwoStream-SLR itself would require HRNet keypoints (not MediaPipe-based)
  and 8-GPU distributed training — a large undertaking we paused given the gloss-data
  blocker above.

## Hardware/account notes
- OSU HPC cluster access exists but requires an intro workshop (scheduled ~Aug 11,
  2026) before use — check if that's happened yet.
- Jetson power mode should be maxed for best performance:
  `sudo nvpmodel -m 0 && sudo jetson_clocks` (note: on this specific board, mode 0
  is the actual max — it does not have the newer "Super Mode"/25W tier some other
  Orin Nano units support).
- Camera is a Logitech ConferenceCam, typically `/dev/video0`, works best in MJPG
  format at 640x480 (`cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))`).

## How to activate the right environment
```bash
cd ~/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText
source shubert_venv/bin/activate
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

## How to run
```bash
# Live capture (primary path) — persistent worker, ~44s/clip after ~18s warmup
python3 auto_segment_v5.py

# One-shot on a file — fp32 is faster here since there's no load to amortize
BYT5_DTYPE=float32 python3 run_shubert.py my_please.mp4
```
Tuning knobs (both have sane defaults; only touch them under memory pressure):
`DINOV2_BATCH_SIZE` (default 32) and `BYT5_DTYPE` (default `bfloat16`).
