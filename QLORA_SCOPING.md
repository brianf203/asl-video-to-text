# QLoRA Fine-Tuning — Scoping Notes (2026-08-07)

Scoping pass for step 5 of `PROJECT_CONTEXT.md` ("QLoRA fine-tune ByT5 on a How2Sign
subset, SHuBERT frozen"). **Conclusion up front: if the goal is latency, this is now a
low-leverage target and should probably not be next. If the goal is translation
accuracy on our domain, it's still worth doing but is gated on several prerequisites.**

## 1. The latency payoff is much smaller than we assumed

We had ByT5 pegged as the largest remaining stage (~20-22s). Splitting that stage
(instrumentation added to `inference.py::generate_text_from_features`) shows it is
mostly **disk I/O, not compute**:

| Sub-stage | Time | % of 56.5s total |
|---|---|---|
| ByT5 checkpoint load (`from_pretrained`, 2.68 GB `pytorch_model.bin`) | 13.2s | 23% |
| `to(device)` + `eval()` | 0.0s | ~0% |
| **ByT5 `generate()` — actual compute** | **7.1s** | **13%** |

Full pipeline for reference (`my_please.mp4`, 111 frames):

| Stage | Time | % |
|---|---|---|
| MediaPipe landmarks | 17.6s | 31% |
| DINOv2 hands | 11.0s | 19% |
| ByT5 checkpoint load | 13.2s | 23% |
| ByT5 generate | 7.1s | 13% |
| DINOv2 face | 7.0s | 12% |
| video read / crops / pose | ~0.5s | ~1% |

**Perception (MediaPipe + DINOv2) is 35.6s = 63% of total.** ByT5's actual inference
compute is 7.1s = 13%.

QLoRA optimizes model weights/compute — i.e. it targets the **7.1s**, not the 13.2s
load. Even if QLoRA made generation *instantaneous and free*, total latency would go
56.5s → ~49.4s (a 12.6% win). By comparison, a persistent worker process that loads the
checkpoint once (see §4) removes 13.2s for a fraction of the effort.

Caveat on framing: the reference paper's QLoRA step is primarily about **adapting the
model to the target domain**, not about speed. If our goal is better translation
quality on our own footage, the latency math above does not apply and QLoRA is judged
on accuracy instead. **This is the key question to settle before investing further —
see §5.**

## 2. Prerequisites we do not currently have

### 2a. Training data — not downloaded (largest single blocker)
- How2Sign is **not present locally**. Only `online_cslr/how2sign-data/download_how2sign.sh`
  exists (from the abandoned Online CSLR direction).
- Disk: **100 GB free** of 233 GB. How2Sign's full RGB video release is on the order of
  hundreds of GB — needs verification, but likely does not fit alongside existing data.
  A curated ~10k-clip subset (what the paper used) would need to be selected carefully
  rather than mirroring the whole dataset.
- **Feature extraction cost is the hidden killer.** Training a ByT5 adapter on frozen
  SHuBERT does not need raw video at train time *if* features are precomputed — but
  precomputing them means running our perception stack over every training clip:

  > 35.6s perception per clip × 10,000 clips ≈ **99 hours ≈ 4.1 days** of continuous
  > Jetson compute, before a single training step runs.

  This is a one-time cost and is parallelizable on a cluster, but it is not something
  to start on the Jetson casually. It also argues for doing extraction wherever the
  training will happen, not here.

### 2b. Training hardware — no viable target yet
- The Jetson has 8 GB *shared* memory and already OOMs at **inference** when a browser
  is open. QLoRA training on-device is not realistic.
- OSU HPC access requires the intro workshop scheduled **~Aug 11, 2026** — 4 days from
  today (Aug 7, 2026). **Nothing here can start in earnest until that lands.** Worth
  confirming the date and registering.

### 2c. Software stack — untested, with a real version-conflict risk
- `peft` and `bitsandbytes` are **not installed**. Both exist on the index
  (bitsandbytes 0.50.0, peft 0.20.0), but an aarch64 wheel that actually imports with
  CUDA sm_87 (Orin) is **unverified** — bitsandbytes on Jetson is historically painful.
  Note this only matters if we train *here*; on an x86 HPC node it is a non-issue.
- **Version conflict risk (important):** this venv runs `transformers==4.30.2` (June
  2023). `inference.py` defines `SignLanguageByT5Encoder` /
  `SignLanguageByT5ForConditionalGeneration` by subclassing T5 internals
  (`transformers.models.t5.modeling_t5 import *`, `T5PreTrainedModel`,
  `parallelize`/`deparallelize`, `_reorder_cache`) from that specific version. Modern
  `peft` expects a much newer `transformers`. Upgrading risks breaking the custom model
  code that currently works. Plan for a **separate training venv** rather than mutating
  the working inference environment.

## 3. What QLoRA would actually attach to

The wiring is already adapter-shaped, which is good news for this approach:

- `inference.py::LinearAdapter` (line 55) holds the SHuBERT→ByT5 path:
  `signhubert_adapter` (frozen SHuBERT) → learnable `layer_weights` (12, one per
  layer) → `final_layer = nn.Linear(representations_dim, out_dim)`.
- `final_layer` **is** the "SHuBERT→ByT5 projection layer" the paper targets. It is a
  single `nn.Linear` and is already trainable in isolation.
- The paper additionally targets ByT5's **query/value projections** — in this codebase
  those live inside the T5 blocks of `SignLanguageByT5ForConditionalGeneration`
  (standard `q`/`v` module names, so a stock `peft` `LoraConfig(target_modules=["q","v"])`
  should match without custom surgery).

So the model-side change is genuinely small. The cost is all in data, hardware, and
environment — not in the modeling code.

Also noted: `checkpoint-11625/` contains a **5.3 GB `optimizer.pt`** (training state,
unnecessary for inference) alongside the 2.68 GB `pytorch_model.bin`. Not a bug, but
relevant to §4.

## 4. Cheaper wins that dominate QLoRA on latency

If latency is the goal, do these first — all are far less work than QLoRA:

1. **Persistent worker process (biggest win, and it compounds).** `auto_segment_v3.py`
   spawns `run_shubert.py` as a fresh subprocess per clip, so *every* per-process cost
   is paid per clip: the 13.2s ByT5 load, both DINOv2 loads, MediaPipe init, and torch
   import. Restructuring to a long-running process that loads models once and consumes
   clips from a queue removes ~13.2s+ per clip and finally makes the DINOv2 cache and
   MediaPipe thread pool (already implemented) pay off across clips instead of only
   within one. **Estimated: 56.5s → ~43s for the first clip and substantially less for
   every subsequent clip.**
2. **Shrink/convert the checkpoint.** Re-save `pytorch_model.bin` as fp16 and/or
   `safetensors` with the optimizer state stripped. Should cut a large chunk of the
   13.2s load. Low risk, quick to test.
3. **Perception is still 63% of total.** If more latency is needed after the above,
   the remaining headroom is in MediaPipe (17.6s) and DINOv2 (18.0s) — e.g. frame
   subsampling, batching the DINOv2 calls, or TensorRT conversion — not in ByT5.

## 5. Recommendation / open question

**If the goal is latency:** do §4.1 and §4.2 first. QLoRA is at best a 12.6% win and
carries a multi-day data-prep cost, a hardware blocker, and environment risk.

**If the goal is translation accuracy on our own domain** (our recorded footage, not
just the bundled examples), QLoRA is the right tool and the modeling work is small
(§3) — but it is gated on: (a) the OSU HPC workshop ~Aug 11, (b) selecting and
downloading a How2Sign subset, and (c) budgeting ~99 Jetson-hours of feature
extraction *or* doing extraction on the cluster.

Suggested sequencing if we want both: ship §4.1 + §4.2 now (days), and in parallel
start the How2Sign subset download + HPC access so QLoRA is unblocked when the cluster
is available.
