Benchmarks behind the PROJECT_CONTEXT.md blocks that closed the ByT5 speed levers
(2026-08-24) and diagnosed the perception wall (2026-08-25). Run from this directory
with `shubert_venv` active.

- `convert_bf16.py <MODELS_BASE>` — re-save checkpoint-11625 at bf16 (1.34GB vs 2.68GB
  fp32), writing `checkpoint-11625-bf16` beside it. This is a SETUP STEP, not just a
  benchmark: the pipeline loads that path (see the repo README, step 6). Halves the
  host-RAM load peak, which is also what lets the benchmarks below load onto the GPU at
  all on a busy desktop. Writes via a temp dir + os.replace so an interrupted run cannot
  leave a half-written checkpoint, and with safe_serialization=False (tied lm_head).
- `byt5_split.py` — splits generate() into SHuBERT adapter / ByT5 encoder / decode, and
  prints the parameter distribution. Result: decode is 97-99% of the stage; the 18-block
  encoder is 62% of the weights and ~2% of the time.
- `decode_bound.py` — the decisive one. Fixes the decode step count and varies num_beams.
  8x the compute costs 1.27x the time => the step is overhead-bound, not weight-bound,
  so weight quantization cannot help.

Both benchmarks read the bf16 checkpoint from $BYT5_BF16_CKPT, defaulting to
<MODELS_BASE>/checkpoint-11625-bf16 — the same one the pipeline itself loads, so if setup
step 6 has been done there is nothing to create. Otherwise:
    python3 convert_bf16.py <MODELS_BASE>
They use SYNTHETIC features — fine here because per-step decode cost is architecture-
determined and content-independent, and the step count is pinned explicitly. They are not
valid for anything quality-related.

Profiling of the ~40ms decode step (2026-08-24 evening block):
- `profile_decode.py`  — full generate() vs a raw decoder loop; CPU-side op table.
  NOTE: its GPU columns read zero because CUPTI is unavailable to a non-root user on
  this box (CUPTI_ERROR_INSUFFICIENT_PRIVILEGES). Ignore them; they are not a
  measurement. The two scripts below were written to get the answer without CUPTI.
- `profile_decode2.py` — the decisive one. The raw loop keeps everything on-device so it
  enqueues asynchronously; comparing CPU-enqueue time against post-synchronize wall time
  shows the GPU tail is 0.0%. Also attempts torch.compile.
- `profile_decode3.py` — beam sweep 1..64 with a linear fit separating the constant
  dispatch floor from per-beam GPU compute, plus the cudagraphs-backend attempt.

Inductor experiment (2026-08-24 night block):
- `inductor_test.py` — validates triton codegen on sm_87, then compiles the raw decoder
  loop. Its "[0] numerics match: False" is a BAD TEST (atol too tight for a bf16 sum over
  512 elements); `inductor_e2e.py` re-checks it properly and it passes.
- `inductor_e2e.py`  — the one that matters: end-to-end generate() eager vs decoder
  compiled with inductor, asserting decoded ids are identical. Set
  TORCHINDUCTOR_CACHE_DIR to a persistent path (NOT /tmp, which clears on reboot).

bf16 checkpoint re-save (2026-08-24 bf16 block):
- `load_peak.py <ckpt_dir> <label>` — measures checkpoint load time, GPU transfer, and the
  host-RAM peak, and prints a sha256 over every loaded parameter. The hash is the point: it
  proves the bf16 re-save is bitwise identical to the fp32 checkpoint cast at load, which is
  stronger evidence than comparing decoded text. Run it once per checkpoint and compare.
  Make the bf16 checkpoint with `convert_bf16.py <MODELS_BASE>`.

MediaPipe hand detection — the perception wall (2026-08-25 block):
- `hand_detect_bound.py` — times the hand landmarker ALONE (no pose/face contention) and
  splits the cost by how many hands came back. That split is the diagnostic: one-hand
  frames cost MORE than two-hand frames, because below `num_hands` MediaPipe re-runs the
  palm detector hunting for the hand it is missing, and against ASL's frequent one-handed
  frames it does that on ~40% of them. Sweeps the two knobs that had never been tuned
  (`min_hand_presence_confidence`, `min_tracking_confidence`) and includes `num_hands=1`
  as a diagnostic ceiling — 1.88-2.05x, but it discards the second hand, so it is not a
  shipping candidate without the 200-clip gate.
  Runs baseline first AND last as a closing control, which is load-bearing here: on one
  run the two identical baselines differed by 9.4%, which is larger than every confidence-
  knob effect measured. Read no single-run ranking from this script without that control.
  Standalone per-detector numbers only — see the docstring, and this project's five
  recorded cases of exactly such numbers evaporating on the live path.
