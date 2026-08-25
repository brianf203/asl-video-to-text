Benchmarks behind the 2026-08-24 (later) PROJECT_CONTEXT.md block — "ByT5 quantization
is dead on arrival". Run from this directory with `shubert_venv` active.

- `convert_bf16.py <outdir>` — re-save checkpoint-11625 at bf16 (1.34GB vs 2.68GB fp32).
  Halves the host-RAM load peak, which is what lets the two benchmarks below load onto the
  GPU at all on a busy desktop. Writes with safe_serialization=False (tied lm_head).
- `byt5_split.py` — splits generate() into SHuBERT adapter / ByT5 encoder / decode, and
  prints the parameter distribution. Result: decode is 97-99% of the stage; the 18-block
  encoder is 62% of the weights and ~2% of the time.
- `decode_bound.py` — the decisive one. Fixes the decode step count and varies num_beams.
  8x the compute costs 1.27x the time => the step is overhead-bound, not weight-bound,
  so weight quantization cannot help.

Both benchmarks read the bf16 checkpoint from $BYT5_BF16_CKPT (default
/home/sllu/byt5_ckpt_bf16). Create it first:
    python3 convert_bf16.py /home/sllu/byt5_ckpt_bf16
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
