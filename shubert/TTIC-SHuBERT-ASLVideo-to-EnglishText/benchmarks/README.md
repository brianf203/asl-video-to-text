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
