#!/usr/bin/env python3
"""Re-save checkpoint-11625 at bfloat16, which is what the pipeline actually loads.

    python3 benchmarks/convert_bf16.py <MODELS_BASE>

Reads  <MODELS_BASE>/checkpoint-11625        (fp32, 2.68GB)
Writes <MODELS_BASE>/checkpoint-11625-bf16   (bf16, 1.34GB)

The result is BITWISE IDENTICAL to what the fp32 checkpoint becomes at load, because
inference.py passes torch_dtype=bfloat16 and the cast happens anyway. Verify with
`benchmarks/load_peak.py <ckpt> <label>` -- both checkpoints hash to the same sha256.

Why: from_pretrained materialises the whole fp32 file before casting, leaving ~3.58GB
resident for a 1.34GB model, and on this board's unified memory that repeatedly OOMs at
.to("cuda"). Loading the bf16 copy is ~1.6x faster and leaves ~850MB more headroom.

Writes to a temp directory and moves it into place only on success, so an interrupted
run cannot leave a half-written checkpoint that the pipeline would then try to load.
"""
import os, shutil, sys, tempfile, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["BYT5_DEVICE"] = "cpu"          # never needs the GPU
os.environ.pop("BYT5_CKPT", None)          # ignore any override; convert the real thing

if len(sys.argv) != 2:
    sys.exit(__doc__)
BASE = os.path.abspath(sys.argv[1])
SRC = os.path.join(BASE, "checkpoint-11625")
DST = os.path.join(BASE, "checkpoint-11625-bf16")
TOK = os.path.join(BASE, "byt5_base")
for p in (SRC, TOK):
    if not os.path.isdir(p):
        sys.exit(f"not found: {p}\nIs {BASE} really MODELS_BASE?")
if os.path.isdir(DST):
    sys.exit(f"already exists: {DST}\nDelete it first if you mean to regenerate.")

import inference
print(f"loading fp32 from {SRC} ...")
model, _tok, _dev = inference._get_cached_model(SRC, TOK, tempfile.gettempdir())
print(f"loaded, dtype={next(model.parameters()).dtype}")

tmp = tempfile.mkdtemp(prefix="ckpt_bf16_", dir=BASE)
try:
    # safe_serialization=False: lm_head and decoder_emb are tied, and safetensors
    # refuses to write tensors that share storage.
    model.save_pretrained(tmp, safe_serialization=False)
    with open(os.path.join(tmp, "PROVENANCE.txt"), "w") as f:
        f.write(
            "DERIVED ARTIFACT - generated locally by benchmarks/convert_bf16.py,\n"
            "not downloaded from HuggingFace.\n\n"
            f"Source: {SRC}\n"
            "This is that checkpoint re-saved at bfloat16. It is bitwise identical to what\n"
            "the pipeline already loads (inference.py passes torch_dtype=bfloat16), verified\n"
            "by hashing every loaded parameter with benchmarks/load_peak.py.\n\n"
            "Regenerate:  python3 benchmarks/convert_bf16.py <MODELS_BASE>\n"
            "Use fp32 instead, without editing code:  export BYT5_CKPT=<...>/checkpoint-11625\n")
    os.replace(tmp, DST)
except BaseException:
    shutil.rmtree(tmp, ignore_errors=True)
    raise
size = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST))
print(f"wrote {DST}  ({size/1e9:.2f} GB)")
print("verify:  python3 benchmarks/load_peak.py <this-dir> bf16")
