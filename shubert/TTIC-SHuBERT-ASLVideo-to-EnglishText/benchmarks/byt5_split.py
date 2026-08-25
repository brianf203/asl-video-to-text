"""Where does the ByT5 stage actually spend its time?

Splits generate() into SHuBERT adapter / ByT5 encoder (18 layers) / autoregressive
decode (6 layers x beams).  This decides whether weight-only quantization can help:
it pays off on bandwidth-bound autoregressive decode, not on a single big encoder pass.
"""
import os, sys, time, torch, numpy as np
sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

MODELS_BASE = ("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/"
               "snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
CKPT = os.path.join(MODELS_BASE, "checkpoint-11625")
TOK  = os.path.join(MODELS_BASE, "byt5_base")

import gc, inference
from transformers import ByT5Tokenizer
def avail(): return int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0])/1048576
BF16 = os.environ.get("BYT5_BF16_CKPT", "/home/sllu/byt5_ckpt_bf16")
print(f"avail before load: {avail():.2f} Gi")
model = inference.SignLanguageByT5ForConditionalGeneration.from_pretrained(
    BF16, torch_dtype=torch.bfloat16)
tokenizer = ByT5Tokenizer.from_pretrained(TOK)
print(f"avail after  load: {avail():.2f} Gi")
gc.collect()
device = torch.device("cuda")
model.to(device); model.eval()
print(f"avail after ->gpu: {avail():.2f} Gi")
dt = next(model.parameters()).dtype
print(f"device={device} dtype={dt}\n")

def nparams(m): return sum(p.numel() for p in m.parameters())
shub = model.encoder.adapter.signhubert_adapter
enc_blocks = model.encoder.block
total = nparams(model)
parts = [("SHuBERT (adapter.signhubert)", nparams(shub)),
         ("ByT5 encoder 18 blocks", nparams(enc_blocks)),
         ("ByT5 decoder 6 blocks", nparams(model.decoder)),
         ("lm_head + decoder_emb", nparams(model.lm_head) + nparams(model.decoder_emb)),
         ("adapter.final_layer", nparams(model.encoder.adapter.final_layer))]
print(f"{'component':32}{'params':>14}{'share':>8}")
for n, p in parts:
    print(f"{n:32}{p:14,}{100*p/total:7.1f}%")
print(f"{'TOTAL':32}{total:14,}\n")

def sync(): torch.cuda.synchronize() if device.type == "cuda" else None

def bench(T, beams=4, reps=3):
    g = torch.Generator().manual_seed(0)
    f = torch.randn(1, T, 384, generator=g).to(device=device, dtype=dt)
    l = torch.randn(1, T, 384, generator=g).to(device=device, dtype=dt)
    r = torch.randn(1, T, 384, generator=g).to(device=device, dtype=dt)
    p = torch.randn(1, T, 14,  generator=g).to(device=device, dtype=dt)
    out = {}
    with torch.no_grad():
        for _ in range(1):   # warmup
            model.encoder(face_features=f, left_hand_features=l,
                          right_hand_features=r, pose_features=p)
        sync()
        ts = []
        for _ in range(reps):
            t0 = time.time()
            model.encoder.adapter(f, l, r, p); sync()
            ts.append(time.time() - t0)
        out["shubert"] = min(ts)
        ts = []
        for _ in range(reps):
            t0 = time.time()
            model.encoder(face_features=f, left_hand_features=l,
                          right_hand_features=r, pose_features=p); sync()
            ts.append(time.time() - t0)
        out["encoder_total"] = min(ts)
        ts, ntok = [], 0
        for _ in range(reps):
            t0 = time.time()
            ids = model.generate(face_features=f, left_hand_features=l,
                                 right_hand_features=r, pose_features=p,
                                 max_length=768, num_beams=beams, early_stopping=True,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
            sync(); ts.append(time.time() - t0); ntok = ids.shape[-1]
        out["generate"] = min(ts); out["out_bytes"] = ntok
    out["byt5_enc"] = out["encoder_total"] - out["shubert"]
    out["decode"] = out["generate"] - out["encoder_total"]
    return out

print(f"{'T frames':>9}{'SHuBERT':>10}{'ByT5enc':>10}{'decode':>10}{'generate':>10}"
      f"{'bytes':>7}   {'decode share':>12}")
for T in (40, 100, 220):
    o = bench(T)
    print(f"{T:>9}{o['shubert']:9.2f}s{o['byt5_enc']:9.2f}s{o['decode']:9.2f}s"
          f"{o['generate']:9.2f}s{o['out_bytes']:7d}   {100*o['decode']/o['generate']:11.0f}%")
