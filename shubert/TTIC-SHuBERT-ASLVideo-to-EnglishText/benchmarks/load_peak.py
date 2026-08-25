"""Measure the checkpoint load peak, and prove the bf16 re-save is weight-identical.

The shipping loader casts the fp32 checkpoint to bf16 at load (torch_dtype), so a
checkpoint re-saved FROM that cast model must produce bitwise-identical parameters.
Hashing the loaded state dict proves that outright -- stronger than comparing one decode.

    python3 load_peak.py <checkpoint_dir> <label>
"""
import os, sys, time, threading, hashlib, torch
sys.path.insert(0,"/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")
CKPT, LABEL = sys.argv[1], sys.argv[2]

def avail_mb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"): return int(l.split()[1])//1024

stop=False; lo=[avail_mb()]
def sampler():
    while not stop:
        v=avail_mb()
        if v<lo[0]: lo[0]=v
        time.sleep(0.05)
t=threading.Thread(target=sampler,daemon=True); t.start()

import inference
from transformers import ByT5Tokenizer
start_avail=avail_mb()
t0=time.time()
model=inference.SignLanguageByT5ForConditionalGeneration.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16)
load_t=time.time()-t0
after_load=avail_mb()
t1=time.time()
model.to("cuda"); model.eval()
gpu_t=time.time()-t1
stop=True; time.sleep(0.2)

# bitwise hash of every parameter, in a stable order
h=hashlib.sha256()
for name,p in sorted(model.state_dict().items()):
    h.update(name.encode())
    h.update(p.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
print(f"{LABEL:>6} | disk {sum(os.path.getsize(os.path.join(CKPT,f)) for f in os.listdir(CKPT))/1e9:5.2f} GB "
      f"| load {load_t:6.1f}s | ->gpu {gpu_t:5.1f}s | peak used {start_avail-lo[0]:5d} MB "
      f"| min avail {lo[0]:5d} MB | sha256 {h.hexdigest()[:16]}")
