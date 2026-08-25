"""Profile the ~40ms ByT5 decode step.

Separates three candidate causes:
  (1) real GPU compute        -> nothing to win
  (2) CUDA kernel-launch overhead (GPU idle, many tiny kernels) -> CUDA graphs win
  (3) generate() Python bookkeeping (beam search, logits processing, stopping)
                              -> a leaner decode loop wins, no dependency change

Method: compare wall time per step across three levels of machinery, and use the
profiler to get GPU-busy time and kernel counts per step.
"""
import os, sys, time, torch
sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")
import inference
from transformers import ByT5Tokenizer

MB = ("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/"
      "snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
BF16 = os.environ.get("BYT5_BF16_CKPT", os.path.join(MB, "checkpoint-11625-bf16"))
model = inference.SignLanguageByT5ForConditionalGeneration.from_pretrained(
    BF16, torch_dtype=torch.bfloat16)
tok = ByT5Tokenizer.from_pretrained(os.path.join(MB, "byt5_base"))
dev, dt = torch.device("cuda"), torch.bfloat16
model.to(dev); model.eval()

T, STEPS, BEAMS = 100, 64, 4
g = torch.Generator().manual_seed(0)
feat = dict(face_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            left_hand_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            right_hand_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            pose_features=torch.randn(1,T,14,generator=g).to(dev,dt))

def gen(beams=BEAMS, steps=STEPS):
    with torch.no_grad():
        model.generate(**feat, max_length=steps, min_length=steps, num_beams=beams,
                       early_stopping=False, pad_token_id=tok.pad_token_id,
                       eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

# ---------- level 1: full generate() ----------
gen(); gen()                                     # warmup
ts=[]
for _ in range(5):
    t0=time.time(); gen(); ts.append(time.time()-t0)
full = min(ts)/STEPS*1000
print(f"\n[1] full generate()            : {min(ts):6.3f}s  {full:6.2f} ms/step")

# ---------- level 2: raw decoder forward loop, KV cache, no generate machinery ----------
with torch.no_grad():
    enc = model.encoder(**feat)
    h = enc[0].repeat_interleave(BEAMS, dim=0)   # mimic beam expansion
def raw_loop(steps=STEPS):
    with torch.no_grad():
        past, ids = None, torch.zeros((BEAMS,1), dtype=torch.long, device=dev)
        for _ in range(steps):
            out = model.decoder(input_ids=ids, past_key_values=past,
                                encoder_hidden_states=h, use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = model.lm_head(out[0])
            ids = logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
raw_loop(); raw_loop()
ts=[]
for _ in range(5):
    t0=time.time(); raw_loop(); ts.append(time.time()-t0)
raw = min(ts)/STEPS*1000
print(f"[2] raw decoder loop (no gen)  : {min(ts):6.3f}s  {raw:6.2f} ms/step")
print(f"    -> generate() bookkeeping  : {full-raw:6.2f} ms/step  ({100*(full-raw)/full:.0f}% of the step)")

# ---------- level 3: profiler on the raw loop - GPU busy vs wall, kernel count ----------
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    raw_loop()
evts = prof.key_averages()
cuda_us = sum(e.self_device_time_total for e in evts)
cpu_us  = sum(e.self_cpu_time_total for e in evts)
nlaunch = sum(e.count for e in evts if e.self_device_time_total > 0)
print(f"\n[3] profiler over {STEPS} raw steps")
print(f"    GPU busy   : {cuda_us/1000:8.1f} ms total   {cuda_us/1000/STEPS:6.2f} ms/step")
print(f"    CPU self   : {cpu_us/1000:8.1f} ms total   {cpu_us/1000/STEPS:6.2f} ms/step")
print(f"    GPU kernels: {nlaunch:8d} total   {nlaunch/STEPS:6.1f} per step")
print(f"    GPU busy fraction of wall: {100*(cuda_us/1000)/(raw*STEPS):.1f}%")

print("\n    top 12 ops by SELF CPU time:")
for e in sorted(evts, key=lambda e:-e.self_cpu_time_total)[:12]:
    print(f"      {e.key[:44]:44} {e.self_cpu_time_total/1000/STEPS:7.3f} ms/step  n={e.count/STEPS:6.1f}")
print("\n    top 8 ops by SELF CUDA time:")
for e in sorted(evts, key=lambda e:-e.self_device_time_total)[:8]:
    print(f"      {e.key[:44]:44} {e.self_device_time_total/1000/STEPS:7.3f} ms/step  n={e.count/STEPS:6.1f}")
