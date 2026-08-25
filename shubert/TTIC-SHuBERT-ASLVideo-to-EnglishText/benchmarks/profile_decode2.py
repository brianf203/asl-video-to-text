"""Decisive launch-bound test without CUPTI, plus the recoverable-win estimate.

The raw decode loop keeps every tensor on the GPU (argmax feeds the next step as a
device tensor), so it enqueues fully asynchronously. That lets us split:
    CPU enqueue time   = t1 - t0   (no sync)
    total wall         = t2 - t0   (after sync)
If enqueue ~= total, the CPU cannot feed the GPU fast enough -> launch/dispatch bound,
and CUDA graphs / a leaner loop are the levers. If total >> enqueue, the GPU is the
bottleneck and nothing but less compute helps.
"""
import os, sys, time, torch
sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")
import inference
from transformers import ByT5Tokenizer
MB = ("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/"
      "snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
BF16 = os.environ.get("BYT5_BF16_CKPT", "/home/sllu/byt5_ckpt_bf16")
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
with torch.no_grad():
    h = model.encoder(**feat)[0].repeat_interleave(BEAMS, dim=0)

def loop(steps=STEPS, dec=None):
    dec = dec or model.decoder
    with torch.no_grad():
        past, ids = None, torch.zeros((BEAMS,1), dtype=torch.long, device=dev)
        for _ in range(steps):
            out = dec(input_ids=ids, past_key_values=past,
                      encoder_hidden_states=h, use_cache=True, return_dict=True)
            past = out.past_key_values
            ids = model.lm_head(out[0])[:, -1:].argmax(-1)

# ---- A: enqueue vs execute ----
loop(); torch.cuda.synchronize(); loop(); torch.cuda.synchronize()
enq, tot = [], []
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.time(); loop(); t1 = time.time()
    torch.cuda.synchronize(); t2 = time.time()
    enq.append(t1-t0); tot.append(t2-t0)
e, t = min(enq), min(tot)
print(f"\n[A] raw decode loop, {STEPS} steps, beams={BEAMS}")
print(f"    CPU enqueue : {e:6.3f}s  {1000*e/STEPS:6.2f} ms/step")
print(f"    total wall  : {t:6.3f}s  {1000*t/STEPS:6.2f} ms/step")
print(f"    GPU tail after CPU finished enqueueing: {1000*(t-e)/STEPS:6.2f} ms/step "
      f"({100*(t-e)/t:.1f}% of wall)")
print(f"    => {'CPU/LAUNCH-BOUND' if (t-e)/t < 0.25 else 'GPU-BOUND'}")

# ---- B: how much is recoverable? torch.compile the decoder ----
print(f"\n[B] torch.compile(model.decoder) - upper bound on the recoverable win")
try:
    t0 = time.time()
    cdec = torch.compile(model.decoder, mode="reduce-overhead", dynamic=True)
    loop(dec=cdec); torch.cuda.synchronize()
    print(f"    compile+warmup: {time.time()-t0:.1f}s")
    loop(dec=cdec); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.time(); loop(dec=cdec)
        torch.cuda.synchronize(); ts.append(time.time()-t0)
    c = min(ts)
    print(f"    compiled    : {c:6.3f}s  {1000*c/STEPS:6.2f} ms/step   "
          f"speedup vs raw {t/c:.2f}x")
except Exception as ex:
    print(f"    FAILED: {type(ex).__name__}: {str(ex)[:300]}")
