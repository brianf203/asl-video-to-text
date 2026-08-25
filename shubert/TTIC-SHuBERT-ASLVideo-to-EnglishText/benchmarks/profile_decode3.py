"""How much of the ~31ms/step is fixed overhead, and can it be recovered here?

[C] Beam sweep. GPU compute per step scales ~linearly with beams; CPU dispatch cost
    is constant in beams (same op count). Fit  t(B) = overhead + k*B  and the
    intercept IS the launch/dispatch floor -- i.e. the prize CUDA graphs would win.
[D] torch.compile(backend="cudagraphs") -- CUDA graphs without needing triton.
"""
import os, sys, time, torch
sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")
import inference
from transformers import ByT5Tokenizer
MB = ("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/"
      "snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
model = inference.SignLanguageByT5ForConditionalGeneration.from_pretrained(
    os.environ.get("BYT5_BF16_CKPT", os.path.join(MB, "checkpoint-11625-bf16")), torch_dtype=torch.bfloat16)
tok = ByT5Tokenizer.from_pretrained(os.path.join(MB, "byt5_base"))
dev, dt = torch.device("cuda"), torch.bfloat16
model.to(dev); model.eval()
T, STEPS = 100, 32
g = torch.Generator().manual_seed(0)
feat = dict(face_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            left_hand_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            right_hand_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            pose_features=torch.randn(1,T,14,generator=g).to(dev,dt))
with torch.no_grad():
    enc = model.encoder(**feat)[0]

def loop(beams, steps=STEPS, dec=None):
    dec = dec or model.decoder
    h = enc.repeat_interleave(beams, dim=0)
    with torch.no_grad():
        past, ids = None, torch.zeros((beams,1), dtype=torch.long, device=dev)
        for _ in range(steps):
            out = dec(input_ids=ids, past_key_values=past, encoder_hidden_states=h,
                      use_cache=True, return_dict=True)
            past = out.past_key_values
            ids = model.lm_head(out[0])[:, -1:].argmax(-1)
    torch.cuda.synchronize()

print(f"\n[C] beam sweep, {STEPS} steps  (dispatch cost is constant in beams; GPU compute scales)")
print(f"{'beams':>6}{'ms/step':>10}{'vs B=1':>9}")
xs, ys = [], []
for b in (1,2,4,8,16,32,64):
    loop(b)                                        # warmup this shape
    ts=[]
    for _ in range(3):
        t0=time.time(); loop(b); ts.append(time.time()-t0)
    ms = 1000*min(ts)/STEPS; xs.append(b); ys.append(ms)
    print(f"{b:>6}{ms:9.2f}{ms/ys[0]:8.2f}x")
# least squares on the linear part (drop B=1 curvature)
import statistics
X, Y = xs[2:], ys[2:]
n=len(X); mx=sum(X)/n; my=sum(Y)/n
k=sum((x-mx)*(y-my) for x,y in zip(X,Y))/sum((x-mx)**2 for x in X)
c=my-k*mx
print(f"\n    fit over beams>=4:  t(B) = {c:.2f} ms + {k:.4f} ms * B")
print(f"    fixed dispatch/launch floor : {c:.2f} ms/step")
print(f"    GPU compute at beams=4      : {k*4:.2f} ms/step")
print(f"    => at beams=4, {100*c/(c+k*4):.0f}% of the step is fixed overhead")
print(f"    => perfect overhead removal would give {(c+k*4)/(k*4):.1f}x on decode")

print(f"\n[D] torch.compile(backend='cudagraphs') - no triton needed")
for target, label in ((model.decoder, "module"), (model.decoder.forward, "forward")):
    try:
        t0=time.time()
        cdec = torch.compile(target, backend="cudagraphs")
        loop(4, dec=cdec)
        print(f"    {label}: compiled+warmup in {time.time()-t0:.1f}s")
        ts=[]
        for _ in range(3):
            t0=time.time(); loop(4, dec=cdec); ts.append(time.time()-t0)
        ms=1000*min(ts)/STEPS
        print(f"    {label}: {ms:.2f} ms/step   speedup {ys[2]/ms:.2f}x")
        break
    except Exception as ex:
        print(f"    {label}: FAILED {type(ex).__name__}: {str(ex)[:200]}")
