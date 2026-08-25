"""Does inductor+triton work on Orin (sm_87), and does it help the decode step?"""
import os, sys, time, torch
sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")

# ---- Test 0: does triton codegen work on this GPU at all? ----
print("[0] trivial inductor compile (validates triton on sm_87)")
def fn(a, b): return (a * b + a.sin()).relu().sum(-1)
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
try:
    t0 = time.time(); c = torch.compile(fn); r = c(a, b); torch.cuda.synchronize()
    ref = fn(a, b)
    ok = torch.allclose(r.float(), ref.float(), atol=1e-2)
    print(f"    OK - compiled in {time.time()-t0:.1f}s, numerics match: {ok}")
except Exception as ex:
    print(f"    FAILED {type(ex).__name__}: {str(ex)[:400]}"); sys.exit(1)

# ---- load model ----
import inference
from transformers import ByT5Tokenizer
MB = ("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/"
      "snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
model = inference.SignLanguageByT5ForConditionalGeneration.from_pretrained(
    os.environ.get("BYT5_BF16_CKPT", os.path.join(MB, "checkpoint-11625-bf16")), torch_dtype=torch.bfloat16)
tok = ByT5Tokenizer.from_pretrained(os.path.join(MB, "byt5_base"))
dev, dt = torch.device("cuda"), torch.bfloat16
model.to(dev); model.eval()
T, STEPS, BEAMS = 100, 32, 4
g = torch.Generator().manual_seed(0)
feat = dict(face_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            left_hand_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            right_hand_features=torch.randn(1,T,384,generator=g).to(dev,dt),
            pose_features=torch.randn(1,T,14,generator=g).to(dev,dt))
with torch.no_grad():
    h = model.encoder(**feat)[0].repeat_interleave(BEAMS, dim=0)

def loop(dec, steps=STEPS, collect=False):
    with torch.no_grad():
        past, ids = None, torch.zeros((BEAMS,1), dtype=torch.long, device=dev)
        outs=[]
        for _ in range(steps):
            out = dec(input_ids=ids, past_key_values=past, encoder_hidden_states=h,
                      use_cache=True, return_dict=True)
            past = out.past_key_values
            ids = model.lm_head(out[0])[:, -1:].argmax(-1)
            if collect: outs.append(int(ids[0,0]))
    torch.cuda.synchronize()
    return outs

# ---- baseline ----
loop(model.decoder); ts=[]
for _ in range(3):
    t0=time.time(); loop(model.decoder); ts.append(time.time()-t0)
base = 1000*min(ts)/STEPS
ref_ids = loop(model.decoder, collect=True)
print(f"\n[1] eager decoder loop : {base:6.2f} ms/step")

# ---- inductor ----
import torch._dynamo as dynamo
for label, kw in (("dynamic=True", dict(dynamic=True)), ("default", {})):
    dynamo.reset()
    print(f"\n[2] torch.compile(decoder.forward, backend='inductor', {label})")
    try:
        t0=time.time()
        cdec = torch.compile(model.decoder.forward, **kw)
        loop(cdec)
        print(f"    compile+warmup : {time.time()-t0:6.1f}s")
        ts=[]
        for _ in range(3):
            t0=time.time(); loop(cdec); ts.append(time.time()-t0)
        ms = 1000*min(ts)/STEPS
        got = loop(cdec, collect=True)
        print(f"    compiled       : {ms:6.2f} ms/step   speedup {base/ms:.2f}x")
        print(f"    output identical to eager: {got == ref_ids}")
        cnt = dynamo.utils.counters["stats"]
        print(f"    dynamo stats: {dict(cnt)}")
    except Exception as ex:
        print(f"    FAILED {type(ex).__name__}: {str(ex)[:400]}")
