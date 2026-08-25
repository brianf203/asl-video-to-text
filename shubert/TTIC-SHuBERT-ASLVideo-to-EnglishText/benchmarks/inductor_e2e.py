import os, sys, time, torch
sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")

# ---- 0: was the trivial-test mismatch real, or my tolerance? ----
print("[0] re-check the trivial compile numerics properly")
def fn(a,b): return (a*b + a.sin()).relu().sum(-1)
torch.manual_seed(0)
a=torch.randn(512,512,device="cuda",dtype=torch.bfloat16); b=torch.randn(512,512,device="cuda",dtype=torch.bfloat16)
ref=fn(a,b); got=torch.compile(fn)(a,b); torch.cuda.synchronize()
rel=((got.float()-ref.float()).abs()/ref.float().abs().clamp(min=1e-6)).max().item()
print(f"    max |rel err| bf16 sum over 512 elems: {rel:.4f}   allclose(rtol=2e-2): {torch.allclose(got.float(),ref.float(),rtol=2e-2)}")
# fp32 control: if fp32 matches tightly, the bf16 gap is accumulation order, not a codegen bug
def fn32(a,b): return (a*b + a.sin()).relu().sum(-1)
a32,b32=a.float(),b.float()
r32=fn32(a32,b32); g32=torch.compile(fn32)(a32,b32); torch.cuda.synchronize()
print(f"    fp32 control allclose(rtol=1e-5): {torch.allclose(g32,r32,rtol=1e-5)}  max rel {((g32-r32).abs()/r32.abs().clamp(min=1e-6)).max().item():.2e}")

import inference
from transformers import ByT5Tokenizer
MB=("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/"
    "snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
model=inference.SignLanguageByT5ForConditionalGeneration.from_pretrained(
    os.environ.get("BYT5_BF16_CKPT", os.path.join(MB, "checkpoint-11625-bf16")), torch_dtype=torch.bfloat16)
tok=ByT5Tokenizer.from_pretrained(os.path.join(MB,"byt5_base"))
dev,dt=torch.device("cuda"),torch.bfloat16
model.to(dev); model.eval()
T,STEPS,BEAMS=100,64,4
g=torch.Generator().manual_seed(0)
feat=dict(face_features=torch.randn(1,T,384,generator=g).to(dev,dt),
          left_hand_features=torch.randn(1,T,384,generator=g).to(dev,dt),
          right_hand_features=torch.randn(1,T,384,generator=g).to(dev,dt),
          pose_features=torch.randn(1,T,14,generator=g).to(dev,dt))

def gen():
    with torch.no_grad():
        out=model.generate(**feat,max_length=STEPS,min_length=STEPS,num_beams=BEAMS,
                           early_stopping=False,pad_token_id=tok.pad_token_id,
                           eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); return out

print("\n[1] end-to-end generate(), EAGER")
gen(); gen()
ts=[]
for _ in range(4):
    t0=time.time(); ref_ids=gen(); ts.append(time.time()-t0)
eager=min(ts); print(f"    {eager:6.3f}s   {1000*eager/STEPS:6.2f} ms/step")

print("\n[2] end-to-end generate(), decoder compiled with inductor (dynamic=True)")
t0=time.time()
model.decoder.forward = torch.compile(model.decoder.forward, dynamic=True)
gen()
print(f"    compile+warmup (COLD or warm inductor cache): {time.time()-t0:6.1f}s")
gen()
ts=[]
for _ in range(4):
    t0=time.time(); got_ids=gen(); ts.append(time.time()-t0)
comp=min(ts)
print(f"    {comp:6.3f}s   {1000*comp/STEPS:6.2f} ms/step   speedup {eager/comp:.2f}x")
print(f"    decoded ids identical to eager: {torch.equal(ref_ids, got_ids)}")
print(f"    text eager   : {tok.decode(ref_ids[0], skip_special_tokens=True)[:80]!r}")
print(f"    text compiled: {tok.decode(got_ids[0], skip_special_tokens=True)[:80]!r}")
print(f"\n    inductor cache dir: {os.environ.get('TORCHINDUCTOR_CACHE_DIR','/tmp/torchinductor_'+os.environ.get('USER','?'))}")
