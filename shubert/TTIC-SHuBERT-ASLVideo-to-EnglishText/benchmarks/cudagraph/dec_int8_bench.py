import os, sys, time, glob
os.environ["BYT5_CUDAGRAPH"]="0"
ROOT="/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch, inference, run_eval, byt5_cudagraph as bc
cfg=run_eval.config
model,tok,dev=inference._get_cached_model(cfg["slt_model_checkpoint"],cfg["slt_tokenizer_checkpoint"],cfg["temp_dir"])
dt=torch.bfloat16
z=np.load(sorted(glob.glob(ROOT+"/benchmarks/quant/features/*.npz"))[1])
inp={k:torch.tensor(z[s],dtype=dt).unsqueeze(0).to(dev) for k,s in (("face_features","face"),("left_hand_features","left"),("right_hand_features","right"),("pose_features","body"))}
def T(f,n=200):
    for _ in range(10): f()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return 1000*(time.time()-t)/n
def bench(tag):
    with torch.no_grad():
        enc=model.encoder(**inp,return_dict=True)[0]
        sd=bc.StaticDecoder(model,4,768); sd.prepare(enc)
        out={}
        for ld in (64,128,256):
            sd._capture(sd.cur, ld); g,o=sd.graphs[(sd.cur,ld)]
            sd.ids.fill_(65); sd.t.fill_(min(40,ld-1))
            out[ld]=T(g.replay)
            g.replay(); torch.cuda.synchronize(); out[f"logits{ld}"]=o.float().clone()
        dec_mb=sum(p.numel()*p.element_size() for n,p in model.decoder.named_parameters())/2**20
        print(f"{tag}: replay ms/step " + "  ".join(f"ld{ld} {out[ld]:.2f}" for ld in (64,128,256)), flush=True)
        return out
a=bench("bf16 decoder")
from torchao.quantization import Int8WeightOnlyConfig, quantize_
def dec_lin(m,fqn): return isinstance(m,torch.nn.Linear) and fqn.startswith("decoder.")
n=sum(1 for f,m in model.named_modules() if dec_lin(m,f))
quantize_(model, Int8WeightOnlyConfig(), filter_fn=dec_lin); torch.cuda.synchronize(); torch.cuda.empty_cache()
print(f"quantized {n} decoder Linears; cuda alloc {torch.cuda.memory_allocated()/2**20:.0f} MB", flush=True)
b=bench("int8 decoder")
for ld in (64,128,256):
    d=(a[f"logits{ld}"]-b[f"logits{ld}"]).abs()
    print(f"ld{ld}: logits max abs diff {d.max():.3f} mean {d.mean():.4f}; argmax agree {(a[f'logits{ld}'].argmax(-1)==b[f'logits{ld}'].argmax(-1)).float().mean():.2f}")
