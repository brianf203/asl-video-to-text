"""Eager-only A/B of PYTORCH_NO_CUDA_MEMORY_CACHING. Read at CUDA init, so it must be
set per-process; this script measures one setting and is run twice."""
import os, sys, time, torch
sys.path.insert(0,"/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")
import inference
from transformers import ByT5Tokenizer
MB=("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/"
    "snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
model=inference.SignLanguageByT5ForConditionalGeneration.from_pretrained(
    os.environ.get("BYT5_BF16_CKPT", os.path.join(MB, "checkpoint-11625-bf16")),torch_dtype=torch.bfloat16)
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
        o=model.generate(**feat,max_length=STEPS,min_length=STEPS,num_beams=BEAMS,
                         early_stopping=False,pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); return o
gen(); gen()
ts=[]
for _ in range(5):
    t0=time.time(); ids=gen(); ts.append(time.time()-t0)
flag=os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING","<unset>")
print(f"PYTORCH_NO_CUDA_MEMORY_CACHING={flag:8} "
      f"min {1000*min(ts)/STEPS:7.2f} ms/step   median {1000*sorted(ts)[2]/STEPS:7.2f}   "
      f"text={tok.decode(ids[0],skip_special_tokens=True)[:40]!r}")
