"""Is ByT5 decode weight-bandwidth-bound (quantization helps) or per-step
overhead-bound (quantization cannot help)?

Test: force a fixed number of decode steps and vary num_beams. Weight traffic per
step is constant in beams; compute scales with beams. If time is flat in beams,
the step is dominated by fixed overhead (launches / Python), and shrinking the
weights cannot move it.
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
dev = torch.device("cuda"); model.to(dev); model.eval()
dt = torch.bfloat16
T, STEPS = 100, 64
g = torch.Generator().manual_seed(0)
f = torch.randn(1,T,384,generator=g).to(dev,dt); l = torch.randn(1,T,384,generator=g).to(dev,dt)
r = torch.randn(1,T,384,generator=g).to(dev,dt); p = torch.randn(1,T,14,generator=g).to(dev,dt)

def run(beams, steps=STEPS):
    with torch.no_grad():
        # min_length == max_length forces exactly `steps` decoder passes
        ids = model.generate(face_features=f, left_hand_features=l, right_hand_features=r,
                             pose_features=p, max_length=steps, min_length=steps,
                             num_beams=beams, early_stopping=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
    return ids.shape[-1]

print("warmup..."); run(1); run(4)
print(f"\nfixed {STEPS} decode steps, T={T}")
print(f"{'beams':>6}{'total':>9}{'ms/step':>10}{'vs beams=1':>12}")
base = None
for b in (1, 2, 4, 8):
    ts = []
    for _ in range(3):
        t0 = time.time(); n = run(b); torch.cuda.synchronize(); ts.append(time.time()-t0)
    t = min(ts); base = base or t
    print(f"{b:>6}{t:8.2f}s{1000*t/STEPS:9.1f}{t/base:11.2f}x")

# Step-count scaling: is time linear in steps (per-step cost) or is there a fixed base?
print(f"\nnum_beams=4, varying steps")
print(f"{'steps':>6}{'total':>9}{'ms/step':>10}")
for s in (16, 32, 64, 128):
    ts = []
    for _ in range(3):
        t0 = time.time(); run(4, s); torch.cuda.synchronize(); ts.append(time.time()-t0)
    t = min(ts)
    print(f"{s:>6}{t:8.2f}s{1000*t/s:9.1f}")
