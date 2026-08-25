import os, sys, torch, shutil
sys.path.insert(0, "/home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText")
os.environ["BYT5_DEVICE"] = "cpu"
MB = ("/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/"
      "snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models")
CKPT, TOK = os.path.join(MB, "checkpoint-11625"), os.path.join(MB, "byt5_base")
OUT = sys.argv[1]
def avail(): return int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0])/1048576
import inference
print(f"avail before: {avail():.2f} Gi")
model, tok, _ = inference._get_cached_model(CKPT, TOK, "/tmp")
print(f"avail after load: {avail():.2f} Gi  dtype={next(model.parameters()).dtype}")
os.makedirs(OUT, exist_ok=True)
# safe_serialization=False: lm_head/decoder_emb are tied, and safetensors refuses shared storage
model.save_pretrained(OUT, safe_serialization=False)
print("saved ->", OUT)
print(f"avail after save: {avail():.2f} Gi")
