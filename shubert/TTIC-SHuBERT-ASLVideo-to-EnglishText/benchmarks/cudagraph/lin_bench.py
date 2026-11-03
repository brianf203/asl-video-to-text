import torch, time, torch.nn.functional as F
from torchao.quantization import Int8WeightOnlyConfig, quantize_
dev="cuda"; dt=torch.bfloat16
def T(f,n=500):
    for _ in range(20): f()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return 1000*(time.time()-t)/n
for (N,K) in ((3968,1536),(1536,3968),(768,1536)):
    x=torch.randn(4,K,device=dev,dtype=dt)
    lin=torch.nn.Linear(K,N,bias=False).to(dev,dt)
    w=lin.weight.data.clone()
    tb=T(lambda: F.linear(x,w))
    # graph-captured bf16 for fair comparison of launch-free time
    g=torch.cuda.CUDAGraph(); s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): F.linear(x,w)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g): yb=F.linear(x,w)
    tbg=T(g.replay)
    q=torch.nn.Sequential(torch.nn.Linear(K,N,bias=False).to(dev,dt)); q[0].weight.data.copy_(w)
    quantize_(q, Int8WeightOnlyConfig())
    tq=T(lambda: q(x))
    scale=(w.float().abs().amax(1)/127).clamp(min=1e-8)
    wq=(w.float()/scale[:,None]).round().clamp(-128,127).to(torch.int8)
    try:
        tp=T(lambda: torch._weight_int8pack_mm(x,wq,scale.to(dt)))
    except Exception as e:
        tp=f"n/a ({type(e).__name__}: {str(e)[:60]})"
    td=T(lambda: F.linear(x,(wq.to(dt)*scale.to(dt)[:,None])))
    print(f"[{N}x{K}] bf16 {tb:.3f} ms (graph {tbg:.3f}) | torchao int8 {tq:.3f} | _weight_int8pack_mm {tp if isinstance(tp,str) else f'{tp:.3f}'} | naive dequant+mm {td:.3f}  (weight bytes bf16 {N*K*2/2**20:.1f} MB)")
