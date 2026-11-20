import sys, time, torch
from t2r.dit import FlowDiT
from t2r.pipeline import CKPT
k = sys.argv[1] if len(sys.argv) > 1 else "0a34fae7"
P = torch.load(f"out/ref/{k}/pipe.pt", weights_only=False)
c = P["dit.shape_slat_flow_model_1024.call0"]
m = FlowDiT(f"{CKPT}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", False)
x = c["x"]["feats"].cuda(); m.set_coords(c["x"]["coords"].cuda()); t = c["t"].cuda(); cond = c["cond"].cuda(); neg = torch.zeros_like(cond)
print("tokens", x.shape[0])
def bench(fn, n=5):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
with torch.no_grad():
    tp = bench(lambda: m.forward_parity(x, t, cond))
    tp2 = bench(lambda: (m.forward_parity(x, t, cond), m.forward_parity(x, t, neg)))
    tf = bench(lambda: m.forward_fast(x, [t.item()], [cond]))
    tf2 = bench(lambda: m.forward_fast(x, [t.item()] * 2, [cond, neg]))
print(f"parity 1 call {tp:.1f} ms | 2 calls (CFG) {tp2:.1f} ms")
print(f"fast   1 call {tf:.1f} ms | batched CFG {tf2:.1f} ms")
from torch.profiler import profile, ProfilerActivity
with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3): m.forward_fast(x, [t.item()] * 2, [cond, neg])
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=70))
