"""t2r LBVH (fast build) vs cumesh.cuBVH unsigned distance on a dumped TRELLIS.2 mesh."""
import sys, time, torch, cumesh, t2r._C as C
k = sys.argv[1] if len(sys.argv) > 1 else "0a34fae7"
G = torch.load(f"out/ref/{k}/glb.pt", weights_only=False)
v, f = [x.cuda() for x in G["cumesh.init"][0]]
print(f"mesh: {v.shape[0]} verts, {f.shape[0]} faces")
torch.cuda.synchronize(); t = time.perf_counter(); ref = cumesh.cuBVH(v, f); torch.cuda.synchronize(); tb_ref = time.perf_counter() - t
t = time.perf_counter(); bvh = C.LBVH(v, f.int()); torch.cuda.synchronize(); tb = time.perf_counter() - t
t = time.perf_counter(); bvh = C.LBVH(v, f.int()); torch.cuda.synchronize(); tb2 = time.perf_counter() - t
print(f"build: cuBVH {tb_ref:.2f} s, LBVH {tb*1e3:.1f} ms (warm {tb2*1e3:.1f} ms)")
g = torch.Generator(device="cuda").manual_seed(0)
lo, hi = v.min(0).values, v.max(0).values
for name, pts in [("random in bbox", lo + (hi - lo) * torch.rand(2_000_000, 3, device="cuda", generator=g)),
                  ("near surface", v[torch.randint(0, v.shape[0], (2_000_000,), device="cuda", generator=g)] + 2e-3 * torch.randn(2_000_000, 3, device="cuda", generator=g))]:
    d0, f0, u0 = ref.unsigned_distance(pts, return_uvw=True); torch.cuda.synchronize()
    t = time.perf_counter(); d1, f1, u1 = bvh.query(pts, True); torch.cuda.synchronize(); tq = time.perf_counter() - t
    t = time.perf_counter(); ref.unsigned_distance(pts, return_uvw=True); torch.cuda.synchronize(); tq0 = time.perf_counter() - t
    same_d = (d0 == d1).float().mean().item()
    print(f"{name}: dist bitwise-equal {same_d*100:.3f}%, max |dd| {(d0 - d1).abs().max().item():.2e}, face agree {(f0 == f1).float().mean().item()*100:.2f}%  "
          f"| query {tq*1e3:.1f} ms vs cuBVH {tq0*1e3:.1f} ms")
    # closest points must agree even when tied faces differ
    q0 = (v[f[f0].long()] * u0[..., None]).sum(1); q1 = (v[f[f1].long()] * u1[..., None]).sum(1)
    print(f"   closest-point max diff {(q0 - q1).norm(dim=1).max().item():.2e}")
    dq = (q0 - q1).norm(dim=1); dd = (d0 - d1).abs()
    for th in (1e-6, 1e-4, 1e-3):
        print(f"   frac closest-point diff > {th:g}: {(dq > th).float().mean().item()*100:.4f}%")
    bad = dq > 1e-3
    if bad.any():
        i = bad.nonzero()[0, 0]
        print("   example: d_ref", d0[i].item(), "d_lbvh", d1[i].item(), "|p-q_ref|", (pts[i] - q0[i]).norm().item(), "|p-q_lbvh|", (pts[i] - q1[i]).norm().item())
