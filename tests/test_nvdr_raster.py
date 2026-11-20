"""Rasterizer sanity on ROCm: nvdiffrast (wave64 port) vs a brute-force pixel-center reference."""
import torch, nvdiffrast.torch as dr, time
torch.manual_seed(0)
ctx = dr.RasterizeCudaContext()

def ref_raster(pos, tri, H, W):
    # pixel centers in NDC, depth test on z/w, returns tri id+1 per pixel (0 = empty)
    ys, xs = torch.meshgrid(torch.arange(H, device='cuda'), torch.arange(W, device='cuda'), indexing='ij')
    px = (xs.float() + 0.5) / W * 2 - 1; py = (ys.float() + 0.5) / H * 2 - 1
    ndc = pos[:, :3] / pos[:, 3:]
    best = torch.full((H, W), float('inf'), device='cuda'); ids = torch.zeros((H, W), dtype=torch.int64, device='cuda')
    for t in range(tri.shape[0]):
        a, b, c = ndc[tri[t, 0]], ndc[tri[t, 1]], ndc[tri[t, 2]]
        area = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        w0 = ((b[0]-px)*(c[1]-py) - (b[1]-py)*(c[0]-px)) / area
        w1 = ((c[0]-px)*(a[1]-py) - (c[1]-py)*(a[0]-px)) / area
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z = w0*a[2] + w1*b[2] + w2*c[2]
        upd = inside & (z < best)
        best = torch.where(upd, z, best); ids = torch.where(upd, torch.full_like(ids, t + 1), ids)
    return ids

# 1) random overlapping triangles with depth
for H, W, T in [(256, 256, 64), (512, 384, 200)]:
    pos = torch.rand(T * 3, 4, device='cuda') * 2 - 1; pos[:, 2] = pos[:, 2] * 0.9; pos[:, 3] = 1
    tri = torch.arange(T * 3, device='cuda', dtype=torch.int32).view(T, 3)
    rast, _ = dr.rasterize(ctx, pos[None], tri, (H, W))
    got = rast[0, ..., 3].round().long()
    exp = ref_raster(pos, tri.long(), H, W)
    mism = (got != exp).float().mean().item()
    print(f"random {H}x{W} T={T}: pixel id mismatch {mism*100:.3f}% (edge/fill-rule pixels only expected)")
    assert mism < 0.01

# 2) dense UV-atlas-like mesh at 4096^2 (grid of tiny non-overlapping tris), like o_voxel's texture bake
n = 700
g = torch.stack(torch.meshgrid(torch.linspace(-0.98, 0.98, n + 1), torch.linspace(-0.98, 0.98, n + 1), indexing='ij'), -1).view(-1, 2).cuda()
pos = torch.cat([g, torch.zeros_like(g[:, :1]), torch.ones_like(g[:, :1])], 1)
i = torch.arange(n, device='cuda'); I, J = torch.meshgrid(i, i, indexing='ij'); v = (I * (n + 1) + J).reshape(-1)
tri = torch.cat([torch.stack([v, v + 1, v + n + 1], 1), torch.stack([v + 1, v + n + 2, v + n + 1], 1)]).int()
torch.cuda.synchronize(); t0 = time.time()
rast, _ = dr.rasterize(ctx, pos[None], tri, (4096, 4096)); torch.cuda.synchronize()
print(f"dense grid: {tri.shape[0]} tris @4096^2 in {time.time()-t0:.3f}s")
ids = rast[0, ..., 3]
covered = (ids > 0).float().mean().item(); print(f"coverage {covered:.4f} (expect ~{0.98**2:.4f})")
assert abs(covered - 0.98**2) < 2e-3
# every triangle should own pixels; barycentrics in [0,1]
cnt = torch.bincount(ids.long().view(-1), minlength=tri.shape[0] + 1)[1:]
print("tris with zero pixels:", (cnt == 0).sum().item(), " min/max px per tri:", cnt.min().item(), cnt.max().item())
uv = rast[0, ..., :2][ids > 0]; print("bary range", uv.min().item(), uv.max().item())
# 3) determinism
r2, _ = dr.rasterize(ctx, pos[None], tri, (4096, 4096))
print("deterministic:", torch.equal(rast, r2))
# 4) interpolate + texture + antialias run
attr = torch.rand(pos.shape[0], 3, device='cuda')
out, _ = dr.interpolate(attr[None], rast, tri); print("interp ok", out.shape, out.isfinite().all().item())
tex = torch.rand(1, 64, 64, 3, device='cuda'); t = dr.texture(tex, out[..., :2].contiguous(), filter_mode='linear'); print("texture ok", t.isfinite().all().item())
aa = dr.antialias(out.contiguous(), rast, pos[None], tri); print("antialias ok", aa.isfinite().all().item())
print("ALL OK")
