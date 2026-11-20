"""t2r grid-sample kernels vs FlexGEMM (official) on dumped to_glb inputs."""
import sys, torch, t2r._C as C
from flex_gemm import kernels
from flex_gemm.ops import utils as fu
from flex_gemm.ops.grid_sample import grid_sample_3d
k = sys.argv[1] if len(sys.argv) > 1 else "0a34fae7"
P = torch.load(f"out/ref/{k}/pipe.pt", weights_only=False); G = torch.load(f"out/ref/{k}/glb.pt", weights_only=False)
feats = P["final"]["attrs"].cuda(); coords = P["final"]["coords"].cuda()
kw = G["grid_sample.in"]; grid = kw["grid"].cuda(); shape = kw["shape"]
c4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1).int()
ref = G["grid_sample.out"].cuda()
again = grid_sample_3d(feats, c4, shape=shape, grid=grid, mode='trilinear')
print("official rerun bitwise == dump:", torch.equal(again, ref))
_, Cc, W, H, D = shape
hk, hv = fu.init_hashmap(shape, int(2.0 * c4.shape[0]), c4.device)
idx, w = kernels.cuda.hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(hk, hv, c4, grid, W, H, D)
for fused in (False, True):
    o = C.weighted_sum8(feats, idx.view(-1, 8), w.view(-1, 8), fused).view_as(ref)
    print(f"weighted_sum8 fused={fused}: bitwise={torch.equal(o, ref)}  maxdiff={(o - ref).abs().max().item():.3e}")
o = C.grid_sample_trilinear(feats, coords.int(), grid, W, H, D).view_as(ref)
print(f"fast fused trilinear: maxdiff={(o - ref).abs().max().item():.3e} mean={(o - ref).abs().mean().item():.3e}")
import time
for name, fn in [("official", lambda: grid_sample_3d(feats, c4, shape=shape, grid=grid, mode='trilinear')),
                 ("t2r fast", lambda: C.grid_sample_trilinear(feats, coords.int(), grid, W, H, D))]:
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(3): fn()
    torch.cuda.synchronize(); print(f"{name}: {(time.perf_counter() - t) / 3 * 1e3:.2f} ms  (M={grid.shape[1]})")
