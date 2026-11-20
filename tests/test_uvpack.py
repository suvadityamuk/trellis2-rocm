"""Grouped parallel xatlas packing (fast build) vs the official cumesh uv_unwrap output on a dumped mesh."""
import os, sys, time, torch, numpy as np, cumesh, t2r._C as C
import nvdiffrast.torch as dr
k = sys.argv[1] if len(sys.argv) > 1 else "0a34fae7"
G = torch.load(f"out/ref/{k}/glb.pt", weights_only=False)
v, f = [x.cuda() for x in G["uv_unwrap.in"]]
m = cumesh.CuMesh(); m.init(v, f); m.remove_degenerate_faces()
m.compute_charts(threshold_cone_half_angle_rad=np.radians(90.0), refine_iterations=0, global_iterations=1, smooth_strength=1)
nv, nf = m.read(); nc, cid, cvm, cf, cvo, cfo = m.read_atlas_charts()
cv = nv[cvm].cpu().contiguous(); cf = cf.cpu().int().contiguous()
def dens(V, F, U):
    V, F, U = V.double(), F.long(), U.double()
    a3 = torch.linalg.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]).norm(dim=1)
    e1, e2 = U[F[:, 1]] - U[F[:, 0]], U[F[:, 2]] - U[F[:, 0]]; a2 = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).abs()
    return (a2.sum() / a3.sum()).sqrt().item()
for ng in [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "256").split(",")]:
    t = time.perf_counter(); vmap, faces, uvs = C.uv_pack_groups(cv, cf, cvo.cpu(), cfo.cpu(), ng, 0, int(os.environ.get("MAXF", "1024"))); dt = time.perf_counter() - t
    print(f"groups={ng}: {dt:.2f} s, {faces.shape[0]} faces, {vmap.shape[0]} verts, uv range [{uvs.min():.3f}, {uvs.max():.3f}]  "
          f"density x4096 {dens(nv.cpu()[cvm.cpu()][vmap.long()], faces, uvs)*4096:.1f}")
def util(uv, fa):
    ctx = dr.RasterizeCudaContext()
    p = torch.cat([uv.cuda() * 2 - 1, torch.zeros_like(uv[:, :1]).cuda(), torch.ones_like(uv[:, :1]).cuda()], 1)[None]
    r, _ = dr.rasterize(ctx, p, fa.cuda().int(), (4096, 4096)); return (r[0, ..., 3] > 0).float().mean().item()
ov, of, ou = G["uv_unwrap.out"][:3]
print(f"texel coverage @4096^2: official {util(ou, of)*100:.1f}%  grouped {util(uvs, faces)*100:.1f}%")
# texel density: mean sqrt(uv area / surface area) over faces
print(f"texels per unit length (x4096): official {dens(ov, of, ou)*4096:.1f}  grouped {dens(nv.cpu()[cvm.cpu()][vmap.long()], faces, uvs)*4096:.1f}")
