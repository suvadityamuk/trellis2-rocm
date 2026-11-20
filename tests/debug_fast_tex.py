"""Bisect the fast-path texture mismatch on dumped official stage inputs."""
import sys, torch, numpy as np, trimesh
sys.path.insert(0, "tools"); import refenv
import t2r.glb as G, t2r._C as C
k = "T"
P = torch.load(f"out/ref/{k}/pipe.pt", weights_only=False); R = torch.load(f"out/ref/{k}/glb.pt", weights_only=False)
attrs, coords = P["final"]["attrs"].cuda(), P["final"]["coords"].cuda()
aabb, vs, gs = G.grid_info(coords, 1 / 1024)
rv, rf, ruv, rvm = [x.cuda() for x in R["uv_unwrap.out"]]
v, f = R["to_glb.in"]["v"].cuda(), R["to_glb.in"]["f"].cuda()
import cumesh
m = cumesh.CuMesh(); m.init(v, f); m.fill_holes(max_hole_perimeter=3e-2); v0, f0 = m.read()
bvh_ref = cumesh.cuBVH(v0, f0); bvh_fast = G.LBVHAdapter(v0, f0)
a_par, mask = G.stage_bake(rv, rf.int(), ruv, v0, f0, bvh_ref, attrs, coords, aabb, vs, gs, 4096, fast=False)
a_fast, mask2 = G.stage_bake(rv, rf.int(), ruv, v0, f0, bvh_fast, attrs, coords, aabb, vs, gs, 4096, fast=True)
print("masks equal:", torch.equal(mask, mask2), " bake attrs max|d|:", (a_par - a_fast)[mask].abs().max().item(), "mean", (a_par - a_fast)[mask].abs().mean().item())
base, mr = C.bake_textures(a_fast, mask, 3, 1)
b = base.cpu().numpy(); ref_bc = R["inpaint"][0][3]
print("fast base rgb vs official inpainted base on known texels: mean|d|", np.abs(b[..., :3].astype(int) - ref_bc.astype(int))[mask.cpu().numpy()].mean())
