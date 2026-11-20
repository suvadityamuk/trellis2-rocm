"""Sub-step timing of cumesh uv_unwrap (official) on a dumped mesh."""
import sys, time, torch, numpy as np, cumesh
from cumesh import xatlas as XA
k = sys.argv[1] if len(sys.argv) > 1 else "0a34fae7"
G = torch.load(f"out/ref/{k}/glb.pt", weights_only=False)
v, f = [x.cuda() for x in G["uv_unwrap.in"]]
m = cumesh.CuMesh(); m.init(v, f)
T = {}; s = time.perf_counter()
def tick(n):
    global s; torch.cuda.synchronize(); e = time.perf_counter(); T[n] = e - s; s = e
m.remove_degenerate_faces(); m.compute_charts(threshold_cone_half_angle_rad=np.radians(90.0), refine_iterations=0, global_iterations=1, smooth_strength=1); tick("cumesh.compute_charts")
nv, nf = m.read(); nc, cid, cvm, cf, cvo, cfo = m.read_atlas_charts(); cv = nv[cvm].cpu(); cf = cf.cpu(); cvo = cvo.cpu(); cfo = cfo.cpu(); tick("read charts")
sizes = (cfo[1:] - cfo[:-1]).numpy(); print(f"{nc} charts; faces={sizes.sum()}; >1024: {(sizes > 1024).sum()} holding {sizes[sizes > 1024].sum() / sizes.sum() * 100:.1f}% of faces; max {sizes.max()}")
a = XA.Atlas()
for i in range(nc): a.add_mesh(cv[cvo[i]:cvo[i + 1]], cf[cfo[i]:cfo[i + 1]] - cvo[i])
tick("xatlas add_mesh loop")
a.compute_charts(); tick("xatlas compute_charts")
a.pack_charts(); tick("xatlas pack_charts")
for i in range(nc): a.get_mesh(i)
tick("get_mesh loop")
for n, t in T.items(): print(f"{n:28s} {t:7.2f}s")
