"""Dump stage inputs/outputs of the official TRELLIS.2 pipeline (ROCm) for bitwise parity checks. Offline only.
Usage: python tools/ref_dump.py [--images ...] [--out out/ref] [--no-glb]
Writes out/ref/<img8>/pipe.pt (pipeline stages) and glb.pt (to_glb stages) plus ref.glb."""
import os, sys, argparse, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refenv
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--images", nargs="+", default=refenv.IMAGES)
ap.add_argument("--out", default="out/ref")
ap.add_argument("--no-glb", action="store_true")
ap.add_argument("--calls", type=int, default=2, help="DiT calls to record per flow model (teacher forcing)")
a = ap.parse_args()

pipe = refenv.load_pipeline()
import o_voxel.postprocess as PP, cumesh
D = {}


def cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
    if hasattr(x, "feats") and hasattr(x, "coords"): return {"feats": cpu(x.feats), "coords": cpu(x.coords)}
    if hasattr(x, "vertices") and hasattr(x, "faces"): return {"v": cpu(x.vertices), "f": cpu(x.faces)}
    if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [cpu(v) for v in x]
    return x


def wrap(obj, name, key, save_in=None):
    fn = getattr(obj, name)
    @functools.wraps(fn)
    def w(*args, **kw):
        out = fn(*args, **kw)
        if save_in: D.update({f"{key}.in.{k}": cpu(v) for k, v in save_in(*args, **kw).items()})
        D[f"{key}.out"] = cpu(out)
        return out
    setattr(obj, name, w)


P = type(pipe)
orig = {m: getattr(P, m) for m in ["preprocess_image", "get_cond", "sample_sparse_structure", "decode_shape_slat", "decode_tex_slat"]}


def inst(m, key_fn):
    f = orig[m]
    def w(self, *args, **kw):
        out = f(self, *args, **kw)
        D[key_fn(*args, **kw)] = cpu(out if not isinstance(out, Image.Image) else np.array(out))
        return out
    setattr(P, m, w)


inst("preprocess_image", lambda *a, **k: "preprocess.out")
inst("get_cond", lambda img, res, *a, **k: f"cond{res}")
inst("sample_sparse_structure", lambda *a, **k: "ss.coords")
inst("decode_shape_slat", lambda *a, **k: "dec_shape.out")
inst("decode_tex_slat", lambda *a, **k: "dec_tex.out")

# randn draws (CPU generator), in order
_randn = torch.randn
def randn_rec(*args, **kw):
    out = _randn(*args, **kw); D.setdefault("randn", []).append(out.clone()); return out
torch.randn = randn_rec

# sampler outputs per stage + first DiT calls (teacher-forcing inputs/outputs)
for sname in ["sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"]:
    s = getattr(pipe, sname)
    fs = s.sample
    def sample_w(model, noise, *args, _fs=fs, **kw):
        out = _fs(model, noise, *args, **kw)
        k = len([x for x in D if x.startswith("sampler.") and x.endswith(".samples")])
        D[f"sampler.{k}.samples"] = cpu(out.samples)
        return out
    s.sample = sample_w

for mname, m in pipe.models.items():
    if "flow" not in mname: continue
    f = m.forward
    def fwd(x, t, cond, *args, _f=f, _n=mname, **kw):
        out = _f(x, t, cond, *args, **kw)
        k = D.setdefault(f"_ncall.{_n}", 0)
        if k < a.calls:
            D[f"dit.{_n}.call{k}"] = {"x": cpu(x), "t": cpu(t), "cond": cpu(cond), "kw": cpu(kw), "out": cpu(out)}
        D[f"_ncall.{_n}"] = k + 1
        return out
    m.forward = fwd

hr = pipe.models["shape_slat_decoder"]
fu = hr.upsample
def up_w(*args, **kw):
    out = fu(*args, **kw); D["shape.hr_coords"] = cpu(out); return out
hr.upsample = up_w

G = {}
_init = cumesh.CuMesh.init
def init_w(self, v, f): G.setdefault("cumesh.init", []).append((cpu(v), cpu(f))); return _init(self, v, f)
cumesh.CuMesh.init = init_w
_simp = cumesh.CuMesh.simplify
def simp_w(self, *args, **kw):
    out = _simp(self, *args, **kw); G.setdefault("simplify.out", []).append(cpu(self.read())); return out
cumesh.CuMesh.simplify = simp_w
_uv = cumesh.CuMesh.uv_unwrap
def uv_w(self, *args, **kw):
    G["uv_unwrap.in"] = cpu(self.read()); out = _uv(self, *args, **kw); G["uv_unwrap.out"] = cpu(out); return out
cumesh.CuMesh.uv_unwrap = uv_w
_gs = PP.grid_sample_3d
def gs_w(*args, **kw):
    out = _gs(*args, **kw); G["grid_sample.in"] = cpu(kw); G["grid_sample.out"] = cpu(out); return out
PP.grid_sample_3d = gs_w
import cv2
_inp = cv2.inpaint
def inp_w(img, mask, r, flag):
    out = _inp(img, mask, r, flag); G.setdefault("inpaint", []).append((img.copy(), mask.copy(), r, out.copy())); return out
PP.cv2 = type("cv2w", (), {"inpaint": staticmethod(inp_w), "INPAINT_TELEA": cv2.INPAINT_TELEA})

for name in a.images:
    k = os.path.splitext(name)[0][:8]; od = os.path.join(a.out, k); os.makedirs(od, exist_ok=True)
    D.clear(); G.clear()
    mesh = pipe.run(Image.open(os.path.join(refenv.IMG_DIR, name)), seed=42)[0]
    D["final"] = {"v": cpu(mesh.vertices), "f": cpu(mesh.faces), "attrs": cpu(mesh.attrs), "coords": cpu(mesh.coords)}
    torch.save({x: y for x, y in D.items() if not x.startswith("_")}, os.path.join(od, "pipe.pt"))
    print(k, "pipe dumped", {x: (tuple(y.shape) if isinstance(y, torch.Tensor) else type(y).__name__) for x, y in D.items() if not x.startswith("_")}, flush=True)
    if a.no_glb: continue
    mesh.simplify(16777216)
    G["to_glb.in"] = {"v": cpu(mesh.vertices), "f": cpu(mesh.faces)}
    glb = PP.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=mesh.layout, voxel_size=mesh.voxel_size, **refenv.GLB_KW)
    glb.export(os.path.join(od, "ref.glb"), extension_webp=True)
    G["glb.vertices"] = np.asarray(glb.vertices); G["glb.faces"] = np.asarray(glb.faces)
    torch.save(G, os.path.join(od, "glb.pt"))
    print(k, "glb dumped", flush=True)
