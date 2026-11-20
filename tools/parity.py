"""Per-stage bitwise checks of t2r (parity mode) against official dumps from tools/ref_dump.py.
Each to_glb stage is fed the official stage inputs, so one mismatch cannot hide behind another.
Usage: python tools/parity.py <stage>... [--images k1 k2] [--ref out/ref]
stages: pipe remesh simplify unwrap bake textures"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import refenv
refenv.load_env_token()
sys.path.insert(0, refenv.TRELLIS)

ap = argparse.ArgumentParser()
ap.add_argument("stages", nargs="+")
ap.add_argument("--images", nargs="+", default=[os.path.splitext(x)[0][:8] for x in refenv.IMAGES])
ap.add_argument("--ref", default="out/ref")
a = ap.parse_args()
FILES = {os.path.splitext(x)[0][:8]: x for x in refenv.IMAGES}
RES = []


def check(img, name, got, ref):
    got = got.detach().cpu() if isinstance(got, torch.Tensor) else torch.as_tensor(np.ascontiguousarray(got))
    ref = ref.detach().cpu() if isinstance(ref, torch.Tensor) else torch.as_tensor(np.ascontiguousarray(ref))
    if got.shape != ref.shape: s = f"SHAPE {tuple(got.shape)} vs {tuple(ref.shape)}"
    elif got.dtype == ref.dtype and torch.equal(got, ref): s = "bitwise"
    else:
        d = (got.double() - ref.double()).abs(); s = f"DIFF n={int((d > 0).sum())}/{d.numel()} max={d.max().item():.3e}"
    RES.append((img, name, s)); print(f"  {img} {name:28s} {s}", flush=True)
    return s == "bitwise"


pipe = None
import t2r.glb as G
cumesh_mod = G.cumesh
for img in a.images:
    P = torch.load(f"{a.ref}/{img}/pipe.pt", weights_only=False) if "pipe" in a.stages else None
    R = torch.load(f"{a.ref}/{img}/glb.pt", weights_only=False) if set(a.stages) - {"pipe"} else None
    if "pipe" in a.stages:
        from PIL import Image
        from t2r.pipeline import Pipeline
        if pipe is None: pipe = Pipeline("parity")
        taps = {}
        pipe.taps = taps
        out = pipe.run(Image.open(os.path.join(refenv.IMG_DIR, FILES[img])), seed=42)
        check(img, "cond512", taps["cond512"], P["cond512"]["cond"])
        check(img, "cond1024", taps["cond1024"], P["cond1024"]["cond"])
        check(img, "ss.coords", taps["ss.coords"], P["ss.coords"])
        check(img, "shape512 samples", taps["sampler.1"], P["sampler.1.samples"]["feats"])
        check(img, "hr_coords", taps["hr_coords"], P["shape.hr_coords"])
        check(img, "shape1024 samples", taps["sampler.2"], P["sampler.2.samples"]["feats"])
        check(img, "tex samples", taps["sampler.3"], P["sampler.3.samples"]["feats"])
        check(img, "final.v", out["vertices"], P["final"]["v"])
        check(img, "final.f", out["faces"], P["final"]["f"])
        check(img, "final.attrs", out["attrs"], P["final"]["attrs"])
        check(img, "final.coords", out["coords"], P["final"]["coords"])
    if R is None: continue
    final = torch.load(f"{a.ref}/{img}/pipe.pt", weights_only=False)["final"]
    attrs, coords = final["attrs"].cuda(), final["coords"].cuda()
    aabb, vs, gs = G.grid_info(coords, 1 / 1024)
    if "remesh" in a.stages or "bake" in a.stages:
        v, f = R["to_glb.in"]["v"].cuda(), R["to_glb.in"]["f"].cuda()
        mesh, v0, f0, bvh = G.stage_remesh(v, f, gs, aabb, fast=False)
        if "remesh" in a.stages:
            rv, rf = mesh.read()
            check(img, "remesh.v", rv, R["cumesh.init"][3][0]); check(img, "remesh.f", rf, R["cumesh.init"][3][1])
    if "simplify" in a.stages:
        cumesh = G.cumesh
        outs = []
        for _ in range(2):
            m = cumesh.CuMesh(); m.init(*[x.cuda() for x in R["cumesh.init"][3]]); m.simplify(1000000, verbose=False); outs.append(m.read())
        check(img, "simplify(run1).v", outs[0][0], R["simplify.out"][1][0])
        check(img, "simplify run1 vs run2 .v", outs[0][0], outs[1][0])
    if "unwrap" in a.stages:
        cumesh = G.cumesh
        m = cumesh.CuMesh(); m.init(*[x.cuda() for x in R["uv_unwrap.in"]])
        ov, of, ouv, on = G.stage_unwrap(m, fast=False)
        rv, rf, ruv, rvm = R["uv_unwrap.out"]
        check(img, "unwrap.v", ov, rv); check(img, "unwrap.f", of, rf); check(img, "unwrap.uv", ouv, ruv)
    if "bake" in a.stages:
        rv, rf, ruv, rvm = [x.cuda() for x in R["uv_unwrap.out"]]
        att, mask = G.stage_bake(rv, rf.int(), ruv, v0, f0, bvh, attrs, coords, aabb, vs, gs, 4096, fast=False)
        check(img, "bake.texel_count", torch.tensor(int(mask.sum())), torch.tensor(R["grid_sample.in"]["grid"].shape[1]))
        check(img, "bake.attrs[mask]", att[mask], R["grid_sample.out"][0])
    if "textures" in a.stages:
        (bc, minv, _, bco), (me, _, _, meo), (ro, _, _, roo), (al, _, _, alo) = R["inpaint"]
        # feed the official uint8 inputs through the same finishing code path (inpaints in parallel)
        import cv2
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(4) as ex:
            outs = [ex.submit(cv2.inpaint, x, minv, r, cv2.INPAINT_TELEA) for x, r in [(bc, 3), (me, 1), (ro, 1), (al, 1)]]
            outs = [o.result() for o in outs]
        for nm, o, r in zip(["base", "metallic", "roughness", "alpha"], outs, [bco, meo, roo, alo]):
            check(img, f"inpaint.{nm}", o, r)

nb = sum(r[2] == "bitwise" for r in RES)
print(f"\n{nb}/{len(RES)} checks bitwise")
