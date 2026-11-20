"""t2r image -> GLB timing, same protocol as bench_ref.py (warm process, warmup, model loading excluded).
Usage: python tools/bench_t2r.py --mode {parity,fast} --out bench/fast.json [--warm-all] [--images ...] [--glb-dir out/t2r_fast]"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refenv
refenv.load_env_token()
import torch
from PIL import Image
from t2r.pipeline import Pipeline
from t2r import glb as G

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="fast", choices=["parity", "fast"])
ap.add_argument("--images", nargs="+", default=refenv.IMAGES)
ap.add_argument("--out", required=True)
ap.add_argument("--glb-dir", default=None)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--warm-all", action="store_true", help="also run every image once untimed before the timed pass")
ap.add_argument("--stages", action="store_true", help="record per-stage times")
a = ap.parse_args()

pipe = Pipeline(a.mode)
glb_dir = a.glb_dir or f"out/t2r_{a.mode}"
os.makedirs(glb_dir, exist_ok=True)


def sync(): torch.cuda.synchronize(); return time.perf_counter()


def run(name, save=True):
    t = {}; st = {}; t0 = s = sync()
    if a.stages: pipe.times = st
    m = pipe.run(Image.open(os.path.join(refenv.IMG_DIR, name)), seed=42)
    t["pipeline"] = (e := sync()) - s; s = e
    # the official example simplifies to nvdiffrast's limit before to_glb
    import t2r_cumesh as cumesh
    cm = cumesh.CuMesh(); cm.init(m["vertices"], m["faces"]); cm.simplify(16777216, verbose=False); v, f = cm.read()
    out = G.to_glb(v, f, m["attrs"], m["coords"], m["voxel_size"], mode=a.mode, times=st if a.stages else None)
    t["to_glb"] = (e := sync()) - s; s = e
    path = os.path.join(glb_dir, os.path.splitext(name)[0][:8] + ".glb")
    if save:
        if a.mode == "fast": G.write_glb(out, path)
        else: out.export(path, extension_webp=True)
    t["export"] = (e := sync()) - s
    t["total"] = e - t0
    if a.stages: t["stages"] = st
    pipe.times = None
    return t


for i in range(a.warmup): print("warmup", {k: v for k, v in run(a.images[0], save=False).items() if k != "stages"}, flush=True)
if a.warm_all:
    for p in a.images: print("warm", p[:8], round(run(p, save=False)["total"], 2), flush=True)
R = {}
for p in a.images:
    k = os.path.splitext(p)[0][:8]
    R[k] = run(p); print(a.mode, k, {x: (round(y, 2) if isinstance(y, float) else {q: round(w, 2) for q, w in y.items()}) for x, y in R[k].items()}, flush=True)
R["_meta"] = {"warmup": a.warmup, "warm_all": a.warm_all, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "hip": torch.version.hip}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(R, open(a.out, "w"), indent=1)
