"""Official TRELLIS.2 image -> GLB timing on ROCm (example.py settings), eager or torch.compile. Offline baseline only.
Usage: python tools/bench_ref.py --mode {eager,compile} --out bench/eager.json [--warm-all] [--images a.png ...] [--glb-dir out/eager]"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refenv
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="eager", choices=["eager", "compile"])
ap.add_argument("--images", nargs="+", default=refenv.IMAGES)
ap.add_argument("--out", required=True)
ap.add_argument("--glb-dir", default=None)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--warm-all", action="store_true", help="also run every image once untimed before the timed pass")
a = ap.parse_args()

pipe = refenv.load_pipeline(math_sdpa=False)  # baseline timing: official code as shipped
import o_voxel.postprocess as PP
if a.mode == "compile":
    torch._dynamo.config.cache_size_limit = 256
    for k, m in pipe.models.items():
        if "flow" in k: m.forward = torch.compile(m.forward, dynamic=True)
    pipe.image_cond_model.model.forward = torch.compile(pipe.image_cond_model.model.forward, dynamic=True)
glb_dir = a.glb_dir or f"out/ref_{a.mode}"
os.makedirs(glb_dir, exist_ok=True)


def sync(): torch.cuda.synchronize(); return time.perf_counter()


def run(name, save=True):
    t = {}; t0 = s = sync()
    mesh = pipe.run(Image.open(os.path.join(refenv.IMG_DIR, name)), seed=42)[0]
    t["pipeline"] = (e := sync()) - s; s = e
    mesh.simplify(16777216)
    glb = PP.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=mesh.layout, voxel_size=mesh.voxel_size, **refenv.GLB_KW)
    t["to_glb"] = (e := sync()) - s; s = e
    if save: glb.export(os.path.join(glb_dir, os.path.splitext(name)[0][:8] + ".glb"), extension_webp=True)
    t["export"] = (e := sync()) - s
    t["total"] = e - t0
    return t


for i in range(a.warmup): print("warmup", run(a.images[0], save=False), flush=True)
if a.warm_all:
    for p in a.images: print("warm", p[:8], round(run(p, save=False)["total"], 2), flush=True)
R = {}
for p in a.images:
    k = os.path.splitext(p)[0][:8]
    R[k] = run(p); print(a.mode, k, {x: round(y, 2) for x, y in R[k].items()}, flush=True)
R["_meta"] = {"warmup": a.warmup, "warm_all": a.warm_all, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "hip": torch.version.hip}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(R, open(a.out, "w"), indent=1)
