"""Image -> textured GLB with the TRELLIS.2 ROCm port.

Usage: python examples/image_to_glb.py IMAGE [--out out.glb] [--mode fast|parity] [--seed 42] [--texture-size 4096]

IMAGE must be RGBA with the background already removed (background removal is not bundled). With no IMAGE, the
T.png example from the official repository is used. Needs the environment from tools/setup_rocm.sh; TRELLIS2_SRC and
TRELLIS2_CKPT override the default ~/src/TRELLIS.2 and ~/ckpts/TRELLIS.2-4B.
"""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import refenv
refenv.load_env_token()  # DINOv3 is gated: HF_TOKEN from the environment or .env
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import torch
from PIL import Image
import t2r_cumesh as cumesh
from t2r.pipeline import Pipeline
from t2r import glb as G

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("image", nargs="?", default=os.path.join(refenv.IMG_DIR, "T.png"))
ap.add_argument("--out", default="out.glb")
ap.add_argument("--mode", default="fast", choices=["fast", "parity"], help="fast: HIP kernels; parity: bitwise match of the official pipeline")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--texture-size", type=int, default=4096)
ap.add_argument("--decimation-target", type=int, default=1000000, help="faces in the exported mesh")
a = ap.parse_args()


def sync(): torch.cuda.synchronize(); return time.perf_counter()


t0 = time.perf_counter()
pipe = Pipeline(a.mode)
t1 = sync()
m = pipe.run(Image.open(a.image), seed=a.seed)
t2 = sync()
# as the official example: simplify to nvdiffrast's face limit, then remesh / unwrap / bake
cm = cumesh.CuMesh(); cm.init(m["vertices"], m["faces"]); cm.simplify(16777216, verbose=False); v, f = cm.read()
out = G.to_glb(v, f, m["attrs"], m["coords"], m["voxel_size"], mode=a.mode, texture_size=a.texture_size, decimation_target=a.decimation_target)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
if a.mode == "fast": G.write_glb(out, a.out)
else: out.export(a.out, extension_webp=True)
t3 = sync()
print(f"{a.out}: load {t1 - t0:.1f} s, image -> mesh + voxels {t2 - t1:.1f} s, to_glb + export {t3 - t2:.1f} s "
      f"(first run in a process includes one-time kernel and cache warmup)")
