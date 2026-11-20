"""Stage-level timing of the official pipeline on ROCm (warm). Wraps pipeline methods and to_glb building blocks.
Usage: python tools/profile_ref.py [image] [--kernels]"""
import os, sys, time, collections, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refenv
import torch
from PIL import Image

img = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "T.png"
pipe = refenv.load_pipeline(math_sdpa=False)  # baseline timing: official code as shipped
import o_voxel.postprocess as PP, cumesh, cv2, nvdiffrast.torch as dr

T = collections.defaultdict(float); N = collections.defaultdict(int); depth = [0]


def timed(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        if depth[0]: return fn(*a, **k)  # only time outermost
        depth[0] += 1; torch.cuda.synchronize(); s = time.perf_counter()
        try: return fn(*a, **k)
        finally:
            torch.cuda.synchronize(); T[name] += time.perf_counter() - s; N[name] += 1; depth[0] -= 1
    return w


P = type(pipe)
for m in ["get_cond", "sample_sparse_structure", "sample_shape_slat", "sample_shape_slat_cascade", "sample_tex_slat", "decode_latent"]:
    setattr(P, m, timed("pipe." + m, getattr(P, m)))
for m in ["init", "fill_holes", "simplify", "uv_unwrap", "compute_vertex_normals", "read"]:
    setattr(cumesh.CuMesh, m, timed("cumesh." + m, getattr(cumesh.CuMesh, m)))
cumesh.cuBVH.__init__ = timed("cuBVH.build", cumesh.cuBVH.__init__)
cumesh.cuBVH.unsigned_distance = timed("cuBVH.udf", cumesh.cuBVH.unsigned_distance)
cumesh.remeshing.remesh_narrow_band_dc = timed("remesh_dc", cumesh.remeshing.remesh_narrow_band_dc)
PP.cumesh = cumesh
PP.cv2 = type("cv2w", (), {"inpaint": staticmethod(timed("cv2.inpaint", cv2.inpaint)), "INPAINT_TELEA": cv2.INPAINT_TELEA})
PP.dr = type("drw", (), {"RasterizeCudaContext": dr.RasterizeCudaContext, "rasterize": staticmethod(timed("dr.rasterize", dr.rasterize)),
                         "interpolate": staticmethod(timed("dr.interpolate", dr.interpolate))})
PP.grid_sample_3d = timed("grid_sample_3d", PP.grid_sample_3d)


def run():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    mesh = pipe.run(Image.open(os.path.join(refenv.IMG_DIR, img)), seed=42)[0]
    torch.cuda.synchronize(); T["== pipeline"] += time.perf_counter() - t0; s = time.perf_counter()
    mesh.simplify(16777216)
    glb = PP.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=mesh.layout, voxel_size=mesh.voxel_size, **refenv.GLB_KW)
    torch.cuda.synchronize(); T["== to_glb"] += time.perf_counter() - s; s = time.perf_counter()
    glb.export("/tmp/_prof.glb", extension_webp=True)
    T["== export"] += time.perf_counter() - s


run()  # warmup (autotune, compile)
T.clear(); N.clear()
if "--kernels" in sys.argv:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=90))
else:
    run()
for k in sorted(T, key=lambda k: -T[k]):
    print(f"{k:40s} {T[k]:8.2f}s  x{N[k]}")
