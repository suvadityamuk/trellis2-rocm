"""Shared setup for running the official TRELLIS.2 pipeline on ROCm (offline reference / baselines only)."""
import os, sys

TRELLIS = os.environ.get("TRELLIS2_SRC", os.path.expanduser("~/src/TRELLIS.2"))
CKPT = os.environ.get("TRELLIS2_CKPT", os.path.expanduser("~/ckpts/TRELLIS.2-4B"))
IMG_DIR = os.path.join(TRELLIS, "assets/example_image")
IMAGES = ["0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp",
          "0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp",
          "0f168a4b1b6e96c72e9627c97a212c27a4572250ff58e25703b9d0c2bc74191a.webp",
          "130c2b18f1651a70f8aa15b2c99f8dba29bb943044d92871f9223bd3e989e8b1.webp",
          "T.png"]
GLB_KW = dict(aabb=[[-0.5] * 3, [0.5] * 3], decimation_target=1000000, texture_size=4096,
              remesh=True, remesh_band=1, remesh_project=0)


def load_env_token():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(p) and "HF_TOKEN" not in os.environ:
        for line in open(p):
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()


def load_pipeline(math_sdpa=None):
    if math_sdpa is None: math_sdpa = os.environ.get("T2R_REF_MATH_SDPA", "1") == "1"
    load_env_token()
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    sys.path.insert(0, TRELLIS)
    from trellis2.pipelines import rembg, Trellis2ImageTo3DPipeline

    class _NoRembg:  # example images carry alpha, so preprocess_image never calls rembg; skip the gated download
        def __init__(self, *a, **k): pass
        def to(self, *a, **k): return self
        def cuda(self): return self
        def cpu(self): return self
        def __call__(self, *a, **k): raise RuntimeError("rembg not available; use RGBA inputs")
    rembg.BiRefNet = _NoRembg
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(CKPT)
    pipe.low_vram = False
    pipe.cuda()
    if math_sdpa:
        # AOTriton's fp32 memory-efficient SDPA on gfx950 (torch 2.13 / ROCm 10.1) intermittently returns wrong results
        # for >2k tokens (DINOv3 at 1024^2 has 4101), so the parity reference pins DINOv3 attention to the MATH backend.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ex = pipe.image_cond_model.extract_features
        def extract_math(image, _ex=ex):
            with sdpa_kernel([SDPBackend.MATH]):
                return _ex(image)
        pipe.image_cond_model.extract_features = extract_math
    return pipe
