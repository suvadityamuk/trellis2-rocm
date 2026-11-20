"""TRELLIS.2-4B image -> mesh + PBR voxels (1024_cascade), the official Trellis2ImageTo3DPipeline.run rebuilt on t2r.

mode="parity": bitwise identical to the official pipeline (with DINOv3 attention on PyTorch's MATH SDPA backend; see
tools/refenv.py for why the official reference pins it too).
mode="fast": t2r fused DiT kernels, batched CFG, fp32 guidance math (tolerance-bounded).

The sparse-structure decoder and the SLat VAE decoders are the official modules (they are ~3% of the warm runtime).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dit import FlowDiT
from . import sampler as S

TRELLIS = os.environ.get("TRELLIS2_SRC", os.path.expanduser("~/src/TRELLIS.2"))
CKPT = os.environ.get("TRELLIS2_CKPT", os.path.expanduser("~/ckpts/TRELLIS.2-4B"))

SHAPE_MEAN = [0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667, -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665]
SHAPE_STD = [5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194, 5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691]
TEX_MEAN = [3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075, 0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717, 1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862]
TEX_STD = [2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822, 2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588, 2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999, 2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190]
SS_P = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=(0.6, 1.0), rescale_t=5.0)
SHAPE_P = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5, guidance_interval=(0.6, 1.0), rescale_t=3.0)
TEX_P = dict(steps=12, guidance_strength=1.0, guidance_rescale=0.0, guidance_interval=(0.6, 0.9), rescale_t=3.0)
PBR_LAYOUT = {'base_color': slice(0, 3), 'metallic': slice(3, 4), 'roughness': slice(4, 5), 'alpha': slice(5, 6)}


def preprocess_image(input: Image.Image) -> Image.Image:
    """Trellis2ImageTo3DPipeline.preprocess_image for RGBA inputs (PIL/numpy, bitwise identical)."""
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    if not has_alpha:
        raise ValueError("t2r expects an RGBA image with a meaningful alpha channel (background removal is not bundled)")
    output = input
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return Image.fromarray((output * 255).astype(np.uint8))


class Dino:
    """DinoV3FeatureExtractor.extract_features + final layer_norm (HF transformers modules, MATH SDPA)."""
    def __init__(self, device="cuda"):
        from transformers import DINOv3ViTModel
        import torchvision.transforms as T
        self.model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").to(device).eval()
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def __call__(self, image: Image.Image, res: int) -> torch.Tensor:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        im = image.resize((res, res), Image.LANCZOS)
        im = np.array(im.convert('RGB')).astype(np.float32) / 255
        x = torch.stack([torch.from_numpy(im).permute(2, 0, 1).float()]).cuda()
        x = self.norm(x).cuda()
        m = self.model
        x = x.to(m.embeddings.patch_embeddings.weight.dtype)
        with sdpa_kernel([SDPBackend.MATH]):
            h = m.embeddings(x, bool_masked_pos=None)
            pe = m.rope_embeddings(x)
            for layer in m.layer:
                h = layer(h, position_embeddings=pe)
        return F.layer_norm(h, h.shape[-1:])


class Pipeline:
    def __init__(self, mode="parity", device="cuda"):
        assert mode in ("parity", "fast")
        self.mode = mode
        sys.path.insert(0, TRELLIS)
        from trellis2 import models as tm
        ck = os.path.join(CKPT, "ckpts")
        self.dino = Dino(device)
        self.ss = FlowDiT(f"{ck}/ss_flow_img_dit_1_3B_64_bf16.safetensors", dense=True)
        self.sh512 = FlowDiT(f"{ck}/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", dense=False)
        self.sh1024 = FlowDiT(f"{ck}/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", dense=False)
        self.tex = FlowDiT(f"{ck}/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", dense=False)
        self.ss_dec = tm.from_pretrained("microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16").to(device).eval()
        self.shape_dec = tm.from_pretrained(f"{ck}/shape_dec_next_dc_f16c32_fp16").to(device).eval()
        self.tex_dec = tm.from_pretrained(f"{ck}/tex_dec_next_dc_f16c32_fp16").to(device).eval()
        dev = torch.device(device)
        self.shape_std, self.shape_mean = torch.tensor(SHAPE_STD)[None].to(dev), torch.tensor(SHAPE_MEAN)[None].to(dev)
        self.tex_std, self.tex_mean = torch.tensor(TEX_STD)[None].to(dev), torch.tensor(TEX_MEAN)[None].to(dev)
        self.times = None
        self.taps = None  # dict -> intermediate tensors for tools/parity.py

    def _tap(self, name, t):
        if self.taps is not None:
            self.taps[name] = t.detach().clone()

    def _tick(self, name):
        if self.times is not None:
            torch.cuda.synchronize(); t = time.perf_counter(); self.times[name] = t - self._t; self._t = t

    def _sample(self, model, noise, cond, neg, P, dense=False, concat=None):
        if self.mode == "parity":
            if dense:
                call = lambda x, t, c: model.forward_parity(x, t, c)
            else:
                call = lambda x, t, c: model.forward_parity(x, t, c, concat)
            return S.sample_parity(call, noise, cond, neg, dense=dense, **P)
        if dense:  # token-major [4096, 8]
            x0 = noise.view(noise.shape[1], -1).t().contiguous()
            out = S.sample_fast(lambda x, ts, cs: model.forward_fast(x, ts, cs), x0, cond, neg, dense=True, **P)
            return out.t().reshape_as(noise).contiguous()
        return S.sample_fast(lambda x, ts, cs: model.forward_fast(x, ts, cs, concat), noise, cond, neg, **P)

    @torch.no_grad()
    def run(self, image: Image.Image, seed: int = 42):
        """-> dict(vertices, faces, attrs, coords, voxel_size, layout) as MeshWithVoxel would hold them."""
        from trellis2.modules.sparse import SparseTensor
        self._t = time.perf_counter()
        # The fast DiTs cache cross-attention K/V per condition tensor; the allocator can hand the next image's
        # condition the same address, so the caches must not outlive a run.
        for m in (self.ss, self.sh512, self.sh1024, self.tex):
            m.kv_cache.clear()
        image = preprocess_image(image)
        torch.manual_seed(seed)
        c512 = self.dino(image, 512); c1024 = self.dino(image, 1024)
        n512, n1024 = torch.zeros_like(c512), torch.zeros_like(c1024)
        self._tap("cond512", c512); self._tap("cond1024", c1024)
        self._tick("cond")
        # sparse structure
        noise = torch.randn(1, 8, 16, 16, 16).to("cuda")
        z = self._sample(self.ss, noise, c512, n512, SS_P, dense=True)
        decoded = self.ss_dec(z) > 0
        decoded = F.max_pool3d(decoded.float(), 2, 2, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        self._tap("ss.coords", coords)
        self._tick("sparse_structure")
        # shape SLat cascade: 512 -> upsample -> 1024
        self.sh512.set_coords(coords)
        noise = torch.randn(coords.shape[0], 32).to("cuda")
        slat = self._sample(self.sh512, noise, c512, n512, SHAPE_P)
        self._tap("sampler.1", slat)
        slat = slat * self.shape_std + self.shape_mean
        hr = self.shape_dec.upsample(SparseTensor(feats=slat, coords=coords), upsample_times=4)
        self._tap("hr_coords", hr)
        quant = torch.cat([hr[:, :1], ((hr[:, 1:] + 0.5) / 512 * (1024 // 16)).int()], dim=1)
        coords = quant.unique(dim=0)
        self.sh1024.set_coords(coords)
        noise = torch.randn(coords.shape[0], 32).to("cuda")
        slat = self._sample(self.sh1024, noise, c1024, n1024, SHAPE_P)
        self._tap("sampler.2", slat)
        shape_slat = slat * self.shape_std + self.shape_mean
        self._tick("shape_slat")
        # texture SLat
        sn = (shape_slat - self.shape_mean) / self.shape_std
        self.tex.set_coords(coords)
        noise = torch.randn(coords.shape[0], 64 - 32).to("cuda")
        tex = self._sample(self.tex, noise, c1024, n1024, TEX_P, concat=sn)
        self._tap("sampler.3", tex)
        tex_slat = tex * self.tex_std + self.tex_mean
        self._tick("tex_slat")
        if self.mode == "parity":
            torch.cuda.empty_cache()  # as the official run(); the fast build keeps the cache (hipMalloc is slow)
        # decode (official modules)
        self.shape_dec.set_resolution(1024)
        meshes, subs = self.shape_dec(SparseTensor(feats=shape_slat, coords=coords), return_subs=True)
        tex_vox = self.tex_dec(SparseTensor(feats=tex_slat, coords=coords), guide_subs=subs) * 0.5 + 0.5
        m = meshes[0]
        m.fill_holes()
        self._tick("decode")
        return dict(vertices=m.vertices, faces=m.faces, attrs=tex_vox.feats, coords=tex_vox.coords[:, 1:],
                    voxel_size=1 / 1024, layout=PBR_LAYOUT)
