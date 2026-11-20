"""Find the nondeterministic op in DINOv3 @1024 (fp32) on ROCm, using a real preprocessed image."""
import os, sys, torch, numpy as np
from PIL import Image
sys.path.insert(0, "tools"); import refenv; refenv.load_env_token()
from transformers import DINOv3ViTModel
import torchvision.transforms as T
m = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").cuda().eval()
P = torch.load("out/ref/0a34fae7/pipe.pt", weights_only=False)
img = Image.fromarray(P["preprocess.out"])
def inp(res):
    i = img.resize((res, res), Image.LANCZOS); a = np.array(i.convert("RGB")).astype(np.float32) / 255
    t = torch.from_numpy(a).permute(2, 0, 1)[None].cuda()
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t)
calls = []
def hook(name):
    def h(mod, i, o): calls.append((name, (o[0] if isinstance(o, tuple) else o).detach().clone()))
    return h
for n, mod in m.named_modules():
    if n.startswith("layer.") and n.count(".") >= 2: mod.register_forward_hook(hook(n))
runs = []
with torch.no_grad():
    for res in [1024]:
        x = inp(res)
        for _ in range(3):
            calls.clear()
            h = m.embeddings(x, bool_masked_pos=None); pe = m.rope_embeddings(x)
            for l in m.layer: h = l(h, position_embeddings=pe)
            runs.append(list(calls))
for i, (n, v) in enumerate(runs[0]):
    if not all(torch.equal(v, r[i][1]) for r in runs[1:]):
        print(f"first nondeterministic call #{i}: {n}  maxdiff {(v - runs[1][i][1]).abs().max().item():.3e}  shape {tuple(v.shape)}")
        print("  previous call:", runs[0][i - 1][0] if i else None); break
else: print("deterministic over 3 runs")
