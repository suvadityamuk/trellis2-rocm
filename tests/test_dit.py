"""t2r FlowDiT vs official DiT calls recorded by ref_dump (teacher forcing): parity bitwise, fast within tolerance."""
import sys, time, torch
from t2r.dit import FlowDiT
from t2r.pipeline import CKPT
k = sys.argv[1] if len(sys.argv) > 1 else "0a34fae7"
P = torch.load(f"out/ref/{k}/pipe.pt", weights_only=False)
ck = f"{CKPT}/ckpts/"
models = {"sparse_structure_flow_model": ("ss_flow_img_dit_1_3B_64_bf16", True),
          "shape_slat_flow_model_512": ("slat_flow_img2shape_dit_1_3B_512_bf16", False),
          "shape_slat_flow_model_1024": ("slat_flow_img2shape_dit_1_3B_1024_bf16", False),
          "tex_slat_flow_model_1024": ("slat_flow_imgshape2tex_dit_1_3B_1024_bf16", False)}
only = sys.argv[2:] or list(models)
for name in only:
    f, dense = models[name]
    m = FlowDiT(ck + f + ".safetensors", dense)
    for ci in range(2):
        c = P.get(f"dit.{name}.call{ci}")
        if c is None: continue
        t = c["t"].cuda(); cond = c["cond"].cuda()
        if dense:
            x = c["x"].cuda(); ref = c["out"].cuda()
            out = m.forward_parity(x, t, cond)
        else:
            x = c["x"]["feats"].cuda(); coords = c["x"]["coords"].cuda(); ref = c["out"]["feats"].cuda()
            m.set_coords(coords)
            cc = c["kw"].get("concat_cond"); cc = cc["feats"].cuda() if cc is not None else None
            out = m.forward_parity(x, t, cond, cc)
        print(f"{name} call{ci} parity: bitwise={torch.equal(out, ref)} maxdiff={(out - ref).abs().max().item():.3e}", flush=True)
        # fast path (single item, same cond)
        xf = x.view(x.shape[1], -1).t().contiguous() if dense else x
        of = m.forward_fast(xf, [t.item()], [cond], cc if not dense else None)
        if dense: of = of.t().reshape_as(ref)
        d = (of - ref).abs(); rel = d.norm() / ref.norm()
        print(f"   fast: maxdiff={d.max().item():.3e} mean={d.mean().item():.3e} rel_l2={rel.item():.3e}", flush=True)
    del m; torch.cuda.empty_cache()
