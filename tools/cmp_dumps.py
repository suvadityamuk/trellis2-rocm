"""Bitwise comparison of two dump directories (e.g. two official runs, or official vs t2r).
Usage: python tools/cmp_dumps.py out/ref out/ref2 [pipe|glb]"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/src/TRELLIS.2"))


def flat(x, p=""):
    if isinstance(x, torch.Tensor): yield p, x
    elif isinstance(x, np.ndarray): yield p, torch.from_numpy(np.ascontiguousarray(x))
    elif isinstance(x, dict):
        for k, v in x.items(): yield from flat(v, f"{p}.{k}" if p else str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x): yield from flat(v, f"{p}[{i}]")
    elif hasattr(x, "vertices") and hasattr(x, "faces"):
        yield f"{p}.v", x.vertices; yield f"{p}.f", x.faces
    elif hasattr(x, "feats") and hasattr(x, "coords"):
        yield f"{p}.feats", x.feats; yield f"{p}.coords", x.coords


def cmp(a, b):
    A, B = dict(flat(a)), dict(flat(b))
    rows = []
    for k in A:
        if k not in B: rows.append((k, "missing in b")); continue
        x, y = A[k].cpu(), B[k].cpu()
        if x.shape != y.shape: rows.append((k, f"shape {tuple(x.shape)} vs {tuple(y.shape)}")); continue
        if x.dtype == y.dtype and torch.equal(x, y): rows.append((k, "bitwise")); continue
        d = (x.double() - y.double()).abs()
        rows.append((k, f"DIFF n={int((d > 0).sum())}/{d.numel()} max={d.max().item():.3e}"))
    return rows


if __name__ == "__main__":
    da, db = sys.argv[1], sys.argv[2]
    kinds = sys.argv[3:] or ["pipe", "glb"]
    for img in sorted(os.listdir(da)):
        for kind in kinds:
            pa, pb = os.path.join(da, img, kind + ".pt"), os.path.join(db, img, kind + ".pt")
            if not (os.path.exists(pa) and os.path.exists(pb)): continue
            rows = cmp(torch.load(pa, weights_only=False), torch.load(pb, weights_only=False))
            nb = sum(r[1] == "bitwise" for r in rows)
            print(f"== {img} {kind}: {nb}/{len(rows)} bitwise")
            for k, s in rows:
                if s != "bitwise": print(f"   {k:50s} {s}")
