"""End-to-end tolerance table: Chamfer (fraction of bbox diagonal) and base-color L1 of each configuration vs official
eager, for every image. Usage: python tools/tolerance.py --out bench/tolerance.json NAME=dir ... (first is the baseline)"""
import os, sys, json, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_glb import compare

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--images", nargs="+", default=["0a34fae7", "0e4984a9", "0f168a4b", "130c2b18", "T"])
ap.add_argument("cols", nargs="+")
a = ap.parse_args()
cols = [c.split("=", 1) for c in a.cols]
base_name, base_dir = cols[0]
jobs = {(n, i): (os.path.join(base_dir, i + ".glb"), os.path.join(d, i + ".glb")) for n, d in cols[1:] for i in a.images}
with ThreadPoolExecutor(min(8, len(jobs))) as ex:
    futs = {k: ex.submit(compare, *v) for k, v in jobs.items()}
    res = {k: f.result() for k, f in futs.items()}
T = {n: {i: {"chamfer": float(res[(n, i)][0]), "color_l1": float(res[(n, i)][1])} for i in a.images} for n, _ in cols[1:]}
json.dump({"baseline": base_name, "results": T}, open(a.out, "w"), indent=1)
print(f"{'image':10s}" + "".join(f"{n:>26s}" for n, _ in cols[1:]))
for i in a.images:
    print(f"{i:10s}" + "".join(f"{T[n][i]['chamfer']:>14.2e} / {T[n][i]['color_l1']:5.1f}   " for n, _ in cols[1:]))
