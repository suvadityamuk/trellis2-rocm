"""bench/*.json -> docs/timing.png (per-image stacked stages) and docs/speedup.png (sum over images)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

V = [("eager", "Official eager"), ("compile", "Official torch.compile"), ("parity", "t2r parity"), ("fast", "t2r fast")]
COL = {"eager": "#2a78d6", "compile": "#eb6834", "parity": "#1baf7a", "fast": "#6250d6"}  # validated categorical slots 1, 2, 3, 7
S = [("pipeline", "pipeline (models)"), ("to_glb", "to_glb (remesh / UV / bake)"), ("export", "GLB export")]
HATCH = {"pipeline": "", "to_glb": "//", "export": ".."}
D = {k: json.load(open(f"bench/{k}.json")) for k, _ in V}
imgs = [i for i in D["eager"] if not i.startswith("_")]
names = [i[:4] for i in imgs]
os.makedirs("docs", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})

fig, ax = plt.subplots(figsize=(11, 4.6)); w = 0.2
for j, (k, lab) in enumerate(V):
    x = np.arange(len(imgs)) + (j - 1.5) * w; b = np.zeros(len(imgs))
    for s, sl in S:
        h = np.array([D[k][i][s] for i in imgs])
        ax.bar(x, h, w * 0.9, bottom=b, color=COL[k], hatch=HATCH[s], edgecolor="white", linewidth=0.6,
               label=lab if s == "pipeline" else None)
        b += h
    for xi, t in zip(x, b): ax.text(xi, t + 2, f"{t:.0f}", ha="center", fontsize=7, color="#52514e")
ax.set_xticks(np.arange(len(imgs)), names); ax.set_ylabel(f"seconds ({D['eager']['_meta']['gpu']})")
ax.set_title("Per-image image->GLB time (stacks: pipeline / to_glb (hatched) / export (dotted))", fontsize=10)
ax.legend(frameon=False, ncol=4, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e4df", linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("docs/timing.png", dpi=150)

fig, ax = plt.subplots(figsize=(7.5, 3.0))
tot = [sum(D[k][i]["total"] for i in imgs) for k, _ in V]
bars = ax.barh([l for _, l in V][::-1], tot[::-1], color=[COL[k] for k, _ in V][::-1], height=0.6)
for b_, t in zip(bars, tot[::-1]): ax.text(t + 8, b_.get_y() + b_.get_height() / 2, f"{t:.0f} s  ({tot[0] / t:.1f}x)", va="center", color="#0b0b0b")
ax.set_xlabel("total seconds, 5 example images (warm process)"); ax.set_xlim(0, max(tot) * 1.28)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig("docs/speedup.png", dpi=150)
print({k: round(t, 1) for (k, _), t in zip(V, tot)})
