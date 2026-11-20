"""GPU wavefront TELEA (fast build) vs OpenCV cv2.inpaint(TELEA) on dumped to_glb textures."""
import sys, time, numpy as np, torch, cv2, t2r._C as C
k = sys.argv[1] if len(sys.argv) > 1 else "0a34fae7"
G = torch.load(f"out/ref/{k}/glb.pt", weights_only=False)
(bc, minv, r0, bc_out), (me, _, r1, me_out), (ro, _, _, ro_out), (al, _, _, al_out) = G["inpaint"]
mask = torch.from_numpy(minv == 0).cuda()
H, W = mask.shape
attrs = torch.zeros(H, W, 6)
for c, a in enumerate([bc[..., 0], bc[..., 1], bc[..., 2], me.squeeze(), ro.squeeze(), al.squeeze()]):
    attrs[..., c] = (torch.from_numpy(a.astype(np.float32)) + 0.5) / 255
attrs = attrs.cuda()
base, mr = C.bake_textures(attrs, mask, r0, r1); torch.cuda.synchronize()
t = time.perf_counter(); base, mr = C.bake_textures(attrs, mask, r0, r1); torch.cuda.synchronize(); dt = time.perf_counter() - t
base, mr = base.cpu().numpy(), mr.cpu().numpy()
hole = minv != 0
print(f"{k}: {hole.sum()} of {hole.size} texels inpainted, GPU bake_textures {dt*1e3:.1f} ms")
print("known texels unchanged:", np.array_equal(base[~hole, :3], bc[~hole]), np.array_equal(mr[~hole, 2], me.squeeze()[~hole]))
for name, got, ref in [("base rgb", base[..., :3], bc_out), ("metallic", mr[..., 2], me_out[..., 0] if me_out.ndim == 3 else me_out),
                       ("roughness", mr[..., 1], ro_out if ro_out.ndim == 2 else ro_out[..., 0]), ("alpha", base[..., 3], al_out if al_out.ndim == 2 else al_out[..., 0])]:
    d = np.abs(got.astype(np.int16) - ref.astype(np.int16))[hole]
    print(f"  {name:9s} vs cv2 on holes: mean |d| {d.mean():.2f}, p99 {np.percentile(d, 99):.0f}, max {d.max()}")
t = time.perf_counter(); cv2.inpaint(bc, minv, r0, cv2.INPAINT_TELEA); print(f"cv2 base color inpaint alone: {time.perf_counter() - t:.2f} s")
