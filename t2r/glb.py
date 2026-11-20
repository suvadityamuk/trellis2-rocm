"""o_voxel.postprocess.to_glb (remesh=True, remesh_band=1, remesh_project=0) + GLB export on t2r.

parity: the official sequence of CuMesh / cuBVH / xatlas / nvdiffrast / FlexGEMM / OpenCV calls with the same
arguments. Two bitwise-safe changes: the FlexGEMM trilinear weighted sum runs as a t2r HIP kernel (bitwise equal to
the Triton kernel, without its per-image autotuning), and the four independent cv2.inpaint calls run in parallel
threads. Export goes through trimesh exactly as the official example.
fast: GPU LBVH instead of cuBVH (CPU-built BVH), grouped parallel xatlas packing, fused trilinear sampling, GPU
wavefront TELEA + texture packing, and a direct GLB writer with WebP method 0 (both textures encoded in parallel).
"""
import io, json, struct, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2
from PIL import Image
import nvdiffrast.torch as dr
import t2r_cumesh as cumesh  # t2r build of CuMesh: caching allocator + device-side fills (bitwise-identical results)

from . import _C

_ctx = None


def _raster_ctx():
    global _ctx
    if _ctx is None:
        _ctx = dr.RasterizeCudaContext()
    return _ctx


class LBVHAdapter:
    """cumesh.cuBVH.unsigned_distance interface over t2r's GPU LBVH."""
    def __init__(self, vertices, faces):
        self.impl = _C.LBVH(vertices.float().contiguous(), faces.int().contiguous())

    def unsigned_distance(self, positions, return_uvw=False):
        d, f, u = self.impl.query(positions.float().reshape(-1, 3), return_uvw)
        return d.view(positions.shape[:-1]), f.view(positions.shape[:-1]), (u.view(*positions.shape[:-1], 3) if return_uvw else None)


def _uv_unwrap_fast(mesh, ngroups=64, max_chart_faces=1024):
    """cumesh uv_unwrap (same CuMesh charting) with grouped parallel xatlas packing (t2r uv_pack_groups)."""
    mesh.remove_degenerate_faces()
    mesh.compute_charts(threshold_cone_half_angle_rad=np.radians(90.0), refine_iterations=0, global_iterations=1, smooth_strength=1)
    new_v, new_f = mesh.read()
    nc, cid, cvm, cf, cvo, cfo = mesh.read_atlas_charts()
    cv = new_v[cvm].float().cpu().contiguous()
    vmap, faces, uvs = _C.uv_pack_groups(cv, cf.int().cpu().contiguous(), cvo.cpu(), cfo.cpu(), ngroups, 0, max_chart_faces)
    vmaps = cvm.cpu()[vmap.long()]
    return new_v.cpu()[vmaps], faces, uvs, vmaps


def grid_info(coords, voxel_size):
    dev = coords.device
    aabb = torch.tensor([[-0.5] * 3, [0.5] * 3], dtype=torch.float32, device=dev)
    vs = torch.tensor(np.array([voxel_size] * 3), dtype=torch.float32, device=dev)
    return aabb, vs, ((aabb[1] - aabb[0]) / vs).round().int()


def stage_remesh(vertices, faces, grid_size, aabb, fast):
    """fill_holes -> BVH -> narrow-band dual-contouring remesh. Returns (CuMesh with remeshed mesh, v0, f0, bvh)."""
    vertices, faces = vertices.cuda(), faces.cuda()
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()
    bvh = LBVHAdapter(vertices, faces) if fast else cumesh.cuBVH(vertices, faces)
    center = aabb.mean(dim=0)
    scale = (aabb[1] - aabb[0]).max().item()
    resolution = grid_size.max().item()
    band = 1
    mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(vertices, faces, center=center, scale=(resolution + 3 * band) / resolution * scale,
                                                      resolution=resolution, band=band, project_back=0, verbose=False, bvh=bvh))
    return mesh, vertices, faces, bvh


def stage_unwrap(mesh, fast):
    if fast:
        out_v, out_f, out_uv, out_vmaps = _uv_unwrap_fast(mesh)
    else:
        out_v, out_f, out_uv, out_vmaps = mesh.uv_unwrap(
            compute_charts_kwargs={"threshold_cone_half_angle_rad": np.radians(90.0), "refine_iterations": 0,
                                   "global_iterations": 1, "smooth_strength": 1},
            return_vmaps=True, verbose=False)
    out_v, out_f, out_uv, out_vmaps = out_v.cuda(), out_f.cuda(), out_uv.cuda(), out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_n = mesh.read_vertex_normals()[out_vmaps]
    return out_v, out_f, out_uv, out_n


def stage_bake(out_v, out_f, out_uv, v0, f0, bvh, attr_volume, coords, aabb, vs, grid_size, texture_size, fast):
    """UV-space raster -> 3D position -> nearest point on the hole-filled input mesh -> trilinear voxel attributes."""
    ctx = _raster_ctx()
    uvs_rast = torch.cat([out_uv * 2 - 1, torch.zeros_like(out_uv[:, :1]), torch.ones_like(out_uv[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
    for i in range(0, out_f.shape[0], 100000):
        rc, _ = dr.rasterize(ctx, uvs_rast, out_f[i:i + 100000], resolution=[texture_size, texture_size])
        m = rc[..., 3:4] > 0
        rc[..., 3:4] += i
        rast = torch.where(m, rc, rast)
    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(out_v.unsqueeze(0), rast, out_f)[0][0]
    valid_pos = pos[mask]
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig = v0[f0[face_id.long()]]
    valid_pos = (orig * uvw.unsqueeze(-1)).sum(dim=1)
    q = ((valid_pos - aabb[0]) / vs).reshape(1, -1, 3)
    C = attr_volume.shape[1]
    attrs = torch.zeros(texture_size, texture_size, C, device='cuda')
    if fast:
        attrs[mask] = _C.grid_sample_trilinear(attr_volume.float().contiguous(), coords.int().contiguous(), q, *grid_size.tolist())
    else:
        attrs[mask] = _grid_sample_parity(attr_volume, coords, grid_size, q).view(-1, C)
    return attrs, mask


def finish_textures_parity(attrs, mask):
    """The official numpy / OpenCV texture finishing, with the four independent inpaints in parallel threads."""
    mask_np = mask.cpu().numpy()
    a = attrs.cpu().numpy()
    base_color = np.clip(a[..., 0:3] * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(a[..., 3:4] * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(a[..., 4:5] * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(a[..., 5:6] * 255, 0, 255).astype(np.uint8)
    mask_inv = (~mask_np).astype(np.uint8)
    with ThreadPoolExecutor(4) as ex:
        fb = ex.submit(cv2.inpaint, base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        fm = ex.submit(cv2.inpaint, metallic, mask_inv, 1, cv2.INPAINT_TELEA)
        fr = ex.submit(cv2.inpaint, roughness, mask_inv, 1, cv2.INPAINT_TELEA)
        fa = ex.submit(cv2.inpaint, alpha, mask_inv, 1, cv2.INPAINT_TELEA)
        return fb.result(), fm.result()[..., None], fr.result()[..., None], fa.result()[..., None]


def to_glb(vertices, faces, attr_volume, coords, voxel_size, mode="parity", texture_size=4096, decimation_target=1000000,
           times=None):
    """Returns trimesh.Trimesh (parity) or a dict for write_glb (fast)."""
    def tick(n):
        if times is not None:
            torch.cuda.synchronize(); t = time.perf_counter(); times[n] = t - tick.t; tick.t = t
    tick.t = time.perf_counter()
    fast = mode == "fast"
    aabb, vs, grid_size = grid_info(coords, voxel_size)
    mesh, v0, f0, bvh = stage_remesh(vertices, faces, grid_size, aabb, fast)
    tick("remesh")
    mesh.simplify(decimation_target, verbose=False)
    tick("simplify")
    out_v, out_f, out_uv, out_n = stage_unwrap(mesh, fast)
    tick("uv_unwrap")
    attrs, mask = stage_bake(out_v, out_f, out_uv, v0, f0, bvh, attr_volume, coords, aabb, vs, grid_size, texture_size, fast)
    tick("bake")
    if fast:
        base, mr = _C.bake_textures(attrs, mask, 3, 1)
        tick("textures")
        return dict(vertices=out_v, faces=out_f, uvs=out_uv, normals=out_n, base=base, mr=mr)
    base_color, metallic, roughness, alpha = finish_textures_parity(attrs, mask)
    tick("textures")
    import trimesh, trimesh.visual
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0, roughnessFactor=1.0, alphaMode='OPAQUE', doubleSided=False)
    v_np, f_np, uv_np, n_np = out_v.cpu().numpy(), out_f.cpu().numpy(), out_uv.cpu().numpy(), out_n.cpu().numpy()
    v_np[:, 1], v_np[:, 2] = v_np[:, 2], -v_np[:, 1]
    n_np[:, 1], n_np[:, 2] = n_np[:, 2], -n_np[:, 1]
    uv_np[:, 1] = 1 - uv_np[:, 1]
    return trimesh.Trimesh(vertices=v_np, faces=f_np, vertex_normals=n_np, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv_np, material=material))


def _grid_sample_parity(feats, coords, grid_size, q):
    """flex_gemm grid_sample_3d(trilinear): FlexGEMM's CUDA neighbor map + weights, t2r's bitwise weighted sum."""
    from flex_gemm import kernels
    from flex_gemm.ops import utils as fu
    c4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1).int()
    W, H, D = grid_size.tolist()
    shape = torch.Size([1, feats.shape[1], W, H, D])
    hk, hv = fu.init_hashmap(shape, int(2.0 * c4.shape[0]), c4.device)
    idx, w = kernels.cuda.hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(hk, hv, c4, q, W, H, D)
    return _C.weighted_sum8(feats, idx.view(-1, 8), w.view(-1, 8), False)


# ---------------------------------------------------------------------------------------------------- GLB writer (fast)
def _webp(arr, method=0, alpha_quality=100):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="WEBP", method=method, alpha_quality=alpha_quality)
    return b.getvalue()


def write_glb(out, path, webp_method=0):
    """Minimal glTF 2.0 binary with the same content as the official trimesh export (positions, normals, uvs, uint32
    indices, PBR material with EXT_texture_webp base color + metallicRoughness, OPAQUE, single-sided)."""
    with ThreadPoolExecutor(2) as ex:
        # alphaMode is OPAQUE, so the baked alpha channel is never used for shading; its lossy-alpha setting is the
        # single biggest encode cost (3-4x the color planes), so the fast build stores it at alpha_quality 0.
        fb = ex.submit(_webp, out["base"].cpu().numpy(), webp_method, 0)
        fm = ex.submit(_webp, out["mr"].cpu().numpy(), webp_method)
        v = out["vertices"].float().clone(); n = out["normals"].float().clone(); uv = out["uvs"].float().clone()
        v[:, 1], v[:, 2] = out["vertices"][:, 2], -out["vertices"][:, 1]
        n[:, 1], n[:, 2] = out["normals"][:, 2], -out["normals"][:, 1]
        # the official code flips v and trimesh's exporter flips it back, so the file stores the raw atlas v
        v, n, uv = v.cpu().numpy(), n.cpu().numpy(), uv.cpu().numpy()
        f = out["faces"].to(torch.int32).cpu().numpy().astype(np.uint32)
        imgs = [fb.result(), fm.result()]
    blobs, views, acc = [], [], []
    def add(data, target=None):
        off = sum(len(b) for b in blobs)
        pad = (-len(data)) % 4
        blobs.append(data + b"\0" * pad)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), **({"target": target} if target else {})})
        return len(views) - 1
    def accessor(arr, typ, ctype, target, minmax=False):
        a = {"bufferView": add(np.ascontiguousarray(arr).tobytes(), target), "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if minmax: a["min"] = arr.min(0).tolist(); a["max"] = arr.max(0).tolist()
        acc.append(a); return len(acc) - 1
    ip = accessor(f.reshape(-1), "SCALAR", 5125, 34963)
    pp = accessor(v, "VEC3", 5126, 34962, True)
    np_ = accessor(n, "VEC3", 5126, 34962)
    tp = accessor(uv, "VEC2", 5126, 34962)
    im = [add(b) for b in imgs]
    gltf = {
        "asset": {"version": "2.0", "generator": "t2r (TRELLIS.2 ROCm)"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": pp, "NORMAL": np_, "TEXCOORD_0": tp}, "indices": ip, "material": 0, "mode": 4}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "baseColorTexture": {"index": 0},
                                                "metallicRoughnessTexture": {"index": 1}, "metallicFactor": 1.0, "roughnessFactor": 1.0},
                       "alphaMode": "OPAQUE", "doubleSided": False}],
        "textures": [{"sampler": 0, "extensions": {"EXT_texture_webp": {"source": 0}}},
                     {"sampler": 0, "extensions": {"EXT_texture_webp": {"source": 1}}}],
        "images": [{"bufferView": im[0], "mimeType": "image/webp"}, {"bufferView": im[1], "mimeType": "image/webp"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "accessors": acc, "bufferViews": views, "buffers": [{"byteLength": sum(len(b) for b in blobs)}],
        "extensionsUsed": ["EXT_texture_webp"], "extensionsRequired": ["EXT_texture_webp"],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    bin_ = b"".join(blobs)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A)); fh.write(js)
        fh.write(struct.pack("<II", len(bin_), 0x004E4942)); fh.write(bin_)
