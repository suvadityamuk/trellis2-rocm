import pytest
import torch

from ._load import k

pytestmark = pytest.mark.kernels_ci


def patch(n, origin, tilt, seed):
    """A gently curved n x n quad patch: (vertices [n*n, 3], faces [2(n-1)^2, 3])."""
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij")
    z = 0.1 * torch.sin(3 * x + tilt) * torch.cos(2 * y) + 0.002 * torch.rand(n, n, generator=g)
    V = torch.stack([x, y, z], -1).reshape(-1, 3) * (0.5 + 0.1 * seed) + torch.tensor(origin, dtype=torch.float32)
    i = torch.arange(n - 1)
    q = (i[:, None] * n + i[None]).reshape(-1)
    F = torch.cat([torch.stack([q, q + 1, q + n], 1), torch.stack([q + 1, q + n + 1, q + n], 1)]).int()
    return V.float(), F


def charts(sizes):
    Vs, Fs, vo, fo = [], [], [0], [0]
    for c, n in enumerate(sizes):
        V, F = patch(n, [2.0 * c, 0.0, 0.0], 0.3 * c, c)
        Vs.append(V); Fs.append(F + vo[-1])
        vo.append(vo[-1] + V.shape[0]); fo.append(fo[-1] + F.shape[0])
    return torch.cat(Vs), torch.cat(Fs), torch.tensor(vo), torch.tensor(fo)


def uv_area(uvs, faces):
    a, b, c = (uvs[faces[:, i].long()].double() for i in range(3))
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).abs()


def check_valid(V, F, vmap, faces, uvs):
    assert vmap.dtype == torch.int32 and faces.dtype == torch.int32 and uvs.dtype == torch.float32
    assert uvs.shape == (vmap.shape[0], 2) and faces.shape == F.shape
    assert vmap.min() >= 0 and vmap.max() < V.shape[0]
    assert faces.min() >= 0 and faces.max() < vmap.shape[0]
    assert uvs.min() >= 0 and uvs.max() <= 1
    # every input triangle appears exactly once, as the same three input vertices
    key = lambda t: sorted(tuple(sorted(r)) for r in t.tolist())
    assert key(vmap[faces.long()]) == key(F)
    area = uv_area(uvs, faces)
    assert (area > 0).all()
    return area


@pytest.mark.parametrize("groups", [1, 4, 64])
def test_valid_atlas(groups):
    V, F, vo, fo = charts([6, 9, 12, 7, 15, 5, 10, 8])
    vmap, faces, uvs = k.pack_charts(V, F, vo, fo, num_groups=groups)
    area = check_valid(V, F, vmap, faces, uvs)
    assert 0.05 < area.sum() <= 1.0  # the atlas is reasonably filled and charts fit in the unit square


def test_no_overlap_between_charts():
    V, F, vo, fo = charts([8, 11, 6, 9])
    vmap, faces, uvs = k.pack_charts(V, F, vo, fo, num_groups=2)
    check_valid(V, F, vmap, faces, uvs)
    # rasterize triangle interiors (shrunk slightly) on a grid; no texel may be claimed by two charts
    R = 512
    owner = torch.full((R, R), -1, dtype=torch.long)
    chart_of_vertex = torch.bucketize(vmap.long(), vo[1:], right=True)
    yy, xx = torch.meshgrid((torch.arange(R) + 0.5) / R, (torch.arange(R) + 0.5) / R, indexing="ij")
    P = torch.stack([xx, yy], -1).reshape(-1, 2).double()
    for f in faces.long():
        a, b, c = uvs[f].double()
        cen = (a + b + c) / 3
        a, b, c = (cen + 0.98 * (p - cen) for p in (a, b, c))
        lo = (torch.minimum(torch.minimum(a, b), c) * R).floor().clamp(0, R - 1).long()
        hi = (torch.maximum(torch.maximum(a, b), c) * R).ceil().clamp(0, R - 1).long()
        sub = P.reshape(R, R, 2)[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1].reshape(-1, 2)
        e = lambda p, q: (q[0] - p[0]) * (sub[:, 1] - p[1]) - (q[1] - p[1]) * (sub[:, 0] - p[0])
        w0, w1, w2 = e(a, b), e(b, c), e(c, a)
        inside = ((w0 > 0) & (w1 > 0) & (w2 > 0)) | ((w0 < 0) & (w1 < 0) & (w2 < 0))
        view = owner[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1].reshape(-1)
        ch = int(chart_of_vertex[f[0]])
        assert ((view[inside] == -1) | (view[inside] == ch)).all()
        view[inside] = ch
        owner[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = view.reshape(hi[1] - lo[1] + 1, hi[0] - lo[0] + 1)


def test_deterministic_across_thread_counts():
    V, F, vo, fo = charts([6, 9, 12, 7, 15, 5, 10, 8])
    ref = k.pack_charts(V, F, vo, fo, num_groups=4, num_threads=1)
    for threads in (0, 2, 4):
        out = k.pack_charts(V, F, vo, fo, num_groups=4, num_threads=threads)
        assert all(torch.equal(a, b) for a, b in zip(ref, out))


def test_large_chart_is_split():
    V, F, vo, fo = charts([40])  # 3042 faces in one chart
    whole = k.pack_charts(V, F, vo, fo, num_groups=4, max_chart_faces=0)
    split = k.pack_charts(V, F, vo, fo, num_groups=4, max_chart_faces=512)
    check_valid(V, F, *whole)
    check_valid(V, F, *split)
    assert split[0].shape[0] > whole[0].shape[0]  # splitting duplicates vertices along the cuts


def test_uniform_texel_density():
    V, F, vo, fo = charts([10, 10, 10, 10, 10, 10])
    vmap, faces, uvs = k.pack_charts(V, F, vo, fo, num_groups=3)
    p = V[vmap.long()].double()
    a, b, c = (p[faces[:, i].long()] for i in range(3))
    area3d = 0.5 * torch.linalg.cross(b - a, c - a).norm(dim=-1)
    chart = torch.bucketize(vmap[faces[:, 0].long()].long(), vo[1:], right=True)
    ratio = torch.stack([uv_area(uvs, faces)[chart == i].sum() / area3d[chart == i].sum() for i in range(6)])
    assert ratio.max() / ratio.min() < 1.25  # charts in different groups share one texel scale


def test_empty_and_bad_inputs():
    e = k.pack_charts(torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.int32), torch.tensor([0]), torch.tensor([0]))
    assert e[0].shape == (0,) and e[1].shape == (0, 3) and e[2].shape == (0, 2)
    V, F, vo, fo = charts([6, 7])
    with pytest.raises(RuntimeError):
        k.pack_charts(V, F.long(), vo, fo)
    with pytest.raises(RuntimeError):
        k.pack_charts(V, F, vo[:-1], fo)
    bad = F.clone(); bad[0, 0] = V.shape[0] - 1  # chart 0 face pointing into chart 1
    with pytest.raises(RuntimeError):
        k.pack_charts(V, bad, vo, fo)


def test_many_small_charts_stress():
    # Thousands of AddMesh calls per atlas while its worker thread drains the queue: this crashed intermittently
    # before the vendored xatlas task scheduler copied tasks under its lock.
    g = torch.Generator().manual_seed(1)
    V, F, vo, fo = charts(torch.randint(2, 5, (30000,), generator=g).tolist())
    ref = k.pack_charts(V, F, vo, fo, num_groups=64)
    assert ref[1].shape == F.shape
    for _ in range(20):
        out = k.pack_charts(V, F, vo, fo, num_groups=64)
        assert all(torch.equal(a, b) for a, b in zip(ref, out))
