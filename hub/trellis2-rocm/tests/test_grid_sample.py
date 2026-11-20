import pytest
import torch

from ._load import k

pytestmark = pytest.mark.kernels_ci


def reference(feats, coords, points, grid_size):
    """Float64 dense reference: weights of the up to 8 surrounding voxel centers, renormalized over occupied voxels."""
    W, H, D = grid_size
    C = feats.shape[1]
    xyz = coords[:, -3:].long()
    dense = torch.zeros(W, H, D, C, dtype=torch.float64, device=feats.device)
    occ = torch.zeros(W, H, D, dtype=torch.bool, device=feats.device)
    dense[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = feats.double()
    occ[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    base = torch.floor(points - 0.5).long()
    q = points.double()
    acc = torch.zeros(points.shape[0], C, dtype=torch.float64, device=feats.device)
    wsum = torch.zeros(points.shape[0], dtype=torch.float64, device=feats.device)
    for i in range(8):
        v = base + torch.tensor([i & 1, (i >> 1) & 1, (i >> 2) & 1], device=feats.device)
        ok = ((v >= 0) & (v < torch.tensor([W, H, D], device=feats.device))).all(-1)
        vc = v.clamp_min(0).minimum(torch.tensor([W - 1, H - 1, D - 1], device=feats.device))
        ok &= occ[vc[:, 0], vc[:, 1], vc[:, 2]]
        # weights from the float32 inputs, as the kernel computes them
        w = (1 - (points - v - 0.5).abs()).prod(-1).double() * ok
        acc += w[:, None] * dense[vc[:, 0], vc[:, 1], vc[:, 2]]
        wsum += w
    return acc / wsum.clamp_min(1e-12)[:, None], wsum > 0


def scene(grid_size, n, C, seed, batch_col=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W, H, D = grid_size
    lin = torch.randperm(W * H * D, device="cuda", generator=g)[:n]
    coords = torch.stack([lin // (H * D), (lin // D) % H, lin % D], -1).int()
    if batch_col:
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1)
    feats = torch.randn(n, C, device="cuda", generator=g)
    points = torch.rand(20000, 3, device="cuda", generator=g) * torch.tensor([W + 2.0, H + 2.0, D + 2.0], device="cuda") - 1
    return feats, coords, points


@pytest.mark.parametrize("C", [1, 2, 3, 4, 5, 6, 8, 11, 32])
def test_matches_reference(C):
    grid = (24, 17, 31)
    feats, coords, points = scene(grid, 6000, C, C)
    out = k.grid_sample_trilinear(feats, coords, points, grid)
    ref, hit = reference(feats, coords, points, grid)
    assert out.shape == (points.shape[0], C) and out.dtype == torch.float32
    torch.testing.assert_close(out.double(), ref, atol=1e-5, rtol=1e-5)
    assert hit.any() and (~hit).any()
    assert (out[~hit] == 0).all()


def test_batch_column_ignored():
    grid = (20, 20, 20)
    feats, coords, points = scene(grid, 3000, 6, 0)
    with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1)
    assert torch.equal(k.grid_sample_trilinear(feats, coords, points, grid), k.grid_sample_trilinear(feats, with_batch, points, grid))


def test_voxel_centers_return_voxel_features():
    grid = (16, 16, 16)
    feats, coords, _ = scene(grid, 2000, 6, 1)
    out = k.grid_sample_trilinear(feats, coords, coords.float() + 0.5, grid)
    torch.testing.assert_close(out, feats, atol=1e-6, rtol=1e-6)


def test_dense_grid_matches_torch_grid_sample():
    W, H, D, C = 12, 10, 14, 5
    g = torch.Generator(device="cuda").manual_seed(2)
    vol = torch.randn(W, H, D, C, device="cuda", generator=g)
    xyz = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), torch.arange(D), indexing="ij"), -1).reshape(-1, 3).int().cuda()
    # interior points only: at the border F.grid_sample pads with zeros while this kernel renormalizes
    pts = torch.rand(8000, 3, device="cuda", generator=g) * torch.tensor([W - 1.0, H - 1.0, D - 1.0], device="cuda") + 0.5
    out = k.grid_sample_trilinear(vol.reshape(-1, C), xyz, pts, (W, H, D))
    size = torch.tensor([W, H, D], device="cuda")
    norm = (pts / size * 2 - 1).flip(-1)  # grid_sample wants (x->last dim) order
    ref = torch.nn.functional.grid_sample(vol.permute(3, 0, 1, 2)[None], norm[None, :, None, None], mode="bilinear", align_corners=False)
    torch.testing.assert_close(out, ref[0, :, :, 0, 0].T, atol=1e-5, rtol=1e-4)


def test_deterministic_and_empty():
    grid = (20, 20, 20)
    feats, coords, points = scene(grid, 3000, 6, 3)
    assert torch.equal(k.grid_sample_trilinear(feats, coords, points, grid), k.grid_sample_trilinear(feats, coords, points, grid))
    assert k.grid_sample_trilinear(feats, coords, points[:0], grid).shape == (0, 6)
    assert (k.grid_sample_trilinear(feats[:0], coords[:0], points, grid) == 0).all()


def test_rejects_bad_inputs():
    grid = (8, 8, 8)
    feats, coords, points = scene(grid, 100, 3, 4)
    with pytest.raises(RuntimeError):
        k.grid_sample_trilinear(feats, coords.long(), points, grid)
    with pytest.raises(RuntimeError):
        k.grid_sample_trilinear(feats, coords, points, (8, 8))
    with pytest.raises(RuntimeError):
        k.grid_sample_trilinear(feats[:50], coords, points, grid)
