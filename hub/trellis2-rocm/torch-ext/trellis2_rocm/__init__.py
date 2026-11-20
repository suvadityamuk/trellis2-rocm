"""GPU mesh and texture kernels from the TRELLIS.2 ROCm port (https://github.com/suvadityamuk/trellis2-rocm)."""

from typing import Sequence, Tuple

import torch

from ._ops import ops


def mesh_udf(vertices: torch.Tensor, faces: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unsigned distance from each point to a triangle mesh, via a GPU linear BVH built on the fly.

    Args:
        vertices: float32 [V, 3] GPU tensor.
        faces: int32 or int64 [F, 3] GPU tensor of vertex indices, F >= 1.
        points: float32 [M, 3] GPU tensor of query points.

    Returns:
        (distance float32 [M], face_id int64 [M], barycentric float32 [M, 3]) of the closest point on the mesh.
        Ties between equally close faces resolve to the lowest face index.
    """
    return ops.mesh_udf(vertices, faces, points)


def telea_inpaint(image: torch.Tensor, mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Fill masked pixels with TELEA inpainting, a GPU alternative to cv2.inpaint(..., cv2.INPAINT_TELEA).

    Pixels are filled in parallel by integer distance level rather than in fast-marching order, so results are close
    to OpenCV's but not identical. Unmasked pixels are returned unchanged.

    Args:
        image: uint8 [H, W] or [H, W, C] (C <= 4) GPU tensor.
        mask: bool or uint8 [H, W] GPU tensor; nonzero marks pixels to inpaint.
        radius: neighborhood radius in pixels.

    Returns:
        Inpainted uint8 tensor with the same shape as image.
    """
    return ops.telea_inpaint(image, mask, radius)


def grid_sample_trilinear(feats: torch.Tensor, coords: torch.Tensor, points: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
    """Trilinearly sample features stored on a sparse voxel grid.

    Voxel (i, j, k) spans [i, i+1) x [j, j+1) x [k, k+1), so its center is at (i + 0.5, j + 0.5, k + 0.5). Each point
    blends the up to 8 voxels whose centers surround it; weights are renormalized over the voxels that exist, and a
    point with no occupied neighbor gets zeros.

    Args:
        feats: float32 [N, C] GPU tensor of per-voxel features (any C).
        coords: int32 [N, 3] (x, y, z) or [N, 4] (batch, x, y, z; the batch column is ignored) GPU tensor of unique
            voxel coordinates inside the grid.
        points: float32 [M, 3] GPU tensor of query points in grid space.
        grid_size: (W, H, D) extent of the grid along x, y, z.

    Returns:
        float32 [M, C] sampled features.
    """
    return ops.grid_sample_trilinear(feats, coords, points, list(grid_size))


def pack_charts(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_offsets: torch.Tensor,
    face_offsets: torch.Tensor,
    num_groups: int = 64,
    num_threads: int = 0,
    max_chart_faces: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parameterize and pack pre-cut mesh charts into one UV atlas with xatlas, in parallel on the CPU.

    Charts are dealt into num_groups groups of similar area; each group is unwrapped and packed by xatlas on its own
    thread at a shared texel scale, and the group atlases are then tiled into a single atlas. This trades a few
    percent of texel density for a large speedup over packing all charts in one xatlas atlas.

    Args:
        vertices: float32 [V, 3] CPU tensor, the vertices of all charts concatenated.
        faces: int32 [F, 3] CPU tensor of indices into vertices; a chart's faces only reference its own vertices.
        vertex_offsets: integer [num_charts + 1] CPU tensor; chart c owns vertices[vertex_offsets[c]:vertex_offsets[c + 1]].
        face_offsets: integer [num_charts + 1] CPU tensor; chart c owns faces[face_offsets[c]:face_offsets[c + 1]].
        num_groups: number of independent xatlas atlases.
        num_threads: worker threads; 0 uses all hardware threads.
        max_chart_faces: charts with more faces are split before unwrapping; 0 disables splitting.

    Returns:
        (vertex_map int32 [V'], faces int32 [F', 3], uvs float32 [V', 2]). Output vertex i is a copy of
        vertices[vertex_map[i]] with UV uvs[i] in [0, 1]; faces index the output vertices. The result is
        deterministic for fixed inputs and arguments, independent of num_threads.
    """
    return ops.pack_charts(vertices, faces, vertex_offsets, face_offsets, num_groups, num_threads, max_chart_faces)


__all__ = ["mesh_udf", "telea_inpaint", "grid_sample_trilinear", "pack_charts"]
