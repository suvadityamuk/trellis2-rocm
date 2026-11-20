// Trilinear sampling of features stored on a sparse voxel grid, in one kernel: hash lookup of the 8 neighbouring
// voxels, trilinear weights normalized over the neighbours that exist, weighted sum.
#include "common.h"

namespace {

constexpr uint32_t kEmpty = 0xffffffffu;

// Open-addressing hash of voxel keys (x*H*D + y*D + z), linear probing, capacity a power of two.
__device__ __forceinline__ uint32_t hash32(uint32_t k) {
  k ^= k >> 16; k *= 0x85ebca6bu; k ^= k >> 13; k *= 0xc2b2ae35u; k ^= k >> 16; return k;
}
__global__ void k_hash_insert(const int* __restrict__ coords, int cstride, int coff, i64 N, int H, int D, uint32_t mask,
                              uint32_t* keys, uint32_t* vals) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= N) return;
  const int* c = coords + i * cstride + coff;
  uint32_t key = ((uint32_t)c[0] * H + (uint32_t)c[1]) * D + (uint32_t)c[2];
  uint32_t s = hash32(key) & mask;
  while (true) {
    uint32_t prev = atomicCAS(&keys[s], kEmpty, key);
    if (prev == kEmpty || prev == key) { vals[s] = (uint32_t)i; return; }
    s = (s + 1) & mask;
  }
}
__device__ __forceinline__ uint32_t hash_find(const uint32_t* keys, const uint32_t* vals, uint32_t mask, uint32_t key) {
  uint32_t s = hash32(key) & mask;
  while (true) {
    uint32_t k = keys[s];
    if (k == key) return vals[s];
    if (k == kEmpty) return kEmpty;
    s = (s + 1) & mask;
  }
}

// q: [M, 3] grid-space points (voxel centers at integer + 0.5). CT > 0 fixes the channel count at compile time.
template <int CT>
__global__ void k_trilinear(const float* __restrict__ feats, const uint32_t* __restrict__ keys, const uint32_t* __restrict__ vals,
                            uint32_t mask, const float* __restrict__ q, i64 M, int W, int H, int D, float* __restrict__ out, int Cdyn) {
  const int C = CT > 0 ? CT : Cdyn;
  i64 m = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (m >= M) return;
  float qx = q[3 * m], qy = q[3 * m + 1], qz = q[3 * m + 2];
  int bx = (int)floorf(qx - 0.5f), by = (int)floorf(qy - 0.5f), bz = (int)floorf(qz - 0.5f);
  uint32_t n[8]; float w[8]; float wsum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int x = bx + (i & 1), y = by + ((i >> 1) & 1), z = bz + ((i >> 2) & 1);
    n[i] = kEmpty; w[i] = 0.f;
    if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
      uint32_t v = hash_find(keys, vals, mask, ((uint32_t)x * H + (uint32_t)y) * D + (uint32_t)z);
      if (v != kEmpty) {
        n[i] = v;
        w[i] = (1.f - fabsf(qx - x - 0.5f)) * (1.f - fabsf(qy - y - 0.5f)) * (1.f - fabsf(qz - z - 0.5f));
        wsum += w[i];
      }
    }
  }
  float inv = 1.f / fmaxf(wsum, 1e-12f);
  float* o = out + m * C;
  if (CT > 0) {
    float acc[CT > 0 ? CT : 1];
#pragma unroll
    for (int c = 0; c < CT; c++) acc[c] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if (n[i] == kEmpty) continue;
      float wi = w[i] * inv;
      const float* f = feats + (i64)n[i] * C;
#pragma unroll
      for (int c = 0; c < CT; c++) acc[c] = fmaf(f[c], wi, acc[c]);
    }
#pragma unroll
    for (int c = 0; c < CT; c++) o[c] = acc[c];
  } else {  // any channel count: same per-channel operation order as the fixed-size path
    for (int c = 0; c < C; c++) {
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) if (n[i] != kEmpty) acc = fmaf(feats[(i64)n[i] * C + c], w[i] * inv, acc);
      o[c] = acc;
    }
  }
}

}  // namespace

// feats float32 [N, C], coords int32 [N, 3] (x, y, z) or [N, 4] (batch, x, y, z; batch ignored), points float32 [M, 3]
// in grid space (voxel (i, j, k) spans [i, i+1) x [j, j+1) x [k, k+1)), grid_size (W, H, D) -> float32 [M, C].
// Points with no occupied neighbour get zeros.
torch::Tensor grid_sample_trilinear(const torch::Tensor& feats, const torch::Tensor& coords, const torch::Tensor& points, std::vector<int64_t> grid_size) {
  TORCH_CHECK(feats.is_cuda() && feats.scalar_type() == at::kFloat && feats.dim() == 2, "feats: float32 [N, C] on the GPU");
  TORCH_CHECK(coords.scalar_type() == at::kInt && coords.dim() == 2 && (coords.size(1) == 3 || coords.size(1) == 4), "coords: int32 [N, 3] or [N, 4]");
  TORCH_CHECK(points.scalar_type() == at::kFloat && points.dim() == 2 && points.size(1) == 3, "points: float32 [M, 3]");
  TORCH_CHECK(coords.device() == feats.device() && points.device() == feats.device(), "all inputs must be on the same device");
  TORCH_CHECK(coords.size(0) == feats.size(0), "feats and coords must have the same number of rows");
  TORCH_CHECK(grid_size.size() == 3, "grid_size: (W, H, D)");
  const i64 W = grid_size[0], H = grid_size[1], D = grid_size[2];
  TORCH_CHECK(W > 0 && H > 0 && D > 0 && W < INT_MAX && H < INT_MAX && D < INT_MAX && (double)W * H * D < 4294967295.0, "grid_size must be positive with W*H*D < 2^32 - 1");
  const c10::DeviceGuard g(feats.device());
  auto f = feats.contiguous(); auto cc = coords.contiguous(); auto Q = points.contiguous();
  const i64 N = f.size(0), M = Q.size(0); const int C = (int)f.size(1);
  TORCH_CHECK(N < (1ll << 30), "too many voxels");
  auto out = torch::zeros({M, C}, f.options());
  if (N == 0 || M == 0 || C == 0) return out;
  uint32_t cap = 1; while (cap < 2 * N) cap <<= 1;
  auto keys = torch::full({(i64)cap}, -1, cc.options()), vals = torch::empty({(i64)cap}, cc.options());
  LAUNCH(k_hash_insert, N, 256, cc.data_ptr<int>(), (int)cc.size(1), (int)cc.size(1) - 3, N, (int)H, (int)D, cap - 1,
         (uint32_t*)keys.data_ptr(), (uint32_t*)vals.data_ptr());
#define TRI(CC) LAUNCH(k_trilinear<CC>, M, 256, f.data_ptr<float>(), (const uint32_t*)keys.data_ptr(), (const uint32_t*)vals.data_ptr(), \
                       cap - 1, Q.data_ptr<float>(), M, (int)W, (int)H, (int)D, out.data_ptr<float>(), C)
  switch (C) {
    case 1: TRI(1); break; case 2: TRI(2); break; case 3: TRI(3); break; case 4: TRI(4); break;
    case 6: TRI(6); break; case 8: TRI(8); break;
    default: TRI(0);
  }
#undef TRI
  return out;
}
