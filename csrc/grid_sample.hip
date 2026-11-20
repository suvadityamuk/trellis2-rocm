// Trilinear sampling of sparse voxel attributes (texture bake in to_glb).
//
// parity: FlexGEMM's indice_weighed_sum_fwd (Triton) given its neighbor map + weights. Every output element is the
//         sequential fp32 sum over the 8 neighbors starting from 0, independent of the Triton tile, so a plain loop
//         reproduces it. FUSED selects whether each step is one fma or a rounded multiply then add.
// fast:   one kernel: hash lookup of the 8 neighbors, FlexGEMM's weights (normalized by their sum), weighted sum.
#include "common.h"

namespace {

constexpr uint32_t kEmpty = 0xffffffffu;

template <bool FUSED>
__global__ void k_weighted_sum8(const float* __restrict__ feats, const uint32_t* __restrict__ idx, const float* __restrict__ w,
                                i64 M, int C, float* __restrict__ out) {
  i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x;
  if (t >= M * C) return;
  i64 m = t / C; int c = (int)(t % C);
  float acc = 0.f;
#pragma unroll
  for (int v = 0; v < 8; v++) {
    uint32_t j = idx[m * 8 + v];
    float x = j != kEmpty ? feats[(i64)j * C + c] : 0.f;
    float wv = w[m * 8 + v];
    if (FUSED) acc = __builtin_fmaf(x, wv, acc);
    else acc = __fadd_rn(acc, __fmul_rn(x, wv));
  }
  out[t] = acc;
}

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

// q: [M, 3] grid-space points (voxel centers at integer + 0.5). out row stride = ostride (>= C).
template <int C>
__global__ void k_trilinear(const float* __restrict__ feats, const uint32_t* __restrict__ keys, const uint32_t* __restrict__ vals,
                            uint32_t mask, const float* __restrict__ q, i64 M, int W, int H, int D, float* __restrict__ out, int ostride) {
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
  float acc[C];
#pragma unroll
  for (int c = 0; c < C; c++) acc[c] = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    if (n[i] == kEmpty) continue;
    float wi = w[i] * inv;
    const float* f = feats + (i64)n[i] * C;
#pragma unroll
    for (int c = 0; c < C; c++) acc[c] = fmaf(f[c], wi, acc[c]);
  }
#pragma unroll
  for (int c = 0; c < C; c++) out[m * ostride + c] = acc[c];
}

}  // namespace

torch::Tensor weighted_sum8(torch::Tensor feats, torch::Tensor idx, torch::Tensor w, bool fused) {
  T2R_CHECK(feats.is_cuda() && feats.scalar_type() == at::kFloat && feats.dim() == 2, "feats: float32 [N, C]");
  T2R_CHECK(idx.numel() == w.numel() && idx.numel() % 8 == 0, "idx/w: [M, 8]");
  const c10::DeviceGuard g(feats.device());
  auto f = feats.contiguous(); auto I = idx.contiguous(); auto Wt = w.contiguous().to(at::kFloat);
  i64 M = I.numel() / 8; int C = (int)f.size(1);
  auto out = torch::empty({M, C}, f.options());
  if (fused) LAUNCH(k_weighted_sum8<true>, M * C, 256, f.data_ptr<float>(), (const uint32_t*)I.data_ptr(), Wt.data_ptr<float>(), M, C, out.data_ptr<float>());
  else LAUNCH(k_weighted_sum8<false>, M * C, 256, f.data_ptr<float>(), (const uint32_t*)I.data_ptr(), Wt.data_ptr<float>(), M, C, out.data_ptr<float>());
  return out;
}

// feats [N, C] fp32, coords int32 [N, 3] or [N, 4] (b, x, y, z), q [M, 3] grid space -> [M, C]
torch::Tensor grid_sample_trilinear(torch::Tensor feats, torch::Tensor coords, torch::Tensor q, int64_t W, int64_t H, int64_t D) {
  T2R_CHECK(feats.is_cuda() && feats.scalar_type() == at::kFloat && feats.dim() == 2, "feats: float32 [N, C]");
  T2R_CHECK(coords.scalar_type() == at::kInt && coords.dim() == 2 && (coords.size(1) == 3 || coords.size(1) == 4), "coords: int32 [N, 3|4]");
  T2R_CHECK(q.scalar_type() == at::kFloat && q.size(-1) == 3, "q: float32 [..., 3]");
  T2R_CHECK(W * H * D < 0xffffffffll, "grid too large for 32-bit keys");
  const c10::DeviceGuard g(feats.device());
  auto f = feats.contiguous(); auto cc = coords.contiguous(); auto Q = q.reshape({-1, 3}).contiguous();
  i64 N = f.size(0), M = Q.size(0); int C = (int)f.size(1);
  uint32_t cap = 1; while (cap < 2 * N) cap <<= 1;
  auto keys = torch::full({(i64)cap}, -1, cc.options()), vals = torch::empty({(i64)cap}, cc.options());
  LAUNCH(k_hash_insert, N, 256, cc.data_ptr<int>(), (int)cc.size(1), (int)cc.size(1) - 3, N, (int)H, (int)D, cap - 1,
         (uint32_t*)keys.data_ptr(), (uint32_t*)vals.data_ptr());
  auto out = torch::empty({M, C}, f.options());
#define TRI(CC) LAUNCH(k_trilinear<CC>, M, 256, f.data_ptr<float>(), (const uint32_t*)keys.data_ptr(), (const uint32_t*)vals.data_ptr(), \
                       cap - 1, Q.data_ptr<float>(), M, (int)W, (int)H, (int)D, out.data_ptr<float>(), C)
  switch (C) {
    case 6: TRI(6); break;
    case 3: TRI(3); break;
    case 8: TRI(8); break;
    default: T2R_CHECK(false, "grid_sample_trilinear: unsupported C=", C);
  }
#undef TRI
  return out;
}
