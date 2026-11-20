// Fused DiT kernels for the fast build (bf16 activations, fp32 math). Rows are tokens of a CFG batch: row r belongs
// to batch item r / N, whose modulation vectors are mods[b, j, :] (j: 0 shift_msa, 1 scale_msa, 2 gate_msa,
// 3 shift_mlp, 4 scale_mlp, 5 gate_mlp). All row kernels assume C == 1536 and 256 threads (4 wave64s) per row.
#include "common.h"
#include <hip/hip_bf16.h>

namespace {

typedef __hip_bfloat16 bf16;
constexpr int C = 1536, H = 12, D = 128, T = 256, PER = C / T;  // 6 elements per thread

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}
// Block-wide sum for 256 threads (4 waves).
__device__ __forceinline__ float block_sum(float v, float* sh) {
  v = wave_sum(v);
  int w = threadIdx.x >> 6, l = threadIdx.x & 63;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  return sh[0] + sh[1] + sh[2] + sh[3];
}

// Optional residual update x += a * gate (gate may be null), then LayerNorm (eps 1e-6, optional affine), then
// optional modulate y * (1 + scale) + shift. Writes the updated residual back to x and the normalized row to y.
template <bool RES, bool GATE, bool AFFINE, bool MOD>
__global__ void __launch_bounds__(T) k_row(bf16* __restrict__ x, const bf16* __restrict__ a, const bf16* __restrict__ mods, int N,
                                           int gate_j, int shift_j, const float* __restrict__ w, const float* __restrict__ bb,
                                           bf16* __restrict__ y) {
  __shared__ float sh[4];
  const int64_t r = blockIdx.x; const int b = (int)(r / N);
  bf16* xr = x + r * C;
  const bf16* mb = mods + (int64_t)b * 6 * C;
  float v[PER];
#pragma unroll
  for (int i = 0; i < PER; i++) {
    int c = threadIdx.x + i * T;
    float xv = b2f(xr[c]);
    if (RES) {
      float av = b2f(a[r * C + c]);
      xv += GATE ? av * b2f(mb[gate_j * C + c]) : av;
      bf16 q = f2b(xv); xr[c] = q; xv = b2f(q);  // the residual stream is bf16
    }
    v[i] = xv;
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < PER; i++) s += v[i];
  const float mean = block_sum(s, sh) * (1.f / C);
  float s2 = 0.f;
#pragma unroll
  for (int i = 0; i < PER; i++) { float d = v[i] - mean; s2 += d * d; }
  const float rstd = rsqrtf(block_sum(s2, sh) * (1.f / C) + 1e-6f);
#pragma unroll
  for (int i = 0; i < PER; i++) {
    int c = threadIdx.x + i * T;
    float o = (v[i] - mean) * rstd;
    if (AFFINE) o = o * w[c] + bb[c];
    if (MOD) o = o * (1.f + b2f(mb[(shift_j + 1) * C + c])) + b2f(mb[shift_j * C + c]);
    y[r * C + c] = f2b(o);
  }
}

// x += a * gate (elementwise, vectorized by 2)
__global__ void k_gate_res(bf16* __restrict__ x, const bf16* __restrict__ a, const bf16* __restrict__ mods, int N, int gate_j, int64_t n) {
  int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) * 2; if (i >= n) return;
  int64_t r = i / C; int c = (int)(i % C); int b = (int)(r / N);
  const bf16* g = mods + ((int64_t)b * 6 + gate_j) * C + c;
  x[i] = f2b(b2f(x[i]) + b2f(a[i]) * b2f(g[0]));
  x[i + 1] = f2b(b2f(x[i + 1]) + b2f(a[i + 1]) * b2f(g[1]));
}

// qkv [R, 3, H, D] (row-major [R, 3C]) -> packed [R, 3, H, D]: q, k RMS-normalized per head (x / max(|x|, 1e-12) *
// gamma * scale), then rotated by phases[r % N] (64 complex pairs per head); v copied. One wave per head, 2 lanes'
// worth (one complex pair) per lane. Block: 256 threads = 4 heads at a time; grid.y walks the 36 (q|k|v, head) slots.
__global__ void __launch_bounds__(T) k_qk_rope(const bf16* __restrict__ qkv, const float* __restrict__ qg, const float* __restrict__ kg,
                                               float scale, const float2* __restrict__ ph, int N, bf16* __restrict__ out) {
  const int64_t r = blockIdx.x;
  const int slot = blockIdx.y * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;  // slot in [0, 36)
  const int part = slot / H, h = slot % H;
  const int64_t base = r * 3 * C + part * C + h * D + 2 * lane;
  float x0 = b2f(qkv[base]), x1 = b2f(qkv[base + 1]);
  if (part == 2) { out[base] = f2b(x0); out[base + 1] = f2b(x1); return; }
  float nrm = sqrtf(wave_sum(x0 * x0 + x1 * x1));
  float inv = 1.f / fmaxf(nrm, 1e-12f);
  const float* g = (part == 0 ? qg : kg) + h * D + 2 * lane;
  float a = x0 * inv * g[0] * scale, bq = x1 * inv * g[1] * scale;
  // the official casts the normalized q/k to bf16 before RoPE
  a = b2f(f2b(a)); bq = b2f(f2b(bq));
  float2 p = ph[(r % N) * 64 + lane];
  out[base] = f2b(a * p.x - bq * p.y);
  out[base + 1] = f2b(a * p.y + bq * p.x);
}

// x [R, H*D] -> RMS-normalized per head (same formula), out [R, H, D]
__global__ void __launch_bounds__(T) k_rms_heads(const bf16* __restrict__ x, int64_t xstride, const float* __restrict__ g, float scale, bf16* __restrict__ out) {
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
  const bf16* xr = x + r * xstride + h * D + 2 * lane;
  float x0 = b2f(xr[0]), x1 = b2f(xr[1]);
  float inv = 1.f / fmaxf(sqrtf(wave_sum(x0 * x0 + x1 * x1)), 1e-12f);
  bf16* o = out + r * C + h * D + 2 * lane;
  o[0] = f2b(x0 * inv * g[h * D + 2 * lane] * scale);
  o[1] = f2b(x1 * inv * g[h * D + 2 * lane + 1] * scale);
}

// kv [L, 2C] -> v contiguous [L, H, D] (k is normalized by k_rms_heads)
__global__ void k_copy_v(const bf16* __restrict__ kv, int64_t L, bf16* __restrict__ v) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i >= L * C) return;
  v[i] = kv[(i / C) * 2 * C + C + i % C];
}

}  // namespace

static void chk(const torch::Tensor& t, at::ScalarType s, const char* n) {
  T2R_CHECK(t.is_cuda() && t.scalar_type() == s && t.is_contiguous(), n, ": expected contiguous ", s);
}
#define BF(t) reinterpret_cast<bf16*>((t).data_ptr())
#define CBF(t) reinterpret_cast<const bf16*>((t).data_ptr())

// y = LN(x) [*w + b] [* (1 + mods[:, shift_j+1]) + mods[:, shift_j]]
torch::Tensor ln_mod(torch::Tensor x, torch::Tensor mods, int64_t shift_j, int64_t N, c10::optional<torch::Tensor> w, c10::optional<torch::Tensor> b) {
  chk(x, at::kBFloat16, "x"); chk(mods, at::kBFloat16, "mods"); T2R_CHECK(x.size(-1) == C, "C must be 1536");
  const c10::DeviceGuard g(x.device());
  auto y = torch::empty_like(x); int64_t R = x.numel() / C;
  bool aff = w.has_value(); const float* wp = aff ? w->data_ptr<float>() : nullptr; const float* bp = aff ? b->data_ptr<float>() : nullptr;
  if (R == 0) return y;
  if (aff) hipLaunchKernelGGL((k_row<false, false, true, false>), dim3(R), dim3(T), 0, cur_stream(), BF(x), nullptr, CBF(mods), (int)N, 0, 0, wp, bp, BF(y));
  else hipLaunchKernelGGL((k_row<false, false, false, true>), dim3(R), dim3(T), 0, cur_stream(), BF(x), nullptr, CBF(mods), (int)N, 0, (int)shift_j, wp, bp, BF(y));
  HIPCHK(hipGetLastError());
  return y;
}
// x += a * mods[:, gate_j]; return LN(x) * w + b    (self-attention residual + norm2)
torch::Tensor gate_res_ln(torch::Tensor x, torch::Tensor a, torch::Tensor mods, int64_t gate_j, int64_t N, torch::Tensor w, torch::Tensor b) {
  chk(x, at::kBFloat16, "x"); chk(a, at::kBFloat16, "a"); chk(mods, at::kBFloat16, "mods");
  const c10::DeviceGuard g(x.device());
  auto y = torch::empty_like(x); int64_t R = x.numel() / C;
  if (R) hipLaunchKernelGGL((k_row<true, true, true, false>), dim3(R), dim3(T), 0, cur_stream(), BF(x), CBF(a), CBF(mods), (int)N, (int)gate_j, 0,
                            w.data_ptr<float>(), b.data_ptr<float>(), BF(y));
  HIPCHK(hipGetLastError());
  return y;
}
// x += a; return LN(x) * (1 + scale_mlp) + shift_mlp     (cross-attention residual + norm3 + modulate)
torch::Tensor res_ln_mod(torch::Tensor x, torch::Tensor a, torch::Tensor mods, int64_t N) {
  chk(x, at::kBFloat16, "x"); chk(a, at::kBFloat16, "a"); chk(mods, at::kBFloat16, "mods");
  const c10::DeviceGuard g(x.device());
  auto y = torch::empty_like(x); int64_t R = x.numel() / C;
  if (R) hipLaunchKernelGGL((k_row<true, false, false, true>), dim3(R), dim3(T), 0, cur_stream(), BF(x), CBF(a), CBF(mods), (int)N, 0, 3, nullptr, nullptr, BF(y));
  HIPCHK(hipGetLastError());
  return y;
}
void gate_res(torch::Tensor x, torch::Tensor a, torch::Tensor mods, int64_t gate_j, int64_t N) {
  chk(x, at::kBFloat16, "x"); chk(a, at::kBFloat16, "a"); chk(mods, at::kBFloat16, "mods");
  const c10::DeviceGuard g(x.device());
  int64_t n = x.numel();
  LAUNCH(k_gate_res, n / 2, 256, BF(x), CBF(a), CBF(mods), (int)N, (int)gate_j, n);
}
// qkv [R, 3C] -> packed normalized/rotated [R, 3, H, D]; phases: complex64 [N, 64]
torch::Tensor qk_rms_rope_pack(torch::Tensor qkv, torch::Tensor qg, torch::Tensor kg, double scale, torch::Tensor phases, int64_t N) {
  chk(qkv, at::kBFloat16, "qkv"); chk(qg, at::kFloat, "qg"); chk(kg, at::kFloat, "kg");
  T2R_CHECK(phases.is_complex() && phases.size(-1) == 64, "phases: complex64 [N, 64]");
  const c10::DeviceGuard g(qkv.device());
  int64_t R = qkv.numel() / (3 * C);
  auto out = torch::empty({R, 3, H, D}, qkv.options());
  auto ph = torch::view_as_real(phases.contiguous()).contiguous();
  if (R) hipLaunchKernelGGL(k_qk_rope, dim3(R, 9), dim3(T), 0, cur_stream(), CBF(qkv), qg.data_ptr<float>(), kg.data_ptr<float>(), (float)scale,
                            reinterpret_cast<const float2*>(ph.data_ptr<float>()), (int)N, BF(out));
  HIPCHK(hipGetLastError());
  return out;
}
torch::Tensor rms_heads(torch::Tensor x, torch::Tensor g, double scale) {
  T2R_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(-1) == 1, "x: bf16 rows");
  const c10::DeviceGuard gd(x.device());
  int64_t R = x.numel() / x.size(-1);
  auto out = torch::empty({R, H, D}, x.options());
  if (R) hipLaunchKernelGGL(k_rms_heads, dim3(R, 3), dim3(T), 0, cur_stream(), CBF(x), x.size(-1), g.data_ptr<float>(), (float)scale, BF(out));
  HIPCHK(hipGetLastError());
  return out;
}
// kv [L, 2C] -> (normalized k [L, H, D], v [L, H, D])
std::vector<torch::Tensor> kv_split_rms(torch::Tensor kv, torch::Tensor g, double scale) {
  chk(kv, at::kBFloat16, "kv");
  const c10::DeviceGuard gd(kv.device());
  int64_t L = kv.size(0);
  auto k = torch::empty({L, H, D}, kv.options()), v = torch::empty({L, H, D}, kv.options());
  if (L) {
    hipLaunchKernelGGL(k_rms_heads, dim3(L, 3), dim3(T), 0, cur_stream(), CBF(kv), 2 * C, g.data_ptr<float>(), (float)scale, BF(k));
    HIPCHK(hipGetLastError());
    LAUNCH(k_copy_v, L * C, 256, CBF(kv), L, BF(v));
  }
  return {k, v};
}
