// Texture finishing for the fast build: float attributes -> uint8 (numpy clip(x*255).astype(uint8)) -> wavefront TELEA
// inpainting -> packed glTF textures, all on the GPU.
//
// OpenCV's INPAINT_TELEA marches pixels one at a time in fast-marching order. Here every unknown pixel gets an integer
// 4-connected distance level L from the known region, and all pixels of level L are filled in parallel from pixels of
// lower levels with TELEA's weights (direction x distance x level). Adapted from the Apache-2.0 telea kernel in
// suvadityamuk/trellis2-cuda (hub/trellis2-kernels); channels are processed in two groups to keep the CPU radii
// (radius 3 for base color, radius 1 for metallic/roughness/alpha).
#include "common.h"

namespace {

// numpy: np.clip(x * 255, 0, 255).astype(np.uint8)  (float32 multiply, clip, truncate)
__device__ __forceinline__ uint8_t to_u8(float x) {
  float v = __fmul_rn(x, 255.f);
  v = fminf(fmaxf(v, 0.f), 255.f);
  return (uint8_t)(int)v;
}

// attrs [P, Cin] float, layout (base r,g,b, metallic, roughness, alpha) -> planar u8 [6, P]; level init from mask.
__global__ void k_prep(const float* __restrict__ attrs, int Cin, const bool* __restrict__ mask, i64 P, uint8_t* __restrict__ img, int* lv) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= P) return;
  bool m = mask[i];
  for (int c = 0; c < 6; c++) img[c * P + i] = m ? to_u8(attrs[i * Cin + c]) : 0;
  lv[i] = m ? 0 : -1;
}

__global__ void k_mark(int* lv, int H, int W, int L, int* cnt) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= (i64)H * W || lv[i] != -1) return;
  int y = (int)(i / W), x = (int)(i % W), p = L - 1;
  if ((x > 0 && lv[i - 1] == p) || (x < W - 1 && lv[i + 1] == p) || (y > 0 && lv[i - W] == p) || (y < H - 1 && lv[i + W] == p)) {
    lv[i] = L; atomicAdd(cnt, 1);
  }
}

// Fill channels [c0, c0 + C) of planar img at level L with radius R.
template <int C>
__global__ void k_fill(const int* __restrict__ lv, int H, int W, int L, int R, uint8_t* img) {
  i64 P = (i64)H * W, i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= P || lv[i] != L) return;
  int y = (int)(i / W), x = (int)(i % W);
  auto kn = [&](int r, int c) { if (r < 0 || c < 0 || r >= H || c >= W) return false; int v = lv[(i64)r * W + c]; return v >= 0 && v < L; };
  auto tv = [&](int r, int c) { return (float)lv[(i64)r * W + c]; };
  auto I = [&](int ch, int r, int c) { return (float)img[ch * P + (i64)r * W + c]; };
  float gx = kn(y, x + 1) ? (kn(y, x - 1) ? (tv(y, x + 1) - tv(y, x - 1)) * 0.5f : tv(y, x + 1) - L) : (kn(y, x - 1) ? L - tv(y, x - 1) : 0.f);
  float gy = kn(y + 1, x) ? (kn(y - 1, x) ? (tv(y + 1, x) - tv(y - 1, x)) * 0.5f : tv(y + 1, x) - L) : (kn(y - 1, x) ? L - tv(y - 1, x) : 0.f);
  float Ia[C] = {}, Jx[C] = {}, Jy[C] = {}, s = 1e-20f;
  for (int k = y - R; k <= y + R; k++)
    for (int l = x - R; l <= x + R; l++) {
      if ((k - y) * (k - y) + (l - x) * (l - x) > R * R || !kn(k, l)) continue;
      float ry = (float)(y - k), rx = (float)(x - l), vl = rx * rx + ry * ry;
      float dir = rx * gx + ry * gy; if (fabsf(dir) <= 0.01f) dir = 1e-6f;
      float w = fabsf(dir / (vl * sqrtf(vl) * (1 + fabsf(tv(k, l) - L))));
      bool r1 = kn(k, l + 1), l1 = kn(k, l - 1), d1 = kn(k + 1, l), u1 = kn(k - 1, l);
#pragma unroll
      for (int c = 0; c < C; c++) {
        float v = I(c, k, l);
        float ix = r1 ? (l1 ? (I(c, k, l + 1) - I(c, k, l - 1)) * 2.f : I(c, k, l + 1) - v) : (l1 ? v - I(c, k, l - 1) : 0.f);
        float iy = d1 ? (u1 ? (I(c, k + 1, l) - I(c, k - 1, l)) * 2.f : I(c, k + 1, l) - v) : (u1 ? v - I(c, k - 1, l) : 0.f);
        Ia[c] += w * v; Jx[c] -= w * ix * rx; Jy[c] -= w * iy * ry;
      }
      s += w;
    }
#pragma unroll
  for (int c = 0; c < C; c++) {
    float sat = Ia[c] / s + (Jx[c] + Jy[c]) / (sqrtf(Jx[c] * Jx[c] + Jy[c] * Jy[c]) + 1e-20f);
    img[c * P + i] = (uint8_t)fminf(fmaxf(rintf(sat + 0.5f), 0.f), 255.f);
  }
}

// planar [6, P] -> baseColor RGBA [P, 4] (r, g, b, alpha) and metallicRoughness [P, 3] (0, roughness, metallic)
__global__ void k_pack(const uint8_t* __restrict__ img, i64 P, uint8_t* __restrict__ base, uint8_t* __restrict__ mr) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= P) return;
  base[4 * i] = img[i]; base[4 * i + 1] = img[P + i]; base[4 * i + 2] = img[2 * P + i]; base[4 * i + 3] = img[5 * P + i];
  mr[3 * i] = 0; mr[3 * i + 1] = img[4 * P + i]; mr[3 * i + 2] = img[3 * P + i];
}

}  // namespace

// attrs [H, W, 6] float32, mask [H, W] bool -> (baseColor RGBA u8 [H, W, 4], metallicRoughness u8 [H, W, 3])
std::vector<torch::Tensor> bake_textures(torch::Tensor attrs, torch::Tensor mask, int64_t r_base, int64_t r_mr) {
  T2R_CHECK(attrs.is_cuda() && attrs.scalar_type() == at::kFloat && attrs.dim() == 3 && attrs.size(2) >= 6, "attrs: float32 [H, W, >=6]");
  T2R_CHECK(mask.scalar_type() == at::kBool && mask.dim() == 2, "mask: bool [H, W]");
  const c10::DeviceGuard g(attrs.device());
  int H = (int)attrs.size(0), W = (int)attrs.size(1), Cin = (int)attrs.size(2);
  i64 P = (i64)H * W;
  auto a = attrs.contiguous(); auto m = mask.contiguous();
  auto u8 = a.options().dtype(at::kByte), i32 = a.options().dtype(at::kInt);
  auto img = torch::empty({6, P}, u8), lv = torch::empty({P}, i32), cnt = torch::zeros({1}, i32);
  auto cnt_host = torch::empty({1}, i32.device(at::kCPU).pinned_memory(true));
  LAUNCH(k_prep, P, 256, a.data_ptr<float>(), Cin, m.data_ptr<bool>(), P, img.data_ptr<uint8_t>(), lv.data_ptr<int>());
  for (int L = 1;; L++) {
    HIPCHK(hipMemsetAsync(cnt.data_ptr<int>(), 0, sizeof(int), cur_stream()));
    LAUNCH(k_mark, P, 256, lv.data_ptr<int>(), H, W, L, cnt.data_ptr<int>());
    LAUNCH(k_fill<3>, P, 256, lv.data_ptr<int>(), H, W, L, (int)r_base, img.data_ptr<uint8_t>());
    LAUNCH(k_fill<3>, P, 256, lv.data_ptr<int>(), H, W, L, (int)r_mr, img.data_ptr<uint8_t>() + 3 * P);
    HIPCHK(hipMemcpyAsync(cnt_host.data_ptr<int>(), cnt.data_ptr<int>(), sizeof(int), hipMemcpyDeviceToHost, cur_stream()));
    HIPCHK(hipStreamSynchronize(cur_stream()));
    if (cnt_host.data_ptr<int>()[0] == 0) break;
  }
  auto base = torch::empty({H, W, 4}, u8), mr = torch::empty({H, W, 3}, u8);
  LAUNCH(k_pack, P, 256, img.data_ptr<uint8_t>(), P, base.data_ptr<uint8_t>(), mr.data_ptr<uint8_t>());
  return {base, mr};
}
