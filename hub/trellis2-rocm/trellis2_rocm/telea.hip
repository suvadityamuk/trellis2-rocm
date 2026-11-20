// TELEA inpainting on the GPU, an alternative to cv2.inpaint(..., cv2.INPAINT_TELEA).
//
// OpenCV marches pixels one at a time in fast-marching order. Here every unknown pixel gets an integer 4-connected
// distance level L from the known region, and all pixels of level L are filled in parallel from pixels of lower levels
// with TELEA's weights (direction x distance x level). Adapted from the Apache-2.0 telea kernel in
// suvadityamuk/trellis2-cuda (hub/trellis2-kernels).
#include "common.h"
#include <climits>

namespace {

template <class M>
__global__ void k_init(const M* __restrict__ mask, i64 P, int* lv) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < P) lv[i] = mask[i] != 0 ? -1 : 0;
}

__global__ void k_mark(int* lv, int H, int W, int L, int* cnt) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= (i64)H * W || lv[i] != -1) return;
  int y = (int)(i / W), x = (int)(i % W), p = L - 1;
  if ((x > 0 && lv[i - 1] == p) || (x < W - 1 && lv[i + 1] == p) || (y > 0 && lv[i - W] == p) || (y < H - 1 && lv[i + W] == p)) {
    lv[i] = L; atomicAdd(cnt, 1);
  }
}

// Fill the C interleaved channels of every level-L pixel of img [H, W, C] with radius R.
template <int C>
__global__ void k_fill(const int* __restrict__ lv, int H, int W, int L, int R, uint8_t* img) {
  i64 P = (i64)H * W, i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= P || lv[i] != L) return;
  int y = (int)(i / W), x = (int)(i % W);
  auto kn = [&](int r, int c) { if (r < 0 || c < 0 || r >= H || c >= W) return false; int v = lv[(i64)r * W + c]; return v >= 0 && v < L; };
  auto tv = [&](int r, int c) { return (float)lv[(i64)r * W + c]; };
  auto I = [&](int ch, int r, int c) { return (float)img[((i64)r * W + c) * C + ch]; };
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
    img[i * C + c] = (uint8_t)fminf(fmaxf(rintf(sat + 0.5f), 0.f), 255.f);
  }
}

}  // namespace

// image uint8 [H, W] or [H, W, C] (C <= 4), mask [H, W] bool or uint8 (nonzero = inpaint) -> inpainted copy of image.
// Masked pixels that no known pixel can reach (an all-masked image) keep their input values.
torch::Tensor telea_inpaint(const torch::Tensor& image, const torch::Tensor& mask, int64_t radius) {
  TORCH_CHECK(image.is_cuda() && image.scalar_type() == at::kByte && (image.dim() == 2 || image.dim() == 3), "image: uint8 [H, W] or [H, W, C] on the GPU");
  TORCH_CHECK(mask.device() == image.device() && mask.dim() == 2 && (mask.scalar_type() == at::kBool || mask.scalar_type() == at::kByte),
              "mask: bool or uint8 [H, W] on the same device as image");
  const i64 H = image.size(0), W = image.size(1); const int C = image.dim() == 3 ? (int)image.size(2) : 1;
  TORCH_CHECK(C >= 1 && C <= 4, "image: 1 to 4 channels, got ", C);
  TORCH_CHECK(mask.size(0) == H && mask.size(1) == W, "mask and image must have the same height and width");
  TORCH_CHECK(radius >= 1 && radius <= 64, "radius must be in [1, 64]");
  TORCH_CHECK(H < INT_MAX && W < INT_MAX, "image too large");
  const c10::DeviceGuard g(image.device());
  auto out = image.contiguous().clone(); auto m = mask.contiguous();
  const i64 P = H * W; if (P == 0) return out;
  auto i32 = out.options().dtype(at::kInt);
  auto lv = torch::empty({P}, i32), cnt = torch::zeros({1}, i32);
  auto cnt_host = torch::empty({1}, i32.device(at::kCPU).pinned_memory(true));
  if (m.scalar_type() == at::kBool) LAUNCH(k_init<bool>, P, 256, m.data_ptr<bool>(), P, lv.data_ptr<int>());
  else LAUNCH(k_init<uint8_t>, P, 256, m.data_ptr<uint8_t>(), P, lv.data_ptr<int>());
  for (int L = 1;; L++) {
    HIPCHK(hipMemsetAsync(cnt.data_ptr<int>(), 0, sizeof(int), cur_stream()));
    LAUNCH(k_mark, P, 256, lv.data_ptr<int>(), (int)H, (int)W, L, cnt.data_ptr<int>());
#define FILL(CC) LAUNCH(k_fill<CC>, P, 256, lv.data_ptr<int>(), (int)H, (int)W, L, (int)radius, out.data_ptr<uint8_t>())
    switch (C) { case 1: FILL(1); break; case 2: FILL(2); break; case 3: FILL(3); break; default: FILL(4); }
#undef FILL
    HIPCHK(hipMemcpyAsync(cnt_host.data_ptr<int>(), cnt.data_ptr<int>(), sizeof(int), hipMemcpyDeviceToHost, cur_stream()));
    HIPCHK(hipStreamSynchronize(cur_stream()));
    if (cnt_host.data_ptr<int>()[0] == 0) break;
  }
  return out;
}
