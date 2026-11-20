// Shared helpers for the trellis2-rocm kernels.
#pragma once
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/torch.h>
#include <cstdint>

// The kernels spell out every fused multiply-add they want (fmaf); keep the compiler from adding more.
#pragma STDC FP_CONTRACT OFF

#define HIPCHK(x) do { hipError_t e_ = (x); TORCH_CHECK(e_ == hipSuccess, "HIP error ", hipGetErrorString(e_), " at ", __FILE__, ":", __LINE__); } while (0)

using i64 = int64_t;

inline hipStream_t cur_stream() { return at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream(); }
inline int cdiv(i64 n, int b) { return (int)((n + b - 1) / b); }

#define LAUNCH(kernel, n, block, ...)                                                          \
  do {                                                                                        \
    if ((n) > 0) {                                                                            \
      hipLaunchKernelGGL(kernel, dim3(cdiv((n), (block))), dim3(block), 0, cur_stream(), __VA_ARGS__); \
      HIPCHK(hipGetLastError());                                                              \
    }                                                                                         \
  } while (0)
