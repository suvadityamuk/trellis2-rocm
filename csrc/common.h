// Shared helpers for the TRELLIS.2 ROCm kernels.
#pragma once
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include <cstdint>

#define T2R_CHECK(cond, ...) TORCH_CHECK(cond, __VA_ARGS__)
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
