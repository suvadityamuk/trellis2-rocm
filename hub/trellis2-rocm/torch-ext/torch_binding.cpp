#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("mesh_udf(Tensor vertices, Tensor faces, Tensor points) -> (Tensor, Tensor, Tensor)");
  ops.impl("mesh_udf", torch::kCUDA, &mesh_udf);
  ops.def("telea_inpaint(Tensor image, Tensor mask, int radius) -> Tensor");
  ops.impl("telea_inpaint", torch::kCUDA, &telea_inpaint);
  ops.def("grid_sample_trilinear(Tensor feats, Tensor coords, Tensor points, int[] grid_size) -> Tensor");
  ops.impl("grid_sample_trilinear", torch::kCUDA, &grid_sample_trilinear);
  // xatlas packing runs on the host.
  ops.def("pack_charts(Tensor vertices, Tensor faces, Tensor vertex_offsets, Tensor face_offsets, int num_groups, int num_threads, int max_chart_faces) -> (Tensor, Tensor, Tensor)");
  ops.impl("pack_charts", torch::kCPU, &pack_charts);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
