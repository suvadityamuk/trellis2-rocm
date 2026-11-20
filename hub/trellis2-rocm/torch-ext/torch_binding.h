#pragma once

#include <tuple>
#include <vector>

#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mesh_udf(const torch::Tensor& vertices, const torch::Tensor& faces, const torch::Tensor& points);
torch::Tensor telea_inpaint(const torch::Tensor& image, const torch::Tensor& mask, int64_t radius);
torch::Tensor grid_sample_trilinear(const torch::Tensor& feats, const torch::Tensor& coords, const torch::Tensor& points, std::vector<int64_t> grid_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pack_charts(const torch::Tensor& vertices, const torch::Tensor& faces, const torch::Tensor& vertex_offsets,
                                                                    const torch::Tensor& face_offsets, int64_t num_groups, int64_t num_threads, int64_t max_chart_faces);
