#include <torch/extension.h>
#include <vector>

torch::Tensor weighted_sum8(torch::Tensor feats, torch::Tensor idx, torch::Tensor w, bool fused);
torch::Tensor grid_sample_trilinear(torch::Tensor feats, torch::Tensor coords, torch::Tensor q, int64_t W, int64_t H, int64_t D);
std::vector<torch::Tensor> bake_textures(torch::Tensor attrs, torch::Tensor mask, int64_t r_base, int64_t r_mr);
void bind_lbvh(pybind11::module& m);
torch::Tensor ln_mod(torch::Tensor x, torch::Tensor mods, int64_t shift_j, int64_t N, c10::optional<torch::Tensor> w, c10::optional<torch::Tensor> b);
torch::Tensor gate_res_ln(torch::Tensor x, torch::Tensor a, torch::Tensor mods, int64_t gate_j, int64_t N, torch::Tensor w, torch::Tensor b);
torch::Tensor res_ln_mod(torch::Tensor x, torch::Tensor a, torch::Tensor mods, int64_t N);
void gate_res(torch::Tensor x, torch::Tensor a, torch::Tensor mods, int64_t gate_j, int64_t N);
torch::Tensor qk_rms_rope_pack(torch::Tensor qkv, torch::Tensor qg, torch::Tensor kg, double scale, torch::Tensor phases, int64_t N);
torch::Tensor rms_heads(torch::Tensor x, torch::Tensor g, double scale);
std::vector<torch::Tensor> kv_split_rms(torch::Tensor kv, torch::Tensor g, double scale);
std::vector<torch::Tensor> uv_pack_groups(torch::Tensor cv, torch::Tensor cf, torch::Tensor vo, torch::Tensor fo, int64_t ngroups, int64_t nthreads, int64_t max_faces);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weighted_sum8", &weighted_sum8, "FlexGEMM indice_weighed_sum_fwd equivalent (8 neighbors)");
  m.def("grid_sample_trilinear", &grid_sample_trilinear, "fused sparse-voxel trilinear sampling");
  m.def("bake_textures", &bake_textures, "uint8 conversion + wavefront TELEA + glTF texture packing");
  m.def("uv_pack_groups", &uv_pack_groups, "parallel grouped xatlas parameterization + packing");
  m.def("ln_mod", &ln_mod); m.def("gate_res_ln", &gate_res_ln); m.def("res_ln_mod", &res_ln_mod); m.def("gate_res", &gate_res);
  m.def("qk_rms_rope_pack", &qk_rms_rope_pack); m.def("rms_heads", &rms_heads); m.def("kv_split_rms", &kv_split_rms);
  bind_lbvh(m);
}
