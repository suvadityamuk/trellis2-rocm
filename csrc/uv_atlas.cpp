// Fast-build UV atlas: CuMesh charts -> xatlas parameterization + packing, run as many independent group atlases in
// parallel native threads, then tiled into one atlas.
//
// The official uv_unwrap adds ~25k CuMesh charts to a single xatlas Atlas from Python (one AddMesh call each) and
// packs them serially (4096 random placement attempts per chart). Here charts are dealt round-robin by area into
// groups; every group runs AddMesh/ComputeCharts/PackCharts single-threaded on its own thread with one shared
// texelsPerUnit (xatlas's own 1024^2-at-75%-utilization estimate over the whole mesh), so texel density is uniform.
// The group atlases are then shelf-packed into one atlas and UVs are normalized by its size, as CuMesh's binding does.
#include <torch/extension.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>
#include "third_party/xatlas/xatlas.h"

extern std::atomic<unsigned> t2r_xa_thread_override;

namespace {
struct GroupOut {
  std::vector<int32_t> vmap;  // global chart-vertex index
  std::vector<int32_t> faces; // group-local output vertex indices
  std::vector<float> uv;      // pixels in the group atlas
  int w = 0, h = 0;
  std::string err;
};
}  // namespace

// cv: [V, 3] float32 chart vertices (concatenated), cf: [F, 3] int32 faces indexing cv (as CuMesh read_atlas_charts
// returns them), vo/fo: [nc+1] chart vertex/face offsets.
// -> (vmap int32 [V'] into cv rows, faces int32 [F', 3], uvs float32 [V', 2] in [0, 1])
std::vector<torch::Tensor> uv_pack_groups(torch::Tensor cv, torch::Tensor cf, torch::Tensor vo, torch::Tensor fo, int64_t ngroups, int64_t nthreads, int64_t max_faces) {
  TORCH_CHECK(cv.device().is_cpu() && cv.scalar_type() == at::kFloat && cv.is_contiguous(), "cv: CPU float32 [V, 3]");
  TORCH_CHECK(cf.device().is_cpu() && cf.scalar_type() == at::kInt && cf.is_contiguous(), "cf: CPU int32 [F, 3]");
  auto VO = vo.to(at::kLong).contiguous(), FO = fo.to(at::kLong).contiguous();
  const int64_t nc = VO.numel() - 1;
  const float* P = cv.data_ptr<float>(); const int32_t* Fi = cf.data_ptr<int32_t>();
  const int64_t* vop = VO.data_ptr<int64_t>(); const int64_t* fop = FO.data_ptr<int64_t>();

  // Pieces: charts with more than max_faces faces are cut by recursive centroid-median splits along their longest
  // axis (xatlas parameterizes each chart serially, so a few huge charts would otherwise set the runtime). Each piece
  // carries compact vertices plus a map back to rows of cv.
  struct Piece { std::vector<float> P; std::vector<uint32_t> I; std::vector<int32_t> gmap; double area = 0; };
  std::vector<std::vector<Piece>> per_chart(nc);
  at::parallel_for(0, nc, 16, [&](int64_t b, int64_t e) {
    std::vector<int32_t> loc;
    for (int64_t c = b; c < e; c++) {
      const int64_t nf = fop[c + 1] - fop[c];
      std::vector<std::vector<int64_t>> leaves;
      std::vector<int64_t> all(nf); std::iota(all.begin(), all.end(), fop[c]);
      if (max_faces <= 0 || nf <= max_faces) leaves.push_back(std::move(all));
      else {
        std::vector<std::vector<int64_t>> stack{std::move(all)};
        auto cen = [&](int64_t f, int k) { return P[3 * Fi[3 * f] + k] + P[3 * Fi[3 * f + 1] + k] + P[3 * Fi[3 * f + 2] + k]; };
        while (!stack.empty()) {
          auto cur = std::move(stack.back()); stack.pop_back();
          if ((int64_t)cur.size() <= max_faces) { leaves.push_back(std::move(cur)); continue; }
          float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
          for (int64_t f : cur) for (int k = 0; k < 3; k++) { float x = cen(f, k); lo[k] = std::min(lo[k], x); hi[k] = std::max(hi[k], x); }
          int ax = 0; for (int k = 1; k < 3; k++) if (hi[k] - lo[k] > hi[ax] - lo[ax]) ax = k;
          auto mid = cur.begin() + cur.size() / 2;
          std::nth_element(cur.begin(), mid, cur.end(), [&](int64_t x, int64_t y) { float a = cen(x, ax), b = cen(y, ax); return a < b || (a == b && x < y); });
          stack.emplace_back(cur.begin(), mid); stack.emplace_back(mid, cur.end());
        }
      }
      loc.assign(vop[c + 1] - vop[c], -1);
      for (auto& lf : leaves) {
        Piece pc;
        for (int64_t f : lf) {
          for (int k = 0; k < 3; k++) {
            int32_t gv = Fi[3 * f + k]; int32_t& l = loc[gv - vop[c]];
            if (l < 0) { l = (int32_t)pc.gmap.size(); pc.gmap.push_back(gv); pc.P.insert(pc.P.end(), P + 3 * gv, P + 3 * gv + 3); }
            pc.I.push_back((uint32_t)l);
          }
          const float* p0 = P + 3 * Fi[3 * f]; const float* p1 = P + 3 * Fi[3 * f + 1]; const float* p2 = P + 3 * Fi[3 * f + 2];
          double ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2], vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
          double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
          pc.area += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        }
        for (int32_t gv : pc.gmap) loc[gv - vop[c]] = -1;
        per_chart[c].push_back(std::move(pc));
      }
    }
  });
  std::vector<Piece> pieces;
  for (auto& v : per_chart) for (auto& pc : v) pieces.push_back(std::move(pc));
  per_chart.clear();
  const int64_t np = (int64_t)pieces.size();

  // Shared texelsPerUnit: xatlas's packCharts estimate (resolution 1024 at 75% utilization) over the whole mesh.
  double total = 0; for (auto& pc : pieces) total += pc.area;
  const float tpu = (float)std::sqrt((1024.0 * 1024.0) / std::max(1.0, total / 0.75));

  // Deal pieces to groups round-robin in decreasing area order (balances both area and count).
  int G = (int)std::max<int64_t>(1, std::min<int64_t>(ngroups, np));
  std::vector<int64_t> order(np); std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return pieces[a].area > pieces[b].area; });
  std::vector<std::vector<int64_t>> groups(G);
  for (int64_t i = 0; i < np; i++) groups[i % G].push_back(order[i]);

  std::vector<GroupOut> out(G);
  t2r_xa_thread_override.store(2);  // xatlas needs >= 2 (its single worker uses thread index 1)
  {
    pybind11::gil_scoped_release nogil;
    std::atomic<int> next{0};
    int T = (int)std::max<int64_t>(1, std::min<int64_t>(nthreads > 0 ? nthreads : std::thread::hardware_concurrency(), G));
    std::vector<std::thread> pool;
    for (int t = 0; t < T; t++) pool.emplace_back([&] {
      for (int g; (g = next.fetch_add(1)) < G;) {
        auto& o = out[g];
        t2r_xatlas::Atlas* atlas = t2r_xatlas::Create();
        for (int64_t pi : groups[g]) {
          const Piece& pc = pieces[pi];
          t2r_xatlas::MeshDecl d;
          d.vertexCount = (uint32_t)pc.gmap.size();
          d.vertexPositionData = pc.P.data();
          d.vertexPositionStride = sizeof(float) * 3;
          d.indexCount = (uint32_t)pc.I.size();
          d.indexData = pc.I.data();
          d.indexFormat = t2r_xatlas::IndexFormat::UInt32;
          auto r = t2r_xatlas::AddMesh(atlas, d, (uint32_t)groups[g].size());
          if (r != t2r_xatlas::AddMeshError::Success) { o.err = t2r_xatlas::StringForEnum(r); break; }
        }
        if (o.err.empty()) {
          t2r_xatlas::AddMeshJoin(atlas);
          t2r_xatlas::ChartOptions co;  // CuMesh Atlas.compute_charts defaults == xatlas defaults
          t2r_xatlas::ComputeCharts(atlas, co);
          t2r_xatlas::PackOptions po;   // CuMesh Atlas.pack_charts defaults, with the shared texel scale
          po.texelsPerUnit = tpu;
          t2r_xatlas::PackCharts(atlas, po);
          o.w = (int)atlas->width; o.h = (int)atlas->height;
          for (uint32_t m = 0; m < atlas->meshCount; m++) {
            const auto& mesh = atlas->meshes[m];
            int32_t base = (int32_t)(o.vmap.size());
            const Piece& pc = pieces[groups[g][m]];
            for (uint32_t i = 0; i < mesh.vertexCount; i++) {
              const auto& v = mesh.vertexArray[i];
              o.vmap.push_back(pc.gmap[v.xref]);
              o.uv.push_back(v.uv[0]); o.uv.push_back(v.uv[1]);
            }
            for (uint32_t i = 0; i < mesh.indexCount; i++) o.faces.push_back(base + (int32_t)mesh.indexArray[i]);
          }
        }
        t2r_xatlas::Destroy(atlas);
      }
    });
    for (auto& th : pool) th.join();
  }
  t2r_xa_thread_override.store(0);
  for (auto& o : out) TORCH_CHECK(o.err.empty(), "xatlas AddMesh failed: ", o.err);

  // Shelf-pack the group atlases (tallest first), 1 texel apart. UVs are normalized by the final W x H, so texel density
  // scales with 1/sqrt(W*H): try a range of shelf widths and keep the smallest area.
  std::vector<int> gord(G); std::iota(gord.begin(), gord.end(), 0);
  std::stable_sort(gord.begin(), gord.end(), [&](int a, int b) { return out[a].h != out[b].h ? out[a].h > out[b].h : out[a].w > out[b].w; });
  double sumA = 0; int maxW = 0;
  for (auto& o : out) { sumA += (double)(o.w + 1) * (o.h + 1); maxW = std::max(maxW, o.w + 1); }
  std::vector<int> ox(G), oy(G), bx(G), by(G); int W = 0, H = 0; double bestA = 1e300;
  for (int k = 0; k <= 64; k++) {
    int rowW = std::max(maxW, (int)std::ceil(std::sqrt(sumA) * (0.7 + 0.02 * k)));
    int x = 0, y = 0, rowH = 0, w = 0;
    for (int g : gord) {
      if (x > 0 && x + out[g].w + 1 > rowW) { y += rowH; x = 0; rowH = 0; }
      bx[g] = x; by[g] = y; x += out[g].w + 1; rowH = std::max(rowH, out[g].h + 1); w = std::max(w, x);
    }
    int h = y + rowH;
    if ((double)w * h < bestA) { bestA = (double)w * h; W = w; H = h; ox = bx; oy = by; }
  }

  int64_t NV = 0, NF = 0; std::vector<int64_t> vbase(G), fbase(G);
  for (int g = 0; g < G; g++) { vbase[g] = NV; fbase[g] = NF; NV += (int64_t)out[g].vmap.size(); NF += (int64_t)out[g].faces.size() / 3; }
  auto vmap = torch::empty({NV}, torch::kInt32), faces = torch::empty({NF, 3}, torch::kInt32), uvs = torch::empty({NV, 2}, torch::kFloat32);
  int32_t* vm = vmap.data_ptr<int32_t>(); int32_t* fp = faces.data_ptr<int32_t>(); float* up = uvs.data_ptr<float>();
  at::parallel_for(0, G, 1, [&](int64_t b, int64_t e) {
    for (int64_t g = b; g < e; g++) {
      const auto& o = out[g];
      for (size_t i = 0; i < o.vmap.size(); i++) {
        vm[vbase[g] + i] = o.vmap[i];
        up[2 * (vbase[g] + i)] = (o.uv[2 * i] + ox[g]) / (float)W;
        up[2 * (vbase[g] + i) + 1] = (o.uv[2 * i + 1] + oy[g]) / (float)H;
      }
      for (size_t i = 0; i < o.faces.size(); i++) fp[3 * fbase[g] + i] = (int32_t)(o.faces[i] + vbase[g]);
    }
  });
  return {vmap, faces, uvs};
}
