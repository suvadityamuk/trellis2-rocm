# TRELLIS.2 on ROCm (t2r)

Image → textured GLB for [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2) (`1024_cascade`, `example.py` `to_glb`
settings) on AMD Instinct GPUs, modeled on [trellis2-cuda](https://github.com/suvadityamuk/trellis2-cuda). Two modes:

- **parity**: bitwise identical to the official PyTorch pipeline running on ROCm, from the input image to the final
  mesh and PBR voxel attributes, and through every deterministic `to_glb` stage.
- **fast**: HIP kernels and restructured CPU/GPU work, tolerance-bounded against the official pipeline's own spread.

Targets gfx942 (MI300X / MI300A / MI325X) and gfx950 (MI350X / MI355X). Measured on an MI350X (VF), torch 2.13 + ROCm 10.1.

## Results (MI350X, 5 example images, warm process, 1 warmup, seconds)

| Image | Official eager | Official torch.compile | t2r parity | t2r fast |
|---|---|---|---|---|
| 0a34 | 149.4 | 155.3 | 117.5 | **8.3** |
| 0e49 | 219.0 | 223.1 | 191.1 | **14.6** |
| 0f16 | 109.4 | 121.0 | 95.6 | **10.7** |
| 130c | 105.1 | 128.1 | 102.2 | **4.8** |
| T.png | 111.5 | 108.0 | 93.9 | **9.3** |
| **Sum** | 694.5 | 735.5 | 600.3 (1.16×) | **47.6 (14.6×)** |

Pipeline + `to_glb` + export, model loading excluded (`tools/bench_ref.py`, `tools/bench_t2r.py`; raw numbers in `bench/`).
By phase (sum over images): models 53.9 → 30.4 s, `to_glb` 608.6 → 14.0 s, export 32.0 → 3.2 s.

![per-image](docs/timing.png)
![total](docs/speedup.png)

## Outputs

![renders](docs/renders.png)

Turntable of the fast build for `T.png`: `docs/T_fast.gif`.

## Getting the official pipeline running on ROCm

The official code does not run on ROCm as shipped: nvdiffrast has no HIP path, CuMesh fails to build, and the sparse
DiTs only support flash-attn/xformers. Ported here (`patches/`, `tools/setup_rocm.sh`):

| Dependency | What it took |
|---|---|
| nvdiffrast v0.4.0 | The CUDA rasterizer is written for 32-lane warps (`blockDim = (32, warps)`). On wave64 CDNA one wavefront holds two logical warps, so every ballot/shuffle/lane-mask is reimplemented on the caller's 32-lane half (`cr_warp32.h`). All inline PTX replaced with exact equivalents (`set.ge.u32` is all-ones, `prmt` nibble bit 3 is sign replication, sub-word `vadd` merges into `c`, `__frcp_rz`, `fma.rm`). Validated against a brute-force rasterizer at 4096² and for determinism. |
| CuMesh | cub's tuple-decomposer radix sort (not in hipCUB) replaced by three stable passes, which give the identical permutation; reduce op; build flags. |
| flash-attention | ROCm CK backend (2.8.4) built for gfx950. |
| FlexGEMM, o-voxel | Build as-is with `GPU_ARCHS`. |

## Parity

Method as in trellis2-cuda: `tools/ref_dump.py` records every stage of the official pipeline, `tools/parity.py` feeds
each t2r stage the official stage inputs and compares bytes.

| Stage | 5 images |
|---|---|
| DINOv3 cond (512, 1024), SS sampling + decode, shape 512 → upsample → 1024, texture, decoders, `fill_holes` | **55/55 bitwise** (`final.v/f/attrs/coords` included) |
| remesh, UV unwrap, texture bake, the four TELEA inpaints | **bitwise** |
| simplify | nondeterministic in the official code (two official runs on identical input differ) |

What parity needed:
- **Same library calls, same layouts.** hipBLASLt through `F.linear`, CK flash-attention with the official
  (packed/varlen) call forms, MIOpen conv3d, FlexGEMM Triton sparse conv with the shared autotune cache, and the same
  ATen elementwise/normalization ops in the same order and dtypes (RoPE complex multiplies, `F.normalize` RMSNorm,
  bf16 rounding points, CPU `torch.randn`).
- **A bitwise HIP replacement for FlexGEMM's trilinear weighted sum.** Each output is a sequential fp32 sum over the 8
  neighbors (unfused multiply-add), independent of the Triton tile, so it reproduces the kernel without its per-image
  autotuning (~2 s/image).
- **A fixed reference.** On gfx950, PyTorch's fp32 memory-efficient SDPA (AOTriton) intermittently returns *wrong*
  results above ~2k tokens (errors up to 1.5e-2 on outputs of magnitude 0.02; correct runs are at 5e-7). DINOv3 at
  1024² has 4101 tokens, so the official 1024 conditioning changes every run (13 distinct outputs in 20 runs). The parity
  reference pins DINOv3 attention to PyTorch's MATH backend (`tools/refenv.py`); with that, two official runs are
  82/82 bitwise. Speed baselines use the official code as shipped.

Parity mode is only modestly faster: it has to keep the official xatlas packing (70–137 s per image). Its savings come
from the FlexGEMM autotuning, parallel inpainting, and `t2r_cumesh`, a build of CuMesh whose allocations go through the
PyTorch caching allocator and whose buffer fills run on the device. Upstream allocates per iteration (`hipMalloc` is slow)
and builds host vectors for `fill()` (~300 MB pageable copies). Neither change alters any arithmetic.

## Fast build

| Change | Effect |
|---|---|
| Grouped parallel xatlas: charts > 1024 faces split, 64 area-balanced group atlases packed in native threads with a shared texel scale, then shelf-tiled | UV unwrap 70–137 s → 1–3 s; texel density −3% vs official |
| GPU LBVH (Morton + hipCUB radix sort + Karras tree) instead of cuBVH's CPU build | BVH 7 s → 0.12 s |
| GPU wavefront TELEA + uint8 conversion + glTF texture packing | 4 inpaints ~20 s → 45 ms |
| `t2r_cumesh` (caching allocator, device fills) | simplify 5.2 → 1.0 s, chart clustering 2.9 → 1.2 s (both also used in parity) |
| Fused DiT kernels (LN+modulate, QK-RMSNorm+RoPE+pack, gated residual + LN), GELU fused in the GEMM epilogue, cross-attention K/V cached per run, CFG cond/uncond batched | DiT step 277 → 177 ms at 13k tokens |
| PyTorch SDPA (AOTriton bf16 flash) for DiT attention | ~20% faster than CK at these shapes, deterministic, as accurate |
| Fused sparse-voxel trilinear sampling; direct GLB writer, WebP method 0, textures encoded in parallel | export 4.9–8.3 s → 0.6–0.7 s |

**Tolerance** (`tools/tolerance.py`: symmetric Chamfer / bbox diagonal, base-color L1 at matched surface samples, vs official eager):

| Image | official compile | official (pinned attention) | t2r parity | t2r fast |
|---|---|---|---|---|
| 0a34 | 2.20e-3 / 22.2 | 2.21e-3 / 21.7 | 2.21e-3 / 21.7 | 2.14e-3 / 17.0 |
| 0e49 | 3.83e-3 / 22.8 | 3.68e-3 / 22.6 | 3.68e-3 / 22.6 | 3.54e-3 / 21.4 |
| 0f16 | 2.74e-3 / 21.4 | 2.75e-3 / 26.3 | 2.75e-3 / 26.3 | 2.66e-3 / 28.7 |
| 130c | 4.32e-3 / 18.9 | 4.34e-3 / 19.3 | 4.34e-3 / 19.3 | 4.30e-3 / 23.7 |
| T | 2.53e-3 / 25.4 | 2.50e-3 / 28.9 | 2.50e-3 / 28.9 | 2.70e-3 / 26.2 |

The fast build falls inside the band of the official configurations against each other. Teacher-forced, a fast DiT
call differs from the official by 0.3–0.6% relative L2.

Fast-build trade-offs: ~1% more UV seams from chart splitting; TELEA fill order differs in texels outside the charts;
the (unused, `alphaMode: OPAQUE`) base-color alpha channel is WebP-encoded at `alpha_quality=0`.

What didn't help: aiter's FMHA (same ~920 TF/s as AOTriton, bitwise-identical output); HIP graphs (DiT forwards are
already 90–95% GPU-busy). Not attempted: FP8 GEMMs.

## Layout

- `csrc/`: HIP kernels (`dit_fast.hip`, `lbvh.hip`, `telea.hip`, `grid_sample.hip`), the grouped xatlas packer
  (`uv_atlas.cpp`, vendored xatlas), bindings. `setup.py` builds `t2r._C` for `T2R_ARCHS` (default `gfx942;gfx950`).
- `t2r/`: `dit.py` (four flow DiTs, parity + fast paths), `sampler.py`, `pipeline.py`, `glb.py` (`to_glb` stages, GLB writer).
- `tools/`: setup, reference dumps, parity checks, benchmarks, tolerance, renders, plots. `tests/`: kernel tests.
- `patches/`: the CuMesh and nvdiffrast ROCm ports.

The sparse-structure and SLat VAE decoders (~3% of the warm runtime) still use the official modules.

## Reproduce

```bash
tools/setup_rocm.sh                                            # deps, ROCm ports, t2r, weights (HF_TOKEN in .env)
python tools/ref_dump.py --out out/ref                         # official stage dumps
PYTHONPATH=. python tools/parity.py pipe remesh simplify unwrap bake textures
python tools/bench_ref.py --mode eager --out bench/eager.json
PYTHONPATH=. python tools/bench_t2r.py --mode fast --stages --out bench/fast.json
python tools/tolerance.py --out bench/tolerance.json eager=out/eager fast=out/fast
```

Only gfx950 was available for testing; the gfx942 code paths are built but untested.
