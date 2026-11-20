#!/usr/bin/env bash
# Official TRELLIS.2 on ROCm (reference + baselines) and the t2r extension. Tested: MI350X (gfx950), torch 2.13+rocm10.1,
# Python 3.14, ROCm SDK from pip. GPU_ARCHS / T2R_ARCHS default to MI300 (gfx942) + MI350 (gfx950).
set -euo pipefail
ARCHS=${ARCHS:-"gfx942;gfx950"}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-$HOME/src}; mkdir -p $SRC/ext
export GPU_ARCHS=$ARCHS PYTORCH_ROCM_ARCH=$ARCHS MAX_JOBS=${MAX_JOBS:-64}
pip install -q imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh "transformers==4.56.2" pandas \
  zstandard kornia timm scipy rtree pybind11 psutil safetensors "huggingface_hub<1.0" hf_transfer matplotlib
pip install -q git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# official repo (pinned as in trellis2-cuda)
if [ ! -d $SRC/TRELLIS.2/.git ]; then
  mkdir -p $SRC/TRELLIS.2 && cd $SRC/TRELLIS.2 && git init -q && git remote add origin https://github.com/microsoft/TRELLIS.2.git
  git fetch -q --depth 1 origin 75fbf0183001ed9876c8dbb35de6b68552ee08bd && git checkout -q FETCH_HEAD
  git submodule update --init --depth 1 o-voxel/third_party/eigen
fi
cd $SRC/ext
# flash-attention (ROCm, CK backend) - the attention backend the official sparse DiTs require
[ -d flash-attention ] || git clone -q --recursive https://github.com/ROCm/flash-attention.git
(cd flash-attention && pip install --no-build-isolation -v . > ../fa_build.log 2>&1)
# FlexGEMM (already HIP-aware)
[ -d FlexGEMM ] || (git clone -q https://github.com/JeffreyXiang/FlexGEMM.git && cd FlexGEMM && git checkout -q 6dd94a859c26ee8246888502eada3dd8ad85532e)
(cd FlexGEMM && pip install --no-build-isolation --no-deps .)
# CuMesh + t2r ROCm patch (hipCUB-compatible face sort, reduce op, build flags)
[ -d CuMesh ] || (git clone -q https://github.com/JeffreyXiang/CuMesh.git && cd CuMesh && git checkout -q 12289e1062f0603f2f0d0771b02e1395d247f26f && git submodule update --init --recursive -q && git apply $ROOT/patches/cumesh-rocm.patch)
(cd CuMesh && pip install --no-build-isolation --no-deps .)
# t2r_cumesh: the same CuMesh with allocations through the PyTorch caching allocator and device-side fills, installed
# next to the upstream build (the t2r inference path uses it; results are bitwise identical)
[ -d CuMesh-t2r ] || (git clone -q https://github.com/JeffreyXiang/CuMesh.git CuMesh-t2r && cd CuMesh-t2r && git checkout -q 12289e1062f0603f2f0d0771b02e1395d247f26f && git submodule update --init --recursive -q && git apply $ROOT/patches/cumesh-t2r-full.patch && cd third_party/cubvh && git apply $ROOT/patches/cumesh-t2r-cubvh-submodule.patch)
(cd CuMesh-t2r && pip install --no-build-isolation --no-deps .)
# nvdiffrast v0.4.0 + t2r ROCm patch (wave64-safe 32-lane logical warps, exact PTX helper replacements)
[ -d nvdiffrast ] || (git clone -q https://github.com/NVlabs/nvdiffrast.git && cd nvdiffrast && git checkout -q 253ac4fcea7de5f396371124af597e6cc957bfae && git apply $ROOT/patches/nvdiffrast-v0.4.0-rocm.patch)
(cd nvdiffrast && pip install --no-build-isolation --no-deps .)
(cd $SRC/TRELLIS.2/o-voxel && pip install --no-build-isolation --no-deps .)
# t2r kernels
(cd $ROOT && T2R_ARCHS=$ARCHS python3 setup.py build_ext --inplace)
# weights (TRELLIS.2-4B local; DINOv3 is gated: put HF_TOKEN in $ROOT/.env)
[ -f $ROOT/.env ] && set -a && . $ROOT/.env && set +a
hf download microsoft/TRELLIS.2-4B --local-dir $HOME/ckpts/TRELLIS.2-4B
python3 -c "import torch, flash_attn, flex_gemm, cumesh, t2r_cumesh, o_voxel, nvdiffrast.torch, t2r._C; print('env ok', torch.__version__, flash_attn.__version__)"
