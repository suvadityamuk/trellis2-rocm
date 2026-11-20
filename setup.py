# Builds t2r._C, the HIP kernels of the TRELLIS.2 ROCm port.
#   T2R_ARCHS="gfx942;gfx950" pip install --no-build-isolation -e .
# gfx942 = MI300X/MI300A/MI325X, gfx950 = MI350X/MI355X. gfx90a (MI200) also builds; kernels are wave64-generic.
import glob, os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

archs = os.environ.get("T2R_ARCHS", "gfx942;gfx950").split(";")
hip_flags = ["-O3", "-std=c++17", "-ffp-contract=off", "-fno-gpu-rdc", "-munsafe-fp-atomics"] + [f"--offload-arch={a}" for a in archs]
srcs = sorted(glob.glob("csrc/*.hip")) + sorted(glob.glob("csrc/*.cpp")) + ["csrc/third_party/xatlas/xatlas.cpp"]

setup(
    name="t2r",
    version="0.1.0",
    packages=["t2r"],
    ext_modules=[CUDAExtension("t2r._C", srcs, include_dirs=[os.path.abspath("csrc")],
                               extra_compile_args={"cxx": ["-O3", "-std=c++17", "-fopenmp"], "nvcc": hip_flags},
                               extra_link_args=["-fopenmp"])],
    cmdclass={"build_ext": BuildExtension},
)
