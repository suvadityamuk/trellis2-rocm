"""Build one Hub variant of trellis2-rocm against the PyTorch + ROCm of the current environment, without Nix.

kernel-builder (`nix run .#build-and-copy`) is the reproducible route and covers the torch x ROCm matrix it supports.
This script is for combinations outside that matrix (for example torch 2.13 + ROCm 10.1): it writes
build/torch<XY>-rocm<XY>-x86_64-linux/ in the same layout, including metadata.json with file digests (but no
provenance record or symbols.json, and linked against the host's glibc rather than manylinux_2_28).

Usage: python build_variant.py [--archs gfx942 gfx950] [--out build]
Needs hipcc (ROCm SDK) on PATH and the torch headers of the active environment.
"""
import argparse, base64, hashlib, json, os, platform, re, shutil, subprocess, sys, sysconfig, tomllib
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils import cpp_extension

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--archs", nargs="+", default=None, help="GPU architectures (default: rocm-archs from build.toml)")
ap.add_argument("--out", default=os.path.join(ROOT, "build"))
ap.add_argument("--rocm", default=None, help="ROCm version X.Y (default: from the torch version string)")
ap.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
ap.add_argument("--hipcc", default=shutil.which("hipcc") or "hipcc")
a = ap.parse_args()

cfg = tomllib.load(open(os.path.join(ROOT, "build.toml"), "rb"))
name = cfg["general"]["name"]
pyname = name.replace("-", "_")
kernel = cfg["kernel"]["mesh"]
archs = sorted(a.archs or kernel["rocm-archs"])
assert torch.version.hip, "this script needs a ROCm build of PyTorch"

tv = torch.__version__.split("+")[0].split(".")
m = re.search(r"rocm(\d+)\.(\d+)", torch.__version__)  # torch.version.hip is the HIP runtime version, not ROCm's
assert m or a.rocm, f"cannot read the ROCm version from torch {torch.__version__}; pass --rocm X.Y"
rv = (a.rocm.split(".") if a.rocm else m.groups())
variant = f"torch{tv[0]}{tv[1]}-rocm{rv[0]}{rv[1]}-{platform.machine()}-linux"

# Unique op namespace per build: a hash of the sources, like kernel-builder's git-revision suffix.
srcs = sorted(cfg["torch"]["src"] + kernel["src"] + [f"torch-ext/{pyname}/__init__.py", "build.toml"])
h = hashlib.sha1()
for s in srcs:
    h.update(s.encode()); h.update(open(os.path.join(ROOT, s), "rb").read())
h.update(variant.encode()); h.update(";".join(archs).encode())
ops_name = f"_{pyname}_rocm_{h.hexdigest()[:7]}"

out = os.path.join(a.out, variant)
obj = os.path.join(a.out, ".obj", variant)
shutil.rmtree(out, ignore_errors=True); shutil.rmtree(obj, ignore_errors=True)
os.makedirs(out); os.makedirs(obj)

inc = [f"-I{p}" for p in cpp_extension.include_paths(device_type="cuda")]
inc += [f"-I{sysconfig.get_paths()['include']}", f"-I{ROOT}/torch-ext"] + [f"-I{ROOT}/{d}" for d in kernel.get("include", [])]
# registration.h is the one kernel-builder ships (a stable-ABI module init)
REG = '''#pragma once
#include <Python.h>
#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)
#define REGISTER_EXTENSION(NAME)                                               \\
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \\
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \\
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \\
    return PyModule_Create(&module);                                           \\
  }
'''
open(os.path.join(obj, "registration.h"), "w").write(REG)
common = ["-O3", "-std=c++17", "-fPIC", "-fvisibility=hidden", "-DROCM_KERNEL", "-DUSE_ROCM=1", "-DPy_LIMITED_API=0x03090000",
          f"-DTORCH_EXTENSION_NAME={ops_name}", f"-I{obj}"] + inc


def compile_one(src):
    o = os.path.join(obj, src.replace("/", "_") + ".o")
    if src.endswith(".hip"):
        cmd = [a.hipcc, "-x", "hip", "-fno-gpu-rdc"] + [f"--offload-arch={x}" for x in archs]
    else:
        cmd = [a.cxx]
    subprocess.run(cmd + common + ["-c", os.path.join(ROOT, src), "-o", o], check=True)
    return o


units = [s for s in cfg["torch"]["src"] + kernel["src"] if s.endswith((".cpp", ".hip"))]
with ThreadPoolExecutor(len(units)) as ex:
    objs = list(ex.map(compile_one, units))
so = f"{ops_name}.abi3.so"
libdirs = cpp_extension.library_paths(device_type="cuda")
subprocess.run([a.cxx, "-shared", "-o", os.path.join(out, so)] + objs + [f"-L{p}" for p in libdirs]
               + ["-lc10", "-ltorch", "-ltorch_cpu", "-lc10_hip", "-ltorch_hip", "-lamdhip64", "-lpthread", "-Wl,--as-needed"], check=True)

shutil.copy(os.path.join(ROOT, "torch-ext", pyname, "__init__.py"), os.path.join(out, "__init__.py"))
open(os.path.join(out, "_ops.py"), "w").write(f'''import torch
from . import {ops_name}
ops = torch.ops.{ops_name}

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"{ops_name}::{{op_name}}"
''')
files = {}
for d, _, fs in os.walk(out):
    for f in fs:
        p = os.path.join(d, f)
        files[os.path.relpath(p, out)] = base64.b64encode(hashlib.sha256(open(p, "rb").read()).digest()).decode()
meta = {"name": name, "id": ops_name, "version": cfg["general"]["version"], "build-layout": "files", "kernels-minver": "0.14.0",
        "license": cfg["general"]["license"], "source": cfg["general"]["source"], "python-depends": [], "kernel-depends": [], "backend": {"type": "rocm", "archs": archs},
        "digest": {"algorithm": "sha256", "files": dict(sorted(files.items()))}}
json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)
open(os.path.join(out, "metadata.json"), "a").write("\n")
print(f"built {out}\n  id {ops_name}, archs {archs}, torch {torch.__version__}")
