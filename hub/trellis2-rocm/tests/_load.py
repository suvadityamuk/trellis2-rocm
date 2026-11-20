"""Loads the kernel under test: the local build next to this directory by default, or the published one when
TRELLIS2_ROCM_FROM_HUB=1 (optionally TRELLIS2_ROCM_REPO / TRELLIS2_ROCM_VERSION)."""
import os
from pathlib import Path

import kernels

if os.environ.get("TRELLIS2_ROCM_FROM_HUB") == "1":
    k = kernels.get_kernel(os.environ.get("TRELLIS2_ROCM_REPO", "suvadityamuk/trellis2-rocm"),
                           version=int(os.environ.get("TRELLIS2_ROCM_VERSION", "1")), trust_remote_code=True)
else:
    k = kernels.get_local_kernel(Path(__file__).resolve().parent.parent, trust_remote_code=True)
