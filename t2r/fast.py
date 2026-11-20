"""Thin Python wrappers over the fused HIP kernels of the fast build (t2r._C)."""
import torch
from . import _C


def block_mods(mod_param, t_emb):
    """(modulation + t_emb).bf16 -> [B, 6C] (the official adds in fp32 then casts)."""
    return (mod_param + t_emb).to(torch.bfloat16).contiguous()


def ln_mod(x, mods, shift_j, N, w=None, b=None):
    return _C.ln_mod(x, mods, shift_j, N, w, b)


def gate_res_ln(x, a, mods, gate_j, N, w, b):
    return _C.gate_res_ln(x, a.contiguous(), mods, gate_j, N, w, b)


def res_ln_mod(x, a, mods, N):
    return _C.res_ln_mod(x, a.contiguous(), mods, N)


def gate_res(x, a, mods, gate_j, N):
    _C.gate_res(x, a.contiguous(), mods, gate_j, N)


def qk_rms_rope_pack(qkv, qg, kg, scale, phases, N):
    return _C.qk_rms_rope_pack(qkv.contiguous(), qg, kg, scale, phases, N)


def rms_heads(x, g, scale):
    return _C.rms_heads(x.contiguous(), g, scale)


def kv_split_rms(kv, g, scale):
    return _C.kv_split_rms(kv.contiguous(), g, scale)
