"""TRELLIS.2 flow DiTs (SparseStructureFlowModel when dense, SLatFlowModel otherwise), without the trellis2 package.

Parity path: the exact sequence of library ops the official modules issue (hipBLASLt GEMMs through F.linear, CK
flash-attention through flash_attn, ATen elementwise/norm kernels), so results are bitwise identical.
Fast path (t2r.fast): fused HIP kernels for LayerNorm+modulate, QK-RMSNorm+RoPE+pack and gated residuals, GELU fused
into the MLP GEMM, a K/V cache for the (image-constant) cross-attention context, and CFG cond/uncond batched into one
varlen call.
"""
import math
import torch
import torch.nn.functional as F
import flash_attn
from safetensors.torch import load_file
from torch.nn.attention import sdpa_kernel, SDPBackend

from . import fast as FK

C, H, D, NB, FF = 1536, 12, 128, 30, 8192
RMS_SCALE = D ** 0.5


def timestep_freqs():
    # TimestepEmbedder.timestep_embedding, computed on the CPU exactly as the official code does
    import numpy as np
    half = 128
    return torch.exp(-np.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half)


def rope_freqs():
    f = torch.arange(21, dtype=torch.float32) / 21
    return 1.0 / (10000.0 ** f)


def rope_phases_from_coords(coords3, device):
    """SparseRotaryPositionEmbedder._get_phases (+ padding) for int coords [N, 3] -> complex64 [N, 64] on device."""
    fr = rope_freqs().to(device)
    ph = torch.outer(coords3.reshape(-1), fr)
    ph = torch.polar(torch.ones_like(ph), ph).reshape(coords3.shape[0], -1)
    pad = torch.polar(torch.ones(ph.shape[0], 1, device=device), torch.zeros(ph.shape[0], 1, device=device))
    return torch.cat([ph, pad], dim=-1)


def dense_rope_phases(res=16):
    """RotaryPositionEmbedder buffer of SparseStructureFlowModel (built on the CPU at module init)."""
    coords = torch.stack(torch.meshgrid(*[torch.arange(res)] * 3, indexing='ij'), dim=-1).reshape(-1, 3)
    fr = rope_freqs()
    ph = torch.outer(coords.reshape(-1), fr)
    ph = torch.polar(torch.ones_like(ph), ph).reshape(coords.shape[0], -1)
    pad = torch.polar(torch.ones(ph.shape[0], 1), torch.zeros(ph.shape[0], 1))
    return torch.cat([ph, pad], dim=-1)


def _attn(q, k, v):
    """[B, L, H, D] views -> [B*L, C]. Fast build: PyTorch SDPA with the AOTriton flash backend (bf16), ~20% faster
    than the CK flash-attention kernels at these sizes on gfx950, deterministic, and at least as accurate."""
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        o = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
    return o.permute(0, 2, 1, 3).reshape(-1, C)


class Block:
    __slots__ = ("mod", "qkvW", "qkvb", "qg", "kg", "o1W", "o1b", "n2w", "n2b", "qW", "qb", "kvW", "kvb", "cqg", "ckg",
                 "o2W", "o2b", "f1W", "f1b", "f2W", "f2b", "qgs", "kgs", "cqgs", "ckgs")


class FlowDiT:
    def __init__(self, path, dense, device="cuda"):
        W = load_file(path, device=str(device))
        f32 = lambda k: W[k].float()
        bf = lambda k: W[k].to(torch.bfloat16)
        self.dense = dense
        self.t0W, self.t0b, self.t2W, self.t2b = f32("t_embedder.mlp.0.weight"), f32("t_embedder.mlp.0.bias"), f32("t_embedder.mlp.2.weight"), f32("t_embedder.mlp.2.bias")
        self.aW, self.ab = f32("adaLN_modulation.1.weight"), f32("adaLN_modulation.1.bias")
        self.inW, self.inb = f32("input_layer.weight"), f32("input_layer.bias")
        self.outW, self.outb = f32("out_layer.weight"), f32("out_layer.bias")
        self.cin, self.cout = self.inW.shape[1], self.outW.shape[0]
        self.blocks = []
        for i in range(NB):
            p = f"blocks.{i}."
            b = Block()
            b.mod = f32(p + "modulation")
            b.qkvW, b.qkvb = bf(p + "self_attn.to_qkv.weight"), bf(p + "self_attn.to_qkv.bias")
            b.qg, b.kg = f32(p + "self_attn.q_rms_norm.gamma"), f32(p + "self_attn.k_rms_norm.gamma")
            b.o1W, b.o1b = bf(p + "self_attn.to_out.weight"), bf(p + "self_attn.to_out.bias")
            b.n2w, b.n2b = f32(p + "norm2.weight"), f32(p + "norm2.bias")
            b.qW, b.qb = bf(p + "cross_attn.to_q.weight"), bf(p + "cross_attn.to_q.bias")
            b.kvW, b.kvb = bf(p + "cross_attn.to_kv.weight"), bf(p + "cross_attn.to_kv.bias")
            b.cqg, b.ckg = f32(p + "cross_attn.q_rms_norm.gamma"), f32(p + "cross_attn.k_rms_norm.gamma")
            b.o2W, b.o2b = bf(p + "cross_attn.to_out.weight"), bf(p + "cross_attn.to_out.bias")
            b.f1W, b.f1b = bf(p + "mlp.mlp.0.weight"), bf(p + "mlp.mlp.0.bias")
            b.f2W, b.f2b = bf(p + "mlp.mlp.2.weight"), bf(p + "mlp.mlp.2.bias")
            # fast path: gamma * sqrt(D) folded, as the official computes (normalize(x) * gamma) * scale in two steps
            b.qgs = b.kgs = b.cqgs = b.ckgs = None
            self.blocks.append(b)
        del W
        self.tfreq = timestep_freqs().to(device)
        self.phases = dense_rope_phases().to(device) if dense else None
        self.kv_cache = {}

    # ---------------------------------------------------------------- shared pieces
    def t_embed(self, t):
        args = t[:, None].float() * self.tfreq[None]
        e = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        e = F.linear(e, self.t0W, self.t0b)
        e = F.linear(F.silu(e), self.t2W, self.t2b)
        return F.linear(F.silu(e), self.aW, self.ab)  # share_mod adaLN_modulation

    def set_coords(self, coords4):
        self.phases = rope_phases_from_coords(coords4[:, 1:], coords4.device)
        self.N = coords4.shape[0]
        self.cu = torch.tensor([0, self.N], dtype=torch.int32, device=coords4.device)

    # ---------------------------------------------------------------- parity (official op sequence)
    @staticmethod
    def _rms(x, gamma):  # MultiHeadRMSNorm / SparseMultiHeadRMSNorm
        return (F.normalize(x.float(), dim=-1) * gamma * RMS_SCALE).to(x.dtype)

    @staticmethod
    def _rope(x, phases):  # apply_rotary_embedding (phases [N, 64] or [1, N, 64] broadcast over heads)
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * phases.unsqueeze(-2)).reshape(*x.shape).to(x.dtype)

    def _block_parity(self, b, x, t_emb, cond_bf):
        dense = self.dense
        sm, cm, gm, sf, cf, gf = (b.mod + t_emb).type(t_emb.dtype).chunk(6, dim=1)
        if dense:  # dense blocks broadcast [1, C] -> [1, 1, C] via unsqueeze(1); same values
            sm, cm, gm, sf, cf, gf = [m.unsqueeze(1) for m in (sm, cm, gm, sf, cf, gf)]
        # self-attention
        h = F.layer_norm(x.float(), (C,), None, None, 1e-6).type(x.dtype)
        h = h * (1 + cm) + sm
        qkv = F.linear(h, b.qkvW, b.qkvb)
        if dense:
            qkv = qkv.reshape(1, 4096, 3, H, D)
            q, k, v = qkv.unbind(dim=2)
            q, k = self._rms(q, b.qg), self._rms(k, b.kg)
            q, k = self._rope(q, self.phases), self._rope(k, self.phases)
            o = flash_attn.flash_attn_func(q, k, v).reshape(1, 4096, C)
        else:
            qkv = qkv.unsqueeze(0).reshape(1, self.N, 3, H, D).squeeze(0)
            q, k, v = qkv.unbind(dim=-3)
            q, k = self._rms(q, b.qg), self._rms(k, b.kg)
            q, k = self._rope(q, self.phases), self._rope(k, self.phases)
            qkv = torch.stack([q, k, v], dim=1)
            o = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, self.cu, self.N).reshape(self.N, C)
        h = F.linear(o, b.o1W, b.o1b)
        x = x + h * gm
        # cross-attention
        h = F.layer_norm(x.float(), (C,), b.n2w, b.n2b, 1e-6).type(x.dtype)
        q = F.linear(h, b.qW, b.qb)
        kv = F.linear(cond_bf, b.kvW, b.kvb)  # [1, L, 2C]
        L = kv.shape[1]
        kv = kv.reshape(1, L, 2, H, D)
        k, v = kv.unbind(dim=-3)
        k = self._rms(k, b.ckg)
        if dense:
            q = self._rms(q.reshape(1, 4096, H, D), b.cqg)
            o = flash_attn.flash_attn_func(q, k, v).reshape(1, 4096, C)
        else:
            q = self._rms(q.reshape(self.N, H, D), b.cqg)
            cuk = torch.tensor([0, L], dtype=torch.int32, device=x.device)
            o = flash_attn.flash_attn_varlen_func(q, k.reshape(L, H, D), v.reshape(L, H, D), self.cu, cuk, self.N, L).reshape(self.N, C)
        x = x + F.linear(o, b.o2W, b.o2b)
        # MLP
        h = F.layer_norm(x.float(), (C,), None, None, 1e-6).type(x.dtype)
        h = h * (1 + cf) + sf
        h = F.linear(F.gelu(F.linear(h, b.f1W, b.f1b), approximate="tanh"), b.f2W, b.f2b)
        return x + h * gf

    def forward_parity(self, x, t, cond, concat=None):
        """x: dense [1, cin, 16, 16, 16] / sparse feats [N, cin] fp32; t: [1] fp32 (already *1000); cond [1, L, 1024]."""
        if self.dense:
            h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        else:
            h = torch.cat([x, concat], dim=-1) if concat is not None else x
        h = F.linear(h, self.inW, self.inb)
        t_emb = self.t_embed(t).type(torch.bfloat16)
        h = h.type(torch.bfloat16)
        cond_bf = cond.type(torch.bfloat16)
        for b in self.blocks:
            h = self._block_parity(b, h, t_emb, cond_bf)
        h = F.layer_norm(h.type(torch.float32), (C,))
        h = F.linear(h, self.outW, self.outb)
        if self.dense:
            h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], 16, 16, 16).contiguous()
        return h

    # ---------------------------------------------------------------- fast path
    def _cross_kv(self, conds):
        """Per-block normalized K and V for the cross-attention context(s). Cached by tensor address within one
        pipeline run (Pipeline.run clears the cache, since the allocator reuses addresses across runs)."""
        key = tuple((c.data_ptr(), c.shape[1], c._version) for c in conds)
        if key not in self.kv_cache:
            self.kv_cache.clear()
            cb = torch.cat([c.reshape(-1, c.shape[-1]) for c in conds], 0).type(torch.bfloat16)
            out = []
            for b in self.blocks:
                kv = F.linear(cb, b.kvW, b.kvb)  # [sum L, 2C]
                k, v = FK.kv_split_rms(kv, b.ckg, RMS_SCALE)
                out.append((k, v))
            lens = [c.shape[1] for c in conds]
            cuk = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=cb.device)
            self.kv_cache[key] = (out, cuk, max(lens))
        return self.kv_cache[key]

    def forward_fast(self, x, ts, conds, concat=None):
        """Batched over CFG: x [N, cin] (sparse feats or dense tokens), ts list of python floats (one per batch item,
        already *1000), conds list of [1, L, 1024]. Returns [B*N, cout] token-major."""
        B = len(conds)
        N = x.shape[0]
        h = torch.cat([x, concat], dim=-1) if concat is not None else x
        h = F.linear(h, self.inW, self.inb).to(torch.bfloat16)  # identical rows for every batch item
        t = torch.tensor(ts, dtype=torch.float32, device=x.device)
        t_emb = self.t_embed(t).type(torch.bfloat16)  # [B, 6C]
        h = h.repeat(B, 1) if B > 1 else h
        kvs, cuk, maxk = self._cross_kv(conds)
        cuq = torch.arange(0, B + 1, dtype=torch.int32, device=x.device) * N
        phases = self.phases if self.phases.dim() == 2 else self.phases.reshape(-1, 64)
        for b, (ck, cv) in zip(self.blocks, kvs):
            mods = FK.block_mods(b.mod, t_emb)  # [B, 6, C] bf16
            hn = FK.ln_mod(h, mods, 0, N, None, None)
            qkv = F.linear(hn, b.qkvW, b.qkvb)
            qkvp = FK.qk_rms_rope_pack(qkv, b.qg, b.kg, RMS_SCALE, phases, N).view(B, N, 3, H, D)
            o = _attn(qkvp[:, :, 0], qkvp[:, :, 1], qkvp[:, :, 2])
            hn = FK.gate_res_ln(h, F.linear(o, b.o1W, b.o1b), mods, 2, N, b.n2w, b.n2b)  # x += o*gate; LN2 (affine)
            q = FK.rms_heads(F.linear(hn, b.qW, b.qb), b.cqg, RMS_SCALE).view(B, N, H, D)
            o = _attn(q, ck.view(B, -1, H, D), cv.view(B, -1, H, D))
            hn = FK.res_ln_mod(h, F.linear(o, b.o2W, b.o2b), mods, N)  # x += o; LN3 + modulate(mlp)
            g = torch._addmm_activation(b.f1b, hn, b.f1W.t(), use_gelu=True)
            FK.gate_res(h, F.linear(g, b.f2W, b.f2b), mods, 5, N)
        h = F.layer_norm(h.type(torch.float32), (C,))
        return F.linear(h, self.outW, self.outb)
