"""FlowEulerGuidanceIntervalSampler (TRELLIS.2) for t2r models.

Parity: the official op sequence (ClassifierFreeGuidanceSamplerMixin / GuidanceIntervalSamplerMixin / FlowEulerSampler):
each tensor op on the same dtype and in the same order, CFG cond/uncond as two separate model calls. Sparse latents use
VarLenTensor.std (biased, sqrt(E[x^2] - E[x]^2)), dense ones torch.std (unbiased).
Fast: cond and uncond are batched into one model call and the guidance arithmetic runs in fp32 on whole tensors.
"""
import numpy as np
import torch


def t_schedule(steps, rescale_t):
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_seq = t_seq.tolist()
    return [(t_seq[i], t_seq[i + 1]) for i in range(steps)]


def _std(x, dense):
    if dense:
        return x.std(dim=list(range(1, x.ndim)), keepdim=True)
    mean = torch.segment_reduce(x.mean(dim=(1,), keepdim=True), reduce="mean", lengths=torch.tensor([x.shape[0]], device=x.device))
    mean2 = torch.segment_reduce((x ** 2).mean(dim=(1,), keepdim=True), reduce="mean", lengths=torch.tensor([x.shape[0]], device=x.device))
    return (mean2 - mean ** 2).sqrt()


def _bcast(s, x, dense):
    # SparseTensor elementwise ops broadcast [B, 1] to the feats via the batch map; values are identical to a plain
    # broadcast, so a broadcast multiply gives the same bits.
    return s if dense else s.expand(x.shape[0], x.shape[1])


def sample_parity(model_call, noise, cond, neg_cond, steps, rescale_t, guidance_strength, guidance_rescale,
                  guidance_interval, sigma_min=1e-5, dense=False):
    """model_call(x, t_tensor, cond) -> v. noise: dense [1, C, 16, 16, 16] or sparse feats [N, C] (fp32)."""
    x_t = noise
    for t, t_prev in t_schedule(steps, rescale_t):
        tt = torch.tensor([1000 * t] * 1, device=x_t.device, dtype=torch.float32)
        gs = guidance_strength if guidance_interval[0] <= t <= guidance_interval[1] else 1
        if gs == 1:
            pred = model_call(x_t, tt, cond)
        elif gs == 0:
            pred = model_call(x_t, tt, neg_cond)
        else:
            pred_pos = model_call(x_t, tt, cond)
            pred_neg = model_call(x_t, tt, neg_cond)
            pred = gs * pred_pos + (1 - gs) * pred_neg
            if guidance_rescale > 0:
                x_0_pos = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred_pos
                x_0_cfg = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred
                std_pos = _std(x_0_pos, dense)
                std_cfg = _std(x_0_cfg, dense)
                x_0_rescaled = x_0_cfg * _bcast(std_pos / std_cfg, x_0_cfg, dense)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = ((1 - sigma_min) * x_t - x_0) / (sigma_min + (1 - sigma_min) * t)
        x_t = x_t - (t - t_prev) * pred
    return x_t


def sample_fast(model_call_batched, noise, cond, neg_cond, steps, rescale_t, guidance_strength, guidance_rescale,
                guidance_interval, sigma_min=1e-5, dense=False):
    """model_call_batched(x [N, C] token-major, [t*1000, ...], [cond, ...]) -> [B*N, C]. noise token-major [N, C]."""
    x_t = noise
    N = x_t.shape[0]
    for t, t_prev in t_schedule(steps, rescale_t):
        gs = guidance_strength if guidance_interval[0] <= t <= guidance_interval[1] else 1
        if gs == 1 or gs == 0:
            pred = model_call_batched(x_t, [1000 * t], [cond if gs == 1 else neg_cond])
        else:
            out = model_call_batched(x_t, [1000 * t] * 2, [cond, neg_cond])
            pred_pos, pred_neg = out[:N], out[N:]
            pred = gs * pred_pos + (1 - gs) * pred_neg
            if guidance_rescale > 0:
                a, b = 1 - sigma_min, sigma_min + (1 - sigma_min) * t
                x_0_pos = a * x_t - b * pred_pos
                x_0_cfg = a * x_t - b * pred
                if dense:
                    r = x_0_pos.std() / x_0_cfg.std()
                else:
                    sd = lambda z: ((z ** 2).mean() - z.mean() ** 2).sqrt()
                    r = sd(x_0_pos) / sd(x_0_cfg)
                x_0 = (guidance_rescale * r + (1 - guidance_rescale)) * x_0_cfg
                pred = (a * x_t - x_0) / b
        x_t = x_t - (t - t_prev) * pred
    return x_t
