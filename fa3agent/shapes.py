"""Authoritative input construction for the config matrix.

The SAME tensors must drive both the baseline (`_vllm_fa3_C`) and a candidate
(`_distill_fa3_C` / `_vllm_fa3_C`) so correctness and perf are apples-to-apples.
The 37-positional-arg ordering of the `fwd` op is encoded in exactly ONE place
(`fwd_args`), mirroring how vLLM's wrapper calls the op.

Layouts (verified against the baseline this session, see .research/golden_scripts/):
  q                : (total_q, num_heads_q, head_dim)     total_q = sum(q_len)
  paged KV cache   : k,v each (num_blocks, block_size, num_heads_kv, head_dim)
  non-paged KV     : k,v each (total_k, num_heads_kv, head_dim)
  block_table      : (batch, max_blocks_per_seq) int32
  cu_seqlens_q     : (batch+1,) int32   (cumulative q lengths)
  cu_seqlens_k     : (batch+1,) int32   (non-paged only)
  seqused_k        : (batch,) int32     (paged only; kv length per sequence)
  q/k/v_descale    : (batch, num_heads_kv) fp32  (fp8 only)
  out              : op allocates if None (bf16 when fp8 inputs)
  softmax_lse      : (num_heads_q, total_q) fp32
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from . import config as C


@dataclass
class Inputs:
    cfg: C.Config
    q: torch.Tensor
    k: torch.Tensor               # paged k_cache, or contiguous k
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: Optional[torch.Tensor]
    seqused_k: Optional[torch.Tensor]
    page_table: Optional[torch.Tensor]
    max_seqlen_q: int
    max_seqlen_k: int
    softmax_scale: float
    q_descale: Optional[torch.Tensor] = None
    k_descale: Optional[torch.Tensor] = None
    v_descale: Optional[torch.Tensor] = None


def _torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, C.TORCH_DTYPE[name])


def make_inputs(cfg: C.Config, seed: int = 0, device: str = "cuda") -> Inputs:
    """Construct identical, deterministic inputs for a config."""
    g = torch.Generator(device=device).manual_seed(seed)
    dt = _torch_dtype(cfg.dtype)
    is_fp8 = cfg.dtype == "fp8e4m3"
    gen_dt = torch.bfloat16 if is_fp8 else dt  # generate in bf16 then cast to fp8

    b, hq, hkv, hd, hdv = cfg.batch, cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim, cfg.hdv
    q_len, k_len, block = cfg.q_len, cfg.k_len, cfg.block_size
    total_q = b * q_len

    q = torch.randn(total_q, hq, hd, generator=g, device=device, dtype=gen_dt)
    cu_q = torch.arange(0, (b + 1) * q_len, q_len, device=device, dtype=torch.int32)
    softmax_scale = hd ** -0.5

    qd = kd = vd = None
    if cfg.paged:
        nblocks_per = (k_len + block - 1) // block
        total_blocks = b * nblocks_per + 4
        kc = torch.randn(total_blocks, block, hkv, hd, generator=g, device=device, dtype=gen_dt)
        vc = torch.randn(total_blocks, block, hkv, hdv, generator=g, device=device, dtype=gen_dt)
        page_table = torch.arange(b * nblocks_per, device=device, dtype=torch.int32).reshape(b, nblocks_per)
        seqused_k = torch.full((b,), k_len, device=device, dtype=torch.int32)
        cu_k = None
        k, v = kc, vc
    else:
        total_k = b * k_len
        k = torch.randn(total_k, hkv, hd, generator=g, device=device, dtype=gen_dt)
        v = torch.randn(total_k, hkv, hdv, generator=g, device=device, dtype=gen_dt)
        cu_k = torch.arange(0, (b + 1) * k_len, k_len, device=device, dtype=torch.int32)
        seqused_k = None
        page_table = None

    if is_fp8:
        f8 = torch.float8_e4m3fn
        q = q.to(f8); k = k.to(f8); v = v.to(f8)
        ds = (b, hkv)
        qd = torch.ones(ds, device=device, dtype=torch.float32)
        kd = torch.ones(ds, device=device, dtype=torch.float32)
        vd = torch.ones(ds, device=device, dtype=torch.float32)

    return Inputs(cfg=cfg, q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                  seqused_k=seqused_k, page_table=page_table,
                  max_seqlen_q=q_len, max_seqlen_k=k_len, softmax_scale=softmax_scale,
                  q_descale=qd, k_descale=kd, v_descale=vd)


def fwd_args(inp: Inputs, out: Optional[torch.Tensor] = None) -> tuple:
    """Build the 37-positional-arg tuple for `torch.ops.<ns>.fwd(*args)`.

    Ordering mirrors config.FWD_SCHEMA and vLLM's wrapper dispatch exactly.
    """
    cfg = inp.cfg
    return (
        inp.q, inp.k, inp.v,
        None, None, None,                          # k_new, v_new, q_v
        out,                                       # out (None -> op allocates)
        inp.cu_seqlens_q, inp.cu_seqlens_k, None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
        None, inp.seqused_k,                       # seqused_q, seqused_k
        inp.max_seqlen_q, inp.max_seqlen_k,        # max_seqlen_q, max_seqlen_k
        inp.page_table, None, None,                # page_table, kv_batch_idx, leftpad_k
        None, None, None,                          # rotary_cos, rotary_sin, seqlens_rotary
        inp.q_descale, inp.k_descale, inp.v_descale,
        inp.softmax_scale, bool(cfg.causal),
        int(cfg.window_left), int(cfg.window_right),
        float(cfg.softcap), True,                  # softcap, is_rotary_interleaved
        None, int(cfg.num_splits), None,           # scheduler_metadata, num_splits, pack_gqa
        0,                                         # sm_margin
        None, 1, 0, None,                          # s_aux, cp_world_size, cp_rank, cp_tot_seqused_k
    )


def call_fwd(namespace: str, inp: Inputs, out: Optional[torch.Tensor] = None):
    """Invoke `torch.ops.<namespace>.fwd` and return (out, softmax_lse)."""
    op = getattr(torch.ops, namespace).fwd
    res = op(*fwd_args(inp, out))
    return res[0], res[1]  # [out, softmax_lse, out_accum, lse_accum]


def ref_fp32(inp: Inputs) -> torch.Tensor:
    """Pure-PyTorch fp32 reference attention for the 2x-error correctness criterion.

    Handles GQA/MQA, causal, sliding window, softcap, paged + non-paged KV.
    Returns out in fp32, shape (total_q, num_heads_q, head_dim_v). Slow oracle only.
    """
    cfg = inp.cfg
    b, hq, hkv, hd, hdv = cfg.batch, cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim, cfg.hdv
    q_len, k_len = cfg.q_len, cfg.k_len
    rep = hq // hkv
    dev = inp.q.device

    q = inp.q.to(torch.float32).view(b, q_len, hq, hd)
    if cfg.paged:
        block = cfg.block_size
        nbp = (k_len + block - 1) // block
        kfull = inp.k.to(torch.float32)  # (nblocks, block, hkv, hd)
        vfull = inp.v.to(torch.float32)
        K = torch.empty(b, k_len, hkv, hd, device=dev, dtype=torch.float32)
        V = torch.empty(b, k_len, hkv, hdv, device=dev, dtype=torch.float32)
        pt = inp.page_table
        for bi in range(b):
            fk = kfull[pt[bi, :nbp].long()].reshape(nbp * block, hkv, hd)
            fv = vfull[pt[bi, :nbp].long()].reshape(nbp * block, hkv, hdv)
            K[bi] = fk[:k_len]; V[bi] = fv[:k_len]
    else:
        K = inp.k.to(torch.float32).view(b, k_len, hkv, hd)
        V = inp.v.to(torch.float32).view(b, k_len, hkv, hdv)

    out = torch.empty(b, q_len, hq, hdv, device=dev, dtype=torch.float32)
    scale = inp.softmax_scale
    qpos = torch.arange(k_len - q_len, k_len, device=dev).view(1, q_len, 1)
    kpos = torch.arange(k_len, device=dev).view(1, 1, k_len)
    mask = torch.zeros(1, q_len, k_len, device=dev, dtype=torch.bool)
    if cfg.causal:
        mask = mask | (kpos > qpos)
    if cfg.window_left >= 0:
        mask = mask | (kpos < (qpos - cfg.window_left))
    if cfg.window_right >= 0:
        mask = mask | (kpos > (qpos + cfg.window_right))
    for h in range(hq):
        hk = h // rep
        qi = q[:, :, h, :]
        ki = K[:, :, hk, :]
        vi = V[:, :, hk, :]
        s = torch.einsum("bqd,bkd->bqk", qi, ki) * scale
        if cfg.softcap and cfg.softcap > 0:
            s = cfg.softcap * torch.tanh(s / cfg.softcap)
        s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, :, h, :] = torch.einsum("bqk,bkd->bqd", p, vi)
    return out.reshape(b * q_len, hq, hdv)
