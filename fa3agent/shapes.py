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


def _cumsum0(xs: list[int]) -> list[int]:
    out, s = [0], 0
    for x in xs:
        s += int(x)
        out.append(s)
    return out


def make_inputs(cfg: C.Config, seed: int = 0, device: str = "cuda") -> Inputs:
    """Construct identical, deterministic inputs for a config (uniform OR ragged/mixed `mix`).

    Q/K/V all use cfg.dtype.  fp8 tensors get deterministic NON-trivial per-(request, kv-head)
    descales (mirroring vLLM's per-tensor scales) so the dequant arithmetic is actually exercised.
    The uniform non-fp8 path is byte-identical to the previous builder.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    is_fp8 = cfg.dtype == "fp8e4m3"
    gen = torch.bfloat16 if is_fp8 else _torch_dtype(cfg.dtype)   # generate in bf16, then cast to fp8

    hq, hkv, hd, hdv, block = cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim, cfg.hdv, cfg.block_size
    reqs = cfg.requests                          # [(q_len, k_len), ...]; uniform = batch copies
    q_lens = [q for q, _ in reqs]
    k_lens = [k for _, k in reqs]
    b = len(reqs)
    total_q = sum(q_lens)

    q = torch.randn(total_q, hq, hd, generator=g, device=device, dtype=gen)
    cu_q = torch.tensor(_cumsum0(q_lens), device=device, dtype=torch.int32)
    softmax_scale = hd ** -0.5

    if cfg.paged:
        nbp = [(k + block - 1) // block for k in k_lens]
        total_blocks = sum(nbp) + 4
        k = torch.randn(total_blocks, block, hkv, hd, generator=g, device=device, dtype=gen)
        v = torch.randn(total_blocks, block, hkv, hdv, generator=g, device=device, dtype=gen)
        page_table = torch.zeros(b, max(nbp), device=device, dtype=torch.int32)
        nb = 0
        for i, nbi in enumerate(nbp):
            page_table[i, :nbi] = torch.arange(nb, nb + nbi, device=device, dtype=torch.int32)
            nb += nbi
        seqused_k = torch.tensor(k_lens, device=device, dtype=torch.int32)
        cu_k = None
    else:
        total_k = sum(k_lens)
        k = torch.randn(total_k, hkv, hd, generator=g, device=device, dtype=gen)
        v = torch.randn(total_k, hkv, hdv, generator=g, device=device, dtype=gen)
        cu_k = torch.tensor(_cumsum0(k_lens), device=device, dtype=torch.int32)
        seqused_k = None
        page_table = None

    qd = kd = vd = None
    if is_fp8:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        ds = (b, hkv)

        def _descale():  # deterministic non-trivial scale in [0.75, 1.25) per (request, kv-head)
            return torch.rand(ds, generator=g, device=device, dtype=torch.float32) * 0.5 + 0.75

        qd = _descale()
        kd = _descale()
        vd = _descale()

    return Inputs(cfg=cfg, q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                  seqused_k=seqused_k, page_table=page_table,
                  max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens), softmax_scale=softmax_scale,
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

    Handles GQA/MQA, causal, sliding window, softcap, paged + non-paged KV, fp8 descales (Q and/or
    KV), head_dim_v != head_dim, and ragged/MIXED batches (`mix`).  Returns out in fp32, shape
    (total_q, num_heads_q, head_dim_v).  Slow per-request oracle only.

    Query positions are the LAST q_len tokens of the k_len-long sequence (decode/append/chunk
    semantics), so the causal diagonal is offset by (k_len - q_len) -- matching FA's varlen path.
    """
    cfg = inp.cfg
    hq, hkv, hd, hdv = cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim, cfg.hdv
    rep = hq // hkv
    dev = inp.q.device
    block = cfg.block_size
    reqs = cfg.requests
    k_lens = [k for _, k in reqs]

    qf = inp.q.to(torch.float32)        # (total_q, hq, hd)
    kf = inp.k.to(torch.float32)        # paged: (nblk, block, hkv, hd); else (total_k, hkv, hd)
    vf = inp.v.to(torch.float32)
    out = torch.empty(qf.shape[0], hq, hdv, device=dev, dtype=torch.float32)
    scale = inp.softmax_scale

    q_off = 0
    for bi, (ql, kl) in enumerate(reqs):
        qi = qf[q_off:q_off + ql]       # (ql, hq, hd)
        if inp.q_descale is not None:   # fp8 query -> dequant (per kv-head, expanded to q-heads)
            qi = qi * inp.q_descale[bi].repeat_interleave(rep).view(1, hq, 1)
        if cfg.paged:
            nbp = (kl + block - 1) // block
            pt = inp.page_table[bi, :nbp].long()
            Ki = kf[pt].reshape(nbp * block, hkv, hd)[:kl]      # (kl, hkv, hd)
            Vi = vf[pt].reshape(nbp * block, hkv, hdv)[:kl]
        else:
            koff = sum(k_lens[:bi])
            Ki = kf[koff:koff + kl]
            Vi = vf[koff:koff + kl]
        if inp.k_descale is not None:
            Ki = Ki * inp.k_descale[bi].view(1, hkv, 1)
        if inp.v_descale is not None:
            Vi = Vi * inp.v_descale[bi].view(1, hkv, 1)

        qpos = torch.arange(kl - ql, kl, device=dev).view(ql, 1)
        kpos = torch.arange(kl, device=dev).view(1, kl)
        mask = torch.zeros(ql, kl, device=dev, dtype=torch.bool)
        if cfg.causal:
            mask |= kpos > qpos
        if cfg.window_left >= 0:
            mask |= kpos < (qpos - cfg.window_left)
        if cfg.window_right >= 0:
            mask |= kpos > (qpos + cfg.window_right)

        for h in range(hq):
            hk = h // rep
            s = torch.einsum("qd,kd->qk", qi[:, h, :], Ki[:, hk, :]) * scale
            if cfg.softcap and cfg.softcap > 0:
                s = cfg.softcap * torch.tanh(s / cfg.softcap)
            s = s.masked_fill(mask, float("-inf"))
            p = torch.softmax(s, dim=-1)
            out[q_off:q_off + ql, h, :] = torch.einsum("qk,kd->qd", p, Vi[:, hk, :])
        q_off += ql
    return out
