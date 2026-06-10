import torch
import vllm.vllm_flash_attn._vllm_fa3_C  # noqa
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func, get_scheduler_metadata

torch.manual_seed(0)
dev='cuda'; dt=torch.bfloat16

def paged_decode():
    # decode: 1 query token per sequence, paged KV cache
    bs=4; nh=8; nkv=2; hd=128; block=16; seqlen=200
    nblocks_per=(seqlen+block-1)//block
    total_blocks=bs*nblocks_per+4
    q=torch.randn(bs,nh,hd,device=dev,dtype=dt)             # (total_q=bs, nh, hd) since 1 tok each
    # KV cache paged: (num_blocks, block, nkv, hd)
    kc=torch.randn(total_blocks,block,nkv,hd,device=dev,dtype=dt)
    vc=torch.randn(total_blocks,block,nkv,hd,device=dev,dtype=dt)
    block_table=torch.arange(bs*nblocks_per,device=dev,dtype=torch.int32).reshape(bs,nblocks_per)
    cu_q=torch.arange(0,bs+1,device=dev,dtype=torch.int32)  # 1 tok per seq
    seqused_k=torch.full((bs,),seqlen,device=dev,dtype=torch.int32)
    out=flash_attn_varlen_func(q=q,k=kc,v=vc,max_seqlen_q=1,cu_seqlens_q=cu_q,
        max_seqlen_k=seqlen,seqused_k=seqused_k,causal=True,block_table=block_table,
        fa_version=3,num_splits=0)
    print('decode out', out.shape, out.dtype, float(out.float().abs().mean()))

def prefill_varlen():
    # prefill: varlen, no paging (cu_seqlens_k), causal
    nh=8; nkv=2; hd=128
    lens=[100,50]; cu=[0]
    for l in lens: cu.append(cu[-1]+l)
    tq=cu[-1]
    q=torch.randn(tq,nh,hd,device=dev,dtype=dt)
    k=torch.randn(tq,nkv,hd,device=dev,dtype=dt)
    v=torch.randn(tq,nkv,hd,device=dev,dtype=dt)
    cu_q=torch.tensor(cu,device=dev,dtype=torch.int32)
    cu_k=torch.tensor(cu,device=dev,dtype=torch.int32)
    out=flash_attn_varlen_func(q=q,k=k,v=v,max_seqlen_q=max(lens),cu_seqlens_q=cu_q,
        max_seqlen_k=max(lens),cu_seqlens_k=cu_k,causal=True,fa_version=3)
    print('prefill out', out.shape, out.dtype, float(out.float().abs().mean()))

def sched_meta():
    bs=4; nh=8; nkv=2; hd=128; seqlen=200
    cache_seqlens=torch.full((bs,),seqlen,device=dev,dtype=torch.int32)
    sm=get_scheduler_metadata(batch_size=bs,max_seqlen_q=1,max_seqlen_k=seqlen,
        num_heads_q=nh,num_heads_kv=nkv,headdim=hd,cache_seqlens=cache_seqlens,
        qkv_dtype=dt,page_size=16,causal=True,num_splits=0)
    print('sched_meta', sm.shape, sm.dtype)

paged_decode(); prefill_varlen(); sched_meta()
print("ALL OK")
