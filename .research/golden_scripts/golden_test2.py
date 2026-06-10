import torch
import vllm.vllm_flash_attn._vllm_fa3_C  # noqa
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
torch.manual_seed(0); dev='cuda'

def run(hd=128, dt=torch.bfloat16, kvdt=None, nh=8, nkv=2, softcap=0.0, window=None, splits=0, fp8=False):
    bs=2; block=16; seqlen=128
    nbp=(seqlen+block-1)//block; tb=bs*nbp+2
    q=torch.randn(bs,nh,hd,device=dev,dtype=dt)
    kc=torch.randn(tb,block,nkv,hd,device=dev,dtype=dt)
    vc=torch.randn(tb,block,nkv,hd,device=dev,dtype=dt)
    qd=kd=vd=None
    if fp8:
        f8=torch.float8_e4m3fn
        q=q.to(f8); kc=kc.to(f8); vc=vc.to(f8)
        descale_shape=(bs,nkv)
        qd=torch.ones(descale_shape,device=dev); kd=torch.ones(descale_shape,device=dev); vd=torch.ones(descale_shape,device=dev)
    bt=torch.arange(bs*nbp,device=dev,dtype=torch.int32).reshape(bs,nbp)
    cu=torch.arange(0,bs+1,device=dev,dtype=torch.int32)
    sk=torch.full((bs,),seqlen,device=dev,dtype=torch.int32)
    r=flash_attn_varlen_func(q=q,k=kc,v=vc,max_seqlen_q=1,cu_seqlens_q=cu,max_seqlen_k=seqlen,
        seqused_k=sk,causal=True,block_table=bt,fa_version=3,softcap=softcap,
        window_size=window,num_splits=splits,q_descale=qd,k_descale=kd,v_descale=vd,
        return_softmax_lse=True)
    out,lse=r
    print(f'hd={hd} dt={str(dt).split(".")[-1]} fp8={fp8} nh/nkv={nh}/{nkv} softcap={softcap} win={window} splits={splits} -> out{tuple(out.shape)} lse{tuple(lse.shape)} mean={float(out.float().abs().mean()):.4f}')

for hd in [64,96,128,192,256]:
    run(hd=hd)
run(dt=torch.float16)
run(fp8=True, hd=128)
run(softcap=30.0)
run(window=[64,0])         # sliding window left=64
run(splits=4)              # forced split-KV
run(nh=16,nkv=2)           # GQA group=8
# raw op: check return arity
import torch as T
out,lse,a,b = T.ops._vllm_fa3_C.fwd(
    T.randn(2,8,128,device=dev,dtype=T.bfloat16),
    T.randn(34,16,2,128,device=dev,dtype=T.bfloat16),
    T.randn(34,16,2,128,device=dev,dtype=T.bfloat16),
    None,None,None,None,
    T.arange(0,3,device=dev,dtype=T.int32),None,None,None,
    T.full((2,),128,device=dev,dtype=T.int32),1,128,
    T.arange(16,device=dev,dtype=T.int32).reshape(2,8),
    None,None,None,None,None,None,None,None,
    0.08838,True,-1,-1,0.0,True,None,0,None,0,None,1,0,None)
print('raw fwd returns 4:', out.shape, lse.shape, type(a), type(b))
print("ALL OK2")
