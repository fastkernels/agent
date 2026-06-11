"""Faithful in-situ A/B correctness gate (cheaper than full decode-equality).

The calibration study (Experiment C) proved that the decisive failure surface -- the LIVE
vLLM paged-KV / broadcast-descale tensors -- does NOT survive clone()/serialization, so an
OFFLINE replay against captured tensors is unfaithful (it silently densifies broadcast
descales and masks the bug).  The only faithful-yet-cheap check is to compare the candidate
against the baseline ON THE LIVE TENSORS during a *short* real vLLM forward -- without the
full 100-token decode loop.

`check_insitu(so_path, namespace, model=None, num_steps=4, max_model_len=512) -> dict`

In an isolated spawn subprocess: load the candidate .so, monkeypatch flash_attn_varlen_func
so each fa_version==3 call runs BOTH the baseline op (which DRIVES the decode, so the run
stays faithful) AND the candidate op on the SAME live args, recording per-call out maxdiff /
mean-bias / argmax-match.  A few decode steps exercise prefill + decode calls across all
layers.  Returns a bounded dict: {ok, n_calls, max_diff, max_bias, argmax_match_all,
first_bad_call, worst, note}.  ok == (max_diff <= atol AND argmax_match_all).

Cost: ONE short forward (~one model load), not two full decodes.  Use as the promotion-tier
faithful gate; keep decode_equiv (full greedy tokens) as the final acceptance gate.
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
from typing import Optional

from .. import config as C

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PROMPT = "The capital of France is"
GPU_MEM_UTIL = 0.6
CHILD_TIMEOUT_S = 900
_RESULT_PREFIX = "INSITU_RESULT "


def check_insitu(so_path: str, namespace: str, model: Optional[str] = None,
                 num_steps: int = 4, max_model_len: int = 512) -> dict:
    model = model or DEFAULT_MODEL
    if not os.path.isfile(so_path):
        return {"ok": False, "note": f"candidate .so not found: {so_path}"}
    payload = {"so_path": so_path, "namespace": namespace, "model": model,
               "num_steps": int(num_steps), "max_model_len": int(max_model_len),
               "prompt": DEFAULT_PROMPT}
    res = _run_child(payload)
    if res is None or res.get("engine_failed"):
        note = (res or {}).get("note", "no result")
        return {"ok": False, "skip": True, "note": f"[skip:vLLM-unavailable] {note}"}
    return res


def _run_child(payload: dict) -> Optional[dict]:
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/data-fast/huggingface")
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # in-process EngineCore so the patch is seen
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "fa3agent.verify.insitu", json.dumps(payload)],
            cwd="/data-fast/agent", env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"engine_failed": True, "note": f"insitu child timed out after {CHILD_TIMEOUT_S}s"}
    except Exception as e:  # noqa
        return {"engine_failed": True, "note": f"launch failed: {type(e).__name__}: {e}"}
    if not proc.stdout:
        tail = (proc.stderr or "").strip().splitlines()
        return {"engine_failed": True, "note": f"no output (rc={proc.returncode}): {' | '.join(tail[-2:])[:200]}"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                return json.loads(line[len(_RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return {"engine_failed": True, "note": f"no result line (rc={proc.returncode}): {' | '.join(tail[-2:])[:200]}"}


# ======================================================================================
# Child (isolated subprocess)
# ======================================================================================
def _dtype_key(t) -> str:
    import torch
    return {torch.bfloat16: "bf16", torch.float16: "fp16",
            torch.float8_e4m3fn: "fp8e4m3"}.get(t.dtype, "bf16")


def _install_ab_patch(namespace: str, stats: dict):
    """Patch flash_attn_varlen_func so fa3 calls run baseline (driver) AND candidate (compare)."""
    import torch
    import importlib
    from vllm.vllm_flash_attn import flash_attn_interface as _fi
    base_op = torch.ops._vllm_fa3_C.fwd
    cand_op = getattr(torch.ops, namespace).fwd

    def _mc(x):
        return x.contiguous() if x is not None and x.stride(-1) != 1 else x

    def patched(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
                cu_seqlens_k=None, seqused_k=None, q_v=None, dropout_p=0.0,
                softmax_scale=None, causal=False, window_size=None, softcap=0.0,
                alibi_slopes=None, deterministic=False, return_attn_probs=False,
                block_table=None, return_softmax_lse=False, out=None,
                scheduler_metadata=None, q_descale=None, k_descale=None, v_descale=None,
                num_splits=0, fa_version=2, s_aux=None, cp_world_size=1, cp_rank=0,
                cp_tot_seqused_k=None):
        if fa_version != 3:
            return _orig(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
                         cu_seqlens_k=cu_seqlens_k, seqused_k=seqused_k, q_v=q_v, dropout_p=dropout_p,
                         softmax_scale=softmax_scale, causal=causal, window_size=window_size,
                         softcap=softcap, alibi_slopes=alibi_slopes, deterministic=deterministic,
                         return_attn_probs=return_attn_probs, block_table=block_table,
                         return_softmax_lse=return_softmax_lse, out=out, scheduler_metadata=scheduler_metadata,
                         q_descale=q_descale, k_descale=k_descale, v_descale=v_descale, num_splits=num_splits,
                         fa_version=fa_version, s_aux=s_aux, cp_world_size=cp_world_size, cp_rank=cp_rank,
                         cp_tot_seqused_k=cp_tot_seqused_k)
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        rw = (-1, -1) if window_size is None else (window_size[0], window_size[1])
        q, k, v = (_mc(x) for x in (q, k, v))
        args = (q, k, v, None, None, q_v, None,  # out=None so the candidate allocates fresh
                cu_seqlens_q, cu_seqlens_k, None, None, seqused_k, max_seqlen_q, max_seqlen_k,
                block_table, None, None, None, None, None, q_descale, k_descale, v_descale,
                softmax_scale, causal, rw[0], rw[1], softcap, True, scheduler_metadata,
                num_splits, None, 0, s_aux, cp_world_size, cp_rank, cp_tot_seqused_k)
        base_list = base_op(*args)
        cand_list = cand_op(*args)
        bo, blse = base_list[0], base_list[1]
        co = cand_list[0]
        # per-call compare on the LIVE tensors
        try:
            d = (co.float() - bo.float())
            maxdiff = float(d.abs().max().item()) if d.numel() else 0.0
            bias = float(d.abs().mean().item()) if d.numel() else 0.0
            # greedy semantics: argmax over the flattened per-token feature must agree
            am = bool((co.float().flatten(1).argmax(-1) == bo.float().flatten(1).argmax(-1)).all().item())
        except Exception:
            maxdiff, bias, am = float("inf"), float("inf"), False
        ci = stats["n"]
        stats["n"] += 1
        phase = "prefill" if (q.shape[0] > stats.get("batch", 1) and max_seqlen_q and max_seqlen_q > 1) else "decode"
        if maxdiff > stats["max_diff"]:
            stats["max_diff"] = maxdiff
            stats["worst"] = {"call": ci, "phase": phase, "maxdiff": maxdiff,
                              "out_shape": list(co.shape), "dtype": _dtype_key(co)}
        stats["max_bias"] = max(stats["max_bias"], bias)
        if not am:
            stats["argmax_match_all"] = False
        if stats["first_bad_call"] < 0 and maxdiff > stats["atol"]:
            stats["first_bad_call"] = ci
            stats["first_bad_phase"] = phase
        # drive the decode with the BASELINE result (faithful)
        return (bo, blse) if return_softmax_lse else bo

    _orig = _fi.flash_attn_varlen_func
    targets = ["vllm.vllm_flash_attn.flash_attn_interface", "vllm.vllm_flash_attn",
               "vllm.v1.attention.backends.fa_utils", "vllm.v1.attention.backends.flash_attn"]
    for m in targets:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        if getattr(mod, "flash_attn_varlen_func", None) is not None:
            mod.flash_attn_varlen_func = patched
    _fi.flash_attn_varlen_func = patched
    return _orig


def _emit(rec: dict):
    print(_RESULT_PREFIX + json.dumps(rec), flush=True)


def _child_main(argv):
    if len(argv) < 2:
        _emit({"ok": False, "note": "no payload"}); return 2
    p = json.loads(argv[1])
    import torch
    stats = {"n": 0, "max_diff": 0.0, "max_bias": 0.0, "argmax_match_all": True,
             "first_bad_call": -1, "first_bad_phase": "", "worst": None,
             "atol": C.ALLCLOSE_TOL["bf16"][0]}
    try:
        torch.ops.load_library(p["so_path"])
        _install_ab_patch(p["namespace"], stats)
    except Exception as e:
        _emit({"ok": False, "engine_failed": True, "note": f"load/patch failed: {type(e).__name__}: {e}"}); return 0
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=p["model"], enforce_eager=True, max_model_len=int(p["max_model_len"]),
                  gpu_memory_utilization=GPU_MEM_UTIL, max_num_seqs=1, disable_log_stats=True)
        sp = SamplingParams(temperature=0.0, max_tokens=int(p["num_steps"]))
        llm.generate([p["prompt"]], sp)
    except Exception as e:
        sys.stderr.write(f"[insitu] engine failed: {type(e).__name__}: {e}\n")
        _emit({"ok": False, "engine_failed": True, "note": f"engine init/run failed: {type(e).__name__}: {e}"}); return 0

    atol = stats["atol"]
    ok = (stats["n"] > 0) and (stats["max_diff"] <= atol) and stats["argmax_match_all"]
    note = (f"in-situ A/B over {stats['n']} live fa3 calls: max_diff={stats['max_diff']:.3e} "
            f"(atol={atol:.1e}) max_bias={stats['max_bias']:.2e} argmax_all={stats['argmax_match_all']}")
    if stats["first_bad_call"] >= 0:
        note += f"; first diverging call #{stats['first_bad_call']} ({stats['first_bad_phase']})"
    _emit({"ok": bool(ok), "n_calls": stats["n"], "max_diff": stats["max_diff"],
           "max_bias": stats["max_bias"], "argmax_match_all": stats["argmax_match_all"],
           "first_bad_call": stats["first_bad_call"], "worst": stats["worst"], "note": note})
    return 0


if __name__ == "__main__":
    try:
        import multiprocessing as _mp
        _mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    sys.exit(_child_main(sys.argv))
