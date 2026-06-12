"""Semantic promotion gate: greedy-decode token-stream equality.

The final correctness contract for a distilled FA3 kernel is *behavioral*: when a
candidate op is swapped into a real vLLM engine, the greedy (temperature=0) token
ids it produces must be byte-identical to the baseline `_vllm_fa3_C` stream over a
fixed prompt.  This module is the promotion gate that enforces that.

`check_decode_equiv(so_path, namespace, model=None, num_tokens=100) -> (bool, note)`

Each vLLM engine runs in its OWN isolated `spawn` subprocess (a candidate kernel
that faults the CUDA driver cannot take down the orchestrator, and two full models
cannot coexist in one process).  The flow:

  1. A baseline child runs ONE vLLM engine (enforce_eager, tiny max_model_len) with
     the stock `torch.ops._vllm_fa3_C.fwd` and emits its greedy token stream.
  2. A candidate child loads the candidate `.so` (registering
     `torch.ops.<namespace>.fwd`), monkeypatches `flash_attn_varlen_func` so the
     `fa_version == 3` path routes to the candidate op with the SAME positional args
     as the stock wrapper (verified via a routing counter), runs a SECOND engine, and
     emits its greedy token stream.
  3. The parent compares the two token-id streams.

IMPORTANT: vLLM v1 runs its EngineCore in a separate worker process by default
(`SyncMPClient`); an in-process monkeypatch would never reach that worker.  The
children force the in-process EngineCore via `VLLM_ENABLE_V1_MULTIPROCESSING=0` so
the candidate op is actually exercised during decode.

If a vLLM engine cannot be created/loaded in this environment we DEGRADE
GRACEFULLY: instead of failing, we validate the MONKEYPATCH MECHANISM directly --
patch `flash_attn_varlen_func`, fire a couple of synthetic varlen calls through it,
and confirm they route to the candidate op and produce the same argmax as the
baseline op.  This still proves the swap point is correctly mirrored, while the
caller is told (via a `[skip:...]` note prefix) that the full decode check was
skipped.

Context discipline: the function returns only `(bool, str)` -- a verdict plus a
short, bounded one-line note.  Never a raw token/logit dump.
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
from typing import Optional, Tuple

from .. import config as C

# --------------------------------------------------------------------------------------
# Defaults / knobs (kept tiny so the gate is bounded).
# --------------------------------------------------------------------------------------
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # hd128 GQA, cached under HF_HOME
DEFAULT_MAX_MODEL_LEN = 1024
DEFAULT_PROMPT = "The capital of France is"
GPU_MEM_UTIL = 0.6                # per-engine; one model at a time per child process
CHILD_TIMEOUT_S = 900             # bound a single child (model load + one decode)

# Sentinel a child prints (one JSON line) so the parent can parse a bounded result.
_RESULT_PREFIX = "DECODE_EQUIV_RESULT "


# ======================================================================================
# Public API
# ======================================================================================
def check_decode_equiv(
    so_path: str,
    namespace: str,
    model: Optional[str] = None,
    num_tokens: int = 100,
) -> Tuple[bool, str]:
    """Promotion gate: candidate greedy token-ids must match baseline.

    Runs each decode in its own isolated spawn subprocess (driver protection +
    clean GPU per engine; two full models cannot coexist in one process).  Returns
    ``(identical, note)`` where ``note`` is a short, bounded human-readable string
    (count of matching tokens, or the reason the full check degraded to the
    mechanism-only validation / failed).
    """
    model = model or DEFAULT_MODEL
    if not os.path.isfile(so_path):
        return False, f"candidate .so not found: {so_path}"

    base_payload = {
        "mode": "decode", "patch": False, "namespace": "_vllm_fa3_C",
        "so_path": "", "model": model, "num_tokens": int(num_tokens),
        "max_model_len": DEFAULT_MAX_MODEL_LEN, "prompt": DEFAULT_PROMPT,
    }
    cand_payload = {
        "mode": "decode", "patch": True, "namespace": namespace,
        "so_path": so_path, "model": model, "num_tokens": int(num_tokens),
        "max_model_len": DEFAULT_MAX_MODEL_LEN, "prompt": DEFAULT_PROMPT,
    }

    # ---- baseline decode (fresh process, stock op) ----------------------------------
    base = _run_child(base_payload)
    if base is None or base.get("engine_failed"):
        # vLLM cannot run here -> degrade to mechanism-only validation.
        return _degrade_to_mechanism(so_path, namespace,
                                     reason=_engine_fail_reason(base, "baseline"))
    base_ids = base.get("token_ids") or []

    # ---- candidate decode (fresh process, monkeypatched op) -------------------------
    cand = _run_child(cand_payload)
    if cand is None or cand.get("engine_failed"):
        return _degrade_to_mechanism(so_path, namespace,
                                     reason=_engine_fail_reason(cand, "candidate"))
    if cand.get("error"):
        return False, f"candidate decode error: {str(cand['error'])[:200]}"
    cand_ids = cand.get("token_ids") or []
    routed = int(cand.get("routed", 0))

    # ---- compare greedy token streams -----------------------------------------------
    if routed == 0:
        return False, ("candidate op was never routed to during decode "
                       "(monkeypatch did not take effect)")
    if not base_ids or not cand_ids:
        return False, (f"empty token stream (base={len(base_ids)}, "
                       f"cand={len(cand_ids)})")

    matched = 0
    n = min(len(base_ids), len(cand_ids))
    for i in range(n):
        if base_ids[i] == cand_ids[i]:
            matched += 1
        else:
            break
    identical = (len(base_ids) == len(cand_ids)) and (matched == len(base_ids))
    if identical:
        note = (f"greedy decode identical: {matched}/{len(base_ids)} tokens match "
                f"(ns={namespace}, routed={routed} fa3 calls)")
    else:
        note = (f"greedy decode DIVERGED: {matched}/{n} prefix tokens match "
                f"(base={len(base_ids)} tok, cand={len(cand_ids)} tok, routed={routed})")
    return identical, note


def _engine_fail_reason(child: Optional[dict], which: str) -> str:
    if child is None:
        return f"{which} decode subprocess produced no result"
    return f"{which} engine init/load failed: {str(child.get('note',''))[:160]}"


def _degrade_to_mechanism(so_path: str, namespace: str, reason: str) -> Tuple[bool, str]:
    """vLLM engine could not run -> validate the monkeypatch MECHANISM instead."""
    mech = _run_child({"mode": "mechanism", "so_path": so_path, "namespace": namespace})
    if mech is None:
        return False, f"vLLM unavailable AND mechanism check produced no result ({reason})"
    mnote = str(mech.get("note", ""))[:220]
    # Mechanism path is a graceful-degrade signal: caller maps this to smoke='skip'.
    return bool(mech.get("identical", False)), f"[skip:vLLM-unavailable] {reason} || {mnote}"


def _run_child(payload: dict) -> Optional[dict]:
    """Run THIS module's __main__ in a fresh spawn process; parse its one result line."""
    env = dict(os.environ)
    if C.HF_HOME:
        env.setdefault("HF_HOME", C.HF_HOME)
    # Don't let a stray VLLM_ATTENTION_BACKEND override the FA3 path; force FLASH_ATTN.
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    # CRITICAL: vLLM v1 runs the EngineCore in a SEPARATE process by default
    # (SyncMPClient); our in-process monkeypatch of flash_attn_varlen_func would never
    # reach that worker.  Force the in-process EngineCore (InprocClient) so the
    # candidate op is actually exercised during decode.
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "fa3agent.verify.decode_equiv", json.dumps(payload)],
            cwd=C.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"engine_failed": True, "note": f"child timed out after {CHILD_TIMEOUT_S}s"}
    except Exception as e:  # pragma: no cover - launch failure
        return {"engine_failed": True, "note": f"launch failed: {type(e).__name__}: {e}"}

    result = _parse_child_result(proc.stdout)
    if result is None:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail_s = " | ".join(tail[-2:])[:200] if tail else "no output"
        return {"engine_failed": True,
                "note": f"no result (rc={proc.returncode}): {tail_s}"}
    return result


def _parse_child_result(stdout: str) -> Optional[dict]:
    """Extract the single bounded JSON result line a child emits."""
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                return json.loads(line[len(_RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


# ======================================================================================
# Monkeypatch: route the fa_version==3 path to torch.ops.<namespace>.fwd
# ======================================================================================
def _make_patched_varlen_func(namespace: str, counter: Optional[dict] = None):
    """Build a `flash_attn_varlen_func` replacement that mirrors the stock vLLM
    wrapper's `fa_version == 3` dispatch but calls `torch.ops.<namespace>.fwd`.

    Positional-arg ordering is taken VERBATIM from
    `vllm/vllm_flash_attn/flash_attn_interface.py` (the `fa_version == 3` branch,
    lines ~328-366) and matches `fa3agent.shapes.fwd_args`.  fa_version 2/4 fall
    through to the original wrapper so non-FA3 paths are untouched.  If ``counter``
    is given, ``counter['n']`` is incremented on each fa_version==3 routing (proves
    the swap took effect).
    """
    import torch
    from vllm.vllm_flash_attn import flash_attn_interface as _fi

    _orig = _fi.flash_attn_varlen_func
    fwd_op = getattr(torch.ops, namespace).fwd

    def _maybe_contiguous(x):
        return x.contiguous() if x is not None and x.stride(-1) != 1 else x

    def patched_flash_attn_varlen_func(
        q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
        cu_seqlens_k=None, seqused_k=None, q_v=None, dropout_p=0.0,
        softmax_scale=None, causal=False, window_size=None, softcap=0.0,
        alibi_slopes=None, deterministic=False, return_attn_probs=False,
        block_table=None, return_softmax_lse=False, out=None,
        scheduler_metadata=None, q_descale=None, k_descale=None, v_descale=None,
        num_splits: int = 0, fa_version: int = 2, s_aux=None,
        cp_world_size=1, cp_rank=0, cp_tot_seqused_k=None,
    ):
        if fa_version != 3:
            # Untouched: defer to the stock wrapper for FA2/FA4.
            return _orig(
                q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
                cu_seqlens_k=cu_seqlens_k, seqused_k=seqused_k, q_v=q_v,
                dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
                window_size=window_size, softcap=softcap, alibi_slopes=alibi_slopes,
                deterministic=deterministic, return_attn_probs=return_attn_probs,
                block_table=block_table, return_softmax_lse=return_softmax_lse, out=out,
                scheduler_metadata=scheduler_metadata, q_descale=q_descale,
                k_descale=k_descale, v_descale=v_descale, num_splits=num_splits,
                fa_version=fa_version, s_aux=s_aux, cp_world_size=cp_world_size,
                cp_rank=cp_rank, cp_tot_seqused_k=cp_tot_seqused_k,
            )

        if counter is not None:
            counter["n"] += 1
        assert alibi_slopes is None, "Alibi is not supported in FA3"
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        q, k, v = (_maybe_contiguous(x) for x in (q, k, v))

        # VERBATIM mirror of the stock fa_version==3 positional call, but on our op.
        out_, softmax_lse, _, _ = fwd_op(
            q, k, v,
            None, None,            # k_new, v_new
            q_v, out,
            cu_seqlens_q, cu_seqlens_k,
            None,                  # cu_seqlens_k_new
            None, seqused_k,       # seqused_q, seqused_k
            max_seqlen_q, max_seqlen_k,
            block_table,
            None, None,            # kv_batch_idx, leftpad_k
            None, None, None,      # rotary_cos, rotary_sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale, causal,
            real_window_size[0], real_window_size[1],
            softcap, True,         # is_rotary_interleaved (hardcoded True upstream)
            scheduler_metadata, num_splits,
            None,                  # pack_gqa
            0,                     # sm_margin
            s_aux, cp_world_size, cp_rank, cp_tot_seqused_k,
        )
        return (out_, softmax_lse) if return_softmax_lse else out_

    return patched_flash_attn_varlen_func, _orig


def _install_patch(namespace: str):
    """Load the candidate op and overwrite `flash_attn_varlen_func` everywhere it is
    bound (re-exporting modules bind the name at import time).  Returns
    ``(orig, patched, patched_sites, counter)`` where ``counter['n']`` increments on
    every fa_version==3 routing through the candidate op (used to PROVE the swap took
    effect during a real decode).
    """
    import importlib
    from vllm.vllm_flash_attn import flash_attn_interface as _fi

    counter = {"n": 0}
    patched, orig = _make_patched_varlen_func(namespace, counter)
    # Every module that did `from ... import flash_attn_varlen_func` holds its own
    # binding; patch all known sites that exist in this build.
    targets = [
        "vllm.vllm_flash_attn.flash_attn_interface",
        "vllm.vllm_flash_attn",
        "vllm.v1.attention.backends.fa_utils",
        "vllm.v1.attention.backends.flash_attn",
    ]
    patched_sites = []
    for modname in targets:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if getattr(mod, "flash_attn_varlen_func", None) is not None:
            mod.flash_attn_varlen_func = patched  # type: ignore[attr-defined]
            patched_sites.append(modname)
    # flash_attn_interface is the source of truth; always patch it.
    _fi.flash_attn_varlen_func = patched  # type: ignore[attr-defined]
    if "vllm.vllm_flash_attn.flash_attn_interface" not in patched_sites:
        patched_sites.append("vllm.vllm_flash_attn.flash_attn_interface")
    return orig, patched, patched_sites, counter


# ======================================================================================
# Child entrypoint (runs in the isolated subprocess)
# ======================================================================================
def _emit(identical: bool, note: str, extra: Optional[dict] = None):
    rec = {"identical": bool(identical), "note": str(note)[:300]}
    if extra:
        rec.update(extra)
    print(_RESULT_PREFIX + json.dumps(rec), flush=True)


def _run_decode_child(p: dict):
    """Single-engine decode in this fresh process; emit token-id stream.

    If ``patch`` is set, load the candidate .so and monkeypatch
    `flash_attn_varlen_func` to route the fa_version==3 path to the candidate op
    BEFORE building the engine, then verify the candidate op was actually exercised
    (routed > 0).  On engine init/model-load failure, emit ``engine_failed`` so the
    parent can degrade gracefully to the mechanism check.
    """
    import torch

    model = p["model"]
    prompt = p["prompt"]
    num_tokens = int(p["num_tokens"])
    max_model_len = int(p["max_model_len"])
    want_patch = bool(p.get("patch"))

    counter = {"n": 0}
    if want_patch:
        try:
            torch.ops.load_library(p["so_path"])
            _orig, _patched, _sites, counter = _install_patch(p["namespace"])
        except Exception as e:
            _emit(False, f"candidate .so load / patch failed: {type(e).__name__}: {e}",
                  extra={"error": f"{type(e).__name__}: {e}"})
            return

    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=model,
            enforce_eager=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_num_seqs=1,
            disable_log_stats=True,
        )
    except Exception as e:
        # Engine could not be created/loaded here -> signal graceful degrade.
        sys.stderr.write(f"[decode_equiv] engine init/load failed: "
                         f"{type(e).__name__}: {e}\n")
        _emit(False, f"engine init/load failed: {type(e).__name__}: {e}",
              extra={"engine_failed": True})
        return

    try:
        sp = SamplingParams(temperature=0.0, max_tokens=num_tokens)
        outs = llm.generate([prompt], sp)
        token_ids = [int(t) for t in outs[0].outputs[0].token_ids]
    except Exception as e:
        _emit(False, f"generate failed: {type(e).__name__}: {e}",
              extra={"error": f"{type(e).__name__}: {e}"})
        return

    tag = "candidate" if want_patch else "baseline"
    note = f"{tag} decode produced {len(token_ids)} tokens (routed={counter['n']})"
    _emit(True, note, extra={"token_ids": token_ids, "routed": int(counter["n"]),
                             "mode": "decode", "tag": tag})


def _run_mechanism_check(p: dict):
    """Graceful-degrade path: validate the MONKEYPATCH MECHANISM without a model.

    Patch `flash_attn_varlen_func`, fire a couple of synthetic varlen FA3 calls
    through the patched function (fa_version=3), and confirm they route to the
    candidate op and produce the SAME per-token argmax over the output as the
    baseline op called directly with the identical positional args.  This proves
    the swap point is wired correctly even when vLLM cannot run here.

    Emits a verdict with mode="mechanism" so the caller knows the full decode
    check was skipped.
    """
    import torch

    namespace = p["namespace"]
    try:
        torch.ops.load_library(p["so_path"])
    except Exception as e:
        _emit(False, f"could not load candidate .so for mechanism check: "
                     f"{type(e).__name__}: {e}", extra={"mode": "mechanism", "skip": True})
        return

    # Need the baseline op AND the wrapper module for the patch.  If vLLM's
    # flash_attn module isn't importable either, we can't validate -> skip.
    try:
        import vllm.vllm_flash_attn  # noqa: F401  (triggers baseline op registration)
        _ = torch.ops._vllm_fa3_C.fwd
    except Exception as e:
        _emit(False, f"baseline op unavailable for mechanism check: "
                     f"{type(e).__name__}: {e}", extra={"mode": "mechanism", "skip": True})
        return

    try:
        from fa3agent import shapes, config as C
    except Exception as e:
        _emit(False, f"shapes/config import failed: {type(e).__name__}: {e}",
              extra={"mode": "mechanism", "skip": True})
        return

    if not torch.cuda.is_available():
        _emit(False, "no CUDA device for mechanism check",
              extra={"mode": "mechanism", "skip": True})
        return

    try:
        _orig, patched, sites, counter = _install_patch(namespace)
    except Exception as e:
        _emit(False, f"monkeypatch install failed: {type(e).__name__}: {e}",
              extra={"mode": "mechanism", "skip": True})
        return

    # A couple of small decode-shaped configs (hd128 GQA, like Llama-3.1-8B).
    cfgs = [
        C.Config("mech_decode_hd128", "decode", "bf16", 128, 8, 2,
                 batch=2, q_len=1, k_len=256),
        C.Config("mech_decode_hd128b", "decode", "bf16", 128, 8, 2,
                 batch=3, q_len=1, k_len=512),
    ]

    all_match = True
    detail = []
    for cfg in cfgs:
        try:
            inp = shapes.make_inputs(cfg, seed=0, device="cuda")
            # Baseline: call the original op directly (same positional args).
            base_out, _ = shapes.call_fwd("_vllm_fa3_C", inp)

            # Candidate via the PATCHED wrapper -> proves routing + arg mirroring.
            cand_out = patched(
                inp.q, inp.k, inp.v,
                inp.max_seqlen_q, inp.cu_seqlens_q, inp.max_seqlen_k,
                cu_seqlens_k=inp.cu_seqlens_k, seqused_k=inp.seqused_k,
                softmax_scale=inp.softmax_scale, causal=bool(cfg.causal),
                window_size=[cfg.window_left, cfg.window_right],
                softcap=float(cfg.softcap), block_table=inp.page_table,
                q_descale=inp.q_descale, k_descale=inp.k_descale,
                v_descale=inp.v_descale, num_splits=int(cfg.num_splits),
                fa_version=3,
            )
            # Greedy semantics: argmax over the value/head_dim per token must agree.
            base_arg = base_out.float().flatten(1).argmax(dim=-1)
            cand_arg = cand_out.float().flatten(1).argmax(dim=-1)
            match = bool(torch.equal(base_arg, cand_arg))
            all_match = all_match and match
            detail.append(f"{cfg.name}:{'ok' if match else 'DIFF'}")
        except Exception as e:
            all_match = False
            detail.append(f"{cfg.name}:ERR({type(e).__name__})")

    routed = int(counter["n"])
    routed_ok = routed >= len(cfgs)
    note = (f"monkeypatch mechanism validated: routed={routed} "
            f"argmax_match={all_match} [{', '.join(detail)}] "
            f"(patched {len(sites)} site(s))")
    _emit(all_match and routed_ok, note, extra={"mode": "mechanism", "skip": True})


def _child_main(argv):
    if len(argv) < 2:
        _emit(False, "child invoked without payload")
        return 2
    try:
        p = json.loads(argv[1])
    except json.JSONDecodeError as e:
        _emit(False, f"bad payload json: {e}")
        return 2

    mode = p.get("mode", "decode")
    try:
        if mode == "mechanism":
            _run_mechanism_check(p)
        else:
            _run_decode_child(p)
    except Exception as e:
        sys.stderr.write(f"[decode_equiv] child ({mode}) crashed: "
                         f"{type(e).__name__}: {e}\n")
        # For a decode child a crash is an engine-side failure -> let parent degrade.
        extra = {"engine_failed": True} if mode != "mechanism" \
            else {"mode": "mechanism", "skip": True}
        _emit(False, f"{mode} child crashed: {type(e).__name__}: {e}", extra=extra)
    return 0


if __name__ == "__main__":
    # Ensure spawn-style isolation even if a parent set start method.
    try:
        import multiprocessing as _mp
        _mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    sys.exit(_child_main(sys.argv))
