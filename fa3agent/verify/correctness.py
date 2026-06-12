"""Correctness verification for the FA3-distillation agent (verify FIRST).

`check_correctness` runs a candidate op over a list of configs and compares its
output + softmax_lse against (a) the prebuilt baseline op and (b) a pure-fp32
PyTorch reference, using the "fp32-reference 2x-error" criterion plus a coarse
allclose gate and an lse-tolerance gate.

A config passes iff ALL of:
  * err_ratio = out_err / (baseline_err + eps) <= config.ERR_MULT
        out_err      = max|cand_out.float() - ref_fp32|
        baseline_err = max|base_out.float() - ref_fp32|
  * torch.allclose(cand_out.float(), base_out.float(), atol, rtol) with
        (atol, rtol) = config.ALLCLOSE_TOL[cfg.dtype]
  * lse_err = max|cand_lse.float() - base_lse.float()| within config.LSE_TOL

Shape mismatches and NaN/inf in the candidate are treated as hard failures
(correct=False with a terse note), never exceptions that escape.

Output is bounded: a list of `common.ConfigResult` (one per config), filling the
correctness fields and leaving the perf fields at 0. No raw tensors leak out.
"""
from __future__ import annotations

import torch

from .. import config as C
from .. import common
from .. import shapes
from .golden import GoldenCache
from ..runlog import logged


_BASELINE_NS = "_vllm_fa3_C"
_EPS = 1e-12


def _ensure_baseline_loaded() -> None:
    if hasattr(torch.ops, _BASELINE_NS) and hasattr(getattr(torch.ops, _BASELINE_NS), "fwd"):
        return
    import vllm.vllm_flash_attn._vllm_fa3_C  # noqa: F401


def _maxabs(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    return float(t.abs().max().item())


def _check_one(cfg: C.Config, namespace: str, seed: int, gc: GoldenCache) -> common.ConfigResult:
    res = common.ConfigResult(config=cfg.name, correct=False)

    # Cached baseline golden (out + lse + fp32 ref + baseline_err). Tensors land
    # on the current cuda device (set by the caller via set_device).
    golden = gc.get(cfg, seed)
    base_out = golden["out"]
    base_lse = golden["lse"]
    ref = golden["ref"]
    baseline_err = float(golden["baseline_err"])

    # Run the candidate on identical inputs.
    inp = shapes.make_inputs(cfg, seed, device="cuda")
    try:
        cand_out, cand_lse = shapes.call_fwd(namespace, inp)
    except Exception as e:  # candidate op blew up
        res.baseline_err = baseline_err
        res.note = f"candidate fwd raised: {type(e).__name__}: {e}"[:200]
        return res

    cand_out_f = cand_out.float()
    cand_lse_f = cand_lse.float()
    base_out_f = base_out.float()
    base_lse_f = base_lse.float()
    ref_f = ref.float()

    # NaN / inf -> hard fail.
    if not torch.isfinite(cand_out_f).all():
        res.baseline_err = baseline_err
        res.note = "candidate out has NaN/inf"
        return res
    if not torch.isfinite(cand_lse_f).all():
        # lse can legitimately be -inf for fully-masked rows; baseline would too.
        bad = (~torch.isfinite(cand_lse_f)) & torch.isfinite(base_lse_f)
        if bad.any():
            res.baseline_err = baseline_err
            res.note = "candidate lse has NaN/inf where baseline finite"
            return res

    # Shape checks against the baseline (the authoritative shapes).
    if cand_out_f.shape != base_out_f.shape:
        res.baseline_err = baseline_err
        res.note = f"out shape {tuple(cand_out_f.shape)} != baseline {tuple(base_out_f.shape)}"
        return res
    if cand_lse_f.shape != base_lse_f.shape:
        res.baseline_err = baseline_err
        res.note = f"lse shape {tuple(cand_lse_f.shape)} != baseline {tuple(base_lse_f.shape)}"
        return res

    # The fp32-reference 2x-error criterion (primary).
    out_err = _maxabs(cand_out_f - ref_f)
    err_ratio = out_err / (baseline_err + _EPS)

    # Mean-abs bias vs baseline (systematic-drift gate; calibrated in config.MEAN_ABS_TOL).
    bias = float((cand_out_f - base_out_f).abs().mean().item()) if cand_out_f.numel() else 0.0

    # lse comparison: mask out non-finite entries (fully-masked rows) on both sides.
    finite = torch.isfinite(cand_lse_f) & torch.isfinite(base_lse_f)
    if finite.any():
        lse_err = _maxabs((cand_lse_f - base_lse_f)[finite])
    else:
        lse_err = 0.0

    # Coarse allclose gate vs baseline output.
    atol, rtol = C.ALLCLOSE_TOL[cfg.dtype]
    allclose_ok = torch.allclose(cand_out_f, base_out_f, atol=atol, rtol=rtol)

    # lse tolerance gate.
    lse_atol, lse_rtol = C.LSE_TOL
    if finite.any():
        lse_ok = torch.allclose(cand_lse_f[finite], base_lse_f[finite],
                                atol=lse_atol, rtol=lse_rtol)
    else:
        lse_ok = True

    ratio_ok = err_ratio <= C.ERR_MULT
    bias_ok = bias <= C.MEAN_ABS_TOL.get(cfg.dtype, 1e-2)

    res.out_err = out_err
    res.baseline_err = baseline_err
    res.lse_err = lse_err
    res.err_ratio = err_ratio
    res.bias = bias
    res.correct = bool(ratio_ok and allclose_ok and lse_ok and bias_ok)

    if not res.correct:
        fails = []
        if not ratio_ok:
            fails.append(f"err_ratio={err_ratio:.3f}>{C.ERR_MULT}")
        if not allclose_ok:
            fails.append(f"allclose fail (atol={atol},rtol={rtol})")
        if not bias_ok:
            fails.append(f"bias={bias:.3e}>{C.MEAN_ABS_TOL.get(cfg.dtype)}")
        if not lse_ok:
            fails.append(f"lse_err={lse_err:.3e}>tol")
        res.note = "; ".join(fails)[:200]
    return res


@logged("correctness")
def check_correctness(namespace: str, cfgs, seed: int = 0,
                      gpu_id: int = 0) -> list[common.ConfigResult]:
    """Verify a candidate op (registered under ``namespace``) for each config.

    Returns one ``common.ConfigResult`` per config with the correctness fields
    populated (perf fields left at 0). Robust to candidate crashes, NaN/inf, and
    shape mismatches: those become ``correct=False`` rows with a terse ``note``.
    """
    torch.cuda.set_device(gpu_id)
    _ensure_baseline_loaded()

    # Make sure the candidate namespace is actually registered; if not, fail all
    # configs uniformly rather than throwing.
    cand_ok = hasattr(torch.ops, namespace) and hasattr(getattr(torch.ops, namespace), "fwd")

    gc = GoldenCache(device="cuda")
    results: list[common.ConfigResult] = []
    for cfg in cfgs:
        if not cand_ok:
            r = common.ConfigResult(config=cfg.name, correct=False,
                                    note=f"namespace torch.ops.{namespace}.fwd not registered")
            try:
                r.baseline_err = gc.golden_baseline_error(cfg, seed)
            except Exception:
                pass
            results.append(r)
            continue
        try:
            results.append(_check_one(cfg, namespace, seed, gc))
        except Exception as e:
            results.append(common.ConfigResult(
                config=cfg.name, correct=False,
                note=f"verify error: {type(e).__name__}: {e}"[:200]))
    return results
