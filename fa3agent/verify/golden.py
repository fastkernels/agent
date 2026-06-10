"""Golden cache for the FA3-distillation agent (verify component).

The "golden" for a config is the prebuilt baseline op's output: the baseline
out + softmax_lse, plus the baseline's own error against the pure-fp32 reference
(`shapes.ref_fp32`). These are deterministic for a given (cfg, seed), so we
compute them ONCE and cache the tensors to disk (torch.save) under
``CHECKPOINT_ROOT/_golden``. Repeated calls (every verify iteration of the
evolution loop) then just torch.load instead of re-running the baseline + the
slow fp32 oracle.

Context discipline: ``get`` returns a small dict of tensors/floats for ONE
config; ``golden_baseline_error`` returns a single float. Never a raw dump.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from .. import config as C
from .. import shapes


# Where cached golden tensors live (outside the repo, with the rest of the run).
GOLDEN_DIR = os.path.join(C.CHECKPOINT_ROOT, "_golden")

# Baseline op namespace (the prebuilt "oracle" shipped inside vLLM).
_BASELINE_NS = "_vllm_fa3_C"

# On-disk cache format version. Bump if the cached dict layout changes so stale
# files are ignored rather than silently mis-read.
_CACHE_VERSION = 1


def _ensure_baseline_loaded() -> None:
    """Import the vLLM extension so torch.ops._vllm_fa3_C.{fwd,...} is registered.

    Idempotent and cheap after the first call.
    """
    if hasattr(torch.ops, _BASELINE_NS) and hasattr(getattr(torch.ops, _BASELINE_NS), "fwd"):
        return
    # Importing the package module registers the TORCH_LIBRARY ops.
    import vllm.vllm_flash_attn._vllm_fa3_C  # noqa: F401


def _cache_key(cfg: C.Config, seed: int) -> str:
    return f"{cfg.name}__seed{seed}__v{_CACHE_VERSION}.pt"


def _cache_path(cfg: C.Config, seed: int) -> str:
    return os.path.join(GOLDEN_DIR, _cache_key(cfg, seed))


class GoldenCache:
    """Per-config cache of baseline out+lse and the baseline-vs-fp32-ref error.

    The cache is two-tier: an in-process dict (fast within a single run) backed
    by a torch.save'd file on disk (fast across processes / loop iterations).
    """

    def __init__(self, cache_dir: str = GOLDEN_DIR, device: str = "cuda"):
        self.cache_dir = cache_dir
        self.device = device
        self._mem: dict[str, dict] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    # -- internal helpers ---------------------------------------------------
    def _disk_path(self, cfg: C.Config, seed: int) -> str:
        return os.path.join(self.cache_dir, _cache_key(cfg, seed))

    def _compute(self, cfg: C.Config, seed: int) -> dict:
        """Run the baseline + fp32 reference once and assemble the golden dict."""
        _ensure_baseline_loaded()
        inp = shapes.make_inputs(cfg, seed, device=self.device)
        base_out, base_lse = shapes.call_fwd(_BASELINE_NS, inp)
        ref = shapes.ref_fp32(inp)  # fp32, shape (total_q, hq, hdv)

        base_out_f = base_out.detach().float()
        baseline_err = float((base_out_f - ref).abs().max().item())

        # Store on CPU so the cache is portable across GPUs / device indices.
        return {
            "version": _CACHE_VERSION,
            "config": cfg.name,
            "seed": int(seed),
            "out": base_out.detach().to("cpu"),
            "lse": base_lse.detach().to("cpu"),
            "ref": ref.detach().to("cpu"),
            "baseline_err": baseline_err,
        }

    # -- public API ---------------------------------------------------------
    def get(self, cfg: C.Config, seed: int = 0) -> dict:
        """Return the cached golden dict for (cfg, seed).

        Keys: ``out`` (baseline out tensor), ``lse`` (baseline softmax_lse),
        ``ref`` (fp32 reference out), ``baseline_err`` (float, baseline-vs-ref
        max abs error), plus ``config``/``seed``/``version`` metadata. Tensors
        are returned on ``self.device``.
        """
        key = _cache_key(cfg, seed)

        # 1) in-process memo
        cached = self._mem.get(key)
        if cached is not None:
            return cached

        # 2) on-disk cache
        path = self._disk_path(cfg, seed)
        if os.path.exists(path):
            try:
                d = torch.load(path, map_location=self.device)
                if isinstance(d, dict) and d.get("version") == _CACHE_VERSION:
                    self._mem[key] = d
                    return d
            except Exception:
                # Corrupt / incompatible cache file -> recompute and overwrite.
                pass

        # 3) compute fresh, then persist (best-effort) and memo on-device.
        d_cpu = self._compute(cfg, seed)
        tmp = path + f".tmp.{os.getpid()}"
        try:
            torch.save(d_cpu, tmp)
            os.replace(tmp, path)  # atomic publish
        except Exception:
            # Disk cache is an optimization; never let it break the call.
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

        d_dev = self._to_device(d_cpu)
        self._mem[key] = d_dev
        return d_dev

    def _to_device(self, d: dict) -> dict:
        out = dict(d)
        for k in ("out", "lse", "ref"):
            t = d.get(k)
            if isinstance(t, torch.Tensor):
                out[k] = t.to(self.device)
        return out

    def golden_baseline_error(self, cfg: C.Config, seed: int = 0) -> float:
        """max|baseline_out.float() - ref_fp32| for (cfg, seed)."""
        return float(self.get(cfg, seed)["baseline_err"])


# Module-level singleton + convenience wrapper so callers can use the function
# form from the interface contract without managing a cache object.
_DEFAULT_CACHE: Optional[GoldenCache] = None


def _default_cache() -> GoldenCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = GoldenCache()
    return _DEFAULT_CACHE


def golden_baseline_error(cfg: C.Config, seed: int = 0) -> float:
    """Module-level convenience: baseline-vs-fp32-ref max abs error for a config."""
    return _default_cache().golden_baseline_error(cfg, seed)
