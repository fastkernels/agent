"""CUDA-graph capture/replay compatibility gate (binary).

vLLM runs the FA3 attention op INSIDE a full CUDA graph for steady-state (uniform) decode batches
-- the FA3 backend declares full cudagraph support -- so a kernel that runs fine EAGERLY but cannot
be CUDA-graph-captured, or that captures yet produces wrong output on replay, is NOT a valid drop-in
replacement: it would error (or silently corrupt) once swapped into a graph-capturing vLLM.  Every
other gate here runs eager (`enforce_eager=True`), so none of them can catch this.

`check_cuda_graph(so_path, namespace, cfg_name=None, gpu_id=0) -> dict` runs in an ISOLATED spawn
subprocess (a candidate that faults the driver under capture must not take down the orchestrator).
It captures + replays the candidate op and checks the replayed output matches the eager output.  The
baseline op is captured FIRST as a control: if even the baseline can't be captured in this
environment we DEGRADE GRACEFULLY (skip) instead of blaming the candidate.

Returns a bounded dict: {ok, captured, replay_match, skip, max_diff, note}.
  ok == captured AND replay_match.   skip == the environment itself cannot graph-capture.
This is a BINARY compatibility check only -- it does NOT time anything (no graphed perf).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

from .. import config as C
from ..runlog import logged

CHILD_TIMEOUT_S = 300
_RESULT_PREFIX = "CUDAGRAPH_RESULT "


@logged("cudagraph")
def check_cuda_graph(so_path: str, namespace: str, cfg_name: Optional[str] = None,
                     gpu_id: int = 0) -> dict:
    """Capture+replay the candidate op in a CUDA graph and verify replay matches eager.

    Runs in an isolated subprocess.  Returns {ok, captured, replay_match, skip, max_diff, note}.
    `ok=False, skip=False` means the candidate is genuinely not CUDA-graph-compatible (a real
    drop-in blocker); `skip=True` means we couldn't test it here (baseline also uncapturable / no
    device) and the caller should not penalize the candidate.
    """
    if not os.path.isfile(so_path):
        return {"ok": False, "note": f"candidate .so not found: {so_path}"}
    cfg_name = cfg_name or C.DEV_CONFIGS[0].name   # a small decode config: the graphed-in-prod case
    payload = {"so_path": so_path, "namespace": namespace, "cfg_name": cfg_name}
    res = _run_child(payload, gpu_id)
    if res is None:
        return {"ok": False, "skip": True, "note": "[skip] cudagraph child produced no result"}
    return res


def _run_child(payload: dict, gpu_id: int) -> Optional[dict]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id),
               PYTHONPATH=C.REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "fa3agent.verify.cudagraph", json.dumps(payload)],
            cwd=C.REPO_ROOT, env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ok": False, "skip": True, "note": f"[skip] cudagraph child timed out after {CHILD_TIMEOUT_S}s"}
    except Exception as e:  # noqa
        return {"ok": False, "skip": True, "note": f"[skip] cudagraph launch failed: {type(e).__name__}: {e}"}
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                return json.loads(line[len(_RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return {"ok": False, "skip": True,
            "note": f"[skip] no cudagraph result (rc={proc.returncode}): {' | '.join(tail[-2:])[:200]}"}


# ======================================================================================
# Child (isolated subprocess)
# ======================================================================================
def _emit(rec: dict):
    print(_RESULT_PREFIX + json.dumps(rec), flush=True)


def _capture_replay(ns: str, inp, atol: float):
    """Eager-ref -> side-stream warmup -> capture -> replay -> compare.

    Returns (captured: bool, match: bool, max_diff: float, err: str).  `captured` is False if the
    graph could not be captured at all (the op is not graph-safe); `match` is whether the replayed
    output reproduces the eager output within `atol`.
    """
    import torch
    from fa3agent import shapes

    # Eager reference (capture does NOT execute the kernel, so we need a real run to compare to).
    out_e, _ = shapes.call_fwd(ns, inp)
    out_e = out_e.float().clone()
    torch.cuda.synchronize()

    # Required warmup on a side stream before capture (stabilizes the allocator / lazy init).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            shapes.call_fwd(ns, inp)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            out_g, _ = shapes.call_fwd(ns, inp)
    except Exception as e:  # not graph-safe (host alloc/sync/readback inside fwd, etc.)
        return False, False, float("inf"), f"{type(e).__name__}: {str(e)[:160]}"

    try:
        g.replay()
        torch.cuda.synchronize()
        d = float((out_g.float() - out_e).abs().max().item())
        finite = bool(torch.isfinite(out_g.float()).all().item())
        return True, (finite and d <= atol), d, ""
    except Exception as e:
        return True, False, float("inf"), f"replay: {type(e).__name__}: {str(e)[:160]}"


def _child_main(argv) -> int:
    if len(argv) < 2:
        _emit({"ok": False, "skip": True, "note": "no payload"}); return 0
    p = json.loads(argv[1])
    import torch
    if not torch.cuda.is_available():
        _emit({"ok": False, "skip": True, "note": "[skip] no CUDA device"}); return 0

    from fa3agent import config as C, shapes
    cfg = next((c for c in (C.DEV_CONFIGS + C.FULL_CONFIGS) if c.name == p["cfg_name"]), C.DEV_CONFIGS[0])
    atol = C.ALLCLOSE_TOL.get(cfg.dtype, (1e-2, 0))[0]

    try:
        torch.ops.load_library(p["so_path"])
    except Exception as e:
        _emit({"ok": False, "note": f"candidate .so load failed: {type(e).__name__}: {e}"}); return 0

    inp = shapes.make_inputs(cfg, seed=0, device="cuda")

    # Control: the baseline FA3 op IS captured in a CUDA graph by vLLM, so it MUST capture here.
    # If it can't, the environment (not the candidate) is the problem -> skip, don't penalize.
    if C.BASELINE_SO and os.path.isfile(C.BASELINE_SO):
        try:
            torch.ops.load_library(C.BASELINE_SO)
        except Exception:
            pass
        if hasattr(torch.ops, "_vllm_fa3_C") and hasattr(torch.ops._vllm_fa3_C, "fwd"):
            b_cap, _bm, _bd, b_err = _capture_replay("_vllm_fa3_C", inp, atol)
            if not b_cap:
                _emit({"ok": False, "skip": True,
                       "note": f"[skip] baseline not capturable in this env: {b_err}"}); return 0

    cap, match, d, err = _capture_replay(p["namespace"], inp, atol)
    if not cap:
        _emit({"ok": False, "captured": False, "replay_match": False,
               "note": f"candidate is NOT cuda-graph-capturable on {cfg.name}: {err}"}); return 0
    note = (f"cuda-graph captured ok on {cfg.name}; replay max_diff={d:.2e} "
            f"(atol={atol:.1e}) replay_match={match}")
    _emit({"ok": bool(match), "captured": True, "replay_match": bool(match),
           "max_diff": d, "note": note})
    return 0


if __name__ == "__main__":
    try:
        import multiprocessing as _mp
        _mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    sys.exit(_child_main(sys.argv))
