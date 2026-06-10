"""NCU profiling -> distilled `ProfileFeedback` (component 2.1).

`profile(so_path, namespace, cfg, gpu_id, run_dir) -> common.ProfileFeedback`

Pipeline:
  1. Discover the kernel name from the .so via `cuobjdump --dump-sass` (FA3 kernels are
     templated; we never hardcode a name) and derive a safe `-k` regex.
  2. Write a tiny standalone torch driver that load_library(so), warms once, and does ONE
     `shapes.call_fwd` launch for `cfg`.
  3. Run that driver under ncu as a SUBPROCESS (CUDA_VISIBLE_DEVICES=gpu_id), capturing one
     kernel instance into `run_dir/rep.ncu-rep`.  `--set full` first; if it times out, retry
     with the explicit `--metrics` subset (fewer passes).
  4. Parse the .ncu-rep via the `ncu_report` module (perf.ncu_utils), classify ONE dominant
     bottleneck with the decision tree from .research/ncu-skill.md, and emit a terse,
     evidence-cited, BOUNDED `ProfileFeedback` (never the 2000-metric dump).
  5. Degrade gracefully on ERR_NVGPUCTRPERM / timeout (clear note, partial feedback).

CRITICAL PROCESS SEPARATION (verified this session): `import ncu_report` and `import torch`
in the same process SEGFAULT.  So:
  * the driver subprocess imports torch + shapes, NEVER ncu_report / this module's parser;
  * THIS module (parent) imports ncu_utils (-> ncu_report) and must NEVER import torch.
We therefore do not import torch or shapes at module scope.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .. import config as C
from .. import common
from . import ncu_utils as U

# ncu collection wall-clock budgets (seconds).  --set full = ~38 replay passes on the seed.
_FULL_TIMEOUT = 480
_SUBSET_TIMEOUT = 240
_NCU_BIN = os.environ.get("NCU_BIN", "ncu")

# How many words each terse field is allowed (CUDAForge contract).
_BOTTLENECK_W = 30
_OPT_W = 35
_PLAN_W = 35


def _clip_words(s: str, n: int) -> str:
    w = s.split()
    return s.strip() if len(w) <= n else " ".join(w[:n]).rstrip(",.;:") + " ..."


def _fmt(v, nd=1) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if abs(f) >= 1e6:
        return f"{f:.2e}"
    return f"{f:.{nd}f}"


# --- 1. kernel name discovery --------------------------------------------------
def discover_kernel_regex(so_path: str, dtype: Optional[str] = None) -> Optional[str]:
    """Return a mangled-name regex selecting the candidate's attention kernel(s).

    Uses `cuobjdump --dump-sass` and keeps function names that look like attention
    kernels (flash / attn / fwd / mha).  Falls back to a concrete escaped name.
    Returns None if nothing plausible is found (caller then profiles unfiltered).
    """
    try:
        out = subprocess.run(["cuobjdump", "--dump-sass", so_path],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    names = re.findall(r"Function\s*:\s*(\S+)", out)
    if not names:
        return None
    pat = re.compile(r"flash|attn|fwd|mha|fmha|attention", re.I)
    cands = [n for n in names if pat.search(n)] or names
    # Prefer a stable common token rather than a full mangled name (templated kernels).
    for tok in ("attn_kernel", "flash_fwd", "flash", "fmha", "attn", "mha", "fwd"):
        if any(tok in n for n in cands):
            return f"regex:{tok}"
    return "regex:" + re.escape(cands[0])


# --- 2. driver script ----------------------------------------------------------
_DRIVER_TMPL = '''\
import sys
sys.path.insert(0, {repo!r})
import torch
torch.ops.load_library({so!r})
from fa3agent import config as C, shapes
cfg = next(c for c in (C.DEV_CONFIGS + C.FULL_CONFIGS) if c.name == {cfg_name!r})
inp = shapes.make_inputs(cfg, seed=0, device="cuda")
# warm once (steady clocks / JIT / allocator), then ONE profiled launch
shapes.call_fwd({ns!r}, inp)
torch.cuda.synchronize()
shapes.call_fwd({ns!r}, inp)
torch.cuda.synchronize()
'''


def _write_driver(run_dir: str, so_path: str, namespace: str, cfg) -> str:
    src = _DRIVER_TMPL.format(repo=C.REPO_ROOT, so=so_path, ns=namespace,
                              cfg_name=cfg.name)
    p = os.path.join(run_dir, "ncu_driver.py")
    Path(p).write_text(src)
    return p


# --- 3. ncu invocation ---------------------------------------------------------
def _ncu_env(gpu_id: int, run_dir: str) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = C.REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # writable HOME silences "Could not deploy stock section files"; per-GPU TMPDIR
    # avoids ncu lock-file contention across the 4-GPU scheduler.
    home = os.path.join(run_dir, "ncu_home")
    tmp = os.path.join(run_dir, f"ncu_tmp_g{gpu_id}")
    os.makedirs(home, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    env["HOME"] = home
    env["TMPDIR"] = tmp
    return env


def _subset_collect_names() -> list[str]:
    """Concrete metric names to collect in the fast fallback pass.

    config.NCU_METRIC_SUBSET resolved through Hopper aliases, plus the SOL /
    stall-ratio / SchedulerStats metrics the classifier relies on.
    """
    names: list[str] = []
    for n in C.NCU_METRIC_SUBSET:
        names.extend(U.HOPPER_ALIASES.get(n, [n]))
    for reason in U.STALL_REASONS:
        names.append(f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio")
    names += [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "gpu__time_duration.sum",
    ]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _run_ncu(driver: str, rep_base: str, kregex: Optional[str], env: dict,
             mode: str, timeout: int):
    """Run ncu once. mode in {'full','subset'}. Returns (returncode, combined_output)."""
    cmd = [_NCU_BIN, "-f", "-o", rep_base, "--target-processes", "all",
           "--replay-mode", "kernel", "-c", "1", "-s", "1"]
    if kregex:
        cmd += ["-k", kregex, "--kernel-name-base", "mangled"]
    if mode == "full":
        cmd += ["--set", "full", "--section", "SchedulerStats",
                "--section", "WarpStateStats"]
    else:
        cmd += ["--metrics", ",".join(_subset_collect_names())]
    cmd += ["python", driver]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        so = e.stdout or ""
        se = e.stderr or ""
        if isinstance(so, bytes):
            so = so.decode("utf-8", "replace")
        if isinstance(se, bytes):
            se = se.decode("utf-8", "replace")
        return 124, f"TIMEOUT after {timeout}s\n" + so + se
    except (FileNotFoundError, OSError) as e:
        return 127, f"ncu binary not launchable ({_NCU_BIN}): {e}"


def _classify_perm_error(text: str) -> Optional[str]:
    if "ERR_NVGPUCTRPERM" in text or "insufficient permissions" in text.lower():
        return ("ncu lacks GPU performance-counter permission (ERR_NVGPUCTRPERM). "
                "Run as root or set NVreg_RestrictProfilingToAdminUsers=0.")
    if "TIMEOUT" in text:
        return "ncu collection timed out"
    if "does not exist or is not an executable" in text or "not launchable" in text:
        return "ncu could not launch (binary/driver path error)"
    return None


# --- 4. bottleneck classification (decision tree from .research/ncu-skill.md) ---
def _classify(action) -> tuple[str, str, str, str, list[str]]:
    """Return (label, bottleneck, optimization, plan, evidence_lines).

    Thresholds + cross-dimension logic follow .research/ncu-skill.md section 3:
      DRAM%peak high(>=60) -> BW-bound;
      DRAM%<10 & long_scoreboard>40% of stalls -> latency-bound;
      tensor pipe ~0 & fp32/fma high -> off-tensor-core;
      tensor>50% -> good;
      register spill (local ld/st > 0) -> spill.
    """
    g = lambda *names: U.metric_or_none(action, *names)  # noqa: E731

    dram_pct = g("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                 "dram__throughput.avg.pct_of_peak_sustained_elapsed") or 0.0
    sm_sol = g("sm__throughput.avg.pct_of_peak_sustained_elapsed") or 0.0
    tensor = g("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
               "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
               "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed") or 0.0
    fma = g("sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active") or 0.0
    occ = g("sm__warps_active.avg.pct_of_peak_sustained_active") or 0.0
    waves = g("launch__waves_per_multiprocessor")
    spill_ld = g("smsp__sass_inst_executed_op_local_ld.sum") or 0.0
    spill_st = g("smsp__sass_inst_executed_op_local_st.sum") or 0.0

    shares = U.stall_shares(action)
    lsb = shares.get("long_scoreboard", 0.0)
    ssb = shares.get("short_scoreboard", 0.0)
    barrier = shares.get("barrier", 0.0) + shares.get("membar", 0.0)
    top_stall = max(shares.items(), key=lambda kv: kv[1], default=("none", 0.0))

    ev = [
        f"SM SOL {_fmt(sm_sol)}% | DRAM %peak {_fmt(dram_pct)}%",
        f"tensor-pipe active {_fmt(tensor, 2)}% | FMA pipe {_fmt(fma, 2)}%",
        f"achieved occ {_fmt(occ)}%"
        + (f" | waves/SM {_fmt(waves, 3)}" if waves is not None else ""),
        f"top stall: {top_stall[0]} {_fmt(top_stall[1])}% "
        f"(long_sb {_fmt(lsb)}%, short_sb {_fmt(ssb)}%, barrier {_fmt(barrier)}%)",
    ]
    if spill_ld or spill_st:
        ev.append(f"register spill: local ld {_fmt(spill_ld, 0)} / st {_fmt(spill_st, 0)}")

    # --- decision tree (ordered: spill first = always-bad, then SOL classes) ---
    if (spill_ld or spill_st) and (lsb > 30 or sm_sol < 30):
        label = "register-spill"
        bn = (f"Register spill: local ld {_fmt(spill_ld, 0)}/st {_fmt(spill_st, 0)} >0, "
              f"long-scoreboard {_fmt(lsb)}% of stalls, SM SOL {_fmt(sm_sol)}%.")
        opt = ("Cut register pressure so spills vanish: add __launch_bounds__, shrink live "
               "register tiles, or reduce per-thread accumulator footprint.")
        plan = ("Set __launch_bounds__ and re-tile so the inner accumulator fits registers; "
                "recheck local ld/st == 0.")
    elif dram_pct >= 60:
        label = "dram-bandwidth-bound"
        bn = (f"DRAM-bandwidth-bound at {_fmt(dram_pct)}% peak HBM (SM SOL {_fmt(sm_sol)}%, "
              f"L2 hit {_fmt(g('lts__t_sector_hit_rate.pct'))}%).")
        opt = ("Cut HBM bytes: keep KV tiles resident in smem/registers across score+PV "
               "passes, use TMA/cp.async bulk loads, avoid re-reading K/V.")
        plan = ("Fuse KV loads into one streamed pass cached in shared memory; verify "
                "dram__bytes_read.sum drops and DRAM %peak falls.")
    elif dram_pct < 10 and lsb > 40:
        label = "latency-bound"
        bn = (f"Latency-bound on L1/global: long-scoreboard {_fmt(lsb)}% of stalls, "
              f"DRAM only {_fmt(dram_pct)}% peak, achieved occ {_fmt(occ)}%.")
        opt = ("Hide load latency: cp.async/TMA double-buffering, unroll loads for ILP, "
               "raise occupancy, ensure coalesced 128B accesses.")
        plan = ("Add a multi-stage async copy pipeline for K/V and increase warps/CTA; "
                "recheck long-scoreboard share and occupancy.")
    elif tensor < 1.0 and fma > 20:
        label = "off-tensor-core"
        bn = (f"Compute off tensor cores: tensor-pipe {_fmt(tensor, 2)}% vs FMA pipe "
              f"{_fmt(fma)}%, SM SOL {_fmt(sm_sol)}%.")
        opt = ("Move the QK^T and PV matmuls onto Hopper tensor cores via wgmma instead of "
               "FMA/CUDA-core loops.")
        plan = ("Rewrite the two inner GEMMs as wgmma.mma_async with smem operand tiles; "
                "recheck tensor-pipe %peak rises above 50%.")
    elif tensor > 50:
        label = "tensor-core-good"
        bn = (f"Healthy tensor-core use: tensor-pipe {_fmt(tensor)}% peak, SM SOL "
              f"{_fmt(sm_sol)}%, DRAM {_fmt(dram_pct)}% -- near compute-bound.")
        opt = ("Marginal gains only: trim the largest residual stall "
               f"({top_stall[0]} {_fmt(top_stall[1])}%) and tail effects.")
        plan = ("Reduce the dominant residual stall / tail; pursue occupancy or "
                "split-K only if it lifts the worst-case config.")
    elif barrier > 20:
        label = "sync-bound"
        bn = (f"Sync overhead: barrier/membar {_fmt(barrier)}% of stalls, SM SOL "
              f"{_fmt(sm_sol)}%, occ {_fmt(occ)}%.")
        opt = ("Reduce __syncthreads/named-barrier waits: overlap producer/consumer warp "
               "groups, fewer barriers, async pipeline.")
        plan = ("Restructure producer/consumer barriers so MMA and copy warps overlap; "
                "recheck barrier stall share.")
    elif (waves is not None and waves < 1) or occ < 30:
        label = "occupancy-bound"
        bn = (f"Under-occupied/small grid: waves/SM {_fmt(waves, 3)}, achieved occ "
              f"{_fmt(occ)}%, SM SOL {_fmt(sm_sol)}% (top stall {top_stall[0]} "
              f"{_fmt(top_stall[1])}%).")
        opt = ("Fill the GPU: split KV across CTAs (split-K) for decode, raise occupancy by "
               "cutting registers/smem, or persistent scheduling.")
        plan = ("Add split-K over the KV dimension so the grid covers all SMs; recheck "
                "waves/SM toward >=1 and occupancy.")
    else:
        label = "mixed/inconclusive"
        bn = (f"No single dominant bottleneck: SM SOL {_fmt(sm_sol)}%, DRAM {_fmt(dram_pct)}%, "
              f"tensor {_fmt(tensor, 2)}%, top stall {top_stall[0]} {_fmt(top_stall[1])}%.")
        opt = ("Profile a larger/representative shape; first attack the top stall reason "
               f"({top_stall[0]}) and lift occupancy.")
        plan = ("Re-profile on a production shape; address the dominant stall, then "
                "re-evaluate the bottleneck class.")
    return label, bn, opt, plan, ev


def _ranked(action, ev: list[str]) -> list[str]:
    """<=5 evidence-cited priorities: metric-derived findings + top NCU rules."""
    out = list(ev[:4])
    for est, name, msg in U.rule_speedups(action):
        if len(out) >= 5:
            break
        first = msg.split(".")[0].strip()[:90] if msg else ""
        tag = f"~{_fmt(est, 0)}% est" if est else "rule"
        out.append(f"NCU rule [{name}] ({tag}): {first}")
    return out[:5]


# --- top-level entry -----------------------------------------------------------
def profile(so_path: str, namespace: str, cfg, gpu_id: int,
            run_dir: str) -> common.ProfileFeedback:
    os.makedirs(run_dir, exist_ok=True)
    fb = common.ProfileFeedback(config=getattr(cfg, "name", str(cfg)))

    if not U.ncu_report_available():
        fb.bottleneck = "partial: ncu_report module not importable (cannot parse reports)."
        fb.plan = "Add an ncu extras/python dir to config.NCU_PYTHON_CANDIDATES."
        return fb

    driver = _write_driver(run_dir, so_path, namespace, cfg)
    kregex = discover_kernel_regex(so_path, getattr(cfg, "dtype", None))
    rep_base = os.path.join(run_dir, "rep")
    rep_path = rep_base + ".ncu-rep"
    env = _ncu_env(gpu_id, run_dir)

    rc, log = _run_ncu(driver, rep_base, kregex, env, "full", _FULL_TIMEOUT)
    used = "full"
    if (rc != 0 or not os.path.exists(rep_path)) and "ERR_NVGPUCTRPERM" not in log:
        # full set too slow / flaky -> fast subset retry
        rc, log2 = _run_ncu(driver, rep_base, kregex, env, "subset", _SUBSET_TIMEOUT)
        log = log[-1500:] + "\n--- subset retry ---\n" + log2
        used = "subset"

    perm = _classify_perm_error(log)
    if not os.path.exists(rep_path):
        note = perm or "ncu produced no report"
        fb.bottleneck = _clip_words(f"partial: {note}.", _BOTTLENECK_W)
        fb.optimization = "Cannot profile; rely on SASS/measurement components this round."
        fb.plan = "Fix profiling permission/timeout, then re-run NCU on a single launch."
        fb.ranked = [f"ncu rc={rc} ({used} pass); see log tail", (log[-300:] or "")]
        return fb

    # --- parse + classify ---
    try:
        report, action = U.load_report(rep_path)
    except Exception as e:
        fb.bottleneck = _clip_words(f"partial: report parse failed ({e}).", _BOTTLENECK_W)
        fb.plan = "Re-collect the .ncu-rep; verify ncu_report version matches ncu."
        fb.report_path = rep_path
        return fb

    try:
        kname = action.name()
    except Exception:
        kname = "?"
    try:
        U.dump_all_metrics(action, os.path.join(run_dir, "metrics_all.json"))
    except Exception:
        pass

    label, bn, opt, plan, ev = _classify(action)
    key_metrics = dict(U.resolve_subset(action))
    key_metrics["_bottleneck_label"] = label
    key_metrics["_kernel"] = kname
    key_metrics["_ncu_set"] = used
    dur = U.metric_or_none(action, "gpu__time_duration.sum")
    if dur is not None:
        key_metrics["gpu__time_duration.sum_ns"] = dur

    fb.bottleneck = _clip_words(bn, _BOTTLENECK_W)
    fb.optimization = _clip_words(opt, _OPT_W)
    fb.plan = _clip_words(plan, _PLAN_W)
    fb.ranked = _ranked(action, ev)
    fb.key_metrics = key_metrics
    fb.report_path = rep_path
    if perm:  # collected but with warnings
        fb.ranked.append(f"note: {perm}")
    return fb
