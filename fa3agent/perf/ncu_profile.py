"""NCU profiling -> distilled `ProfileFeedback` (component 2.1).

`profile(so_path, namespace, cfg, gpu_id, run_dir) -> common.ProfileFeedback`

Pipeline:
  1. TIMING PASS: a cheap one-metric ncu run captures the launches and ranks the candidate's
     subkernels by duration, so we deep-profile (and `-k`-target) the DOMINANT kernel -- a split-KV
     candidate runs a heavy partial kernel + a tiny combine kernel -- and report the per-kernel time
     split.  Falls back to a name-based `-k` regex from `cuobjdump --dump-sass` if timing is unavailable.
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
from ..runlog import logged

# ncu collection wall-clock budgets (seconds).  --set full = ~38 replay passes on the seed.
_FULL_TIMEOUT = 480
_SUBSET_TIMEOUT = 240
_NCU_BIN = os.environ.get("NCU_BIN", "ncu")

# Phase-A timing pass: capture this many launches with one cheap metric (covers the warmup +
# profiled iterations of a few-kernel forward) so we can rank the candidate's subkernels by
# duration and deep-profile the DOMINANT one (not, e.g., a tiny split-KV combine kernel).
_TIMING_LAUNCHES = 16
_TIMING_TIMEOUT = 150

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
# torch/aten/library helper kernels that may be linked into the candidate .so but are NOT the
# attention kernel (and do NOT launch during the profiling driver) -- must be excluded, else the
# -k regex matches a kernel that never runs and ncu writes no report (observed on wgmma rewrites).
_ATEN_PAT = re.compile(
    r"at6native|_GLOBAL__N__|elementwise|Philox|arange|RangeFactories|thrust|cub|"
    r"vectorized|CatArray|reduce_kernel|fill_|index_elementwise|c10",
    re.I)

# Tokens by which a candidate's own attention kernels self-identify (seed: attn_kernel; split-KV:
# attn_partial_kernel / attn_combine_kernel).  Used to pick candidate kernels out of a launch trace.
_ATTN_PAT = re.compile(r"flash|attn|fwd|mha|fmha|attention", re.I)
# Helper/library kernels to drop from the timing trace.  Broader than _ATEN_PAT, which targets
# cuobjdump's MANGLED `Function:` names; these match the DEMANGLED names ncu_report returns.
_HELPER_PAT = re.compile(
    r"at::native|\bat::|_GLOBAL__N__|elementwise|Philox|RangeFactories|thrust::|cub::|"
    r"vectorized_elementwise|CatArray|reduce_kernel|fill_|index_elementwise|c10::|"
    r"DeviceReduce|distribution_|direct_copy",
    re.I)


def discover_kernel_regex(so_path: str, dtype: Optional[str] = None) -> Optional[str]:
    """Return a mangled-name regex selecting the candidate's OWN attention kernel.

    Uses `cuobjdump --dump-sass`, EXCLUDES torch/aten helper kernels, then prefers an
    attention-looking name (flash/attn/fwd/mha).  Falls back to the first non-aten kernel's
    escaped name.  Returns None only if nothing plausible is found (caller profiles unfiltered).
    """
    try:
        out = subprocess.run(["cuobjdump", "--dump-sass", so_path],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    names = re.findall(r"Function\s*:\s*(\S+)", out)
    if not names:
        return None
    own = [n for n in names if not _ATEN_PAT.search(n)] or names   # drop torch/aten helpers
    cands = [n for n in own if _ATTN_PAT.search(n)] or own
    # Return a BROAD alternation matching ANY of the candidate's attention-kernel variants -- a
    # candidate may ship several (e.g. scalar attn_kernel AND wgmma attn_mma_kernel); a single
    # narrow token ('attn_kernel') would miss 'attn_mma_kernel' and ncu would match a kernel that
    # never launches for the profiled config -> no report. With -c 1 -s 1 ncu profiles the first
    # matching launch (the combine kernel is excluded, so the main attention kernel is profiled).
    present = [t for t in ("attn", "flash", "fmha", "mha", "attention", "fwd") if any(t in n for n in cands)]
    if present:
        return "regex:" + "|".join(present)
    return "regex:" + re.escape(cands[0])


def _short_kname(name: str) -> str:
    """Trim template args / parameter lists from a kernel name for compact display."""
    return (name or "?").split("<")[0].split("(")[0].strip() or (name or "?")


def _kregex_for(dominant: str, names: list[str]) -> Optional[str]:
    """Build a `-k` regex selecting ONLY the dominant kernel among `names` (the launched set).

    Picks an identifier token in the dominant kernel's name that no other launched kernel shares
    (e.g. 'partial' vs an 'attn_combine_kernel'); that token also occurs in the mangled symbol, so
    it works with --kernel-name-base mangled.  Prefers an attention-identity token so the filter
    stays off torch helper kernels; falls back to the longest unique (then longest) identifier.
    """
    toks = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", dominant or "")))
    if not toks:
        return None
    others = [n for n in names if n != dominant]
    uniq = [t for t in toks if not any(t in o for o in others)]
    attn_uniq = [t for t in uniq if _ATTN_PAT.search(t)]
    pick = (max(attn_uniq, key=len) if attn_uniq
            else max(uniq, key=len) if uniq
            else max(toks, key=len))
    return "regex:" + re.escape(pick)


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


def _time_kernels(driver: str, rep_base: str, env: dict) -> list[tuple[str, float]]:
    """Phase A: a cheap single-metric ncu pass capturing the first _TIMING_LAUNCHES launches, used
    to rank the candidate's OWN kernels by duration (dominant first).

    Returns [(kernel_name, time_ns)] or [] if profiling is unavailable here (caller then falls back
    to the name-only `-k` heuristic).  Helper/library kernels are dropped; if any kernels carry an
    attention-identity token we keep only those, else all non-helper kernels.
    """
    rep = rep_base + ".ncu-rep"
    cmd = [_NCU_BIN, "-f", "-o", rep_base, "--target-processes", "all", "--replay-mode", "kernel",
           "-c", str(_TIMING_LAUNCHES), "--metrics", "gpu__time_duration.sum", "python", driver]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMING_TIMEOUT, env=env)
    except Exception:  # noqa - timing pass is best-effort; caller degrades gracefully
        return []
    if not os.path.exists(rep):
        return []
    try:
        _report, actions = U.load_all_actions(rep)
    except Exception:
        return []
    times: dict[str, float] = {}
    for a in actions:
        try:
            name = a.name()
        except Exception:
            continue
        if not name or _HELPER_PAT.search(name):
            continue
        d = U.metric_or_none(a, "gpu__time_duration.sum")
        if d is None:
            continue
        try:
            times[name] = float(d)   # last (most-warmed) occurrence per kernel wins
        except (TypeError, ValueError):
            continue
    if not times:
        return []
    # Prefer the candidate's attention kernels; if none self-identify, keep all non-helper kernels.
    attn = {n: t for n, t in times.items() if _ATTN_PAT.search(n)}
    chosen = attn or times
    return sorted(chosen.items(), key=lambda kv: -kv[1])


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
@logged("ncu_profile")
def profile(so_path: str, namespace: str, cfg, gpu_id: int,
            run_dir: str) -> common.ProfileFeedback:
    os.makedirs(run_dir, exist_ok=True)
    fb = common.ProfileFeedback(config=getattr(cfg, "name", str(cfg)))

    if not U.ncu_report_available():
        fb.bottleneck = "partial: ncu_report module not importable (cannot parse reports)."
        fb.plan = "Add an ncu extras/python dir to config.NCU_PYTHON_CANDIDATES."
        return fb

    driver = _write_driver(run_dir, so_path, namespace, cfg)
    rep_base = os.path.join(run_dir, "rep")
    rep_path = rep_base + ".ncu-rep"
    env = _ncu_env(gpu_id, run_dir)

    # Phase A: cheap timing pass -> rank the candidate's subkernels by duration and target the
    # DOMINANT one.  A split-KV candidate launches a heavy partial-attention kernel + a tiny
    # combine kernel; the old broad `-k` regex matched both and `-c 1` often captured the combine.
    ranked = _time_kernels(driver, os.path.join(run_dir, "rep_timing"), env)
    dtype = getattr(cfg, "dtype", None)
    if ranked:
        kregex = _kregex_for(ranked[0][0], [n for n, _ in ranked]) or discover_kernel_regex(so_path, dtype)
    else:
        kregex = discover_kernel_regex(so_path, dtype)

    # Phase B: deep-profile the targeted (dominant) kernel.
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

    # Per-subkernel time split from the timing pass: tells the brain WHERE time goes across a
    # multi-kernel forward and which kernel these metrics describe.
    kernels_line = None
    if ranked:
        tot = sum(d for _, d in ranked) or 1.0
        key_metrics["_kernels"] = [
            {"name": _short_kname(n), "time_ns": round(d, 1), "pct": round(100.0 * d / tot, 1)}
            for n, d in ranked
        ]
        if len(ranked) > 1:
            summ = " / ".join(f"{_short_kname(n)} {100.0 * d / tot:.0f}%" for n, d in ranked[:4])
            kernels_line = f"kernels by time: {summ} (profiled: {_short_kname(kname)})"

    fb.bottleneck = _clip_words(bn, _BOTTLENECK_W)
    fb.optimization = _clip_words(opt, _OPT_W)
    fb.plan = _clip_words(plan, _PLAN_W)
    base_ranked = _ranked(action, ev)
    fb.ranked = ([kernels_line] + base_ranked)[:5] if kernels_line else base_ranked
    fb.key_metrics = key_metrics
    fb.report_path = rep_path
    if perm:  # collected but with warnings
        fb.ranked.append(f"note: {perm}")
    return fb
