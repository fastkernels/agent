"""SASS analysis (sec 2.3).

Disassemble a built `.so`/cubin with `cuobjdump` and produce a DISTILLED, BOUNDED
instruction-mix histogram per kernel plus register / shared-mem / spill usage, and
(optionally) a diff vs a baseline `.so` or an oracle cubin.

The whole point of this component is context discipline: cuobjdump on a real FA3
build emits hundreds of MB of SASS across >1000 template instantiations and takes
minutes. We never return that. Instead we:

  * use the near-instant `--dump-resource-usage` to enumerate kernels + regs/smem/spills,
  * dump SASS for only a BOUNDED set of relevant kernels (`--function <name>` per kernel),
  * count only the ~16 FA3-on-Hopper instruction classes that matter,
  * return a small dict of histograms + a handful of human-readable diff lines.

Public API (interfaces.py):
    analyze(so_path, namespace, baseline_so=None, oracle_cubin=None) -> dict
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections import OrderedDict
from typing import Optional

from ..runlog import logged

# --------------------------------------------------------------------------------------
# Instruction taxonomy: FA3-on-Hopper classes that drive a perf verdict.  Each class maps
# to a list of SASS-opcode PREFIXES (matched against the leading mnemonic token, modifiers
# stripped).  Order matters: the first matching class wins, so put specific before generic
# (e.g. LDSM/STSM before LDS/STS, LDGSTS before LDG/STG, *GMMA before HMMA).
# --------------------------------------------------------------------------------------
INSTR_CLASSES: "OrderedDict[str, list[str]]" = OrderedDict([
    # warpgroup MMA (the thing that makes a Hopper kernel fast)
    ("HGMMA", ["HGMMA"]),                 # bf16/fp16 wgmma
    ("QGMMA", ["QGMMA"]),                 # fp8 wgmma
    ("IGMMA", ["IGMMA", "BGMMA"]),        # int8 / bit wgmma
    # TMA / bulk async copy
    ("UTMALDG", ["UTMALDG", "UTMASTG", "UTMAPF", "UTMAREDG", "UTMACCTL", "UTMACMDFLUSH"]),
    ("UBLKCP", ["UBLKCP", "UBLKPF", "UBLKRED"]),
    # Ampere-style async copy (pre-TMA pipelining idiom)
    ("LDGSTS", ["LDGSTS", "LDGDEPBAR"]),
    # shared-matrix load/store (feeds wgmma / epilogue)
    ("LDSM_STSM", ["LDSM", "STSM"]),
    # barriers / synchronization (mbarrier sync-unit + CTA barrier + warpgroup sync)
    ("BAR_SYNC", ["BAR", "SYNCS", "WARPGROUP", "DEPBAR", "ELECT", "MEMBAR", "FENCE",
                  "UCGABAR", "WARPSYNC", "BSYNC", "BSSY", "NANOSLEEP", "ACQBULK"]),
    # plain (non-warpgroup) tensor core
    ("HMMA", ["HMMA"]),
    ("IMMA", ["IMMA", "BMMA"]),
    # FP32 / FP16 scalar math (softmax, rescale, the "FFMA-bound" smell of a naive kernel)
    ("FFMA", ["FFMA", "FADD", "FMUL", "FSEL", "FSET"]),
    ("HFMA2", ["HFMA2", "HADD2", "HMUL2", "HSET2", "HMNMX2"]),
    ("MUFU", ["MUFU"]),                   # exp2/rcp/rsqrt -> softmax transcendentals
    # global load/store
    ("LDG", ["LDG", "LDGMC"]),
    ("STG", ["STG"]),
    # shared load/store (non-matrix)
    ("LDS", ["LDS"]),
    ("STS", ["STS", "STAS"]),
    # register spill -- MUST be zero on a good kernel
    ("LDL_SPILL", ["LDL"]),
    ("STL_SPILL", ["STL"]),
])

# The handful we surface in the top-line summary / diff (the load-bearing verdict signals).
KEY_CLASSES = ["HGMMA", "QGMMA", "UTMALDG", "UBLKCP", "LDGSTS", "LDSM_STSM",
               "HMMA", "FFMA", "MUFU", "LDG", "LDL_SPILL", "STL_SPILL"]
SPILL_CLASSES = ["LDL_SPILL", "STL_SPILL"]

# Bounds (context discipline).  Never dump SASS for more than this many kernels, and never
# spend more than this many seconds in cuobjdump for one analyze() call.
MAX_KERNELS_SASS = 8        # how many kernels we actually disassemble per .so (>= #kernel families,
                            # so the family-cover ranking can sample one of EACH distinct kernel)
MAX_KERNELS_REPORT = 8      # how many kernels appear in the returned dict
CUOBJDUMP_TIMEOUT_S = 90    # hard wall on any single cuobjdump invocation
TOTAL_SASS_BUDGET_S = 120   # total time budget for all per-kernel SASS dumps

# Itanium-mangled FREE function -> the source identifier is length-prefixed, e.g.
# `_Z18attn_prefill_wgmmaI...` carries `18` then the 18-char name `attn_prefill_wgmma`.  We use
# that base identifier as a coarse "kernel family" key so we sample one representative of EACH
# distinct kernel (a scalar `attn_kernel` vs a `attn_prefill_wgmma`), instead of 6 instantiations
# of whichever name the ranking happens to favour.
_MANGLED_FREE_RE = re.compile(r"^_Z(\d+)([A-Za-z_]\w*)")

# Resource-usage line, e.g.:
#   REG:168 STACK:8 SHARED:1024 LOCAL:0 CONSTANT[0]:3328 TEXTURE:0 ...
_RES_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")
_FUNC_RE = re.compile(r"^\s*Function\s+(\S+?):?\s*$")
# A SASS instruction line begins with an address comment /*XXXX*/ (>=4 hex), optional
# predicate @P0 / @!P0, then the mnemonic.
_SASS_LINE_RE = re.compile(r"^\s*/\*[0-9a-fA-F]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)")


def _cuobjdump_path() -> Optional[str]:
    return shutil.which("cuobjdump")


def _classify(mnemonic: str) -> Optional[str]:
    """Map a SASS mnemonic (modifiers already trimmed at first '.') to an instr class."""
    base = mnemonic.split(".", 1)[0]
    for cls, prefixes in INSTR_CLASSES.items():
        for p in prefixes:
            if base == p or base.startswith(p):
                return cls
    return None


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 124, "<cuobjdump timeout>"
    except Exception as e:  # noqa: BLE001
        return 1, f"<cuobjdump error: {e}>"


def _parse_resource_usage(text: str) -> "OrderedDict[str, dict]":
    """Parse `cuobjdump --dump-resource-usage` -> {func_name: {regs,stack,shared,local}}."""
    out: "OrderedDict[str, dict]" = OrderedDict()
    cur = None
    for line in text.splitlines():
        m = _FUNC_RE.match(line)
        if m:
            cur = m.group(1)
            continue
        if cur is not None:
            r = _RES_RE.search(line)
            if r:
                out[cur] = dict(
                    regs=int(r.group(1)),
                    stack=int(r.group(2)),
                    shared=int(r.group(3)),
                    local=int(r.group(4)),
                )
                cur = None
    return out


def _histogram_from_sass(text: str) -> dict:
    """Count instr classes across a SASS dump (may span multiple Function blocks)."""
    hist = {cls: 0 for cls in INSTR_CLASSES}
    total = 0
    for line in text.splitlines():
        m = _SASS_LINE_RE.match(line)
        if not m:
            continue
        total += 1
        cls = _classify(m.group(1))
        if cls:
            hist[cls] += 1
    hist = {k: v for k, v in hist.items() if v}  # drop zero classes (bounded output)
    return {"instr_total": total, **hist}


def _short_name(fn: str, maxlen: int = 72) -> str:
    """Trim a giant demangled/mangled template name to something loggable."""
    # Pull a stable, informative middle if it is the FA3 mangled name.
    for tag in ("FlashAttnFwdCombine", "FlashAttnFwdSm90", "FlashAttnFwd", "attn_kernel"):
        if tag in fn:
            idx = fn.find(tag)
            frag = fn[idx: idx + maxlen]
            return frag + ("…" if idx + maxlen < len(fn) else "")
    if len(fn) <= maxlen:
        return fn
    return fn[: maxlen - 1] + "…"


def _family_key(fn: str) -> str:
    """Coarse 'kernel family' key (the source base identifier) so we don't waste the SASS budget
    on many instantiations of the SAME kernel and -- crucially -- so the bounded sample always
    covers EACH distinct kernel.  For an Itanium-mangled free function the identifier is
    length-prefixed (`_Z18attn_prefill_wgmmaI...` -> 'attn_prefill_wgmma'); the regex over-captures
    into the template suffix, so we clamp to the declared length.  Falls back to the leading
    token for namespaced/other names."""
    m = _MANGLED_FREE_RE.match(fn)
    if m:
        n = int(m.group(1))
        ident = m.group(2)
        return ident[:n] if len(ident) >= n else ident
    return re.split(r"[IL<(]", fn, 1)[0][:48]


def _rank_kernels(res: "OrderedDict[str, dict]", namespace: Optional[str]) -> list[str]:
    """Pick the most analysis-worthy kernels (bounded), guaranteeing per-family COVERAGE.

    Heuristics:
       - prefer the obvious attention mainloop (highest reg usage / has 'FlashAttnFwdSm90'
         or 'attn_kernel'), de-prioritise tiny combine/setup kernels,
       - dedupe near-identical template instantiations by their resource fingerprint so we
         don't waste budget disassembling 50 copies of the same shape,
       - then pull the best representative of EACH distinct kernel family to the FRONT.  This last
         step is load-bearing: the summary verdict (uses_wgmma / uses_tma) is computed from the
         sampled set, so a wgmma kernel whose symbol does NOT contain 'attn_kernel' (e.g.
         `attn_prefill_wgmma`) must still be sampled -- otherwise scalar `attn_kernel`/`*_coop`
         instantiations monopolise the budget and the verdict mis-reads 'no wgmma' though HGMMA is
         present (the step041 trap)."""
    names = list(res.keys())

    def score(fn: str) -> tuple:
        r = res[fn]
        is_main = 1 if ("FlashAttnFwdSm90" in fn or "attn_kernel" in fn
                        or (namespace and namespace in fn)) else 0
        is_combine = 1 if "Combine" in fn or "combine" in fn else 0
        # higher regs & shared usually == the real compute kernel
        return (is_main, -is_combine, r.get("regs", 0), r.get("shared", 0))

    # dedupe by resource fingerprint, keep first (already source-ordered)
    seen: set = set()
    deduped: list[str] = []
    for fn in sorted(names, key=score, reverse=True):
        r = res[fn]
        fp = (r.get("regs"), r.get("stack"), r.get("shared"), r.get("local"))
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(fn)

    # family cover: the score-best (already first-seen) member of each family leads, so the bounded
    # sample contains one of every distinct kernel before any second instantiation.
    cover: list[str] = []
    rest: list[str] = []
    seen_fam: set = set()
    for fn in deduped:
        fam = _family_key(fn)
        if fam in seen_fam:
            rest.append(fn)
        else:
            seen_fam.add(fam)
            cover.append(fn)
    return cover + rest


def _select_targets(res: "OrderedDict[str, dict]", targets: list[str]) -> list[str]:
    """Select the resource-usage kernels whose family matches any NCU-named `targets` kernel.

    `targets` are demangled kernel names from the NCU profile (`key_metrics._kernel(s)`); we reduce
    each to its base identifier and match by FAMILY (exact base, via `_family_key`) so 'attn_kernel'
    does not also grab 'attn_kernel_coop'/'attn_kernel_prefill_wgmma'.  Returns the matching kernels
    (deduped by resource fingerprint, highest-reg first) or [] when nothing matches (caller falls
    back to the family-cover ranking)."""
    want: set = set()
    for t in targets or []:
        if not t:
            continue
        tok = str(t).split("<", 1)[0].split("(", 1)[0].strip()
        tok = tok.split("::")[-1].strip()
        if tok:
            want.add(tok)
    if not want:
        return []
    matches = [fn for fn in res.keys() if _family_key(fn) in want]
    matches.sort(key=lambda fn: res[fn].get("regs", 0), reverse=True)
    seen: set = set()
    out: list[str] = []
    for fn in matches:
        r = res[fn]
        fp = (r.get("regs"), r.get("stack"), r.get("shared"), r.get("local"))
        if fp in seen:
            continue
        seen.add(fp)
        out.append(fn)
    return out


def _analyze_one(so_path: str, namespace: Optional[str],
                 max_kernels: int, targets: Optional[list[str]] = None) -> dict:
    """Resource-usage for all kernels + bounded per-kernel SASS histograms.

    When `targets` is given (demangled kernel names from the NCU profile), disassemble exactly the
    matching kernel family/families -- i.e. the kernel(s) that actually DOMINATE the profiled
    config's runtime -- so the verdict describes the hot path, not just whatever the name heuristic
    surfaces.  Falls back to the family-cover ranking when no target matches."""
    cuobj = _cuobjdump_path()
    if cuobj is None:
        return {"error": "cuobjdump not found on PATH"}
    if not os.path.exists(so_path):
        return {"error": f"file not found: {so_path}"}

    rc, ru_text = _run([cuobj, "--dump-resource-usage", so_path], CUOBJDUMP_TIMEOUT_S)
    res = _parse_resource_usage(ru_text) if rc == 0 else OrderedDict()

    if res and targets:
        ranked = _select_targets(res, targets) or _rank_kernels(res, namespace)
    else:
        ranked = _rank_kernels(res, namespace) if res else []
    # If resource-usage gave us nothing (unusual), fall back to a single full SASS dump.
    if not ranked:
        rc, sass = _run([cuobj, "--dump-sass", so_path], CUOBJDUMP_TIMEOUT_S)
        agg = _histogram_from_sass(sass) if rc == 0 else {"instr_total": 0}
        return {"kernels": [{"name": "<all>", "instr_mix": agg,
                             "regs": 0, "smem": 0, "spills": 0}],
                "num_kernels_total": 0, "num_kernels_analyzed": 1,
                "agg_instr_mix": agg}

    deadline = time.time() + TOTAL_SASS_BUDGET_S
    kernels: list[dict] = []
    agg: dict = {}
    for fn in ranked[:max_kernels]:
        if time.time() > deadline:
            break
        rc, sass = _run([cuobj, "--dump-sass", "--function", fn, so_path],
                        CUOBJDUMP_TIMEOUT_S)
        mix = _histogram_from_sass(sass) if rc == 0 else {"instr_total": 0}
        r = res.get(fn, {})
        spills = int(r.get("stack", 0)) + int(r.get("local", 0)) \
            + mix.get("LDL_SPILL", 0) + mix.get("STL_SPILL", 0)
        kernels.append({
            "name": _short_name(fn),
            "instr_mix": mix,
            "regs": r.get("regs", 0),
            "smem": r.get("shared", 0),
            "spills": spills,
        })
        for k, v in mix.items():
            agg[k] = agg.get(k, 0) + v

    return {
        "kernels": kernels[:MAX_KERNELS_REPORT],
        "num_kernels_total": len(res),
        "num_kernels_analyzed": len(kernels),
        "agg_instr_mix": agg,
    }


def _summarize(agg: dict) -> dict:
    """One-line verdict signals from the aggregate histogram."""
    has_wgmma = agg.get("HGMMA", 0) + agg.get("QGMMA", 0) + agg.get("IGMMA", 0) > 0
    has_tma = agg.get("UTMALDG", 0) + agg.get("UBLKCP", 0) > 0
    spills = agg.get("LDL_SPILL", 0) + agg.get("STL_SPILL", 0)
    verdict = []
    if not has_wgmma:
        verdict.append("NO wgmma (HGMMA/QGMMA) -> not using Hopper tensor cores")
    if not has_tma:
        verdict.append("NO TMA (UTMALDG/UBLKCP) -> not using async bulk copy")
    if spills:
        verdict.append(f"register SPILL present ({spills} LDL/STL)")
    if has_wgmma and has_tma and not spills:
        verdict.append("uses wgmma + TMA, no spills")
    return {
        "uses_wgmma": has_wgmma,
        "uses_fp8_wgmma": agg.get("QGMMA", 0) > 0,   # fp8 e4m3/e5m2 wgmma (vs bf16 HGMMA)
        "uses_tma": has_tma,
        "spills": spills,
        "wgmma_count": agg.get("HGMMA", 0) + agg.get("QGMMA", 0) + agg.get("IGMMA", 0),
        "tma_count": agg.get("UTMALDG", 0) + agg.get("UBLKCP", 0),
        "ffma_count": agg.get("FFMA", 0),
        "verdict": "; ".join(verdict) if verdict else "ok",
    }


def _diff(cand_agg: dict, ref_agg: dict, ref_label: str) -> list[str]:
    """Distilled (<=~8 line) histogram diff candidate-vs-reference for the key classes."""
    lines: list[str] = []
    cs = _summarize(cand_agg)
    rs = _summarize(ref_agg)
    if rs["uses_wgmma"] and not cs["uses_wgmma"]:
        lines.append(f"candidate has 0 wgmma vs {ref_label} {rs['wgmma_count']} "
                     f"(HGMMA/QGMMA) -- candidate is NOT using Hopper tensor cores")
    if rs["uses_tma"] and not cs["uses_tma"]:
        lines.append(f"candidate has 0 TMA vs {ref_label} {rs['tma_count']} "
                     f"(UTMALDG/UBLKCP) -- candidate is NOT using async bulk copy")
    if cs["spills"] and not rs["spills"]:
        lines.append(f"candidate has {cs['spills']} register-spill LDL/STL "
                     f"vs {ref_label} 0 -- reduce register pressure")
    # generic per-class delta on key classes (only report meaningful gaps)
    for cls in KEY_CLASSES:
        c = cand_agg.get(cls, 0)
        r = ref_agg.get(cls, 0)
        if c == r:
            continue
        if cls in SPILL_CLASSES and c > r:
            lines.append(f"{cls}: candidate {c} vs {ref_label} {r} (spill -- bad)")
        elif (r == 0) != (c == 0):  # presence flip is the actionable signal
            lines.append(f"{cls}: candidate {c} vs {ref_label} {r}")
    return lines[:8]


@logged("sass_analyze")
def analyze(so_path: str, namespace: str, baseline_so: Optional[str] = None,
            oracle_cubin: Optional[str] = None,
            targets: Optional[list[str]] = None) -> dict:
    """Distilled SASS analysis of a built `.so`/cubin.

    Returns a bounded dict::

        {
          "ok": bool,
          "so_path": str,
          "kernels": [ {name, instr_mix, regs, smem, spills}, ...],   # <= MAX_KERNELS_REPORT
          "summary": {uses_wgmma, uses_tma, spills, wgmma_count, ..., verdict},
          "num_kernels_total": int,
          "num_kernels_analyzed": int,
          # present only if a reference was given:
          "baseline_diff": [ "candidate has 0 HGMMA vs baseline 36 ...", ...],  # <= 8 lines
          "oracle_diff":   [ ... ],
        }

    `targets` (optional) restricts the disassembly to the named kernel family/families -- pass the
    NCU-profiled dominant kernel name(s) to get a HOT-PATH verdict (does the kernel that dominates
    the worst config use wgmma/TMA) instead of a whole-binary 'is it present anywhere' verdict.

    Never returns raw SASS text.  cuobjdump cost is bounded by MAX_KERNELS_SASS +
    TOTAL_SASS_BUDGET_S (a full FA3 .so dump would otherwise be ~minutes / hundreds of MB).
    """
    out: dict = {"ok": False, "so_path": so_path}

    if _cuobjdump_path() is None:
        out["error"] = "cuobjdump not found on PATH"
        return out

    cand = _analyze_one(so_path, namespace, MAX_KERNELS_SASS, targets=targets)
    if "error" in cand and "kernels" not in cand:
        out["error"] = cand["error"]
        return out

    cand_agg = cand.get("agg_instr_mix", {})
    out.update({
        "ok": True,
        "kernels": cand.get("kernels", []),
        "summary": _summarize(cand_agg),
        "num_kernels_total": cand.get("num_kernels_total", 0),
        "num_kernels_analyzed": cand.get("num_kernels_analyzed", 0),
    })

    if baseline_so and os.path.exists(baseline_so) and baseline_so != so_path:
        base = _analyze_one(baseline_so, "_vllm_fa3_C", MAX_KERNELS_SASS)
        base_agg = base.get("agg_instr_mix", {})
        out["baseline_diff"] = _diff(cand_agg, base_agg, "baseline")
        out["baseline_summary"] = _summarize(base_agg)

    if oracle_cubin and os.path.exists(oracle_cubin):
        ora = _analyze_one(oracle_cubin, "oracle", MAX_KERNELS_SASS)
        ora_agg = ora.get("agg_instr_mix", {})
        out["oracle_diff"] = _diff(cand_agg, ora_agg, "oracle")
        out["oracle_summary"] = _summarize(ora_agg)

    return out
