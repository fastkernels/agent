"""fa3agent.trial -- the in-loop candidate evaluator the brain runs itself.

Workstream A (AVO / K-Search local refinement): this frees the brain from the old "don't build"
contract.  The brain runs, from the repo root, DURING a step:

    python -m fa3agent.trial <candidate.cu> [flags]

to compile + correctness-check + benchmark (and optionally NCU-profile) a candidate IN PLACE -- the
SAME build / verify / measure path the driver uses for the authoritative eval -- and gets back a
compact JSON verdict.  So it can iterate edit -> trial -> diagnose within a single step instead of
writing one blind candidate and waiting a whole round for feedback.  The driver still re-runs the
AUTHORITATIVE eval on whatever file the brain leaves on disk; these trials are throwaway scratch.

Run context comes from FA3_TRIAL_* env the driver sets when it launches the brain; standalone
defaults let it also be run by hand:

    FA3_TRIAL_CONFIG_SET    config set name (default "dev")
    FA3_TRIAL_SEED          correctness seed (default 0)
    FA3_TRIAL_GPUS          comma-separated GPU ids the trial may use (default config.GPU_IDS)
    FA3_TRIAL_SCRATCH       dir for throwaway trial builds (default <cwd>/.fa3_trials)
    FA3_TRIAL_CHAMPION_MIN  champion min_ratio to beat (annotates the verdict; optional)

Isolation / safety: each invocation is its OWN short-lived process that builds + loads exactly ONE
.so under a UNIQUE namespace and then exits, so there is no TORCH_LIBRARY double-registration across
trials (the footgun the driver sidesteps with per-attempt namespaces; reloading a second .so under
the same op namespace re-runs registration -> std::terminate).  NCU profiling stays out-of-process
(it segfaults co-imported with torch), reusing the driver's perf._profile_main.  The verdict is
BOUNDED (distilled) -- never a raw dump.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

from . import config as C


def _truncate(s, n: int = 600) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " \u2026[truncated]"


def _geomean(xs) -> float:
    vals = [x for x in xs if x and x > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _summary_line(v: dict) -> str:
    """A single human-readable headline (the machine-parseable detail is the TRIAL_JSON line)."""
    if not v.get("built"):
        why = v.get("error") or "; ".join(v.get("violations") or [])
        return f"[trial] BUILD/ANTICHEAT FAIL ({v.get('stage')}): {_truncate(why, 220)}"
    if not v.get("correct"):
        fails = ", ".join(f["config"] for f in (v.get("failed") or [])[:6])
        return (f"[trial] INCORRECT ({v.get('num_correct')}/{v.get('num_configs')} configs pass); "
                f"failing: {fails}")
    if "min_ratio" not in v:
        return f"[trial] BUILDS + correct ({v.get('num_correct')}/{v.get('num_configs')}); perf skipped"
    line = (f"[trial] OK: correct {v['num_correct']}/{v['num_configs']} | "
            f"min_ratio={v['min_ratio']:.3f} geomean={v['geomean']:.3f}")
    if v.get("champion_min_ratio") is not None:
        line += f" | champion min={v['champion_min_ratio']:.3f} beats={v.get('beats_champion')}"
    if v.get("bottleneck"):
        line += f"\n[trial] worst={v.get('worst_config')} bottleneck: {_truncate(v['bottleneck'], 160)}"
    return line


def _emit(verdict: dict, json_only: bool) -> None:
    if not json_only:
        print(_summary_line(verdict), flush=True)
    print("TRIAL_JSON: " + json.dumps(verdict), flush=True)


def _sass_summary(so_path: str, ns: str) -> dict:
    """Best-effort SASS instruction-mix verdict (wgmma/TMA presence, spills) for the built .so."""
    try:
        from .sass.analyze import analyze
        s = analyze(so_path, ns)
    except Exception as e:  # noqa - best-effort
        return {"ok": False, "error": repr(e)[:160]}
    if not isinstance(s, dict) or not s.get("ok"):
        return {"ok": False}
    summ = s.get("summary") or {}
    v = summ.get("verdict")
    if isinstance(v, list):
        v = "; ".join(str(x) for x in v)
    return {"ok": True,
            "uses_wgmma": summ.get("uses_wgmma"),
            "uses_tma": summ.get("uses_tma"),
            "uses_fp8_wgmma": summ.get("uses_fp8_wgmma"),
            "verdict": (str(v)[:220] if v else "")}


def _profile_bottleneck(so_path: str, ns: str, cfg_name: str, gpu: int, reports_dir: str,
                        uses_wgmma) -> str:
    """One-line NCU bottleneck for the worst config, OUT-OF-PROCESS (ncu_report + torch co-segfault).

    Mirrors orchestrator._profile_subprocess: the gpu is selected via CUDA_VISIBLE_DEVICES, so the
    in-mask device index passed to _profile_main is always 0."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=C.REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    wflag = "-" if uses_wgmma is None else ("1" if uses_wgmma else "0")
    cmd = [sys.executable, "-m", "fa3agent.perf._profile_main",
           so_path, ns, cfg_name, "0", reports_dir, wflag]
    try:
        p = subprocess.run(cmd, cwd=C.REPO_ROOT, env=env, capture_output=True, text=True, timeout=1200)
        for line in p.stdout.splitlines():
            if line.startswith("PROFILE_JSON:"):
                d = json.loads(line[len("PROFILE_JSON:"):])
                return _truncate(d.get("bottleneck") or "", 240)
    except Exception:  # noqa - best-effort
        return ""
    return ""


def _prune_scratch(scratch: str, keep: int = 3) -> None:
    """Keep only the most recent `keep` trial build dirs in the scratch root (the brain may run many
    trials per step; each build dir holds a .so + objects).  Throwaway scratch -- excluded from the
    durable archive and wiped on a fresh run -- so this just bounds live-disk growth.  Best-effort."""
    try:
        subs = [os.path.join(scratch, d) for d in os.listdir(scratch)
                if d.startswith("_fa3_trial_") and os.path.isdir(os.path.join(scratch, d))]
    except OSError:
        return
    if len(subs) <= keep:
        return
    subs.sort(key=lambda p: os.path.getmtime(p))
    import shutil
    for old in subs[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def _resolve_configs(config_set: str, subset: str):
    cfgs = C.get_configs(config_set)
    subset = (subset or "").strip()
    if subset:
        want = {n.strip() for n in subset.split(",") if n.strip()}
        sub = [c for c in cfgs if c.name in want]
        if sub:
            return sub
    return cfgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m fa3agent.trial",
        description="In-loop build + correctness + perf trial for a candidate .cu (the same "
                    "pipeline the driver uses); prints a compact JSON verdict.")
    ap.add_argument("candidate", help="path to the candidate .cu to trial")
    ap.add_argument("--configs", default="",
                    help="comma-separated subset of config names to check/measure (default: all)")
    ap.add_argument("--no-perf", action="store_true",
                    help="build + correctness only (fast; use while fixing a compile/correctness bug)")
    ap.add_argument("--profile", action="store_true",
                    help="also attach the NCU bottleneck for the worst config (slower)")
    ap.add_argument("--json-only", action="store_true", help="print only the TRIAL_JSON line")
    args = ap.parse_args(argv)
    json_only = args.json_only

    cand = os.path.abspath(args.candidate)
    if not os.path.isfile(cand):
        _emit({"ok": False, "built": False, "stage": "input",
               "error": f"candidate not found: {cand}"}, json_only)
        return 2

    config_set = os.environ.get("FA3_TRIAL_CONFIG_SET", "dev")
    try:
        cfgs = _resolve_configs(config_set, args.configs)
    except Exception as e:  # noqa
        _emit({"ok": False, "built": False, "stage": "input",
               "error": f"bad config set {config_set!r}: {e!r}"}, json_only)
        return 2

    seed = int(os.environ.get("FA3_TRIAL_SEED", "0") or "0")
    gpus = [int(g) for g in (os.environ.get("FA3_TRIAL_GPUS")
                             or ",".join(map(str, C.GPU_IDS))).split(",") if g.strip()]
    if not gpus:
        gpus = list(C.GPU_IDS) or [0]
    scratch = os.environ.get("FA3_TRIAL_SCRATCH") or os.path.join(os.getcwd(), ".fa3_trials")
    champ_min_env = os.environ.get("FA3_TRIAL_CHAMPION_MIN")
    champ_min = float(champ_min_env) if champ_min_env not in (None, "") else None

    ns = f"_fa3_trial_{os.getpid()}_{int(time.time() * 1000) % 1_000_000}"
    bdir = os.path.join(scratch, ns)
    try:
        os.makedirs(scratch, exist_ok=True)
        _prune_scratch(scratch, keep=3)
        os.makedirs(bdir, exist_ok=True)
    except OSError as e:
        _emit({"ok": False, "built": False, "stage": "input",
               "error": f"cannot create scratch dir {bdir}: {e!r}"}, json_only)
        return 2

    verdict: dict = {"ok": False, "built": False, "correct": False, "namespace": ns,
                     "config_set": config_set, "num_configs": len(cfgs),
                     "configs": [c.name for c in cfgs], "champion_min_ratio": champ_min}

    # Torch-heavy imports are deferred until after arg parsing so --help / bad args stay instant.
    from .sandbox.anticheat import check as anticheat_check
    from .sandbox.build import build_candidate

    # (1) source anti-cheat (same forbidden-lib scan the driver runs before building)
    ac = anticheat_check(cand, None)
    if not ac.ok:
        verdict.update(stage="anticheat", violations=list(ac.violations)[:8])
        _emit(verdict, json_only)
        return 1

    # (2) build (toolkit-only, sm_90a) under a unique namespace
    br = build_candidate(cand, ns, bdir)
    if not br.ok:
        verdict.update(stage="build", error=_truncate(br.error or br.log),
                       env_fatal=bool(br.env_fatal))
        _emit(verdict, json_only)
        return 1
    so_path = br.so_path
    verdict["built"] = True
    verdict["build_s"] = round(br.duration_s, 1)

    # (3) binary anti-cheat (nm/ldd vs forbidden dynamic libs)
    ac2 = anticheat_check(cand, so_path)
    if not ac2.ok:
        verdict.update(stage="anticheat_binary", violations=list(ac2.violations)[:8])
        _emit(verdict, json_only)
        return 1

    # (4) SASS instruction mix (best-effort; tells the brain whether wgmma/TMA actually landed)
    verdict["sass"] = _sass_summary(so_path, ns)

    # (5) CORRECTNESS FIRST -- never benchmark an incorrect kernel
    from .verify.correctness import check_correctness
    try:
        cres = check_correctness(ns, cfgs, seed=seed, gpu_id=gpus[0])
    except Exception as e:  # noqa
        verdict.update(stage="correctness", error=f"correctness raised: {e!r}"[:300])
        _emit(verdict, json_only)
        return 1
    per: list[dict] = []
    failed: list[dict] = []
    n_ok = 0
    for r in cres:
        ok = bool(getattr(r, "correct", False))
        n_ok += int(ok)
        entry = {"config": r.config, "correct": ok}
        per.append(entry)
        if not ok:
            failed.append({"config": r.config,
                           "err_ratio": round(float(getattr(r, "err_ratio", 0.0) or 0.0), 2),
                           "note": _truncate(getattr(r, "note", "") or "", 160)})
    verdict["num_correct"] = n_ok
    verdict["correct"] = (n_ok == len(cfgs))
    if not verdict["correct"]:
        verdict.update(stage="correctness", failed=failed[:8])
        verdict["per_config"] = per
        _emit(verdict, json_only)
        return 1

    # (6) PERF (ratios vs the FA3 baseline; min/geomean are the optimization targets)
    if not args.no_perf:
        from .perf.measure import measure_all_subprocess
        from . import promotion
        perf = measure_all_subprocess(ns, cfgs, so_path, gpu_ids=gpus)
        rmap = {r.config: r for r in perf if r}
        ratios: list[float] = []
        for p in per:
            rr = rmap.get(p["config"])
            ratio = float(getattr(rr, "ratio", 0.0)) if rr else 0.0
            p["ratio"] = round(ratio, 3)
            ratios.append(ratio)
        pos = [x for x in ratios if x > 0]
        verdict["min_ratio"] = round(min(pos), 3) if pos else 0.0
        verdict["geomean"] = round(_geomean(ratios), 3)
        verdict["weighted_geomean"] = round(
            promotion.weighted_geomean(ratios, [c.weight for c in cfgs]), 3)
        if champ_min is not None:
            verdict["beats_champion"] = verdict["min_ratio"] > champ_min
        worst = min(per, key=lambda p: (p.get("ratio", 0.0) if p.get("ratio", 0.0) > 0 else 1e9))
        verdict["worst_config"] = worst["config"]
    verdict["per_config"] = per
    verdict["ok"] = True

    # (7) optional NCU bottleneck for the worst config
    if args.profile and not args.no_perf:
        verdict["bottleneck"] = _profile_bottleneck(
            so_path, ns, verdict.get("worst_config") or cfgs[0].name, gpus[0], bdir,
            (verdict.get("sass") or {}).get("uses_wgmma"))

    _emit(verdict, json_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
