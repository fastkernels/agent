"""The optimization loop.

generate (Claude headless) -> anti-cheat -> build -> CORRECTNESS FIRST -> profile -> optimize,
with per-step checkpointing, an AVO-style supervisor watchdog, typed in-loop distillation, and a
regenerated evolution plot.  Terminates on (correctness AND >=1.0x baseline on ALL configs AND
greedy-decode equivalence) or an optional step budget.

Each candidate gets a UNIQUE op namespace (_distill_fa3_C_sNNN) so successive .so loads in the
long-running driver process don't double-register the same ops.
"""
from __future__ import annotations

import math
import os
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from typing import Optional

from . import config as C
from . import common, checkpoint, llm, prompts
from .common import BuildResult, AntiCheatResult, ConfigResult, EvalResult, StepRecord
from .gpu_pool import GPUPool


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


class Orchestrator:
    def __init__(self, run_id: str, config_set: str = "dev", max_steps: Optional[int] = None,
                 model: Optional[str] = None, effort: Optional[str] = None, seed: int = 0,
                 decode_tokens: int = 100,
                 decode_model: Optional[str] = None, do_decode_equiv: bool = True,
                 profile_enabled: bool = True):
        self.run_id = run_id
        self.cfgs = C.get_configs(config_set)
        self.config_set = config_set
        self.max_steps = max_steps
        self.model = model or C.HEADLESS_MODEL
        self.effort = effort if effort is not None else C.HEADLESS_EFFORT
        self.seed = seed
        self.decode_tokens = decode_tokens
        self.decode_model = decode_model
        self.do_decode_equiv = do_decode_equiv
        self.profile_enabled = profile_enabled
        self.pool = GPUPool()
        self.paths = checkpoint.init_run(run_id, dict(
            config_set=config_set, max_steps=max_steps, model=self.model, effort=self.effort,
            configs=[c.name for c in self.cfgs], created=common.now()))
        self.comp = self._load_components()
        self.kstore = None
        try:
            self.kstore = self.comp["KnowledgeStore"](notes_dir=os.path.join(self.paths["knowledge"]))
            self.kstore.seed()
        except Exception as e:  # knowledge store optional / best-effort
            self._log(f"[warn] knowledge store unavailable: {e!r}")

    # ---- component loading (defensive: a missing component degrades, not crashes) --------
    def _load_components(self) -> dict:
        comp: dict = {}
        def grab(name, mod, attr):
            try:
                m = __import__(mod, fromlist=[attr])
                comp[name] = getattr(m, attr)
            except Exception as e:
                comp[name] = None
                self._log(f"[warn] component {name} ({mod}.{attr}) unavailable: {e!r}")
        grab("build_candidate", "fa3agent.sandbox.build", "build_candidate")
        grab("anticheat", "fa3agent.sandbox.anticheat", "check")
        grab("check_correctness", "fa3agent.verify.correctness", "check_correctness")
        grab("check_decode_equiv", "fa3agent.verify.decode_equiv", "check_decode_equiv")
        grab("check_insitu", "fa3agent.verify.insitu", "check_insitu")
        grab("run_sanitizer", "fa3agent.verify.sanitizer", "run_sanitizer")
        grab("measure", "fa3agent.perf.measure", "measure")
        # NCU profiling runs OUT-OF-PROCESS (see _profile_subprocess): ncu_report + torch
        # segfault if co-imported, and this process loads torch via build/correctness/perf.
        grab("sass_analyze", "fa3agent.sass.analyze", "analyze")
        grab("update_plot", "fa3agent.plot.evolution", "update_plot")
        grab("get_hints", "fa3agent.reflib.hints", "get_hints")
        try:
            from fa3agent.knowledge.store import KnowledgeStore
            comp["KnowledgeStore"] = KnowledgeStore
        except Exception as e:
            comp["KnowledgeStore"] = None
            self._log(f"[warn] KnowledgeStore unavailable: {e!r}")
        return comp

    def _log(self, msg: str):
        print(f"[fa3agent {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _ns(self, step: int) -> str:
        return f"{C.DISTILL_NAMESPACE}_s{step:03d}"

    def _profile_subprocess(self, so_path: str, ns: str, cfg_name: str, gpu: int) -> dict:
        """Run NCU profiling in a torch-free subprocess (ncu_report + torch segfault together)."""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONPATH=C.REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [sys.executable, "-m", "fa3agent.perf._profile_main",
               so_path, ns, cfg_name, "0", self.paths["reports"]]
        try:
            p = subprocess.run(cmd, cwd=C.REPO_ROOT, env=env, capture_output=True, text=True, timeout=1200)
            for line in p.stdout.splitlines():
                if line.startswith("PROFILE_JSON:"):
                    return json.loads(line[len("PROFILE_JSON:"):])
            return {"bottleneck": f"(profile produced no JSON; rc={p.returncode}; {p.stderr[-300:]})"}
        except subprocess.TimeoutExpired:
            return {"bottleneck": "(profile timed out)"}
        except Exception as e:  # noqa
            return {"bottleneck": f"(profile subprocess error: {e!r})"}

    # ---- candidate generation -----------------------------------------------------------
    def _generate(self, step: int, prev: Optional[dict], best: Optional[dict]) -> tuple[str, Optional[str]]:
        cid = f"step{step:03d}"
        cand_path = os.path.join(self.paths["candidates"], f"{cid}.cu")
        session_id = prev.get("session_id") if prev else None

        if best is None:
            prompt = prompts.bootstrap_prompt(cand_path, [c.name for c in self.cfgs])
        else:
            kh = ""
            try:
                if self.comp.get("get_hints"):
                    kh = self.comp["get_hints"]("wgmma bf16") or ""
            except Exception:
                pass
            focus = prompts.FOCUS_LEVERS[step % len(prompts.FOCUS_LEVERS)]
            prompt = prompts.refine_prompt(
                cand_path, self.paths["best"],
                last_eval_summary=(prev.get("eval", {}) or {}).get("__summary__", str(prev.get("status"))) if prev else "n/a",
                profile_feedback=(prev.get("profile", {}) or {}).get("bottleneck", "") if prev else "",
                knowledge_hints=kh[:1500],
                focus=focus,
                supervisor_note=self._supervisor_note)

        res = llm.run_claude(prompt, cwd=C.REPO_ROOT, model=self.model, effort=self.effort,
                             session_id=session_id)
        # If the brain didn't produce a file, bootstrap from the seed (guarantees a candidate).
        if not os.path.exists(cand_path):
            if best is None and os.path.exists(os.path.join(C.PKG_DIR, "seed_kernel.cu")):
                shutil.copyfile(os.path.join(C.PKG_DIR, "seed_kernel.cu"), cand_path)
                self._log(f"[gen] brain produced no file; seeded {cid} from seed_kernel.cu")
            elif os.path.exists(self.paths["best"]):
                shutil.copyfile(self.paths["best"], cand_path)
                self._log(f"[gen] brain produced no file; reusing champion for {cid}")
            else:
                return cand_path, (res.session_id or session_id)
        return cand_path, (res.session_id or session_id)

    # ---- evaluation ---------------------------------------------------------------------
    def _evaluate(self, step: int, cid: str, ns: str, so_path: str) -> tuple[EvalResult, dict]:
        ev = EvalResult(num_configs=len(self.cfgs))
        profile = {}
        # CORRECTNESS FIRST
        try:
            cres = self.comp["check_correctness"](ns, self.cfgs, seed=self.seed, gpu_id=self.pool.gpu_ids[0])
        except Exception as e:
            ev.error = f"correctness raised: {e!r}"
            return ev, profile
        ev.per_config = cres
        ev.num_correct = sum(1 for r in cres if r.correct)
        ev.correct = ev.num_correct == len(self.cfgs)
        if not ev.correct:
            return ev, profile  # never profile/perf an incorrect kernel

        # PERF (one kernel per GPU at a time)
        if self.comp.get("measure"):
            def _meas(cfg, gpu):
                try:
                    return self.comp["measure"](ns, cfg, gpu)
                except Exception as e:
                    return ConfigResult(config=cfg.name, correct=True, ratio=0.0, note=f"measure err: {e!r}")
            perf = self.pool.map_exclusive(_meas, self.cfgs)
            by = {r.config: r for r in perf if r}
            for r in ev.per_config:
                if r.config in by:
                    r.ratio = by[r.config].ratio
                    r.cand_latency_ms = by[r.config].cand_latency_ms
                    r.base_latency_ms = by[r.config].base_latency_ms
            ratios = [r.ratio for r in ev.per_config]
            ev.min_ratio = min([x for x in ratios if x > 0], default=0.0)
            ev.geomean_ratio = _geomean(ratios)
            ev.weighted_geomean_ratio = _geomean([r.ratio for r, c in zip(ev.per_config, self.cfgs) for _ in range(int(c.weight))])

        # PROFILE the worst config (guides next optimization) — OUT-OF-PROCESS (torch-free).
        if self.profile_enabled and ev.per_config:
            worst = min(ev.per_config, key=lambda r: (r.ratio if r.ratio > 0 else 1e9))
            wcfg = next((c for c in self.cfgs if c.name == worst.config), self.cfgs[0])
            gpu = self.pool.gpu_ids[0]
            with self.pool.exclusive(gpu):
                profile = self._profile_subprocess(so_path, ns, wcfg.name, gpu)

        # NOTE: the faithful vLLM gates (in-situ A/B, then full decode-equality) are run at the
        # PROMOTION tier in run(), not here -- they need a model load and only matter for a
        # candidate that is correct AND improves the best ratio (cost-tiered verification).
        return ev, profile

    # ---- one step -----------------------------------------------------------------------
    def _step(self, step: int, prev: Optional[dict], best: Optional[dict]) -> StepRecord:
        cid = f"step{step:03d}"
        ns = self._ns(step)
        self._log(f"=== STEP {step} ({cid}, ns={ns}) ===")
        rec = StepRecord(step=step, cid=cid, parent=(best.get("cid") if best else None), status="generated")

        cand_path, session_id = self._generate(step, prev, best)
        rec.session_id = session_id
        if not os.path.exists(cand_path):
            rec.status = "no_candidate"
            return rec

        # anti-cheat (source) — fast, before building
        if self.comp.get("anticheat"):
            ac: AntiCheatResult = self.comp["anticheat"](cand_path, None)
            rec.anticheat = asdict(ac)
            if not ac.ok:
                rec.status = "cheat"
                self._log(f"[anticheat] REJECT: {ac.violations}")
                self._distill(rec, f"Anti-cheat rejected: {ac.violations}", ntype="alg")
                return rec

        # build
        bdir = checkpoint.build_dir(self.run_id, cid)
        br: BuildResult = self.comp["build_candidate"](cand_path, ns, bdir)
        rec.build = asdict(br)
        if not br.ok:
            rec.status = "build_failed"
            self._log(f"[build] FAILED: {br.error[:300]}")
            self._distill(rec, f"Build failed: {br.error[:400]}", ntype="ks")
            if br.env_fatal:
                rec.status = "env_fatal"
            return rec

        # anti-cheat (binary)
        if self.comp.get("anticheat"):
            ac2 = self.comp["anticheat"](cand_path, br.so_path)
            rec.anticheat = asdict(ac2)
            if not ac2.ok:
                rec.status = "cheat"
                self._distill(rec, f"Anti-cheat (binary) rejected: {ac2.violations}", ntype="alg")
                return rec

        ev, profile = self._evaluate(step, cid, ns, br.so_path)
        ev_d = asdict(ev); ev_d["__summary__"] = ev.summary()
        rec.eval = ev_d
        rec.profile = profile
        rec.status = "correct" if ev.correct else "incorrect"
        self._log(f"[eval] {ev.summary()}")

        # distill knowledge from the outcome (incl. negative knowledge)
        self._distill_step(rec, ev, profile)
        return rec

    # ---- knowledge distillation ---------------------------------------------------------
    def _distill(self, rec: StepRecord, body: str, ntype: str = "ks"):
        rec.knowledge_note = body[:500]
        if self.kstore:
            try:
                self.kstore.add_note(ntype, f"step{rec.step} {rec.status}", body,
                                     tags=[rec.status, self.config_set], confidence="experimental", step=rec.step)
            except Exception:
                pass

    def _distill_step(self, rec: StepRecord, ev: EvalResult, profile: dict):
        if ev.correct:
            body = (f"Correct candidate. min_ratio={ev.min_ratio:.3f} geomean={ev.geomean_ratio:.3f}. "
                    f"Bottleneck: {profile.get('bottleneck','?')}. decode_equiv={ev.decode_equiv_ok}.")
            self._distill(rec, body, ntype="pf")
        else:
            bad = [r.config for r in ev.per_config if not r.correct][:4]
            body = f"INCORRECT on configs {bad}. " + (ev.error or "see per-config err_ratio/lse_err.")
            self._distill(rec, body, ntype="ks")

    # ---- main loop ----------------------------------------------------------------------
    def run(self):
        steps = checkpoint.load_steps(self.run_id)
        start = (steps[-1]["step"] + 1) if steps else 0
        prev = steps[-1] if steps else None
        best = checkpoint.best_record(self.run_id)
        self._best_min_ratio = (best.get("eval", {}) or {}).get("min_ratio", 0.0) if best else -1.0
        self._stall = 0
        self._supervisor_note = ""
        if start > 0:
            self._log(f"RESUMING from step {start} (best min_ratio={self._best_min_ratio:.3f})")

        step = start
        while self.max_steps is None or step < (start + self.max_steps if steps else self.max_steps):
            try:
                rec = self._step(step, prev, best)
            except Exception as e:
                self._log(f"[error] step {step} crashed: {e!r}\n{traceback.format_exc()[:1000]}")
                rec = StepRecord(step=step, cid=f"step{step:03d}", parent=None, status="error",
                                 knowledge_note=str(e)[:300])

            # ---- faithful vLLM gates, COST-TIERED (only for a correct promotion candidate) ----
            # Tier 2: in-situ A/B on LIVE vLLM tensors (cheap, faithful -- catches live-layout
            #         bugs the synthetic gate misses, e.g. broadcast-descale OOB).
            # Tier 3: full 100-token greedy decode-equality (only when the candidate also meets
            #         the perf target -- the final acceptance gate for the residual near-tie risk).
            if self.do_decode_equiv and rec.status == "correct" and rec.eval:
                mr = rec.eval.get("min_ratio", 0.0)
                promo_cand = (best is None) or (mr >= self._best_min_ratio - C.PERF_NOISE_BAND)
                so = (rec.build or {}).get("so_path"); ns = self._ns(rec.step)
                if promo_cand and so and self.comp.get("check_insitu"):
                    gpu = self.pool.gpu_ids[0]
                    with self.pool.exclusive(gpu):
                        ir = self.comp["check_insitu"](so, ns)
                    rec.eval["insitu"] = ir
                    self._log(f"[insitu] {ir.get('note','')}")
                    if ir.get("ok") is False and not ir.get("skip"):
                        rec.status = "insitu_fail"   # correct on synthetic, diverges on live -> not promotable
                    elif mr >= (1.0 - C.PERF_NOISE_BAND) and self.comp.get("check_decode_equiv"):
                        gpu = self.pool.gpu_ids[0]
                        with self.pool.exclusive(gpu):
                            ok, note = self.comp["check_decode_equiv"](so, ns, model=self.decode_model,
                                                                       num_tokens=self.decode_tokens)
                        rec.eval["decode_equiv_ok"] = ok; rec.eval["decode_equiv_note"] = note
                        self._log(f"[decode_equiv] ok={ok}: {note}")

            # promotion: first correct candidate, or one that improves the MINIMUM ratio
            improved = False
            if rec.status == "correct" and rec.eval:
                mr = rec.eval.get("min_ratio", 0.0)
                if best is None or mr >= self._best_min_ratio - C.PERF_NOISE_BAND:
                    rec.is_best = True
                    improved = True
                    self._best_min_ratio = max(self._best_min_ratio, mr)
                    checkpoint.promote_best(self.run_id, rec.cid)
                    best = asdict(rec)
                    self._log(f"[promote] {rec.cid} -> champion (min_ratio={mr:.3f})")

            checkpoint.append_step(self.run_id, rec)
            self._update_plot()

            # supervisor watchdog (AVO-style)
            if improved:
                self._stall = 0; self._supervisor_note = ""
            else:
                self._stall += 1
                if self._stall >= 3:
                    self._supervisor_note = ("You have not improved in several rounds. Step back, "
                        "review the profiler bottleneck, and try a DIFFERENT subsystem/approach this round.")

            # success?
            if rec.status == "correct" and rec.eval:
                mr = rec.eval.get("min_ratio", 0.0)
                de = rec.eval.get("decode_equiv_ok", None)
                if mr >= (1.0 - C.PERF_NOISE_BAND) and (de is True or not self.do_decode_equiv):
                    self._log(f"SUCCESS: correct + min_ratio {mr:.3f} >= 1.0 on all configs + decode_equiv={de}")
                    return {"status": "success", "step": step, "min_ratio": mr}
            if rec.status == "env_fatal":
                self._log("ABORT: environment-fatal error (unfixable by agent).")
                return {"status": "env_fatal", "step": step}

            prev = asdict(rec)
            step += 1
        self._log(f"Reached step budget ({self.max_steps}). best min_ratio={self._best_min_ratio:.3f}")
        return {"status": "budget_reached", "step": step, "best_min_ratio": self._best_min_ratio}

    def _update_plot(self):
        if self.comp.get("update_plot"):
            try:
                self.comp["update_plot"](self.paths["run"], self.paths["jsonl"],
                                         os.path.join(self.paths["plots"], "evolution.png"))
            except Exception as e:
                self._log(f"[warn] plot failed: {e!r}")
