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
import re
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from typing import Optional

from . import config as C
from . import common, checkpoint, llm, prompts, promotion, runlog
from .common import BuildResult, AntiCheatResult, ConfigResult, EvalResult, StepRecord
from .gpu_pool import GPUPool
from .memory import MemoryStore
from . import levers
from .levers import LeverController
from .plateau import PlateauState


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _doc_query(focus: str, bottleneck: str) -> str:
    """Build a docs.retrieve query from the step's focus lever + the bottleneck.

    The raw bottleneck string carries metric noise (e.g. '78.3%', '1.05e+06', 'SM SOL 0.2%')
    that pollutes the lexical scorer, so numbers/symbols are stripped; combining the focus
    (the technique to apply) with the cleaned symptom keywords retrieves both the how-to and
    the diagnostic docs (verified: this beats querying the raw bottleneck verbatim).
    """
    bt = bottleneck or ""
    if bt.startswith("("):          # status placeholder like "(profile timed out)" -> ignore
        bt = ""
    bt = re.sub(r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?\s*%?", " ", bt)  # drop numeric/metric noise
    bt = re.sub(r"[<>=/:+,.]", " ", bt)
    return (focus + " " + bt).strip()


def _sass_line(s) -> str:
    """Distill a sass.analyze result dict to one prompt line (kernel instruction-mix verdict)."""
    if not isinstance(s, dict) or not s.get("ok"):
        return ""
    summ = s.get("summary") or {}
    v = summ.get("verdict")
    if isinstance(v, list):
        v = "; ".join(str(x) for x in v)
    line = str(v) if v else ""
    bd = s.get("baseline_diff") or []
    if bd:
        line += (" | " if line else "") + "vs baseline: " + " | ".join(str(x) for x in bd[:2])
    return line[:300]


def _knowledge_line(notes) -> str:
    """Distill knowledge-store query results to a short injected block (top notes)."""
    out = []
    for n in (notes or [])[:3]:
        out.append(f"[{n.get('id', '?')}] {n.get('title', '')}: {n.get('snippet', '')}")
    return "\n".join(out)[:1500]


def _profile_digest(profile) -> str:
    """Bounded (<=12-line) digest of the FULL distilled NCU feedback (B1).

    The classifier computes a recommended `optimization`, a `plan`, an evidence-cited `ranked`
    list (per-kernel time split + cited metrics + NCU rules), and a `_bottleneck_label`, all of
    which are checkpointed but otherwise dropped from the prompt -- only the one-line `bottleneck`
    was forwarded.  This turns the rest into a compact actionable block (the diagnose->steer hand-off).
    """
    if not isinstance(profile, dict) or not profile:
        return ""
    bt = profile.get("bottleneck", "") or ""
    if bt.startswith("("):          # status placeholder (timeout / no-JSON) -> nothing actionable
        return ""
    km = profile.get("key_metrics") or {}
    lines: list[str] = []
    label = km.get("_bottleneck_label")
    kern = km.get("_kernel")
    if label:
        head = f"bottleneck class: {label}"
        if kern:
            head += f" (profiled kernel: {str(kern).split('<')[0][:48]})"
        lines.append(head)
    if profile.get("optimization"):
        lines.append(f"recommended optimization: {profile['optimization']}")
    if profile.get("plan"):
        lines.append(f"plan: {profile['plan']}")
    for r in (profile.get("ranked") or [])[:6]:
        if r:
            lines.append(f"- {str(r)[:160]}")
    return "\n".join(lines[:12])[:1800]


# Bottleneck label -> the corrected-cause / next-move hint that goes into a causal lesson (E1).
# Substring-matched against the NCU bottleneck label (most specific first).
_NEXT_HINT = [
    ("overlap", "softmax/GEMM don't overlap; a targeted edit won't fix it -> needs warp-"
                "specialization (producer/consumer split) + O moved out of registers"),
    ("off-tensor", "GEMMs not on tensor cores -> move QK/PV to wgmma.mma_async"),
    ("register-spill", "register spilling -> split consumers / setmaxnreg to cut live registers"),
    ("latency", "memory-latency bound -> async copy (TMA/cp.async.bulk.tensor) + multistage pipeline"),
    ("occupancy", "low occupancy -> persistent/split-K/L2-aware scheduling or clusters+TMA multicast"),
    ("grid", "grid/occupancy limited -> persistent kernel / split-K"),
    ("epilogue", "epilogue bound -> async store (STSM / TMA store)"),
    ("dram", "DRAM-bandwidth bound -> improve reuse/tiling/SMEM swizzle"),
    ("bandwidth", "bandwidth bound -> improve reuse/tiling/SMEM swizzle"),
]


def _next_hint(bottleneck: str) -> str:
    bn = (bottleneck or "").lower()
    for key, hint in _NEXT_HINT:
        if key in bn:
            return hint
    return "this lever didn't move the bottleneck -> try a DIFFERENT subsystem next"


def _label_of(profile: Optional[dict]) -> str:
    """The NCU `_bottleneck_label` (the machine-readable class used to pick a lever), falling back
    to the human one-liner.  Empty when there is no profile yet (bootstrap)."""
    if not profile:
        return ""
    km = profile.get("key_metrics") or {}
    return km.get("_bottleneck_label") or profile.get("bottleneck", "") or ""


def _sass_flags(rec: Optional[dict]) -> tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """(uses_tma, uses_wgmma, uses_fp8_wgmma) from a step record's SASS summary, or all-None.

    Feeds the lever selector's foundation gates (B3b): an overlap/off-tensor-core bottleneck on a
    kernel that has wgmma but NO TMA should land the TMA pipeline before warp-specialization; a slow
    fp8 worst-config with wgmma but NO fp8 wgmma (QGMMA) should land fp8 tensor cores first.
    Unknown (no SASS) -> all None so the gates stay conservative (they fire only on explicit False).

    Prefers `sass_hot` (the NCU-dominant kernel of the worst config) over `sass` (whole-binary
    presence) so the gate reflects whether the BOTTLENECK kernel uses the technique -- not merely
    whether some other kernel in the .so happens to (e.g. a prefill wgmma path while the slow decode
    config still runs scalar).  Falls back to the whole-binary verdict when no hot verdict exists."""
    s = None
    if isinstance(rec, dict):
        s = rec.get("sass_hot") or rec.get("sass")
    if not isinstance(s, dict) or not s.get("ok"):
        return None, None, None
    summ = s.get("summary") or {}
    return summ.get("uses_tma"), summ.get("uses_wgmma"), summ.get("uses_fp8_wgmma")


def _sass_has(sass: Optional[dict], signal: str) -> bool:
    """True iff the built kernel's SASS summary confirms `signal` (e.g. uses_wgmma).  An unreadable
    / failed SASS counts as NOT confirmed -- the completion gate then treats the functionality as
    not-yet-adopted (conservative; better to re-engage the brain than to wave through a blind miss)."""
    if not signal or not isinstance(sass, dict) or not sass.get("ok"):
        return False
    return bool((sass.get("summary") or {}).get(signal))


def _looks_like_mem_bug(ev) -> bool:
    """True when a correctness failure smells like a memory bug worth a sanitizer pass."""
    if getattr(ev, "error", ""):
        return True
    pats = ("nan", "inf", "illegal", "out of bound", "out-of-bound", "misalign", "shape",
            "mismatch", "address", "fault")
    for r in getattr(ev, "per_config", []) or []:
        if not getattr(r, "correct", True):
            note = (getattr(r, "note", "") or "").lower()
            if any(p in note for p in pats):
                return True
    return False


class Orchestrator:
    def __init__(self, run_id: str, config_set: str = "dev", max_steps: Optional[int] = None,
                 model: Optional[str] = None, effort: Optional[str] = None, seed: int = 0,
                 decode_tokens: int = 100,
                 decode_model: Optional[str] = None, do_decode_equiv: bool = True,
                 profile_enabled: bool = True, resume: bool = False):
        self.run_id = run_id
        self.resume = resume
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
        # F1/F2: the LIVE run dir is under WORK_ROOT (scratch); a FRESH run wipes WORK_ROOT so the
        # brain's working set holds only the current run; a RESUME restores it from the archive.
        self.paths = checkpoint.init_run(run_id, dict(
            config_set=config_set, max_steps=max_steps, model=self.model, effort=self.effort,
            configs=[c.name for c in self.cfgs], created=common.now()), resume=resume)
        # F5: isolate the brain's Claude chat history in a per-run config dir under the workdir
        # (wiped with WORK_ROOT on a fresh run, excluded from the archive mirror).  On a fresh run
        # also drop any prior-run transcripts under the default ~/.claude (fallback for a CLI that
        # ignores CLAUDE_CONFIG_DIR), scoped to the brain's own project slug.
        self._brain_claude_dir = os.path.join(self.paths["run"], ".claude")
        if not resume:
            llm.wipe_default_brain_transcripts()
        llm.seed_brain_claude_config(self._brain_claude_dir)
        # E2: cross-step memory (MEMORY.md + state) lives in the LIVE workdir; on resume it is
        # restored from the archive by F2 before this point, so MemoryStore picks it back up.
        self._memory = MemoryStore(self.paths["run"])
        # B3: bottleneck-driven lever selection + persisted per-lever outcome ledger (lever_state.json
        # in the workdir, mirrored to the archive); replaces the old step%5 focus rotation.
        self._levers = LeverController(self.paths["run"])
        # CUTLASS->PTX oracle: a concrete Hopper reference (wgmma+TMA instruction sequence) built
        # ONCE out-of-band and reused by the completion gate when a foundation capability is still
        # missing.  Cached per (signal, head_dim) so it never rebuilds inside the per-step loop.
        self._oracle_refs: dict = {}
        # C2: shared plateau/branch state (champ_history + STUCK + knee + rewrite-branch FSM),
        # persisted in branches.json (workdir, mirrored).  Drives the protected rewrite branch.
        self._plateau = PlateauState(self.paths["run"])
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
        grab("check_cuda_graph", "fa3agent.verify.cudagraph", "check_cuda_graph")
        grab("run_sanitizer", "fa3agent.verify.sanitizer", "run_sanitizer")
        grab("measure", "fa3agent.perf.measure", "measure")
        # NCU profiling runs OUT-OF-PROCESS (see _profile_subprocess): ncu_report + torch
        # segfault if co-imported, and this process loads torch via build/correctness/perf.
        grab("sass_analyze", "fa3agent.sass.analyze", "analyze")
        grab("build_oracle", "fa3agent.sass.cutlass_oracle", "build_oracle")
        grab("update_plot", "fa3agent.plot.evolution", "update_plot")
        grab("get_hints", "fa3agent.reflib.hints", "get_hints")
        grab("retrieve", "fa3agent.docs.retrieve", "retrieve")
        grab("get_section", "fa3agent.docs.retrieve", "get_section")
        try:
            from fa3agent.knowledge.store import KnowledgeStore
            comp["KnowledgeStore"] = KnowledgeStore
        except Exception as e:
            comp["KnowledgeStore"] = None
            self._log(f"[warn] KnowledgeStore unavailable: {e!r}")
        return comp

    def _log(self, msg: str):
        # The general run narrative goes to stdout, which cmd_run tees into
        # <run_dir>/run.log (plus whatever the launcher does with stdout).
        print(f"[fa3agent {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _ns(self, step: int) -> str:
        return f"{C.DISTILL_NAMESPACE}_s{step:03d}"

    def _brain_env(self, cid: str) -> dict:
        """Environment handed to the headless brain: its isolated Claude config dir (F5) PLUS the
        FA3_TRIAL_* context the in-loop `fa3agent.trial` tool reads (Workstream A).  With these set,
        the brain can build + correctness-check + benchmark a candidate ITSELF during the step (the
        same pipeline the driver uses) and iterate before committing.  GPUs are free here -- the
        orchestrator is blocked on the brain subprocess for the whole generation -- so the trial may
        use the entire pool.  Trial builds land in a per-step scratch dir under the step build dir."""
        env = {"CLAUDE_CONFIG_DIR": self._brain_claude_dir}
        if C.TRIAL_IN_LOOP:
            env.update({
                "FA3_TRIAL_CONFIG_SET": self.config_set,
                "FA3_TRIAL_SEED": str(self.seed),
                "FA3_TRIAL_GPUS": ",".join(str(g) for g in self.pool.gpu_ids),
                "FA3_TRIAL_SCRATCH": os.path.join(checkpoint.build_dir(self.run_id, cid), "trials"),
            })
            champ = getattr(self, "_best_min_ratio", -1.0)
            if champ is not None and champ >= 0:
                env["FA3_TRIAL_CHAMPION_MIN"] = f"{champ:.4f}"
        return env

    def _profile_subprocess(self, so_path: str, ns: str, cfg_name: str, gpu: int,
                            uses_wgmma: Optional[bool] = None) -> dict:
        """Run NCU profiling in a torch-free subprocess (ncu_report + torch segfault together).

        `uses_wgmma` (SASS ground truth, from this step's sass.analyze) is forwarded so the
        bottleneck classifier can detect overlap-bound and avoid the off-tensor-core misfire (B2).
        """
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONPATH=C.REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        wflag = "-" if uses_wgmma is None else ("1" if uses_wgmma else "0")
        cmd = [sys.executable, "-m", "fa3agent.perf._profile_main",
               so_path, ns, cfg_name, "0", self.paths["reports"], wflag]
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
    def _generate(self, step: int, prev: Optional[dict], best: Optional[dict],
                  lever=None, focus: str = "", best_path_override: Optional[str] = None,
                  branch: Optional[dict] = None) -> tuple[str, Optional[str]]:
        cid = f"step{step:03d}"
        cand_path = os.path.join(self.paths["candidates"], f"{cid}.cu")
        if not focus:
            focus = lever.focus if lever else (prompts.BRANCH_REFACTOR_FOCUS if branch is not None else "")
        # During a branch the brain continues from the branch SEED/TIP, not the trunk champion.
        read_path = best_path_override or self.paths["best"]
        # Session resume is OFF by default: replaying the full prior conversation as the request
        # fails on the Cortex relay (api_retry loop). Fresh session per step; context is in the
        # refine prompt + the champion .cu on disk. See config.HEADLESS_RESUME_SESSION.
        prev_sid = prev.get("session_id") if prev else None
        session_id = prev_sid if C.HEADLESS_RESUME_SESSION else None

        if best is None:
            prompt = prompts.bootstrap_prompt(cand_path, [c.name for c in self.cfgs])
        else:
            bottleneck = (prev.get("profile", {}) or {}).get("bottleneck", "") if prev else ""
            dq = _doc_query(focus, bottleneck)
            # Reference-lib hints for THIS step's lever/technique (NOT a hardcoded topic): the
            # readable pure-PTX model for what the brain is being steered to do this step.
            hints_topic = (lever.hints if lever else "") or "wgmma"
            kh = ""
            try:
                if self.comp.get("get_hints"):
                    kh = self.comp["get_hints"](hints_topic) or ""
            except Exception:
                pass
            # Proactively surface NVIDIA docs so they reach the brain even on steps where it
            # wouldn't query itself (it can still call docs.retrieve/get_section for more).
            # (a) #2 curated: the authoritative sm_90a section(s) for THIS step's focus technique
            #     -- deterministic, on-target, independent of the lexical scorer.
            focus_docs = ""
            gs = self.comp.get("get_section")
            if gs:
                # B3: the curated docs for THIS step's lever (selected by bottleneck, not step%5).
                fmap = lever.doc_sections if lever else []
                parts = []
                for fdoc, fhead in fmap:
                    try:
                        s = gs(fdoc, fhead)
                    except Exception:
                        s = ""
                    if s and not s.startswith("("):          # skip "(no section ...)" misses
                        parts.append(s[:2200])               # bound each section
                focus_docs = "\n\n".join(parts)[:4200]
            # (b) #1 scope-balanced lexical hints: guarantee a design-guide + Hopper-spec + PTX
            #     snippet for the bottleneck instead of one k=2 pull the huge PTX ISA dominates.
            dh = ""
            rt = self.comp.get("retrieve")
            if rt:
                chunks = []
                for sc in ("design-guides", "architecture-specs", "cuda-docs"):
                    try:
                        r = rt(dq, k=1, scope=sc)
                    except Exception:
                        r = ""
                    if r and not r.startswith("("):          # skip "(no relevant sections ...)"
                        chunks.append(r)
                dh = ("\n\n" + "-" * 8 + "\n").join(chunks)
            # Feed the run's OWN distilled knowledge back in -- the brain rarely queries the store
            # itself, so inject the top notes keyed to the same focus+bottleneck query.
            kn = ""
            try:
                if self.kstore:
                    kn = _knowledge_line(self.kstore.query(dq, k=3))
            except Exception:
                pass
            sass_fb = _sass_line(prev.get("sass")) if prev else ""
            profile_digest = _profile_digest(prev.get("profile")) if prev else ""
            # E2: persisted cross-step memory digest (causal lessons + dead-ends + champion state),
            # injected EVERY step so the fresh session retains what prior steps learned.
            memory_digest = self._memory.digest()
            # B3: when the lever is STRUCTURAL, soften the "make TARGETED edits" bias -- a larger
            # warp-spec/TMA/layout REWRITE is justified and expected for this bottleneck.
            structural_carveout = ""
            if lever is not None and lever.kind == "structural":
                structural_carveout = (
                    f"This step's focus is a STRUCTURAL lever ({lever.id}) for the '{bottleneck}' "
                    "bottleneck: a larger restructuring (warp-specialization / producer-consumer "
                    "split / TMA pipeline / moving O out of registers) is JUSTIFIED here — do NOT "
                    "limit yourself to a one-line tweak. Keep every config correct.")
            # C2/C3: inside the protected branch, override the bias entirely (this IS a rewrite) and
            # inject the rewrite journal so the half-built rewrite survives the fresh session.
            branch_note = ""
            if branch is not None:
                structural_carveout = (
                    f"You are in a PROTECTED REWRITE BRANCH (rung {branch.get('rung')}, seed "
                    f"{branch.get('seed_cid')}, budget {branch.get('budget')}). Acceptance is "
                    "BRANCH-LOCAL: you MAY regress vs the current trunk champion as long as every "
                    "config stays CORRECT — this is the one place a large structural rewrite is "
                    "funded. Build the rewrite incrementally; do not revert to the old structure.")
                branch_note = self._branch_journal_digest()
            prompt = prompts.refine_prompt(
                cand_path, read_path,
                last_eval_summary=(prev.get("eval", {}) or {}).get("__summary__", str(prev.get("status"))) if prev else "n/a",
                profile_feedback=bottleneck,
                profile_digest=profile_digest,
                sass_feedback=sass_fb,
                knowledge_hints=kh[:1500],
                knowledge_notes=kn,
                doc_hints=dh[:3000],
                focus_docs=focus_docs,
                focus=focus,
                memory_digest=memory_digest,
                structural_carveout=structural_carveout,
                branch_note=branch_note,
                supervisor_note=self._supervisor_note)

        brain_log = os.path.join(checkpoint.build_dir(self.run_id, cid), "brain.log")
        mode = "bootstrap" if best is None else f"refine (lever {(lever.id if lever else '?')})"
        self._log(f"[gen] invoking brain ({self.model}, effort={self.effort}) {mode}; "
                  f"streaming -> {brain_log}")
        t0 = time.time()
        # E3: give the brain read access to the LIVE workdir (MEMORY.md / rewrite_journal / sidecars)
        # via --add-dir; F4: NEVER pass a CHECKPOINT_ROOT path.  F5: isolate its Claude config dir.
        res = llm.run_claude(prompt, cwd=C.REPO_ROOT, model=self.model, effort=self.effort,
                             session_id=session_id, stream_log=brain_log,
                             allowed_dirs=[self.paths["run"]],
                             env_extra=self._brain_env(cid))
        dt = time.time() - t0
        # Per-step audit of the brain's I/O: the FULL prompt/context it was given + what it
        # returned, written to builds/stepNNN/prompt.log (the live token stream stays in the
        # sibling brain.log transcript).
        runlog.record("prompt", label=f"brain[{cid}]",
                      note=f"{mode}; transcript -> {brain_log}",
                      inputs={"model": self.model, "effort": self.effort,
                              "session_id": session_id, "cwd": C.REPO_ROOT,
                              "candidate_path": cand_path, "prompt": prompt},
                      output={"ok": res.ok, "session_id": res.session_id, "error": res.error,
                              "candidate_written": os.path.exists(cand_path), "result_text": res.text},
                      duration_s=dt, max_chars=1_000_000)
        self._log(f"[gen] brain returned in {dt:.0f}s (ok={res.ok}); "
                  f"candidate {'written' if os.path.exists(cand_path) else 'MISSING'}")
        # If the brain didn't produce a file, bootstrap from the seed (guarantees a candidate).
        if not os.path.exists(cand_path):
            if best is None and os.path.exists(os.path.join(C.PKG_DIR, "seed_kernel.cu")):
                shutil.copyfile(os.path.join(C.PKG_DIR, "seed_kernel.cu"), cand_path)
                self._log(f"[gen] brain produced no file; seeded {cid} from seed_kernel.cu")
            elif os.path.exists(read_path):
                shutil.copyfile(read_path, cand_path)
                self._log(f"[gen] brain produced no file; reusing {os.path.basename(read_path)} for {cid}")
            else:
                return cand_path, (res.session_id or session_id)
        return cand_path, (res.session_id or session_id)

    # ---- evaluation ---------------------------------------------------------------------
    def _evaluate(self, step: int, cid: str, ns: str, so_path: str,
                  uses_wgmma: Optional[bool] = None,
                  precomputed_cres: Optional[list] = None) -> tuple[EvalResult, dict]:
        ev = EvalResult(num_configs=len(self.cfgs))
        profile = {}
        # CORRECTNESS FIRST.  Reuse the in-step completion-gate's correctness result when present
        # (it already ran the same synthetic check on this exact build) instead of re-running it.
        if precomputed_cres is not None:
            cres = precomputed_cres
            self._log(f"[correctness] reusing completion-gate result "
                      f"({sum(1 for r in cres if r.correct)}/{len(self.cfgs)} passed)")
        else:
            self._log(f"[correctness] checking {len(self.cfgs)} configs (synthetic pre-filter)...")
            try:
                cres = self.comp["check_correctness"](ns, self.cfgs, seed=self.seed, gpu_id=self.pool.gpu_ids[0])
            except Exception as e:
                ev.error = f"correctness raised: {e!r}"
                return ev, profile
        ev.per_config = cres
        ev.num_correct = sum(1 for r in cres if r.correct)
        ev.correct = ev.num_correct == len(self.cfgs)
        if not ev.correct:
            bad = [r.config for r in cres if not r.correct][:6]
            self._log(f"[correctness] {ev.num_correct}/{len(self.cfgs)} passed; failing: {bad}")
            return ev, profile  # never profile/perf an incorrect kernel
        self._log(f"[correctness] {ev.num_correct}/{len(self.cfgs)} passed")

        # PERF: measure each config in its OWN process pinned to one GPU (1 kernel/GPU), up to
        # len(gpu_ids) concurrent. Process isolation (not threads) avoids the GIL/host-launch
        # contention that inflated the baseline/candidate ratio ~2-3x in the old threaded sweep.
        if self.comp.get("measure"):
            self._log(f"[perf] measuring {len(self.cfgs)} configs (1 process/GPU, isolated timing, "
                      f"{len(self.pool.gpu_ids)} GPUs)...")
            from fa3agent.perf.measure import measure_all_subprocess

            def _on(r, gpu):
                self._log(f"[perf]   {r.config}: ratio={r.ratio:.3f} "
                          f"(cand {r.cand_latency_ms:.3f}ms / base {r.base_latency_ms:.3f}ms) gpu{gpu}")
            perf = measure_all_subprocess(ns, self.cfgs, so_path, self.pool.gpu_ids, on_result=_on)
            by = {r.config: r for r in perf if r}
            for r in ev.per_config:
                if r.config in by:
                    r.ratio = by[r.config].ratio
                    r.cand_latency_ms = by[r.config].cand_latency_ms
                    r.base_latency_ms = by[r.config].base_latency_ms
            ratios = [r.ratio for r in ev.per_config]
            ev.min_ratio = min([x for x in ratios if x > 0], default=0.0)
            ev.geomean_ratio = _geomean(ratios)
            # Proper weighted geomean (handles fractional weights; the old int()-duplication both
            # truncated fractional weights and dropped weight<1 configs entirely).
            ev.weighted_geomean_ratio = promotion.weighted_geomean(
                ratios, [c.weight for c in self.cfgs])

        # PROFILE the worst config (guides next optimization) — OUT-OF-PROCESS (torch-free).
        if self.profile_enabled and ev.per_config:
            worst = min(ev.per_config, key=lambda r: (r.ratio if r.ratio > 0 else 1e9))
            wcfg = next((c for c in self.cfgs if c.name == worst.config), self.cfgs[0])
            gpu = self.pool.gpu_ids[0]
            self._log(f"[ncu] profiling worst config '{wcfg.name}' on gpu{gpu} (out-of-process)...")
            with self.pool.exclusive(gpu):
                profile = self._profile_subprocess(so_path, ns, wcfg.name, gpu, uses_wgmma=uses_wgmma)
            # Tag the profile with the worst config's identity/dtype so the lever selector's fp8
            # foundation gate can see the min-ratio config is fp8 (e4m3) without re-deriving it.
            if isinstance(profile, dict):
                profile["worst_config"] = wcfg.name
                profile["worst_dtype"] = wcfg.dtype
            self._log(f"[ncu] bottleneck: {(profile or {}).get('bottleneck','(none)')[:140]}")

        # NOTE: the faithful vLLM gates (in-situ A/B, then full decode-equality) are run at the
        # PROMOTION tier in run(), not here -- they need a model load and only matter for a
        # candidate that is correct AND improves the best ratio (cost-tiered verification).
        return ev, profile

    # ---- build + scan (one attempt) -----------------------------------------------------
    def _build_and_scan(self, cid: str, ns: str, cand_path: str, rec: StepRecord):
        """anti-cheat(source) -> build -> anti-cheat(binary) -> SASS for ONE candidate file.

        Returns (phase, BuildResult|None) where phase is one of:
          'cheat'        — source or binary anti-cheat rejected (sets rec.anticheat);
          'env_fatal'    — build failed, unfixable-by-agent;
          'build_failed' — build failed (fixable);
          'built'        — compiled OK, SASS analyzed (sets rec.build, rec.sass).
        Used both for the single legacy pass and for each turn of the in-step completion loop, so
        re-engaging the brain just re-runs this on the rewritten candidate."""
        if self.comp.get("anticheat"):
            ac: AntiCheatResult = self.comp["anticheat"](cand_path, None)
            rec.anticheat = asdict(ac)
            if not ac.ok:
                self._log(f"[anticheat] REJECT: {ac.violations}")
                return "cheat", None

        bdir = checkpoint.build_dir(self.run_id, cid)
        self._log(f"[build] compiling {cid} (toolkit-only, sm_90a)...")
        br: BuildResult = self.comp["build_candidate"](cand_path, ns, bdir)
        rec.build = asdict(br)
        if not br.ok:
            self._log(f"[build] FAILED ({br.duration_s:.0f}s): {br.error[:300]}")
            return ("env_fatal" if br.env_fatal else "build_failed"), br
        self._log(f"[build] ok ({br.duration_s:.0f}s) -> {os.path.basename(br.so_path or '')}")

        if self.comp.get("anticheat"):
            ac2 = self.comp["anticheat"](cand_path, br.so_path)
            rec.anticheat = asdict(ac2)
            if not ac2.ok:
                self._log(f"[anticheat] REJECT (binary): {ac2.violations}")
                return "cheat", br

        # SASS analysis of the built kernel (its own instruction mix: wgmma/TMA presence, register
        # spills) -- run once per successful build, independent of correctness; distilled into the
        # next refine prompt.  cuobjdump-only (no GPU); bounded inside sass.analyze.
        if self.comp.get("sass_analyze") and br.so_path:
            try:
                rec.sass = self.comp["sass_analyze"](br.so_path, ns)
            except Exception as e:  # noqa - best-effort
                rec.sass = {"ok": False, "error": repr(e)}
            sl = _sass_line(rec.sass)
            if sl:
                self._log(f"[sass] {sl[:150]}")
        return "built", br

    def _attach_hot_sass(self, rec: StepRecord, so_path: Optional[str], ns: str,
                         profile: Optional[dict]) -> None:
        """Re-run SASS targeting EXACTLY the kernel(s) NCU found dominant for the worst config, and
        store the verdict on rec.sass_hot.  This is the hot-path signal the next step's lever
        foundation gate reads (via _sass_flags): 'is the BOTTLENECK kernel on tensor cores?' rather
        than 'does any kernel in the .so contain wgmma?'.  Best-effort; leaves sass_hot None (so
        _sass_flags falls back to the whole-binary verdict) when NCU named no kernel."""
        if not (self.comp.get("sass_analyze") and so_path and isinstance(profile, dict)):
            return
        km = profile.get("key_metrics") or {}
        names: list[str] = []
        kn = km.get("_kernel")
        if kn:
            names.append(kn)
        for k in (km.get("_kernels") or []):
            n = k.get("name") if isinstance(k, dict) else None
            if n and n not in names:
                names.append(n)
        names = [n for n in names if n and n != "?"]
        if not names:
            return
        try:
            hot = self.comp["sass_analyze"](so_path, ns, targets=names)
        except Exception as e:  # noqa - best-effort
            hot = {"ok": False, "error": repr(e)}
        rec.sass_hot = hot
        sl = _sass_line(hot)
        if sl:
            kn0 = names[0].split("<", 1)[0].split("(", 1)[0]
            self._log(f"[sass:hot] dominant kernel ({kn0[:48]}) verdict: {sl[:130]}")

    def _check_correctness_only(self, ns: str):
        """Run just the synthetic correctness check (no perf/profile) for the completion gate."""
        if not self.comp.get("check_correctness"):
            return None
        self._log(f"[completion] checking correctness ({len(self.cfgs)} configs)...")
        try:
            return self.comp["check_correctness"](ns, self.cfgs, seed=self.seed,
                                                  gpu_id=self.pool.gpu_ids[0])
        except Exception as e:  # noqa
            self._log(f"[completion] correctness check raised: {e!r}")
            return None

    def _oracle_cfg(self):
        """Representative config for the oracle build (drives the wgmma N-tile via head_dim).  Prefer
        a head_dim=128 config (the dominant decode/prefill shape); else the first available config."""
        for c in (self.cfgs or []):
            if getattr(c, "head_dim", None) == 128:
                return c
        return (self.cfgs or [None])[0]

    def _oracle_ptx_ref(self, want: str) -> str:
        """A literal Hopper PTX reference for a still-MISSING instruction class (uses_tma /
        uses_wgmma / uses_fp8_wgmma), sliced from a CUTLASS / inline oracle built ONCE out of band.

        This is the concrete sequence (descriptor + mbarrier expect_tx + cp.async.bulk.tensor +
        wgmma.fence + wgmma.mma_async + commit/wait, in order) the brain should MIRROR -- handed to it
        by the completion gate exactly when the gate proves the capability is absent.  Cached per
        head_dim so it never rebuilds inside the per-step loop; '' when no oracle is available."""
        builder = self.comp.get("build_oracle")
        if not builder or not want:
            return ""
        cfg = self._oracle_cfg()
        key = getattr(cfg, "head_dim", 128)
        if key not in self._oracle_refs:
            wd = os.path.join(self.paths["run"], "builds", "_oracle")
            try:
                res = builder(cfg, wd)
                ref = res.get("ptx_ref") or ""
                self._oracle_refs[key] = ref
                self._log(f"[oracle] tier={res.get('oracle')} hd={key} "
                          f"ptx_ref={'yes' if ref else 'no'}: {(res.get('note') or '')[:80]}")
            except Exception as e:  # noqa - best-effort, never crash a step
                self._oracle_refs[key] = ""
                self._log(f"[oracle] build failed: {e!r}")
        return self._oracle_refs.get(key, "")

    def _complete_followup(self, cid: str, cand_path: str, lever, want: str, gaps: list,
                           session_id: Optional[str], attempt: int, ptx_ref: str = "") -> Optional[str]:
        """Re-engage the brain to FINISH an incomplete introducing-step before it is committed.

        By default a FRESH session (the relay chokes on real `--resume`, see HEADLESS_RESUME_SESSION)
        whose prompt re-reads the in-progress candidate + the precise gaps -- functionally the brain
        continues its own just-written work.  When HEADLESS_RESUME_SESSION is on, the prior session
        is resumed for true continuity.  Returns the (possibly new) session id."""
        technique = levers.VERIFY_LABELS.get(want, want)
        focus = lever.focus if lever else ""
        prompt = prompts.completion_followup(cand_path, focus, technique, gaps, ptx_ref=ptx_ref)
        brain_log = os.path.join(checkpoint.build_dir(self.run_id, cid), f"brain_complete{attempt}.log")
        sid = session_id if C.HEADLESS_RESUME_SESSION else None
        self._log(f"[completion] invoking brain ({self.model}, effort={self.effort}) "
                  f"resume={'yes' if sid else 'fresh'}; streaming -> {brain_log}")
        t0 = time.time()
        res = llm.run_claude(prompt, cwd=C.REPO_ROOT, model=self.model, effort=self.effort,
                             session_id=sid, stream_log=brain_log,
                             allowed_dirs=[self.paths["run"]],
                             env_extra=self._brain_env(cid))
        dt = time.time() - t0
        runlog.record("prompt", label=f"complete[{cid}#{attempt}]",
                      note=f"completion follow-up; transcript -> {brain_log}",
                      inputs={"model": self.model, "effort": self.effort, "session_id": sid,
                              "candidate_path": cand_path, "gaps": gaps, "prompt": prompt},
                      output={"ok": res.ok, "session_id": res.session_id, "error": res.error,
                              "candidate_written": os.path.exists(cand_path), "result_text": res.text},
                      duration_s=dt, max_chars=1_000_000)
        self._log(f"[completion] brain returned in {dt:.0f}s (ok={res.ok})")
        return res.session_id or session_id

    # ---- one step -----------------------------------------------------------------------
    def _step(self, step: int, prev: Optional[dict], best: Optional[dict],
              branch: Optional[dict] = None, seed_path: Optional[str] = None) -> StepRecord:
        cid = f"step{step:03d}"
        ns = self._ns(step)
        # Per-tool call logs for this step land in builds/stepNNN/<tool>.log (one dir per step,
        # never a cross-step file).  Exported via env so the brain's own `python -c` tool calls
        # and the measure/profile/verify subprocesses (which inherit this environment) write to
        # the SAME per-step dir.  The general narrative stays on stdout -> run.log.
        runlog.set_dir(checkpoint.build_dir(self.run_id, cid))
        runlog.set_step(step)
        self._log(f"=== STEP {step} ({cid}, ns={ns}) ===")
        lever = None
        lever_id = None
        src_rec = None   # the kernel this step builds FROM (champion / branch tip); its SASS is the
        # "before" state the in-step completion gate compares against to decide "new functionality".
        if branch is not None:
            # Inside the protected branch: rung-0 is a behavior-preserving refactor (D1); later
            # rungs walk the transform catalog keyed by the branch bottleneck (D2).
            rung = branch.get("rung", 0)
            focus_lever = branch.get("focus_lever")
            if focus_lever:
                # Layer 4 CAPABILITY branch: every step develops the SAME all-or-nothing lever (e.g.
                # tma-load) with the completion gate + oracle, seeded from the champion.  Bypasses the
                # rung-0 refactor + catalog walk so the budget is spent landing the one capability.
                lever = levers.lever_by_id(focus_lever)
                lever_id = focus_lever
                src = prev if (isinstance(prev, dict) and prev.get("sass")) else best
                src_rec = src
                focus = (f"{lever.focus}  [NCU success criterion: {lever.success}]" if lever
                         else prompts.BRANCH_REFACTOR_FOCUS)
                self._log(f"[branch] capability-dev rung {rung}: focus_lever={focus_lever} "
                          f"({lever.kind if lever else '?'})")
            elif rung == 0 and not branch.get("escalated"):
                focus = prompts.BRANCH_REFACTOR_FOCUS
                lever_id = "refactor-rung0"
                self._log(f"[branch] rung 0 refactor (seed {branch.get('seed_cid')})")
            else:
                label = _label_of((prev or {}).get("profile")) or _label_of(best.get("profile"))
                # B3b: SASS of the branch tip (prev) if known, else the seed/champion (best).
                src = prev if (isinstance(prev, dict) and prev.get("sass")) else best
                src_rec = src
                uses_tma, uses_wgmma, uses_fp8_wgmma = _sass_flags(src)
                worst_dtype = (((src or {}).get("profile")) or {}).get("worst_dtype")
                lever = self._levers.select(label, step, uses_tma=uses_tma, uses_wgmma=uses_wgmma,
                                            uses_fp8_wgmma=uses_fp8_wgmma, worst_dtype=worst_dtype)
                lever_id = lever.id
                self._log(f"[branch] rung {rung} D2 transform: bottleneck={label or '(none)'} "
                          f"-> {lever.id} ({lever.kind})")
                focus = f"{lever.focus}  [NCU success criterion: {lever.success}]"
        elif best is not None:
            # B3: pick the lever from the CHAMPION's bottleneck label (not step%5); skip exhausted.
            # B3b: gate on the champion's SASS (wgmma+TMA presence) so an overlap-bound kernel that
            # still lacks TMA lands the TMA pipeline before warp-specialization.
            label = _label_of(best.get("profile"))
            src_rec = best
            uses_tma, uses_wgmma, uses_fp8_wgmma = _sass_flags(best)
            worst_dtype = (best.get("profile") or {}).get("worst_dtype")
            lever = self._levers.select(label, step, uses_tma=uses_tma, uses_wgmma=uses_wgmma,
                                        uses_fp8_wgmma=uses_fp8_wgmma, worst_dtype=worst_dtype)
            lever_id = lever.id
            self._log(f"[lever] bottleneck={label or '(none)'} -> {lever.id} ({lever.kind})")
            focus = f"{lever.focus}  [NCU success criterion: {lever.success}]"
        else:
            focus = "(bootstrap)"
        rec = StepRecord(step=step, cid=cid, parent=(best.get("cid") if best else None),
                         status="generated", focus=focus, lever_id=lever_id)

        cand_path, session_id = self._generate(step, prev, best, lever, focus=focus,
                                               best_path_override=seed_path, branch=branch)
        rec.session_id = session_id
        if not os.path.exists(cand_path):
            rec.status = "no_candidate"
            return rec

        # In-step COMPLETION GATE (config.COMPLETION_GATE): when this step's lever is meant to
        # INTRODUCE a new functionality (it carries a SASS `verify` signal AND the source kernel
        # lacks it), verify the BUILT candidate actually adopts it + stays correct; if not, re-engage
        # the brain on the SAME file to finish the change before the step is committed.  General
        # over every such lever (no per-optimization rule) -- trigger + check come from metadata.
        want = (lever.verify if lever else "") or ""
        src_flag = None
        if want:
            _st, _sw, _sf = _sass_flags(src_rec)
            src_flag = {"uses_tma": _st, "uses_wgmma": _sw, "uses_fp8_wgmma": _sf}.get(want)
        # "introducing" iff the source did NOT already have the signal (False or unknown -> verify;
        # only an explicit True means it is a refinement of an existing path -> skip the gate).
        check_adoption = bool(C.COMPLETION_GATE and want and lever is not None and src_flag is not True)

        cres = None            # completion-gate correctness result, reused by _evaluate when set
        attempt = 0
        # Each in-process build/load within ONE step MUST use a distinct op namespace: build_candidate
        # loads the .so in-process to read its schema, and reloading a second .so under the SAME
        # TORCH_LIBRARY namespace re-runs registration during dlopen -> std::terminate/SIGABRT (no
        # Python traceback).  So a completion-fix rebuild gets ns_try = <ns>_c<attempt>; the FINAL
        # ns_try is recorded on rec.build and threaded to eval + the run()-level gates via _built_ns.
        ns_try = ns
        while True:
            ns_try = ns if attempt == 0 else f"{ns}_c{attempt}"
            phase, br = self._build_and_scan(cid, ns_try, cand_path, rec)
            if phase == "cheat":
                rec.status = "cheat"
                self._distill(rec, f"Anti-cheat rejected: {(rec.anticheat or {}).get('violations')}",
                              ntype="alg")
                return rec
            if phase == "env_fatal":
                rec.status = "env_fatal"
                self._distill(rec, f"Build failed (env): {(br.error if br else '')[:400]}", ntype="ks")
                return rec
            if not check_adoption:
                break    # not an introducing step -> single pass (legacy behavior)

            # What is still missing for this step to have fully/correctly landed its functionality?
            gaps: list[str] = []
            cres = None
            instr_missing = False
            if phase == "build_failed":
                gaps.append(f"BUILD FAILED — the new code does not compile: {(br.error if br else '')[:350]}")
            else:  # built ok
                if not _sass_has(rec.sass, want):
                    instr_missing = True
                    gaps.append(f"{levers.VERIFY_LABELS.get(want, want)} — but SASS shows it is NOT "
                                f"present, so the intended path was not actually taken")
                cres = self._check_correctness_only(ns_try)
                if cres is not None:
                    bad = [r.config for r in cres if not r.correct]
                    if bad:
                        gaps.append(f"correctness regressed — wrong on configs: {bad[:6]}")
            if not gaps:
                self._log(f"[completion] '{lever.id}' landed cleanly "
                          f"(attempt {attempt}): {want}=True, correct")
                break
            if attempt >= C.MAX_COMPLETION_ATTEMPTS or not session_id:
                self._log(f"[completion] '{lever.id}' still incomplete after {attempt} fix "
                          f"attempt(s); committing as-is. gaps={gaps}")
                break
            attempt += 1
            # When the instruction class is genuinely absent from the SASS, hand the brain a concrete
            # Hopper PTX reference (CUTLASS/inline oracle) for the exact wgmma/TMA sequence it must
            # emit -- this is the support the brain lacked when it kept planning TMA but never wrote it.
            ptx_ref = self._oracle_ptx_ref(want) if instr_missing else ""
            self._log(f"[completion] attempt {attempt}/{C.MAX_COMPLETION_ATTEMPTS}: re-engaging brain "
                      f"to finish '{lever.id}'. gaps={gaps}"
                      + ("  [+oracle PTX ref]" if ptx_ref else ""))
            session_id = self._complete_followup(cid, cand_path, lever, want, gaps, session_id,
                                                 attempt, ptx_ref=ptx_ref)
            rec.session_id = session_id
            rec.completion_attempts = attempt
            if not os.path.exists(cand_path):
                self._log("[completion] candidate file vanished after follow-up; committing prior build")
                break

        # Couldn't get a buildable candidate (introducing step exhausted its budget still failing,
        # or a non-introducing build failure) -> same terminal handling as before.
        if phase in ("build_failed", "env_fatal"):
            rec.status = "build_failed" if phase == "build_failed" else "env_fatal"
            self._distill(rec, f"Build failed: {(br.error if br else '')[:400]}", ntype="ks")
            return rec

        # SASS ground truth (HGMMA/QGMMA present) -> steers the NCU bottleneck classifier (B2).
        uses_wgmma = None
        if isinstance(rec.sass, dict) and rec.sass.get("ok"):
            uses_wgmma = (rec.sass.get("summary") or {}).get("uses_wgmma")
        ev, profile = self._evaluate(step, cid, ns_try, br.so_path, uses_wgmma=uses_wgmma,
                                     precomputed_cres=cres)
        # Hot-path SASS verdict: re-analyze EXACTLY the kernel(s) NCU found dominant for the worst
        # config, so the next step's lever foundation gate (via _sass_flags) sees whether the
        # BOTTLENECK kernel is on tensor cores -- not merely whether ANY kernel in the .so has wgmma.
        self._attach_hot_sass(rec, br.so_path, ns_try, profile)

        # Sanitizer (memcheck) ONLY when a built kernel fails correctness in a way that smells like a
        # memory bug (raised / NaN-inf / illegal / shape) -- a bounded report instead of leaving the
        # brain to run raw compute-sanitizer.
        san = None
        if (self.comp.get("run_sanitizer") and br.so_path and not ev.correct
                and _looks_like_mem_bug(ev)):
            cfg0 = min(self.cfgs, key=lambda c: sum(q * k for q, k in c.requests) * c.num_heads_q)
            self._log(f"[sanitizer] suspected memory bug -> memcheck on {cfg0.name}...")
            try:
                with self.pool.exclusive(self.pool.gpu_ids[0]):
                    san = self.comp["run_sanitizer"](br.so_path, ns, cfg0, tool="memcheck")
                self._log(f"[sanitizer] clean={san.get('clean')} errors={san.get('num_errors')}: "
                          f"{((san.get('errors') or ['-'])[0] or '')[:80]}")
            except Exception as e:  # noqa
                san = {"clean": None, "note": f"sanitizer error: {e!r}"}

        ev_d = asdict(ev); ev_d["__summary__"] = ev.summary()
        if san is not None:
            ev_d["sanitizer"] = san
            ev_d["__summary__"] += (f" | sanitizer(memcheck): clean={san.get('clean')} "
                                    f"n_err={san.get('num_errors')}")
        rec.eval = ev_d
        rec.profile = profile
        rec.status = "correct" if ev.correct else "incorrect"
        self._log(f"[eval] {ev.summary()}")

        # distill knowledge from the outcome (incl. negative knowledge) -> E1 causal lesson note
        self._distill_step(rec, ev, profile, best)
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

    def _causal_lesson(self, rec: StepRecord, champ_before: Optional[dict],
                       promoted: bool) -> tuple[dict, str]:
        """Author a causal lesson (E1): what CHANGED -> hypothesis -> outcome DELTA -> corrected
        cause/next move.  Pure over `rec` + the champion-before-this-step, so it is usable both from
        `_distill_step` (provisional promote) and `run()` (final outcome -> MEMORY.md)."""
        focus = rec.focus or "(unspecified change)"
        prof = rec.profile or {}
        new_bn = prof.get("bottleneck", "") or ""
        champ = champ_before or {}
        champ_eval = champ.get("eval", {}) or {}
        prior_bn = (champ.get("profile", {}) or {}).get("bottleneck", "") or ""

        change = "initial seed kernel" if not champ_before else f"applied: {focus}"
        # Honest accounting (Layer 5): annotate with the MEASURED adoption so memory reflects what the
        # build actually did, not the lever's intent -- otherwise a step that "applied ... TMA" but
        # emitted zero UTMALDG gets recorded as success and the brain believes TMA is already done.
        lv = levers.lever_by_id(rec.lever_id) if getattr(rec, "lever_id", None) else None
        if champ_before and lv is not None and lv.verify:
            adopted = _sass_has(rec.sass_hot, lv.verify) or _sass_has(rec.sass, lv.verify)
            change += f"  [SASS: {lv.verify} {'present' if adopted else 'STILL ABSENT'}]"
        hypothesis = (f"relieve prior bottleneck: {prior_bn}" if prior_bn
                      else "establish a correct, faster baseline")
        status = rec.status
        ev = rec.eval or {}
        cause = ""
        if status == "correct":
            f_new = ev.get("weighted_geomean_ratio", 0.0)
            m_new = ev.get("min_ratio", 0.0)
            if champ_before:
                f_old = champ_eval.get("weighted_geomean_ratio", 0.0)
                m_old = champ_eval.get("min_ratio", 0.0)
                outcome = (f"correct; F {f_old:.3f}->{f_new:.3f} ({f_new - f_old:+.3f}), "
                           f"min_ratio {m_old:.3f}->{m_new:.3f}; "
                           f"{'PROMOTED' if promoted else 'rejected (did not clear bar)'}")
            else:
                outcome = (f"correct; F={f_new:.3f} min_ratio={m_new:.3f}; "
                           f"{'PROMOTED' if promoted else 'rejected'}")
            cause = (f"new bottleneck: {new_bn}" if (promoted and new_bn)
                     else _next_hint(new_bn or prior_bn))
        elif status == "incorrect":
            bad = [r.get("config") for r in (ev.get("per_config") or [])
                   if isinstance(r, dict) and not r.get("correct")][:4]
            outcome = f"INCORRECT (broke correctness on {bad})"
            cause = ev.get("error") or "diverged vs FA3; see per-config err_ratio/lse_err"
        elif status in ("build_failed", "env_fatal"):
            outcome = "build failed"
            cause = ((rec.build or {}).get("error") or "")[:200]
        elif status == "cheat":
            outcome = "anti-cheat rejected"
            cause = str((rec.anticheat or {}).get("violations"))[:200]
        elif status in ("insitu_fail", "cudagraph_fail"):
            outcome = f"{status} (correct on synthetic, failed the faithful vLLM gate)"
            cause = "not a valid drop-in; see gate note in eval"
        else:
            outcome = status or "no outcome"
        lesson = {"step": rec.step, "cid": rec.cid, "change": change, "hypothesis": hypothesis,
                  "outcome": outcome, "cause": cause, "promoted": promoted}
        note = f"{change} | {hypothesis} | {outcome}" + (f" | {cause}" if cause else "")
        return lesson, note

    def _distill_step(self, rec: StepRecord, ev: EvalResult, profile: dict,
                      champ_before: Optional[dict]):
        # E1: record a causal lesson as the knowledge note (replaces the old "Correct candidate.
        # min_ratio=..." boilerplate).  `promoted` here is PROVISIONAL (the cost-tier gates in run()
        # can still demote a correct candidate); run() re-authors the final lesson into MEMORY.md.
        # Under STRICT_PROGRESS (accounting fix C) the provisional "promoted" note must match run()'s
        # real-improvement test, not the looser bar -- otherwise a within-noise hold reads "PROMOTED"
        # in the in-step note while run() correctly treats it as no-progress.
        _progress = (promotion.is_real_improvement if C.PROMOTION_STRICT_PROGRESS
                     else promotion.meets_promotion_bar)
        promoted = bool(ev.correct and rec.eval and _progress(
            promotion.objective_score(rec.eval, C.PROMOTION_OBJECTIVE), self._champ_score))
        _, note = self._causal_lesson(rec, champ_before, promoted)
        self._distill(rec, note, ntype=("pf" if ev.correct else "ks"))

    def _record_memory(self, rec: StepRecord, champ_before: Optional[dict], improved: bool):
        """E2/B3: record the lever outcome, append the FINAL causal lesson, refresh MEMORY.md."""
        try:
            # B3: score the lever applied this step (landed / regressed / no_effect / failed) and
            # update its persisted ledger (which also drives C2's structural-stall STUCK trigger).
            if rec.lever_id:
                correct = rec.status == "correct"
                dF = None
                if rec.eval and champ_before:
                    dF = (rec.eval.get("weighted_geomean_ratio", 0.0)
                          - (champ_before.get("eval", {}) or {}).get("weighted_geomean_ratio", 0.0))
                outcome = self._levers.classify_outcome(improved, correct, dF)
                # Adoption-aware exhaustion: for a foundation lever (wgmma/TMA/fp8), tell the ledger
                # whether its instruction class actually landed in the SASS.  A "selected but never
                # emitted" step (the brain substituted an easier path) must NOT exhaust the capability
                # the way TMA was -- record_outcome treats adopted=False as not-implemented, not as a
                # technique failure.  adopted stays None when there's no verify signal / no readable SASS.
                adopted = None
                lv = levers.lever_by_id(rec.lever_id)
                if lv is not None and lv.verify:
                    if _sass_has(rec.sass_hot, lv.verify) or _sass_has(rec.sass, lv.verify):
                        adopted = True
                    elif ((isinstance(rec.sass_hot, dict) and rec.sass_hot.get("ok"))
                          or (isinstance(rec.sass, dict) and rec.sass.get("ok"))):
                        adopted = False     # SASS readable and the instruction class is absent
                self._levers.record_outcome(rec.lever_id, outcome, step=rec.step, adopted=adopted)
                self._memory.set_lever_ledger(self._levers.ledger_summary())
            lesson, _ = self._causal_lesson(rec, champ_before, improved)
            self._memory.add_lesson(lesson)
            if improved and rec.eval:
                self._memory.set_champion(
                    rec.cid, rec.step, rec.eval.get("weighted_geomean_ratio", 0.0),
                    rec.eval.get("min_ratio", 0.0),
                    bottleneck=(rec.profile or {}).get("bottleneck"))
            else:
                self._memory.set_bottleneck((champ_before or {}).get("profile", {}).get("bottleneck")
                                            if champ_before else None)
            # negative memory: a correct-but-rejected attempt (or a faithful-gate failure) is a
            # dead-end for that lever -- record it so a fresh session won't retry it the same way.
            if rec.status == "correct" and not improved and rec.focus:
                self._memory.add_dead_end(
                    f"{rec.focus[:120]} -> rejected (did not clear the promotion bar; bottleneck "
                    f"stayed {(rec.profile or {}).get('bottleneck', '?')})")
            elif rec.status in ("insitu_fail", "cudagraph_fail") and rec.focus:
                self._memory.add_dead_end(f"{rec.focus[:120]} -> {rec.status}")
            self._memory.commit()
        except Exception as e:  # memory is best-effort
            self._log(f"[warn] memory update failed: {e!r}")

    # ---- C2: plateau / protected rewrite branch ----------------------------------------
    def _loc_of(self, cid: Optional[str]) -> int:
        """Champion .cu line count (entanglement proxy for the knee)."""
        if not cid:
            return 0
        p = os.path.join(self.paths["candidates"], f"{cid}.cu")
        try:
            with open(p) as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 0

    def _candidate_path(self, cid: Optional[str]) -> Optional[str]:
        """Path to a candidate .cu (the branch seed/tip), with sensible fallbacks."""
        if cid:
            p = os.path.join(self.paths["candidates"], f"{cid}.cu")
            if os.path.exists(p):
                return p
        if os.path.exists(self.paths["best"]):
            return self.paths["best"]
        seed = os.path.join(C.PKG_DIR, "seed_kernel.cu")
        return seed if os.path.exists(seed) else None

    def _find_step_rec(self, cid: str) -> Optional[dict]:
        for s in checkpoint.load_steps(self.run_id):
            if s.get("cid") == cid:
                return s
        return None

    def _resolve_branch_seed(self, best: Optional[dict]) -> str:
        """BRANCH_SEED (D1 will extend): default pre_plateau = the C2 knee; champion / seed otherwise."""
        mode = C.BRANCH_SEED
        if mode == "champion" and best:
            return best.get("cid") or "step000"
        if mode == "seed":
            return "step000"
        return self._plateau.knee_cid("step000")   # pre_plateau (default)

    def _open_branch(self, best: Optional[dict], reasons: list):
        anchor_cid = best.get("cid") if best else None
        anchor_F = (best.get("eval", {}) or {}).get("weighted_geomean_ratio") if best else None
        seed_cid = self._resolve_branch_seed(best)
        b = self._plateau.open_branch(anchor_cid, anchor_F, seed_cid, budget=C.BRANCH_BUDGET)
        self._log(f"[branch] STUCK ({','.join(reasons)}) -> opening protected rewrite branch: "
                  f"anchor={anchor_cid} (F={anchor_F}), seed={seed_cid}, budget={b['budget']}")
        self._memory.add_lesson({
            "step": -1, "cid": seed_cid, "change": f"[branch OPEN] seed={seed_cid}",
            "hypothesis": f"STUCK ({', '.join(reasons)}); restructure from the pre-plateau knee "
                          "where trunk micro-opts can't (warp-spec / TMA / O-out-of-registers)",
            "outcome": "rewrite branch opened (branch-local acceptance; deferred adopt-or-revert)",
            "cause": "trunk structural levers exhausted", "promoted": False})
        self._init_branch_journal(b, reasons)
        self._memory.commit()

    def _capability_branch_target(self, rec: StepRecord) -> Optional[str]:
        """Layer 4 trigger: if this trunk step just FAILED to land an all-or-nothing capability, return
        the lever id to fund through a dedicated branch -- else None.

        Fires only when (a) the step's lever is a `branch_dev` rewrite (tma-load / persistent / cluster
        multicast), (b) its instruction class is STILL absent from the built SASS, and (c) the trunk has
        already spent its in-step budget on it (the lever is exhausted -- i.e. NOT_IMPL_CAP in-step
        completion-gate attempts failed).  At that point the single-step trunk can't add it, so escalate
        to a multi-step, branch-local-acceptance rewrite instead of leaving it permanently absent."""
        lid = rec.lever_id
        if not lid or not levers.is_branch_dev(lid):
            return None
        lv = levers.lever_by_id(lid)
        if lv and lv.verify and (_sass_has(rec.sass_hot, lv.verify) or _sass_has(rec.sass, lv.verify)):
            return None    # capability actually present -> nothing to fund
        if not self._levers.is_exhausted(lid):
            return None    # trunk still has in-step budget to try it -> don't escalate yet
        return lid

    def _open_capability_branch(self, best: Optional[dict], lever_id: str):
        """Open a CAPABILITY rewrite branch (Layer 4): seeded from the CHAMPION (we ADD the capability
        to the best kernel, not restructure from the knee), focused on one all-or-nothing lever, with
        branch-local acceptance over BRANCH_BUDGET steps then adopt-or-revert."""
        anchor_cid = best.get("cid") if best else None
        anchor_F = (best.get("eval", {}) or {}).get("weighted_geomean_ratio") if best else None
        seed_cid = anchor_cid or self._resolve_branch_seed(best)
        lv = levers.lever_by_id(lever_id)
        b = self._plateau.open_branch(anchor_cid, anchor_F, seed_cid,
                                      budget=C.BRANCH_BUDGET, focus_lever=lever_id)
        self._log(f"[branch] CAPABILITY-FUND '{lever_id}' (trunk could not land {lv.verify if lv else '?'} "
                  f"in-step) -> champion-seeded branch from {seed_cid}, budget={b['budget']}")
        self._memory.add_lesson({
            "step": -1, "cid": seed_cid, "change": f"[branch OPEN: capability {lever_id}]",
            "hypothesis": f"trunk repeatedly failed to land {lever_id} ({lv.verify if lv else '?'}) "
                          "in a single step; fund it over multiple steps with branch-local acceptance",
            "outcome": "capability rewrite branch opened (regressions tolerated; must stay correct)",
            "cause": "all-or-nothing structural change rejected by the single-step promotion bar",
            "promoted": False})
        self._init_branch_journal(b, [f"capability:{lever_id}"])
        self._memory.commit()

    def _branch_gate_tip(self, so_path: Optional[str], cid: str) -> bool:
        """Run the faithful vLLM gates on the branch tip at adopt time (cuda-graph + in-situ).
        Missing components / skips do NOT block (mirrors the trunk gate semantics)."""
        if not so_path or not os.path.exists(so_path):
            return False
        ns = self._built_ns(self._find_step_rec(cid)) or self._ns_cid(cid)
        gpu = self.pool.gpu_ids[0]
        try:
            if self.comp.get("check_cuda_graph"):
                with self.pool.exclusive(gpu):
                    cg = self.comp["check_cuda_graph"](so_path, ns)
                if cg.get("ok") is False and not cg.get("skip"):
                    self._log(f"[branch] tip {cid} fails cuda-graph gate -> not adoptable")
                    return False
            if self.do_decode_equiv and self.comp.get("check_insitu"):
                with self.pool.exclusive(gpu):
                    ir = self.comp["check_insitu"](so_path, ns)
                if ir.get("ok") is False and not ir.get("skip"):
                    self._log(f"[branch] tip {cid} fails in-situ gate -> not adoptable")
                    return False
        except Exception as e:  # noqa - gate failure should not crash the loop
            self._log(f"[branch] tip gate raised: {e!r} -> treating as non-adoptable")
            return False
        return True

    def _ns_cid(self, cid: str) -> str:
        try:
            return self._ns(int(cid.replace("step", "")))
        except Exception:
            return f"{C.DISTILL_NAMESPACE}_{cid}"

    def _built_ns(self, rec) -> Optional[str]:
        """The op namespace the candidate .so was ACTUALLY compiled with -- it may differ from the
        base self._ns(step) when an in-step completion-fix rebuilt under <ns>_c<attempt>.  Every
        downstream consumer that loads the .so (perf re-measure, cuda-graph / in-situ / decode gates,
        branch-tip gate) must use THIS, not the base namespace, or torch.ops.<ns> won't resolve."""
        b = (rec.get("build") if isinstance(rec, dict) else getattr(rec, "build", None)) or {}
        return b.get("namespace")

    def _branch_finish(self, best: Optional[dict]) -> Optional[dict]:
        """Budget reached: ADOPT the branch tip iff it matches-or-beats the frozen champion within
        tol AND clears the faithful gates; else REVERT (champion stays the anchor)."""
        b = self._plateau.branch or {}
        tip_cid = b.get("tip_cid"); tip_F = b.get("tip_F"); tip_so = b.get("tip_so")
        anchor_F = b.get("anchor_F")
        gate_ok = self._branch_gate_tip(tip_so, tip_cid) if tip_F is not None else False
        adopt = self._plateau.decide_adopt(gate_ok=gate_ok)
        if adopt and tip_cid:
            tip_rec = self._find_step_rec(tip_cid)
            checkpoint.promote_best(self.run_id, tip_cid)
            mr = (tip_rec.get("eval", {}) or {}).get("min_ratio", 0.0) if tip_rec else 0.0
            self._best_min_ratio = max(self._best_min_ratio, mr)
            self._champ_score = promotion.next_champ_score(
                self._champ_score, promotion.objective_score(tip_rec.get("eval", {}) if tip_rec else {},
                                                             C.PROMOTION_OBJECTIVE),
                C.PROMOTION_OBJECTIVE)
            self._plateau.on_promotion(b.get("step_indices", [tip_cid])[-1]
                                       if b.get("step_indices") else 0,
                                       tip_cid, tip_F, self._loc_of(tip_cid))
            if tip_rec:
                tip_rec["is_best"] = True
            self._memory.set_champion(tip_cid, (tip_rec or {}).get("step", 0), tip_F, mr,
                                      bottleneck=((tip_rec or {}).get("profile", {}) or {}).get("bottleneck"))
            self._log(f"[branch] ADOPT tip {tip_cid} (F={tip_F} >= anchor {anchor_F}) -> new champion")
            self._plateau.close_branch(True, note=f"adopted {tip_cid} F={tip_F}")
            best = tip_rec or best
        else:
            self._memory.add_dead_end(
                f"rewrite branch from {b.get('seed_cid')} REVERTED (tip_F={tip_F} vs frozen "
                f"{anchor_F}); that restructuring path did not match the champion")
            self._log(f"[branch] REVERT (tip_F={tip_F} vs anchor {anchor_F}, gate_ok={gate_ok}); "
                      f"champion stays {b.get('anchor_cid')}")
            self._plateau.close_branch(False, note=f"reverted tip_F={tip_F}")
        self._memory.commit()
        return best

    # ---- C3: branch continuity (rewrite_journal.md, injected per branch step) -----------
    @property
    def _journal_path(self) -> str:
        return os.path.join(self.paths["run"], "rewrite_journal.md")

    def _init_branch_journal(self, b: dict, reasons: list):
        """Start a fresh rewrite journal for a branch (the half-built rewrite must survive the fresh
        per-step session; this is the branch-scoped instance of the E memory channel)."""
        try:
            with open(self._journal_path, "w") as fh:
                fh.write(
                    "# Rewrite-branch journal (protected structural rewrite)\n\n"
                    f"- opened because STUCK: {', '.join(reasons)}\n"
                    f"- frozen champion (revert anchor): {b.get('anchor_cid')} "
                    f"(F={b.get('anchor_F')})\n"
                    f"- seed: {b.get('seed_cid')}  budget: {b.get('budget')} steps\n"
                    "- acceptance is BRANCH-LOCAL: a candidate may regress vs the trunk champion but "
                    "MUST stay correct; at budget end the tip is adopted only if it matches the "
                    "frozen champion within tol AND clears the faithful gates.\n\n"
                    "## Rungs\n")
        except Exception:
            pass

    def _update_branch_journal(self, rec: StepRecord, b: dict, became_tip: bool):
        """Append this branch step's rung: what changed, build/correctness, NCU delta, tip status."""
        try:
            ev = rec.eval or {}
            line = (f"- rung {b.get('rung')} step {rec.step} [{rec.status}]"
                    f"{' -> NEW TIP' if became_tip else ''}: {(rec.focus or '')[:140]}\n"
                    f"    F={ev.get('weighted_geomean_ratio')} min_ratio={ev.get('min_ratio')} "
                    f"bottleneck={(rec.profile or {}).get('bottleneck', '?')[:80]}\n")
            with open(self._journal_path, "a") as fh:
                fh.write(line)
        except Exception:
            pass

    def _branch_journal_digest(self, max_chars: int = 2500) -> str:
        """Bounded tail of the journal, injected into each branch step's prompt (survives the
        fresh session)."""
        try:
            with open(self._journal_path) as fh:
                txt = fh.read()
            return txt if len(txt) <= max_chars else "...\n" + txt[-max_chars:]
        except Exception:
            return ""

    def _is_promo_candidate(self, rec: StepRecord) -> bool:
        """Does this correct candidate clear the promotion bar under the active objective (C1)?
        Gates the expensive cost-tier verification (cuda-graph / in-situ / decode) AND promotion."""
        if not (rec.status == "correct" and rec.eval):
            return False
        score = promotion.objective_score(rec.eval, C.PROMOTION_OBJECTIVE)
        return promotion.meets_promotion_bar(score, self._champ_score)

    def _candidate_is_champion_dup(self, rec: StepRecord) -> bool:
        """True iff this step's candidate .cu is byte-identical to the CURRENT champion best.cu
        (the brain re-emitted the champion unchanged -- a 'hold').  Promotion for this step hasn't
        happened yet, so best.cu still holds the prior champion; a match means zero real change.
        Used by the accounting fix (C) to refuse re-promoting / counting such a no-op as progress."""
        try:
            cand = os.path.join(self.paths["candidates"], f"{rec.cid}.cu")
            champ = self.paths["best"]
            if not (os.path.exists(cand) and os.path.exists(champ)):
                return False
            with open(cand, "rb") as a, open(champ, "rb") as b:
                return a.read() == b.read()
        except OSError:
            return False

    def _persist_step(self, rec: StepRecord):
        """Common per-step tail: append the record, mirror the workdir to the archive (F3), replot."""
        checkpoint.append_step(self.run_id, rec)
        checkpoint.mirror_to_archive(self.run_id)
        self._update_plot()

    def _maybe_terminate(self, rec: StepRecord, step: int) -> Optional[dict]:
        """Success (correct + >=1.0x on all configs, re-measured) or env-fatal termination."""
        if rec.status == "correct" and rec.eval:
            mr = rec.eval.get("min_ratio", 0.0)
            de = rec.eval.get("decode_equiv_ok", None)
            if mr >= (1.0 - C.PERF_NOISE_BAND) and (de is True or not self.do_decode_equiv):
                so = (rec.build or {}).get("so_path")
                mr2 = mr
                if so and self.comp.get("measure"):
                    from fa3agent.perf.measure import measure_all_subprocess
                    rem = measure_all_subprocess(self._built_ns(rec) or self._ns(rec.step),
                                                 self.cfgs, so, self.pool.gpu_ids)
                    rr = [r.ratio for r in rem if r.ratio > 0]
                    mr2 = min(rr) if rr else 0.0
                    self._log(f"[verify] success re-measure: min_ratio {mr:.3f} -> {mr2:.3f} (isolated)")
                    rec.eval["min_ratio_recheck"] = mr2
                if mr2 >= (1.0 - C.PERF_NOISE_BAND):
                    self._log(f"SUCCESS: correct + min_ratio {mr2:.3f} >= 1.0 on all configs + decode_equiv={de}")
                    return {"status": "success", "step": step, "min_ratio": mr2}
                self._log(f"[verify] re-measure {mr2:.3f} below bar -> NOT terminating; continuing.")
        if rec.status == "env_fatal":
            self._log("ABORT: environment-fatal error (unfixable by agent).")
            return {"status": "env_fatal", "step": step}
        return None

    def _record_branch_memory(self, rec: StepRecord, champ_before: Optional[dict], became_tip: bool):
        """Branch-scoped memory: add a causal lesson tagged to the branch rung WITHOUT mutating the
        trunk champion or the lever-exhaustion ledger (those are trunk concepts)."""
        try:
            b = self._plateau.branch or {}
            lesson, _ = self._causal_lesson(rec, champ_before, promoted=False)
            lesson["change"] = f"[branch rung {b.get('rung')}] " + lesson["change"]
            if became_tip:
                lesson["outcome"] = "NEW BRANCH TIP — " + lesson["outcome"]
            self._memory.add_lesson(lesson)
            self._memory.commit()
        except Exception as e:
            self._log(f"[warn] branch memory update failed: {e!r}")

    # ---- main loop ----------------------------------------------------------------------
    def run(self):
        steps = checkpoint.load_steps(self.run_id)
        start = (steps[-1]["step"] + 1) if steps else 0
        prev = steps[-1] if steps else None
        best = checkpoint.best_record(self.run_id)
        self._best_min_ratio = (best.get("eval", {}) or {}).get("min_ratio", 0.0) if best else -1.0
        # Champion score under the ACTIVE promotion objective (C1).  -1.0 == no champion yet
        # (meets_promotion_bar treats <0 as "always promote").  For "min" this mirrors
        # _best_min_ratio; for "weighted_geomean" it tracks the champion's (non-monotone) F.
        self._champ_score = (promotion.objective_score(best.get("eval", {}), C.PROMOTION_OBJECTIVE)
                             if best else -1.0)
        self._stall = 0
        self._supervisor_note = ""
        if start > 0:
            self._log(f"RESUMING from step {start} (best min_ratio={self._best_min_ratio:.3f}, "
                      f"objective={C.PROMOTION_OBJECTIVE}, champ_score={self._champ_score:.3f})")

        # C2: re-derive the champion-F series on a cold resume (branches.json may be empty).
        self._plateau.rebuild_from_steps(steps, self._loc_of)

        step = start
        while self.max_steps is None or step < (start + self.max_steps if steps else self.max_steps):
            champ_before = best   # champion going INTO this step (promotion below may rebind `best`)

            # ---- C2: protected rewrite branch path (branch-local acceptance, deferred adopt) ----
            if C.ENABLE_REWRITE_BRANCH and self._plateau.branch_active():
                bm = self._plateau.branch
                seed_cid = bm.get("tip_cid") if bm.get("tip_F") is not None else bm.get("seed_cid")
                seed_path = self._candidate_path(seed_cid)
                branch_prev = self._find_step_rec(seed_cid) or prev
                try:
                    rec = self._step(step, branch_prev, best, branch=bm, seed_path=seed_path)
                except Exception as e:
                    self._log(f"[error] branch step {step} crashed: {e!r}\n{traceback.format_exc()[:1000]}")
                    rec = StepRecord(step=step, cid=f"step{step:03d}", parent=seed_cid, status="error",
                                     knowledge_note=str(e)[:300])
                correct = rec.status == "correct"
                F = (rec.eval or {}).get("weighted_geomean_ratio") if rec.eval else None
                mr = (rec.eval or {}).get("min_ratio") if rec.eval else None
                so = (rec.build or {}).get("so_path")
                tip_before = bm.get("tip_F")
                self._plateau.record_branch_step(step, rec.cid, correct, F, mr, so)
                became_tip = correct and F is not None and (tip_before is None or F > tip_before)
                self._record_branch_memory(rec, champ_before, became_tip)
                self._update_branch_journal(rec, self._plateau.branch or bm, became_tip)
                self._persist_step(rec)
                # D1: if nothing has been correct after a few branch steps, the knee substrate is
                # blocking the rewrite -> escalate to the from-scratch seed.  NOT for a capability
                # branch: it is champion-seeded to ADD one feature, so a bare-seed restart would throw
                # away the champion's gains -- let it ride its budget on the champion substrate.
                b = self._plateau.branch or {}
                if (not b.get("focus_lever") and b.get("tip_F") is None and not b.get("escalated")
                        and b.get("steps_used", 0) >= C.BRANCH_ESCALATE_AFTER
                        and b.get("seed_cid") != "step000"):
                    self._plateau.escalate_to_seed("step000")
                    self._log("[branch] no correct tip yet -> escalating BRANCH_SEED to from-scratch seed")
                elif became_tip:
                    self._plateau.advance_rung()   # D2: walk to the next transform rung
                term = self._maybe_terminate(rec, step)
                if term is not None:
                    return term
                if self._plateau.budget_exhausted():
                    best = self._branch_finish(best)
                    prev = best   # next trunk step continues from the (adopted or restored) champion
                else:
                    prev = asdict(rec) if correct else branch_prev
                step += 1
                continue

            try:
                rec = self._step(step, prev, best)
            except Exception as e:
                self._log(f"[error] step {step} crashed: {e!r}\n{traceback.format_exc()[:1000]}")
                rec = StepRecord(step=step, cid=f"step{step:03d}", parent=None, status="error",
                                 knowledge_note=str(e)[:300])

            # ---- accounting (Workstream C): REAL improvement vs HOLD ---------------------------
            # Decide ONCE, up front, whether this candidate is genuine progress.  A byte-identical
            # re-emit of the champion, or a "win" that sits inside the measurement-noise band, is NOT
            # progress -- counting either as a promotion reset the stall/plateau counters, ratcheted
            # the min bar on noise, and marked the lever "landed", so the agent coasted (re-promoting
            # the same kernel every step while the plateau detector stayed silent and the rewrite
            # branch never opened -- the "defeatist hold").  Only a real improvement runs the
            # expensive faithful gates below and promotes; everything else is a HOLD that flows into
            # the non-promotion accounting so STUCK fires.  FA3_PROMOTION_STRICT_PROGRESS=0 restores
            # the legacy "any bar-clear is progress" behavior.
            champ_identical = self._candidate_is_champion_dup(rec)
            if C.PROMOTION_STRICT_PROGRESS:
                real_improve = bool(
                    rec.status == "correct" and rec.eval is not None
                    and (best is None or (not champ_identical and promotion.is_real_improvement(
                        promotion.objective_score(rec.eval, C.PROMOTION_OBJECTIVE), self._champ_score))))
            else:
                real_improve = self._is_promo_candidate(rec)
                champ_identical = False

            # ---- CUDA-graph compatibility gate (binary) ----
            # vLLM captures the FA3 op in a FULL CUDA graph for steady-state decode, so a candidate
            # that can't be captured -- or diverges on replay -- is not a valid drop-in even when it
            # is correct + fast EAGERLY.  Every other gate runs eager and misses this.  Run it for a
            # correct promotion candidate; a real (non-skip) failure blocks promotion and is fed back.
            if rec.status == "correct" and rec.eval and self.comp.get("check_cuda_graph"):
                if real_improve:   # only verify a genuine improvement (a hold is the champion, already gated)
                    so = (rec.build or {}).get("so_path"); ns = self._built_ns(rec) or self._ns(rec.step)
                    if so:
                        gpu = self.pool.gpu_ids[0]
                        with self.pool.exclusive(gpu):
                            cg = self.comp["check_cuda_graph"](so, ns)
                        rec.eval["cudagraph"] = cg
                        self._log(f"[cudagraph] {cg.get('note', '')}")
                        if cg.get("ok") is False and not cg.get("skip"):
                            rec.status = "cudagraph_fail"   # not graph-capturable -> not a valid drop-in
                            self._distill(rec,
                                "CUDA-graph INCOMPATIBLE: " + str(cg.get("note", "")) +
                                " The kernel MUST be cuda-graph-capturable (vLLM captures decode in a full "
                                "graph): no host sync / host allocation / device->host readback or host-side "
                                "launch decisions inside fwd; derive the grid from shapes + the precomputed "
                                "scheduler_metadata, not from reading tensor values on the host.", ntype="ks")

            # ---- faithful vLLM gates, COST-TIERED (only for a correct promotion candidate) ----
            # Tier 2: in-situ A/B on LIVE vLLM tensors (cheap, faithful -- catches live-layout
            #         bugs the synthetic gate misses, e.g. broadcast-descale OOB).
            # Tier 3: full 100-token greedy decode-equality (only when the candidate also meets
            #         the perf target -- the final acceptance gate for the residual near-tie risk).
            if self.do_decode_equiv and rec.status == "correct" and rec.eval:
                mr = rec.eval.get("min_ratio", 0.0)
                promo_cand = real_improve   # only verify a genuine improvement (holds skip the gates)
                so = (rec.build or {}).get("so_path"); ns = self._built_ns(rec) or self._ns(rec.step)
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

            # promotion (C1) + accounting (C): ONLY a real improvement (decided up front, and still
            # correct after the faithful gates) promotes and counts as progress.  A correct candidate
            # that merely clears the noise band -- a byte-identical re-emit or a within-noise "win" --
            # is a HOLD: keep the champion and fall through to the non-promotion accounting
            # (rejects++/stall++) so the plateau detector + rewrite branch actually fire instead of
            # the agent coasting on re-promoted holds.
            improved = False
            if real_improve and rec.status == "correct":   # gates above may have demoted to *_fail
                mr = rec.eval.get("min_ratio", 0.0)
                score = promotion.objective_score(rec.eval, C.PROMOTION_OBJECTIVE)
                rec.is_best = True
                improved = True
                self._best_min_ratio = max(self._best_min_ratio, mr)   # success/termination gate
                self._champ_score = promotion.next_champ_score(
                    self._champ_score if best is not None else None, score, C.PROMOTION_OBJECTIVE)
                checkpoint.promote_best(self.run_id, rec.cid)
                best = asdict(rec)
                # C2: extend the champion-F series (drives STUCK + knee).
                self._plateau.on_promotion(rec.step, rec.cid,
                                           rec.eval.get("weighted_geomean_ratio", 0.0),
                                           self._loc_of(rec.cid))
                self._log(f"[promote] {rec.cid} -> champion (objective={C.PROMOTION_OBJECTIVE} "
                          f"score={score:.3f}; min_ratio={mr:.3f}, "
                          f"F={rec.eval.get('weighted_geomean_ratio', 0.0):.3f})")
            elif rec.status == "correct" and rec.eval and self._is_promo_candidate(rec):
                # cleared the bar (a promotable tie) but NOT a real improvement -> a HOLD, not progress.
                why = ("byte-identical re-emit of the champion" if champ_identical
                       else "within measurement noise (no gain beyond the band)")
                self._log(f"[hold] {rec.cid} is correct and clears the bar but is {why} "
                          f"(min_ratio={rec.eval.get('min_ratio', 0.0):.3f} vs champ_score "
                          f"{self._champ_score:.3f}) -> NOT promoted, counted as no-progress")

            # E1/E2: author the FINAL causal lesson for this step + refresh persisted MEMORY.md
            # (before the F3 mirror, so the up-to-date memory lands in the durable archive too).
            self._record_memory(rec, champ_before, improved)
            self._persist_step(rec)

            # supervisor watchdog (AVO-style) + C2 plateau bookkeeping
            if improved:
                self._stall = 0; self._supervisor_note = ""
            else:
                self._plateau.on_nonpromotion()
                self._stall += 1
                if self._stall >= 3:
                    self._supervisor_note = ("You have not improved in several rounds. Step back, "
                        "review the profiler bottleneck, and try a DIFFERENT subsystem/approach this round.")

            # C2/Layer 4: open a protected rewrite branch when stuck.  PRIORITY: if the trunk just
            # gave up on an all-or-nothing capability (e.g. tma-load exhausted its in-step budget with
            # TMA still absent), fund THAT capability in a champion-seeded branch; otherwise fall back
            # to the generic restructure-from-knee branch (score-plateau / structural-stall / rejects).
            if (C.ENABLE_REWRITE_BRANCH and not improved and best is not None
                    and not self._plateau.branch_active()):
                cap_lever = self._capability_branch_target(rec)
                if cap_lever:
                    self._open_capability_branch(best, cap_lever)
                else:
                    recent_struct = self._levers.recent_structural_outcomes(C.STUCK_STRUCTURAL_STALL)
                    reasons = self._plateau.stuck_reasons(recent_struct)
                    if reasons:
                        self._open_branch(best, reasons)

            term = self._maybe_terminate(rec, step)
            if term is not None:
                return term

            # Feedback source for the NEXT step.  On a PROMOTION the brain edits the new champion,
            # so feed back its own profile/eval/sass.  On a NON-promotion the brain continues from
            # the (unchanged) champion -- so feed back the CHAMPION's profile/eval/sass, NOT the
            # discarded regression's (whose metrics describe a kernel that is being thrown away).
            # The "what the last attempt tried and why it regressed" is carried by the memory
            # channel (MEMORY.md, E1/E2), not by these kernel-describing fields.
            if improved or best is None:
                prev = asdict(rec)
            else:
                prev = dict(best)
                prev["session_id"] = rec.session_id   # keep session continuity for optional resume
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
