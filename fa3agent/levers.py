"""Bottleneck-driven optimization levers (Workstreams B3 / B3b / D2).

ONE source of truth -- the `CATALOG` below -- keyed by the NCU `_bottleneck_label`.  B3 reads it to
pick the next lever for the CURRENT bottleneck (replacing the old `step % 5` rotation); D2 walks the
same list rung-by-rung inside a rewrite branch.  Each `Lever` carries:

  - ``bottlenecks``: the NCU labels (from ncu_profile._classify) it addresses;
  - ``focus``: the instruction text injected as SUGGESTED FOCUS;
  - ``doc_sections``: curated (file, heading) refs get_section()'d into the prompt;
  - ``kind``: ``structural`` (a layout/warp-spec/pipeline rewrite -- consumed by C2's structural-
    stall STUCK trigger) or ``micro`` (a targeted tweak);
  - ``success``: the NCU success criterion D2 checks after the rung;
  - ``rung``: ordering for the D2 ladder (lower = earlier).

The `LeverController` tracks each lever's outcome (landed / regressed / no_effect / failed) in a
persisted ``lever_state.json`` (in the run workdir, mirrored to the archive by F3), stops
re-suggesting an exhausted lever, and exposes the ledger for MEMORY.md + the structural-stall
history for C2.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

STRUCTURAL = "structural"
MICRO = "micro"

# After this many consecutive non-landing outcomes (no intervening land) a lever is EXHAUSTED for
# the rest of the run and no longer suggested for its bottleneck (forces a different approach).
LEVER_MAX_FAILS = int(os.environ.get("FA3_LEVER_MAX_FAILS", "2"))
# A FOUNDATION lever (one with a `verify` SASS signal -- wgmma / TMA / fp8-wgmma) whose capability is
# still ABSENT from the build did not actually get its technique tested: the brain never emitted the
# instruction class (the completion gate handles that), so a "no_effect"/"regressed" perf verdict is
# NOT evidence the technique fails.  Such non-implementations are tracked separately and only exhaust
# the lever after this many in a row -- so a needed capability (e.g. TMA) is re-attempted instead of
# being permanently locked out the first time the brain substitutes an easier path.
LEVER_NOT_IMPL_CAP = int(os.environ.get("FA3_LEVER_NOT_IMPL_CAP", "3"))
# |dF| <= this counts as "no effect" rather than a regression (measurement noise band).
LEVER_NOEFFECT_EPS = float(os.environ.get("FA3_LEVER_NOEFFECT_EPS", "0.005"))


@dataclass
class Lever:
    id: str
    bottlenecks: tuple
    focus: str
    doc_sections: list
    kind: str
    success: str
    rung: int
    # reflib.get_hints query for THIS lever's technique (populated from _HINTS_TOPIC below); the
    # readable pure-PTX model mined from ThunderKittens/DeepGEMM/flashinfer/FlashMLA.
    hints: str = ""
    # SASS-summary key whose presence CONFIRMS this lever actually landed its new functionality
    # (populated from _VERIFY_SIGNAL below).  Empty for levers with no single binary signature
    # (micro tweaks, occupancy/scheduling).  Drives the in-step COMPLETION GATE (config.COMPLETION
    # _GATE): when set AND the source kernel lacks the signal, the step is "introducing" it and the
    # driver verifies adoption on the built candidate before committing the step.
    verify: str = ""
    # ALL-OR-NOTHING flag (populated from _BRANCH_DEV below): a big structural rewrite (TMA pipeline,
    # persistent/L2 schedule, cluster multicast) that regresses-before-it-helps and so cannot land in
    # a single trunk step under the promotion bar.  When the trunk fails to land such a capability
    # (Layer 4), the orchestrator funds it through a dedicated, champion-seeded rewrite branch with
    # branch-local acceptance (regressions tolerated; must stay correct) over several steps.
    branch_dev: bool = False


# ---- the catalog (B3 lever list == D2 transform catalog) --------------------------------------
CATALOG: list = [
    Lever(
        id="wgmma-tensorcore",
        bottlenecks=("off-tensor-core", "mixed/inconclusive"),
        focus=("Use wgmma.mma_async (m64nNk16 bf16, N>=64) for the QK^T and P*V GEMMs instead of "
               "scalar FFMA; load Q/K/V tiles to shared memory with cp.async.bulk.tensor (TMA) + "
               "mbarrier producer/consumer."),
        doc_sections=[("kernel-patterns.md", "Async execution model"),
                      ("hopper-h200.md", "Tensor Cores (4th Generation)")],
        kind=STRUCTURAL,
        success="tensor-pipe %active rises above 50%",
        rung=1),
    # FOUNDATION (split from the old conflated "tma-multistage-pipeline", Layer 3): land the TMA load
    # PRODUCER first -- the cp.async.bulk.tensor + mbarrier transaction that feeds K/V tiles to smem.
    # This is the gated capability (verify=uses_tma); the multistage RING that hides latency on top of
    # it is a separate tactic below, so a "no_effect" on the ring no longer exhausts the TMA producer.
    Lever(
        id="tma-load",
        bottlenecks=("latency-bound", "dram-bandwidth-bound"),
        focus=("Load K/V (and Q) tiles through TMA: issue cp.async.bulk.tensor.* from a CUtensorMap "
               "descriptor into shared memory, signalled by an mbarrier with arrive.expect_tx / "
               "try_wait, instead of LDG/cp.async.ca; ensure 128B-coalesced, swizzle-compatible "
               "tiles.  Establish the async-copy PRODUCER (the multistage ring is a separate step)."),
        doc_sections=[("kernel-patterns.md", "Async execution model"),
                      ("kernel-patterns.md", "Global→shared async copy")],
        kind=STRUCTURAL,
        success="UTMALDG/UBLKCP present in SASS; long-scoreboard stall share drops",
        rung=2),
    # TACTIC (split from the old conflated lever): a 2-4 stage software pipeline OVER an existing TMA
    # producer.  No verify signal -- it refines a path that must already have TMA (so a no_effect here
    # is a real tactic result, not a missing capability).
    Lever(
        id="multistage-pipeline",
        bottlenecks=("latency-bound", "dram-bandwidth-bound"),
        focus=("Hide load latency with a multistage async pipeline ON TOP of the TMA load: prefetch "
               "the next K/V tile (cp.async.bulk.tensor) into a 2-4 stage smem ring with rotating "
               "mbarriers while the current tile is consumed by wgmma; keep KV resident across the "
               "score+PV passes so the copy fully overlaps compute."),
        doc_sections=[("kernel-patterns.md", "Software pipelining (double buffering)"),
                      ("kernel-patterns.md", "Async execution model")],
        kind=STRUCTURAL,
        success="long-scoreboard stall share drops and DRAM %peak falls",
        rung=3),
    Lever(
        id="fp8-wgmma-promote",
        # Reached via the fp8 FOUNDATION gate in select() (worst config is e4m3 + no QGMMA in SASS),
        # not by an NCU label -- the synthetic 'fp8-scalar-fallback' tag below never matches a real
        # classifier verdict, so this lever only fires for a genuinely fp8-on-scalar worst config.
        bottlenecks=("fp8-scalar-fallback",),
        focus=("Put the fp8 (e4m3) path on tensor cores: use wgmma.mma_async.sync.aligned with the "
               ".f32.e4m3.e4m3 variant for QK^T and P*V instead of the scalar FFMA fallback the fp8 "
               "config is on now.  Accumulate inside the wgmma f32 registers but PROMOTE to a clean "
               "fp32 accumulator every Nc K-elements (DeepGEMM-style two-level accumulation) to bound "
               "the ~fp22 wgmma accumulation error; apply the per-tile q/k/v descale factors at the "
               "promotion boundary.  Keep the bf16 path unchanged."),
        doc_sections=[("kernel-patterns.md", "DeepGemm FP8 Hopper"),
                      ("kernel-patterns.md", "Async execution model")],
        kind=STRUCTURAL,
        success="fp8 worst-config ratio rises; QGMMA present in SASS; tensor-pipe %active rises",
        rung=3),
    Lever(
        id="warp-specialize",
        bottlenecks=("overlap-bound", "sync-bound"),
        focus=("Warp-specialize: a producer warpgroup issues TMA loads, consumer warpgroups run "
               "wgmma; use setmaxnreg to rebalance registers; named barriers to order the two math "
               "warpgroups so softmax/correction overlaps the GEMM pipeline instead of serializing."),
        doc_sections=[("kernel-patterns.md", "Warp specialization"),
                      ("kernel-patterns.md", "Software pipelining (double buffering)")],
        kind=STRUCTURAL,
        success="long-scoreboard/wait stall share drops and tensor-pipe %active rises",
        rung=4),
    Lever(
        id="ping-pong-overlap",
        bottlenecks=("overlap-bound",),
        focus=("Ping-pong two consumer warpgroups so softmax never stalls the tensor core: while "
               "warpgroup A runs the P*V wgmma for tile i, warpgroup B computes the softmax "
               "(rowmax/rowsum/exp2 + correction) for tile i+1, then they swap.  Order the hand-off "
               "with named barriers (bar.sync / mbarrier) and a 2-stage smem ring so the MUFU/exp "
               "and the wgmma pipelines run concurrently instead of alternating idle."),
        doc_sections=[("kernel-patterns.md", "Warp specialization"),
                      ("kernel-patterns.md", "Reduction & online softmax")],
        kind=STRUCTURAL,
        success="tensor-pipe %active stays high while MUFU/transcendental runs; SM SOL rises",
        rung=5),
    Lever(
        # The register-reduction lever.  Reached two ways: the register-spill verdict (kill spills),
        # and -- via the select() priority gate (2b) -- the register/occupancy-STARVED tensor-core
        # verdict (`register-occupancy-bound`: a ~1-block/SM wgmma kernel), where a smaller
        # register/smem footprint is the most direct way to fit >1 block/SM and lift occupancy.
        id="setmaxnreg-split-registers",
        bottlenecks=("register-spill",),
        focus=("Cut register/smem pressure so more blocks fit per SM (raise occupancy) and spills "
               "vanish: add __launch_bounds__, split consumer warpgroups with setmaxnreg, shrink "
               "live register tiles / per-thread accumulator footprint, move O out of registers "
               "into shared memory if needed."),
        doc_sections=[("kernel-patterns.md", "Occupancy tuning"),
                      ("kernel-patterns.md", "Warp specialization")],
        kind=STRUCTURAL,
        success="achieved occupancy rises (more blocks resident/SM) and local ld/st spills == 0",
        rung=5),
    Lever(
        id="softmax-log2",
        # MICRO lever: only for a healthy tensor-core kernel (NOT overlap-bound -- a log2 softmax
        # tweak cannot fix tensor cores sitting idle; that needs warp-specialization / TMA overlap).
        bottlenecks=("tensor-core-good",),
        focus=("Online softmax in the log2 domain (exp2, fold log2(e) into the scale); branchless "
               "predicated rescaling on fully-unmasked K-blocks; overlap the correction with the "
               "P*V GEMM pipeline."),
        doc_sections=[("kernel-patterns.md", "Reduction & online softmax")],
        kind=MICRO,
        success="transcendental/MUFU stall share drops; SM SOL rises",
        rung=6),
    Lever(
        id="decode-splitkv",
        bottlenecks=("occupancy-bound",),
        focus=("For decode (small M), pack heads/tokens into the wgmma M dimension or use split-KV "
               "across the KV sequence with an LSE-weighted combine to raise occupancy."),
        doc_sections=[("kernel-patterns.md", "Occupancy tuning"),
                      ("kernel-patterns.md", "Persistent kernels & scheduling")],
        kind=STRUCTURAL,
        success="waves/SM moves toward >=1 and achieved occupancy rises",
        rung=7),
    Lever(
        id="persistent-l2-scheduling",
        bottlenecks=("occupancy-bound", "dram-bandwidth-bound"),
        focus=("Use a persistent kernel with an L2-aware / persistent tile schedule: launch one CTA "
               "per SM that loops over output tiles, order tiles to reuse K/V resident in L2 "
               "(cudaStreamSetAttribute access policy / cp.async L2 hints)."),
        doc_sections=[("kernel-patterns.md", "Persistent kernels & scheduling"),
                      ("kernel-patterns.md", "L2 cache control")],
        kind=STRUCTURAL,
        success="waves/SM >= 1 and L2 hit-rate rises / DRAM %peak falls",
        rung=8),
    Lever(
        id="clusters-tma-multicast",
        bottlenecks=("dram-bandwidth-bound", "occupancy-bound"),
        focus=("Use thread-block clusters (sm_90) + TMA multicast: a cluster of CTAs shares one "
               "TMA load via cp.async.bulk.tensor multicast into distributed shared memory "
               "(cluster.map_shared_rank), cutting redundant HBM reads of shared K/V."),
        doc_sections=[("kernel-patterns.md", "L2 cache control"),
                      ("kernel-patterns.md", "Persistent kernels & scheduling")],
        kind=STRUCTURAL,
        success="dram__bytes_read.sum drops and DRAM %peak falls",
        rung=9),
    Lever(
        id="async-store-epilogue",
        bottlenecks=("tensor-core-good", "occupancy-bound"),
        focus=("Make the epilogue async: stage the O tile through shared memory and store with STSM "
               "/ cp.async.bulk.tensor (TMA store) so the write-out overlaps the next tile's compute "
               "instead of stalling at the end."),
        doc_sections=[("kernel-patterns.md", "Async execution model"),
                      ("kernel-patterns.md", "Global→shared async copy")],
        kind=STRUCTURAL,
        success="end-of-kernel store stalls drop; tail effect shrinks",
        rung=10),
    Lever(
        id="faster-barriers",
        bottlenecks=("sync-bound",),
        focus=("Upgrade synchronization: replace __syncthreads with mbarrier arrive/try_wait + "
               "named barriers scoped to the producer/consumer warpgroups; minimize barrier count "
               "and use transaction barriers for TMA completion."),
        doc_sections=[("kernel-patterns.md", "Async execution model"),
                      ("kernel-patterns.md", "Warp specialization")],
        kind=MICRO,
        success="barrier/membar stall share drops",
        rung=11),
    Lever(
        id="tile-swizzle-tuning",
        bottlenecks=("tensor-core-good", "mixed/inconclusive", "dram-bandwidth-bound"),
        focus=("Tune tile sizes (block_m/block_n), pipeline stages (2-4), and SMEM swizzle for the "
               "head dim to kill bank conflicts; request >48KB dynamic SMEM via cudaFuncSetAttribute."),
        doc_sections=[("kernel-patterns.md", "Shared memory swizzling"),
                      ("kernel-patterns.md", "GEMM tiling")],
        kind=MICRO,
        success="shared-memory bank-conflict / short-scoreboard stalls drop; SM SOL rises",
        rung=12),
    Lever(
        id="vectorized-loads",
        bottlenecks=("dram-bandwidth-bound", "latency-bound"),
        focus=("On any non-TMA load path (e.g. the scalar/decode fallback) issue wide 128-bit "
               "vectorized, coalesced global loads: cast Q/K/V to float4/int4 (or use "
               "__ldg/ld.global.nc.v4) so each thread moves 16 B and a warp moves one 128 B cache "
               "line per transaction; align/pad the head-dim leading stride to keep accesses "
               "contiguous.  A micro fallback when the structural TMA pipeline is not applicable."),
        doc_sections=[("kernel-patterns.md", "Vectorized loads"),
                      ("kernel-patterns.md", "Global memory coalescing")],
        kind=MICRO,
        success="L2/DRAM sectors-per-request drops toward 4; long-scoreboard stall share falls",
        rung=13),
    Lever(
        id="epilogue-store-microopts",
        bottlenecks=("tensor-core-good", "mixed/inconclusive"),
        focus=("Squeeze the O write-out without changing structure: coalesce the register->GMEM "
               "stores (vectorize to 128-bit, contiguous per warp), use streaming/write-back cache "
               "hints (__stwt / st.global.cs) on the final O so it does not evict resident K/V from "
               "L2, and use the wgmma accumulate flag (scale-d=0 on the first k-step) to skip "
               "explicitly zeroing the accumulator each tile."),
        doc_sections=[("kernel-patterns.md", "L2 cache control"),
                      ("kernel-patterns.md", "Global memory coalescing")],
        kind=MICRO,
        success="epilogue store stalls drop; L2 hit-rate holds; no extra accumulator-init overhead",
        rung=14),
]

_BY_ID = {lv.id: lv for lv in CATALOG}
# Deterministic fallback rotation when there is no bottleneck label or all matches are exhausted.
_FALLBACK_ORDER = [lv.id for lv in sorted(CATALOG, key=lambda l: l.rung)]

# Reference-library hint topic per lever (the reflib.get_hints query string).  Kept beside the
# catalog so get_hints tracks the ACTIVE lever instead of one hardcoded topic.  Each string is
# chosen to resolve (via reflib.hints._ALIASES) to the on-topic reflib snippet(s) for the lever's
# technique; tile-swizzle -> descriptor/swizzle, warp-spec -> warp_specialization + pipeline, etc.
_HINTS_TOPIC = {
    "wgmma-tensorcore": "wgmma",
    "tma-load": "tma load",
    "multistage-pipeline": "tma load pipeline",
    "fp8-wgmma-promote": "wgmma descriptor",
    "warp-specialize": "warp specialization pipeline",
    "ping-pong-overlap": "pipeline softmax",
    "setmaxnreg-split-registers": "warp specialization register",
    "softmax-log2": "online softmax",
    "decode-splitkv": "split-kv",
    "persistent-l2-scheduling": "pipeline",
    "clusters-tma-multicast": "tma load",
    "async-store-epilogue": "tma store",
    "faster-barriers": "mbarrier",
    "tile-swizzle-tuning": "swizzle descriptor",
    "vectorized-loads": "ldmatrix",
    "epilogue-store-microopts": "tma store epilogue",
}
for _lv in CATALOG:
    _lv.hints = _HINTS_TOPIC.get(_lv.id, _lv.id.replace("-", " "))

# Per-lever ADOPTION signal for the in-step completion gate: the SASS-summary key that must be
# True once the lever's new functionality is really present in the compiled kernel.  Only the
# levers whose whole point is to introduce a distinct instruction CLASS get one -- so the gate
# fires for "did the brain actually put it on tensor cores / TMA?" and stays silent for tweaks
# that have no clean binary signature.  Adding a future foundation lever = one entry here.
_VERIFY_SIGNAL = {
    "wgmma-tensorcore": "uses_wgmma",
    "tma-load": "uses_tma",
    "fp8-wgmma-promote": "uses_fp8_wgmma",
}
for _lv in CATALOG:
    _lv.verify = _VERIFY_SIGNAL.get(_lv.id, "")

# ALL-OR-NOTHING structural rewrites (Layer 4): too big to land in one trunk step under the promotion
# bar, so the trunk funds them through a dedicated champion-seeded rewrite branch when it can't land
# them in-step.  Each must be a real, separable capability (the TMA producer, the persistent/L2
# schedule, the cluster multicast) -- NOT a micro tweak.
_BRANCH_DEV = {"tma-load", "persistent-l2-scheduling", "clusters-tma-multicast"}
for _lv in CATALOG:
    _lv.branch_dev = _lv.id in _BRANCH_DEV


def is_branch_dev(lever_id: str) -> bool:
    """True iff `lever_id` is an all-or-nothing structural rewrite the trunk should fund through a
    protected rewrite branch (resolves aliases, so the retired combined id maps to tma-load)."""
    lv = lever_by_id(lever_id)
    return bool(lv and lv.branch_dev)

# Human-readable name of each adoption signal, for the completion follow-up prompt (what the brain
# must make actually appear in the built kernel).
VERIFY_LABELS = {
    "uses_wgmma": ("wgmma.mma_async tensor-core instructions (HGMMA/QGMMA must appear in the "
                   "compiled SASS) — the QK^T and/or P*V GEMMs must run on tensor cores, not scalar FFMA"),
    "uses_tma": ("TMA async bulk copies (cp.async.bulk.tensor; UTMALDG/UBLKCP must appear in the "
                 "compiled SASS) — the K/V (and/or Q) tile loads must go through the TMA pipeline"),
    "uses_fp8_wgmma": ("fp8 wgmma (QGMMA must appear in the compiled SASS) — the fp8/e4m3 path must "
                       "run on tensor cores via the .e4m3 wgmma variant, not the scalar FFMA fallback"),
}


# Retired lever ids -> their replacement, so a pre-split lever_state.json (history + ledger entries
# referencing "tma-multistage-pipeline") still resolves after the Layer-3 split.  The old combined
# lever's verify signal (uses_tma) matches tma-load, so the foundation continuity is preserved.
_ALIASES = {"tma-multistage-pipeline": "tma-load"}


def lever_by_id(lever_id: str) -> Optional[Lever]:
    return _BY_ID.get(lever_id) or _BY_ID.get(_ALIASES.get(lever_id, ""))


class LeverController:
    """Selects the next lever from the bottleneck label + tracks per-lever outcomes."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.state_path = os.path.join(run_dir, "lever_state.json")
        self.state = self._load()

    def _default_state(self) -> dict:
        return {"levers": {}, "history": []}

    def _load(self) -> dict:
        try:
            with open(self.state_path) as fh:
                st = json.load(fh)
            base = self._default_state()
            base.update(st or {})
            return base
        except Exception:
            return self._default_state()

    def _save(self):
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.state, fh, indent=2)
            os.replace(tmp, self.state_path)
        except Exception:
            pass

    def _entry(self, lever_id: str) -> dict:
        e = self.state["levers"].get(lever_id)
        if e is None:
            e = {"tried": 0, "landed": 0, "regressed": 0, "no_effect": 0, "failed": 0,
                 "fail_streak": 0, "not_impl": 0, "not_impl_streak": 0, "exhausted": False}
            self.state["levers"][lever_id] = e
        return e

    def is_exhausted(self, lever_id: str) -> bool:
        return bool(self.state["levers"].get(lever_id, {}).get("exhausted"))

    def reset_exhaustion(self, foundation_only: bool = True) -> list:
        """Clear `exhausted` + the fail/not-impl streaks so a technique can be re-attempted.

        A FOUNDATION capability (wgmma / TMA / fp8-wgmma) is a prerequisite, not a tactic: it must
        never be foreclosed for the rest of a long run by a handful of early non-implementations or
        regressions.  Called on a plateau and after a rollback so the resumed run can try it again
        (e.g. once the brain has a better reference for the hard instruction sequence).  Returns the
        list of lever ids that were reopened."""
        reopened = []
        for lid, e in self.state["levers"].items():
            if not e.get("exhausted"):
                continue
            if foundation_only and not (lever_by_id(lid) and lever_by_id(lid).verify):
                continue
            e["exhausted"] = False
            e["fail_streak"] = 0
            e["not_impl_streak"] = 0
            reopened.append(lid)
        if reopened:
            self._save()
        return reopened

    # ---- selection --------------------------------------------------------------------
    def select(self, bottleneck_label: Optional[str], step: int,
               uses_tma: Optional[bool] = None, uses_wgmma: Optional[bool] = None,
               uses_fp8_wgmma: Optional[bool] = None, worst_dtype: Optional[str] = None) -> Lever:
        """Pick the lever for the current bottleneck.

        Order of preference:
          1. FP8 foundation gate: when the slowest config is fp8 (e4m3) and SASS shows NO fp8 wgmma
             (QGMMA), that config is on the scalar FFMA fallback -- promote it onto tensor cores
             first.  Gated on `uses_wgmma is True` so we only extend an existing wgmma+TMA path with
             the .e4m3 variant (don't force fp8 wgmma on a kernel with no tensor cores at all).
          2. TMA/tensor-core FOUNDATION gate (B3b): a tensor-core / overlap verdict cannot be fixed
             without the wgmma+TMA producer that feeds the consumers.  If the kernel still has NO
             wgmma, land that first; if it has wgmma but NO TMA, land the TMA multistage pipeline
             first.  `uses_*` are the SASS ground truth for the current champion/seed; a gate fires
             only when a flag is explicitly False (unknown -> skip) and the lever is not exhausted.
          3. A matching, non-exhausted lever, preferring STRUCTURAL over micro -- so a structural
             bottleneck never burns a step on a micro tweak.  Within the structural pool, explore
             BREADTH-first (least-tried, then lowest rung) so the several structural levers for one
             bottleneck (e.g. warp-specialize AND ping-pong for overlap-bound) each get a shot
             instead of hammering the lowest rung until it exhausts.
          4. A deterministic rotation over non-exhausted levers (no label / all matches exhausted).
        """
        label = (bottleneck_label or "").strip()

        # (1) fp8 foundation gate: a slow fp8 worst-config with no QGMMA is on the scalar fallback.
        if (worst_dtype and "fp8" in worst_dtype and uses_fp8_wgmma is False
                and uses_wgmma is True and not self.is_exhausted("fp8-wgmma-promote")):
            return _BY_ID["fp8-wgmma-promote"]

        # (2) foundation gate: overlap/off-tensor-core needs wgmma+TMA before anything else helps.
        if label in ("overlap-bound", "off-tensor-core"):
            if uses_wgmma is False and not self.is_exhausted("wgmma-tensorcore"):
                return _BY_ID["wgmma-tensorcore"]
            if uses_tma is False and not self.is_exhausted("tma-load"):
                return _BY_ID["tma-load"]

        # (2b) register/occupancy-STARVED tensor cores: a wgmma kernel pinned to ~1 block/SM (e.g.
        # 250 regs -> ~12% occupancy) is gated by its register/smem footprint, so attack registers
        # FIRST and persistently -- the most direct lever for THIS kernel -- before exploring the
        # breadth-first occupancy pool (split-K / persistent / clusters).  Breadth-first would
        # otherwise defer register reduction behind the untried occupancy levers (least-tried-first)
        # whenever it has any prior try.  Once it stops paying off (exhausts), fall through to (3),
        # which treats register-occupancy-bound as a plain occupancy case (split-K / persistent / ...).
        if (label == "register-occupancy-bound"
                and not self.is_exhausted("setmaxnreg-split-registers")):
            return _BY_ID["setmaxnreg-split-registers"]

        # (3) matching lever, STRUCTURAL-first; explore structural levers breadth-first.
        if label:
            # register-occupancy-bound shares the occupancy-bound lever pool once register reduction
            # (handled by gate 2b above) is exhausted.
            pool_label = "occupancy-bound" if label == "register-occupancy-bound" else label
            matches = [lv for lv in CATALOG if pool_label in lv.bottlenecks
                       and not self.is_exhausted(lv.id)]
            if matches:
                structural = [lv for lv in matches if lv.kind == STRUCTURAL]
                pool = structural or matches
                pool.sort(key=lambda lv: (self._entry(lv.id)["tried"], lv.rung))
                return pool[0]

        # (4) fallback: rotate through non-exhausted levers deterministically by step
        open_ids = [i for i in _FALLBACK_ORDER if not self.is_exhausted(i)]
        if not open_ids:
            open_ids = _FALLBACK_ORDER  # everything exhausted -> reuse rotation (degraded)
        return _BY_ID[open_ids[step % len(open_ids)]]

    # ---- outcome tracking -------------------------------------------------------------
    def record_outcome(self, lever_id: str, outcome: str, step: Optional[int] = None,
                       adopted: Optional[bool] = None):
        """`outcome` in {landed, regressed, no_effect, failed}.  Updates the ledger + exhaustion.

        `adopted` (foundation levers only): whether the lever's `verify` instruction class actually
        appeared in the built SASS.  When a foundation lever was selected but its capability is still
        ABSENT (`adopted is False`), the technique never really ran -- the brain substituted an easier
        path -- so we DON'T count it toward the normal fail-streak exhaustion (which would lock the
        capability out the way TMA was).  Instead it counts toward a separate not-implemented streak
        that only exhausts after LEVER_NOT_IMPL_CAP in a row.  `adopted is None` (no/unreadable SASS,
        or a lever with no verify signal) -> normal accounting."""
        lv = lever_by_id(lever_id)
        if lv is None:
            return
        e = self._entry(lever_id)
        e["tried"] += 1
        # A foundation lever whose capability is still absent did not get a fair test of the technique.
        not_implemented = (outcome == "not_implemented") or (bool(lv.verify) and adopted is False)
        if not_implemented:
            e["not_impl"] = e.get("not_impl", 0) + 1
            e["not_impl_streak"] = e.get("not_impl_streak", 0) + 1
            if e["not_impl_streak"] >= LEVER_NOT_IMPL_CAP:
                e["exhausted"] = True       # the brain can't land it unaided -> stop forcing it
            tag = "not_implemented"
        else:
            e["not_impl_streak"] = 0        # capability present now -> the technique got a real test
            if outcome == "landed":
                e["landed"] += 1
                e["fail_streak"] = 0
            else:
                e[outcome] = e.get(outcome, 0) + 1
                e["fail_streak"] += 1
                if e["fail_streak"] >= LEVER_MAX_FAILS:
                    e["exhausted"] = True
            tag = outcome
        self.state["history"].append([step, lever_id, lv.kind, tag])
        self.state["history"] = self.state["history"][-200:]
        self._save()

    @staticmethod
    def classify_outcome(promoted: bool, correct: bool, dF: Optional[float]) -> str:
        if promoted:
            return "landed"
        if not correct:
            return "failed"
        if dF is not None and dF < -LEVER_NOEFFECT_EPS:
            return "regressed"
        return "no_effect"

    # ---- introspection for MEMORY.md (E2) + the C2 structural-stall trigger -----------
    def kind_of(self, lever_id: str) -> Optional[str]:
        lv = lever_by_id(lever_id)
        return lv.kind if lv else None

    def recent_structural_outcomes(self, n: int) -> list:
        """The last `n` STRUCTURAL-lever outcomes (newest last) -- C2's structural-stall trigger."""
        structural = [h for h in self.state.get("history", []) if h[2] == STRUCTURAL]
        return [h[3] for h in structural[-n:]]

    def ledger_summary(self) -> dict:
        out = {}
        for lid, e in self.state["levers"].items():
            if e.get("tried"):
                tag = "EXHAUSTED" if e.get("exhausted") else "open"
                ni = f" not_impl={e['not_impl']}" if e.get("not_impl") else ""
                out[lid] = (f"tried={e['tried']} landed={e['landed']} regressed={e['regressed']} "
                            f"no_effect={e['no_effect']} failed={e['failed']}{ni} [{tag}]")
        return out
