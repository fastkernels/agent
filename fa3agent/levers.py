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
    Lever(
        id="tma-multistage-pipeline",
        bottlenecks=("latency-bound", "dram-bandwidth-bound"),
        focus=("Hide load latency with a multistage async pipeline: load K/V tiles via "
               "cp.async.bulk.tensor (TMA) into a 2-4 stage smem ring with mbarrier transaction "
               "barriers; keep KV resident across the score+PV passes; ensure 128B-coalesced access."),
        doc_sections=[("kernel-patterns.md", "Software pipelining (double buffering)"),
                      ("kernel-patterns.md", "Async execution model")],
        kind=STRUCTURAL,
        success="long-scoreboard stall share drops and DRAM %peak falls",
        rung=2),
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
        id="setmaxnreg-split-registers",
        bottlenecks=("register-spill",),
        focus=("Cut register pressure so spills vanish: add __launch_bounds__, split consumer "
               "warpgroups with setmaxnreg, shrink live register tiles / per-thread accumulator "
               "footprint, move O out of registers into shared memory if needed."),
        doc_sections=[("kernel-patterns.md", "Occupancy tuning"),
                      ("kernel-patterns.md", "Warp specialization")],
        kind=STRUCTURAL,
        success="local ld/st (spill) instructions == 0",
        rung=5),
    Lever(
        id="softmax-log2",
        bottlenecks=("tensor-core-good", "overlap-bound"),
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
]

_BY_ID = {lv.id: lv for lv in CATALOG}
# Deterministic fallback rotation when there is no bottleneck label or all matches are exhausted.
_FALLBACK_ORDER = [lv.id for lv in sorted(CATALOG, key=lambda l: l.rung)]


def lever_by_id(lever_id: str) -> Optional[Lever]:
    return _BY_ID.get(lever_id)


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
                 "fail_streak": 0, "exhausted": False}
            self.state["levers"][lever_id] = e
        return e

    def is_exhausted(self, lever_id: str) -> bool:
        return bool(self.state["levers"].get(lever_id, {}).get("exhausted"))

    # ---- selection --------------------------------------------------------------------
    def select(self, bottleneck_label: Optional[str], step: int) -> Lever:
        """Pick the lever for the current bottleneck: prefer a matching, non-exhausted lever not yet
        tried, then by lowest rung; fall back to a deterministic rotation over non-exhausted levers."""
        label = (bottleneck_label or "").strip()
        if label:
            matches = [lv for lv in CATALOG if label in lv.bottlenecks
                       and not self.is_exhausted(lv.id)]
            if matches:
                matches.sort(key=lambda lv: (self._entry(lv.id)["tried"], lv.rung))
                return matches[0]
        # fallback: rotate through non-exhausted levers deterministically by step
        open_ids = [i for i in _FALLBACK_ORDER if not self.is_exhausted(i)]
        if not open_ids:
            open_ids = _FALLBACK_ORDER  # everything exhausted -> reuse rotation (degraded)
        return _BY_ID[open_ids[step % len(open_ids)]]

    # ---- outcome tracking -------------------------------------------------------------
    def record_outcome(self, lever_id: str, outcome: str, step: Optional[int] = None):
        """`outcome` in {landed, regressed, no_effect, failed}.  Updates the ledger + exhaustion."""
        lv = lever_by_id(lever_id)
        if lv is None:
            return
        e = self._entry(lever_id)
        e["tried"] += 1
        if outcome == "landed":
            e["landed"] += 1
            e["fail_streak"] = 0
        else:
            e[outcome] = e.get(outcome, 0) + 1
            e["fail_streak"] += 1
            if e["fail_streak"] >= LEVER_MAX_FAILS:
                e["exhausted"] = True
        self.state["history"].append([step, lever_id, lv.kind, outcome])
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
                out[lid] = (f"tried={e['tried']} landed={e['landed']} regressed={e['regressed']} "
                            f"no_effect={e['no_effect']} failed={e['failed']} [{tag}]")
        return out
