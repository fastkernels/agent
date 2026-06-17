"""Unit test for B2: overlap-bound bottleneck class + off-tensor-core gating on SASS uses_wgmma.

Run from /data-fast/agent (no GPU/torch needed; ncu_report not required for _classify):
    python -m fa3agent.tests.test_ncu_classify

Drives ncu_profile._classify with a SYNTHETIC NCU `action` (metric -> value) across scenarios:
  - wgmma present + idle pipe + scoreboard/wait stalls  -> overlap-bound
  - wgmma present but NCU reads ~0 tensor% + high FMA    -> NOT off-tensor-core (gated)
  - no wgmma + ~0 tensor% + high FMA                     -> off-tensor-core
  - unknown wgmma (None) + ~0 tensor% + high FMA         -> off-tensor-core (legacy behavior kept)
"""
from __future__ import annotations

import sys

from fa3agent.perf import ncu_profile


class _Metric:
    def __init__(self, v):
        self._v = v

    def value(self):
        return self._v


class _Action:
    """Minimal stand-in for an ncu_report action: action[name].value() or KeyError."""
    def __init__(self, metrics: dict):
        self._m = metrics

    def __getitem__(self, name):
        if name in self._m:
            return _Metric(self._m[name])
        raise KeyError(name)


def _stalls(**ratios) -> dict:
    """Build the Hopper `.ratio` stall metrics from short reason names."""
    return {f"smsp__average_warps_issue_stalled_{r}_per_issue_active.ratio": v
            for r, v in ratios.items()}


def _base(tensor=0.0, fma=0.0, dram=10.0, sm_sol=40.0, occ=50.0, waves=2.0, **stalls) -> dict:
    m = {
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": tensor,
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": fma,
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": dram,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": sm_sol,
        "sm__warps_active.avg.pct_of_peak_sustained_active": occ,
        "launch__waves_per_multiprocessor": waves,
        "smsp__sass_inst_executed_op_local_ld.sum": 0.0,
        "smsp__sass_inst_executed_op_local_st.sum": 0.0,
    }
    m.update(_stalls(**stalls))
    return m


def _ok(msg):
    print(f"  PASS: {msg}")


def _fail(msg):
    print(f"  FAIL: {msg}")
    return 1


def main() -> int:
    failures = 0

    # (1) wgmma present, pipe moderate but idle behind scoreboard/wait -> overlap-bound
    a = _Action(_base(tensor=20.0, fma=5.0, long_scoreboard=5.0, wait=3.0,
                       short_scoreboard=0.5, barrier=0.5))
    label, *_ = ncu_profile._classify(a, uses_wgmma=True)
    print(f"[1] wgmma present + idle pipe -> {label}")
    if label == "overlap-bound":
        _ok("overlap-bound detected")
    else:
        failures += _fail(f"expected overlap-bound, got {label}")

    # (2) wgmma present, NCU mis-reads ~0 tensor% + high FMA, LOW stalls -> NOT off-tensor-core
    a = _Action(_base(tensor=0.0, fma=30.0, long_scoreboard=0.2, wait=0.1,
                       not_selected=2.0))
    label, *_ = ncu_profile._classify(a, uses_wgmma=True)
    print(f"[2] wgmma present + NCU ~0 tensor% -> {label}")
    if label != "off-tensor-core":
        _ok(f"off-tensor-core correctly gated off (got {label})")
    else:
        failures += _fail("off-tensor-core misfired despite uses_wgmma=True")

    # (3) NO wgmma + ~0 tensor% + high FMA -> off-tensor-core
    a = _Action(_base(tensor=0.0, fma=30.0, long_scoreboard=0.2, wait=0.1, not_selected=2.0))
    label, *_ = ncu_profile._classify(a, uses_wgmma=False)
    print(f"[3] no wgmma + ~0 tensor% -> {label}")
    if label == "off-tensor-core":
        _ok("off-tensor-core fires when wgmma absent")
    else:
        failures += _fail(f"expected off-tensor-core, got {label}")

    # (4) unknown wgmma (None) -> legacy behavior: off-tensor-core still fires
    a = _Action(_base(tensor=0.0, fma=30.0, long_scoreboard=0.2, wait=0.1, not_selected=2.0))
    label, *_ = ncu_profile._classify(a, uses_wgmma=None)
    print(f"[4] unknown wgmma -> {label}")
    if label == "off-tensor-core":
        _ok("off-tensor-core fires when wgmma unknown (legacy default)")
    else:
        failures += _fail(f"expected off-tensor-core (legacy), got {label}")

    # (5) wgmma present + scoreboard stalls but WARP-STARVED (occ << floor, waves < 1) -> the stalls
    #     come from too few resident warps, not an overlap failure: occupancy-bound, NOT overlap-bound.
    #     (This is the real step-34/36 attn_prefill_wgmma kernel: 250 regs -> ~12.5% occ, ~1 block/SM.)
    a = _Action(_base(tensor=13.0, fma=10.0, occ=12.5, waves=0.5,
                      long_scoreboard=5.0, wait=3.0, short_scoreboard=0.5, barrier=0.5))
    label, *_ = ncu_profile._classify(a, uses_wgmma=True)
    print(f"[5] wgmma present + low occ (12.5%, <1 wave) -> {label}")
    if label == "occupancy-bound":
        _ok("warp-starved tensor-core kernel -> occupancy-bound (not overlap-bound)")
    else:
        failures += _fail(f"expected occupancy-bound for a starved wgmma kernel, got {label}")

    # (6) same stalls but ADEQUATE occupancy -> a genuine overlap problem stays overlap-bound
    a = _Action(_base(tensor=13.0, fma=10.0, occ=45.0, waves=2.0,
                      long_scoreboard=5.0, wait=3.0, short_scoreboard=0.5, barrier=0.5))
    label, *_ = ncu_profile._classify(a, uses_wgmma=True)
    print(f"[6] wgmma present + adequate occ (45%) -> {label}")
    if label == "overlap-bound":
        _ok("adequately-occupied stalled wgmma kernel stays overlap-bound")
    else:
        failures += _fail(f"expected overlap-bound at adequate occupancy, got {label}")

    print()
    if failures == 0:
        print("SMOKE: PASS (B2 classify)")
    else:
        print(f"SMOKE: FAIL ({failures} failing checks)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
