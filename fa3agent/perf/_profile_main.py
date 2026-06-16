"""Torch-free entrypoint to run NCU profiling in an isolated subprocess.

ncu_report and torch SEGFAULT if imported in the same process.  The orchestrator has torch
loaded (build/correctness/perf), so it must invoke profiling out-of-process via this runner,
which imports ncu_profile (-> ncu_report) but NEVER torch.  Prints the ProfileFeedback as JSON.

Usage:  python -m fa3agent.perf._profile_main <so_path> <namespace> <cfg_name> <gpu_id> <run_dir> [uses_wgmma]

`uses_wgmma` (optional 6th arg) is the SASS-derived ground truth passed from the orchestrator:
"1" (wgmma present), "0" (absent), or "-"/absent (unknown).  It steers the bottleneck classifier
(overlap-bound enablement + the off-tensor-core gate) -- see ncu_profile._classify.
"""
import json
import sys
from dataclasses import asdict

from fa3agent import config as C
from fa3agent.perf.ncu_profile import profile


def _cfg_by_name(name):
    for c in list(C.FULL_CONFIGS) + list(C.DEV_CONFIGS):
        if c.name == name:
            return c
    raise SystemExit(f"unknown config {name}")


def _parse_uses_wgmma(argv) -> "bool | None":
    if len(argv) <= 6:
        return None
    v = argv[6]
    return True if v == "1" else (False if v == "0" else None)


if __name__ == "__main__":
    so_path, ns, cfg_name, gpu_id, run_dir = sys.argv[1:6]
    uses_wgmma = _parse_uses_wgmma(sys.argv)
    try:
        pf = profile(so_path, ns, _cfg_by_name(cfg_name), int(gpu_id), run_dir, uses_wgmma=uses_wgmma)
        print("PROFILE_JSON:" + json.dumps(asdict(pf)))
    except Exception as e:  # noqa
        print("PROFILE_JSON:" + json.dumps({"bottleneck": f"(profile runner failed: {e!r})",
                                             "config": cfg_name, "ranked": [], "key_metrics": {}}))
