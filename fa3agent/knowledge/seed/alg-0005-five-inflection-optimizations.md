---
id: alg-0005
type: alg
title: The 5 AVO inflection optimizations - high-value moves to try
tags: [optimization, branchless-rescale, pipeline-overlap, register-rebalance, bitmask-causal, single-pass-softmax]
confidence: source-reported
source: AVO sections 6-8, .research/avo.md
---
AVO's five largest gains were architectural inflection points (discrete jumps, not
gradual). Use as a checklist of high-value moves; each requires JOINT reasoning
across multiple HW subsystems, not isolated param tuning.

1. QK-PV interleaving + BITMASK causal masking (AVO v8). Interleave the two GEMMs;
   represent causal mask as a bitmask rather than per-element compare.
2. SINGLE-PASS softmax restructuring (v13). See [[alg-0004]].
3. BRANCHLESS predicated accumulator rescale + lighter fence (v19->v20, the largest
   single win: non-causal +8.1%, causal +1.6%). v19 used a conditional branch to
   skip rescale when max unchanged -> warp-sync overhead every K-block AND blocks
   lighter fences. Fix: ALWAYS compute the rescale factor, use a predicated select
   substituting 1.0 when no rescale needed (a multiply-by-1 is cheap vs sync). No
   branch => no warp divergence => can swap a blocking memory fence for a lighter
   non-blocking (ordering-only) fence. Applies only to FULLY-UNMASKED K-blocks
   (so causal gains less - keeps branched logic for masked blocks).
4. CORRECTION / PV-GEMM pipeline overlap (v29->v30, +1.1%/+0.4%). In a dual-Q-stage
   pipeline, let the correction warp start normalizing stage-1 output as soon as
   its PV GEMM finishes, overlapping with stage-2 PV GEMM (instead of waiting for both).
5. Per-warp-group REGISTER rebalance via setmaxnreg (v32->v33, +2.1%/~0%). Profiling
   showed the correction group spilling to local mem while softmax had headroom;
   moved 8 regs from softmax to other groups. DRIVEN BY SPILL PROFILING.

Diminishing returns: v1-v20 = naive->competitive (biggest gains); v21-v40 = smaller
compounding cycle-level scheduling / resource-allocation gains.
See [[ks-0004]] for setmaxnreg, [[pf-0001]] for the spill metrics that drove #5.
