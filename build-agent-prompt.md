# Goal

Build an autonomous agent that distills vLLM's fork of Flash Attention 3 (FA3) into a **single, self-contained CUDA file targeting Hopper**. The distilled kernel must:

- Have **no dependency on external libraries** (e.g. CUTLASS/CuTe, cuBLAS, etc.).
- Produce the **same output** as the original implementation (within the correctness criteria below).
- **Match or exceed** the original's performance across all configurations vLLM supports.

The agent should run as an optimization loop: generate a candidate, verify correctness, then profile and optimize, learning from each iteration until the targets are met.

---

## 1. Expected output

### Deliverable
- A single `.cu` file implementing the distilled attention kernel(s).

### Termination / success criteria
The agent keeps looping until **either**:
1. The optional max-step budget is reached (see §3), **or**
2. **Both** of the following hold simultaneously:
   - **Performance:** at least **1.0x** of the original Flash Attention across **all** sequence lengths / configs supported by vLLM, **and**
   - **Correctness:** the acceptance criteria below pass.

### Correctness acceptance criteria
Define a rigorous acceptance test such that, if we swap vLLM's Flash Attention module with the standalone distilled implementation, **the decoded tokens are unchanged**. Concretely:
- Tokens are selected by `argmax` under greedy decoding.
- The match must hold for **up to 100 sequential decoded tokens**.
- This requires very high numerical accuracy in **both prefill and decode** phases.

### Forbidden dependencies (anti-cheating)
The output `.cu` file **must not** use any of:
- CUTLASS, CuTe
- cuBLAS / cuBLASLt, cuDNN
- FlashInfer
- the original `_vllm_fa3_C`
- Triton
- any `at::scaled_dot_product_attention` / `at::native` attention path

**Allowed torch usage (boundary only, never in the compute path):** torch may appear *only* in the `mha_fwd` host function for tensor unpacking, output allocation, and the `TORCH_LIBRARY_EXPAND` registration. Torch **must not** appear inside any `__global__` kernel or be used to compute attention.

---

## 2. Agent components

Build the components below. Use your judgment on the best implementation for each (subagent, skill, RAG, tool, etc.).

> **Context discipline (applies to every component):** Components must **not** flood the main agent's context. Each component should extract/produce only the concise, processed insight needed (from profiling, docs, etc.) and pass *that* distilled information to the main agent — never raw dumps.

### 2.1 Profiling component (NCU / Nsys)
- Profile every candidate in detail and drive performance optimization **after** correctness is achieved. NCU is the more likely to be useful of the two.
- NCU emits a large volume of data that cannot be fed back to the agent directly without flooding its context. This component must **parse** that output and return only **concise, actionable feedback**.
- For reference on parsing/handling NCU output, see the NCU Report Skill <ncu-report-skill> and CUDAForge (<cuda-forge-code>).
- Where possible, return not just overall metrics but **per-operation detail** so the main agent can identify the specific bottleneck (memory-bound, latency-bound, etc.).
- For full usage details on the tools, consult NVIDIA's NCU/Nsys docs <ncu/nsys docs>.

### 2.2 Correctness / debugging component (Compute Sanitizer)
**When to use:** Run this component **only** when the kernel logic appears correct but the results are nonetheless wrong, and/or when repeated runs produce slightly different results (a strong sign of a race, uninitialized memory, or a synchronization bug). Do **not** run it routinely on every candidate — it is a targeted debugging tool, not a standard verification step.

Provide a mechanism to detect and debug:
1. **Race conditions** — use Compute Sanitizer's `racecheck`.
2. **Memory issues** (out-of-bounds reads/writes, misaligned accesses, hardware faults) — use `memcheck`.
3. **Uninitialized device global memory access** — use `initcheck`.
4. **Incorrect use of synchronization primitives** — use `synccheck`.

Reference the Compute Sanitizer docs <compute-sanitizer docs>, which cover all of the tools above.

### 2.3 SASS analysis component
- Analyze the SASS produced by the current candidate, and **compare** it against SASS from the original Flash Attention kernels or from equivalent kernels written with CUTLASS/CuTe or FlashInfer.
- Build this using <sass-analysis> and <cuda-binary-utilities>.

### 2.4 NVIDIA docs navigation component
- Provide a way to efficiently navigate the extensive NVIDIA docs (see what's available at <cuda-docs>) to guide building and debugging.
- The agent should use it to determine how to implement each operation (e.g. GEMM) most effectively for the target architecture / use case / shapes, and to identify transformations needed to reach the architecture's performance ceiling (e.g. `tcgen05` for GEMMs on Blackwell).
- Also leverage our other reference docs: GPU architecture tech specs (<architecture-specs>) and the GEMM/kernel design guides (<design-guides>).

### 2.5 Anti-cheating enforcement component
Make it **impossible** for the agent to cheat (e.g. by using cuBLAS/CUTLASS/torch/CuTe or any other forbidden library). For example:
- Compile with an include path containing **only** the CUDA toolkit — no CUTLASS/CuTe headers in the compile/run sandbox, so `#include <cutlass/...>` fails.
- Scan the built `.so` with `nm -D` / `ldd` for forbidden dynamic symbols/links (cutlass, `cute::`, cublas, cudnn, flashinfer, triton, etc.).

### 2.6 CUTLASS / CuTe / CuTeDSL inspiration component
Use the CUTLASS/CuTe/CuTeDSL library — its source <cutlass source> and docs <cutlass docs> — to inspire improvements to the current kernel and its subkernels, in two ways:
1. **Read** the source for ideas on how to improve the code.
2. **Implement** the current kernel/subkernel in CUTLASS/CuTe/CuTeDSL, run it, and if it's faster than the current candidate, inspect the resulting optimized/generated code/PTX/SASS to derive concrete hints the agent can reuse.

### 2.7 Reference-library optimization component
Similarly, productively use the following libraries (comparing source code, or running code and analyzing the resulting CUDA/PTX/SASS) to hint the agent toward optimizations:
- Thunderkittens (source: <thunderkittens source>)
- NVIDIA cccl (code: <cccl code>, docs: <cccl docs>)
- NVIDIA TransformerEngine (code: <transformerengine code>, docs: <transformerengine docs>)
- FlashInfer (code: <flashinfer code>)
- DeepGEMM (code: <deepgemm code>)
- FlashMLA (code: <flashmla code>)

---

## 3. Agent structure

- **LLM backend:** All (sub)agents must use Claude Code in headless mode, launched with `--dangerously-skip-permissions`.
- **Step budget:** Include an **optional** flag to limit the total number of loops/steps. If unset, the agent loops **indefinitely**, with the only valid termination being that it has reached **both** the 1.0x performance and the correctness requirements.
- **Correctness before performance:** Never waste time optimizing an incorrect kernel. After generating a kernel, **check correctness first**; if incorrect, fix that before doing any performance work.
- **In-loop distillation / learning:** The agent should evolve and learn from each iteration via some form of in-loop distillation. Distinguish between knowledge types — kernel-specific, profiling-specific, algorithmic/methodological, etc. — so each component can be improved separately with its own knowledge. At each step the agent should leverage what it learned previously **without** carrying a giant context.
- **Checkpointing & resume:** Checkpoint state (kernels, knowledge, state, logs, etc.) after **every** step so the lineage can be analyzed later and the run can resume from the most recent completed step after an interruption. Store checkpoints in `/code/users/goliaro/cuda-agent-checkpoints`. Add a mechanism to make resuming easy.
- **Evolution plot:** In addition to checkpointing, maintain a plot (similar to AVO's paper, fig <avo_mha_causal_neurips>), updated after each step, tracking the agent's evolutionary trajectory.
- **GPU parallelism:**
  - Parallelize across the machine's available GPUs when possible.
  - For **correctness checks**, you may run multiple kernels on the same GPU (given enough memory).
  - For **performance/profiling**, **never** run more than one kernel at a time on the same GPU.
- **Performance measurement:** Build a robust mechanism to reliably measure a candidate kernel's latency with CUDA events, ensuring proper warmups and repeats.
- **Knowledge sources:** Ensure the agent can take advantage of <KernelWiki> and the Flashinfer-Bench MLSys context solutions (<solutions>).

---

## 4. Validate after building

After constructing the agent, **briefly test it and each of its components end to end** before handing it off — don't assume it works untested.
- **Per-component smoke tests:** Exercise each component (§2) on a small, representative input and confirm it runs and returns the expected *distilled* output — e.g. the profiling component parses NCU output into concise feedback, Compute Sanitizer is invoked correctly, the SASS analyzer produces a comparison, the docs/library components return usable hints, and the anti-cheating checks actually reject a forbidden include/symbol.
- **End-to-end loop test:** Run the full loop for a small number of steps (e.g. via the step-budget flag) to confirm the candidate→correctness→profile→optimize cycle, checkpointing/resume, the evolution plot, and multi-GPU scheduling all work together.
- Keep these tests **brief** — the goal is to surface obvious breakage and wiring issues, not to fully converge the kernel. Report what you tested, what passed, and anything left unverified.

---

## 5. Useful references

- **FA3 distillation scope:** See <FA3 repo guide> for which functions from the Flash Attention repository the agent must distill, and which files are used by / compiled into FA3. Ignore files not listed there.
- **NVIDIA AVO agent** paper (<paper>) — inspiration for how to build the agent.
- **Kernel Design Agents (KDA)** code (<code>) and their Flashinfer-Bench MLSys context code (<code>) — inspiration for the agent and its components.
- **SyFI's MLSys Contest** (<code>) — inspiration.
- **UT Austin's CUCO agent** paper (<paper>) and code (<code>) — its focus is networking, so not all techniques transfer, but the harness and evolutionary components are worth studying.
- **Dr. Kernel's** paper (<paper>) and code (<code>) — inspiration for the sandbox/tooling/infra.
