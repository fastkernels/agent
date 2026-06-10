# SASS Analysis Reference

Consolidated reference for reading, analyzing, and optimizing NVIDIA SASS (Streaming
ASSembler) machine code. Combines what were previously three files: the SASS reading
guide, the Blackwell instruction-latency notes, and the per-architecture instruction-set
tables (the last copied from NVIDIA's CUDA Binary Utilities documentation,
https://docs.nvidia.com/cuda/cuda-binary-utilities/#instruction-set-reference).

## Contents

1. [SASS basics and generating SASS](#1-sass-basics-and-generating-sass)
2. [Instruction format](#2-instruction-format)
3. [Control codes (scheduling)](#3-control-codes-scheduling)
4. [PTX → SASS mapping](#4-ptx--sass-mapping)
5. [Key mnemonics and approximate latencies](#5-key-mnemonics-and-approximate-latencies)
6. [Ampere A100 (SM80) microarchitecture notes](#6-ampere-a100-sm80-microarchitecture-notes)
7. [Hopper H800/H100 (SM90) microarchitecture notes](#7-hopper-h800h100-sm90-microarchitecture-notes)
8. [Blackwell B200 (SM100) microarchitecture notes](#8-blackwell-b200-sm100-microarchitecture-notes)
9. [Identifying performance issues from SASS](#9-identifying-performance-issues-from-sass)
10. [Correlating with Nsight Compute](#10-correlating-with-nsight-compute)
11. [Tools and resources](#11-tools-and-resources)
12. [Per-architecture instruction-set tables](#12-per-architecture-instruction-set-tables)

---

## 1. SASS basics and generating SASS

SASS is the machine-level assembly for NVIDIA GPUs — architecture-specific, not portable
across major SM versions. On Volta and later, every instruction is 128 bits (16 bytes),
with scheduling control bits embedded per instruction (Maxwell/Pascal instead used one
control word per group of three instructions).

Pipeline: **CUDA C++ → PTX (virtual ISA) → SASS (real machine code)**. PTX is compiled to
SASS by `ptxas` at build time, or by the driver JIT at load time for PTX-only fatbins.

```bash
# Compile to cubin, then dump SASS
nvcc --cubin -arch=sm_90 -O3 -lineinfo kernel.cu
cuobjdump --dump-sass kernel.cubin

# Or directly from an executable / fatbin (filter one arch if several embedded)
cuobjdump --dump-sass ./my_app
cuobjdump --dump-sass --gpu-architecture sm_90 ./my_app

# Source line correlation
nvdisasm -g kernel.cubin            # -gi to include inlined-function info

# Resource usage (registers, shared memory, spills)
cuobjdump --dump-resource-usage kernel.cubin
nvcc -arch=sm_90 -O3 -Xptxas -v kernel.cu 2>&1 | grep "ptxas info"

# Control flow graph (DOT format) and register liveness
nvdisasm -cfg kernel.cubin > cfg.dot
nvdisasm -plr kernel.cubin
```

Key `cuobjdump` flags: `--dump-sass`, `--dump-ptx`, `--dump-resource-usage`,
`--gpu-architecture <arch>`, `--sort-functions`.
Key `nvdisasm` flags: `-c` (code only), `-g` / `-gi` (source lines), `-hex` (show
encoding, including the control bits), `-cfg` (control-flow graph), `-plr` (register
liveness), `-json` (JSON output).

Compiler Explorer (godbolt.org) supports CUDA and shows live PTX and SASS for quick
experiments without a local toolchain.

## 2. Instruction format

```
/*0080*/    @P0 FFMA.FTZ R4, |R2|.reuse, -RZ, R5 ;   /* 0x800000ff02047223 */
                                                        /* 0x001fc800000e0400 */
```

| Part | Meaning |
|---|---|
| `/*0080*/` | Instruction address (byte offset, multiple of 16) |
| `@P0` | Guard predicate (execute only if P0 true) |
| `FFMA` | Opcode |
| `.FTZ` | Modifier (flush-to-zero) |
| `R4` | Destination register |
| `\|R2\|.reuse` | Source 1: absolute value + operand-reuse flag |
| `-RZ` | Source 2: negated zero register |
| `R5` | Source 3 |
| `/* hex */` | 128-bit encoding (two 64-bit words; second word holds control bits) |

### Operand types

| Notation | Meaning |
|---|---|
| `R0`–`R254` | 32-bit general registers |
| `R255` / `RZ` | Zero register (always 0) |
| `R0R1` / `R2.64` | 64-bit register pair |
| `UR0`–`URx` | Uniform registers (one value per warp) |
| `P0`–`P6` | Predicate registers (per-thread boolean) |
| `P7` / `PT` | Always-true predicate |
| `UP0`–... / `UPT` | Uniform predicates (Ampere+) |
| `c[X][Y]` | Constant memory bank X, offset Y |
| `desc[URX][RY]` | Memory descriptor (Hopper+) |
| `gdesc[URX]` | Global memory descriptor (Hopper+) |
| `tmem[URX]` | Tensor memory (Blackwell SM100) |
| `0x1234` | Immediate constant |

### Predicate syntax

`@P0 INST` (execute if true), `@!P0 INST` (if false), `@PT INST` (always),
`@!PT INST` (never — a scheduling-slot NOP).

### Common modifiers

| Modifier | Meaning |
|---|---|
| `.FTZ` | Flush denormals to zero |
| `.SAT` | Saturate to [0,1] |
| `.RN/.RM/.RP/.RZ` | Rounding mode |
| `.E` | Extended (64-bit) address |
| `.128/.64/.32` | Data size |
| `.STRONG/.SYS/.CTA/.GPU` | Memory ordering / scope |
| `.reuse` | Keep operand in the operand-collector cache |
| `.WIDE` | Wide result (32×32→64) |
| `.HI/.LO` | High/low half of a wide result |

## 3. Control codes (scheduling)

On Volta+ (SM70+), each 128-bit instruction embeds its own scheduling metadata. **Not
officially documented** — reverse-engineered by maxas, CuAssembler, and academic papers.
`nvdisasm` does not decode these bits; view them via `-hex` and decode with CuAssembler
or the interactive viewers listed in §11.

| Field | Bits | Description |
|---|---|---|
| Stall count | 4 (0–15) | Min cycles before the next instruction issues |
| Yield flag | 1 | Hint: prefer switching to another warp |
| Write barrier | 3 (0–5) | Scoreboard armed when this instruction finishes writing |
| Read barrier | 3 (0–5) | Scoreboard armed when this instruction finishes reading |
| Wait barrier mask | 6 | Which scoreboards must clear before issue |
| Reuse flags | 4 | Which operands to cache in the operand collector |

**Stall count** handles fixed-latency dependencies: 0 = back-to-back issue, 1–4 typical
for arithmetic chains (FFMA dependent latency ≈ 4 cycles), 5–15 for longer spacing.

**Barriers (6 scoreboards)** handle variable-latency producers (memory loads, tensor
ops). The compiler interleaves independent work between producer and consumer:

```
LDG.E R4, [R2]        ; write-barrier = 1 (arms scoreboard 1 when the load lands)
IADD3 R6, R7, R8, RZ  ; independent work, no barrier wait
FFMA  R9, R10, R11, RZ
IMAD  R12, R4, R5, RZ ; wait-barrier-mask = 0b000010 (waits on scoreboard 1)
```

**Yield**: set → scheduler prefers another warp (good after loads); clear → prefers the
current warp (good in tight compute loops). When yield is set, the reuse flags are
ineffective.

## 4. PTX → SASS mapping

Knowing which SASS op a PTX instruction lowers to makes disassembly much faster to read.
(Exact mnemonics vary slightly by CUDA version; confirm with `cuobjdump` on your build.)

| PTX | SASS | Arch |
|---|---|---|
| `cp.async` | `LDGSTS` | SM80+ |
| `cp.async.commit_group` / `wait_group` | `LDGDEPBAR` / `DEPBAR` | SM80+ |
| `cp.async.bulk.tensor.*` (TMA load/store) | `UTMALDG` / `UTMASTG` | SM90+ |
| `cp.async.bulk` (non-tensor bulk copy) | `UBLKCP` (`UBLKRED` for reduce) | SM90+ |
| `prefetch.tensormap` | `UTMAPF` | SM90+ |
| `wgmma.mma_async` (FP16/BF16) | `HGMMA` | SM90 |
| `wgmma.mma_async` (FP8) | `QGMMA` | SM90 |
| `wgmma.mma_async` (INT8) / (bin) | `IGMMA` / `BGMMA` | SM90 |
| `mma.sync` (FP16/TF32) / (INT) / (FP64) | `HMMA` / `IMMA` / `DMMA` | SM70+/SM80+ |
| `tcgen05.mma` kind f16/tf32 | `UTCHMMA` | SM100 |
| `tcgen05.mma` kind f8f6f4 / mx | `UTCQMMA` (FP8 family) / `UTCOMMA` (FP4) | SM100 |
| `tcgen05.mma` integer kinds | `UTCIMMA` | SM100 |
| `tcgen05.cp` (SMEM→TMEM) | `UTCCP` | SM100 |
| `tcgen05.ld` / `tcgen05.st` | `LDTM` / `STTM` (also `LDT`/`STT`) | SM100 |
| `tcgen05.shift` | `UTCSHIFT` | SM100 |
| `ldmatrix` / `stmatrix` | `LDSM` / `STSM` | SM75+ / SM90+ |
| `mbarrier.*` | `SYNCS` (sync-unit ops) | SM90+ |
| `setmaxnreg` | `USETMAXREG` | SM90+ |
| `elect.sync` | `ELECT` | SM90+ |
| `redux.sync` | `REDUX` | SM80+ |
| `barrier.cluster.*` | `UCGABAR_ARV` / `UCGABAR_WAIT` | SM90+ |
| Cluster launch control (CLC) | `UGETNEXTWORKID` | SM100 |
| `griddepcontrol.launch_dependents` | `PREEXIT` | SM90+ |

Note on Blackwell MMA naming: NVIDIA's ISA listing has both the uniform tensor-core
family (`UTCHMMA`/`UTCQMMA`/`UTCOMMA`/`UTCIMMA`, the tcgen05 path) and plain warp-level
`OMMA`/`QMMA` ("FP4/FP8 MMA across a warp", the `mma.sync`-style path used on consumer
SM120 and for non-tcgen05 code). The B200 microbenchmark paper (arXiv 2512.02189) reports
the mapping with the shortened names HMMA/QMMA/OMMA/IMMA; in disassembly, expect the
`UTC*` forms for tcgen05.

## 5. Key mnemonics and approximate latencies

Latencies are dependent-issue latencies from microbenchmark literature (Jia et al. for
Volta/Turing, follow-ups for Ampere, arXiv 2501.12084 for Ampere/Ada/Hopper, arXiv
2507.10789 / 2512.02189 for Blackwell). Treat them as approximate; they vary with clocks
and CUDA version.

### Floating point

| Mnemonic | Description | Latency |
|---|---|---|
| `FFMA` / `FADD` / `FMUL` | FP32 FMA / add / mul | ~4 cycles (SM70–SM100) |
| `HFMA2` / `HADD2` / `HMUL2` | Packed FP16×2 | ~4 cycles |
| `DFMA` | FP64 FMA | ~4 cycles issue; throughput limited by FP64 unit count |
| `MUFU` | Transcendental (rsqrt, exp2, lg2, sin...) | issued to SFU; result latency notably longer than core ALU (~ low tens of cycles), ¼-rate issue |

### Integer / address (the "overhead" pipe)

| Mnemonic | Description | Latency |
|---|---|---|
| `IMAD` | Integer multiply-add (also used to emulate MOV/shifts) | ~4–5 cycles |
| `IADD3` / `LEA` / `LOP3` / `SHF` / `PRMT` / `MOV` | ALU ops | ~4 cycles |

(The original notes listed 2 cycles for several ALU ops; microbenchmark literature for
Volta+ consistently measures the common ALU dependent latency at ~4 cycles, so these have
been revised. Sub-4-cycle dependent issue is not achievable on these pipelines.)

### Tensor core

| Mnemonic | Description | Latency |
|---|---|---|
| `HMMA.16816.F16` | mma m16n8k16 FP16 | ~24 cycles completion (A100/RTX4090/H800 all ≈24.0–24.6); m16n8k8 ≈16–18 |
| `HMMA` TF32 | mma m16n8k8 TF32 | ~24.5 (H800) / 26.3 (A100) / 33.4 (RTX4090) |
| `IMMA` | mma m16n8k32 INT8 | ~24 cycles; `DMMA` shape-dependent |
| `HGMMA`/`QGMMA`/`IGMMA` | SM90 warpgroup MMA | scales with N: ~32 (n64) / 64 (n128) / 128 (n256); small N: n32 ≈ 24 (SS) / 16 (RS), n16 ≈ 20/13, n8 ≈ 18/13 |
| `UTC*MMA` | SM100 tcgen05 MMA | ~11 cycles, nearly constant across tile sizes |

Sparse and dense mma have equal latency; sparse raises throughput instead. On Hopper,
plain `mma` reaches only ~63% of peak on average (vs 95%+ on A100) and sparse mma
averages only ~1.42× over dense — wgmma is required to saturate SM90 tensor cores
(see §7).

### Global memory

Measured dependent-load latencies in cycles (P-chase, arXiv 2501.12084; CUDA 12.6):

| Level | RTX 4090 (SM89) | A100 PCIe (SM80) | H800 PCIe (SM90) |
|---|---|---|---|
| L1 hit | 32 | 33 | 32 |
| L2 hit | 273 | 203–408 | 264–502 |
| Global (DRAM) | 571 | 566 | 656 |

The L2 ranges on A100/H800 are real, not noise: their L2 is **two partitions**, and GPCs
are partitioned to match. Fine-grained P-chase separates four cases — A100: near hit 208,
far hit 357, near miss 475, far miss 623; H800: near hit 258, far hit 414, near miss 556,
far miss 744. Average global latency ≈ (near miss + far miss)/2. RTX 4090's L2 is
monolithic, hence a single value. Roughly: L2 ≈ 6.5× L1 latency; global ≈ 2.1× L2.

| Mnemonic | Description | Notes |
|---|---|---|
| `LDG` | Global load | latencies above |
| `STG` | Global store | variable |
| `LDGSTS` | cp.async global→shared | non-blocking |
| `UTMALDG` | TMA load | ≈ global latency + ~170 cycles (TMA unit + mbarrier overhead); bypasses L1, only L2-cached |
| `ATOMG` / `RED` | Global atomic / reduction | variable |

### Shared memory

| Mnemonic | Description | Latency |
|---|---|---|
| `LDS` / `STS` | Shared load / store | ~29–31 cycles measured on SM80/SM89/SM90 (bank conflicts add) |
| `LDS` via DSM interface | own SMEM through `mapa` path | ~33 cycles (interface overhead even without SM-to-SM hop) |
| Inter-SM DSM access | cluster size 2 | ~181 cycles (~30% below L2; 6.1× better than a global-memory bounce at ~1110); CS 2→16: 184–213 |
| `LDSM` / `STSM` | ldmatrix / stmatrix | ~LDS-like |
| `ATOMS` | Shared atomic | variable |

### Control flow & special

`BRA` (branch), `EXIT`, `BSSY`/`BSYNC` (divergence set/reconverge), `BAR.SYNC`
(`__syncthreads()`), `DEPBAR` (cp.async wait; `DEPBAR.LE SB0, 0x0` = wait all),
`MEMBAR` (`__threadfence()`, `.CTA/.GPU/.SYS`), `WARPSYNC`, `S2R` (special register read,
e.g. `S2R R0, SR_TID.X`), `VOTE`, `SHFL`, `REDUX` (Ampere+), `ELECT` (SM90+), `NOP`.

## 6. Ampere A100 (SM80) microarchitecture notes

Sources: the A100 columns of arXiv 2501.12084 (same methodology and CUDA 12.6 run as the
Hopper numbers in §7), plus the NVIDIA GA100 whitepaper for configuration facts. For
per-instruction clock-cycle tables beyond what is here, the canonical references are
Abdelkhalik et al., arXiv 2208.11174 (HPEC'22, PTX/SASS instruction latencies on A100)
and Sun et al., arXiv 2206.02874 (TPDS'23, mma/mma.sp tensor-core latency, throughput,
and numerics on Turing/Ampere); their tables are not reproduced here.

Configuration (A100 PCIe 40 GB): 108 SMs × 64 FP32 cores (separate 64-wide INT32 pipe,
32 FP64 units/SM), 1410 MHz boost, 40 MB L2 in **2 partitions**, 192 KB combined L1/SMEM
per SM (up to 164 KB SMEM carve-out), 432 3rd-gen tensor cores, HBM2e 1555 GB/s.
Consumer GA10x (SM86/89) differs significantly: 128 KB combined L1/SMEM, doubled FP32 at
the cost of a shared INT path, 2 FP64 units/SM.

### Measured memory behavior

Latencies are in §5 (L1 33, SMEM 29, L2 near/far hit 208/357, near/far miss 475/623,
global 566 cycles). Throughput: L1 ~107 B/clk/SM, SMEM ~126 B/clk/SM, L2 ~2008 B/clk
aggregate, global 1407 GB/s (90% of peak). The partitioned-L2 near/far asymmetry already
exists here — Hopper inherited the design, Blackwell B200 doubled it to 4 partitions.

### Tensor core (mma) — measured on A100

| Instruction | Dense LAT / TFLOPS | Sparse LAT / TFLOPS |
|---|---|---|
| m16n8k16 FP16/FP16 | 24.6 / 310.6 | 24.5 / 622.8 |
| m16n8k8 FP16/FP16 | 17.7 / 310.0 | 17.3 / 408.4 |
| m16n8k8 TF32/FP32 | 26.3 / 151.5 | 26.7 / 301.5 |
| m16n8k32 INT8/INT32 | 26.0 / 607.6 | 26.6 / 1210 |

Dense mma reaches >95% of the A100's peak (312 TFLOPS FP16) — unlike Hopper, where mma
leaves ~37% on the table and wgmma is required (§7). Sparse mma achieves the theoretical
2× only at the larger k shapes; small-shape sparse (m16n8k8 FP16: 408 vs 624 theoretical)
underdelivers. Sparse and dense latencies are equal — sparsity buys throughput, not
latency.

### Ampere-specific SASS landmarks

`LDGSTS` (cp.async) debuts here: the first asynchronous global→SMEM copy, bypassing the
register file, paired with `LDGDEPBAR`/`DEPBAR` commit/wait groups — the SM80 pipelining
idiom that TMA later replaced. Also new vs Turing: `DMMA` (FP64 tensor core), `REDUX`,
`HMNMX2`, uniform predicates (`UPx`), BF16/TF32 types, `mma.sp` (sparse). L2 residency
control (`cudaAccessPolicyWindow`) lets kernels pin a working set in the 40 MB L2.
Asynchronous barriers (`cuda::barrier`) exist but are software-managed in SMEM — no
hardware transaction barrier (that arrives with SM90 mbarrier+TMA). Absent entirely on
SM80, so their SASS will never appear in A100 disassembly: TMA (`UTMALDG`...), bulk
copies (`UBLKCP`), clusters/DSM (`UCGABAR_*`, `mapa`), `*GMMA`, `ELECT`, `USETMAXREG`,
DPX hardware (DPX intrinsics compile but run as plain emulation), and everything
TMEM/tcgen05.

## 7. Hopper H800/H100 (SM90) microarchitecture notes

Source: arXiv 2501.12084 (Luo et al., "Dissecting the NVIDIA Hopper Architecture", v2
Sep 2025; extends their IPDPS'24 paper). Measured on H800 PCIe vs A100 PCIe and RTX 4090,
CUDA 12.6. H800 PCIe has the same compute configuration as H100 PCIe (114 SMs, 50 MB L2,
2039 GB/s HBM2e, 350 W), so these numbers are a good SM90 proxy; SXM parts clock higher.
Artifact: https://github.com/HPMLL/NVIDIA-Hopper-Benchmark.

### Memory throughput

| Level | RTX 4090 | A100 | H800 |
|---|---|---|---|
| L1 (B/clk/SM, float4) | 121 | 107 | 124 |
| SMEM (B/clk/SM) | 126 | 126 | 128 |
| L2 (B/clk, aggregate) | 1708 | 2008 | 4472 |
| Global (% of theoretical) | 92% | 90% | 91% |

L1/SMEM per-clock bandwidth is unchanged across the three generations; Hopper's big jump
is L2 (2.6× RTX 4090, 2.2× A100 per clock; L2:global ratio 4.23×). On RTX 4090, scalar
FP32 access reaches only half the L1/SMEM throughput of `float4` (MIO throttle) — use
vectorized 128-bit accesses.

### wgmma behavior (H800)

Latency scales with N as listed in §5. Key measured findings:

- **N ≥ 64 is required for near-peak throughput** (95%+ of peak with zero inputs). Below
  n64, SS-mode latency exceeds RS and throughput drops steeply (n32 SS ≈ 477 vs 710 RS
  TFLOPS FP16).
- SS vs RS makes no difference for N ≥ 64 — async execution hides the SMEM reads.
- **Sparse wgmma**: SS latency 144 vs RS 128 cycles, and SS *underperforms* RS by a wide
  margin, because SS reads the full m×k from SMEM and prunes 2:4 in-instruction (2× the
  SMEM traffic), while RS reads pre-pruned m×k/2 from registers. Prefer RS for sparse.
- **Power throttling is a real measurement and deployment hazard**: continuously issuing
  wgmma with random inputs hits the H800-PCIe 350 W limit and drops clocks below the
  1620 MHz spec (zero inputs stay under 200 W and look deceptively fast). Random-input
  dense FP16 lands at ~595–600 TFLOPS vs ~729 zero-input.
- Energy: wgmma dense is only ~0.67× the TFLOPS/W of mma dense — wgmma buys peak
  performance, not efficiency.

### TMA (UTMALDG/UTMASTG)

- Latency ≈ global-memory latency + ~170 cycles (TMA unit + mbarrier init/arrive/wait);
  one fewer P-chase inflection point than LDG because TMA bypasses L1 (L2-only caching).
- Throughput: >1800 GB/s (≈ peak) with large per-instruction loads and enough CTAs, for
  non-tensor, 2D-, and 3D-tensor copies.
- **1D *tensor* loads are capped** at a 256-element box per dimension (≤2 KB even with
  8-byte types) — use non-tensor 1D bulk copies instead.
- Box shape matters: grow the x (contiguous) dimension; large y/z box dimensions reduce
  throughput, increasingly so with more CTAs.
- Application level: TMA + producer-consumer pipelining in a CUTLASS-style FP16 GEMM
  reaches ~600 TFLOPS vs ~450 without TMA (1.5×) for n > 7000 with CTA tile 128×256×64.

### Distributed shared memory (cluster SM-to-SM)

- Going through the DSM interface costs even on local SMEM: 33 cycles vs 29 direct, and
  ~205 GB/s vs 225 GB/s peak per SM.
- Inter-SM: ~181 cycles at cluster size 2 (≈30% lower latency than L2; 6.1× lower than
  bouncing through global), rising to 184–213 cycles at cluster sizes up to 16.
- SM-to-SM network peak ≈ 3.28 TB/s at CS=2, dropping to ≈2.78 TB/s at CS=4 and further
  as more blocks contend; broadcast-style patterns degrade worst with cluster size.
- The `LoadBalancing` cluster scheduling policy (blocks may share an SM) is slightly
  fastest in most patterns. Histogram-style apps gained >30% with well-chosen CS; optimal
  CS depends on block size (small clusters for large blocks).

### DPX (VIMNMX / VIADDMNMX / VIMNMX3)

Hardware-accelerated per-SM on Hopper (throughput scales with SM count and dips when
grid size just exceeds an SM multiple). 16-bit fused min/max+add/ReLU ops gain up to 13×
over the software emulation used on SM80/SM89; some simple 32-bit ops (e.g.
`__viaddmax_s32` → VIMNMX) show no gain over the old IMNMX path. Smith-Waterman with S16
gains 4.75×. Not all DPX intrinsics are wins — check the instruction-level numbers before
restructuring an algorithm around them.

### mma vs wgmma summary

Plain `mma` (HMMA) on Hopper averages only ~63% of peak (vs >95% on A100), and sparse mma
only ~1.42× dense. The H800's higher mma energy efficiency (≈1.5× A100's TFLOPS/W) makes
mma attractive for efficiency-bound work, but saturating SM90 tensor cores requires wgmma
with N ≥ 64 — which is exactly what the GMMA SASS in §5 shows compilers emitting.

## 8. Blackwell B200 (SM100) microarchitecture notes

Sources: arXiv 2512.02189 (B200 vs H200 microbenchmarks, Dec 2025), arXiv 2507.10789
(consumer GB203/SM120 in RTX 5080 vs GH100/H100-PCIe, Jul 2025), SemiAnalysis
"Dissecting Nvidia Blackwell". **The two papers measure different chips** — GB203
numbers (smaller L1/L2, 2 FP64 units/SM, GDDR7) must not be read as B200 figures. The
original notes mixed them; the tables below are relabeled accordingly.

Consumer SM120 (GB203) compute findings worth knowing when reading RTX-Blackwell SASS:
unified INT32/FP32 cores cut mixed INT/FP-workload latency by ~50% vs GH100 (no separate
INT pipe to ping-pong between), while GH100 wins on pure-INT32, pure-FP32, and FP64
throughput. GB203's 2 FP64 units/SM (vs 64 on GH100) appear to exist mainly for ISA
completeness — the paper suggests FP64 may be partially emulated via FP32/tensor paths.

### Tensor core (tcgen05) — B200, verified

| Property | Value |
|---|---|
| Single-instruction latency (FP16, dep-chain) | 11.0–11.4 cycles, ~constant across tile size (m64n64k16 → m256n256k16) |
| vs Hopper wgmma | 32 / 64 / 128 cycles for n64/n128/n256 → 2.9–11.2× lower latency |
| FP8 / FP6 / FP4 latency | ~11.8 / 12.3 / 12.6 cycles |
| FP64 MMA | separate DMMA path (NOT tcgen05/TMEM); ~14 cycles; **doubled FP64 units** vs claim of 2/SM |
| FP16-in / FP32-accum penalty | halves throughput vs FP16 accum — the accumulator datapath is the limiter |
| Warp-level issue | single thread issues per CTA; no 4-warp wgmma sync → 18–23% fewer scheduler stalls in memory-bound kernels |
| M=64 shapes | ~50% of theoretical peak (hardware characteristic, per CUTLASS docs and community benchmarks); prefer M=128/256 and max N |
| Useful MMA ILP | ~6 independent instructions; saturation ≈ 25 active warps (SemiAnalysis/community microbenchmarks — not in the arXiv papers) |

Measured DGEMM FP64 on B200: 36.3 TFLOPS = 80.7% of the 40 TFLOPS peak (vs H200 18.9,
55.6% of 34) — consistent with doubled FP64 units plus TMEM-assisted accumulation.

### Memory — labeled by chip

| Access | Value | Chip / source |
|---|---|---|
| Register file | direct (0 extra) | all |
| TMEM end-to-end access, cache-miss path | ≈420 cycles (58% lower than Hopper's ~1000-cycle global path) | B200. **Note**: this is the end-to-end miss-path comparison, not a raw on-chip read latency |
| TMEM bandwidth | 16 TB/s read, 8 TB/s write per SM, additive with L1/SMEM bandwidth | B200 |
| TMEM optimal tile | 64×64 elements (4 KB FP8); <32×32 → ~45% of peak BW; >128×128 → multi-phase transfers, −30% throughput | B200 |
| SMEM | ~20–30 cycles; 128 B/cycle per SM | both |
| L1 hit | ~30–40 cycles | GB203 measured; B200 similar order |
| L1/SMEM capacity | 256 KB combined per SM (B200/SM100); **128 KB on consumer SM120/GB203** | per-chip — original notes listed 128 KB for "Blackwell" |
| L2 hit | 358 cycles (GB203, monolithic) vs 273 (GH100, 2 partitions; consistent with H800's 258/414 near/far hit, §7) | GB203 paper. B200's L2 is **4 partitions**; its hit latency was not reported there |
| Global (DRAM) | ~877 cycles (GB203 GDDR7) vs ~659 (GH100 HBM, matching the H800's 656 in §7) | GB203 paper; do not assume 877 for B200 HBM3e |
| STREAM Triad sustained | B200 7.48 TB/s (93.5% of 8 TB/s) vs H200 4.38 | B200 paper |

### Synchronization cost estimates

| Operation | Estimated cost |
|---|---|
| `mbarrier.arrive` / `try_wait` hit | ~4–8 cycles |
| `mbarrier.try_wait` miss (spin) | 100+ cycles |
| `bar.sync` / `__syncthreads` | ~20–30 cycles |
| `cluster.sync` | ~50–100 cycles (cluster-size dependent) |
| `fence.proxy.async` | ~4 cycles |

These are order-of-magnitude estimates carried over from the original repo notes — they
do not appear in either cited arXiv paper, so treat them as unverified. Barrier cost
compounds regardless — e.g. 4 barriers per K-block × 112 K-blocks ≈ 1800 barrier cycles
per tile (~0.9 µs at 2 GHz).

### Register file

64K × 32-bit registers per SM; max 255/thread. At 128 threads, 255 regs/thread uses
49.8% of the file; at 256 threads, 99.7%. Low usage (e.g. 48 regs at 128 threads) leaves
large headroom — the compiler may be spilling or recomputing values that could be cached;
try `--maxrregcount=255` or `128`. cuBLAS's BF16 GEMM uses 255 registers.

### Other B200 facts (from the microbenchmark paper)

Dual-die, 208 B transistors, 148 SMs across 8 GPCs, 8 HBM3e stacks, NV-HBI unifying the
dies into one device. Hardware Decompression Engine: output throughput ~84–539 GB/s by
format (ANS 539, Bitcomp 462, zstd 155, gzip slowest at 84; the paper's prose quotes
"42–462", but 42 is gzip's *input* rate and ANS tops its own table), latency ≈0.2–1.3 ms
on 100 MB blocks, output ceiling ≈ 200 GB/s in the LZ4 compression-ratio sweep; optimal
batching 32 KB chunks × 16 concurrent (small files) or 256 KB × 4 (large). TMEM hit rates
of 61–82% in end-to-end workloads.

### Optimization implications (memory-bound / skinny-M kernels)

1. Instruction count per pipeline stage matters — fewer non-MMA instructions = more idle
   time filled, better TMA-compute overlap.
2. Register headroom is often free performance (see above).
3. Barrier overhead compounds per K-block; amortize by grouping K-blocks per barrier
   cycle.
4. tcgen05 at M=64 caps at ~50% peak — the tensor op finishes quickly, so the surrounding
   scalar work dominates; minimize it.

Compiler flags worth sweeping: `--maxrregcount={128,255}`, `-Xptxas -O3`,
`-Xptxas --allow-expensive-optimizations=true`, `--use_fast_math`, and with/without
`-lineinfo` (debug info can perturb register allocation).

## 9. Identifying performance issues from SASS

1. **Latency not hidden** — producer and consumer adjacent:
   `LDG.E R4,[R2]` immediately followed by `FFMA R6,R4,...` stalls on the scoreboard.
   Fix: interleave independent instructions.
2. **Register spilling** — any `LDL`/`STL` (local memory) traffic. Fix: reduce pressure
   (`__launch_bounds__`, simplify, recompute less).
3. **NOP padding** — `NOP` with stall=15 chains: compiler found no useful work (very
   short kernel or insufficient ILP).
4. **Warp divergence** — `BSSY`/`BSYNC` pairs around branches; deep nesting = costly
   reconvergence.
5. **Scoreboard pressure** — count distinct barriers in use. All 6 = maximum
   variable-latency ops in flight; fewer = room to overlap more loads/MMA.

Checklist: instruction mix (FFMA/HMMA vs LDG/LDS vs IMAD/NOP/BAR overhead) · distance
from loads to consumers · stall-count distribution · LDL/STL presence · LDSM:HMMA ratio ·
LDGSTS+DEPBAR pipelining · BSSY/BSYNC complexity · register count vs occupancy.

## 10. Correlating with Nsight Compute

| ncu stall reason | SASS cause |
|---|---|
| Long scoreboard | Waiting on LDG/LDL result |
| Short scoreboard | Waiting on LDS/HMMA/MUFU result |
| Wait | Fixed-latency dependency (stall count) |
| Math pipe throttle | FMA/tensor pipe FIFO full (compute saturated) |
| Barrier | At BAR.SYNC |
| LG throttle | Too many outstanding LDG/STG |
| MIO throttle | Too many outstanding LDS/STS/MUFU |
| Not selected | Eligible but another warp chosen (occupancy may be reducible) |

Use `ncu --section SourceCounters` with `-lineinfo` to map stalls to source lines, and
`ncu --page source` to view SASS alongside source with per-instruction metrics.

## 11. Tools and resources

| Tool | Purpose |
|---|---|
| `cuobjdump --dump-sass` | Extract SASS from binaries |
| `nvdisasm` | CFG, liveness, source correlation, hex encodings |
| `ncu --page source` | SASS with per-instruction profiling metrics |
| Compiler Explorer (godbolt.org) | Live CUDA → PTX/SASS in the browser |
| [CuAssembler](https://github.com/cloudcores/CuAssembler) | Unofficial assembler; decodes control codes |
| [DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS) | Community instruction docs (SM89/SM90a) |
| [maxas wiki](https://github.com/NervanaSystems/maxas/wiki/Control-Codes) | Control-code format reference (Maxwell-era, foundational) |
| [Kuter Dinel's viewer](https://kuterdinel.com/nvidia-sass-control-code-viewer.html) | Interactive control-code decoder |
| arXiv 2208.11174 | Ampere A100 PTX/SASS instruction latencies (Abdelkhalik et al., HPEC'22) |
| arXiv 2206.02874 | Tensor cores via microbenchmarks on Turing/Ampere — mma/mma.sp latency, throughput, numerics (Sun et al., TPDS'23) |
| NVIDIA GA100 whitepaper | Official Ampere architecture spec (images.nvidia.com, "nvidia-ampere-architecture-whitepaper") |
| arXiv 2501.12084 | Hopper (SM90) microbenchmarks — memory, wgmma, TMA, DSM, DPX; artifact: github.com/HPMLL/NVIDIA-Hopper-Benchmark |
| arXiv 2512.02189 | B200 (SM100) microbenchmarks — TMEM, tcgen05, DE |
| arXiv 2507.10789 | Consumer Blackwell GB203 (SM120) vs GH100 microbenchmarks |
| SemiAnalysis "Dissecting Nvidia Blackwell" | tcgen05 PTX/SASS deep-dive |

## 12. Per-architecture instruction-set tables

What changes between architectures, at a glance:

- **Turing → Ampere/Ada**: adds `LDGSTS`/`LDGDEPBAR` (cp.async), `DMMA`, `REDUX`,
  `HMNMX2`, `I2FP`/`F2IP`, `UF2FP`. Drops `RTT`.
- **Ampere → Hopper**: adds the warpgroup MMA family (`HGMMA`/`QGMMA`/`IGMMA`/`BGMMA`),
  TMA (`UTMALDG`/`UTMASTG`/`UTMAREDG`/`UTMAPF`/`UTMACCTL`/`UTMACMDFLUSH`), bulk copies
  (`UBLKCP`/`UBLKPF`/`UBLKRED`), DSMEM async (`STAS`/`REDAS`), the sync unit (`SYNCS`,
  mbarrier ops), cluster barriers (`UCGABAR_*`, `CGAERRBAR`), `ELECT`, `WARPGROUP`,
  `USETMAXREG` (setmaxnreg), `FENCE`, `LDGMC`, `PREEXIT`, SIMD int ops (`VIADD`,
  `VIMNMX`...), `ACQBULK`, `ENDCOLLECTIVE`.
- **Hopper → Blackwell**: drops the `*GMMA` warpgroup family; adds the tensor-core memory
  family (`UTCHMMA`/`UTCIMMA`/`UTCOMMA`/`UTCQMMA`, `UTCCP`, `UTCBAR`, `UTCSHIFT`,
  `UTCATOMSWS`, `LDT`/`LDTM`/`STT`/`STTM` for TMEM), warp-level `OMMA`/`QMMA` (FP4/FP8),
  a much richer uniform datapath (`UFADD`/`UFFMA`/`UFMUL`, uniform conversions/compares,
  `CREDUX`, `LDCU`/`CS2UR`), CLC (`UGETNEXTWORKID`), `UMEMSETS`, `UREDGR`/`USTGR`,
  `UVIRTCOUNT`, `ACQSHMINIT`, plus FP32 pair ops (`FADD2`/`FFMA2`/`FMUL2`,
  `FMNMX3`...). Drops `BMMA` and texture-era `MOV32I` remains.

The complete tables below are reproduced from NVIDIA's CUDA Binary Utilities
documentation (instruction-set reference). Note the source lists Blackwell as Compute
Capability 10.0 **and 12.0** in one table — SM100 and SM120 share the listed opcode set,
but SM120 lacks TMEM, so the tcgen05/TMEM rows (`UTC*`, `LDTM`/`STTM`) apply to SM100
only, while `OMMA`/`QMMA` are the SM120 MMA path. Check NVIDIA's page for the version
matching your CUDA toolkit, as the set grows over time.

## 12.1. Turing Instruction Set[](#turing-instruction-set "Permalink to this headline")

> The Turing architecture (Compute Capability 7.5) have the following instruction set format:

```
(instruction) (destination) (source1), (source2) ...
```

Valid destination and source locations include:

* RX for registers
* URX for uniform registers
* SRX for special system-controlled registers
* PX for predicate registers
* c[X][Y] for constant memory

[Table 6](#turing-turing-instruction-set-table) lists valid instructions for the Turing GPUs.

Table 6. Turing Instruction Set[](#turing-turing-instruction-set-table "Permalink to this table")


| Opcode | Description |
| --- | --- |
| **Floating Point Instructions** | |
| FADD | FP32 Add |
| FADD32I | FP32 Add |
| FCHK | Floating-point Range Check |
| FFMA32I | FP32 Fused Multiply and Add |
| FFMA | FP32 Fused Multiply and Add |
| FMNMX | FP32 Minimum/Maximum |
| FMUL | FP32 Multiply |
| FMUL32I | FP32 Multiply |
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FSWZADD | FP32 Swizzle Add |
| MUFU | FP32 Multi Function Operation |
| HADD2 | FP16 Add |
| HADD2\_32I | FP16 Add |
| HFMA2 | FP16 Fused Mutiply Add |
| HFMA2\_32I | FP16 Fused Mutiply Add |
| HMMA | Matrix Multiply and Accumulate |
| HMUL2 | FP16 Multiply |
| HMUL2\_32I | FP16 Multiply |
| HSET2 | FP16 Compare And Set |
| HSETP2 | FP16 Compare And Set Predicate |
| DADD | FP64 Add |
| DFMA | FP64 Fused Mutiply Add |
| DMUL | FP64 Multiply |
| DSETP | FP64 Compare And Set Predicate |
| **Integer Instructions** | |
| BMMA | Bit Matrix Multiply and Accumulate |
| BMSK | Bitfield Mask |
| BREV | Bit Reverse |
| FLO | Find Leading One |
| IABS | Integer Absolute Value |
| IADD | Integer Addition |
| IADD3 | 3-input Integer Addition |
| IADD32I | Integer Addition |
| IDP | Integer Dot Product and Accumulate |
| IDP4A | Integer Dot Product and Accumulate |
| IMAD | Integer Multiply And Add |
| IMMA | Integer Matrix Multiply and Accumulate |
| IMNMX | Integer Minimum/Maximum |
| IMUL | Integer Multiply |
| IMUL32I | Integer Multiply |
| ISCADD | Scaled Integer Addition |
| ISCADD32I | Scaled Integer Addition |
| ISETP | Integer Compare And Set Predicate |
| LEA | LOAD Effective Address |
| LOP | Logic Operation |
| LOP3 | Logic Operation |
| LOP32I | Logic Operation |
| POPC | Population count |
| SHF | Funnel Shift |
| SHL | Shift Left |
| SHR | Shift Right |
| VABSDIFF | Absolute Difference |
| VABSDIFF4 | Absolute Difference |
| **Conversion Instructions** | |
| F2F | Floating Point To Floating Point Conversion |
| F2I | Floating Point To Integer Conversion |
| I2F | Integer To Floating Point Conversion |
| I2I | Integer To Integer Conversion |
| I2IP | Integer To Integer Conversion and Packing |
| FRND | Round To Integer |
| **Movement Instructions** | |
| MOV | Move |
| MOV32I | Move |
| MOVM | Move Matrix with Transposition or Expansion |
| PRMT | Permute Register Pair |
| SEL | Select Source with Predicate |
| SGXT | Sign Extend |
| SHFL | Warp Wide Register Shuffle |
| **Predicate Instructions** | |
| PLOP3 | Predicate Logic Operation |
| PSETP | Combine Predicates and Set Predicate |
| P2R | Move Predicate Register To Register |
| R2P | Move Register To Predicate Register |
| **Load/Store Instructions** | |
| LD | Load from generic Memory |
| LDC | Load Constant |
| LDG | Load from Global Memory |
| LDL | Load within Local Memory Window |
| LDS | Load within Shared Memory Window |
| LDSM | Load Matrix from Shared Memory with Element Size Expansion |
| ST | Store to Generic Memory |
| STG | Store to Global Memory |
| STL | Store to Local Memory |
| STS | Store to Shared Memory |
| MATCH | Match Register Values Across Thread Group |
| QSPC | Query Space |
| ATOM | Atomic Operation on Generic Memory |
| ATOMS | Atomic Operation on Shared Memory |
| ATOMG | Atomic Operation on Global Memory |
| RED | Reduction Operation on Generic Memory |
| CCTL | Cache Control |
| CCTLL | Cache Control |
| ERRBAR | Error Barrier |
| MEMBAR | Memory Barrier |
| CCTLT | Texture Cache Control |
| **Uniform Datapath Instructions** | |
| R2UR | Move from Vector Register to a Uniform Register |
| S2UR | Move Special Register to Uniform Register |
| UBMSK | Uniform Bitfield Mask |
| UBREV | Uniform Bit Reverse |
| UCLEA | Load Effective Address for a Constant |
| UFLO | Uniform Find Leading One |
| UIADD3 | Uniform Integer Addition |
| UIADD3.64 | Uniform Integer Addition |
| UIMAD | Uniform Integer Multiplication |
| UISETP | Integer Compare and Set Uniform Predicate |
| ULDC | Load from Constant Memory into a Uniform Register |
| ULEA | Uniform Load Effective Address |
| ULOP | Logic Operation |
| ULOP3 | Logic Operation |
| ULOP32I | Logic Operation |
| UMOV | Uniform Move |
| UP2UR | Uniform Predicate to Uniform Register |
| UPLOP3 | Uniform Predicate Logic Operation |
| UPOPC | Uniform Population Count |
| UPRMT | Uniform Byte Permute |
| UPSETP | Uniform Predicate Logic Operation |
| UR2UP | Uniform Register to Uniform Predicate |
| USEL | Uniform Select |
| USGXT | Uniform Sign Extend |
| USHF | Uniform Funnel Shift |
| USHL | Uniform Left Shift |
| USHR | Uniform Right Shift |
| VOTEU | Voting across SIMD Thread Group with Results in Uniform Destination |
| **Texture Instructions** | |
| TEX | Texture Fetch |
| TLD | Texture Load |
| TLD4 | Texture Load 4 |
| TMML | Texture MipMap Level |
| TXD | Texture Fetch With Derivatives |
| TXQ | Texture Query |
| **Surface Instructions** | |
| SUATOM | Atomic Op on Surface Memory |
| SULD | Surface Load |
| SURED | Reduction Op on Surface Memory |
| SUST | Surface Store |
| **Control Instructions** | |
| BMOV | Move Convergence Barrier State |
| BPT | BreakPoint/Trap |
| BRA | Relative Branch |
| BREAK | Break out of the Specified Convergence Barrier |
| BRX | Relative Branch Indirect |
| BRXU | Relative Branch with Uniform Register Based Offset |
| BSSY | Barrier Set Convergence Synchronization Point |
| BSYNC | Synchronize Threads on a Convergence Barrier |
| CALL | Call Function |
| EXIT | Exit Program |
| JMP | Absolute Jump |
| JMX | Absolute Jump Indirect |
| JMXU | Absolute Jump with Uniform Register Based Offset |
| KILL | Kill Thread |
| NANOSLEEP | Suspend Execution |
| RET | Return From Subroutine |
| RPCMOV | PC Register Move |
| RTT | Return From Trap |
| WARPSYNC | Synchronize Threads in Warp |
| YIELD | Yield Control |
| **Miscellaneous Instructions** | |
| B2R | Move Barrier To Register |
| BAR | Barrier Synchronization |
| CS2R | Move Special Register to Register |
| DEPBAR | Dependency Barrier |
| GETLMEMBASE | Get Local Memory Base Address |
| LEPC | Load Effective PC |
| NOP | No Operation |
| PMTRIG | Performance Monitor Trigger |
| R2B | Move Register to Barrier |
| S2R | Move Special Register to Register |
| SETCTAID | Set CTA ID |
| SETLMEMBASE | Set Local Memory Base Address |
| VOTE | Vote Across SIMD Thread Group |

## 12.2. NVIDIA Ampere GPU and Ada Instruction Set[](#nvidia-ampere-gpu-and-ada-instruction-set "Permalink to this headline")

The NVIDIA Ampere GPU and Ada architectures (Compute Capability 8.0, 8.6, and 8.9) have the following instruction set format:

```
(instruction) (destination) (source1), (source2) ...
```

Valid destination and source locations include:

* RX for registers
* URX for uniform registers
* SRX for special system-controlled registers
* PX for predicate registers
* UPX for uniform predicate registers
* c[X][Y] for constant memory

[Table 7](#ampere-ampere-instruction-set-table) lists valid instructions for the NVIDIA Ampere architecrture and Ada GPUs.

Table 7. NVIDIA Ampere GPU and Ada Instruction Set[](#ampere-ampere-instruction-set-table "Permalink to this table")


| Opcode | Description |
| --- | --- |
| **Floating Point Instructions** |  |
| FADD | FP32 Add |
| FADD32I | FP32 Add |
| FCHK | Floating-point Range Check |
| FFMA32I | FP32 Fused Multiply and Add |
| FFMA | FP32 Fused Multiply and Add |
| FMNMX | FP32 Minimum/Maximum |
| FMUL | FP32 Multiply |
| FMUL32I | FP32 Multiply |
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FSWZADD | FP32 Swizzle Add |
| MUFU | FP32 Multi Function Operation |
| HADD2 | FP16 Add |
| HADD2\_32I | FP16 Add |
| HFMA2 | FP16 Fused Mutiply Add |
| HFMA2\_32I | FP16 Fused Mutiply Add |
| HMMA | Matrix Multiply and Accumulate |
| HMNMX2 | FP16 Minimum / Maximum |
| HMUL2 | FP16 Multiply |
| HMUL2\_32I | FP16 Multiply |
| HSET2 | FP16 Compare And Set |
| HSETP2 | FP16 Compare And Set Predicate |
| DADD | FP64 Add |
| DFMA | FP64 Fused Mutiply Add |
| DMMA | Matrix Multiply and Accumulate |
| DMUL | FP64 Multiply |
| DSETP | FP64 Compare And Set Predicate |
| **Integer Instructions** |  |
| BMMA | Bit Matrix Multiply and Accumulate |
| BMSK | Bitfield Mask |
| BREV | Bit Reverse |
| FLO | Find Leading One |
| IABS | Integer Absolute Value |
| IADD | Integer Addition |
| IADD3 | 3-input Integer Addition |
| IADD32I | Integer Addition |
| IDP | Integer Dot Product and Accumulate |
| IDP4A | Integer Dot Product and Accumulate |
| IMAD | Integer Multiply And Add |
| IMMA | Integer Matrix Multiply and Accumulate |
| IMNMX | Integer Minimum/Maximum |
| IMUL | Integer Multiply |
| IMUL32I | Integer Multiply |
| ISCADD | Scaled Integer Addition |
| ISCADD32I | Scaled Integer Addition |
| ISETP | Integer Compare And Set Predicate |
| LEA | LOAD Effective Address |
| LOP | Logic Operation |
| LOP3 | Logic Operation |
| LOP32I | Logic Operation |
| POPC | Population count |
| SHF | Funnel Shift |
| SHL | Shift Left |
| SHR | Shift Right |
| VABSDIFF | Absolute Difference |
| VABSDIFF4 | Absolute Difference |
| **Conversion Instructions** |  |
| F2F | Floating Point To Floating Point Conversion |
| F2I | Floating Point To Integer Conversion |
| I2F | Integer To Floating Point Conversion |
| I2I | Integer To Integer Conversion |
| I2IP | Integer To Integer Conversion and Packing |
| I2FP | Integer to FP32 Convert and Pack |
| F2IP | FP32 Down-Convert to Integer and Pack |
| FRND | Round To Integer |
| **Movement Instructions** |  |
| MOV | Move |
| MOV32I | Move |
| MOVM | Move Matrix with Transposition or Expansion |
| PRMT | Permute Register Pair |
| SEL | Select Source with Predicate |
| SGXT | Sign Extend |
| SHFL | Warp Wide Register Shuffle |
| **Predicate Instructions** |  |
| PLOP3 | Predicate Logic Operation |
| PSETP | Combine Predicates and Set Predicate |
| P2R | Move Predicate Register To Register |
| R2P | Move Register To Predicate Register |
| **Load/Store Instructions** |  |
| LD | Load from generic Memory |
| LDC | Load Constant |
| LDG | Load from Global Memory |
| LDGDEPBAR | Global Load Dependency Barrier |
| LDGSTS | Asynchronous Global to Shared Memcopy |
| LDL | Load within Local Memory Window |
| LDS | Load within Shared Memory Window |
| LDSM | Load Matrix from Shared Memory with Element Size Expansion |
| ST | Store to Generic Memory |
| STG | Store to Global Memory |
| STL | Store to Local Memory |
| STS | Store to Shared Memory |
| MATCH | Match Register Values Across Thread Group |
| QSPC | Query Space |
| ATOM | Atomic Operation on Generic Memory |
| ATOMS | Atomic Operation on Shared Memory |
| ATOMG | Atomic Operation on Global Memory |
| RED | Reduction Operation on Generic Memory |
| CCTL | Cache Control |
| CCTLL | Cache Control |
| ERRBAR | Error Barrier |
| MEMBAR | Memory Barrier |
| CCTLT | Texture Cache Control |
| **Uniform Datapath Instructions** |  |
| R2UR | Move from Vector Register to a Uniform Register |
| REDUX | Reduction of a Vector Register into a Uniform Register |
| S2UR | Move Special Register to Uniform Register |
| UBMSK | Uniform Bitfield Mask |
| UBREV | Uniform Bit Reverse |
| UCLEA | Load Effective Address for a Constant |
| UF2FP | Uniform FP32 Down-convert and Pack |
| UFLO | Uniform Find Leading One |
| UIADD3 | Uniform Integer Addition |
| UIADD3.64 | Uniform Integer Addition |
| UIMAD | Uniform Integer Multiplication |
| UISETP | Integer Compare and Set Uniform Predicate |
| ULDC | Load from Constant Memory into a Uniform Register |
| ULEA | Uniform Load Effective Address |
| ULOP | Logic Operation |
| ULOP3 | Logic Operation |
| ULOP32I | Logic Operation |
| UMOV | Uniform Move |
| UP2UR | Uniform Predicate to Uniform Register |
| UPLOP3 | Uniform Predicate Logic Operation |
| UPOPC | Uniform Population Count |
| UPRMT | Uniform Byte Permute |
| UPSETP | Uniform Predicate Logic Operation |
| UR2UP | Uniform Register to Uniform Predicate |
| USEL | Uniform Select |
| USGXT | Uniform Sign Extend |
| USHF | Uniform Funnel Shift |
| USHL | Uniform Left Shift |
| USHR | Uniform Right Shift |
| VOTEU | Voting across SIMD Thread Group with Results in Uniform Destination |
| **Texture Instructions** |  |
| TEX | Texture Fetch |
| TLD | Texture Load |
| TLD4 | Texture Load 4 |
| TMML | Texture MipMap Level |
| TXD | Texture Fetch With Derivatives |
| TXQ | Texture Query |
| **Surface Instructions** |  |
| SUATOM | Atomic Op on Surface Memory |
| SULD | Surface Load |
| SURED | Reduction Op on Surface Memory |
| SUST | Surface Store |
| **Control Instructions** |  |
| BMOV | Move Convergence Barrier State |
| BPT | BreakPoint/Trap |
| BRA | Relative Branch |
| BREAK | Break out of the Specified Convergence Barrier |
| BRX | Relative Branch Indirect |
| BRXU | Relative Branch with Uniform Register Based Offset |
| BSSY | Barrier Set Convergence Synchronization Point |
| BSYNC | Synchronize Threads on a Convergence Barrier |
| CALL | Call Function |
| EXIT | Exit Program |
| JMP | Absolute Jump |
| JMX | Absolute Jump Indirect |
| JMXU | Absolute Jump with Uniform Register Based Offset |
| KILL | Kill Thread |
| NANOSLEEP | Suspend Execution |
| RET | Return From Subroutine |
| RPCMOV | PC Register Move |
| WARPSYNC | Synchronize Threads in Warp |
| YIELD | Yield Control |
| **Miscellaneous Instructions** |  |
| B2R | Move Barrier To Register |
| BAR | Barrier Synchronization |
| CS2R | Move Special Register to Register |
| DEPBAR | Dependency Barrier |
| GETLMEMBASE | Get Local Memory Base Address |
| LEPC | Load Effective PC |
| NOP | No Operation |
| PMTRIG | Performance Monitor Trigger |
| S2R | Move Special Register to Register |
| SETCTAID | Set CTA ID |
| SETLMEMBASE | Set Local Memory Base Address |
| VOTE | Vote Across SIMD Thread Group |

## 12.3. Hopper Instruction Set[](#hopper-instruction-set "Permalink to this headline")

The Hopper architecture (Compute Capability 9.0) has the following instruction set format:

```
(instruction) (destination) (source1), (source2) ...
```

Valid destination and source locations include:

* RX for registers
* URX for uniform registers
* SRX for special system-controlled registers
* PX for predicate registers
* UPX for uniform predicate registers
* c[X][Y] for constant memory
* desc[URX][RY] for memory descriptors
* gdesc[URX] for global memory descriptors

[Table 8](#hopper-hopper-instruction-set-table) lists valid instructions for the Hopper GPUs.

Table 8. Hopper Instruction Set[](#hopper-hopper-instruction-set-table "Permalink to this table")


| Opcode | Description |
| --- | --- |
| **Floating Point Instructions** | |
| FADD | FP32 Add |
| FADD32I | FP32 Add |
| FCHK | Floating-point Range Check |
| FFMA32I | FP32 Fused Multiply and Add |
| FFMA | FP32 Fused Multiply and Add |
| FMNMX | FP32 Minimum/Maximum |
| FMUL | FP32 Multiply |
| FMUL32I | FP32 Multiply |
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FSWZADD | FP32 Swizzle Add |
| MUFU | FP32 Multi Function Operation |
| HADD2 | FP16 Add |
| HADD2\_32I | FP16 Add |
| HFMA2 | FP16 Fused Mutiply Add |
| HFMA2\_32I | FP16 Fused Mutiply Add |
| HMMA | Matrix Multiply and Accumulate |
| HMNMX2 | FP16 Minimum / Maximum |
| HMUL2 | FP16 Multiply |
| HMUL2\_32I | FP16 Multiply |
| HSET2 | FP16 Compare And Set |
| HSETP2 | FP16 Compare And Set Predicate |
| DADD | FP64 Add |
| DFMA | FP64 Fused Mutiply Add |
| DMMA | Matrix Multiply and Accumulate |
| DMUL | FP64 Multiply |
| DSETP | FP64 Compare And Set Predicate |
| **Integer Instructions** | |
| BMMA | Bit Matrix Multiply and Accumulate |
| BMSK | Bitfield Mask |
| BREV | Bit Reverse |
| FLO | Find Leading One |
| IABS | Integer Absolute Value |
| IADD | Integer Addition |
| IADD3 | 3-input Integer Addition |
| IADD32I | Integer Addition |
| IDP | Integer Dot Product and Accumulate |
| IDP4A | Integer Dot Product and Accumulate |
| IMAD | Integer Multiply And Add |
| IMMA | Integer Matrix Multiply and Accumulate |
| IMNMX | Integer Minimum/Maximum |
| IMUL | Integer Multiply |
| IMUL32I | Integer Multiply |
| ISCADD | Scaled Integer Addition |
| ISCADD32I | Scaled Integer Addition |
| ISETP | Integer Compare And Set Predicate |
| LEA | LOAD Effective Address |
| LOP | Logic Operation |
| LOP3 | Logic Operation |
| LOP32I | Logic Operation |
| POPC | Population count |
| SHF | Funnel Shift |
| SHL | Shift Left |
| SHR | Shift Right |
| VABSDIFF | Absolute Difference |
| VABSDIFF4 | Absolute Difference |
| VHMNMX | SIMD FP16 3-Input Minimum / Maximum |
| VIADD | SIMD Integer Addition |
| VIADDMNMX | SIMD Integer Addition and Fused Min/Max Comparison |
| VIMNMX | SIMD Integer Minimum / Maximum |
| VIMNMX3 | SIMD Integer 3-Input Minimum / Maximum |
| **Conversion Instructions** | |
| F2F | Floating Point To Floating Point Conversion |
| F2I | Floating Point To Integer Conversion |
| I2F | Integer To Floating Point Conversion |
| I2I | Integer To Integer Conversion |
| I2IP | Integer To Integer Conversion and Packing |
| I2FP | Integer to FP32 Convert and Pack |
| F2IP | FP32 Down-Convert to Integer and Pack |
| FRND | Round To Integer |
| **Movement Instructions** | |
| MOV | Move |
| MOV32I | Move |
| MOVM | Move Matrix with Transposition or Expansion |
| PRMT | Permute Register Pair |
| SEL | Select Source with Predicate |
| SGXT | Sign Extend |
| SHFL | Warp Wide Register Shuffle |
| **Predicate Instructions** | |
| PLOP3 | Predicate Logic Operation |
| PSETP | Combine Predicates and Set Predicate |
| P2R | Move Predicate Register To Register |
| R2P | Move Register To Predicate Register |
| **Load/Store Instructions** | |
| FENCE | Memory Visibility Guarantee for Shared or Global Memory |
| LD | Load from generic Memory |
| LDC | Load Constant |
| LDG | Load from Global Memory |
| LDGDEPBAR | Global Load Dependency Barrier |
| LDGMC | Reducing Load |
| LDGSTS | Asynchronous Global to Shared Memcopy |
| LDL | Load within Local Memory Window |
| LDS | Load within Shared Memory Window |
| LDSM | Load Matrix from Shared Memory with Element Size Expansion |
| STSM | Store Matrix to Shared Memory |
| ST | Store to Generic Memory |
| STG | Store to Global Memory |
| STL | Store to Local Memory |
| STS | Store to Shared Memory |
| STAS | Asynchronous Store to Distributed Shared Memory With Explicit Synchronization |
| SYNCS | Sync Unit |
| MATCH | Match Register Values Across Thread Group |
| QSPC | Query Space |
| ATOM | Atomic Operation on Generic Memory |
| ATOMS | Atomic Operation on Shared Memory |
| ATOMG | Atomic Operation on Global Memory |
| REDAS | Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization |
| REDG | Reduction Operation on Generic Memory |
| CCTL | Cache Control |
| CCTLL | Cache Control |
| ERRBAR | Error Barrier |
| MEMBAR | Memory Barrier |
| CCTLT | Texture Cache Control |
| **Uniform Datapath Instructions** | |
| R2UR | Move from Vector Register to a Uniform Register |
| REDUX | Reduction of a Vector Register into a Uniform Register |
| S2UR | Move Special Register to Uniform Register |
| UBMSK | Uniform Bitfield Mask |
| UBREV | Uniform Bit Reverse |
| UCGABAR\_ARV | CGA Barrier Synchronization |
| UCGABAR\_WAIT | CGA Barrier Synchronization |
| UCLEA | Load Effective Address for a Constant |
| UF2FP | Uniform FP32 Down-convert and Pack |
| UFLO | Uniform Find Leading One |
| UIADD3 | Uniform Integer Addition |
| UIADD3.64 | Uniform Integer Addition |
| UIMAD | Uniform Integer Multiplication |
| UISETP | Integer Compare and Set Uniform Predicate |
| ULDC | Load from Constant Memory into a Uniform Register |
| ULEA | Uniform Load Effective Address |
| ULEPC | Uniform Load Effective PC |
| ULOP | Logic Operation |
| ULOP3 | Logic Operation |
| ULOP32I | Logic Operation |
| UMOV | Uniform Move |
| UP2UR | Uniform Predicate to Uniform Register |
| UPLOP3 | Uniform Predicate Logic Operation |
| UPOPC | Uniform Population Count |
| UPRMT | Uniform Byte Permute |
| UPSETP | Uniform Predicate Logic Operation |
| UR2UP | Uniform Register to Uniform Predicate |
| USEL | Uniform Select |
| USETMAXREG | Release, Deallocate and Allocate Registers |
| USGXT | Uniform Sign Extend |
| USHF | Uniform Funnel Shift |
| USHL | Uniform Left Shift |
| USHR | Uniform Right Shift |
| VOTEU | Voting across SIMD Thread Group with Results in Uniform Destination |
| **Warpgroup Instructions** | |
| BGMMA | Bit Matrix Multiply and Accumulate Across Warps |
| HGMMA | Matrix Multiply and Accumulate Across a Warpgroup |
| IGMMA | Integer Matrix Multiply and Accumulate Across a Warpgroup |
| QGMMA | FP8 Matrix Multiply and Accumulate Across a Warpgroup |
| WARPGROUP | Warpgroup Synchronization |
| WARPGROUPSET | Set Warpgroup Counters |
| **Tensor Memory Access Instructions** | |
| UBLKCP | Bulk Data Copy |
| UBLKPF | Bulk Data Prefetch |
| UBLKRED | Bulk Data Copy from Shared Memory with Reduction |
| UTMACCTL | TMA Cache Control |
| UTMACMDFLUSH | TMA Command Flush |
| UTMALDG | Tensor Load from Global to Shared Memory |
| UTMAPF | Tensor Prefetch |
| UTMAREDG | Tensor Store from Shared to Global Memory with Reduction |
| UTMASTG | Tensor Store from Shared to Global Memory |
| **Texture Instructions** | |
| TEX | Texture Fetch |
| TLD | Texture Load |
| TLD4 | Texture Load 4 |
| TMML | Texture MipMap Level |
| TXD | Texture Fetch With Derivatives |
| TXQ | Texture Query |
| **Surface Instructions** | |
| SUATOM | Atomic Op on Surface Memory |
| SULD | Surface Load |
| SURED | Reduction Op on Surface Memory |
| SUST | Surface Store |
| **Control Instructions** | |
| ACQBULK | Wait for Bulk Release Status Warp State |
| BMOV | Move Convergence Barrier State |
| BPT | BreakPoint/Trap |
| BRA | Relative Branch |
| BREAK | Break out of the Specified Convergence Barrier |
| BRX | Relative Branch Indirect |
| BRXU | Relative Branch with Uniform Register Based Offset |
| BSSY | Barrier Set Convergence Synchronization Point |
| BSYNC | Synchronize Threads on a Convergence Barrier |
| CALL | Call Function |
| CGAERRBAR | CGA Error Barrier |
| ELECT | Elect a Leader Thread |
| ENDCOLLECTIVE | Reset the MCOLLECTIVE mask |
| EXIT | Exit Program |
| JMP | Absolute Jump |
| JMX | Absolute Jump Indirect |
| JMXU | Absolute Jump with Uniform Register Based Offset |
| KILL | Kill Thread |
| NANOSLEEP | Suspend Execution |
| PREEXIT | Dependent Task Launch Hint |
| RET | Return From Subroutine |
| RPCMOV | PC Register Move |
| WARPSYNC | Synchronize Threads in Warp |
| YIELD | Yield Control |
| **Miscellaneous Instructions** | |
| B2R | Move Barrier To Register |
| BAR | Barrier Synchronization |
| CS2R | Move Special Register to Register |
| DEPBAR | Dependency Barrier |
| GETLMEMBASE | Get Local Memory Base Address |
| LEPC | Load Effective PC |
| NOP | No Operation |
| PMTRIG | Performance Monitor Trigger |
| S2R | Move Special Register to Register |
| SETCTAID | Set CTA ID |
| SETLMEMBASE | Set Local Memory Base Address |
| VOTE | Vote Across SIMT Thread Group |

## 12.4. Blackwell Instruction Set[](#blackwell-instruction-set "Permalink to this headline")

The Blackwell architecture (Compute Capability 10.0 and 12.0) has the following instruction set format:

```
(instruction) (destination) (source1), (source2) ...
```

Valid destination and source locations include:

* RX for registers
* URX for uniform registers
* SRX for special system-controlled registers
* PX for predicate registers
* UPX for uniform predicate registers
* c[X][Y] for constant memory
* desc[URX][RY] for memory descriptors
* gdesc[URX] for global memory descriptors
* tmem[URX] for tensor memory

[Table 9](#blackwell-instruction-set-table) lists valid instructions for the Blackwell GPUs.

Table 9. Blackwell Instruction Set[](#id13 "Permalink to this table")


| Opcode | Description |
| --- | --- |
| **Floating Point Instructions** |  |
| FADD | FP32 Add |
| FADD2 | FP32 Add |
| FADD32I | FP32 Add |
| FCHK | Floating-point Range Check |
| FFMA32I | FP32 Fused Multiply and Add |
| FFMA | FP32 Fused Multiply and Add |
| FFMA2 | FP32 Fused Multiply and Add |
| FHADD | FP32 Addition |
| FHFMA | FP32 Fused Multiply and Add |
| FMNMX | FP32 Minimum/Maximum |
| FMNMX3 | 3-Input Floating-point Minimum / Maximum |
| FMUL | FP32 Multiply |
| FMUL2 | FP32 Multiply |
| FMUL32I | FP32 Multiply |
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FSWZADD | FP32 Swizzle Add |
| MUFU | FP32 Multi Function Operation |
| HADD2 | FP16 Add |
| HADD2\_32I | FP16 Add |
| HFMA2 | FP16 Fused Mutiply Add |
| HFMA2\_32I | FP16 Fused Mutiply Add |
| HMMA | Matrix Multiply and Accumulate |
| HMNMX2 | FP16 Minimum / Maximum |
| HMUL2 | FP16 Multiply |
| HMUL2\_32I | FP16 Multiply |
| HSET2 | FP16 Compare And Set |
| HSETP2 | FP16 Compare And Set Predicate |
| DADD | FP64 Add |
| DFMA | FP64 Fused Mutiply Add |
| DMMA | Matrix Multiply and Accumulate |
| DMUL | FP64 Multiply |
| DSETP | FP64 Compare And Set Predicate |
| OMMA | FP4 Matrix Multiply and Accumulate Across a Warp |
| QMMA | FP8 Matrix Multiply and Accumulate Across a Warp |
| **Integer Instructions** |  |
| BMSK | Bitfield Mask |
| BREV | Bit Reverse |
| FLO | Find Leading One |
| IABS | Integer Absolute Value |
| IADD | Integer Addition |
| IADD3 | 3-input Integer Addition |
| IADD32I | Integer Addition |
| IDP | Integer Dot Product and Accumulate |
| IDP4A | Integer Dot Product and Accumulate |
| IMAD | Integer Multiply And Add |
| IMMA | Integer Matrix Multiply and Accumulate |
| IMNMX | Integer Minimum/Maximum |
| IMUL | Integer Multiply |
| IMUL32I | Integer Multiply |
| ISCADD | Scaled Integer Addition |
| ISCADD32I | Scaled Integer Addition |
| ISETP | Integer Compare And Set Predicate |
| LEA | LOAD Effective Address |
| LOP | Logic Operation |
| LOP3 | Logic Operation |
| LOP32I | Logic Operation |
| POPC | Population count |
| SHF | Funnel Shift |
| SHL | Shift Left |
| SHR | Shift Right |
| VABSDIFF | Absolute Difference |
| VABSDIFF4 | Absolute Difference |
| VHMNMX | SIMD FP16 3-Input Minimum / Maximum |
| VIADD | SIMD Integer Addition |
| VIADDMNMX | SIMD Integer Addition and Fused Min/Max Comparison |
| VIMNMX | SIMD Integer Minimum / Maximum |
| VIMNMX3 | SIMD Integer 3-Input Minimum / Maximum |
| **Conversion Instructions** |  |
| F2F | Floating Point To Floating Point Conversion |
| F2I | Floating Point To Integer Conversion |
| I2F | Integer To Floating Point Conversion |
| I2I | Integer To Integer Conversion |
| I2IP | Integer To Integer Conversion and Packing |
| I2FP | Integer to FP32 Convert and Pack |
| F2IP | FP32 Down-Convert to Integer and Pack |
| FRND | Round To Integer |
| **Movement Instructions** |  |
| MOV | Move |
| MOV32I | Move |
| MOVM | Move Matrix with Transposition or Expansion |
| PRMT | Permute Register Pair |
| SEL | Select Source with Predicate |
| SGXT | Sign Extend |
| SHFL | Warp Wide Register Shuffle |
| **Predicate Instructions** |  |
| PLOP3 | Predicate Logic Operation |
| PSETP | Combine Predicates and Set Predicate |
| P2R | Move Predicate Register To Register |
| R2P | Move Register To Predicate Register |
| **Load/Store Instructions** |  |
| FENCE | Memory Visibility Guarantee for Shared or Global Memory |
| LD | Load from generic Memory |
| LDC | Load Constant |
| LDG | Load from Global Memory |
| LDGDEPBAR | Global Load Dependency Barrier |
| LDGMC | Reducing Load |
| LDGSTS | Asynchronous Global to Shared Memcopy |
| LDL | Load within Local Memory Window |
| LDS | Load within Shared Memory Window |
| LDSM | Load Matrix from Shared Memory with Element Size Expansion |
| STSM | Store Matrix to Shared Memory |
| ST | Store to Generic Memory |
| STG | Store to Global Memory |
| STL | Store to Local Memory |
| STS | Store to Shared Memory |
| STAS | Asynchronous Store to Distributed Shared Memory With Explicit Synchronization |
| SYNCS | Sync Unit |
| MATCH | Match Register Values Across Thread Group |
| QSPC | Query Space |
| ATOM | Atomic Operation on Generic Memory |
| ATOMS | Atomic Operation on Shared Memory |
| ATOMG | Atomic Operation on Global Memory |
| REDAS | Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization |
| REDG | Reduction Operation on Generic Memory |
| CCTL | Cache Control |
| CCTLL | Cache Control |
| ERRBAR | Error Barrier |
| MEMBAR | Memory Barrier |
| CCTLT | Texture Cache Control |
| **Uniform Datapath Instructions** |  |
| CREDUX | Coupled Reduction of a Vector Register into a Uniform Register |
| CS2UR | Load a Value from Constant Memory into a Uniform Register |
| LDCU | Load a Value from Constant Memory into a Uniform Register |
| R2UR | Move from Vector Register to a Uniform Register |
| REDUX | Reduction of a Vector Register into a Uniform Register |
| S2UR | Move Special Register to Uniform Register |
| UBMSK | Uniform Bitfield Mask |
| UBREV | Uniform Bit Reverse |
| UCGABAR\_ARV | CGA Barrier Synchronization |
| UCGABAR\_WAIT | CGA Barrier Synchronization |
| UCLEA | Load Effective Address for a Constant |
| UFADD | Uniform Uniform FP32 Addition |
| UF2F | Uniform Float-to-Float Conversion |
| UF2FP | Uniform FP32 Down-convert and Pack |
| UF2I | Uniform Float-to-Integer Conversion |
| UF2IP | Uniform FP32 Down-Convert to Integer and Pack |
| UFFMA | Uniform FP32 Fused Multiply-Add |
| UFLO | Uniform Find Leading One |
| UFMNMX | Uniform Floating-point Minimum / Maximum |
| UFMUL | Uniform FP32 Multiply |
| UFRND | Uniform Round to Integer |
| UFSEL | Uniform Floating-Point Select |
| UFSET | Uniform Floating-Point Compare and Set |
| UFSETP | Uniform Floating-Point Compare and Set Predicate |
| UI2F | Uniform Integer to Float conversion |
| UI2FP | Uniform Integer to FP32 Convert and Pack |
| UI2I | Uniform Saturating Integer-to-Integer Conversion |
| UI2IP | Uniform Dual Saturating Integer-to-Integer Conversion and Packing |
| UIABS | Uniform Integer Absolute Value |
| UIMNMX | Uniform Integer Minimum / Maximum |
| UIADD3 | Uniform Integer Addition |
| UIADD3.64 | Uniform Integer Addition |
| UIMAD | Uniform Integer Multiplication |
| UISETP | Uniform Integer Compare and Set Uniform Predicate |
| ULEA | Uniform Load Effective Address |
| ULEPC | Uniform Load Effective PC |
| ULOP | Uniform Logic Operation |
| ULOP3 | Uniform Logic Operation |
| ULOP32I | Uniform Logic Operation |
| UMOV | Uniform Move |
| UP2UR | Uniform Predicate to Uniform Register |
| UPLOP3 | Uniform Predicate Logic Operation |
| UPOPC | Uniform Population Count |
| UPRMT | Uniform Byte Permute |
| UPSETP | Uniform Predicate Logic Operation |
| UR2UP | Uniform Register to Uniform Predicate |
| USEL | Uniform Select |
| USETMAXREG | Release, Deallocate and Allocate Registers |
| USGXT | Uniform Sign Extend |
| USHF | Uniform Funnel Shift |
| USHL | Uniform Left Shift |
| USHR | Uniform Right Shift |
| UGETNEXTWORKID | Uniform Get Next Work ID |
| UMEMSETS | Initialize Shared Memory |
| UREDGR | Uniform Reduction on Global Memory with Release |
| USTGR | Uniform Store to Global Memory with Release |
| UVIADD | Uniform SIMD Integer Addition |
| UVIMNMX | Uniform SIMD Integer Minimum / Maximum |
| UVIRTCOUNT | Virtual Resource Management |
| VOTEU | Voting across SIMD Thread Group with Results in Uniform Destination |
| **Tensor Memory Access Instructions** |  |
| UBLKCP | Bulk Data Copy |
| UBLKPF | Bulk Data Prefetch |
| UBLKRED | Bulk Data Copy from Shared Memory with Reduction |
| UTMACCTL | TMA Cache Control |
| UTMACMDFLUSH | TMA Command Flush |
| UTMALDG | Tensor Load from Global to Shared Memory |
| UTMAPF | Tensor Prefetch |
| UTMAREDG | Tensor Store from Shared to Global Memory with Reduction |
| UTMASTG | Tensor Store from Shared to Global Memory |
| **Tensor Core Memory Instructions** |  |
| LDT | Load Matrix from Tensor Memory to Register File |
| LDTM | Load Matrix from Tensor Memory to Register File |
| STT | Store Matrix to Tensor Memory from Register File |
| STTM | Store Matrix to Tensor Memory from Register File |
| UTCATOMSWS | Perform Atomic operation on SW State Register |
| UTCBAR | Tensor Core Barrier |
| UTCCP | Asynchonous data copy from Shared Memory to Tensor Memory |
| UTCHMMA | Uniform Matrix Multiply and Accumulate |
| UTCIMMA | Uniform Matrix Multiply and Accumulate |
| UTCOMMA | Uniform Matrix Multiply and Accumulate |
| UTCQMMA | Uniform Matrix Multiply and Accumulate |
| UTCSHIFT | Shift elements in Tensor Memory |
| **Texture Instructions** |  |
| TEX | Texture Fetch |
| TLD | Texture Load |
| TLD4 | Texture Load 4 |
| TMML | Texture MipMap Level |
| TXD | Texture Fetch With Derivatives |
| TXQ | Texture Query |
| **Surface Instructions** |  |
| SUATOM | Atomic Op on Surface Memory |
| SULD | Surface Load |
| SURED | Reduction Op on Surface Memory |
| SUST | Surface Store |
| **Control Instructions** |  |
| ACQBULK | Wait for Bulk Release Status Warp State |
| ACQSHMINIT | Wait for Shared Memory Initialization Release Status Warp State |
| BMOV | Move Convergence Barrier State |
| BPT | BreakPoint/Trap |
| BRA | Relative Branch |
| BREAK | Break out of the Specified Convergence Barrier |
| BRX | Relative Branch Indirect |
| BRXU | Relative Branch with Uniform Register Based Offset |
| BSSY | Barrier Set Convergence Synchronization Point |
| BSYNC | Synchronize Threads on a Convergence Barrier |
| CALL | Call Function |
| CGAERRBAR | CGA Error Barrier |
| ELECT | Elect a Leader Thread |
| ENDCOLLECTIVE | Reset the MCOLLECTIVE mask |
| EXIT | Exit Program |
| JMP | Absolute Jump |
| JMX | Absolute Jump Indirect |
| JMXU | Absolute Jump with Uniform Register Based Offset |
| KILL | Kill Thread |
| NANOSLEEP | Suspend Execution |
| PREEXIT | Dependent Task Launch Hint |
| RET | Return From Subroutine |
| RPCMOV | PC Register Move |
| WARPSYNC | Synchronize Threads in Warp |
| YIELD | Yield Control |
| **Miscellaneous Instructions** |  |
| B2R | Move Barrier To Register |
| BAR | Barrier Synchronization |
| CS2R | Move Special Register to Register |
| DEPBAR | Dependency Barrier |
| GETLMEMBASE | Get Local Memory Base Address |
| LEPC | Load Effective PC |
| NOP | No Operation |
| PMTRIG | Performance Monitor Trigger |
| S2R | Move Special Register to Register |
| SETCTAID | Set CTA ID |
| SETLMEMBASE | Set Local Memory Base Address |
| VOTE | Vote Across SIMT Thread Group |

