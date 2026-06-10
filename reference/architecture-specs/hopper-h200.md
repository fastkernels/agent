<!-- originally from https://github.com/xinhaoc/ferret/blob/main/docs/architecture/hopper-h100.md -->
# Hopper H100 / H200 Architecture Deep-Dive

The Hopper family covers the H100 (SXM5, PCIe, NVL) and the H200 (SXM, NVL). All share
the same GH100 die and SM design; the H200 is a memory upgrade (HBM3e, 141 GB, 4.8 TB/s)
on the identical compute fabric, so the architecture below applies to both unless noted.

## SM Architecture

- **SM count**: 132 SMs (H100 SXM5, H100 NVL, H200 SXM/NVL), 114 SMs (H100 PCIe). Full GH100 die: 144 SMs
- **Per SM**: 128 FP32 cores, 4 fourth-gen Tensor Cores, 4 warp schedulers
- **FP32 throughput**: 2x per cycle per SM compared to A100 (doubled FP32 cores)
- **Max**: 64 warps/SM, 2048 threads/SM, 32 blocks/SM
- **Register file**: 256 KB (64K × 32-bit registers), max 255 regs/thread
- **Shared memory + L1**: 256 KB combined (33% larger than A100's 192 KB)
  * Configurable shared memory up to 228 KB
  * Carveout options: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB
  * Static shared memory limited to 48 KB; dynamic above 48 KB requires `cudaFuncSetAttribute` opt-in
- **Compute capability**: 9.0 (9.0a for architecture-specific features like wgmma). Same CC 9.0 for H100 and H200 — no recompilation needed to target H200.


## Product Variants

| Variant   | SMs | GPU Memory      | Bandwidth   | TDP        | Interconnect                     |
| --------- | --- | --------------- | ----------- | ---------- | -------------------------------- |
| H100 SXM5 | 132 | 80 GB HBM3      | 3.35 TB/s   | 700W       | NVLink 900 GB/s, PCIe Gen5       |
| H100 PCIe | 114 | 80 GB HBM2e     | ~2 TB/s     | 350W       | NVLink 600 GB/s, PCIe Gen5       |
| H100 NVL  | 132 | 94 GB HBM3/GPU  | ~3.9 TB/s   | 400W/GPU   | NVLink bridge 600 GB/s, PCIe Gen5|
| H200 SXM  | 132 | 141 GB HBM3e    | 4.8 TB/s    | up to 700W | NVLink 900 GB/s, PCIe Gen5       |
| H200 NVL  | 132 | 141 GB HBM3e    | 4.8 TB/s    | up to 600W | 2/4-way NVLink bridge 900 GB/s, PCIe Gen5 |

**Notes on variants:**

- **H100 PCIe** runs at lower clocks and has fewer active SMs (114), so its dense Tensor
  throughput is ~756 TFLOPS FP16 vs 989 on SXM5. It also ships with HBM2e rather than HBM3.
- **H100 NVL** is a dual-GPU PCIe bridge product. Per-GPU Tensor throughput matches SXM5
  (989 dense FP16); each GPU carries 94 GB HBM3 (188 GB per bridged pair) at ~3.9 TB/s.
- **H200 SXM and H200 NVL publish identical Tensor Core throughput** per NVIDIA datasheet
  3418659 (AUG24), equal to H100 SXM5. The H200 advantage is entirely memory: 141 GB
  HBM3e at 4.8 TB/s vs the H100's 80 GB. In practice H200 NVL runs ~15-18% slower than
  H200 SXM due to its lower power cap, but the published peaks are the same.


## Memory Hierarchy

### L1 Cache

- Part of the unified 256 KB data cache per SM
- Also serves as texture cache
- Whatever remains after shared memory carveout


### L2 Cache

- **50 MB** (25% larger than A100's 40 MB). Same on H100 and H200.
- Partitioned crossbar structure for data localization
- Supports data compression/decompression
- Persistence control: pin frequently accessed data via CUDA APIs (same as Ampere)


### HBM (H100 HBM3 / H200 HBM3e)

- **H100 SXM5**: 80 GB, 5 HBM3 stacks, 10 × 512-bit memory controllers. **3.35 TB/s** bandwidth (~64% increase over A100-SXM's 2.04 TB/s)
- **H100 PCIe**: 80 GB HBM2e, over 2 TB/s bandwidth
- **H100 NVL**: 94 GB HBM3 per GPU, ~3.9 TB/s per GPU
- **H200 SXM / NVL**: 141 GB HBM3e, 4.8 TB/s (~1.4x the H100 SXM5). Same GH100 die, upgraded memory subsystem.
- **Inline compression**: Available through CUDA driver API. Allows higher effective bandwidth by compressing data on the bus. Check via `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED)`


## Tensor Cores (4th Generation)

### Supported Data Types

FP8 (E4M3 and E5M2), FP16, BF16, TF32, FP64, INT8 — all with structured sparsity support.

**New**: FP8 support via Transformer Engine. E4M3 (higher precision, max 448) for forward pass; E5M2 (wider range, max 57344) for backward pass.

Note: neither H100 nor H200 supports FP4 — that requires Blackwell's 5th-gen Tensor Cores.

### Peak Performance (H100 SXM5 / H200 SXM / H200 NVL — identical)

| Data Type         | Dense       | With 2:4 Sparsity |
| ----------------- | ----------- | ----------------- |
| FP8               | 1979 TFLOPS | 3958 TFLOPS       |
| FP16              | 989 TFLOPS  | 1979 TFLOPS       |
| BF16              | 989 TFLOPS  | 1979 TFLOPS       |
| TF32              | 495 TFLOPS  | 989 TFLOPS        |
| FP64              | 60 TFLOPS   | —                 |
| INT8              | 2000 TOPS   | 3958 TOPS         |
| FP32 (CUDA cores) | 66.9 TFLOPS | —                 |
| FP64 (CUDA cores) | 33.5 TFLOPS | —                 |

H100 PCIe (lower clocks, 114 SMs) is roughly 0.76x these figures: FP8 1513 / FP16 756 /
TF32 378 dense.

### Structured Sparsity

2:4 pattern (exactly 2 non-zero values in every 4 elements). Doubles tensor core throughput. Workflow: train dense → prune to 2:4 → fine-tune remaining weights.

### Transformer Engine

Automatically manages FP8/FP16 precision per layer. Analyzes output statistics and dynamically scales tensors into the representable range. Up to 9x faster training and 30x faster inference vs A100 on large language models.

### MMA Instruction Sizes (PTX)

- WGMMA (warpgroup MMA): 128 threads (4 warps) cooperate on one MMA
- Shapes: m64nNk16 for FP16/BF16, m64nNk8 for TF32, m64nNk32 for FP8
- Operands can be read **directly from shared memory** (SS mode) or registers+shared (RS mode)


## New Features vs Ampere

### TMA (Tensor Memory Accelerator)

Dedicated hardware unit for asynchronous multi-dimensional tensor copies:

- Transfers entire tiles (1D through 5D) between global and shared memory in a **single instruction**
- **Single-thread programming**: one thread issues the TMA command for the entire thread block. All other threads are free to compute.
- **No register usage**: data moves directly between memory spaces without passing through registers. Eliminates register pressure from data movement.
- Supports element-wise **reduction operations** (add/min/max/bitwise) for global writes
- Can transfer between shared memory of **different SMs within a cluster** (multicast)
- Exposed through `cp.async.bulk.tensor` PTX instructions
- Synchronized via `cuda::barrier` / `cuda::pipeline` with transaction-byte tracking


**TMA programming pattern:**

```
1. Create TMA descriptor on host (cuTensorMapEncodeTiled)
2. In kernel: one thread issues cp.async.bulk.tensor with mbarrier address
3. TMA hardware completes the copy and signals the mbarrier
4. Consumer threads wait on mbarrier, then read from shared memory
```

### Thread Block Clusters

New level in the CUDA hierarchy: **threads → blocks → clusters → grid**

- A cluster is a group of thread blocks **guaranteed to be concurrently scheduled** onto SMs within the same GPC
- Max portable cluster size: **8 blocks**. Non-portable: up to 16 (requires `cudaFuncAttributeNonPortableClusterSizeAllowed` opt-in)
- Hardware-accelerated barriers across multiple SMs within a cluster
- Dedicated SM-to-SM network within GPC
- **Occupancy impact**: larger clusters may reduce max active blocks. Use `cudaOccupancyMaxActiveClusters` to compute occupancy.


### Distributed Shared Memory (DSMEM)

- All threads in a cluster can directly **load, store, and perform atomics** on shared memory of other thread blocks in the cluster
- **~7x faster** than exchanging data via global memory
- Dedicated SM-to-SM network for cluster communication
- Generic address space mapping for direct pointer access
- Supports asynchronous copy operations with barrier synchronization


**Tuning**: Accesses should be coalesced and aligned to 32-byte segments. Non-unit stride access patterns should use local shared memory with padding instead.

### Asynchronous Transaction Barriers

Enhanced mbarrier that tracks both **thread arrivals AND data transaction bytes**:

- When TMA completes a transfer, it automatically signals the mbarrier by completing the expected byte count
- Barrier flips phase only when BOTH all thread arrivals AND all expected bytes are satisfied
- Supports **sleeping while waiting** — waiting threads can sleep until the barrier condition is met, reducing power consumption and freeing SM resources
- Critical for efficient producer-consumer patterns with TMA


### Warp Specialization

Enabled by TMA's single-thread programming model:

- **Producer warps**: issue TMA loads (only need ~24-40 registers)
- **Consumer warps**: perform tensor core MMA (get up to 240 registers)
- Register redistribution at runtime via `setmaxnreg.inc/dec.sync.aligned`
- Communication through shared memory + async transaction barriers
- See `docs/patterns/async-execution.md` for full pipeline details


### DPX Instructions

Accelerate dynamic programming algorithms by up to 7x over A100:

- Support for fused operands in DP inner loops
- Operations on signed/unsigned 32-bit int and 16-bit short2 types
- 128 operations per cycle per SM
- Use cases: Smith-Waterman (genomics), Floyd-Warshall (route optimization)


### Interconnect

| Feature                  | Spec                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| **NVLink 4th gen**       | 18 links, 900 GB/s bidirectional total (1.5x A100). 7x PCIe Gen 5 bandwidth. Same on H100 SXM5 and H200 SXM.        |
| **NVSwitch 3rd gen**     | 64 ports, 13.6 Tbps. Hardware-accelerated collectives (all\_gather, reduce\_scatter, broadcast atomics via SHARP). |
| **NVLink Switch System** | Up to 256 GPUs across 32 nodes. 57.6 TB/s all-to-all.                                                              |
| **PCIe Gen 5**           | 128 GB/s total (2x Gen 4). Native atomic operations. SR-IOV support.                                               |


### Second-Generation MIG

- 3x more compute and 2x more bandwidth per instance vs A100 MIG
- Up to 7 instances with dedicated NVDEC/NVJPG per instance
- Each instance has fully isolated paths: crossbar ports, L2 banks, memory controllers, DRAM buses
- H100: up to 7 MIGs @ 10 GB each. H200: up to 7 MIGs @ 18 GB each (benefits from the larger 141 GB pool).


## Compilation

```
nvcc -arch=sm_90 -O3 -lineinfo kernel.cu           # baseline Hopper (H100 and H200)
nvcc -arch=sm_90a -O3 -lineinfo kernel.cu           # architecture-specific (wgmma, TMA features)
nvcc -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \     # + PTX for forward compat
     kernel.cu
```

Use `sm_90a` for wgmma, advanced TMA, and cluster-specific features. Not forward-compatible.
H100 and H200 share CC 9.0, so a single `sm_90` build runs on both.

## Tuning Recommendations

1. Apps following Ampere best practices should see speedups without code changes
2. Use `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout` to configure shared memory (up to 228 KB)
3. **TMA**: use single-thread issue model, pair with `cuda::barrier` / `cuda::pipeline` for synchronization. Set transaction bytes exactly.
4. **Clusters**: compute occupancy with `cudaOccupancyMaxActiveClusters`. Prefer portable cluster size of 8 for compatibility.
5. **DSMEM**: coalesce accesses to 32-byte aligned segments. Avoid non-unit stride — use local shared memory with padding instead.
6. **Warp specialization**: dedicate producer warps for TMA, consumer warps for MMA. Redistribute registers via `setmaxnreg`.
7. Ensure coalesced global memory access. Minimize redundant global memory reads.
8. Avoid long sequences of diverged execution within the same warp.
9. Use inline compression for bandwidth-bound kernels via CUDA driver API.
10. For FP8: use Transformer Engine with HYBRID format (E4M3 forward, E5M2 backward).
11. **H200 vs H100**: since compute peaks are identical, the H200 wins on memory-bound
    kernels (larger working sets, bigger KV caches, fewer memory stalls). Compute-bound
    kernels see little difference. Re-check the roofline ridge point: H200's higher
    bandwidth (4.8 vs 3.35 TB/s) shifts more kernels into the compute-bound regime.


## Physical Specs

- TSMC 4N process, 80 billion transistors, 814 mm² die (GH100, shared by H100 and H200)
- H100 SXM5 TDP: 700W, H100 PCIe TDP: 350W, H100 NVL: 400W/GPU
- H200 SXM TDP: up to 700W, H200 NVL TDP: up to 600W
