# Reference Papers

LaTeX sources for a set of papers on agentic / RL-based GPU kernel generation,
plus their accompanying code repositories (included as git submodules).

## Papers (LaTeX sources)

- **`cuco/`** — *CUCo: An Agentic Framework for Compute and Communication Co-design*; a training-free agent that co-optimizes computation and communication in CUDA kernels for up to 1.57× speedup on multi-GPU workloads.
- **`cudaforge/`** — *CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization*; a training-free multi-agent workflow that iteratively generates, tests, and tunes CUDA kernels using hardware profiling feedback.
- **`drkernel/`** — *Dr.Kernel: Reinforcement Learning Done Right for Triton Kernel Generations*; studies multi-turn RL (TRLOO, profiling-based rewards) to train LLMs that produce fast, correct Triton kernels.
- **`nvidia-avo/`** — *AVO: Agentic Variation Operators for Autonomous Evolutionary Search*; replaces classical evolutionary operators with coding agents to evolve attention kernels that surpass cuDNN and FlashAttention-4 on NVIDIA Blackwell (B200).

## Code (git submodules)

- **`cuco-code/`** — code for the CUCo paper ([UT-InfraAI/cuco](https://github.com/UT-InfraAI/cuco)).
- **`cudaforge-code/`** — code for the CudaForge paper ([OptimAI-Lab/CudaForge](https://github.com/OptimAI-Lab/CudaForge)).
- **`KernelGYM/`** — distributed GPU environment and RL training code for the Dr.Kernel paper ([hkust-nlp/KernelGYM](https://github.com/hkust-nlp/KernelGYM)).