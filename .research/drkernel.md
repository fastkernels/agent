# Dr.Kernel / KernelGYM — Sandbox / Tooling / Infra Distillation

Source: `/data-fast/agent/reference/related-works/KernelGYM/` (code) +
`/data-fast/agent/reference/related-works/drkernel/` (COLM 2026 paper).
Focus: build/run/profile sandbox, worker/queue arch, multi-GPU scheduling, resource/timeout control,
failure isolation, result flow. Target reuse: our autonomous FA3->single-.cu agent on H200 x4 / CUDA 13.

NOTE: KernelGYM is built for **Triton RL training at scale** (thousands of kernels/batch), so a lot of its
machinery is about throughput + RL reward signals. Our agent is the opposite scale: a SMALL number of
candidate .cu files, each needing rock-solid correctness + reliable latency. So we adopt the *isolation,
timing, and failure-recovery patterns*, and drop the Redis/FastAPI distributed layer (overkill) unless we
want multi-GPU fan-out — which we do for the "1 kernel per GPU for perf, share for correctness" requirement.

---

## 1. High-level architecture (server–worker split)

Four design principles (paper §kernelgym):
1. **Serialized execution** — profiling is contention-sensitive; enforce **one-GPU-one-task**. No two
   kernels timed on the same GPU concurrently (prevents context pollution / unreliable timing).
2. **Elastic scalability** — GPU workers add/remove dynamically without interrupting the run.
3. **Fault isolation + self-recovery** — generated kernels routinely cause illegal memory access /
   unrecoverable CUDA errors; isolate at task level, auto-recover.
4. **Rich environmental feedback** — pass/fail + single speedup is not enough; expose structured
   profiling summaries + hacking detection.

Components (`kernelgym/`):
- **API server** (`server/api/server.py`, FastAPI, port 10907) — REST submit/query + worker register.
- **Task Manager** (`server/task_manager.py`) — Redis-backed priority queues + worker registry +
  timeout requeue.
- **GPU Worker** (`worker/gpu_worker.py`) — one per physical GPU; pulls tasks, runs each in a
  **subprocess**; main process stays CUDA-clean.
- **Subprocess Worker Pool** (`worker/subprocess_pool.py`) — persistent spawn subprocesses that own CUDA;
  auto-restart on CUDA error or task-count cap.
- **Worker Monitor** (`worker/worker_monitor.py`) — liveness via heartbeat + PID checks; restarts dead
  workers, resets GPU.
- **Backends / Toolkits / Workflows** (`backend/`, `toolkit/`, `workflow/`) — pluggable compile/load/run +
  correctness/perf/profiling logic.

Data flow: client -> API `/evaluate` -> TaskManager.submit_task (Redis LPUSH onto priority queue) ->
GPUWorker.get_next_task (RPOP) -> worker_pool.execute_task -> subprocess runs toolkit.evaluate ->
result back through mp.Queue -> TaskManager.complete_task (Redis HSET result) -> client polls
`/results/{task_id}`.

---

## 2. THE KEY PATTERN — CUDA fault isolation via subprocesses (MOST REUSABLE)

This is the single most important infra idea for us. Generated kernels crash GPUs; you cannot let that
take down the orchestrator.

### 2.1 Main process never touches CUDA
- `GPUWorker._initialize_gpu()` validates the GPU using **`nvidia-smi` ONLY** (`gpu_diagnostics.py
  test_gpu_health_nvidia_smi`) — parses `--query-gpu=name,memory.total --format=csv,noheader,nounits`,
  5s timeout. Deliberately does NOT call `torch.cuda.*` in the parent, so a CUDA error in a child can
  never corrupt the parent context.
- The parent only stores `device_id` (parsed from `"cuda:N"`). All CUDA work happens in children.

### 2.2 Two isolation strategies present
- **`task_executor.py` (IsolatedTaskExecutor)**: spawn a FRESH subprocess per task. Cleanest isolation,
  but pays torch+CUDA init (~2.5s, per paper) every task.
- **`subprocess_pool.py` (SubprocessWorkerPool)**: PERSISTENT subprocess(es) that init torch+CUDA ONCE,
  then loop pulling tasks over `mp.Queue`. `mp.get_context('spawn')` for full isolation (no inherited
  CUDA state). This is the production path used by `GPUWorker`.

**For our agent**: use the per-candidate fresh-subprocess model (task_executor style). We don't have
thousands of tasks so spawn cost is negligible, and fresh-process-per-candidate gives the strongest
isolation + cleanest timing. Reserve the persistent-pool optimization only for huge config sweeps where
init dominates.

### 2.3 Restart triggers (subprocess_pool)
A subprocess is marked dead / restarted when ANY of:
- It reports `worker_exiting: True` after detecting a CUDA error (classification below).
- Task **timeout** (queue.Empty on result) — process killed (`terminate`->`join(2)`->`kill`).
- `tasks_processed >= max_tasks_per_worker` (default **1** in settings = per-task isolation; prevents
  VRAM accumulation / fragmentation across tasks).
After shutdown the pool **sleeps 2.0s** before respawn (let GPU resources release; immediate restart
under load causes slow/timeout CUDA init). Emergency recovery recreates a worker if the pool empties
(sleeps 3.0s first).

### 2.4 CUDA-error classification (inside subprocess, `_persistent_worker_loop`)
A caught exception is a hard CUDA error (=> exit + restart) if:
```
"CUDA" in error_type or "CUDA" in error_message or "cuda" in error_message.lower()
or error_type in ["RuntimeError","CudaError"]   # note: ANY RuntimeError counts!
```
plus a profiler-dropout signal `PROFILER_NO_CUDA_EVENTS`. On hard error: best-effort
`_aggressive_gpu_cleanup` + `torch.cuda.synchronize()` (wrapped in try/except — sync often fails after a
real illegal-access) then break the loop so the parent respawns. Non-CUDA errors return
`worker_exiting: False` and the subprocess keeps serving.

`_is_memory_error` (`code_retry_manager.py`) is the canonical "GPU-corrupting error" matcher:
patterns = illegal memory access / illegal address / cuda error / cudaErrorIllegalAddress /
unspecified launch failure / device-side assert triggered / misaligned address / segmentation fault /
invalid memory / out of memory.

### 2.5 `_aggressive_gpu_cleanup(device_id)` recipe (after every task + before exit)
`torch.cuda.synchronize` -> `empty_cache` -> `gc.collect` -> `empty_cache` again ->
`reset_peak_memory_stats` + `reset_accumulated_memory_stats` -> final `synchronize`. All in try/except
(post-error syncs expected to fail). Only helps *recoverable* states; truly corrupted contexts are only
cleared by killing the process — which is why the process-per-candidate model matters.

---

## 3. Timing / latency measurement (DIRECTLY reusable)

`toolkit/kernelbench/timing.py :: time_execution_with_cuda_event`:
- `num_warmup` (default 3) iterations, each followed by `torch.cuda.synchronize(device)`.
- Then `num_trials` (default 10; settings `default_num_trials=100`) timed iterations, EACH with its own
  `torch.cuda.Event(enable_timing=True)` start/end pair: `start.record() / kernel() / end.record()`,
  then `synchronize(device)`, then `start.elapsed_time(end)` (ms).
- Per-trial events (not one pair around a loop) -> gives a distribution.
- `get_timing_stats` returns mean/std/min/max (3 sig figs) + num_trials + hardware name + device.
  **Reported runtime = mean.**

Gotchas / improvements for OUR perf bar (must match-or-beat baseline across ALL configs):
- KernelGYM uses **mean** of 10 trials. For latency comparison prefer **median or min** (min = least-noise
  estimate of true kernel cost); bump trials (>=50–100) + warmup (>=10–20) for stable H200 numbers,
  especially tiny decode kernels where launch overhead dominates.
- Per-trial sync includes per-call sync overhead; for very fast kernels add a "N-launches-then-one-sync"
  measurement to separate launch overhead from kernel time. KernelGYM only does per-call-sync.
- Pin device, set seeds, re-create inputs per measurement set. They `set_seed` before `get_inputs`, and
  set `CUDA_VISIBLE_DEVICES` to the single device for triton runs so the child sees one GPU
  (`pipeline.eval_kernel_against_ref`).

---

## 4. Correctness checking (reusable protocol)

`toolkit/kernelbench/correctness.py :: run_and_check_correctness`:
- `num_correct_trials` (settings: quick=3, standard=5, thorough=10) randomized inputs.
- Deterministic seeding: master `torch.manual_seed(42)`, derive per-trial seeds via `torch.randint`.
  Before EACH trial: `set_seed(trial_seed)` then build inputs, `set_seed` again then build the reference
  model, `set_seed` again then build the candidate model. (Seeding model builds matters if init random.)
- Run reference, `synchronize`, run candidate, `synchronize`.
- Check **shape first** (early-out on mismatch), then `torch.allclose(out, out_new, atol=1e-2, rtol=1e-2)`.
  On mismatch records `max_difference` / `avg_difference` per trial.
- Pass only if ALL trials pass. Any exception during candidate run -> `runtime_error` (compiled=True,
  correctness=False) with `get_error_name(e)`. All under `torch.no_grad()`.

For OUR agent: 1e-2/1e-2 is LOOSE (fine for fp16 KernelBench). Our real bar is "greedy/argmax decoding of
up to 100 tokens unchanged vs baseline" — so the correctness gate should be (a) tight allclose vs the
prebuilt `_vllm_fa3_C.fwd` baseline AND (b) an end-to-end argmax-equivalence check over the actual vLLM
configs. Reuse the multi-seed, multi-trial, shape-first, exception-as-runtime-error structure; tighten
numeric tolerance and add the token-equivalence harness on top.

---

## 5. Profiling feedback (torch.profiler path; we mostly use NCU, but the SHAPE matters)

`toolkit/kernelbench/profiling.py`:
- `profiling_context(enabled)` wraps `torch.profiler.profile(activities=[CPU,CUDA], record_shapes,
  profile_memory, with_stack)`. Overhead notes: record_shapes cheap, profile_memory ~5%, with_stack
  ~15-20% (default off).
- `extract_profiling_metrics(prof)` -> `prof.key_averages()` -> per-kernel dict
  `{name, cuda_time_us, cpu_time_us, count, [cuda_memory_usage]}`, **sorted by cuda_time_us desc**, plus
  totals (`total_cuda_time_us`, `total_self_cuda_time_us`, kernel_count) + memory_stats
  (allocated/reserved/max). Uses defensive `_safe_metric` trying multiple attr names across torch versions
  (`device_time_total`/`cuda_time_total`/...). **Lesson: profiler attr names drift between torch versions
  — probe defensively.**
- Empty-capture handling: events captured but 0 CUDA kernels -> warn; sets
  `profiling_warning="no_cuda_events"`. Pipeline RETRIES profiling up to `profiling_retry_count`
  (default 1) when empty (`run_profiling_only`). Persistent empty capture raises `PROFILER_NO_CUDA_EVENTS`
  treated like a CUDA error (restart subprocess + bounded retry).
- Profiling runs on a SEPARATE pass (`num_profiling_trials = min(10, num_trials)`) AFTER the timing pass,
  keeping timing clean of profiler overhead. **Adopt: never time and profile in the same pass.**

Our "concise actionable feedback" (component 2.1) is exactly this distilled top-N kernel table (name,
time, %, count) + memory. For NCU we produce an analogous concise table; the principle (rank by self time,
attribute to operators, report coverage %) carries over directly.

---

## 6. Anti-cheating / hacking detection (relevant to component 2.5 + correctness gate)

Paper's two failure modes (§pitfalls):
1. **Reward hacking** — copy the torch reference (passes correctness, ~1.0x), OR emit a kernel that is
   never executed (e.g. branch on `self.training` to skip kernel path, inflating measured speedup).
2. **Lazy optimization** — fuse/optimize only one trivial op, leave the rest to torch; clears Fast@1 but
   never Fast@1.2.

Detection (`toolkit/kernelbench/triton_detect.py`, `pipeline._run_triton_detection_step`):
- **Execution-based hacking check**: monkeypatch Triton's launch path (`JITFunction.run/launch/__call__/
  __getitem__`, `Autotuner`, low-level `CUDAKernel`, etc.) to RECORD which kernels actually executed, in
  BOTH no-grad and grad modes (`used = used_no_grad AND used_with_grad`) — catches the `self.training`
  trick. Triton backend executing ZERO triton kernels in either mode -> `decoy_kernel=True`, runtime=-1
  (treated incorrect).
- **Coverage**: `compute_triton_kernel_coverage` cross-references captured custom-kernel names against the
  profiler's kernel list -> `num_custom_kernels / num_total_kernels` ratio + custom-CUDA-time /
  total-CUDA-time ratio. Profiler shows kernels but 0 custom -> decoy. A `TorchOpsCallHook` hooks
  `OpOverload.__call__` to catch custom `torch.ops::ns::op` calls (non-core namespaces, with a CUDA tensor
  arg) — directly relevant to our CUDA backend (our kernel registers via TORCH_LIBRARY, invoked as
  `torch.ops._C...`).

**For OUR agent (FA3 distillation)**: the analog of "decoy/hacking" is the candidate secretly calling the
prebuilt baseline `_vllm_fa3_C` or another lib instead of its own kernel. Our enforcement (2.5) =
toolkit-only include path + `nm`/`ldd` scan of the compiled .so (no CUTLASS/cuBLAS/FlashInfer/
`_vllm_fa3_C` symbols). KernelGYM's *runtime execution hook* idea is a strong complement: at runtime, hook
`torch.ops` to verify OUR registered op actually executed and no forbidden op was called — cheap, catches
what static `nm`/`ldd` can miss (e.g. dlopen).
Early static validation (`toolkit/validation.py :: early_kernel_validation`) is the cheap pre-gate:
`compile(code,"<string>","exec")` for syntax, required-import presence checks, required-class presence,
entry-point instantiation — all WITHOUT touching the GPU. Adopt a CUDA-flavored version (must contain
`__global__`, must register the expected op name, must NOT include/link forbidden libs).

---

## 7. Compilation / sandboxed build (CUDA-specific, reusable)

`toolkit/kernelbench/loading.py` + `compile.py`:
- CUDA path uses PyTorch inline extension build. Build dir isolation: prepend
  `os.environ['TORCH_EXTENSIONS_DIR'] = '<build_dir>'` to the source so each candidate compiles into its
  own cache dir (avoids cross-candidate cache collisions; critical when compiling many variants).
- Triton path loads via a NamedTemporaryFile .py module (`load_custom_model_with_tempfile`) + `importlib`,
  cleaned up in `graceful_eval_cleanup` (close + os.remove). For us (single .cu), the per-candidate build
  dir + temp source is the relevant trick.
- Compile wrapped capturing stdout/stderr (`redirect_stdout/redirect_stderr` to StringIO) -> returns
  `{compiled, stdout, stderr, error}`. **Lock-file errors** (`"lock" in err or "No such file or
  directory"`) treated as transient -> return None to trigger RETRY rather than hard compile-fail.
  Concurrent torch-extension builds share a lock; expect transient lock failures under parallelism.
- `os.environ["TORCH_USE_CUDA_DSA"]="1"` set before compile (device-side assertions) for better error
  localization.

---

## 8. Queue / scheduler / multi-GPU (adopt selectively)

`server/task_manager.py` (Redis):
- Priority queues HIGH/NORMAL/LOW (`...:queue:priority:{p}`), plus optional per-worker queues
  (`...:queue:worker:{wid}`). `submit_task` LPUSH, `get_next_task` RPOP (FIFO per priority). Worker queue
  checked before priority queues -> lets you PIN a task to a specific GPU/worker (relevant to our "perf
  measurement = dedicated 1 GPU" requirement: pin perf tasks to a worker, let correctness tasks float).
- **Round-robin load balancer** (`WorkerLoadBalancer.get_next_worker`) only considers workers with
  heartbeat <=30s old.
- **Timeout requeue** (`_queue_wait_monitor`, every 20s): scans per-worker queues; task waited > 
  `queue_wait_timeout` (min(task_timeout, default 180s)) or stale assignment -> requeue to priority queue.
  Keeps throughput up when a worker dies mid-assignment.
- Result/state persisted in Redis HSETs (`...:task:{id}` status, `...:result:{id}` payload). Survives
  restarts -> effectively their checkpoint store for in-flight tasks.

**For our agent**: full Redis+FastAPI is heavier than needed for a handful of candidates. But the concepts
map to a lightweight local scheduler:
- A simple in-process asyncio queue / `concurrent.futures` with N=4 GPU slots.
- Pin a candidate's PERF run to exactly one GPU (no co-tenancy) for clean timing; allow CORRECTNESS runs
  to share GPUs (tolerance checks, contention-insensitive). Realizes one-GPU-one-task for timing while
  parallelizing correctness across 4x H200.
- Persist task->result mapping to JSON/sqlite for checkpoint+resume (Redis HSET is the pattern). Each
  completed candidate record {code hash, compiled, correct, latency, ncu summary} is a checkpoint row +
  evolution-plot data point.

Settings worth copying (`config/settings.py`):
- `worker_pool_size=1`, `max_tasks_per_worker=1` => **strict serial, per-task isolation** (their own
  default for reliability over throughput). We want this for perf runs.
- `default_timeout=600`; per-task hard timeout in GPUWorker = **35s** (overrideable per task via payload
  `timeout`) "to account for ~30x-second tasks". For us, set per-config timeouts generously above expected
  kernel+compile time; treat timeout as hard fail (no retry).
- `worker_monitor_heartbeat_timeout=120`, `worker_monitor_interval=30`, `restart_cooldown=60`.

---

## 9. Retry policy (opinionated, important)

`code_retry_manager.py`:
- **NEVER retry memory/illegal-access errors** — treated as deterministic hard failures (retrying just
  crashes another GPU). Correct: a kernel with an illegal access reliably re-crashes.
- **Timeouts are NOT retried** (subprocess_pool: timeout -> kill worker, do NOT requeue, "to free up the
  worker queue / avoid blocking"). A hung kernel hangs again.
- Only **profiler-dropout** (`PROFILER_NO_CUDA_EVENTS`) and transient **compile lock-file** errors get
  bounded retries (max 2). These are infra flakes, not kernel bugs.
- Retry bookkeeping keyed by `original_task_id` (strips `_retry` suffix), Redis counters with TTL.

Takeaway: classify failures into {kernel-bug (no retry), infra-flake (bounded retry)}. Compile lock
contention and profiler dropouts are the only infra-flakes worth retrying. Everything memory/timeout is a
real candidate defect — record it and move on.

---

## 10. Worker liveness / self-recovery (multi-GPU robustness)

`worker_monitor.py` (run with `--persistent`):
- Tracks `:expected_workers` (SET) + `:expected_worker:{id}` (device/host) + `:worker_process:{id}` (pid),
  populated by `start_all_with_monitor.sh` at launch.
- Restart a worker if: heartbeat key expired/missing, recorded PID dead (`os.kill(pid,0)`),
  `cuda_error_shutdown` flag set, or heartbeat stale > timeout.
- On restart: kill old process **group** (`os.setsid` at spawn + `os.killpg`, SIGTERM then SIGKILL after
  10s), attempt **GPU reset** (`nvidia-smi --gpu-reset -i N`, falls back to a torch
  empty_cache/reset_stats subprocess since gpu-reset usually needs root), sleep 5s, relaunch
  `single_worker`, then **wait 45s** before re-checking (so init doesn't look like a crash).
- Heartbeats: worker writes Redis hash every 10s with `expire(120)`; if API rejects heartbeat (409/410)
  the worker self-evicts to avoid "ghost" entries.

For us: even with a lightweight local scheduler, the pattern — spawn children in their own process group,
escalate SIGTERM->SIGKILL, attempt GPU cleanup, grace-period before declaring a fresh child dead — is
worth replicating for the 4-GPU pool so one wedged kernel/GPU can't stall the evolution loop.

---

## 11. Startup / config (operational)

- `setup.sh`: pip reqs + pydantic-settings + iproute2 + local redis.
- `scripts/auto_configure.sh`: detects host IP + GPU list, picks free ports from candidate pools, writes
  `.env`. `--force` overwrite, `--use-indexed-ports`.
- `start_all_with_monitor.sh`: sources `.env`, starts redis (if local+down), API server, worker monitor
  (`--persistent`), then one `single_worker` per GPU in `GPU_DEVICES`, registering each into Redis
  expected-worker keys. Logs per-component to `logs/`. 0.3s stagger between worker launches.
- Multi-node: workers point at remote `REDIS_HOST`/`API_HOST`, get a server-assigned `node_id`
  (`/node/allocate`), worker_id = `{node_id}_gpu_{N}`.

We don't need Redis/FastAPI; but "one long-lived process per GPU, registered centrally, monitored, logs to
per-GPU files" is the right topology for our 4x H200 perf pool.

---

## 12. Concrete things to ADOPT in our agent

1. **Process-per-candidate execution** (spawn): parent never imports torch.cuda; every compile+run+time
   happens in a child so a crash/illegal-access/hang is contained. Strongest reliability lever.
2. **nvidia-smi-only GPU health check** in the parent (no CUDA init in orchestrator).
3. **CUDA-event timing**: warmup then per-trial Event pairs + per-trial sync; report a distribution. Use
   median/min + more trials than KernelGYM's mean/10 for our strict perf bar; add a separate
   "launch-overhead" measurement for tiny decode kernels.
4. **Separate timing pass from profiling pass** (never profile during the timed pass).
5. **Failure taxonomy + retry policy**: memory/illegal-access + timeout = hard fail, no retry; compile
   lock-file + profiler-dropout = bounded retry (<=2). Reuse `_is_memory_error` pattern list verbatim.
6. **Per-candidate isolated build dir** via `TORCH_EXTENSIONS_DIR`; capture stdout/stderr; treat lock-file
   compile errors as transient.
7. **Runtime execution hook** (hook `torch.ops.OpOverload.__call__`) to assert OUR op actually ran and no
   forbidden op was invoked — complements static `nm`/`ldd` anti-cheat scan.
8. **Cheap pre-GPU static validation** (syntax compile, required `__global__`/op-registration present,
   forbidden includes/libs absent) before spending a GPU slot.
9. **Pin perf runs to a dedicated GPU (1 task/GPU); let correctness runs share GPUs** — realizes the
   one-GPU-one-task-for-timing principle while using all 4 H200s.
10. **Checkpoint = persisted task->result records** (their Redis HSET; we use file/sqlite) keyed by
    candidate/code hash, written on completion -> resume + evolution-plot data source.
11. **`_aggressive_gpu_cleanup` recipe** after each task in long-lived children (if we ever use a pool).
12. **Worker monitor pattern**: process-group spawn, SIGTERM->SIGKILL escalation, grace period before
    declaring a relaunched child dead, GPU cleanup on restart.

---

## 13. Gotchas / pitfalls observed

- ANY `RuntimeError` is treated as a CUDA error by the subprocess loop (over-broad) — for us, classify
  more precisely (compile vs correctness vs CUDA) so we don't needlessly kill a child on a benign Python
  error.
- `torch.cuda.synchronize()` after a real illegal-access usually THROWS — all cleanup must be try/except;
  the only reliable recovery is killing the process. `nvidia-smi --gpu-reset` usually needs root and "may
  not be available in compute mode" — don't rely on it; rely on process death + (worst case) leaving that
  GPU out of the pool.
- Profiler attribute names differ across torch versions — probe multiple names; empty captures happen and
  must be retried/handled, not treated as decoy.
- allclose tolerance 1e-2/1e-2 is too loose for our token-equivalence goal — tighten and add an argmax
  end-to-end check.
- `set_seed` must be called before BOTH input gen and EACH model construction for determinism.
- Immediate subprocess restart after a crash causes slow/timeout CUDA init under load — insert a short
  (2-3s) delay before respawn.
- Timeouts must NOT be retried (hangs reproduce) and must hard-kill the process group, else the queue
  wedges.

---

## 14. File index (absolute paths)

- Worker/isolation: `/data-fast/agent/reference/related-works/KernelGYM/kernelgym/worker/gpu_worker.py`,
  `.../worker/subprocess_pool.py`, `.../worker/task_executor.py`, `.../worker/worker_monitor.py`,
  `.../worker/single_worker.py`
- Timing/profiling/correctness/compile:
  `.../kernelgym/toolkit/kernelbench/{timing,profiling,correctness,compile,loading,pipeline,triton_detect}.py`
- Scheduling/retry/errors: `.../kernelgym/server/task_manager.py`, `.../server/code_retry_manager.py`,
  `.../utils/error_classifier.py`, `.../utils/gpu_diagnostics.py`
- Anti-cheat/validation: `.../kernelgym/toolkit/validation.py`, `.../toolkit/kernelbench/triton_detect.py`
- Config/ops: `.../kernelgym/config/settings.py`, `.../start_all_with_monitor.sh`,
  `.../scripts/auto_configure.sh`
- Paper: `.../drkernel/sections/kernelgym.tex`, `.../sections/pitfalls.tex`
