"""Component interface contract.

The orchestrator imports the concrete functions below; the components are built
independently (in parallel) and MUST match these signatures and return types so
integration is mechanical. Every component returns ONLY distilled, bounded output
(context discipline) — never raw dumps to the orchestrator/LLM.

All components import shared types from fa3agent.common and config from fa3agent.config,
and drive baseline/candidate via fa3agent.shapes (make_inputs / fwd_args / call_fwd).

--------------------------------------------------------------------------------
sandbox/build.py        (build)
  build_candidate(cu_path, namespace, work_dir, extra_defs=None) -> common.BuildResult
     Compile a candidate .cu into a .so for sm_90a using TOOLKIT-ONLY includes (torch +
     CUDA; NO cutlass). Defines -DFA3_LIB_NAME=<namespace>. On success the .so registers
     torch.ops.<namespace>.{fwd,get_scheduler_metadata}; verify by loading and dumping the
     schema, comparing to config.FWD_SCHEMA arity/types. Classify env-fatal errors.

sandbox/anticheat.py    (2.5)
  check(cu_path, so_path=None) -> common.AntiCheatResult
     Source scan (FORBIDDEN_INCLUDE_SUBSTR / FORBIDDEN_SOURCE_TOKENS, ignoring comments)
     + binary scan (nm -D / ldd vs FORBIDDEN_DYNAMIC_SUBSTR / ALLOWED_LINK_SUBSTR).
     ok=False with violations[] if any forbidden lib/symbol/include is present.

--------------------------------------------------------------------------------
verify/golden.py        (golden cache)
  class GoldenCache: get(cfg, seed=0) -> dict; caches baseline out+lse + fp32-ref error.
  golden_baseline_error(cfg, seed=0) -> float

verify/correctness.py   (verify FIRST)
  check_correctness(namespace, cfgs, seed=0, gpu_id=0) -> list[common.ConfigResult]
     Per cfg: candidate via shapes.call_fwd, compare out+lse to baseline + fp32 ref.
     Pass iff candidate_err <= ERR_MULT*baseline_err AND allclose tol. (ratio left 0.)

verify/decode_equiv.py  (promotion gate)
  check_decode_equiv(so_path, namespace, model=None, num_tokens=100) -> (bool, note)
     Monkeypatch vLLM flash_attn_varlen_func -> torch.ops.<ns>.fwd, tiny-model greedy
     decode num_tokens (temp=0, enforce_eager), compare token ids to baseline. Subprocess.

verify/cudagraph.py     (compat gate — binary)
  check_cuda_graph(so_path, namespace, cfg_name=None, gpu_id=0) -> dict
     Capture+replay the candidate op in a CUDA graph (baseline captured first as a control);
     verify replay matches eager. {ok, captured, replay_match, skip, max_diff, note}. Subprocess.
     A non-skip ok=False means the kernel isn't graph-capturable -> not a valid vLLM drop-in.

verify/sanitizer.py     (2.2 — on-demand only)
  run_sanitizer(so_path, namespace, cfg, tool='memcheck') -> dict
     tool in {memcheck,racecheck,initcheck,synccheck}. Minimal driver under compute-sanitizer
     (subprocess); parse to {clean: bool, errors: [..]} (distilled).

--------------------------------------------------------------------------------
perf/measure.py         (sec 3)
  measure(namespace, cfg, gpu_id, baseline_ns='_vllm_fa3_C') -> common.ConfigResult
     CUDA-event ratio: warm to steady clocks, L2-flush between samples, median+p90;
     baseline AND candidate INTERLEAVED in-process (clocks unlockable). ONE kernel/GPU.

perf/ncu_profile.py     (2.1)
  profile(so_path, namespace, cfg, gpu_id, run_dir) -> common.ProfileFeedback
     2-pass ncu collect, parse via ncu_report module (perf/ncu_utils), classify bottleneck,
     emit terse one-bottleneck JSON + <=5 ranked evidence-cited list. ~24-metric subset only.

--------------------------------------------------------------------------------
sass/analyze.py         (2.3)
  analyze(so_path, namespace, baseline_so=None, oracle_cubin=None) -> dict
     cuobjdump/nvdisasm; instruction-mix histogram, reg/smem/spill; diff vs baseline/oracle.

sass/cutlass_oracle.py  (2.6)
  build_oracle(cfg, work_dir) -> dict
     Minimal CuTe/CUTLASS or CuTeDSL fmha/gemm OUT OF BAND, dump PTX/SASS, distilled hints.

docs/retrieve.py        (2.4)
  retrieve(query, k=5, scope=None) -> str   # top-k short snippets over reference docs.

reflib/hints.py         (2.7)
  get_hints(topic, k=3) -> str   # file:line + snippet from TK/DeepGEMM/FlashMLA/flashinfer/...

--------------------------------------------------------------------------------
knowledge/store.py      (typed in-loop distillation)
  class KnowledgeStore:
    add_note(ntype, title, body, **frontmatter) -> str   # ntype in {ks,pf,alg}
    query(q, ntype=None, k=5) -> list[dict]; get(note_id, follow=False) -> str
    grep(pattern) -> list[str]; seed()

plot/evolution.py       (AVO-style)
  update_plot(run_dir, candidates_jsonl, out_png=None) -> str
     X=committed version, Y=speedup; running-best min+geomean line, per-config dashed,
     baseline=1.0 line. Regenerated each step.
"""
