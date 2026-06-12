"""Component 2.7 -- reference-library optimization hints.

get_hints(topic, k=3) -> str

Mine the vendored reference libraries (under reference/libraries/) for *readable
models* of the raw Hopper primitives we must hand-write as pure inline-PTX in a
single self-contained .cu (TMA / mbarrier / WGMMA / shared-mem matrix descriptor /
online-softmax / warp specialization / pipeline / split-KV).

Strategy: a CURATED topic -> [(file, grep-pattern)] index, hand-built from the exact
file:line pointers in .research/ref-libs.md.  We ripgrep each pattern, take the top
matches, and return `file:line` + a short (~15-line) snippet for the top-k.  Output is
strictly bounded (context discipline): never a raw file dump.

stdlib + ripgrep only.  CPU-only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Optional, Tuple

from ..runlog import logged

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Resolve the reference-library root relative to this file so the component is
# location-independent: <repo>/fa3agent/reflib/hints.py -> <repo>/reference/libraries
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_REF_ROOT = os.path.join(_REPO_ROOT, "reference", "libraries")

# --------------------------------------------------------------------------- #
# Output bounds (context discipline)
# --------------------------------------------------------------------------- #

_SNIPPET_BEFORE = 4          # lines of context before the match
_SNIPPET_AFTER = 11          # lines of context after the match
_MAX_SNIPPET_LINES = 16      # hard cap on snippet height
_MAX_LINE_CHARS = 160        # hard cap on per-line width
_MAX_TOTAL_CHARS = 8000      # hard cap on whole return string
_DEFAULT_K = 3
_MAX_K = 8                   # never return more than this regardless of k

# --------------------------------------------------------------------------- #
# Curated topic -> [(relative_file, grep_pattern)] index.
#
# Files are relative to _REF_ROOT.  Patterns are ripgrep regexes; the FIRST match
# in the file is used as the snippet anchor.  Entries are listed best-first; the
# top-k across all matched topics (de-duplicated by file) are returned.
#
# Cross-checked against .research/ref-libs.md exact file:line pointers.
# --------------------------------------------------------------------------- #

_TK = "ThunderKittens"
_DG = "DeepGEMM"
_FI = "flashinfer"
_MLA = "FlashMLA"

_TOPIC_INDEX = {
    # ----- WGMMA tensor-core matmul (bf16 m64nNk16) -------------------------- #
    "wgmma": [
        (f"{_TK}/include/ops/group/mma/base/64x64.impl",
         r"wgmma\.mma_async\.sync\.aligned\.m64n64k16\.f32\.bf16\.bf16"),
        (f"{_TK}/include/ops/group/mma/warpgroup.cuh",
         r"wgmma\.fence\.sync\.aligned"),
        (f"{_DG}/deep_gemm/include/deep_gemm/mma/sm90.cuh",
         r"BF16MMASelector"),
        (f"{_FI}/include/flashinfer/mma.cuh",
         r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32"),
    ],

    # ----- WGMMA shared-memory matrix descriptor encoding ------------------- #
    "descriptor": [
        (f"{_TK}/include/types/shared/descriptor.cuh",
         r"matrix_descriptor_raw"),
        (f"{_TK}/include/types/shared/descriptor.cuh",
         r"chunk_descriptor"),
        (f"{_DG}/deep_gemm/include/deep_gemm/mma/sm90.cuh",
         r"make_smem_desc"),
        (f"{_DG}/deep_gemm/include/deep_gemm/mma/sm90.cuh",
         r"make_gmma_desc"),
    ],

    # ----- TMA tensor-memory-accelerator async copy (device PTX) ------------ #
    "tma": [
        (f"{_TK}/include/ops/thread/memory/tile/tma.cuh",
         r"cp\.async\.bulk\.tensor\.5d\.shared::cluster\.global"),
        (f"{_TK}/include/ops/thread/memory/tile/tma.cuh",
         r"cp\.async\.bulk\.tensor\.5d\.global\.shared::cta"),
        (f"{_TK}/include/ops/thread/util/tma.cuh",
         r"mbarrier\.arrive\.expect_tx\.shared::cta"),
        (f"{_DG}/deep_gemm/include/deep_gemm/ptx/tma.cuh",
         r"tma_load_1d"),
    ],

    # ----- HOST-side TMA descriptor creation (cuTensorMapEncodeTiled) ------- #
    "tma_host": [
        (f"{_TK}/include/types/global/tma.cuh",
         r"cuTensorMapEncodeTiled"),
        (f"{_TK}/include/types/global/tma.cuh",
         r"create_tensor_map"),
    ],

    # ----- mbarrier init / arrive / try_wait.parity phase idiom ------------- #
    "mbarrier": [
        (f"{_TK}/include/ops/thread/util/sync.cuh",
         r"mbarrier\.init\.shared::cta"),
        (f"{_TK}/include/ops/thread/util/sync.cuh",
         r"mbarrier\.try_wait\.parity\.shared::cta"),
        (f"{_TK}/include/ops/thread/util/tma.cuh",
         r"mbarrier\.arrive\.expect_tx\.shared::cta"),
        (f"{_DG}/deep_gemm/include/deep_gemm/ptx/tma.cuh",
         r"mbarrier_wait_and_flip_phase"),
    ],

    # ----- online softmax (exp2 + log2(e) scale, running max/sum) ----------- #
    "online_softmax": [
        (f"{_TK}/kernels/attention/mha_h100_lcf/mha_h100_lcf.cu",
         r"exp2\("),
        (f"{_FI}/include/flashinfer/attention/prefill.cuh",
         r"update_mdo_states"),
    ],

    # ----- warp specialization (producer/consumer + register reconfig) ------ #
    "warp_specialization": [
        (f"{_DG}/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh",
         r"warpgroup_reg_dealloc|warpgroup_reg_alloc|reg_dealloc|reg_alloc"),
        (f"{_DG}/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh",
         r"get_next_block"),
        (f"{_TK}/kernels/attention/mha_h100/mha_h100.cu",
         r"decrease_registers|increase_registers"),
    ],

    # ----- multi-stage software pipeline (full/empty barriers, phase) ------- #
    "pipeline": [
        (f"{_DG}/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh",
         r"empty_barriers\[.*\]->wait"),
        (f"{_DG}/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh",
         r"arrive_and_expect_tx"),
        (f"{_DG}/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh",
         r"kNumQStages"),
    ],

    # ----- split-KV / combine epilogue (long-context decode) ---------------- #
    "split_kv": [
        (f"{_MLA}/csrc/sm90/decode/dense/splitkv_mla.cuh",
         r"is_no_split|n_split_idx"),
        (f"{_MLA}/csrc/smxx/decode/combine/combine.cu",
         r"split|lse|accum"),
        (f"{_DG}/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh",
         r"kNumKVStages"),
    ],

    # ----- WGMMA fence / commit / wait-group bracketing --------------------- #
    "wgmma_fence": [
        (f"{_TK}/include/ops/group/mma/warpgroup.cuh",
         r"wgmma\.fence\.sync\.aligned"),
        (f"{_TK}/include/ops/group/mma/warpgroup.cuh",
         r"wgmma\.commit_group\.sync\.aligned"),
        (f"{_TK}/include/ops/group/mma/warpgroup.cuh",
         r"wgmma\.wait_group\.sync\.aligned"),
    ],

    # ----- TMA store + fence.proxy.async (O / logsumexp epilogue) ----------- #
    "tma_store": [
        (f"{_TK}/include/ops/thread/memory/tile/tma.cuh",
         r"fence\.proxy\.async\.shared::cta"),
        (f"{_TK}/include/ops/thread/memory/tile/tma.cuh",
         r"cp\.async\.bulk\.tensor\.5d\.global\.shared::cta"),
    ],

    # ----- non-WGMMA fallback (Ampere ldmatrix + mma.sync) ------------------ #
    "ldmatrix": [
        (f"{_FI}/include/flashinfer/mma.cuh",
         r"ldmatrix_m8n8x4"),
        (f"{_FI}/include/flashinfer/cp_async.cuh",
         r"cp\.async"),
    ],
}

# Keyword aliases -> canonical topic keys.  Matched as substrings against the
# whitespace/underscore-normalized query.  Multi-word aliases are checked too.
_ALIASES = {
    # wgmma
    "wgmma": "wgmma",
    "mma_async": "wgmma",
    "m64n": "wgmma",
    "tensor core": "wgmma",
    "tensorcore": "wgmma",
    "matmul": "wgmma",
    "gemm": "wgmma",
    "bf16": "wgmma",
    # descriptor
    "descriptor": "descriptor",
    "smem desc": "descriptor",
    "smem_desc": "descriptor",
    "gmma desc": "descriptor",
    "matrix descriptor": "descriptor",
    "swizzle": "descriptor",
    # tma device load
    "tma load": "tma",
    "tma_load": "tma",
    "cp.async.bulk": "tma",
    "bulk tensor": "tma",
    "async copy": "tma",
    # tma host
    "tma descriptor": "tma_host",
    "cutensormap": "tma_host",
    "tensor map": "tma_host",
    "tensormap": "tma_host",
    "encode tiled": "tma_host",
    "create_tensor_map": "tma_host",
    "host descriptor": "tma_host",
    # mbarrier
    "mbarrier": "mbarrier",
    "barrier": "mbarrier",
    "semaphore": "mbarrier",
    "phase": "mbarrier",
    "try_wait": "mbarrier",
    "arrive": "mbarrier",
    "expect_tx": "mbarrier",
    "expect bytes": "mbarrier",
    # online softmax
    "online softmax": "online_softmax",
    "softmax": "online_softmax",
    "exp2": "online_softmax",
    "running max": "online_softmax",
    "rowmax": "online_softmax",
    "rowsum": "online_softmax",
    "logsumexp": "online_softmax",
    "lse": "online_softmax",
    # warp specialization
    "warp specialization": "warp_specialization",
    "warp spec": "warp_specialization",
    "warp-specialization": "warp_specialization",
    "producer": "warp_specialization",
    "consumer": "warp_specialization",
    "register reconfig": "warp_specialization",
    "reg_alloc": "warp_specialization",
    "reg_dealloc": "warp_specialization",
    "warpgroup": "warp_specialization",
    # pipeline
    "pipeline": "pipeline",
    "multi-stage": "pipeline",
    "multistage": "pipeline",
    "double buffer": "pipeline",
    "ring buffer": "pipeline",
    "stages": "pipeline",
    "full barrier": "pipeline",
    "empty barrier": "pipeline",
    # split-kv
    "split-kv": "split_kv",
    "split kv": "split_kv",
    "splitkv": "split_kv",
    "split-k": "split_kv",
    "combine": "split_kv",
    "decode": "split_kv",
    # wgmma fence
    "wgmma fence": "wgmma_fence",
    "fence": "wgmma_fence",
    "commit_group": "wgmma_fence",
    "wait_group": "wgmma_fence",
    # tma store
    "tma store": "tma_store",
    "store_async": "tma_store",
    "epilogue": "tma_store",
    "proxy fence": "tma_store",
    # ldmatrix / ampere fallback
    "ldmatrix": "ldmatrix",
    "ampere": "ldmatrix",
    "m16n8": "ldmatrix",
    "fallback": "ldmatrix",
    "mma.sync": "ldmatrix",
}


# --------------------------------------------------------------------------- #
# Topic resolution
# --------------------------------------------------------------------------- #

def _normalize(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def _resolve_topics(topic: str) -> List[str]:
    """Map a free-form query to an ordered, de-duplicated list of canonical topic keys."""
    norm = _normalize(topic)
    ordered: List[str] = []

    # 1) exact canonical key (allow caller to pass the canonical name directly)
    raw_key = topic.strip().lower().replace(" ", "_").replace("-", "_")
    if raw_key in _TOPIC_INDEX:
        ordered.append(raw_key)

    # 2) alias substring matches; prefer longer aliases first (more specific)
    for alias in sorted(_ALIASES, key=len, reverse=True):
        alias_norm = _normalize(alias)
        if alias_norm and alias_norm in norm:
            key = _ALIASES[alias]
            if key not in ordered:
                ordered.append(key)

    return ordered


# --------------------------------------------------------------------------- #
# ripgrep + snippet extraction
# --------------------------------------------------------------------------- #

def _ripgrep_first(pattern: str, abs_path: str) -> Optional[int]:
    """Return the 1-based line number of the first match, or None."""
    if not os.path.isfile(abs_path):
        return None
    if shutil.which("rg") is None:
        return _python_grep_first(pattern, abs_path)
    try:
        proc = subprocess.run(
            ["rg", "--no-heading", "--line-number", "--max-count", "1",
             "--", pattern, abs_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return _python_grep_first(pattern, abs_path)
    if proc.returncode != 0 or not proc.stdout:
        return None
    first = proc.stdout.splitlines()[0]
    # format: "<lineno>:<text>"
    head, _, _ = first.partition(":")
    try:
        return int(head)
    except ValueError:
        return None


def _python_grep_first(pattern: str, abs_path: str) -> Optional[int]:
    """Pure-stdlib fallback if ripgrep is unavailable."""
    import re
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))
    try:
        with open(abs_path, "r", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                if rx.search(line):
                    return i
    except OSError:
        return None
    return None


def _read_snippet(abs_path: str, line_no: int) -> List[str]:
    """Read ~_MAX_SNIPPET_LINES lines centered on line_no (1-based)."""
    start = max(1, line_no - _SNIPPET_BEFORE)
    end = line_no + _SNIPPET_AFTER
    out: List[str] = []
    try:
        with open(abs_path, "r", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                if i < start:
                    continue
                if i > end:
                    break
                text = line.rstrip("\n")
                if len(text) > _MAX_LINE_CHARS:
                    text = text[:_MAX_LINE_CHARS - 3] + "..."
                marker = ">" if i == line_no else " "
                out.append(f"{i:>5}{marker}| {text}")
                if len(out) >= _MAX_SNIPPET_LINES:
                    break
    except OSError:
        return []
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@logged("get_hints")
def get_hints(topic: str, k: int = _DEFAULT_K) -> str:
    """Return up to ``k`` bounded reference-library hints for ``topic``.

    Each hint is a `file:line` pointer (path relative to reference/libraries/) plus
    a ~15-line source snippet, mined from ThunderKittens / DeepGEMM / flashinfer /
    FlashMLA for readable models of raw Hopper primitives.

    Returns a single distilled, bounded string. Never raw-dumps a file.
    """
    if not isinstance(topic, str) or not topic.strip():
        return "[reflib] empty topic; try one of: " + ", ".join(sorted(_TOPIC_INDEX))
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = _DEFAULT_K
    k = max(1, min(k, _MAX_K))

    topics = _resolve_topics(topic)
    if not topics:
        return (f"[reflib] no topic match for {topic!r}. "
                "Known topics: " + ", ".join(sorted(_TOPIC_INDEX)) + ".")

    # Collect (file, pattern) entries across matched topics, in priority order.
    seen_files = set()
    hits: List[Tuple[str, str, int]] = []   # (abs_path, rel_path, line_no)
    for tkey in topics:
        for rel_path, pattern in _TOPIC_INDEX.get(tkey, ()):
            if rel_path in seen_files:
                continue
            abs_path = os.path.join(_REF_ROOT, rel_path)
            line_no = _ripgrep_first(pattern, abs_path)
            if line_no is None:
                continue
            seen_files.add(rel_path)
            hits.append((abs_path, rel_path, line_no))
            if len(hits) >= k:
                break
        if len(hits) >= k:
            break

    if not hits:
        return (f"[reflib] matched topics {topics} but found no live snippets under "
                f"{_REF_ROOT} (files missing?).")

    header = f"[reflib] topic={topic!r} -> {','.join(topics)}  ({len(hits)} hit(s))"
    blocks: List[str] = [header]
    for abs_path, rel_path, line_no in hits:
        snippet = _read_snippet(abs_path, line_no)
        loc = f"reference/libraries/{rel_path}:{line_no}"
        block = ["", f"=== {loc} ==="]
        block.extend(snippet)
        blocks.append("\n".join(block))

    result = "\n".join(blocks)
    if len(result) > _MAX_TOTAL_CHARS:
        result = result[:_MAX_TOTAL_CHARS - 3] + "..."
    return result


if __name__ == "__main__":  # tiny manual smoke
    for q in ("wgmma bf16", "tma descriptor", "online softmax",
              "warp specialization", "mbarrier phase", "split-kv"):
        print("#" * 70)
        print(get_hints(q, k=2))
