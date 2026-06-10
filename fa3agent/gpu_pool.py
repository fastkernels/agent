"""Multi-GPU lease scheduler.

Rules (spec section 3): for PERFORMANCE/PROFILING never run more than one kernel at a
time on a GPU; for CORRECTNESS multiple kernels may share a GPU.  Perf/NCU/sanitizer run
as subprocesses pinned with CUDA_VISIBLE_DEVICES, so "one kernel per GPU" == one
exclusive lease per GPU id at a time.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterable, Any

from . import config as C


class GPUPool:
    def __init__(self, gpu_ids: list[int] | None = None):
        self.gpu_ids = list(gpu_ids if gpu_ids is not None else C.GPU_IDS)
        self._locks = {g: threading.Lock() for g in self.gpu_ids}
        self._rr = 0
        self._rr_lock = threading.Lock()

    @contextmanager
    def exclusive(self, gpu_id: int):
        """Hold a GPU exclusively (perf/NCU/sanitizer)."""
        lk = self._locks[gpu_id]
        lk.acquire()
        try:
            yield gpu_id
        finally:
            lk.release()

    def next_gpu(self) -> int:
        with self._rr_lock:
            g = self.gpu_ids[self._rr % len(self.gpu_ids)]
            self._rr += 1
            return g

    def map_exclusive(self, fn: Callable[[Any, int], Any], items: Iterable[Any]) -> list:
        """Run fn(item, gpu_id) across items, guaranteeing <=1 task per GPU at a time."""
        items = list(items)
        results: list = [None] * len(items)
        if not items:
            return results

        def worker(idx_item):
            idx, item = idx_item
            gpu = self.gpu_ids[idx % len(self.gpu_ids)]
            with self.exclusive(gpu):
                results[idx] = fn(item, gpu)

        with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as ex:
            list(ex.map(worker, enumerate(items)))
        return results

    def map_shared(self, fn: Callable[[Any, int], Any], items: Iterable[Any],
                   max_per_gpu: int = 1) -> list:
        """Run fn(item, gpu_id) across items; correctness may share GPUs."""
        items = list(items)
        results: list = [None] * len(items)
        if not items:
            return results

        def worker(idx_item):
            idx, item = idx_item
            gpu = self.gpu_ids[idx % len(self.gpu_ids)]
            results[idx] = fn(item, gpu)

        workers = max(1, len(self.gpu_ids) * max_per_gpu)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(worker, enumerate(items)))
        return results
