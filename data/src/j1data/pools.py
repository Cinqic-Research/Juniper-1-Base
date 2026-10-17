"""Worker pools that fail fast when a worker dies.

``multiprocessing.Pool`` silently replaces a worker that is killed (for example by the
kernel OOM killer or a SIGKILL) and then waits forever for the lost task, so a stage hangs
instead of failing and being resumed. ``concurrent.futures.ProcessPoolExecutor`` marks the
pool broken and raises ``BrokenProcessPool`` in the parent; committed units stay committed
and the next run resumes from them.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any


def _exit_with_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(1.0)
    os._exit(1)  # the parent died (for example SIGKILL): do not linger holding memory


def _initialize(parent: int, initializer: Callable[..., None] | None, initargs: tuple[Any, ...]) -> None:
    threading.Thread(target=_exit_with_parent, args=(parent,), daemon=True).start()
    if initializer is not None:
        initializer(*initargs)


def run_unordered(
    function: Callable[[Any], Any],
    jobs: Iterable[Any],
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    max_tasks_per_child: int | None = None,
) -> Iterator[Any]:
    """Yield ``function(job)`` results in completion order; raise if any job or worker fails.

    ``max_tasks_per_child`` recycles worker memory by running the jobs in batches of
    ``workers * max_tasks_per_child``, each in a fresh pool (the executor's own option of that
    name can hang on Python 3.12 when a worker retires while tasks are queued)."""
    pending = list(jobs)
    size = len(pending) if max_tasks_per_child is None else max(1, workers * max_tasks_per_child)
    for start in range(0, len(pending), max(size, 1)):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize,
            initargs=(os.getpid(), initializer, initargs),
        ) as pool:
            futures = [pool.submit(function, job) for job in pending[start : start + size]]
            try:
                for future in concurrent.futures.as_completed(futures):
                    yield future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
