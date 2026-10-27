"""Cooperative cancellation for DSL solving."""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Iterator


class SolveCancelledError(RuntimeError):
    """Raised when a DSL solve is canceled by the caller."""


class LocalSearchBudgetExpired(RuntimeError):
    """A bounded optional witness construction ran out of time."""


_LOCAL_SEARCH_BUDGET = ContextVar("dsl_local_search_budget", default=None)


@contextmanager
def local_search_budget(deadline, cancel_token=None):
    token = _LOCAL_SEARCH_BUDGET.set((deadline, cancel_token))
    try:
        yield
    finally:
        _LOCAL_SEARCH_BUDGET.reset(token)


def check_local_search_budget():
    budget = _LOCAL_SEARCH_BUDGET.get()
    if budget is None:
        return
    deadline, cancel_token = budget
    if cancel_token is not None:
        cancel_token.raise_if_requested()
    if deadline is not None and time.perf_counter() >= deadline:
        raise LocalSearchBudgetExpired("optional witness construction budget expired")


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._solvers = set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            solvers = list(self._solvers)
        for solver in solvers:
            try:
                solver.interrupt()
            except Exception:
                pass

    def requested(self) -> bool:
        return self._event.is_set()

    def raise_if_requested(self) -> None:
        if self.requested():
            raise SolveCancelledError("DSL solve canceled")

    @contextmanager
    def track_solver(self, solver) -> Iterator[None]:
        with self._lock:
            self._solvers.add(solver)
            should_interrupt = self.requested()
        if should_interrupt:
            solver.interrupt()
        try:
            yield
        finally:
            with self._lock:
                self._solvers.discard(solver)
