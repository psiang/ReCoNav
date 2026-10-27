"""Public request and evidence vocabulary for benchmark route solving."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryMode(str, Enum):
    FEASIBLE = "FEASIBLE"
    PENALTY_AT_MOST = "PENALTY_AT_MOST"
    MIN_PENALTY = "MIN_PENALTY"
    LEX_OPTIMAL = "LEX_OPTIMAL"


@dataclass(frozen=True)
class WitnessRequirements:
    """Exact per-clause soft outcomes required of a returned witness.

    The triples are ``(kind, index_within_kind, outcome)``.  Outcomes are
    binary: zero means that the soft clause is satisfied and one means that
    it is violated.  These requirements constrain witness selection only;
    they never change the task's reported preference objective.
    """

    outcomes: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        allowed = {"avoid", "via", "count", "order"}
        seen: set[tuple[str, int]] = set()
        for kind, index, outcome in self.outcomes:
            if kind not in allowed:
                raise ValueError(f"unsupported witness requirement kind: {kind}")
            if index < 0:
                raise ValueError("witness requirement index must be nonnegative")
            if outcome not in {0, 1}:
                raise ValueError("witness requirement outcome must be zero or one")
            key = (kind, index)
            if key in seen:
                raise ValueError(f"duplicate witness requirement: {key}")
            seen.add(key)


@dataclass(frozen=True)
class SolveRequest:
    mode: QueryMode = QueryMode.LEX_OPTIMAL
    penalty_bound: int | None = None
    witness_requirements: WitnessRequirements | None = None

    def __post_init__(self) -> None:
        if self.mode is QueryMode.PENALTY_AT_MOST:
            if self.penalty_bound is None or self.penalty_bound < 0:
                raise ValueError("PENALTY_AT_MOST requires a nonnegative integer bound")
        elif self.penalty_bound is not None:
            raise ValueError(f"{self.mode.value} does not accept a penalty bound")
        if self.witness_requirements is not None and self.mode is not QueryMode.FEASIBLE:
            raise ValueError("witness requirements are supported only in FEASIBLE mode")


FEASIBLE = SolveRequest(QueryMode.FEASIBLE)
MIN_PENALTY = SolveRequest(QueryMode.MIN_PENALTY)
LEX_OPTIMAL = SolveRequest(QueryMode.LEX_OPTIMAL)
