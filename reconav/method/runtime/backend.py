"""Common bounded solver for all methods; replay only the predicted contract."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import threading
import time
from collections.abc import Callable

from reconav.solver.cancellation import CancellationToken
from reconav.solver.dsl_parser import parse_dsl
from reconav.solver.execution_profiles import ExecutionProfile, program_features
from reconav.solver.query_evidence import FEASIBLE, MIN_PENALTY, QueryMode, SolveRequest
from reconav.solver.backend import solve_dsl_route


class QueryBudgetExhausted(TimeoutError):
    """No unresolved query may turn a time budget into a negative fact."""


def _empty_soft_count_offset(grounded: dict) -> float:
    """Lowering omits empty sets, whose soft COUNT cost is a constant.

    Restoring that known constant preserves the argmin of the emitted DSL.
    Nothing else is repaired or inferred from the route's eventual score.
    """
    offset = 0.0
    for item in grounded.get("prefer", []):
        if item.get("preference") != "count" or item.get("grounded_ids"):
            continue
        op = str(item["op"]).lower().replace(" ", "_")
        count = int(item["count"])
        if op not in {"at_least", "at_most", "exactly"}:
            raise ValueError(f"unknown empty COUNT operator {op}")
        satisfied = (0 >= count if op == "at_least" else
                     0 <= count if op == "at_most" else count == 0)
        if not satisfied:
            offset += float(item.get("weight", 1))
    return offset


def _predicted_replay(dsl, asg, route, grounded, robot_position):
    from reconav.benchmark.route_evaluator import evaluate_grounded_route
    from reconav.solver.dsl_parser import parse_dsl

    if route.get("status") != "SAT" or not route.get("waypoints"):
        return route
    if robot_position is None:
        start = parse_dsl(dsl).start().target
        if start.kind != "pos":
            raise ValueError("predicted-contract replay requires the actual START")
        x, height, z = (float(v) for v in start.values)
        robot_position = {"x": x, "floor_height": height, "y": z}
    result = deepcopy(route)
    offset = _empty_soft_count_offset(grounded)
    reported = result.get("optimal_soft_cost")
    if isinstance(reported, (int, float)) and math.isfinite(reported):
        result["optimal_soft_cost"] = float(reported) + offset
    result["lowering_constant_soft_cost"] = offset
    replay = evaluate_grounded_route(asg, grounded, result, dict(robot_position))
    result["predicted_contract_replay"] = replay
    hard_errors = [item for item in replay.get("constraints", [])
                   if not item.get("satisfied") and item.get("component")
                   not in {"prefer", "prefer_order", "preference_optimality"}]
    actual = replay.get("preference_cost")
    corrected = result.get("optimal_soft_cost")
    mismatch = (not isinstance(actual, (int, float)) or not math.isfinite(actual)
                or not isinstance(corrected, (int, float)) or not math.isfinite(corrected)
                or abs(actual - corrected) > 1e-9)
    if hard_errors or mismatch:
        result.update(status="UNKNOWN", status_reason="predicted-contract lowering/replay mismatch",
                      optimality_established=False, state_search_complete=False,
                      replay_rejected_waypoints=result.get("waypoints", []), waypoints=[])
    return result


class BoundedBackend:
    def __init__(self, *, per_query_seconds=90.0, total_seconds=360.0, solve: Callable | None = None):
        self.per_query_seconds = float(per_query_seconds)
        self.total_seconds = float(total_seconds)
        self.solve = solve or solve_dsl_route
        self.cache: dict[tuple, dict] = {}
        self.queries: list[dict] = []
        self.elapsed = 0.0

    def fork_remaining(self):
        """Independent stage snapshot, retaining evidence and only unused budget."""
        branch = BoundedBackend(per_query_seconds=self.per_query_seconds,
                                total_seconds=max(0.0, self.total_seconds - self.elapsed),
                                solve=self.solve)
        branch.cache = deepcopy(self.cache)
        return branch

    def __call__(self, dsl, asg, *, profile=ExecutionProfile.FULL_HARD_SOFT,
                 grounded_contract=None, robot_position=None,
                 request: SolveRequest | None = None):
        profile = ExecutionProfile(profile)
        if request is None:
            features = program_features(parse_dsl(dsl))
            request = (
                MIN_PENALTY
                if profile is ExecutionProfile.FULL_HARD_SOFT
                and any(item in {"prefer", "prefer_order"} for item in features)
                else FEASIBLE
            )
        context = json.dumps([grounded_contract, robot_position], sort_keys=True, separators=(",", ":"))
        request_key = (
            request.mode.value,
            request.penalty_bound,
            None
            if request.witness_requirements is None
            else request.witness_requirements.outcomes,
        )
        key = (id(asg), dsl, profile.value, context, request_key)
        if key in self.cache:
            return deepcopy(self.cache[key])
        remaining = self.total_seconds - self.elapsed
        if remaining <= 0:
            raise QueryBudgetExhausted("total candidate/solver budget exhausted")
        token = CancellationToken()
        timer = threading.Timer(min(remaining, self.per_query_seconds), token.cancel)
        timer.daemon = True
        record = {
            "dsl": dsl,
            "profile": profile.value,
            "query_mode": request.mode.value,
            "query_bound": request.penalty_bound,
            "budget_seconds": min(remaining, self.per_query_seconds),
        }
        self.queries.append(record)
        started = time.perf_counter()
        timer.start()
        try:
            route = self.solve(
                dsl,
                asg,
                profile=profile,
                cancel_token=token,
                request=request,
            )
            if grounded_contract is not None:
                route = _predicted_replay(dsl, asg, route, grounded_contract, robot_position)
            record.update(
                status=route.get("status"),
                status_reason=route.get("status_reason"),
                query_outcome=route.get("query_outcome"),
                penalty_proven=route.get("penalty_proven"),
                optimality_established=route.get("optimality_established"),
                proof_domain=route.get("proof_domain"),
                predicted_contract_replayed=grounded_contract is not None,
            )
            status = route.get("status")
            hard_unsat = (
                status == "UNSAT" and route.get("query_outcome") == "HARD_UNSAT"
            )
            sat_evidence_complete = (
                status == "SAT"
                and (
                    request.mode is QueryMode.FEASIBLE
                    or (
                        request.mode is QueryMode.MIN_PENALTY
                        and route.get("penalty_proven") is True
                    )
                    or (
                        request.mode is QueryMode.LEX_OPTIMAL
                        and route.get("optimality_established") is True
                    )
                    or request.mode is QueryMode.PENALTY_AT_MOST
                )
            )
            if hard_unsat or sat_evidence_complete:
                self.cache[key] = deepcopy(route)
            return route
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            timer.cancel()
            record["seconds"] = time.perf_counter() - started
            self.elapsed += record["seconds"]
