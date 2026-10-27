"""Static semantic checks for COUNT constraints."""

from __future__ import annotations

from dataclasses import dataclass

from z3 import And, Distinct, Int, Not, Solver, sat

from reconav.scene.asg import ASG
from reconav.solver.dsl_ast import CountSpec, DSLProgram, OrderConstraint
from reconav.solver.target_grounding import ground_target


class CountConsistencyError(ValueError):
    """Raised when COUNT constraints are statically impossible."""


@dataclass(frozen=True)
class _CountError:
    spec: CountSpec
    message: str


def check_count_semantics(program: DSLProgram, asg: ASG) -> None:
    """Reject COUNT constraints whose interval or grounded candidate set is impossible."""
    if not program.count_specs:
        return

    errors: list[_CountError] = []
    errors.extend(_check_count_intervals(program))
    errors.extend(_check_grounded_counts(program, asg))
    if errors:
        lines = [f"  line {err.spec.source.number}: {err.spec.source.text}: {err.message}" for err in errors]
        raise CountConsistencyError("count constraints are inconsistent:\n" + "\n".join(lines))


def _check_count_intervals(program: DSLProgram) -> list[_CountError]:
    milestones = program.milestone_by_name()
    names = list(milestones)
    pos = {name: Int(f"count_pos_{_z3_safe_name(name)}") for name in names}
    solver = _base_solver(program, names, pos)
    errors: list[_CountError] = []

    for spec in program.count_specs:
        if spec.after not in milestones:
            errors.append(_CountError(spec, f"unknown AFTER milestone '{spec.after}'"))
            continue
        if spec.until not in milestones:
            errors.append(_CountError(spec, f"unknown UNTIL milestone '{spec.until}'"))
            continue
        if spec.after.startswith("_via_") or spec.until.startswith("_via_"):
            errors.append(_CountError(spec, "COUNT scope cannot reference anonymous VIA milestones"))
            continue
        if spec.after == spec.until:
            errors.append(_CountError(spec, "AFTER and UNTIL milestones must be different"))
            continue

        solver.push()
        solver.add(pos[spec.after] < pos[spec.until])
        if solver.check() != sat:
            errors.append(_CountError(spec, "AFTER milestone cannot occur before UNTIL under current order constraints"))
        solver.pop()
    return errors


def _check_grounded_counts(program: DSLProgram, asg: ASG) -> list[_CountError]:
    errors: list[_CountError] = []
    for spec in program.count_specs:
        grounded = ground_target(asg, spec.target, spec.source.number, start_target=program.original_start_target or program.start().target)
        candidate_count = len(grounded.ids)
        if spec.op in {"at_least", "exactly"} and spec.count > candidate_count:
            errors.append(
                _CountError(
                    spec,
                    f"requires {spec.count} {grounded.kind} target(s), but only {candidate_count} candidate(s) ground",
                )
            )
    return errors


def _base_solver(program: DSLProgram, names: list[str], pos: dict[str, object]) -> Solver:
    solver = Solver()
    n = len(names)
    for name in names:
        solver.add(pos[name] >= 0, pos[name] < n)
    solver.add(Distinct(*[pos[name] for name in names]))
    solver.add(pos["START"] == 0)
    solver.add(pos["GOAL"] == n - 1)
    for constraint in program.order_constraints:
        if not all(name in pos for name in (constraint.first, constraint.second, *constraint.names)):
            continue
        solver.add(_constraint_expr(constraint, pos, names))
    return solver


def _constraint_expr(constraint: OrderConstraint, pos: dict[str, object], all_names: list[str]):
    a = pos[constraint.first]
    b = pos[constraint.second]
    if constraint.kind == "before":
        return a < b
    if constraint.kind == "next":
        return b == a + 1
    if constraint.kind == "between_only":
        allowed = set(constraint.names)
        terms = [a < b]
        for name in allowed:
            terms.append(And(a < pos[name], pos[name] < b))
        for name in all_names:
            if name in {"START", "GOAL", constraint.first, constraint.second} or name in allowed:
                continue
            terms.append(Not(And(a < pos[name], pos[name] < b)))
        return And(terms)
    if constraint.kind == "between_without":
        forbidden = set(constraint.names)
        terms = [a < b]
        for name in forbidden:
            terms.append(Not(And(a < pos[name], pos[name] < b)))
        return And(terms)
    raise ValueError(f"unsupported order constraint kind: {constraint.kind}")


def _z3_safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
