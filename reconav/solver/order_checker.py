"""Z3 consistency checks for milestone order constraints."""

from __future__ import annotations

from z3 import And, Bool, Distinct, Int, Not, Solver, sat

from reconav.solver.dsl_ast import DSLProgram, OrderConstraint


class OrderConsistencyError(ValueError):
    """Raised when milestone order constraints are internally inconsistent."""



def check_order_satisfiable(program: DSLProgram) -> None:
    """Validate order constraints with Z3 and raise an unsat-core error on conflicts."""
    milestones = program.milestone_by_name()
    _validate_references(program, milestones)

    names = list(milestones)
    hard_constraints = [
        constraint
        for constraint in program.order_constraints
        if all(name in milestones for name in (constraint.first, constraint.second, *constraint.names))
    ]
    n = len(names)
    pos = {name: Int(f"pos_{_z3_safe_name(name)}") for name in names}
    solver = Solver()

    for name in names:
        solver.add(pos[name] >= 0, pos[name] < n)
    solver.add(Distinct(*[pos[name] for name in names]))
    solver.add(pos["START"] == 0)
    solver.add(pos["GOAL"] == n - 1)

    tracked: dict[str, OrderConstraint] = {}
    for idx, constraint in enumerate(hard_constraints):
        expr = _constraint_expr(constraint, pos, names)
        label = f"order_{idx}_line_{constraint.source.number}"
        tracked[label] = constraint
        solver.assert_and_track(expr, Bool(label))

    if solver.check() == sat:
        return

    core = [str(x) for x in solver.unsat_core()]
    conflicts = [tracked[x] for x in core if x in tracked]
    if not conflicts:
        raise OrderConsistencyError("order constraints are inconsistent")

    messages = tuple(f"line {c.source.number}: {c.source.text}" for c in conflicts)
    raise OrderConsistencyError(
        "order constraints are inconsistent:\n" + "\n".join(f"  {m}" for m in messages)
    )


def _validate_references(program: DSLProgram, milestones: dict[str, object]) -> None:
    soft_names = set(program.soft_event_by_name())
    event_names = set(milestones) | soft_names
    if "START" not in milestones:
        raise ValueError("missing START statement")
    if "GOAL" not in milestones:
        raise ValueError("missing GOAL statement")
    for constraint in program.order_constraints:
        refs = [constraint.first, constraint.second, *constraint.names]
        for name in refs:
            if name not in event_names:
                raise ValueError(
                    f"line {constraint.source.number}: unknown milestone '{name}' in {constraint.source.text}"
                )
            if name.startswith("_via_"):
                raise ValueError(
                    f"line {constraint.source.number}: anonymous milestone '{name}' cannot be referenced"
                )


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
