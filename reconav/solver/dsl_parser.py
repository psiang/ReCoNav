"""Navigation DSL parser."""

from __future__ import annotations

import logging
import math
import re
import shlex

from reconav.scene.asg import ASG
from reconav.solver.avoid_checker import AvoidConsistencyError
from reconav.solver.cancellation import CancellationToken
from reconav.solver.count_checker import CountConsistencyError, check_count_semantics
from reconav.solver.cross_region import CrossRegionPathResult, cross_region_nav
from reconav.solver.dsl_ast import (
    AvoidSpec,
    CountSpec,
    DSLProgram,
    MilestoneSpec,
    OrderConstraint,
    SoftAvoidSpec,
    SoftCountSpec,
    SoftOrderSpec,
    SoftViaSpec,
    SourceLine,
    TargetQualifier,
    TargetSelector,
)
from reconav.solver.order_checker import check_order_satisfiable
from reconav.solver.query_evidence import LEX_OPTIMAL, SolveRequest

_log = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAMED_STMT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$")
_ORDER_RE = re.compile(r"^ORDER\s+([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", re.IGNORECASE)
_RESERVED_NAMES = {
    "START",
    "GOAL",
    "VIA",
    "OBJ",
    "REGION",
    "PORTAL",
    "LEVEL",
    "POS",
    "CATEGORY",
    "ORDER",
    "BEFORE",
    "AFTER",
    "NEXT",
    "BETWEEN",
    "ONLY",
    "WITHOUT",
    "AVOID",
    "COUNT",
    "AT",
    "LEAST",
    "MOST",
    "EXACTLY",
    "PREFER",
    "IN",
}


def _normalize_category(raw: str) -> str:
    return " ".join(raw.strip().lower().replace("_", " ").split())


def parse_dsl(dsl_text: str) -> DSLProgram:
    """Parse DSL text into a typed program."""
    program = DSLProgram()
    anonymous_count = 0

    for line_number, raw_line in enumerate(dsl_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        source = SourceLine(line_number, line)
        try:
            anonymous_count = _parse_line(
                line,
                source,
                program,
                anonymous_count,
            )
        except Exception as exc:
            raise ValueError(f"failed to parse line {line_number}: {line}\n{exc}") from exc

    _validate_milestones(program)
    _validate_avoid_specs(program)
    _validate_count_specs(program)
    _validate_soft_specs(program)
    _add_state_interval_order_constraints(program)
    _log.info(
        "DSL parsed: milestones=%s order_constraints=%s",
        [m.name for m in program.milestones],
        len(program.order_constraints),
    )
    return program


def _parse_line(
    line: str,
    source: SourceLine,
    program: DSLProgram,
    anonymous_count: int,
) -> int:
    if line.upper().startswith("PREFER "):
        _parse_prefer(_split_statement(line, source), source, program, None)
        return anonymous_count
    if line.upper().startswith("AVOID "):
        program.avoid_specs.append(_parse_avoid(_split_statement(line, source), source))
        return anonymous_count
    if line.upper().startswith("COUNT "):
        program.count_specs.append(_parse_count(_split_statement(line, source), source))
        return anonymous_count

    named = _NAMED_STMT_RE.match(line)
    if named:
        name, rest = named.group(1), named.group(2).strip()
        _validate_user_name(name, source)
        parts = _split_statement(rest, source)
        if len(parts) >= 2 and parts[0].upper() == "PREFER":
            _parse_prefer(parts, source, program, name)
            return anonymous_count
        if len(parts) >= 2 and parts[0].upper() == "VIA":
            target = _parse_target(parts[1:], source, allow_multi_id=True)
            if target.kind == "pos":
                raise ValueError("VIA POS is not supported")
            program.milestones.append(
                MilestoneSpec(name=name, role="via", target=target, source=source, anonymous=False)
            )
            return anonymous_count
        raise ValueError("named statements currently support '<name>: VIA ...' and '<name>: PREFER VIA ...'")

    order_match = _ORDER_RE.match(line)
    if order_match:
        program.order_constraints.append(_parse_order(order_match.group(1), order_match.group(2), source))
        return anonymous_count

    parts = _split_statement(line, source)
    cmd = parts[0].upper()
    if cmd == "START":
        target = _parse_target(parts[1:], source)
        program.milestones.append(MilestoneSpec("START", "start", target, source))
        return anonymous_count
    if cmd == "GOAL":
        target = _parse_target(parts[1:], source)
        if target.kind == "pos":
            raise ValueError("GOAL POS is not supported; use GOAL OBJ <id> or GOAL REGION <id>")
        if target.kind == "portal":
            raise ValueError("GOAL PORTAL is not supported; use VIA PORTAL plus a GOAL OBJ/REGION")
        program.milestones.append(MilestoneSpec("GOAL", "goal", target, source))
        return anonymous_count
    if cmd == "VIA":
        target = _parse_target(parts[1:], source, allow_multi_id=True)
        if target.kind == "pos":
            raise ValueError("VIA POS is not supported")
        name = f"_via_{anonymous_count}"
        program.milestones.append(MilestoneSpec(name, "via", target, source, anonymous=True))
        return anonymous_count + 1

    raise ValueError(f"unrecognized directive: {line}")


def _parse_prefer(parts: list[str], source: SourceLine, program: DSLProgram, name: str | None) -> None:
    if len(parts) < 2 or parts[0].upper() != "PREFER":
        raise ValueError("PREFER syntax requires AVOID, COUNT, VIA, or ORDER")
    kind = parts[1].upper()
    if kind == "AVOID":
        if name is not None:
            raise ValueError("PREFER AVOID cannot be named")
        program.soft_avoid_specs.append(_parse_prefer_avoid(parts, source))
        return
    if kind == "COUNT":
        if name is not None:
            raise ValueError("PREFER COUNT cannot be named")
        program.soft_count_specs.append(_parse_prefer_count(parts, source))
        return
    if kind == "VIA":
        program.soft_via_specs.append(_parse_prefer_via(parts, source, name))
        return
    if kind == "ORDER":
        if name is not None:
            raise ValueError("PREFER ORDER cannot be named")
        program.soft_order_specs.append(_parse_prefer_order(parts, source))
        return
    raise ValueError("PREFER supports only AVOID, COUNT, VIA, or ORDER")


def _parse_prefer_avoid(parts: list[str], source: SourceLine) -> SoftAvoidSpec:
    upper = [part.upper() for part in parts]
    try:
        after_pos = upper.index("AFTER")
    except ValueError as exc:
        raise ValueError(
            "PREFER AVOID requires AFTER <milestone> UNTIL <milestone>"
        ) from exc
    target = _parse_target(parts[2:after_pos], source, allow_multi_id=True)
    if target.kind not in {"obj", "region", "portal", "level"}:
        raise ValueError("PREFER AVOID supports only OBJ, REGION, PORTAL, or LEVEL targets")
    after, until = _parse_scoped_tail(parts, after_pos, source, "PREFER AVOID")
    return SoftAvoidSpec(target=target, after=after, until=until, source=source)


def _parse_prefer_via(parts: list[str], source: SourceLine, name: str | None) -> SoftViaSpec:
    target = _parse_target(parts[2:], source, allow_multi_id=True)
    if target.kind == "pos":
        raise ValueError("PREFER VIA POS is not supported")
    return SoftViaSpec(name=name, target=target, source=source)


def _parse_prefer_count(parts: list[str], source: SourceLine) -> SoftCountSpec:
    upper = [part.upper() for part in parts]
    try:
        after_pos = upper.index("AFTER")
    except ValueError as exc:
        raise ValueError(
            "PREFER COUNT requires AFTER <milestone> UNTIL <milestone>"
        ) from exc
    op, op_start, count_pos = _parse_count_op(parts, after_pos, source)
    if count_pos + 1 != after_pos:
        raise ValueError("PREFER COUNT quantity must appear immediately before AFTER")
    target = _parse_target(parts[2:op_start], source, allow_multi_id=True)
    if target.kind == "pos":
        raise ValueError("PREFER COUNT supports only OBJ, REGION, or PORTAL targets")
    count = _parse_nonnegative_int(parts[count_pos], "PREFER COUNT")
    after, until = _parse_scoped_tail(parts, after_pos, source, "PREFER COUNT")
    return SoftCountSpec(
        target=target,
        op=op,
        count=count,
        after=after,
        until=until,
        source=source,
    )


def _parse_prefer_order(parts: list[str], source: SourceLine) -> SoftOrderSpec:
    order_expr = " ".join(parts[2:])
    match = _ORDER_RE.match("ORDER " + order_expr)
    if not match:
        raise ValueError("PREFER ORDER requires BEFORE(...), AFTER(...), NEXT(...), or BETWEEN(...)")
    constraint = _parse_order(match.group(1), match.group(2), source)
    return SoftOrderSpec(constraint=constraint, source=source)


def _parse_scoped_tail(parts: list[str], pos: int, source: SourceLine, context: str) -> tuple[str, str]:
    if pos + 4 != len(parts) or parts[pos].upper() != "AFTER" or parts[pos + 2].upper() != "UNTIL":
        raise ValueError(f"{context} scope syntax is AFTER <milestone> UNTIL <milestone>")
    after = parts[pos + 1]
    until = parts[pos + 3]
    _validate_ref_name(after, source)
    _validate_ref_name(until, source)
    if after == until:
        raise ValueError(f"{context} after and until milestones must be different")
    return after, until


def _split_statement(text: str, source: SourceLine) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"line {source.number}: invalid quoting: {exc}") from exc


_QUALIFIER_STARTS = {"IN", "BETWEEN", "NEAR", "NEAREST", "FARTHEST"}


def _parse_target(parts: list[str], source: SourceLine, *, allow_multi_id: bool = False) -> TargetSelector:
    target, pos = _parse_target_prefix(parts, source, allow_pos=True, allow_multi_id=allow_multi_id)
    if target.kind == "pos":
        if pos != len(parts):
            raise ValueError("POS target does not support qualifiers")
        return target
    qualifiers, pos = _parse_qualifiers(
        parts,
        pos,
        source,
        target_kind=target.kind,
    )
    if pos != len(parts):
        raise ValueError(f"unexpected token in target: {parts[pos]}")
    return TargetSelector(target.kind, target.selector, target.values, tuple(qualifiers))


def _parse_target_prefix(
    parts: list[str],
    source: SourceLine,
    *,
    allow_pos: bool,
    allow_multi_id: bool,
) -> tuple[TargetSelector, int]:
    if not parts:
        raise ValueError("target requires OBJ, REGION, PORTAL, LEVEL, or POS")
    kind_raw = parts[0].upper()
    if len(parts) < 2:
        raise ValueError("target requires OBJ, REGION, PORTAL, LEVEL, or POS plus arguments")
    if kind_raw == "POS":
        if not allow_pos:
            raise ValueError("POS is not supported in target qualifiers")
        if len(parts) < 4:
            raise ValueError("POS target syntax is POS <x> <floor_height> <z>")
        try:
            x = float(parts[1])
            floor_height = float(parts[2])
            z = float(parts[3])
        except ValueError as exc:
            raise ValueError("POS target requires numeric x, floor_height, and z") from exc
        if not math.isfinite(x) or not math.isfinite(floor_height) or not math.isfinite(z):
            raise ValueError("POS coordinates must be finite")
        return TargetSelector(kind="pos", selector="coord", values=(x, floor_height, z)), 4

    kind_map = {"OBJ": "obj", "REGION": "region", "PORTAL": "portal", "LEVEL": "level"}
    if kind_raw not in kind_map:
        raise ValueError(f"unsupported target kind: {parts[0]}")
    kind = kind_map[kind_raw]
    if kind == "portal" and parts[1].upper() in _QUALIFIER_STARTS:
        return TargetSelector(kind=kind, selector="any", values=()), 1

    if parts[1].upper() == "CATEGORY":
        if len(parts) < 3:
            raise ValueError(f"{parts[0]} CATEGORY requires a category label")
        category = _normalize_category(parts[2])
        if not category:
            raise ValueError(f"{parts[0]} CATEGORY requires a non-empty category label")
        return TargetSelector(kind=kind, selector="category", values=(category,)), 3

    pos = 1
    values: list[int] = []
    while pos < len(parts):
        if parts[pos].upper() in _QUALIFIER_STARTS or parts[pos].upper() in {"AFTER", "UNTIL", "AND"}:
            break
        try:
            values.append(int(parts[pos]))
        except ValueError as exc:
            raise ValueError(f"{parts[0]} id target requires integer indices, got '{parts[pos]}'") from exc
        pos += 1
        if not allow_multi_id:
            break
    if not values:
        raise ValueError(f"{parts[0]} id target requires at least one index")
    if not allow_multi_id and pos < len(parts) and parts[pos].upper() not in _QUALIFIER_STARTS:
        raise ValueError(f"{parts[0]} id target requires exactly one index")
    return TargetSelector(kind=kind, selector="id", values=tuple(values)), pos


def _parse_qualifiers(
    parts: list[str],
    pos: int,
    source: SourceLine,
    *,
    target_kind: str,
) -> tuple[list[TargetQualifier], int]:
    qualifiers: list[TargetQualifier] = []
    while pos < len(parts):
        tok = parts[pos].upper()
        if tok == "IN":
            if pos + 1 >= len(parts) or parts[pos + 1].upper() not in {"REGION", "LEVEL"}:
                raise ValueError("IN qualifier syntax is IN REGION|LEVEL <ids...|CATEGORY label>")
            target, consumed = _parse_target_prefix(parts[pos + 1:], source, allow_pos=False, allow_multi_id=True)
            pos += 1 + consumed
            if target.kind not in {"region", "level"}:
                raise ValueError("IN qualifier supports only REGION or LEVEL targets")
            qualifiers.append(TargetQualifier(f"in_{target.kind}", target))
            continue
        if tok == "BETWEEN":
            first, consumed = _parse_target_prefix(parts[pos + 1 :], source, allow_pos=False, allow_multi_id=True)
            pos += 1 + consumed
            if first.kind != "region":
                raise ValueError("BETWEEN qualifier supports only REGION targets")
            if pos >= len(parts) or parts[pos].upper() != "AND":
                raise ValueError("BETWEEN qualifier syntax is BETWEEN REGION <...> AND REGION <...>")
            second, consumed = _parse_target_prefix(parts[pos + 1 :], source, allow_pos=False, allow_multi_id=True)
            pos += 1 + consumed
            if second.kind != "region":
                raise ValueError("BETWEEN qualifier supports only REGION targets")
            qualifiers.append(TargetQualifier("between_regions", first, second_target=second))
            continue
        if tok in {"NEAR", "NEAREST", "FARTHEST"}:
            if target_kind != "obj":
                raise ValueError(
                    f"{tok} qualifier requires an OBJ target"
                )
            kind = {
                "NEAR": "near",
                "NEAREST": "nearest_to",
                "FARTHEST": "farthest_from",
            }[tok]
            pos += 1
            if tok == "NEAREST":
                if pos >= len(parts) or parts[pos].upper() != "TO":
                    raise ValueError("NEAREST qualifier syntax is NEAREST TO <target>")
                pos += 1
            elif tok == "FARTHEST":
                if pos >= len(parts) or parts[pos].upper() != "FROM":
                    raise ValueError("FARTHEST qualifier syntax is FARTHEST FROM <target>")
                pos += 1
            if pos < len(parts) and parts[pos].upper() == "START":
                if kind == "near":
                    raise ValueError("NEAR START is not supported; use a unique OBJ or NEAR EXISTS OBJ")
                qualifiers.append(TargetQualifier(
                    kind, TargetSelector("pos", "coord", ()), reference_mode="start"
                ))
                pos += 1
                continue
            reference_mode = "unique"
            if pos < len(parts) and parts[pos].upper() == "EXISTS":
                if kind != "near":
                    raise ValueError("EXISTS reference is supported only for NEAR")
                reference_mode = "exists"
                pos += 1
            reference, consumed = _parse_target_prefix(
                parts[pos:], source, allow_pos=False, allow_multi_id=reference_mode == "exists",
            )
            if reference.kind != "obj":
                raise ValueError(f"{tok} qualifier reference must be OBJ or original START")
            pos += consumed
            qualifiers.append(TargetQualifier(kind, reference, reference_mode=reference_mode))
            continue
        raise ValueError(f"unexpected qualifier token: {parts[pos]}")
    return qualifiers, pos


def _parse_avoid(parts: list[str], source: SourceLine) -> AvoidSpec:
    if len(parts) < 7 or parts[0].upper() != "AVOID":
        raise ValueError("AVOID requires target plus AFTER <milestone> UNTIL <milestone>")
    upper = [p.upper() for p in parts]
    try:
        after_pos = upper.index("AFTER")
        until_pos = upper.index("UNTIL")
    except ValueError as exc:
        raise ValueError("AVOID requires AFTER <milestone> UNTIL <milestone>") from exc
    if after_pos >= until_pos:
        raise ValueError("AVOID requires AFTER before UNTIL")
    if after_pos < 3:
        raise ValueError("AVOID target is missing")
    if until_pos != after_pos + 2 or until_pos + 2 != len(parts):
        raise ValueError("AVOID scope syntax is AFTER <milestone> UNTIL <milestone>")

    after = parts[after_pos + 1]
    until = parts[until_pos + 1]
    _validate_ref_name(after, source)
    _validate_ref_name(until, source)
    if after == until:
        raise ValueError("AVOID after and until milestones must be different")

    target = _parse_target(parts[1:after_pos], source, allow_multi_id=True)
    if target.kind not in {"obj", "region", "portal", "level"}:
        raise ValueError("AVOID supports only OBJ, REGION, PORTAL, or LEVEL targets")
    return AvoidSpec(
        kind=target.kind,
        selector=target.selector,
        values=target.values,
        qualifiers=target.qualifiers,
        after=after,
        until=until,
        source=source,
    )


def _parse_count(parts: list[str], source: SourceLine) -> CountSpec:
    if len(parts) < 8 or parts[0].upper() != "COUNT":
        raise ValueError("COUNT syntax is COUNT <target> AT LEAST|AT MOST|EXACTLY <n> AFTER <milestone> UNTIL <milestone>")
    upper = [p.upper() for p in parts]
    try:
        after_pos = upper.index("AFTER")
        until_pos = upper.index("UNTIL")
    except ValueError as exc:
        raise ValueError("COUNT requires AFTER <milestone> UNTIL <milestone>") from exc
    if after_pos >= until_pos:
        raise ValueError("COUNT requires AFTER before UNTIL")
    if until_pos != after_pos + 2 or until_pos + 2 != len(parts):
        raise ValueError("COUNT scope syntax is AFTER <milestone> UNTIL <milestone>")

    op, op_start, count_pos = _parse_count_op(parts, after_pos, source)
    if count_pos + 1 != after_pos:
        raise ValueError("COUNT quantity must appear immediately before AFTER")
    count = _parse_nonnegative_int(parts[count_pos], "COUNT")

    target = _parse_target(parts[1:op_start], source, allow_multi_id=True)
    if target.kind == "pos":
        raise ValueError("COUNT supports only OBJ, REGION, PORTAL, or LEVEL targets")

    after = parts[after_pos + 1]
    until = parts[until_pos + 1]
    _validate_ref_name(after, source)
    _validate_ref_name(until, source)
    if after == until:
        raise ValueError("COUNT after and until milestones must be different")
    return CountSpec(target=target, op=op, count=count, after=after, until=until, source=source)


def _parse_count_op(parts: list[str], after_pos: int, source: SourceLine) -> tuple[str, int, int]:
    if after_pos >= 2 and parts[after_pos - 2].upper() == "EXACTLY":
        return "exactly", after_pos - 2, after_pos - 1
    if after_pos >= 3 and parts[after_pos - 3].upper() == "AT":
        mode = parts[after_pos - 2].upper()
        if mode == "LEAST":
            return "at_least", after_pos - 3, after_pos - 1
        if mode == "MOST":
            return "at_most", after_pos - 3, after_pos - 1
    raise ValueError(f"line {source.number}: COUNT requires AT LEAST <n>, AT MOST <n>, or EXACTLY <n>")


def _parse_nonnegative_int(raw: str, context: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{context} requires a non-negative integer, got '{raw}'") from exc
    if value < 0:
        raise ValueError(f"{context} requires a non-negative integer")
    return value


def _parse_order(op_raw: str, body: str, source: SourceLine) -> OrderConstraint:
    op = op_raw.upper()
    args = _split_args(body)
    if op == "BEFORE":
        _expect_arg_count(op, args, 2)
        return OrderConstraint("before", args[0], args[1], (), source)
    if op == "AFTER":
        _expect_arg_count(op, args, 2)
        return OrderConstraint("before", args[1], args[0], (), source)
    if op == "NEXT":
        _expect_arg_count(op, args, 2)
        return OrderConstraint("next", args[0], args[1], (), source)
    if op == "BETWEEN":
        if len(args) != 3:
            raise ValueError(f"ORDER BETWEEN expects 3 arguments, got {len(args)}")
        _validate_ref_name(args[0], source)
        _validate_ref_name(args[1], source)
        mode, names = _parse_between_mode(args[2], source)
        return OrderConstraint(mode, args[0], args[1], tuple(names), source)
    raise ValueError(f"unsupported ORDER operator: {op_raw}")


def _parse_between_mode(raw: str, source: SourceLine) -> tuple[str, list[str]]:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", raw.strip())
    if not match:
        raise ValueError("BETWEEN third argument must be ONLY(...) or WITHOUT(...)")
    mode = match.group(1).upper()
    names = _split_args(match.group(2))
    if not names:
        raise ValueError(f"{mode} requires at least one milestone name")
    for name in names:
        _validate_ref_name(name, source)
    if mode == "ONLY":
        return "between_only", names
    if mode == "WITHOUT":
        return "between_without", names
    raise ValueError("BETWEEN third argument must be ONLY(...) or WITHOUT(...)")


def _split_args(raw: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced ')' in ORDER expression")
        elif ch == "," and depth == 0:
            args.append(raw[start:i].strip())
            start = i + 1
    if depth != 0:
        raise ValueError("unbalanced '(' in ORDER expression")
    tail = raw[start:].strip()
    if tail:
        args.append(tail)
    if any(not arg for arg in args):
        raise ValueError("empty argument in ORDER expression")
    for arg in args:
        if "(" not in arg:
            _validate_ref_name(arg, SourceLine(0, raw))
    return args


def _expect_arg_count(op: str, args: list[str], expected: int) -> None:
    if len(args) != expected:
        raise ValueError(f"ORDER {op} expects {expected} arguments, got {len(args)}")
    for arg in args:
        _validate_ref_name(arg, SourceLine(0, op))


def _validate_milestones(program: DSLProgram) -> None:
    names: set[str] = set()
    starts = 0
    goals = 0
    for milestone in program.milestones:
        if milestone.name in names:
            raise ValueError(f"line {milestone.source.number}: duplicate milestone name '{milestone.name}'")
        names.add(milestone.name)
        starts += int(milestone.role == "start")
        goals += int(milestone.role == "goal")
    if starts != 1:
        raise ValueError(f"expected exactly one START statement, got {starts}")
    if goals != 1:
        raise ValueError(f"expected exactly one GOAL statement, got {goals}")


def _validate_avoid_specs(program: DSLProgram) -> None:
    for spec in program.avoid_specs:
        _validate_state_scope(program, spec.after, spec.until, spec.source, "AVOID")


def _validate_count_specs(program: DSLProgram) -> None:
    for spec in program.count_specs:
        _validate_state_scope(program, spec.after, spec.until, spec.source, "COUNT")


def _validate_soft_specs(program: DSLProgram) -> None:
    hard_names = set(program.milestone_by_name())
    soft_names = program.soft_event_by_name()
    for name in soft_names:
        if name in hard_names:
            raise ValueError(f"soft event name '{name}' conflicts with a hard milestone")
    for spec in [*program.soft_avoid_specs, *program.soft_count_specs]:
        _validate_state_scope(program, spec.after, spec.until, spec.source, "PREFER")
    event_names = hard_names | set(soft_names)
    for spec in program.soft_order_specs:
        _validate_order_refs(spec.constraint, event_names, spec.source)
    for constraint in program.order_constraints:
        _validate_order_refs(constraint, event_names, constraint.source)


def _validate_order_refs(
    constraint: OrderConstraint, event_names: set[str], source: SourceLine
) -> None:
    for name in (constraint.first, constraint.second, *constraint.names):
        if name not in event_names:
            raise ValueError(f"line {source.number}: unknown milestone or soft event '{name}' in {source.text}")
        if name.startswith("_via_"):
            raise ValueError(f"line {source.number}: anonymous milestone '{name}' cannot be referenced")


def _validate_state_scope(
    program: DSLProgram, after: str, until: str, source: SourceLine, directive: str
) -> None:
    milestones = program.milestone_by_name()
    for name in (after, until):
        if name not in milestones:
            raise ValueError(f"line {source.number}: unknown milestone '{name}' in {source.text}")
        if name.startswith("_via_"):
            raise ValueError(f"line {source.number}: anonymous milestone '{name}' cannot scope {directive}")


def _add_state_interval_order_constraints(program: DSLProgram) -> None:
    existing = {(c.kind, c.first, c.second, c.names) for c in program.order_constraints}
    for spec in [*program.avoid_specs, *program.count_specs]:
        key = ("before", spec.after, spec.until, ())
        if key not in existing:
            program.order_constraints.append(OrderConstraint("before", spec.after, spec.until, (), spec.source))
            existing.add(key)


def _validate_user_name(name: str, source: SourceLine) -> None:
    _validate_ref_name(name, source)
    if name.upper() in _RESERVED_NAMES:
        raise ValueError(f"line {source.number}: reserved milestone name '{name}'")
    if name.startswith("_via_"):
        raise ValueError(f"line {source.number}: names starting with '_via_' are reserved for anonymous VIA")


def _validate_ref_name(name: str, source: SourceLine) -> None:
    if not _NAME_RE.match(name):
        prefix = f"line {source.number}: " if source.number else ""
        raise ValueError(f"{prefix}invalid milestone name '{name}'")


def solve_program(
    program: DSLProgram,
    asg: ASG,
    cancel_token: CancellationToken | None = None,
    request: SolveRequest = LEX_OPTIMAL,
) -> CrossRegionPathResult:
    """Run solver-native checks and execute an already parsed program."""
    check_order_satisfiable(program)
    try:
        # Required endpoint membership alone does not imply an entry event:
        # START may already be in the zone. The event product handles AVOID.
        check_count_semantics(program, asg)
    except (AvoidConsistencyError, CountConsistencyError) as error:
        return CrossRegionPathResult(
            [],
            [],
            [],
            status="UNSAT",
            status_reason=f"static task certificate: {error}",
            physical_domain_complete=True,
            query_graph_complete=True,
            state_search_complete=True,
            optimality_established=False,
            proof_domain="grounded-static-task-semantics",
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="HARD_UNSAT",
            certificate_kind="static-task-semantics",
        )
    from reconav.solver.cancellation import SolveCancelledError

    try:
        result = cross_region_nav(
            asg,
            program,
            cancel_token=cancel_token,
            request=request,
        )
        if result.status == "SAT":
            from dataclasses import replace
            from reconav.solver.replay import replay_solution
            replay = replay_solution(program, asg, result)
            if not replay.get("ok") or not replay.get("penalty_matches_solver"):
                return replace(result, status="UNKNOWN", status_reason="independent semantic replay disagrees with query search", optimality_established=False, state_search_complete=False, semantic_replay=replay)
            result = replace(result, semantic_replay=replay)
        return result
    except SolveCancelledError:
        return CrossRegionPathResult(
            [],
            [],
            [],
            status="UNKNOWN",
            status_reason="query budget/cancellation interrupted graph or state expansion",
            physical_domain_complete=True,
            query_graph_complete=False,
            state_search_complete=False,
            optimality_established=False,
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="CANCELLED",
            cancelled=True,
            incomplete=True,
        )
