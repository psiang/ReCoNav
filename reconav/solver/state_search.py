"""Complete finite visibility-graph x route-event-state search."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
import heapq
import itertools
import math
import os
import time

from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from reconav.scene.geometry import Point3
from reconav.scene.physical import within_semantic_clearance
from reconav.solver.avoid_runtime import AvoidRuntime
from reconav.solver.cancellation import SolveCancelledError
from reconav.solver.count_runtime import CountRuntime, CountState
from reconav.solver.order_runtime import OrderRuntime
from reconav.solver.prefer_runtime import PreferRuntime
from reconav.solver.qualifier_geometry import object_geometry, UnknownRelationGeometry
from reconav.solver.query_evidence import LEX_OPTIMAL, QueryMode, SolveRequest
from reconav.solver.query_graph import AttachedQuery, get_complete_query_graph, QUERY_GRAPH_VERSION
from reconav.solver.route_events import object_encounter_entries
from reconav.solver.segment_events import EVENT_POSITION_EPS, EVENT_SEMANTICS_VERSION
from reconav.solver.target_grounding import ground_target
from reconav.scene.navigation import get_geometry_graph


_TOTAL_QUERY_CACHE_BYTES = int(
    os.environ.get("BENCHMARK_QUERY_CACHE_BYTES", str(256 * 1024 * 1024))
)
_EVENT_CACHE_BUDGET_BYTES = max(1, _TOTAL_QUERY_CACHE_BYTES // 2)
_RAW_EVENT_CACHE = OrderedDict()
_RAW_EVENT_CACHE_BYTES = 0
_RAW_EVENT_CACHE_EVICTIONS = 0


@dataclass(frozen=True)
class ProductState:
    node: int
    progress: tuple[str, ...]
    counts: CountState
    preference: object


@dataclass(frozen=True)
class StepWitness:
    source: int
    target: int
    portal: int | None
    names: tuple[tuple[str, str, int], ...]
    fractions: tuple[float, ...]
    observations: tuple[tuple[int, str, int, float, int, int], ...]


def _active(spec, progress, targets=()):
    return (
        spec.after in progress
        and spec.until not in progress
        and spec.after not in targets
        and spec.until not in targets
    )


def _target_components(physical, asg, target) -> frozenset[int]:
    if target.kind == "pos":
        x, height, z = (float(value) for value in target.values)
        candidates = []
        for region_id in physical.regions:
            local = physical.locate_local_component(region_id, Point(x, z))
            if local is not None:
                candidates.append(
                    (
                        abs(physical.floor_height(region_id) - height),
                        physical.region_to_global_component[(region_id, local)],
                    )
                )
        if not candidates:
            return frozenset()
        best = min(value[0] for value in candidates)
        return frozenset(component for delta, component in candidates if delta <= best + 1e-6)
    grounded = ground_target(asg, target, 0, start_target=target)
    return _endpoint_global_components(physical, asg, grounded)


def _endpoint_global_components(physical, asg, grounded) -> frozenset[int]:
    """Use the complete semantic endpoint domain for disconnection proofs.

    Warm object anchors are a search aid, and can omit a disconnected component
    that still intersects the object's legal approach zone. Missing geometry is
    inconclusive here; the ordinary grounding/search path retains its UNKNOWN.
    """
    from reconav.solver.topology_relaxation import _grounded_components

    try:
        components = _grounded_components(physical, asg, grounded)
    except UnknownRelationGeometry:
        return frozenset()
    return frozenset(physical.region_to_global_component[item] for item in components)


def _internal_via_avoid_certificate(program, asg, grounded, avoid, physical):
    start = program.start().target
    if start.kind != "pos":
        return None
    x, _height, z = (float(value) for value in start.values)
    start_point = Point(x, z)
    for item in avoid.avoids:
        if item.spec.after != "START" or item.spec.until != "GOAL" or not item.object_ids:
            continue
        outside = True
        for object_id in item.object_ids:
            try:
                shape, _level, _region = object_geometry(asg, object_id)
            except UnknownRelationGeometry:
                outside = False
                break
            if within_semantic_clearance(start_point.distance(shape), physical.config.endpoint_clearance_m):
                outside = False
                break
        if not outside:
            continue
        for milestone in program.milestones:
            if milestone.role != "via":
                continue
            target = grounded[milestone.name]
            if target.kind == "obj" and target.ids and target.ids <= item.object_ids:
                return {
                    "after": item.spec.after,
                    "until": item.spec.until,
                    "via": milestone.name,
                    "via_candidate_ids": sorted(target.ids),
                    "avoid_object_ids": sorted(item.object_ids),
                    "start_strictly_outside": True,
                    "occurrence_semantics": EVENT_SEMANTICS_VERSION,
                }
    return None


def _fast_witness_satisfies_requirements(
    result,
    program,
    requirements: tuple[tuple[str, int, int], ...],
) -> bool:
    """Check a replayed sparse witness against the same per-clause contract."""

    replay = result.semantic_replay
    if not isinstance(replay, dict) or not replay.get("ok"):
        return False
    records = {
        str(item.get("name")): bool(item.get("satisfied"))
        for item in replay.get("constraints", ())
        if isinstance(item, dict)
    }
    offsets = {
        "via": 0,
        "avoid": len(program.soft_via_specs),
        "count": len(program.soft_via_specs) + len(program.soft_avoid_specs),
    }
    lengths = {
        "via": len(program.soft_via_specs),
        "avoid": len(program.soft_avoid_specs),
        "count": len(program.soft_count_specs),
        "order": len(program.soft_order_specs),
    }
    for kind, index, expected in requirements:
        if index < 0 or index >= lengths.get(kind, 0):
            return False
        name = (
            f"prefer_order[{index}]"
            if kind == "order"
            else f"prefer[{offsets[kind] + index}]"
        )
        satisfied = records.get(name)
        if satisfied is None or int(not satisfied) != expected:
            return False
    return True


def solve_event_product(
    asg,
    program,
    cancel_token=None,
    *,
    request: SolveRequest = LEX_OPTIMAL,
):
    from reconav.solver.cross_region import CrossRegionPathResult

    started = time.perf_counter()
    if cancel_token is not None:
        cancel_token.raise_if_requested()
    original_start = program.original_start_target or program.start().target
    program.original_start_target = original_start
    physical = get_geometry_graph(asg)
    order = OrderRuntime(program, asg)
    avoid = AvoidRuntime(program, asg)
    count = CountRuntime(program, asg)
    prefer = PreferRuntime(program, asg)
    grounded = {
        item.name: ground_target(
            asg,
            item.target,
            item.source.number,
            start_target=original_start,
        )
        for item in program.milestones
        if item.name != "START"
    }
    goal = grounded["GOAL"]
    if len(goal.ids) != 1:
        raise ValueError(f"GOAL must be resolved before route search: {sorted(goal.ids)}")
    goal_id = next(iter(goal.ids))
    start_components = _target_components(physical, asg, original_start)
    goal_components = _endpoint_global_components(physical, asg, goal)
    topology_payload = {
        "graph_id": physical.graph_id,
        "start_components": sorted(start_components),
        "goal_components": sorted(goal_components),
    }
    if start_components and goal_components and not start_components.intersection(goal_components):
        return CrossRegionPathResult(
            [],
            [],
            [],
            status="UNSAT",
            status_reason="physical component/portal topology separates endpoints",
            query_graph_complete=False,
            state_search_complete=True,
            proof_domain=f"G0={physical.graph_id};component-portal-connectivity",
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="HARD_UNSAT",
            certificate_kind="component-portal-disconnection",
            certificate_payload=topology_payload,
            timing={"total_seconds": time.perf_counter() - started},
            counters={"full_query_graph_created": 0, "expanded_states": 0},
        )
    conflict = _internal_via_avoid_certificate(
        program,
        asg,
        grounded,
        avoid,
        physical,
    )
    if conflict is not None:
        return CrossRegionPathResult(
            [],
            [],
            [],
            status="UNSAT",
            status_reason="required internal VIA necessarily enters active AVOID zone",
            query_graph_complete=False,
            state_search_complete=True,
            proof_domain=f"G0={physical.graph_id};grounded-occurrence-conflict",
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="HARD_UNSAT",
            certificate_kind="internal-via-avoid-conflict",
            certificate_payload=conflict,
            timing={"total_seconds": time.perf_counter() - started},
            counters={"full_query_graph_created": 0, "expanded_states": 0},
        )

    # Exhaustion of an optimistic component/event product is a hard UNSAT
    # certificate when even the maximum possible soft penalty is allowed.
    # This is independent of failure to construct a sparse positive witness.
    if program.avoid_specs or program.count_specs:
        from reconav.solver.topology_relaxation import prove_penalty_bound
        maximum_soft_penalty = sum(len(specs) for specs in (
            program.soft_via_specs, program.soft_avoid_specs,
            program.soft_count_specs, program.soft_order_specs,
        ))
        hard_topology = prove_penalty_bound(
            asg, program, physical, maximum_soft_penalty,
            deadline=time.perf_counter() + 0.5, cancel_token=cancel_token,
        )
        if hard_topology.outcome == "BOUND_EXCLUDED":
            return CrossRegionPathResult(
                [], [], [], status="UNSAT",
                status_reason="complete component/event relaxation excludes every hard-feasible route",
                query_graph_complete=False, state_search_complete=True,
                proof_domain=f"G0={physical.graph_id};component-portal-event-relaxation",
                requested_mode=request.mode.value, requested_bound=request.penalty_bound,
                query_outcome="HARD_UNSAT", certificate_kind="component-event-hard-exclusion",
                certificate_payload={
                    **hard_topology.certificate,
                    "maximum_possible_soft_penalty": maximum_soft_penalty,
                },
                timing={"total_seconds": time.perf_counter() - started},
                counters={"full_query_graph_created": 0, "topology_states": hard_topology.states_visited},
            )

    fast = None
    cached_topology_bound = None
    cached_bound = None
    if request.mode in {
        QueryMode.FEASIBLE,
        QueryMode.PENALTY_AT_MOST,
        QueryMode.MIN_PENALTY,
        QueryMode.LEX_OPTIMAL,
    }:
        from reconav.solver.fast_witness import fast_complete_hard_witness

        fast = fast_complete_hard_witness(
            asg,
            program,
            physical,
            # Scoped COUNT/AVOID steering may need several graph traversals.
            # A two-second cutoff discarded valid ~2.2s witnesses and sent
            # otherwise easy queries into a much larger complete product.
            # This optional window remains subject to the caller's total
            # cancellation budget; only replayed witnesses can establish SAT.
            deadline=time.perf_counter() + 10.0,
            requirements=(
                request.witness_requirements.outcomes
                if request.witness_requirements is not None else ()
            ),
            cancel_token=cancel_token,
        )
        seek_zero_penalty = (
            fast is not None
            and float(fast.witness_penalty or 0.0) > 0.0
            and (
                request.mode is QueryMode.MIN_PENALTY
                or (
                    request.mode is QueryMode.PENALTY_AT_MOST
                    and float(fast.witness_penalty or 0.0) > float(request.penalty_bound)
                )
            )
        )
        if seek_zero_penalty:
            # A cheap exact lower bound can settle an unavoidable penalty
            # before the optional bounded zero-vector construction consumes
            # a short caller budget. A non-exclusion remains inconclusive.
            from reconav.solver.topology_relaxation import prove_penalty_bound
            incumbent_penalty = float(fast.witness_penalty or 0.0)
            rounded_penalty = int(round(incumbent_penalty))
            if math.isclose(incumbent_penalty, rounded_penalty, abs_tol=1e-9):
                cached_bound = (
                    rounded_penalty - 1 if request.mode is QueryMode.MIN_PENALTY
                    else int(request.penalty_bound)
                )
                cached_topology_bound = prove_penalty_bound(
                    asg, program, physical, cached_bound,
                    deadline=time.perf_counter() + 0.5, cancel_token=cancel_token,
                )
                if cached_topology_bound.outcome == "BOUND_EXCLUDED":
                    seek_zero_penalty = False
        if seek_zero_penalty:
            # A replayed all-zero vector supplies its own global lower bound
            # because penalties are nonnegative. Search for that witness before
            # invoking the complete product; keep the incumbent if steering
            # fails, since a failed optional construction proves nothing.
            zero_requirements = tuple(
                (kind, index, 0)
                for kind, specs in (
                    ("via", program.soft_via_specs),
                    ("avoid", program.soft_avoid_specs),
                    ("count", program.soft_count_specs),
                    ("order", program.soft_order_specs),
                )
                for index in range(len(specs))
            )
            try:
                zero = fast_complete_hard_witness(
                    asg, program, physical,
                    deadline=time.perf_counter() + 10.0,
                    requirements=zero_requirements,
                    cancel_token=cancel_token,
                )
            except SolveCancelledError:
                return replace(
                    fast,
                    requested_mode=request.mode.value,
                    requested_bound=request.penalty_bound,
                    query_outcome="SAT_INCOMPLETE",
                    status_reason="valid incumbent retained after zero-penalty search cancellation",
                    cancelled=True, incomplete=True,
                    timing={**fast.timing, "total_seconds": time.perf_counter() - started},
                    counters={**fast.counters, "full_query_graph_created": 0},
                )
            if zero is not None:
                fast = zero
            fast = replace(
                fast,
                counters={
                    **fast.counters,
                    "zero_penalty_directed_attempt": 1,
                    "zero_penalty_directed_found": int(zero is not None),
                },
            )
        if fast is not None and request.witness_requirements is not None:
            if _fast_witness_satisfies_requirements(
                fast,
                program,
                request.witness_requirements.outcomes,
            ):
                fast = replace(
                    fast,
                    counters={
                        **fast.counters,
                        "witness_requirements_checked": len(
                            request.witness_requirements.outcomes
                        ),
                    },
                )
            else:
                # An ordinary sparse witness is not a negative fact about the
                # requested outcome.  Continue into the complete product and
                # never retain this route as a cancellation fallback.
                fast = None
        if fast is not None and request.mode is QueryMode.FEASIBLE:
            counters = dict(fast.counters)
            counters["full_query_graph_created"] = 0
            return replace(
                fast,
                requested_mode=request.mode.value,
                requested_bound=request.penalty_bound,
                timing={
                    **fast.timing,
                    "total_seconds": time.perf_counter() - started,
                },
                counters=counters,
            )
        if fast is not None and request.mode is QueryMode.PENALTY_AT_MOST and (
            float(fast.witness_penalty or 0.0) <= float(request.penalty_bound)
        ):
            counters = dict(fast.counters)
            counters["full_query_graph_created"] = 0
            return replace(
                fast,
                requested_mode=request.mode.value,
                requested_bound=request.penalty_bound,
                timing={
                    **fast.timing,
                    "total_seconds": time.perf_counter() - started,
                },
                counters=counters,
            )
        if fast is not None and request.mode is QueryMode.MIN_PENALTY and math.isclose(
            float(fast.witness_penalty or 0.0),
            0.0,
            abs_tol=1e-9,
        ):
            return replace(
                fast,
                requested_mode=request.mode.value,
                query_outcome="MIN_PENALTY",
                penalty_lb=0.0,
                penalty_ub=0.0,
                penalty_proven=True,
                certificate_kind="nonnegative-penalty+witness",
                certificate_payload={"nonnegative_unit_penalties": True},
                timing={
                    **fast.timing,
                    "total_seconds": time.perf_counter() - started,
                },
                counters={**fast.counters, "full_query_graph_created": 0},
            )

    topology_bound = None
    bound_to_check = None
    if request.mode is QueryMode.PENALTY_AT_MOST:
        bound_to_check = int(request.penalty_bound)
    elif request.mode is QueryMode.MIN_PENALTY and fast is not None:
        witness_penalty = float(fast.witness_penalty or 0.0)
        rounded = int(round(witness_penalty))
        if rounded > 0 and math.isclose(witness_penalty, rounded, abs_tol=1e-9):
            bound_to_check = rounded - 1
    if bound_to_check is not None:
        from reconav.solver.topology_relaxation import prove_penalty_bound

        topology_bound = cached_topology_bound if cached_bound == bound_to_check else None
        if topology_bound is None:
            topology_bound = prove_penalty_bound(
                asg,
                program,
                physical,
                bound_to_check,
                deadline=time.perf_counter() + 0.5,
                cancel_token=cancel_token,
            )
        if topology_bound.outcome == "BOUND_EXCLUDED":
            lower_bound = float(bound_to_check + 1)
            if fast is not None:
                upper_bound = float(fast.witness_penalty or 0.0)
                proven = math.isclose(lower_bound, upper_bound, abs_tol=1e-9)
                return replace(
                    fast,
                    requested_mode=request.mode.value,
                    requested_bound=request.penalty_bound,
                    query_outcome=(
                        "BOUND_EXCLUDED"
                        if request.mode is QueryMode.PENALTY_AT_MOST
                        else "MIN_PENALTY"
                    ),
                    penalty_lb=lower_bound,
                    penalty_ub=upper_bound,
                    penalty_proven=proven,
                    certificate_kind="component-portal-bound+witness",
                    certificate_payload=topology_bound.certificate,
                    timing={
                        **fast.timing,
                        "total_seconds": time.perf_counter() - started,
                    },
                    counters={
                        **fast.counters,
                        "full_query_graph_created": 0,
                        "topology_states": topology_bound.states_visited,
                    },
                )
            return CrossRegionPathResult(
                [],
                [],
                [],
                status="UNKNOWN",
                status_reason=(
                    f"component/portal relaxation excludes penalty <= {bound_to_check}; "
                    "hard feasibility remains unresolved"
                ),
                proof_domain=f"G0={physical.graph_id};component-portal-relaxation",
                requested_mode=request.mode.value,
                requested_bound=request.penalty_bound,
                query_outcome="BOUND_EXCLUDED",
                penalty_lb=lower_bound,
                certificate_kind="component-portal-bound-exclusion",
                certificate_payload=topology_bound.certificate,
                timing={"total_seconds": time.perf_counter() - started},
                counters={
                    "full_query_graph_created": 0,
                    "topology_states": topology_bound.states_visited,
                },
            )

    endpoint_object_ids: set[int] = set()
    for item in grounded.values():
        if item.kind == "obj":
            endpoint_object_ids.update(item.ids)
    watcher_object_ids: set[int] = set()
    for item in avoid.avoids:
        watcher_object_ids.update(item.object_ids)
    for item in (*count.counts, *prefer.avoids, *prefer.counts, *prefer.vias):
        if item.kind == "obj":
            watcher_object_ids.update(item.ids)

    graph_started = time.perf_counter()
    try:
        base = get_complete_query_graph(asg, cancel_token)
        query = AttachedQuery(
            base,
            asg,
            original_start,
            cancel_token,
            active_object_ids=endpoint_object_ids | watcher_object_ids,
        )
    except SolveCancelledError:
        if fast is None:
            raise
        return replace(
            fast,
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="SAT_INCOMPLETE",
            status_reason="valid sparse-G0 incumbent retained after cancellation",
            cancelled=True,
            incomplete=True,
            timing={
                **fast.timing,
                "total_seconds": time.perf_counter() - started,
            },
            counters={**fast.counters, "full_query_graph_created": 0},
        )
    graph_seconds = time.perf_counter() - graph_started
    domain = (
        f"{QUERY_GRAPH_VERSION}/{EVENT_SEMANTICS_VERSION};"
        f"G0={base.physical.graph_id};finite published semantic vertices+anchors"
    )

    missing = (endpoint_object_ids | watcher_object_ids) & base.missing_semantic_objects
    if missing:
        return CrossRegionPathResult(
            [],
            [],
            [],
            status="UNKNOWN",
            status_reason=f"semantic geometry missing for {sorted(missing)}",
            proof_domain=domain,
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="INCOMPLETE",
            incomplete=True,
        )
    objects_by_region: dict[int, set[int]] = defaultdict(set)
    for object_id in watcher_object_ids:
        obj = asg.get_object(object_id)
        if obj is None:
            raise ValueError(f"missing object {object_id}")
        for region_id in {obj.region_index, *obj.body_band_footprint_2d_by_region}:
            objects_by_region[region_id].add(object_id)

    goal_regions = (
        {asg.get_object(goal_id).region_index}
        if goal.kind == "obj"
        else {goal_id}
        if goal.kind == "region"
        else {region.index for region in asg.regions if region.level == goal_id}
    )
    exact_goal_components: set[int] = set()
    for region_id in goal_regions:
        for component_id in range(len(base.physical.regions[region_id].components)):
            exact_goal_components.add(
                base.physical.region_to_global_component[(region_id, component_id)]
            )
    if goal.kind == "obj":
        exact_goal_components = {
            base.physical.region_to_global_component[
                (query.nodes[node_id].region, query.nodes[node_id].component)
            ]
            for node_id in base.object_nodes.get(goal_id, ())
        }
        shape, _level, region = base.object_shapes[goal_id]
        for node_id in range(base.base_size, len(query.nodes)):
            node = query.nodes[node_id]
            if (
                node.region == region
                and within_semantic_clearance(
                    Point(node.x, node.z).distance(shape),
                    base.physical.config.endpoint_clearance_m,
                )
            ):
                exact_goal_components.add(
                    base.physical.region_to_global_component[(node.region, node.component)]
                )
    exact_start_components = {
        base.physical.region_to_global_component[
            (query.nodes[node_id].region, query.nodes[node_id].component)
        ]
        for node_id in query.start_ids
    }
    if not exact_goal_components:
        return CrossRegionPathResult(
            [],
            [],
            [],
            status="UNKNOWN",
            status_reason="no complete legal endpoint domain for GOAL",
            proof_domain=domain,
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="INCOMPLETE",
            incomplete=True,
        )
    if not exact_start_components.intersection(exact_goal_components):
        return CrossRegionPathResult(
            [],
            [],
            [],
            status="UNSAT",
            status_reason="declared physical components and portal contracts separate endpoints",
            query_graph_complete=True,
            state_search_complete=True,
            proof_domain=domain,
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="HARD_UNSAT",
            certificate_kind="complete-endpoint-component-disconnection",
            certificate_payload=topology_payload,
        )

    scope_specs = [
        item.spec
        for item in (*avoid.avoids, *count.counts, *prefer.avoids, *prefer.counts)
    ]
    endpoint_cache = {}

    def eligible(node_id, portal_id=None, entry_region=False, initial=False):
        key = (node_id, portal_id, entry_region, initial)
        if key in endpoint_cache:
            return endpoint_cache[key]
        node = query.nodes[node_id]
        point = Point(node.x, node.z)
        result = []
        for name, target in grounded.items():
            matches = []
            if target.kind == "obj":
                for entity_id in target.ids:
                    shape, _level, region = base.object_shapes[entity_id]
                    if (
                        region == node.region
                        and within_semantic_clearance(
                            point.distance(shape),
                            base.physical.config.endpoint_clearance_m,
                        )
                    ):
                        matches.append(entity_id)
            elif target.kind == "region" and node.region in target.ids and (
                entry_region or initial or name == "GOAL"
            ):
                matches.append(node.region)
            elif target.kind == "level" and base.levels[node.region] in target.ids and (
                entry_region or initial or name == "GOAL"
            ):
                matches.append(base.levels[node.region])
            elif target.kind == "portal" and portal_id is not None and portal_id in target.ids:
                matches.append(portal_id)
            result.extend((name, target.kind, int(value)) for value in matches)
        endpoint_cache[key] = tuple(result)
        return endpoint_cache[key]

    def choices(progress, available):
        options = [(progress, ())]
        seen = {(progress, ())}
        cursor = 0
        while cursor < len(options):
            prior, hits = options[cursor]
            cursor += 1
            if prior and prior[-1] == "GOAL":
                continue
            for name, kind, index in available:
                if name in prior:
                    continue
                if any(name == spec.until and spec.after not in prior for spec in scope_specs):
                    continue
                following = order._observe_one(prior, name)
                if following is None:
                    continue
                if name == "GOAL" and not order.accepting(following):
                    continue
                record = (following, (*hits, (name, kind, index)))
                if record not in seen:
                    seen.add(record)
                    options.append(record)
        return options

    def is_forbidden(progress, targets, entities):
        for item in avoid.avoids:
            if not _active(item.spec, progress, targets):
                continue
            pools = {
                "obj": item.object_ids,
                "region": item.region_ids,
                "portal": item.portal_ids,
                "level": item.level_ids,
            }
            if any(entity_id in pools.get(kind, ()) for kind, entity_id in entities):
                return True
        return False

    def observe(progress, counts, preference, targets, entities):
        if is_forbidden(progress, targets, entities):
            return None
        following_count = count.observe(
            counts,
            progress,
            targets,
            entities,
            include_until_transition=False,
        )
        if following_count is None:
            return None
        step = prefer.observe(preference, progress, targets, entities)
        if not math.isfinite(step.penalty):
            return None
        return following_count, step.state, step.penalty

    event_trees = {}
    for region_id, ids in objects_by_region.items():
        ordered = tuple(sorted(ids))
        event_trees[region_id] = (
            ordered,
            STRtree([base.object_shapes[entity_id][0] for entity_id in ordered]),
        )
    counters = {
        "full_query_graph_created": 1,
        "expanded_states": 0,
        "distinct_count_keys": 0,
        "edge_event_cache_hits": 0,
        "edge_event_cache_misses": 0,
        "object_encounter_calls": 0,
        "watcher_objects": len(watcher_object_ids),
    }

    def edge_events(source, target, portal):
        global _RAW_EVENT_CACHE_BYTES, _RAW_EVENT_CACHE_EVICTIONS
        if not watcher_object_ids and portal is None:
            # Ordinary spatial edges cannot create any event relevant to this
            # query.  Do not allocate keys or LineStrings for empty answers.
            return ()
        a, b = query.nodes[source], query.nodes[target]
        key = (
            base.physical.graph_id,
            EVENT_SEMANTICS_VERSION,
            tuple(sorted(watcher_object_ids)),
            (a.region, round(a.x, 9), round(a.z, 9)),
            (b.region, round(b.x, 9), round(b.z, 9)),
            portal,
        )
        cached = _RAW_EVENT_CACHE.get(key)
        if cached is not None:
            counters["edge_event_cache_hits"] += 1
            _RAW_EVENT_CACHE.move_to_end(key)
            return cached
        counters["edge_event_cache_misses"] += 1
        groups = defaultdict(list)
        if watcher_object_ids:
            pa, pb = query.point3(source), query.point3(target)
            radius = base.physical.config.endpoint_clearance_m
            bounds = box(
                min(a.x, b.x) - radius,
                min(a.z, b.z) - radius,
                max(a.x, b.x) + radius,
                max(a.z, b.z) + radius,
            )
            ids = set()
            for region_id in {a.region, b.region}:
                if region_id in event_trees:
                    ordered, tree = event_trees[region_id]
                    ids.update(ordered[int(index)] for index in tree.query(bounds))
            counters["object_encounter_calls"] += 1
            for event in object_encounter_entries(
                asg,
                ids,
                [pa, pb],
                [a.region, b.region],
            ):
                fraction = max(0.0, min(1.0, float(event.segment_fraction)))
                groups[round(fraction, 10)].append(("obj", event.object_id))
        if portal is not None:
            groups[1.0].append(("portal", portal))
            if a.region != b.region:
                groups[1.0].append(("region", b.region))
            if base.levels[a.region] != base.levels[b.region]:
                groups[1.0].append(("level", base.levels[b.region]))
        result = tuple(
            (fraction, tuple(sorted(set(values))))
            for fraction, values in sorted(groups.items())
        )
        estimated = 96 + 48 * sum(len(values) for _, values in result)
        _RAW_EVENT_CACHE[key] = result
        _RAW_EVENT_CACHE_BYTES += estimated
        while (
            _RAW_EVENT_CACHE_BYTES > _EVENT_CACHE_BUDGET_BYTES
            and len(_RAW_EVENT_CACHE) > 1
        ):
            _old_key, old = _RAW_EVENT_CACHE.popitem(last=False)
            _RAW_EVENT_CACHE_BYTES -= 96 + 48 * sum(
                len(values) for _, values in old
            )
            _RAW_EVENT_CACHE_EVICTIONS += 1
        counters["event_cache_budget_bytes"] = _EVENT_CACHE_BUDGET_BYTES
        counters["event_cache_estimated_bytes"] = _RAW_EVENT_CACHE_BYTES
        counters["event_cache_evictions"] = _RAW_EVENT_CACHE_EVICTIONS
        return result


    target_geometry_cache = {}

    def target_geometry(target):
        key = (target.kind, tuple(sorted(target.ids)))
        cached = target_geometry_cache.get(key)
        if cached is not None:
            return cached
        if target.kind == "obj":
            geometry = unary_union(
                [
                    base.object_shapes[entity_id][0].buffer(
                        base.physical.config.endpoint_clearance_m
                    )
                    for entity_id in target.ids
                ]
            )
        elif target.kind == "region":
            geometry = unary_union(
                [base.physical.regions[region_id].support for region_id in target.ids]
            )
        elif target.kind == "level":
            geometry = unary_union(
                [
                    base.physical.regions[region_id].support
                    for region_id, level_id in base.levels.items()
                    if level_id in target.ids
                ]
            )
        elif target.kind == "portal":
            landings = []
            for connector in base.physical.portal_connectors:
                if not connector.validated or connector.portal_id not in target.ids:
                    continue
                for landing_id in (connector.node_a, connector.node_b):
                    landing = base.physical.nodes[landing_id]
                    landings.append(Point(landing.x, landing.y))
            geometry = unary_union(landings) if landings else Point()
        else:
            geometry = Point()
        target_geometry_cache[key] = geometry
        return geometry

    def target_lower_bound(node_id, target):
        node = query.nodes[node_id]
        point = Point(node.x, node.z)
        geometry = target_geometry(target)
        if geometry.is_empty:
            return 0.0
        return float(point.distance(geometry))

    milestone_by_name = program.milestone_by_name()
    heuristic_cache = {}
    fixed_chain = None
    next_constraints = [
        item for item in order.constraints if item.kind == "next"
    ]
    if (
        len(next_constraints) == len(milestone_by_name) - 1
        and len(next_constraints) == len(order.constraints)
    ):
        successor = {item.first: item.second for item in next_constraints}
        chain = ["START"]
        while chain[-1] in successor and successor[chain[-1]] not in chain:
            chain.append(successor[chain[-1]])
        if len(chain) == len(milestone_by_name) and chain[-1] == "GOAL":
            fixed_chain = tuple(chain)
    fixed_chain_tail = {}
    if fixed_chain is not None:
        remaining = 0.0
        fixed_chain_tail[fixed_chain[-1]] = 0.0
        for left, right in reversed(list(zip(fixed_chain, fixed_chain[1:]))):
            remaining += float(
                target_geometry(grounded[left]).distance(
                    target_geometry(grounded[right])
                )
            ) if left != "START" else 0.0
            fixed_chain_tail[left] = remaining

    def lower_bound(state):
        """Distance to one currently observable mandatory route event.

        This is admissible because every named milestone must still be reached.
        For a fixed NEXT chain it points at the next ordered target ID rather than the
        final goal. For a partial order it uses the closest event that can
        legally occur next.
        """

        key = (state.node, state.progress)
        cached = heuristic_cache.get(key)
        if cached is not None:
            return cached
        if (
            fixed_chain is not None
            and tuple(state.progress) == fixed_chain[: len(state.progress)]
            and len(state.progress) < len(fixed_chain)
        ):
            next_name = fixed_chain[len(state.progress)]
            value = target_lower_bound(state.node, grounded[next_name])
            value += fixed_chain_tail[next_name]
            heuristic_cache[key] = value
            return value
        candidates = []
        for name in milestone_by_name:
            if name == "START" or name in state.progress:
                continue
            following = order._observe_one(state.progress, name)
            if following is None:
                continue
            if name == "GOAL" and not order.accepting(following):
                continue
            candidates.append(target_lower_bound(state.node, grounded[name]))
        value = (
            min(candidates)
            if candidates
            else target_lower_bound(state.node, goal)
        )
        heuristic_cache[key] = value
        return value

    counter = itertools.count()
    queue = []
    dist = {}
    parent = {}
    witness = {}
    roots = {}
    count_keys: set[CountState] = set()

    def search_priority(state, cost):
        if request.mode is QueryMode.MIN_PENALTY:
            # The proof-bearing penalty stays the primary key. Distance only
            # orders work inside an equal-penalty layer.
            return (cost[0], cost[1] + lower_bound(state))
        if request.mode is QueryMode.FEASIBLE:
            return (0.0, cost[1] + lower_bound(state))
        return (cost[0], cost[1] + lower_bound(state))

    def enqueue(state, cost, prior=None, step=None, root=None):
        if (
            request.mode is QueryMode.PENALTY_AT_MOST
            and cost[0] > float(request.penalty_bound)
        ):
            return
        if cost >= dist.get(state, (math.inf, math.inf)):
            return
        dist[state] = cost
        count_keys.add(state.counts)
        if prior is not None:
            parent[state] = prior
            witness[state] = step
        else:
            roots[state] = root
        heapq.heappush(
            queue,
            (search_priority(state, cost), next(counter), False, state, cost),
        )

    for start_id in query.start_ids:
        node = query.nodes[start_id]
        start_progress = order._observe_one((), "START")
        if start_progress is None:
            continue
        for progress, hits in choices(
            start_progress,
            eligible(start_id, initial=True),
        ):
            preference = prefer.initial_state(progress, node.region)
            initial = ProductState(
                start_id,
                progress,
                count.initial_state(),
                preference,
            )
            enqueue(initial, (0.0, 0.0), root=hits)

    expanded = 0
    incumbent = None
    try:
        while queue:
            if cancel_token is not None:
                cancel_token.raise_if_requested()
            _priority, _sequence, terminal, state, cost = heapq.heappop(queue)
            if terminal:
                counters["distinct_count_keys"] = len(count_keys)
                penalty_proven = request.mode in {
                    QueryMode.MIN_PENALTY,
                    QueryMode.LEX_OPTIMAL,
                }
                length_proven = request.mode is QueryMode.LEX_OPTIMAL
                return reconstruct(
                    state,
                    cost,
                    query,
                    program,
                    parent,
                    witness,
                    roots,
                    domain,
                    expanded,
                    request,
                    counters,
                    started,
                    graph_seconds,
                    penalty_proven=penalty_proven,
                    length_proven=length_proven,
                )
            if cost != dist.get(state):
                continue
            expanded += 1
            counters["expanded_states"] = expanded
            if state.progress and state.progress[-1] == "GOAL":
                if order.accepting(state.progress) and count.accepting(state.counts):
                    if (
                        request.witness_requirements is not None
                        and not prefer.satisfies_requirements(
                            state.preference,
                            request.witness_requirements.outcomes,
                        )
                    ):
                        continue
                    terminal_penalty = prefer.final_penalty(state.preference)
                    if math.isfinite(terminal_penalty):
                        final = (cost[0] + terminal_penalty, cost[1])
                        if incumbent is None or final < incumbent[1]:
                            incumbent = (state, final)
                        if request.mode is QueryMode.FEASIBLE or (
                            request.mode is QueryMode.PENALTY_AT_MOST
                            and final[0] <= float(request.penalty_bound)
                        ):
                            counters["distinct_count_keys"] = len(count_keys)
                            return reconstruct(
                                state,
                                final,
                                query,
                                program,
                                parent,
                                witness,
                                roots,
                                domain,
                                expanded,
                                request,
                                counters,
                                started,
                                graph_seconds,
                                penalty_proven=False,
                                length_proven=False,
                            )
                        if request.mode is QueryMode.MIN_PENALTY:
                            priority = (final[0], -1.0)
                        else:
                            priority = final
                        heapq.heappush(
                            queue,
                            (priority, next(counter), True, state, final),
                        )
                continue
            for move in query.neighbors(state.node, cancel_token):
                target = move.target
                node = query.nodes[target]
                if (
                    base.physical.region_to_global_component[
                        (node.region, node.component)
                    ]
                    not in exact_goal_components
                ):
                    continue
                events = edge_events(state.node, target, move.portal_id)
                current_count = state.counts
                current_preference = state.preference
                incremental = 0.0
                observations = []
                failed = False
                end_entities = ()
                for fraction, entities in events:
                    if fraction >= 1.0 - EVENT_POSITION_EPS:
                        end_entities = (*end_entities, *entities)
                        continue
                    deltas = count.observation_deltas_for_event(
                        current_count,
                        state.progress,
                        (),
                        entities,
                    )
                    observed = observe(
                        state.progress,
                        current_count,
                        current_preference,
                        (),
                        entities,
                    )
                    if observed is None:
                        failed = True
                        break
                    next_count, next_preference, penalty = observed
                    for delta in deltas:
                        observations.append(
                            (
                                delta.constraint_index,
                                delta.entity_kind,
                                delta.entity_id,
                                fraction,
                                len(state.progress) - 1,
                                len(state.progress) - 1,
                            )
                        )
                    current_count, current_preference = next_count, next_preference
                    incremental += penalty
                if failed:
                    continue
                available = eligible(
                    target,
                    move.portal_id,
                    query.nodes[state.node].region != node.region,
                )
                for progress, hits in choices(state.progress, available):
                    names = tuple(item[0] for item in hits)
                    deltas = count.observation_deltas_for_event(
                        current_count,
                        state.progress,
                        names,
                        end_entities,
                    )
                    observed = observe(
                        state.progress,
                        current_count,
                        current_preference,
                        names,
                        end_entities,
                    )
                    if observed is None:
                        continue
                    final_count, final_preference, penalty = observed
                    if "GOAL" in names and not count.accepting(final_count):
                        continue
                    record = list(observations)
                    for delta in deltas:
                        record.append(
                            (
                                delta.constraint_index,
                                delta.entity_kind,
                                delta.entity_id,
                                1.0,
                                len(state.progress) - 1,
                                len(progress) - 1,
                            )
                        )
                    successor = ProductState(
                        target,
                        progress,
                        final_count,
                        final_preference,
                    )
                    next_cost = (
                        cost[0] + incremental + penalty,
                        cost[1] + move.length,
                    )
                    step = StepWitness(
                        state.node,
                        target,
                        move.portal_id,
                        hits,
                        tuple(fraction for fraction, _ in events),
                        tuple(record),
                    )
                    enqueue(successor, next_cost, prior=state, step=step)
    except SolveCancelledError:
        if incumbent is None:
            if fast is None:
                raise
            return replace(
                fast,
                requested_mode=request.mode.value,
                requested_bound=request.penalty_bound,
                query_outcome="SAT_INCOMPLETE",
                status_reason="valid sparse-G0 incumbent retained after cancellation",
                cancelled=True,
                incomplete=True,
                timing={
                    **fast.timing,
                    "query_graph_seconds": graph_seconds,
                    "total_seconds": time.perf_counter() - started,
                },
                counters={
                    **fast.counters,
                    "full_query_graph_created": 1,
                    "expanded_states": expanded,
                },
            )
        counters["distinct_count_keys"] = len(count_keys)
        result = reconstruct(
            incumbent[0],
            incumbent[1],
            query,
            program,
            parent,
            witness,
            roots,
            domain,
            expanded,
            request,
            counters,
            started,
            graph_seconds,
            penalty_proven=False,
            length_proven=False,
        )
        return replace(
            result,
            status_reason="valid incumbent retained after cancellation",
            query_outcome="SAT_INCOMPLETE",
            cancelled=True,
            incomplete=True,
        )

    counters["distinct_count_keys"] = len(count_keys)
    if request.mode is QueryMode.PENALTY_AT_MOST:
        return CrossRegionPathResult(
            [],
            [],
            [],
            status="UNKNOWN",
            status_reason=(
                f"complete finite query product excludes penalty <= {request.penalty_bound}; "
                "hard feasibility remains unresolved by this bounded query"
            ),
            query_graph_complete=True,
            state_search_complete=True,
            proof_domain=domain,
            requested_mode=request.mode.value,
            requested_bound=request.penalty_bound,
            query_outcome="BOUND_EXCLUDED",
            penalty_lb=float(request.penalty_bound + 1),
            certificate_kind="finite-product-bound-exclusion",
            certificate_payload={
                "excluded_penalty_bound": request.penalty_bound,
                "expanded_states": expanded,
                "graph_id": physical.graph_id,
            },
            timing={
                "query_graph_seconds": graph_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            counters=counters,
        )
    return CrossRegionPathResult(
        [],
        [],
        [],
        status="UNSAT",
        status_reason="complete finite query visibility/event product exhausted",
        query_graph_complete=True,
        state_search_complete=True,
        proof_domain=domain,
        requested_mode=request.mode.value,
        requested_bound=request.penalty_bound,
        query_outcome="HARD_UNSAT",
        certificate_kind="finite-product-exhaustion",
        certificate_payload={"expanded_states": expanded},
        timing={
            "query_graph_seconds": graph_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        counters=counters,
    )


def reconstruct(
    state,
    cost,
    query,
    program,
    parent,
    witness,
    roots,
    domain,
    expanded,
    request,
    counters,
    started,
    graph_seconds,
    *,
    penalty_proven,
    length_proven,
):
    from reconav.solver.cross_region import (
        CrossRegionPathResult,
        RouteCountObservation,
        RouteMilestoneHit,
    )

    chain = []
    current = state
    while current in parent:
        chain.append(witness[current])
        current = parent[current]
    chain.reverse()
    points = [query.point3(current.node)]
    regions = [query.nodes[current.node].region]
    portals = []
    hits = []
    trace = []

    def add_hit(name, kind, index, position):
        item = program.milestone_by_name()[name]
        value = item.target.values[0] if item.target.values else index
        if item.target.kind == "pos":
            value = index
        hits.append(
            RouteMilestoneHit(
                name,
                item.role,
                kind,
                item.target.selector,
                value,
                index,
                points[position],
                position,
                len(hits),
            )
        )

    start = program.start()
    start_id = int(start.target.values[0]) if start.target.kind == "obj" else regions[0]
    add_hit("START", start.target.kind, start_id, 0)
    for name, kind, index in roots[current] or ():
        add_hit(name, kind, index, 0)
    for step in chain:
        pa, pb = query.point3(step.source), query.point3(step.target)
        fractions = sorted(
            {
                *(fraction for fraction in step.fractions if EVENT_POSITION_EPS < fraction < 1.0 - EVENT_POSITION_EPS),
                1.0,
            }
        )
        positions = {}
        for fraction in fractions:
            point = Point3(
                pa.x + (pb.x - pa.x) * fraction,
                pa.y + (pb.y - pa.y) * fraction,
                pa.z + (pb.z - pa.z) * fraction,
            )
            points.append(point)
            regions.append(
                query.nodes[step.target].region
                if fraction == 1.0
                else query.nodes[step.source].region
            )
            positions[fraction] = len(points) - 1
        if step.portal is not None:
            portals.append(step.portal)
        for name, kind, index in step.names:
            add_hit(name, kind, index, len(points) - 1)
        for constraint_index, kind, index, fraction, before, after in step.observations:
            position = positions[min(positions, key=lambda value: abs(value - fraction))]
            trace.append(
                RouteCountObservation(
                    f"count[{constraint_index}]",
                    kind,
                    index,
                    points[position],
                    before,
                    after,
                    position,
                    len(trace),
                )
            )
    counters = dict(counters)
    counters["expanded_states"] = expanded
    outcome = (
        "LEX_OPTIMAL"
        if length_proven
        else "MIN_PENALTY"
        if penalty_proven
        else "SAT_WITNESS"
    )
    return CrossRegionPathResult(
        points,
        regions,
        portals,
        hits,
        cost[0],
        trace,
        status="SAT",
        status_reason=f"{outcome.lower()} in finite event product; expanded_states={expanded}",
        query_graph_complete=True,
        state_search_complete=penalty_proven or length_proven,
        optimality_established=length_proven,
        proof_domain=domain,
        requested_mode=request.mode.value,
        requested_bound=request.penalty_bound,
        query_outcome=outcome,
        witness_penalty=cost[0],
        witness_length=cost[1],
        penalty_lb=cost[0] if penalty_proven else 0.0,
        penalty_ub=cost[0],
        penalty_proven=penalty_proven,
        length_lb=cost[1] if length_proven else 0.0,
        length_ub=cost[1],
        length_proven=length_proven,
        length_condition=("minimum-penalty layer" if length_proven else None),
        certificate_kind=(
            "finite-product-lexicographic-exhaustion"
            if length_proven
            else "finite-product-penalty-exhaustion"
            if penalty_proven
            else "replayed-hard-witness"
        ),
        certificate_payload={"expanded_states": expanded},
        timing={
            "query_graph_seconds": graph_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        counters=counters,
    )
