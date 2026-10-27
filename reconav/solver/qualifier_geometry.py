"""Pure, shared spatial relation semantics (never consult navigation reachability).

Distance is to the *semantic* footprint, not an anchor or a robot-inflated
obstacle.  A missing possible witness is not the same thing as an empty class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import weakref

from shapely.geometry import Point
from shapely.ops import unary_union

from reconav.scene.asg import ASG

NEAR_RADIUS_M = 2.0
DISTANCE_TIE_EPS_M = 1e-6
SEMANTIC_RELATION_VERSION = "distance-same-region-near-reference-level-extreme-v2"


class UnknownRelationGeometry(ValueError):
    """A denotation cannot be certified from the available semantic geometry."""


@dataclass(frozen=True)
class DistanceEvaluation:
    ids: frozenset[int]
    distances: dict[int, float]
    witnesses: dict[int, tuple[int, ...]]
    comparison_domain: str | None = None
    resolved_reference_level: int | None = None
    comparison_ids: tuple[int, ...] = ()


_GEOMETRY_CACHE = {}
_GEOMETRY_FAILURE_CACHE = {}
_RELATION_CACHE = {}
_ASG_REFS = {}


def object_geometry(asg: ASG, object_id: int):
    identity = id(asg)
    reference = _ASG_REFS.get(identity)
    if reference is None or reference() is not asg:
        for cache in (_GEOMETRY_CACHE, _GEOMETRY_FAILURE_CACHE, _RELATION_CACHE):
            for key in [key for key in cache if key[0] == identity]:
                cache.pop(key, None)
        def cleanup(ref, identity=identity):
            if _ASG_REFS.get(identity) is ref:
                _ASG_REFS.pop(identity, None)
                for cache in (_GEOMETRY_CACHE, _GEOMETRY_FAILURE_CACHE, _RELATION_CACHE):
                    for key in [key for key in cache if key[0] == identity]:
                        cache.pop(key, None)
        _ASG_REFS[identity] = weakref.ref(asg, cleanup)
    key = (identity, int(object_id))
    if key in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[key]
    if key in _GEOMETRY_FAILURE_CACHE:
        raise UnknownRelationGeometry(_GEOMETRY_FAILURE_CACHE[key])

    def unavailable(reason):
        # Like positive geometry, this is a fact about this immutable ASG
        # snapshot only. Store text rather than an exception/traceback that
        # would retain the ASG. There is no navigation/configuration-dependent
        # fact here, and a cached UNKNOWN never becomes an empty denotation.
        _GEOMETRY_FAILURE_CACHE[key] = reason
        raise UnknownRelationGeometry(reason)

    obj = asg.get_object(int(object_id))
    if obj is None:
        unavailable(f"missing object {object_id}")
    # Unassigned mesh objects may have enormous footprints. Metadata alone
    # already makes their relation geometry UNKNOWN; do not union those mesh
    # contours on every attempted distance description before discovering it.
    if obj.level_index is None or obj.region_index is None:
        unavailable(
            f"object {object_id} has no known region/level"
        )
    region = asg.get_region(int(obj.region_index))
    if region is None or int(region.level) != int(obj.level_index):
        unavailable(
            f"object {object_id} has inconsistent region/level metadata"
        )
    parts = obj.navigation_footprint_2d()
    if not parts:
        unavailable(f"object {object_id} has no semantic footprint")
    shape = unary_union([part.to_shapely() for part in parts])
    if shape.is_empty or not shape.is_valid:
        unavailable(f"object {object_id} has invalid semantic footprint")
    result = (shape, int(obj.level_index), int(obj.region_index))
    _GEOMETRY_CACHE[key] = result
    return result


def original_start_point(asg: ASG, start_target):
    """Resolve only the original task START; never accept a local handoff."""
    if start_target is None:
        raise UnknownRelationGeometry("distance qualifier requires original task START")
    if start_target.kind != "pos" or len(start_target.values) != 3:
        raise UnknownRelationGeometry("FROM START requires the task's fixed START POS")
    x, height, z = (float(value) for value in start_target.values)
    if not all(math.isfinite(value) for value in (x, height, z)):
        raise UnknownRelationGeometry("non-finite START coordinates")
    point = Point(x, z)
    candidates = []
    for region in asg.regions:
        shape = unary_union([part.to_shapely() for part in region.polygon_2d])
        if shape.buffer(DISTANCE_TIE_EPS_M).covers(point):
            candidates.append((abs(float(region.get_height()) - height), int(region.level)))
    if not candidates:
        raise UnknownRelationGeometry("START has no region/level at its original coordinate")
    minimum = min(delta for delta, _level in candidates)
    levels = {level for delta, level in candidates if delta <= minimum + DISTANCE_TIE_EPS_M}
    if len(levels) != 1:
        raise UnknownRelationGeometry("START level is ambiguous")
    return point, next(iter(levels))


def _evaluate_object_distance_uncached(
    asg: ASG,
    candidate_ids: Iterable[int],
    relation: str,
    *,
    reference_ids: Iterable[int] = (),
    reference_mode: str = "unique",
    start_target=None,
) -> DistanceEvaluation:
    """Evaluate unique OBJ, explicit existential OBJ-class, or original START.

    EXISTS is permitted only for NEAR. Self-reference cannot be an existential
    witness. All denotations include unavailable/unreachable peers; this layer
    must not turn navigation sampling exclusions into language uniqueness.
    """
    if relation not in {"near", "nearest_to", "farthest_from"}:
        raise ValueError(f"unsupported distance relation {relation!r}")
    if reference_mode not in {"unique", "exists", "start"}:
        raise ValueError(f"unsupported reference mode {reference_mode!r}")
    ids = tuple(sorted(set(int(value) for value in candidate_ids)))
    refs = tuple(sorted(set(int(value) for value in reference_ids)))
    if reference_mode == "start":
        if relation == "near":
            raise ValueError("START reference is supported only for nearest/farthest")
        reference_shape, reference_level = original_start_point(asg, start_target)
        reference_shapes = {-1: (reference_shape, reference_level, None)}
    else:
        if reference_mode == "unique" and len(refs) != 1:
            raise ValueError(f"singular reference is not unique: {list(refs)}")
        if reference_mode == "exists" and relation != "near":
            raise ValueError("existential class references are supported only for NEAR")
        if not refs:
            return DistanceEvaluation(frozenset(), {}, {})
        reference_shapes = {}
        missing_refs = []
        for ref in refs:
            try:
                reference_shapes[ref] = object_geometry(asg, ref)
            except UnknownRelationGeometry:
                missing_refs.append(ref)
    distances: dict[int, float] = {}
    witnesses: dict[int, tuple[int, ...]] = {}
    unresolved = []
    for candidate_id in ids:
        try:
            shape, level, region = object_geometry(asg, candidate_id)
        except UnknownRelationGeometry:
            unresolved.append(candidate_id)
            continue
        comparable = {
            ref: float(shape.distance(ref_shape))
            for ref, (ref_shape, ref_level, ref_region) in reference_shapes.items()
            if ref_level == level
            and ref != candidate_id
            and (relation != "near" or ref_region == region)
        }
        if reference_mode == "exists":
            hits = tuple(sorted(ref for ref, value in comparable.items()
                                if value <= NEAR_RADIUS_M + DISTANCE_TIE_EPS_M))
            if hits:
                witnesses[candidate_id] = hits
                distances[candidate_id] = min(comparable.values())
                continue
            possibly_missing = []
            for ref in missing_refs:
                if ref == candidate_id:
                    continue
                obj = asg.get_object(ref)
                if obj is None:
                    possibly_missing.append(ref)
                    continue
                if obj.level_index not in {None, level}:
                    continue
                if relation == "near" and obj.region_index not in {None, region}:
                    continue
                possibly_missing.append(ref)
            if possibly_missing:
                unresolved.append(candidate_id)
            elif comparable:
                distances[candidate_id] = min(comparable.values())
            continue
        if reference_mode == "unique" and missing_refs:
            unresolved.append(candidate_id)
            continue
        if comparable:
            distances[candidate_id] = min(comparable.values())
            witnesses[candidate_id] = tuple(comparable)
    if unresolved:
        raise UnknownRelationGeometry(f"distance denotation unresolved for objects {sorted(set(unresolved))}")
    if relation == "near":
        selected = frozenset(i for i, value in distances.items()
                             if value <= NEAR_RADIUS_M + DISTANCE_TIE_EPS_M)
    elif not distances:
        selected = frozenset()
    else:
        extreme = min(distances.values()) if relation == "nearest_to" else max(distances.values())
        selected = frozenset(i for i, value in distances.items()
                             if abs(value - extreme) <= DISTANCE_TIE_EPS_M)
    reference_levels = {
        ref_level for _shape, ref_level, _region in reference_shapes.values()
    }
    resolved_reference_level = (
        next(iter(reference_levels)) if len(reference_levels) == 1 else None
    )
    comparison_ids = tuple(sorted(distances))
    return DistanceEvaluation(
        selected,
        distances,
        {i: witnesses.get(i, ()) for i in selected},
        "reference_level" if relation in {"nearest_to", "farthest_from"} else None,
        resolved_reference_level,
        comparison_ids,
    )


def evaluate_object_distance(asg, candidate_ids, relation, *, reference_ids=(), reference_mode="unique", start_target=None):
    """Memoize immutable ASG-derived distance truth, not stored task proofs."""
    ids = tuple(sorted(set(int(x) for x in candidate_ids)))
    refs = tuple(sorted(set(int(x) for x in reference_ids)))
    start = tuple(start_target.values) if start_target is not None else ()
    key = (id(asg), ids, relation, refs, reference_mode, start)
    owner = _ASG_REFS.get(id(asg))
    if owner is not None and owner() is asg and key in _RELATION_CACHE:
        return _RELATION_CACHE[key]
    result = _evaluate_object_distance_uncached(asg, ids, relation, reference_ids=refs,
        reference_mode=reference_mode, start_target=start_target)
    if len(_RELATION_CACHE) >= 65536:
        _RELATION_CACHE.clear()
    _RELATION_CACHE[key] = result
    return result
