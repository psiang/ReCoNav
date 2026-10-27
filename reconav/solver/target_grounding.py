"""Ground DSL target selectors to ASG entity ids."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reconav.scene.asg import (
    ASG,
    ASGLevelEntity,
    ASGObjectEntity,
    ASGPortalEntity,
    ASGRegionEntity,
)
from reconav.solver.dsl_ast import AvoidSpec, TargetSelector
from reconav.solver.cancellation import check_local_search_budget
from reconav.scene.vocabulary import (
    default_object_categories,
    object_confusion_groups,
    object_head_alias,
    region_confusion_groups,
)

NEAR_RADIUS_M = 2.0
_DISTANCE_TIE_EPS_M = 1e-6

@dataclass(frozen=True)
class GroundedTarget:
    kind: str
    ids: frozenset[int]
    geom: BaseGeometry | None = None


def normalize_category(raw: str) -> str:
    return " ".join(str(raw).strip().lower().replace("_", " ").split())


def target_from_avoid(spec: AvoidSpec) -> TargetSelector:
    return TargetSelector(spec.kind, spec.selector, spec.values, spec.qualifiers)


def ground_target(
    asg: ASG, target: TargetSelector, line: int, *, start_target: TargetSelector | None = None
) -> GroundedTarget:
    if target.kind == "pos":
        raise ValueError(f"line {line}: POS cannot be grounded to an ASG entity")
    ids = _base_ids(asg, target, line)
    ordinary = tuple(
        qualifier
        for qualifier in target.qualifiers
        if qualifier.kind not in {"nearest_to", "farthest_from"}
    )
    extremes = tuple(
        qualifier
        for qualifier in target.qualifiers
        if qualifier.kind in {"nearest_to", "farthest_from"}
    )
    for qualifier in ordinary:
        if qualifier.kind == "in_region":
            ref = ground_target(asg, qualifier.target, line, start_target=start_target)
            if ref.kind != "region":
                raise ValueError(f"line {line}: IN qualifier requires a REGION reference")
            ids = _filter_in_regions(asg, target.kind, ids, ref.ids)
        elif qualifier.kind == "in_level":
            ref = ground_target(asg, qualifier.target, line, start_target=start_target)
            if ref.kind != "level":
                raise ValueError(f"line {line}: IN qualifier requires a LEVEL reference")
            ids = _filter_in_levels(asg, target.kind, ids, ref.ids)
        elif qualifier.kind == "between_regions":
            if target.kind != "portal":
                raise ValueError(f"line {line}: BETWEEN qualifier is supported only for PORTAL targets")
            if qualifier.second_target is None:
                raise ValueError(f"line {line}: BETWEEN qualifier requires two REGION references")
            first = ground_target(asg, qualifier.target, line, start_target=start_target)
            second = ground_target(asg, qualifier.second_target, line, start_target=start_target)
            if first.kind != "region" or second.kind != "region":
                raise ValueError(f"line {line}: BETWEEN qualifier requires REGION references")
            ids = _filter_between_regions(asg, ids, first.ids, second.ids)
        elif qualifier.kind == "near":
            if target.kind != "obj":
                raise ValueError(
                    f"line {line}: {qualifier.kind} requires an OBJ target"
                )
            from reconav.solver.qualifier_geometry import evaluate_object_distance
            mode = qualifier.reference_mode
            reference_ids = ()
            if mode != "start":
                reference = ground_target(asg, qualifier.target, line, start_target=start_target)
                if reference.kind != "obj":
                    raise ValueError(f"line {line}: distance reference must be OBJ")
                reference_ids = reference.ids
            ids = evaluate_object_distance(
                asg, ids, qualifier.kind, reference_ids=reference_ids,
                reference_mode=mode, start_target=start_target,
            ).ids
        else:
            raise ValueError(f"line {line}: unsupported target qualifier {qualifier.kind}")
        if not ids:
            raise ValueError(f"line {line}: target qualifiers removed all candidates")
    comparison_ids = ids
    extreme_results: list[frozenset[int]] = []
    for qualifier in extremes:
        if target.kind != "obj":
            raise ValueError(
                f"line {line}: {qualifier.kind} requires an OBJ target"
            )
        if qualifier.comparison_domain != "reference_level":
            raise ValueError(
                f"line {line}: {qualifier.kind} requires reference_level domain"
            )
        from reconav.solver.qualifier_geometry import evaluate_object_distance
        mode = qualifier.reference_mode
        reference_ids = ()
        if mode != "start":
            reference = ground_target(
                asg, qualifier.target, line, start_target=start_target
            )
            if reference.kind != "obj":
                raise ValueError(f"line {line}: distance reference must be OBJ")
            reference_ids = reference.ids
        evaluated = evaluate_object_distance(
            asg,
            comparison_ids,
            qualifier.kind,
            reference_ids=reference_ids,
            reference_mode=mode,
            start_target=start_target,
        )
        if evaluated.comparison_domain != "reference_level":
            raise ValueError("distance evaluator lost reference-level domain")
        extreme_results.append(evaluated.ids)
    if extreme_results:
        ids = frozenset.intersection(*extreme_results)
        if not ids:
            raise ValueError(f"line {line}: target qualifiers removed all candidates")
    return GroundedTarget(target.kind, ids, _grounded_geom(asg, target.kind, ids))


def _base_ids(asg: ASG, target: TargetSelector, line: int) -> frozenset[int]:
    if target.selector == "id":
        ids = frozenset(int(v) for v in target.values)
        missing = [idx for idx in ids if _entity(asg, target.kind, idx) is None]
        if missing:
            raise ValueError(f"line {line}: {target.kind} {missing[0]} not found")
        return ids
    if target.selector == "category":
        category = normalize_category(str(target.values[0]))
        if target.kind == "obj":
            categories = default_object_categories()
            canonical = (
                categories[category].canonical
                if category in categories
                else object_head_alias(category)
            )
            denotation = object_confusion_groups(categories).get(
                canonical, frozenset({canonical})
            )
            ids = frozenset(
                entity.index
                for entity in asg.objects
                if normalize_category(entity.label) in denotation
            )
        elif target.kind == "region":
            groups = region_confusion_groups(
                frozenset(normalize_category(region.label) for region in asg.regions)
            )
            denotation = groups.get(category, frozenset({category}))
            ids = frozenset(
                entity.index
                for entity in asg.regions
                if normalize_category(entity.label) in denotation
            )
        elif target.kind == "portal" and category == "portal":
            from reconav.scene.vocabulary import ORDINARY_PORTAL_LABELS
            ids = frozenset(entity.index for entity in asg.portals
                            if normalize_category(entity.label) in ORDINARY_PORTAL_LABELS)
        else:
            ids = frozenset(
                entity.index
                for entity in _entities(asg, target.kind)
                if normalize_category(entity.label) == category
            )
        if not ids:
            raise ValueError(f"line {line}: no {target.kind} matches category '{category}'")
        return ids
    if target.selector == "any":
        ids = frozenset(entity.index for entity in _entities(asg, target.kind))
        if not ids:
            raise ValueError(f"line {line}: no {target.kind} entities found")
        return ids
    raise ValueError(f"line {line}: unsupported target selector {target.selector}")


def _filter_in_regions(asg: ASG, kind: str, ids: frozenset[int], region_ids: frozenset[int]) -> frozenset[int]:
    out: set[int] = set()
    for idx in ids:
        if kind == "obj":
            obj = asg.get_object(idx)
            if obj is not None and obj.region_index in region_ids:
                out.add(idx)
        elif kind == "portal":
            portal = asg.get_portal(idx)
            if portal is None:
                continue
            sides = {portal.region_a, portal.region_b}
            matches = region_ids.issubset(sides) if len(region_ids) > 1 else bool(sides & region_ids)
            if matches:
                out.add(idx)
        elif kind == "region":
            if idx in region_ids:
                out.add(idx)
        else:
            raise ValueError(f"IN REGION is not supported for {kind}")
    return frozenset(out)


def _filter_in_levels(asg: ASG, kind: str, ids: frozenset[int], level_ids: frozenset[int]) -> frozenset[int]:
    out: set[int] = set()
    for idx in ids:
        if kind == "obj":
            obj = asg.get_object(idx)
            region = asg.get_region(obj.region_index) if obj is not None else None
            if region is not None and region.level in level_ids:
                out.add(idx)
        elif kind == "region":
            region = asg.get_region(idx)
            if region is not None and region.level in level_ids:
                out.add(idx)
        elif kind == "level":
            if idx in level_ids:
                out.add(idx)
        else:
            raise ValueError(f"IN LEVEL is not supported for {kind}")
    return frozenset(out)


def _filter_between_regions(
    asg: ASG,
    ids: frozenset[int],
    first_region_ids: frozenset[int],
    second_region_ids: frozenset[int],
) -> frozenset[int]:
    out: set[int] = set()
    for idx in ids:
        portal = asg.get_portal(idx)
        if portal is None:
            continue
        a = portal.region_a
        b = portal.region_b
        if (a in first_region_ids and b in second_region_ids) or (a in second_region_ids and b in first_region_ids):
            out.add(idx)
    return frozenset(out)


def _grounded_geom(asg: ASG, kind: str, ids: frozenset[int]):
    return _union_geoms(_geom_for(asg, kind, idx) for idx in ids)


def _entities(asg: ASG, kind: str) -> Iterable[ASGObjectEntity | ASGRegionEntity | ASGPortalEntity | ASGLevelEntity]:
    if kind == "obj":
        return asg.objects
    if kind == "region":
        return asg.regions
    if kind == "portal":
        return asg.portals
    if kind == "level":
        return asg.levels
    raise ValueError(f"unsupported target kind: {kind}")


def _entity(asg: ASG, kind: str, idx: int):
    if kind == "obj":
        return asg.get_object(idx)
    if kind == "region":
        return asg.get_region(idx)
    if kind == "portal":
        return asg.get_portal(idx)
    if kind == "level":
        return asg.get_level(idx)
    raise ValueError(f"unsupported target kind: {kind}")


def _geom_for(asg: ASG, kind: str, idx: int):
    check_local_search_budget()
    if kind == "obj":
        obj = asg.get_object(idx)
        if obj is None:
            return None
        geom = _union_geoms(
            poly.to_shapely() for poly in obj.navigation_footprint_2d()
        )
        if geom is not None:
            return geom
        c = obj.get_navigation_centroid()
        return Point(c.x, c.y)
    if kind == "region":
        region = asg.get_region(idx)
        if region is None:
            return None
        return _union_geoms(poly.to_shapely() for poly in region.polygon_2d)
    if kind == "portal":
        portal = asg.get_portal(idx)
        if portal is None:
            return None
        return LineString([(portal.anchor_a.x, portal.anchor_a.y), (portal.anchor_b.x, portal.anchor_b.y)])
    if kind == "level":
        return _union_geoms(
            poly.to_shapely()
            for region in asg.regions_for_level(idx)
            for poly in region.polygon_2d
        )
    raise ValueError(f"unsupported target kind: {kind}")


def _union_geoms(geoms):
    valid = [geom for geom in geoms if geom is not None and not geom.is_empty]
    if not valid:
        return None
    return unary_union(valid) if len(valid) > 1 else valid[0]
