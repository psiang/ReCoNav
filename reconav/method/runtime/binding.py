"""Learned execution binding and exact, joint progressive instance selection."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import math
import json
import hashlib
from collections.abc import Mapping

from reconav.benchmark.route_evaluator import evaluate_grounded_route
from reconav.method.grounding.contracts import GroundingBinding
from reconav.method.grounding.grounded_contract import ASGBinding, GroundedOccurrence, SceneGroundedContract
from reconav.method.grounding.label_grounding import collect_nominal_commitments, validate_relation_signatures, vocabulary_from_entity_payload
from reconav.method.grounding.scene_binding import (
    align_frozen_label_grounding,
    bind_candidate_scene,
    _node_at_path,
    _nominal_path,
    start_target,
)
from reconav.solver.dsl_ast import TargetQualifier, TargetSelector
from reconav.solver.query_evidence import FEASIBLE, MIN_PENALTY
from reconav.solver.target_grounding import ground_target
from reconav.method.lowering.dsl import lower_grounded_contract
from reconav.method.runtime.backend import QueryBudgetExhausted


class BindingResolutionError(ValueError):
    def __init__(self, code: str, message: str, report: dict, *, infrastructure: bool = False):
        super().__init__(message)
        self.code = code
        self.report = report
        self.infrastructure = infrastructure


def identity_paths(contract) -> tuple[str, ...]:
    paths = ["$.route.goal.target"]
    paths.extend(f"$.route.via_events[{i}].target" for i, _ in enumerate(contract.route.via_events))
    # Current navigation benchmark has no carry/WIDTH. Reject before lowering,
    # rather than silently removing a hallucinated manipulation requirement.
    if contract.constraints.carry or contract.constraints.robot_width:
        raise ValueError("carry/robot-width are not supported by the current navigation execution contract")
    raw = contract.model_dump(mode="json")
    return tuple(_nominal_path(_node_at_path(raw, path), path) for path in paths)


def _nominal_identity(node):
    return json.dumps({k: node.get(k, [] if k == "qualifiers" else None)
                       for k in ("head_text", "source_text", "qualifiers")}, sort_keys=True)


def coreference_groups(contract, scene) -> dict[str, str]:
    """Only explicit coreferences share decisions. Repeated nouns do not."""
    root = {o.path: o.path for o in scene.occurrences}
    raw = contract.model_dump(mode="json")
    plain = [o for o in scene.occurrences if ".antecedent" not in o.path]
    for occurrence in scene.occurrences:
        if ".antecedent" not in occurrence.path:
            continue
        node = _node_at_path(raw, occurrence.path)
        matches = [o for o in plain
                   if _nominal_identity(_node_at_path(raw, o.path)) == _nominal_identity(node)]
        if len(matches) > 1:
            raise ValueError(f"coreference at {occurrence.path} has more than one source occurrence")
        if len(matches) == 1:
            root[occurrence.path] = matches[0].path
    return root


def label_grounding_artifact(contract, bindings, *, validation_success=True, diagnostics=(), determination=None):
    collection = collect_nominal_commitments(contract)
    result = {
        "stage1_gate_passed": True, "occurrence_count": len(collection.commitments),
        "all_grounding_items": [{"head_text": o.head_text, "source_text": o.source_text, "asg_kind": None}
                                for o in collection.commitments],
        "final_bindings": [b.model_dump(mode="json") if hasattr(b, "model_dump") else dict(b) for b in bindings],
        "validation": {"executed": True, "parse_success": True, "success": bool(validation_success),
                       "diagnostics": list(diagnostics)},
    }
    if determination is not None:
        result["determination"] = determination.model_dump(mode="json")
    return result


def _canonical_learned_path(path: str) -> str:
    """Normalize transport-only path roots; semantic paths are contract-relative."""
    prefix = "$.semantic_contract."
    return "$." + path[len(prefix):] if path.startswith(prefix) else path


def learned_scene(contract, decisions, asg, robot_position):
    """Validate the learned contract without resolving/replacing its choices.

    The baselines own all instance IDs, including constraint/reference sets.
    Check their declared types, denotations, qualifier relations and explicit
    coreferences without replacing their choices. A committed route target may
    supply legal existential witnesses; constraint sets must remain complete.
    """
    collection = collect_nominal_commitments(contract)
    paths = [o.path for o in collection.commitments]
    normalized = [
        d.model_copy(update={"path": _canonical_learned_path(d.path)})
        if _canonical_learned_path(d.path) != d.path
        else d
        for d in decisions
    ]
    by_path = {d.path: d for d in normalized}
    if len(by_path) != len(normalized) or set(by_path) != set(paths):
        raise ValueError("binding paths must match every nominal occurrence exactly once")
    occurrences = []
    labels = []
    lookup = {"object": asg.get_object, "region": asg.get_region, "portal": asg.get_portal, "level": asg.get_level}
    vocabulary = vocabulary_from_entity_payload(asg.to_entity_dict()).model_dump()
    raw = contract.model_dump(mode="json")
    identities = set(identity_paths(contract))
    for i, occurrence in enumerate(collection.commitments):
        item = by_path[occurrence.path]
        ids = item.grounded_ids
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ID in {item.path}")
        if item.asg_kind not in occurrence.possible_kinds:
            raise ValueError(f"binding violates structural type at {item.path}")
        if item.asg_label not in vocabulary[item.asg_kind]:
            raise ValueError(f"unsupported ASG label {item.asg_kind}/{item.asg_label}")
        if any(lookup[item.asg_kind](v) is None for v in ids):
            raise ValueError(f"binding contains nonexistent {item.asg_kind} ID at {item.path}")
        if item.path in identities and len(ids) != 1:
            raise ValueError(f"learned identity commitment must contain one ID at {item.path}")
        node = _node_at_path(raw, item.path)
        if ".references[" in item.path and node.get("quantifier", "unique") == "unique" and len(ids) != 1:
            raise ValueError(f"singular qualifier reference must contain one ID at {item.path}")
        label = GroundingBinding(head_text=occurrence.head_text, asg_kind=item.asg_kind, asg_label=item.asg_label)
        labels.append(label)
        occurrences.append(GroundedOccurrence(position=i, path=item.path, owner_path=occurrence.owner_path,
            head_text=occurrence.head_text, source_text=occurrence.source_text,
            asg_binding=ASGBinding(asg_kind=item.asg_kind, asg_label=item.asg_label, grounded_ids=list(ids))))
    errors = validate_relation_signatures(collection, labels)
    if errors:
        raise ValueError("; ".join(errors))
    artifact = label_grounding_artifact(contract, labels)
    scene = SceneGroundedContract(semantic_contract=raw, occurrences=occurrences)
    roots = coreference_groups(contract, scene)
    for path, root in roots.items():
        if root != path and set(by_path[path].grounded_ids) != set(by_path[root].grounded_ids):
            raise ValueError(f"learned coreference is inconsistent: {path} -> {root}")
    _validate_learned_denotations(
        contract,
        artifact,
        decisions=normalized,
        asg=asg,
        robot_position=robot_position,
    )
    return scene, artifact


def _validate_learned_denotations(
    contract,
    artifact,
    *,
    decisions,
    asg,
    robot_position,
):
    """Reject hallucinated learned IDs without repairing or replacing them.

    The learned method owns every instance commitment. Validation uses those
    exact commitments when a qualifier contains a unique/existential reference;
    it never replaces a learned reference with deterministic local uniqueness.
    """
    aligned = align_frozen_label_grounding(contract, artifact)
    del aligned  # positional/artifact consistency was checked by alignment
    start = start_target(robot_position)
    decisions_by_path = {item.path: item for item in decisions}
    raw = contract.model_dump(mode="json")
    identity = set(identity_paths(contract))
    commitments = collect_nominal_commitments(contract).commitments

    # First validate every learned ID against the method's own declared
    # kind/label class, independently of qualifiers.
    for position, occurrence in enumerate(commitments):
        decision = decisions_by_path[occurrence.path]
        base = _learned_base_selector(decision)
        legal_base = ground_target(
            asg, base, position + 1, start_target=start
        ).ids
        chosen = set(int(value) for value in decision.grounded_ids)
        if not chosen.issubset(set(int(value) for value in legal_base)):
            raise ValueError(
                f"learned binding is outside its declared label denotation at "
                f"{occurrence.path}: chosen={sorted(chosen)}"
            )

    # Then validate owner denotations with qualifier references fixed exactly
    # to the IDs that the learned method itself committed.
    for position, occurrence in enumerate(commitments):
        decision = decisions_by_path[occurrence.path]
        selector = _learned_selector(
            raw,
            occurrence.path,
            decisions_by_path,
        )
        legal = ground_target(
            asg,
            selector,
            position + 1,
            start_target=start,
        )
        legal_ids = set(int(value) for value in legal.ids)
        chosen = set(int(value) for value in decision.grounded_ids)
        if not chosen.issubset(legal_ids):
            raise ValueError(
                f"learned binding is outside its declared qualifier denotation at "
                f"{occurrence.path}: chosen={sorted(chosen)}"
            )

        node = _node_at_path(raw, occurrence.path)
        inside_reference = (
            ".qualifiers[" in occurrence.path
            and ".references[" in occurrence.path
        )
        if occurrence.path in identity:
            if len(chosen) != 1:
                raise ValueError(
                    f"learned identity commitment must contain one ID at {occurrence.path}"
                )
        elif inside_reference:
            quantifier = str(node.get("quantifier", "unique"))
            if quantifier == "unique" and len(chosen) != 1:
                raise ValueError(
                    f"singular qualifier reference must contain one ID at {occurrence.path}"
                )
            if quantifier == "exists":
                # A GOAL/VIA is already one learned identity. A nonempty subset
                # can witness its existential qualifier, including nested
                # references. The owner was checked against these exact IDs
                # above; no automatic completion or identity selection occurs.
                # Under AVOID/COUNT (including preferences), keep the full
                # reference set so a witness cannot shrink a quantified set.
                route_witness = any(
                    occurrence.path.startswith(f"{path}.qualifiers[")
                    for path in identity
                )
                if not chosen:
                    raise ValueError(
                        f"existential qualifier reference has no witness at "
                        f"{occurrence.path}"
                    )
                if not route_witness and chosen != legal_ids:
                    raise ValueError(
                        f"existential qualifier binding does not equal its declared set "
                        f"denotation at {occurrence.path}"
                    )
        elif occurrence.path.startswith("$.constraints.avoid[") or occurrence.path.startswith(
            "$.constraints.count["
        ):
            if chosen != legal_ids:
                raise ValueError(
                    f"learned set binding does not equal its declared denotation at "
                    f"{occurrence.path}"
                )


def _learned_base_selector(decision) -> TargetSelector:
    kind = {
        "object": "obj",
        "region": "region",
        "portal": "portal",
        "level": "level",
    }[decision.asg_kind]
    if decision.asg_label == "any":
        return TargetSelector(kind, "any", ())
    return TargetSelector(kind, "category", (decision.asg_label,))


def _learned_selector(raw_contract, path, decisions_by_path) -> TargetSelector:
    """Compile a learned occurrence using its own committed reference IDs."""
    node = _node_at_path(raw_contract, path)
    if node.get("kind") == "coreference":
        antecedent_path = f"{path}.antecedent"
        return _learned_selector(
            raw_contract, antecedent_path, decisions_by_path
        )
    if node.get("kind") != "nominal":
        raise ValueError(f"{path} is not a learned nominal occurrence")
    try:
        decision = decisions_by_path[path]
    except KeyError as exc:
        raise ValueError(f"missing learned occurrence at {path}") from exc

    base = _learned_base_selector(decision)
    qualifiers = []
    for q_index, qualifier in enumerate(node.get("qualifiers") or []):
        q_path = f"{path}.qualifiers[{q_index}]"
        references = qualifier.get("references") or []
        ref_selectors = [
            _learned_reference_selector(
                raw_contract,
                reference,
                f"{q_path}.references[{r_index}]",
                decisions_by_path,
            )
            for r_index, reference in enumerate(references)
        ]
        relation = qualifier.get("relation")
        if relation == "between":
            if len(ref_selectors) != 2:
                raise ValueError(f"BETWEEN at {q_path} requires two references")
            qualifiers.append(
                TargetQualifier(
                    "between_regions",
                    ref_selectors[0],
                    second_target=ref_selectors[1],
                )
            )
            continue
        if len(ref_selectors) != 1:
            raise ValueError(f"{relation} at {q_path} requires one reference")
        if relation == "in":
            if ref_selectors[0].kind not in {"region", "level"}:
                raise ValueError(f"IN at {q_path} requires region/level")
            qualifier_kind = f"in_{ref_selectors[0].kind}"
        elif relation in {"near", "nearest_to", "farthest_from"}:
            qualifier_kind = str(relation)
        else:
            raise ValueError(f"unsupported qualifier {relation!r}")

        reference_node = references[0]
        while reference_node.get("kind") == "coreference":
            reference_node = reference_node["antecedent"]
        mode = (
            "start"
            if reference_node.get("kind") == "start"
            else str(reference_node.get("quantifier", "unique"))
        )
        qualifiers.append(
            TargetQualifier(
                qualifier_kind,
                ref_selectors[0],
                reference_mode=mode,
                comparison_domain=(
                    "reference_level"
                    if qualifier_kind in {"nearest_to", "farthest_from"}
                    else None
                ),
            )
        )
    return TargetSelector(
        base.kind,
        base.selector,
        base.values,
        tuple(qualifiers),
    )


def _learned_reference_selector(
    raw_contract,
    node,
    path,
    decisions_by_path,
) -> TargetSelector:
    kind = node.get("kind")
    if kind == "start":
        return TargetSelector("pos", "coord", ())
    if kind == "coreference":
        return _learned_reference_selector(
            raw_contract,
            node["antecedent"],
            f"{path}.antecedent",
            decisions_by_path,
        )
    if kind != "nominal":
        raise ValueError(f"unsupported learned reference kind at {path}: {kind}")
    try:
        decision = decisions_by_path[path]
    except KeyError as exc:
        raise ValueError(f"missing learned qualifier reference at {path}") from exc
    dsl_kind = {
        "object": "obj",
        "region": "region",
        "portal": "portal",
        "level": "level",
    }[decision.asg_kind]
    values = tuple(int(value) for value in decision.grounded_ids)
    if not values:
        raise ValueError(f"learned qualifier reference has no IDs at {path}")
    return TargetSelector(dsl_kind, "id", values)


@dataclass(frozen=True)
class IdentityCandidateContext:
    scene: SceneGroundedContract
    roots: dict[str, str]
    domains: dict[str, tuple[int, ...]]
    all_occurrences: tuple[dict, ...]

    @property
    def total_assignments(self) -> int:
        return math.prod(len(values) for values in self.domains.values())


def build_identity_candidate_context(contract, label_grounding, asg, robot_position):
    """Shared local candidate construction for ReCoNav and Forced-Early.

    This function performs only semantic/ASG denotation and explicit-coreference
    intersection. It does not use route constraints, preferences, path length, or
    benchmark gold to remove candidates.
    """
    candidates = bind_candidate_scene(contract, label_grounding, asg, robot_position)
    roots = coreference_groups(contract, candidates)
    by_path = {occurrence.path: occurrence for occurrence in candidates.occurrences}
    domains: dict[str, tuple[int, ...]] = {}
    for path in identity_paths(contract):
        root = roots[path]
        domain = set(by_path[path].asg_binding.grounded_ids)
        for alias, parent in roots.items():
            if parent == root:
                domain &= set(by_path[alias].asg_binding.grounded_ids)
        domains[root] = tuple(sorted(domain))
    all_occurrences = tuple(
        {
            "path": occurrence.path,
            "kind": occurrence.asg_binding.asg_kind,
            "label": occurrence.asg_binding.asg_label,
            "ids": list(occurrence.asg_binding.grounded_ids),
        }
        for occurrence in candidates.occurrences
    )
    return IdentityCandidateContext(
        scene=candidates,
        roots=roots,
        domains=domains,
        all_occurrences=all_occurrences,
    )


def _bound_scene(candidate_scene, roots, chosen):
    scene = candidate_scene.model_copy(deep=True)
    for occurrence in scene.occurrences:
        root = roots[occurrence.path]
        if root in chosen:
            value = chosen[root]
            if value not in occurrence.asg_binding.grounded_ids:
                raise ValueError("explicit coreference denotations disagree")
            occurrence.asg_binding.grounded_ids = [value]
    return scene


def seeded_random_identity_choice(
    report: Mapping[str, object],
    *,
    seed: int,
    scene_id: str,
    task_id: str,
    prefer_proved_feasible: bool = False,
) -> tuple[dict[str, int] | None, dict[str, object]]:
    """Deterministically choose one unresolved assignment without learned reasoning.

    Forced-Early receives only its post-assurance local candidate product. Progressive
    execution fallback may additionally inherit the admissible assignment set
    already established by its ordinary hard/preference evidence.
    """
    raw_domains = report.get("candidate_domains")
    if not isinstance(raw_domains, Mapping):
        raw_domains = report.get("ambiguous_domains")
    if not isinstance(raw_domains, Mapping):
        return None, {
            "status": "not_applicable",
            "reason": "candidate domains unavailable",
        }
    domains = {
        str(path): tuple(sorted({int(value) for value in values}))
        for path, values in raw_domains.items()
        if isinstance(values, list) and values
    }
    ambiguous = {
        path: values for path, values in domains.items() if len(values) > 1
    }
    if not ambiguous:
        return None, {
            "status": "not_applicable",
            "reason": "no multi-candidate identity domains",
            "candidate_domains": {
                path: list(values) for path, values in domains.items()
            },
        }

    def normalized_assignment(value) -> dict[str, int] | None:
        if not isinstance(value, Mapping):
            return None
        out = {}
        for path, domain in domains.items():
            raw = value.get(path)
            if raw is None:
                return None
            candidate = int(raw)
            if candidate not in domain:
                return None
            out[path] = candidate
        return out

    explicit = report.get("unresolved_assignments")
    options: list[dict[str, int]] = []
    option_source = None
    if isinstance(explicit, list):
        for item in explicit:
            normalized = normalized_assignment(item)
            if normalized is not None and normalized not in options:
                options.append(normalized)
        if options:
            option_source = "unresolved_assignments"

    rows = report.get("assignments")
    if (
        not options
        and report.get("hard_filter_complete") is True
        and isinstance(rows, list)
    ):
        hard_sat_rows: list[tuple[dict[str, int], Mapping[str, object] | None]] = []
        proved_preference_costs: list[float] = []
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            hard = item.get("hard_evidence")
            if not isinstance(hard, Mapping) or hard.get("status") != "SAT":
                continue
            normalized = normalized_assignment(item.get("chosen"))
            if normalized is None:
                continue
            preference = item.get("preference_evidence")
            pref_mapping = preference if isinstance(preference, Mapping) else None
            hard_sat_rows.append((normalized, pref_mapping))
            if (
                pref_mapping is not None
                and pref_mapping.get("penalty_proven") is True
                and isinstance(pref_mapping.get("soft_cost"), (int, float))
                and math.isfinite(float(pref_mapping["soft_cost"]))
            ):
                proved_preference_costs.append(float(pref_mapping["soft_cost"]))

        best_proved_cost = (
            min(proved_preference_costs)
            if proved_preference_costs
            else None
        )
        preference_dominated = False
        for normalized, preference in hard_sat_rows:
            if (
                best_proved_cost is not None
                and preference is not None
                and preference.get("penalty_proven") is True
                and isinstance(preference.get("soft_cost"), (int, float))
                and math.isfinite(float(preference["soft_cost"]))
                and float(preference["soft_cost"]) > best_proved_cost + 1e-9
            ):
                preference_dominated = True
                continue
            if normalized not in options:
                options.append(normalized)

        if options:
            option_source = (
                "hard_feasible_minus_proved_preference_dominated"
                if preference_dominated
                else "proved_hard_feasible_assignments"
            )

    if prefer_proved_feasible and report.get("hard_filter_complete") is not True and isinstance(rows, list):
        feasible = []
        for item in rows:
            if not isinstance(item, Mapping) or not isinstance(item.get("hard_evidence"), Mapping):
                continue
            if item["hard_evidence"].get("status") != "SAT":
                continue
            normalized = normalized_assignment(item.get("chosen"))
            if normalized is not None and (not options or normalized in options) and normalized not in feasible:
                feasible.append(normalized)
        if feasible:
            options = feasible
            option_source = "execution_from_proved_hard_feasible_assignments"

    proved_unsat: set[str] = set()
    if isinstance(rows, list):
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            hard = item.get("hard_evidence")
            if not isinstance(hard, Mapping):
                continue
            if not (
                hard.get("hard_infeasible_proved") is True
                or (
                    hard.get("status") == "UNSAT"
                    and hard.get("query_outcome") == "HARD_UNSAT"
                )
            ):
                continue
            normalized = normalized_assignment(item.get("chosen"))
            if normalized is not None:
                proved_unsat.add(
                    json.dumps(normalized, sort_keys=True, separators=(",", ":"))
                )

    paths = tuple(sorted(domains))
    total = 1
    for path in paths:
        total *= len(domains[path])

    digest_input = json.dumps(
        {
            "seed": int(seed),
            "scene_id": str(scene_id),
            "task_id": str(task_id),
            "candidate_domains": {
                path: list(domains[path]) for path in paths
            },
            "options": (
                sorted(
                    json.dumps(item, sort_keys=True, separators=(",", ":"))
                    for item in options
                )
                if options
                else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    number = int(digest, 16)

    if options:
        options.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            )
        )
        chosen_full = options[number % len(options)]
    else:
        # When proof is incomplete, retain every assignment not already proved
        # HARD_UNSAT; known SAT does not receive an automatic tie-break advantage.
        if total <= 0:
            return None, {
                "status": "unresolved",
                "reason": "empty candidate product",
            }

        def decode(index: int) -> dict[str, int]:
            out: dict[str, int] = {}
            for path in reversed(paths):
                values = domains[path]
                out[path] = values[index % len(values)]
                index //= len(values)
            return {path: out[path] for path in paths}

        chosen_full = None
        start = number % total
        for offset in range(total):
            candidate = decode((start + offset) % total)
            serialized = json.dumps(
                candidate, sort_keys=True, separators=(",", ":")
            )
            if serialized not in proved_unsat:
                chosen_full = candidate
                break
        if chosen_full is None:
            return None, {
                "status": "unresolved",
                "reason": "every candidate assignment is proved HARD_UNSAT",
                "candidate_domains": {
                    path: list(values) for path, values in domains.items()
                },
            }
        option_source = "candidate_product_minus_proved_hard_unsat"

    choices = {
        path: int(chosen_full[path])
        for path in sorted(ambiguous)
    }
    evidence_filtered = bool(proved_unsat)
    if option_source in {
        "unresolved_assignments",
        "proved_hard_feasible_assignments",
        "execution_from_proved_hard_feasible_assignments",
        "hard_feasible_minus_proved_preference_dominated",
    }:
        evidence_filtered = True

    return choices, {
        "status": "resolved",
        "mechanism": "seeded_random_commit",
        "reason": (
            "deterministic pseudo-random choice from the current admissible "
            "candidate set"
        ),
        "candidate_domains": {
            path: list(values) for path, values in domains.items()
        },
        "option_source": option_source,
        "selection_digest": digest,
        "seed": int(seed),
        "validated_choices": choices,
        "chosen_assignment": chosen_full,
        "certified_identity": False,
        "uses_global_constraint_filtering": evidence_filtered,
        "selection_uses_path_length": False,
    }


class ForcedEarlyResolver:
    """Build Forced-Early's local frontier without global constraint resolution.

    Singleton local denotations are committed immediately. A multi-candidate
    identity is reported without consulting route constraints or preferences;
    verification/repair may first repair semantics/grounding, after which Forced-Early can collapse
    any residual ambiguity with its non-learned seeded tie-break.
    """

    def __init__(self):
        self.history: list[dict] = []

    def __call__(self, contract, label_grounding, asg, robot_position):
        context = build_identity_candidate_context(
            contract, label_grounding, asg, robot_position
        )
        report = {
            "status": "started",
            "candidate_source": "ASG+semantic_closure",
            "candidate_domains": {
                path: list(values) for path, values in context.domains.items()
            },
            "all_occurrences": list(context.all_occurrences),
            "total_assignments": context.total_assignments,
            "complete": True,
            "selection_uses_path_length": False,
            "uses_global_constraint_filtering": False,
        }
        self.history.append(report)
        if any(not values for values in context.domains.values()):
            report.update(status="empty_identity_domain")
            raise BindingResolutionError(
                "IDENTITY_REQUIRED_EMPTY",
                "required identity has no local candidates",
                report,
            )
        ambiguous = {
            path: values
            for path, values in context.domains.items()
            if len(values) > 1
        }
        if ambiguous:
            report.update(
                status="ambiguous",
                mechanism="forced_early_unique_block",
                ambiguous_domains={
                    path: list(values) for path, values in ambiguous.items()
                },
            )
            raise BindingResolutionError(
                "IDENTITY_REQUIRED_AMBIGUOUS",
                "forced early unique binding blocks on a multi-candidate local denotation",
                report,
            )
        chosen = {
            path: values[0] for path, values in context.domains.items()
        }
        report.update(
            status="resolved",
            mechanism="local_unique",
            chosen=chosen,
        )
        return _bound_scene(context.scene, context.roots, chosen), report


class PinnedIdentityResolver:
    """Replay one validated in-domain identity choice without search."""

    def __init__(
        self,
        choices: Mapping[str, int],
        *,
        mechanism: str = "pinned_identity_choice",
    ):
        self.choices = {str(path): int(value) for path, value in choices.items()}
        self.mechanism = str(mechanism)
        self.history: list[dict] = []

    def __call__(self, contract, label_grounding, asg, robot_position):
        context = build_identity_candidate_context(
            contract, label_grounding, asg, robot_position
        )
        ambiguous = {
            path: values
            for path, values in context.domains.items()
            if len(values) > 1
        }
        if set(self.choices) != set(ambiguous):
            raise BindingResolutionError(
                "IDENTITY_REQUIRED_AMBIGUOUS",
                "pinned identity choice paths do not match current ambiguous domains",
                {
                    "status": "invalid_forced_choice",
                    "expected_paths": sorted(ambiguous),
                    "received_paths": sorted(self.choices),
                    "candidate_domains": {
                        path: list(values)
                        for path, values in context.domains.items()
                    },
                },
            )
        chosen: dict[str, int] = {}
        for path, values in context.domains.items():
            if not values:
                raise BindingResolutionError(
                    "IDENTITY_REQUIRED_EMPTY",
                    "required identity has no local candidates",
                    {"status": "empty_identity_domain", "path": path},
                )
            if len(values) == 1:
                chosen[path] = values[0]
                continue
            value = self.choices[path]
            if value not in values:
                raise BindingResolutionError(
                    "IDENTITY_REQUIRED_AMBIGUOUS",
                    f"pinned choice {value} is outside candidate domain {path}",
                    {
                        "status": "invalid_forced_choice",
                        "path": path,
                        "chosen": value,
                        "candidate_domain": list(values),
                    },
                )
            chosen[path] = value
        report = {
            "status": "resolved",
            "mechanism": self.mechanism,
            "candidate_source": "ASG+semantic_closure",
            "candidate_domains": {
                path: list(values) for path, values in context.domains.items()
            },
            "chosen": chosen,
            "complete": True,
            "selection_uses_path_length": False,
            "uses_global_constraint_filtering": False,
        }
        self.history.append(report)
        return _bound_scene(context.scene, context.roots, chosen), report


class ReplayedFrontierResolver:
    """Replay one frozen candidate resolution frontier result while it remains authoritative.

    verification inspects the exact preceding candidate resolution artifact rather than repeating that
    commitment step. If deterministic verification grounding changes the commitment
    frontier, the frozen evidence is stale and the supplied fresh fallback
    resolver is used instead.
    """

    def __init__(self, frozen_report: Mapping[str, object] | None, fallback):
        self.frozen_report = (
            deepcopy(dict(frozen_report))
            if isinstance(frozen_report, Mapping)
            else None
        )
        self.fallback = fallback
        self.frozen_valid = True
        # Replayed evidence is already present in the shared candidate resolution history. Only
        # genuinely new fallback work is appended here.
        self.history: list[dict] = []

    def invalidate_frozen(self) -> None:
        """Advance the frontier: any verification grounding repair invalidates old candidate resolution evidence."""
        self.frozen_valid = False

    @staticmethod
    def _domains(context) -> dict[str, list[int]]:
        return {
            str(path): [int(value) for value in values]
            for path, values in context.domains.items()
        }

    def _fresh(self, contract, label_grounding, asg, robot_position):
        before = len(self.fallback.history)
        try:
            return self.fallback(contract, label_grounding, asg, robot_position)
        finally:
            self.history.extend(
                deepcopy(self.fallback.history[before:])
            )

    def __call__(self, contract, label_grounding, asg, robot_position):
        report = self.frozen_report
        if not self.frozen_valid or not isinstance(report, Mapping):
            return self._fresh(contract, label_grounding, asg, robot_position)

        context = build_identity_candidate_context(
            contract, label_grounding, asg, robot_position
        )
        current_domains = self._domains(context)
        frozen_domains = report.get("candidate_domains")
        if not isinstance(frozen_domains, Mapping):
            return self._fresh(contract, label_grounding, asg, robot_position)
        normalized_frozen = {
            str(path): [int(value) for value in values]
            for path, values in frozen_domains.items()
            if isinstance(values, list)
        }
        if normalized_frozen != current_domains:
            return self._fresh(contract, label_grounding, asg, robot_position)

        failure = report.get("replay_failure")
        if isinstance(failure, Mapping):
            raise BindingResolutionError(
                str(failure.get("code") or "IDENTITY_REQUIRED_AMBIGUOUS"),
                str(failure.get("message") or "shared candidate resolution resolution failed"),
                deepcopy(dict(report)),
                infrastructure=bool(failure.get("infrastructure", False)),
            )

        chosen_raw = report.get("chosen")
        if report.get("status") != "resolved" or not isinstance(chosen_raw, Mapping):
            return self._fresh(contract, label_grounding, asg, robot_position)
        chosen = {str(path): int(value) for path, value in chosen_raw.items()}
        if set(chosen) != set(current_domains):
            return self._fresh(contract, label_grounding, asg, robot_position)
        for path, values in current_domains.items():
            if chosen[path] not in values:
                return self._fresh(contract, label_grounding, asg, robot_position)
        return (
            _bound_scene(context.scene, context.roots, chosen),
            deepcopy(dict(report)),
        )


class ProgressiveResolver:
    """Progressively determine joint identities from minimum sufficient evidence.

    The order is deliberate:
      1. build the complete local GOAL/VIA candidate product;
      2. test every assignment for HARD feasibility only;
      3. if exactly one hard-feasible assignment remains, commit it immediately;
      4. only if multiple hard-feasible assignments remain may an explicit
         preference select among them;
      5. UNKNOWN/incomplete evidence never eliminates a candidate.

    Route length and numerical ID are never identity tie-breakers.
    """

    def __init__(self, backend, *, max_assignments=128):
        self.backend = backend
        self.max_assignments = max_assignments
        self.history: list[dict] = []

    @staticmethod
    def _has_explicit_preference(contract) -> bool:
        return (
            any(
                via.preference_source_text is not None
                for via in contract.route.via_events
            )
            or any(
                constraint.preference_source_text is not None
                for constraint in (
                    *contract.constraints.avoid,
                    *contract.constraints.count,
                )
            )
            or any(
                getattr(order, "preference_source_text", None) is not None
                for family in (
                    contract.orders.precedence,
                    contract.orders.consecutive,
                    contract.orders.between_only,
                    contract.orders.between_without,
                )
                for order in family
            )
        )

    @staticmethod
    def _hard_replay_errors(asg, grounded, route, robot_position):
        replay = evaluate_grounded_route(
            asg,
            grounded,
            route,
            dict(robot_position),
        )
        return replay, [
            item
            for item in replay.get("constraints", [])
            if not item.get("satisfied")
            and item.get("component")
            not in {
                "prefer",
                "prefer_order",
                "preference_optimality",
            }
        ]

    def __call__(self, contract, label_grounding, asg, robot_position):
        report = {
            "status": "started",
            "candidate_source": "ASG+semantic_closure",
            "assignments": [],
            "complete": False,
            "selection_uses_path_length": False,
            "hard_query_mode": FEASIBLE.mode.value,
            "preference_query_mode": None,
        }
        self.history.append(report)
        try:
            context = build_identity_candidate_context(
                contract,
                label_grounding,
                asg,
                robot_position,
            )
            candidates = context.scene
            roots = context.roots
            domains = context.domains
            report["candidate_domains"] = {
                path: list(values)
                for path, values in domains.items()
            }
            report["all_occurrences"] = list(context.all_occurrences)

            if any(not values for values in domains.values()):
                report.update(
                    status="empty_identity_domain",
                    complete=True,
                )
                raise BindingResolutionError(
                    "IDENTITY_REQUIRED_EMPTY",
                    "required identity has no candidates",
                    report,
                )

            total = context.total_assignments
            report["total_assignments"] = total
            if total == 1:
                chosen = {
                    path: values[0]
                    for path, values in domains.items()
                }
                report.update(
                    status="resolved",
                    mechanism="local_unique",
                    chosen=chosen,
                    complete=True,
                    evidence_stages=[],
                    final_route_query_mode="MIN_PENALTY",
                )
                return _bound_scene(candidates, roots, chosen), report

            names = tuple(domains)
            hard_feasible = []
            has_preference = self._has_explicit_preference(contract)
            report["has_explicit_preference"] = has_preference

            # Phase 1: hard feasibility only.  Soft clauses may exist in the
            # program, but FEASIBLE does not require proving their optimum.
            for ordinal, values in enumerate(
                product(*(domains[path] for path in names))
            ):
                if ordinal >= self.max_assignments:
                    break

                chosen = dict(zip(names, values))
                scene = _bound_scene(candidates, roots, chosen)
                row = {
                    "chosen": chosen,
                    "hard_evidence": {
                        "query_mode": FEASIBLE.mode.value,
                    },
                }
                report["assignments"].append(row)

                try:
                    lowered = lower_grounded_contract(
                        contract,
                        scene,
                        robot_position,
                        asg,
                    )
                except Exception as exc:
                    row["hard_evidence"].update(
                        status="LOWERING_ERROR",
                        status_reason=f"{type(exc).__name__}: {exc}",
                    )
                    report.update(
                        status="semantic_lowering_conflict",
                        complete=False,
                        lowering_error=row["hard_evidence"][
                            "status_reason"
                        ],
                    )
                    raise BindingResolutionError(
                        "SEMANTIC_LOWERING_CONFLICT",
                        "candidate assignment cannot be lowered deterministically: "
                        + row["hard_evidence"]["status_reason"],
                        report,
                        infrastructure=False,
                    ) from exc

                row["dsl"] = lowered.dsl
                try:
                    route = self.backend(
                        lowered.dsl,
                        asg,
                        grounded_contract=lowered.grounded_constraints,
                        robot_position=robot_position,
                        request=FEASIBLE,
                    )
                    hard = row["hard_evidence"]
                    hard.update(
                        status=route.get("status", "UNKNOWN"),
                        status_reason=route.get("status_reason"),
                        query_outcome=route.get("query_outcome"),
                        proof_domain=route.get("proof_domain"),
                        query_graph_complete=route.get(
                            "query_graph_complete", False
                        ),
                        state_search_complete=route.get(
                            "state_search_complete", False
                        ),
                        witness_waypoint_count=len(
                            route.get("waypoints", [])
                        ),
                    )

                    if (
                        hard["status"] == "SAT"
                        and route.get("waypoints")
                    ):
                        replay, hard_errors = self._hard_replay_errors(
                            asg,
                            lowered.grounded_constraints,
                            route,
                            robot_position,
                        )
                        if hard_errors:
                            raise BindingResolutionError(
                                "BACKEND_ASSERTION_FAILURE",
                                "candidate FEASIBLE witness failed hard replay",
                                report,
                                infrastructure=True,
                            )
                        hard["hard_replay_pass"] = True
                        hard["witness_soft_cost"] = replay.get(
                            "preference_cost"
                        )
                        hard["witness_route_cost"] = route.get("cost")
                        hard_feasible.append(
                            (scene, row, lowered)
                        )
                    elif (
                        hard["status"] == "UNSAT"
                        and hard["query_outcome"] == "HARD_UNSAT"
                    ):
                        hard["hard_infeasible_proved"] = True
                    else:
                        # An UNKNOWN, interrupted search, or uncertified UNSAT
                        # cannot eliminate this assignment.
                        hard["status"] = "UNKNOWN"
                        hard["status_reason"] = (
                            hard["status_reason"]
                            or "hard-feasibility evidence is incomplete"
                        )
                except QueryBudgetExhausted:
                    row["hard_evidence"].update(
                        status="UNKNOWN",
                        status_reason="total_solver_budget_exhausted",
                    )
                    break
                except BindingResolutionError:
                    raise
                except Exception as exc:
                    row["hard_evidence"].update(
                        status="UNKNOWN",
                        status_reason=f"{type(exc).__name__}: {exc}",
                    )

            hard_complete = (
                len(report["assignments"]) == total
                and not any(
                    row["hard_evidence"].get("status") == "UNKNOWN"
                    for row in report["assignments"]
                )
            )
            report["hard_filter_complete"] = hard_complete
            report["hard_feasible_count"] = len(hard_feasible)

            if not hard_complete:
                report.update(status="unknown", complete=False)
                raise BindingResolutionError(
                    "SOLVER_UNKNOWN",
                    "hard candidate filtering is incomplete; no unresolved "
                    "assignment was discarded as false",
                    report,
                    infrastructure=True,
                )

            if not hard_feasible:
                report.update(status="infeasible", complete=True)
                raise BindingResolutionError(
                    "HARD_SAT_NOT_ESTABLISHED",
                    "all complete identity assignments are hard-infeasible "
                    "in the declared solver domain",
                    report,
                )

            # Hard constraints have priority over soft preferences.  Once they
            # leave a unique assignment, identity is already determined.
            if len(hard_feasible) == 1:
                selected, row, _lowered = hard_feasible[0]
                report.update(
                    status="resolved",
                    mechanism="hard_constraint_resolved",
                    chosen=row["chosen"],
                    complete=True,
                    evidence_stages=[FEASIBLE.mode.value],
                    final_route_query_mode="MIN_PENALTY",
                )
                return selected, report

            if not has_preference:
                report.update(
                    status="ambiguous",
                    complete=True,
                    mechanism="multiple_hard_feasible_witnesses",
                    hard_feasible_assignments=[
                        row["chosen"]
                        for _scene, row, _lowered in hard_feasible
                    ],
                    unresolved_assignments=[
                        row["chosen"]
                        for _scene, row, _lowered in hard_feasible
                    ],
                )
                raise BindingResolutionError(
                    "IDENTITY_REQUIRED_AMBIGUOUS",
                    "multiple hard-feasible identity assignments; "
                    "no explicit preference can distinguish them",
                    report,
                )

            # Phase 2: preferences are consulted only among assignments already
            # proved hard-feasible.
            report["preference_query_mode"] = MIN_PENALTY.mode.value
            preference_complete = True
            preferred = []
            for scene, row, lowered in hard_feasible:
                pref = {
                    "query_mode": MIN_PENALTY.mode.value,
                }
                row["preference_evidence"] = pref
                try:
                    route = self.backend(
                        lowered.dsl,
                        asg,
                        grounded_contract=lowered.grounded_constraints,
                        robot_position=robot_position,
                        request=MIN_PENALTY,
                    )
                    pref.update(
                        status=route.get("status", "UNKNOWN"),
                        status_reason=route.get("status_reason"),
                        query_outcome=route.get("query_outcome"),
                        proof_domain=route.get("proof_domain"),
                        penalty_proven=route.get(
                            "penalty_proven", False
                        ),
                        penalty_lb=route.get("penalty_lb"),
                        penalty_ub=route.get("penalty_ub"),
                        witness_waypoint_count=len(
                            route.get("waypoints", [])
                        ),
                    )
                    if (
                        pref["status"] == "SAT"
                        and route.get("waypoints")
                    ):
                        replay, hard_errors = self._hard_replay_errors(
                            asg,
                            lowered.grounded_constraints,
                            route,
                            robot_position,
                        )
                        if hard_errors:
                            raise BindingResolutionError(
                                "BACKEND_ASSERTION_FAILURE",
                                "candidate MIN_PENALTY witness failed hard replay",
                                report,
                                infrastructure=True,
                            )
                        cost = replay.get("preference_cost")
                        pref["soft_cost"] = cost
                        pref["witness_route_cost"] = route.get("cost")
                        if (
                            route.get("penalty_proven") is not True
                            or not isinstance(cost, (int, float))
                            or not math.isfinite(cost)
                        ):
                            pref["status"] = "UNKNOWN"
                            pref["status_reason"] = (
                                "candidate preference minimum is not proved"
                            )
                            preference_complete = False
                        else:
                            preferred.append((scene, row, float(cost)))
                    elif (
                        pref["status"] == "UNSAT"
                        and pref["query_outcome"] == "HARD_UNSAT"
                    ):
                        # This contradicts the earlier FEASIBLE witness for the
                        # exact same assignment and contract.
                        raise BindingResolutionError(
                            "BACKEND_ASSERTION_FAILURE",
                            "MIN_PENALTY reported HARD_UNSAT after a certified "
                            "FEASIBLE witness for the same candidate",
                            report,
                            infrastructure=True,
                        )
                    else:
                        pref["status"] = "UNKNOWN"
                        pref["status_reason"] = (
                            pref["status_reason"]
                            or "preference evidence is incomplete"
                        )
                        preference_complete = False
                except QueryBudgetExhausted:
                    pref.update(
                        status="UNKNOWN",
                        status_reason="total_solver_budget_exhausted",
                    )
                    preference_complete = False
                    break
                except BindingResolutionError:
                    raise
                except Exception as exc:
                    pref.update(
                        status="UNKNOWN",
                        status_reason=f"{type(exc).__name__}: {exc}",
                    )
                    preference_complete = False

            report["preference_filter_complete"] = (
                preference_complete
                and len(preferred) == len(hard_feasible)
            )
            if not report["preference_filter_complete"]:
                report.update(status="unknown", complete=False)
                raise BindingResolutionError(
                    "SOLVER_UNKNOWN",
                    "preference comparison is incomplete; no hard-feasible "
                    "candidate was discarded as suboptimal",
                    report,
                    infrastructure=True,
                )

            optimum = min(cost for _scene, _row, cost in preferred)
            winners = [
                (scene, row, cost)
                for scene, row, cost in preferred
                if abs(cost - optimum) <= 1e-9
            ]
            if len(winners) != 1:
                report.update(
                    status="ambiguous",
                    complete=True,
                    mechanism="explicit_preference_tie",
                    best_soft_cost=optimum,
                    unresolved_assignments=[
                        row["chosen"]
                        for _scene, row, _cost in winners
                    ],
                )
                raise BindingResolutionError(
                    "IDENTITY_REQUIRED_AMBIGUOUS",
                    "explicit preference does not uniquely choose an identity; "
                    "path length is not an identity tie-break",
                    report,
                )

            selected, row, _cost = winners[0]
            report.update(
                status="resolved",
                mechanism="preference_selected",
                chosen=row["chosen"],
                complete=True,
                best_soft_cost=optimum,
                evidence_stages=[
                    FEASIBLE.mode.value,
                    MIN_PENALTY.mode.value,
                ],
                final_route_query_mode="MIN_PENALTY",
            )
            return selected, report

        except BindingResolutionError:
            raise
        except Exception as exc:
            from reconav.solver.qualifier_geometry import UnknownRelationGeometry

            report.update(
                status="unresolved",
                error=f"{type(exc).__name__}: {exc}",
            )
            infrastructure = isinstance(
                exc,
                (
                    UnknownRelationGeometry,
                    QueryBudgetExhausted,
                    OSError,
                ),
            )
            raise BindingResolutionError(
                "SOLVER_UNKNOWN"
                if infrastructure
                else "IDENTITY_REQUIRED_AMBIGUOUS",
                str(exc),
                report,
                infrastructure=infrastructure,
            ) from exc
