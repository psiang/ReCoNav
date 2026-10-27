"""Bind frozen occurrence-level semantics to ASG instances deterministically."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reconav.scene.asg import ASG
from reconav.solver.dsl_ast import TargetQualifier, TargetSelector
from reconav.solver.target_grounding import ground_target
from reconav.method.grounding.contracts import GroundingBinding
from reconav.method.grounding.grounded_contract import (
    ASGBinding,
    GroundedOccurrence,
    SceneGroundedContract,
)
from reconav.method.grounding.label_grounding import NominalCommitment, collect_nominal_commitments
from reconav.method.semantic.contracts import SceneIndependentSemanticContract

_KIND_TO_DSL = {"object": "obj", "region": "region", "portal": "portal", "level": "level"}
_PATH_PART = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?")


@dataclass(frozen=True)
class AlignedOccurrence:
    position: int
    commitment: NominalCommitment
    binding: GroundingBinding


def align_frozen_label_grounding(
    contract: SceneIndependentSemanticContract,
    artifact: Mapping[str, Any],
) -> tuple[AlignedOccurrence, ...]:
    """Reconstruct and validate the frozen positional occurrence alignment."""
    if not artifact.get("stage1_gate_passed"):
        raise ValueError("label grounding artifact did not pass the frozen Stage-1 gate")
    validation = artifact.get("validation")
    if not isinstance(validation, Mapping) or validation.get("success") is not True:
        raise ValueError("label grounding artifact is not validated")

    commitments = collect_nominal_commitments(contract).commitments
    request_items = artifact.get("all_grounding_items")
    bindings = artifact.get("final_bindings")
    if not isinstance(request_items, list) or not isinstance(bindings, list):
        raise ValueError("label grounding artifact is missing positional occurrence data")
    if len(commitments) != len(request_items) or len(commitments) != len(bindings):
        raise ValueError(
            "label grounding occurrence cardinality mismatch: "
            f"collector={len(commitments)}, items={len(request_items)}, bindings={len(bindings)}"
        )
    if artifact.get("occurrence_count") != len(commitments):
        raise ValueError("label grounding occurrence_count does not match the collector")

    aligned: list[AlignedOccurrence] = []
    for position, (commitment, item, raw_binding) in enumerate(
        zip(commitments, request_items, bindings)
    ):
        if not isinstance(item, Mapping) or (
            item.get("head_text") != commitment.head_text
            or item.get("source_text") != commitment.source_text
        ):
            raise ValueError(f"label grounding request mismatch at occurrence {position}")
        binding = GroundingBinding.model_validate(raw_binding)
        if binding.head_text != commitment.head_text:
            raise ValueError(f"label grounding binding mismatch at occurrence {position}")
        aligned.append(AlignedOccurrence(position, commitment, binding))
    return tuple(aligned)


def start_target(robot_position: Mapping[str, Any]) -> TargetSelector:
    try:
        values = (
            float(robot_position["x"]),
            float(robot_position["floor_height"]),
            float(robot_position["y"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("task robot_position requires numeric x, floor_height, and y") from exc
    return TargetSelector("pos", "coord", values)


def semantic_target_selector(
    contract: SceneIndependentSemanticContract,
    path: str,
    aligned: tuple[AlignedOccurrence, ...],
) -> TargetSelector:
    """Compile one Stage-1 referent occurrence into existing target semantics."""
    node = _node_at_path(contract.model_dump(mode="json"), path)
    by_path = {item.commitment.path: item.binding for item in aligned}
    return _selector_from_node(node, path, by_path)


def bind_scene(
    contract: SceneIndependentSemanticContract,
    label_grounding_artifact: Mapping[str, Any],
    asg: ASG,
    robot_position: Mapping[str, Any],
) -> SceneGroundedContract:
    """Add deterministic ASG IDs without modifying the frozen semantic artifact."""
    empty_set_paths = frozenset(
        [
            *(
                f"$.constraints.avoid[{index}].target"
                for index, _ in enumerate(contract.constraints.avoid)
            ),
            *(
                f"$.constraints.count[{index}].target"
                for index, _ in enumerate(contract.constraints.count)
            ),
        ]
    )
    return _bind_scene(contract, label_grounding_artifact, asg, robot_position, empty_set_paths)


def bind_candidate_scene(contract, label_grounding_artifact, asg, robot_position):
    """Preserve complete denotations before a runtime identity policy commits IDs."""
    empty_set_paths = frozenset([
        *(f"$.constraints.avoid[{i}].target" for i, _ in enumerate(contract.constraints.avoid)),
        *(f"$.constraints.count[{i}].target" for i, _ in enumerate(contract.constraints.count)),
    ])
    return _bind_scene(contract, label_grounding_artifact, asg, robot_position,
                       empty_set_paths, require_executable=False)


def _bind_scene(
    contract: SceneIndependentSemanticContract,
    label_grounding_artifact: Mapping[str, Any],
    asg: ASG,
    robot_position: Mapping[str, Any],
    allow_empty_set_paths: frozenset[str],
    *, require_executable: bool = True,
) -> SceneGroundedContract:
    aligned = align_frozen_label_grounding(contract, label_grounding_artifact)
    start = start_target(robot_position)
    occurrences: list[GroundedOccurrence] = []
    for item in aligned:
        selector = semantic_target_selector(contract, item.commitment.path, aligned)
        try:
            grounded = ground_target(asg, selector, item.position + 1, start_target=start)
            grounded_ids = sorted(grounded.ids)
        except ValueError as exc:
            if (
                item.commitment.owner_path not in allow_empty_set_paths
                or not _empty_candidate_error(exc)
            ):
                raise
            grounded_ids = []
        occurrences.append(
            GroundedOccurrence(
                position=item.position,
                path=item.commitment.path,
                owner_path=item.commitment.owner_path,
                head_text=item.commitment.head_text,
                source_text=item.commitment.source_text,
                asg_binding=ASGBinding(
                    asg_kind=item.binding.asg_kind,
                    asg_label=item.binding.asg_label,
                    grounded_ids=grounded_ids,
                ),
            )
        )
    scene = SceneGroundedContract(
        semantic_contract=contract.model_dump(mode="json"),
        occurrences=occurrences,
    )
    if require_executable:
        _validate_executable_cardinality(contract, scene, aligned, asg, start)
    return scene


def _empty_candidate_error(exc: ValueError) -> bool:
    message = str(exc)
    return (
        " not found" in message
        or " no " in message
        and " matches category " in message
        or " entities found" in message
        or "target qualifiers removed all candidates" in message
    )


def binding_for_referent(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    path: str,
) -> ASGBinding:
    node = _node_at_path(contract.model_dump(mode="json"), path)
    nominal_path = _nominal_path(node, path)
    by_path = {item.path: item.asg_binding for item in scene.occurrences}
    try:
        return by_path[nominal_path]
    except KeyError as exc:
        raise ValueError(f"no grounded nominal occurrence at {nominal_path}") from exc


def _validate_executable_cardinality(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    aligned: tuple[AlignedOccurrence, ...],
    asg: ASG,
    start: TargetSelector,
) -> None:
    required: list[tuple[str, Any, str]] = [
        ("hard GOAL", contract.route.goal.target, "$.route.goal.target")
    ]
    required.extend(
        ("carry width source", item.payload, f"$.constraints.carry[{index}].payload")
        for index, item in enumerate(contract.constraints.carry)
    )
    required.extend(
        ("reference width source", item.reference, f"$.constraints.robot_width[{index}].reference")
        for index, item in enumerate(contract.constraints.robot_width)
        if hasattr(item, "reference")
    )
    for role, _referent, path in required:
        selector = semantic_target_selector(contract, _nominal_path_for_contract(contract, path), aligned)
        grounded = ground_target(asg, selector, 1, start_target=start)
        if len(grounded.ids) != 1:
            raise ValueError(f"{role} target is ambiguous: candidates {sorted(grounded.ids)}")

    for index, event in enumerate(contract.route.via_events):
        path = f"$.route.via_events[{index}].target"
        selector = semantic_target_selector(contract, _nominal_path_for_contract(contract, path), aligned)
        grounded = ground_target(asg, selector, 1, start_target=start)
        if not grounded.ids:
            role = "hard VIA" if event.preference_source_text is None else "soft VIA"
            raise ValueError(f"{role} target has no candidates")


def _nominal_path_for_contract(contract: SceneIndependentSemanticContract, path: str) -> str:
    node = _node_at_path(contract.model_dump(mode="json"), path)
    return _nominal_path(node, path)


def _nominal_path(node: Mapping[str, Any], path: str) -> str:
    kind = node.get("kind")
    if kind == "coreference":
        antecedent = node.get("antecedent")
        if not isinstance(antecedent, Mapping):
            raise ValueError(f"invalid coreference at {path}")
        return _nominal_path(antecedent, f"{path}.antecedent")
    if kind != "nominal":
        raise ValueError(f"{path} is not a nominal referent")
    return path


def _selector_from_node(
    node: Mapping[str, Any],
    path: str,
    by_path: Mapping[str, GroundingBinding],
) -> TargetSelector:
    kind = node.get("kind")
    if kind == "start":
        return TargetSelector("pos", "coord", ())
    if kind == "coreference":
        antecedent = node.get("antecedent")
        if not isinstance(antecedent, Mapping):
            raise ValueError(f"invalid coreference at {path}")
        return _selector_from_node(antecedent, f"{path}.antecedent", by_path)
    if kind != "nominal":
        raise ValueError(f"unsupported referent kind at {path}: {kind}")
    try:
        binding = by_path[path]
    except KeyError as exc:
        raise ValueError(f"missing positional label grounding binding for {path}") from exc
    dsl_kind = _KIND_TO_DSL[binding.asg_kind]
    selector = "any" if binding.asg_label == "any" else "category"
    values = () if selector == "any" else (binding.asg_label,)
    qualifiers: list[TargetQualifier] = []
    for index, qualifier in enumerate(node.get("qualifiers") or []):
        q_path = f"{path}.qualifiers[{index}]"
        references = qualifier.get("references") or []
        ref_selectors = [
            _selector_from_node(ref, f"{q_path}.references[{ref_index}]", by_path)
            for ref_index, ref in enumerate(references)
        ]
        relation = qualifier.get("relation")
        if relation == "between":
            if len(ref_selectors) != 2:
                raise ValueError(f"BETWEEN at {q_path} requires two references")
            qualifiers.append(
                TargetQualifier("between_regions", ref_selectors[0], second_target=ref_selectors[1])
            )
        else:
            if len(ref_selectors) != 1:
                raise ValueError(f"{relation} at {q_path} requires one reference")
            qualifier_kind = None
            if relation == "in":
                if ref_selectors[0].kind not in {"region", "level"}:
                    raise ValueError(f"IN at {q_path} requires a region or level reference")
                qualifier_kind = f"in_{ref_selectors[0].kind}"
            if relation in {"near", "nearest_to", "farthest_from"}:
                qualifier_kind = str(relation)
            if qualifier_kind is None:
                raise ValueError(f"unsupported qualifier at {q_path}: {relation}")
            reference_node = references[0]
            while reference_node.get("kind") == "coreference":
                reference_node = reference_node["antecedent"]
            mode = ("start" if reference_node.get("kind") == "start"
                    else str(reference_node.get("quantifier", "unique")))
            if mode == "exists" and relation != "near":
                raise ValueError("existential reference is supported only for NEAR")
            if mode == "start" and relation not in {"nearest_to", "farthest_from"}:
                raise ValueError("START reference is supported only for nearest/farthest")
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
    return TargetSelector(dsl_kind, selector, values, tuple(qualifiers))


def _node_at_path(root: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not path.startswith("$"):
        raise ValueError(f"invalid AST path: {path}")
    node: Any = root
    consumed = "$"
    for match in _PATH_PART.finditer(path[1:]):
        field, index = match.groups()
        consumed += match.group(0)
        if not isinstance(node, Mapping) or field not in node:
            raise ValueError(f"invalid AST path at {consumed}")
        node = node[field]
        if index is not None:
            if not isinstance(node, list) or int(index) >= len(node):
                raise ValueError(f"invalid AST path at {consumed}")
            node = node[int(index)]
    if consumed != path or not isinstance(node, Mapping):
        raise ValueError(f"invalid AST path: {path}")
    return node
