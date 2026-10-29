"""Gold-free repair triggering and readable deterministic diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from reconav.scene.asg import ASG
from reconav.method.verification.contract_index import ContractIndex, ContractNode
from reconav.method.verification.diagnostics import VerificationDiagnostic
from reconav.method.grounding.label_grounding import (
    collect_nominal_commitments,
    determine_groundings,
    vocabulary_from_entity_payload,
)
from reconav.method.semantic.contracts import SceneIndependentSemanticContract

_BACKEND_CODES = {
    "BACKEND_ERROR",
    "SOLVER_UNKNOWN",
    "BACKEND_ASSERTION_FAILURE",
    "CARRY_ELABORATION_MISMATCH",
    "LOWERING_ASSERTION_FAILURE",
    "UPSTREAM_INFRASTRUCTURE_FAILURE",
}

# These identifiers are part of the paper's model inputs.
_MODEL_CHECKPOINTS = {
    "contract_validation": "L3-S1",
    "label_grounding": "L3-G",
    "scene_binding": "L3-B",
    "route_lowering": "L3-L",
    "route_verification": "L3-X",
}


def repair_eligibility(
    *,
    final_status: str,
    diagnostics: Sequence[VerificationDiagnostic | Mapping[str, Any]],
) -> tuple[bool, str]:
    """Return the sole repair eligibility decision for a final verification state."""
    blocked = [_diagnostic_dict(item) for item in diagnostics if _field(item, "severity") == "BLOCK"]
    if final_status != "BLOCK" or not blocked:
        return False, "verification final status is not BLOCK"
    if any(item["code"] in _BACKEND_CODES or item.get("needs_backend_fix") for item in blocked):
        return False, "backend/infrastructure failure"
    if any(item.get("repairable") for item in blocked):
        return False, "deterministic verification repair remains available"
    semantic = any(
        item["code"] == "HARD_SAT_NOT_ESTABLISHED"
        or item.get("category") in {"ARTIFACT", "COMMITMENT", "GROUNDING"}
        for item in blocked
    )
    if not semantic:
        return False, "failure is not plausibly semantic"
    return True, "final semantic BLOCK after deterministic repair exhaustion"


def build_error(diagnostics: Sequence[VerificationDiagnostic | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "code": item["code"],
            "checkpoint": _MODEL_CHECKPOINTS[item["checkpoint"]],
            "category": item["category"],
            "assertion": item["assertion"],
            "expected": item.get("expected"),
            "observed": item.get("observed"),
        }
        for item in map(_diagnostic_dict, diagnostics)
        if item["severity"] == "BLOCK"
    ]


def build_warnings(
    diagnostics: Sequence[VerificationDiagnostic | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return all current non-blocking diagnostics without interpreting them."""
    return [
        {
            "code": item["code"],
            "checkpoint": _MODEL_CHECKPOINTS[item["checkpoint"]],
            "category": item["category"],
            "node_or_path": "$.layer1" if item["node_or_path"] == "$.label_grounding" else item["node_or_path"],
            "assertion": item["assertion"],
            "expected": item.get("expected"),
            "observed": item.get("observed"),
            "affected_commitments": item.get("affected_commitments", []),
        }
        for item in map(_diagnostic_dict, diagnostics)
        if item["severity"] == "WARNING"
    ]


def build_error_location(
    diagnostics: Sequence[VerificationDiagnostic | Mapping[str, Any]],
    index: ContractIndex,
    contract: SceneIndependentSemanticContract | None = None,
) -> list[dict[str, Any]]:
    """Resolve diagnostic paths/handles to readable semantic nodes."""
    nodes = {node.handle: node for node in index.nodes}
    located: list[ContractNode] = []
    paths: list[str] = []
    for diagnostic in map(_diagnostic_dict, diagnostics):
        if diagnostic["severity"] != "BLOCK":
            continue
        path = str(diagnostic.get("node_or_path") or "$")
        paths.append(path)
        residual = (diagnostic.get("detail") or {}).get("residual_positions")
        if contract is not None and _is_grounding_failure(diagnostic) and residual:
            commitments = collect_nominal_commitments(contract).commitments
            residual_paths = [
                commitments[position].path
                for position in residual
                if isinstance(position, int) and 0 <= position < len(commitments)
            ]
            residual_nodes = [
                node for node in index.nodes if node.path in residual_paths
            ]
            if residual_nodes:
                located.extend(residual_nodes)
                continue
        seeds = set(diagnostic.get("affected_commitments") or [])
        seeds.add(path)
        for seed in seeds:
            if seed == "$.route":
                located.extend(
                    node
                    for node in index.nodes
                    if node.kind == "EVENT" and node.path.startswith("$.route.")
                )
                continue
            if seed in nodes:
                located.append(nodes[seed])
                continue
            exact = [node for node in index.nodes if node.path == seed]
            if exact:
                located.extend(exact)
                continue
            located.extend(
                node
                for node in index.nodes
                if node.path.startswith(seed + ".")
                or seed.startswith(node.path + ".")
            )
            for edge in index.edges:
                if edge.path == seed or seed.startswith(edge.path + "."):
                    located.extend(
                        nodes[handle]
                        for handle in (edge.from_handle, edge.to_handle)
                        if handle in nodes
                    )
    if not located and any(path == "$.route" for path in paths):
        located = [node for node in index.nodes if node.path.startswith("$.route.")]
    unique = {node.handle: node for node in located}
    return [
        _node_dict(unique[handle], contract)
        for handle in sorted(unique, key=_handle_key)
    ]


def complete_asg_label_vocabulary(asg: ASG) -> dict[str, list[str]]:
    """Return every legal label for each repair grounding output kind."""
    return vocabulary_from_entity_payload(asg.to_entity_dict()).model_dump(mode="json")


def build_grounding_output_domain_audit(
    contract: SceneIndependentSemanticContract,
    label_grounding: Mapping[str, Any] | None,
    asg: ASG,
    index: ContractIndex,
    *,
    visible_vocabulary: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Prove that every non-fixed output slot sees its complete legal domain."""
    legal = complete_asg_label_vocabulary(asg)
    vocabulary_source = legal if visible_vocabulary is None else visible_vocabulary
    visible = {
        kind: sorted({str(label) for label in labels})
        for kind, labels in vocabulary_source.items()
    }
    collection = collect_nominal_commitments(contract)
    determine = determine_groundings(
        collection, vocabulary_from_entity_payload(asg.to_entity_dict())
    )
    fully = dict(
        zip(
            determine.value("grounding.fully_determined_positions", ()),
            determine.value("grounding.fully_determined", ()),
        )
    )
    structural = dict(
        zip(
            determine.value("grounding.structurally_determined_positions", ()),
            determine.value("grounding.structurally_determined", ()),
        )
    )
    raw_bindings = (label_grounding or {}).get("final_bindings")
    current = raw_bindings if isinstance(raw_bindings, list) else []
    slots = []
    failures = []
    for position, commitment in enumerate(collection.commitments):
        current_binding = current[position] if position < len(current) else None
        if position in fully:
            classification = "fully_determined"
            required_kinds: list[str] = []
        elif position in structural:
            classification = "kind_determined_label_unresolved"
            required_kinds = [structural[position].asg_kind]
        else:
            classification = "kind_and_label_unresolved"
            required_kinds = sorted(commitment.possible_kinds)
            if not required_kinds:
                # This is an inconsistent *input* structure, not a missing
                # output vocabulary. Semantic repair can move/remove the
                # wrongly attached qualifier and must see every legal kind.
                # The edited contract is still checked by label grounding/verification afterwards.
                classification = "inconsistent_structure_requires_semantic_repair"
                required_kinds = sorted(legal)
        if isinstance(current_binding, Mapping) and position not in fully:
            current_kind = current_binding.get("asg_kind")
            if current_kind in legal and current_kind not in required_kinds:
                required_kinds.append(str(current_kind))
                required_kinds.sort()
        missing_kinds = [kind for kind in required_kinds if kind not in visible]
        missing_labels = {
            kind: sorted(set(legal[kind]) - set(visible.get(kind, ())))
            for kind in required_kinds
            if kind in legal and set(legal[kind]) - set(visible.get(kind, ()))
        }
        if missing_kinds or missing_labels:
            failures.append(
                {
                    "code": "INCOMPLETE_GROUNDING_OUTPUT_DOMAIN",
                    "position": position,
                    "path": commitment.path,
                    "handle": index.handle_for_path(commitment.path),
                    "allowed_kinds": sorted(commitment.possible_kinds),
                    "required_kinds": required_kinds,
                    "missing_vocabulary_kinds": missing_kinds,
                    "missing_legal_labels": missing_labels,
                }
            )
        slots.append(
            {
                "position": position,
                "path": commitment.path,
                "handle": index.handle_for_path(commitment.path),
                "classification": classification,
                "allowed_kinds": sorted(commitment.possible_kinds),
                "required_kinds": required_kinds,
                "current_binding": current_binding,
                "determined_binding": (
                    fully[position].model_dump(mode="json")
                    if position in fully
                    else None
                ),
            }
        )
    dynamic_missing = {
        kind: sorted(set(labels) - set(visible.get(kind, ())))
        for kind, labels in legal.items()
        if set(labels) - set(visible.get(kind, ()))
    }
    if dynamic_missing:
        failures.append(
            {
                "code": "INCOMPLETE_GROUNDING_OUTPUT_DOMAIN",
                "path": "<dynamic-semantic-edit>",
                "handle": None,
                "allowed_kinds": sorted(legal),
                "required_kinds": sorted(legal),
                "missing_vocabulary_kinds": [
                    kind for kind in legal if kind not in visible
                ],
                "missing_legal_labels": dynamic_missing,
            }
        )
    return {
        "passed": not failures,
        "visible_label_vocabulary": visible,
        "legal_label_vocabulary": legal,
        "visible_vocabulary_kinds": sorted(visible),
        "dynamic_semantic_edit_domains_complete": not dynamic_missing,
        "slots": slots,
        "failures": failures,
    }


def _node_dict(
    node: ContractNode,
    contract: SceneIndependentSemanticContract | None = None,
) -> dict[str, Any]:
    value = {
        "handle": node.handle,
        "semantic_type": node.kind,
        "role": node.role,
        "path": node.path,
        "source_text": node.source_text,
    }
    if contract is not None:
        value["semantic_content"] = _semantic_content(contract, node.path)
    return value


def _semantic_content(
    contract: SceneIndependentSemanticContract,
    path: str,
) -> Any:
    value: Any = contract.model_dump(mode="json")
    for field, index in re.findall(r"(?:^\$\.|\.)([^.\[]+)(?:\[(\d+)\])?", path):
        if not isinstance(value, Mapping) or field not in value:
            return None
        value = value[field]
        if index:
            if not isinstance(value, list) or int(index) >= len(value):
                return None
            value = value[int(index)]
    return value


def _is_grounding_failure(diagnostic: Mapping[str, Any]) -> bool:
    return diagnostic.get("category") == "GROUNDING" or str(
        diagnostic.get("code", "")
    ).startswith("GROUNDING_")


def _field(value: VerificationDiagnostic | Mapping[str, Any], name: str) -> Any:
    return getattr(value, name) if isinstance(value, VerificationDiagnostic) else value.get(name)


def _diagnostic_dict(value: VerificationDiagnostic | Mapping[str, Any]) -> dict[str, Any]:
    return value.model_dump(mode="json") if isinstance(value, VerificationDiagnostic) else dict(value)


def _handle_key(handle: str) -> tuple[str, int]:
    return handle[:1], int(handle[1:])
